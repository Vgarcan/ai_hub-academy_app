import copy
import json
import time

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from ai_hub.models import ExecutionSession, ExecutionStepRun, GameActionRun
from ai_hub.services.agent_runtime import (
    apply_mapping,
    execute_agent,
    execute_agent_deliberate,
    prepare_agent_payload,
)
from ai_hub.services.contracts import validate_payload
from ai_hub.services.game_agent_resolution import resolve_game_entry_agent
from ai_hub.services.tools_runtime import TOOL_POLICY_ALL, TOOL_POLICY_GAME_CONTEXT_ONLY


AGENT_TOOL_RUNTIME_RESOLVED = "resolved"
AGENT_TOOL_RUNTIME_LEGACY = "legacy_preexecute"
AGENT_TOOL_RUNTIME_CHOICES = {
    AGENT_TOOL_RUNTIME_RESOLVED,
    AGENT_TOOL_RUNTIME_LEGACY,
}
DEFAULT_GAME_FINISH_ACTIONS = ["finish", "final", "complete", "stop"]
DEFAULT_GAME_STOP_STATUSES = ["success", "complete", "completed", "done", "final"]
DEFAULT_GAME_MEMORY_MAX_ENTRIES = 20
DEFAULT_GAME_OBSERVATIONS_MAX_ENTRIES = 8
DEFAULT_GAME_OBSERVATION_MAX_CHARS = 2000
DEFAULT_GAME_PREVIOUS_RESPONSE_MAX_CHARS = 2000
DEFAULT_GAME_MEMORY_ENTRY_MAX_CHARS = 500
GAME_RESERVED_PAYLOAD_KEYS = (
    "goal",
    "goal_text",
    "iteration",
    "max_iterations",
    "memory",
    "scoped_memory",
    "observations",
    "previous_response",
    "available_actions",
    "game_policy",
    "game_response_contract",
)


def _session_runtime_config(session: ExecutionSession) -> dict:
    runtime_config = session.runtime_config or {}
    if not isinstance(runtime_config, dict):
        raise ValidationError("Execution session runtime_config must be a JSON object.")
    return dict(runtime_config)


def _resolve_agent_tool_runtime(session: ExecutionSession, runtime_config: dict | None = None) -> str:
    config = runtime_config if runtime_config is not None else _session_runtime_config(session)
    runtime = str(
        config.get(
            "agent_tool_runtime",
            getattr(settings, "AI_HUB_DEFAULT_AGENT_TOOL_RUNTIME", AGENT_TOOL_RUNTIME_RESOLVED),
        )
        or ""
    ).strip().lower()
    if runtime not in AGENT_TOOL_RUNTIME_CHOICES:
        raise ValidationError(
            f"Unknown agent tool runtime '{runtime}'. "
            f"Use '{AGENT_TOOL_RUNTIME_RESOLVED}' or '{AGENT_TOOL_RUNTIME_LEGACY}'."
        )
    return runtime


def _session_workspace(session: ExecutionSession):
    if session.runtime_kind != ExecutionSession.RuntimeKind.GAME:
        return None
    from ai_hub.services.game_policy import get_session_workspace

    return get_session_workspace(session)


def _execute_session_agent(
    *,
    session: ExecutionSession,
    step_run: ExecutionStepRun,
    agent,
    payload: dict,
    agent_tool_runtime: str,
    tool_policy: str,
) -> dict:
    if agent_tool_runtime == AGENT_TOOL_RUNTIME_LEGACY:
        output_payload = execute_agent(
            agent,
            payload,
            tool_policy=tool_policy,
            workspace=_session_workspace(session),
            execution_context={"session": session, "step_run": step_run},
        )
    else:
        workspace = _session_workspace(session)
        output_payload = execute_agent_deliberate(
            agent,
            payload,
            workspace=workspace,
            execution_context={"session": session, "step_run": step_run},
            tool_policy=tool_policy,
            allow_plain_final=True,
            unwrap_final_answer=True,
            # Generic deliberate calls do not yet persist a resumable model
            # checkpoint. Approval-requiring work must use a governed GAME
            # action, whose dispatcher owns approval and resume.
            allow_approval_requests=False,
        )
        if output_payload.get("status") != "final":
            detail = output_payload.get("error") or output_payload.get("requested_tool") or "unknown error"
            raise ValidationError(
                "Resolved agent tool runtime stopped with status "
                f"'{output_payload.get('status', 'unknown')}': {detail}"
            )
    output_payload["agent_tool_runtime"] = agent_tool_runtime
    return output_payload


def _create_step_run(session: ExecutionSession, step) -> ExecutionStepRun:
    return ExecutionStepRun.objects.create(
        session=session,
        order=step.order,
        pipeline_step=step,
        agent=step.agent,
        action_name="agent_call",
        status=ExecutionStepRun.Status.RUNNING,
    )


def _mapped_path_exists(source: dict, source_key: str) -> bool:
    current = source
    for part in str(source_key).split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _apply_step_output_mapping(output_payload: dict, mapping: dict) -> dict:
    missing_paths = [
        str(source_key)
        for source_key in (mapping or {}).values()
        if not _mapped_path_exists(output_payload, source_key)
    ]
    if missing_paths:
        raise ValidationError(
            "Pipeline step output is missing mapped source paths: "
            f"{', '.join(missing_paths)}."
        )
    return apply_mapping(output_payload, mapping or {})


def _execution_error_metadata(error: Exception) -> dict:
    category = getattr(error, "category", "")
    if not category:
        category = (
            "validation_error"
            if isinstance(error, ValidationError)
            else "execution_error"
        )
    return {
        "category": str(category),
        "type": type(error).__name__,
        "detail": str(error),
    }


def _agent_attempt_metadata(agent, status: str, error: Exception | None = None) -> dict:
    metadata = {
        "agent_id": agent.pk,
        "agent": agent.name,
        "status": status,
    }
    if error is not None:
        metadata["error"] = _execution_error_metadata(error)
    return metadata


def _fallback_recovery_metadata(
    *,
    primary_agent,
    primary_error: Exception,
    fallback_agent,
    fallback_status: str,
    final_outcome: str,
    fallback_error: Exception | None = None,
) -> dict:
    return {
        "attempted": True,
        "primary": _agent_attempt_metadata(
            primary_agent,
            "failed",
            primary_error,
        ),
        "fallback": _agent_attempt_metadata(
            fallback_agent,
            fallback_status,
            fallback_error,
        ),
        "final_outcome": final_outcome,
    }


def _create_game_step_run(session: ExecutionSession, agent, order: int) -> ExecutionStepRun:
    return ExecutionStepRun.objects.create(
        session=session,
        order=order,
        agent=agent,
        action_name="game_iteration",
        status=ExecutionStepRun.Status.RUNNING,
    )


def _mark_session_failed(session: ExecutionSession, context: dict, error: Exception) -> None:
    if session.runtime_kind == ExecutionSession.RuntimeKind.GAME:
        context.update(
            {
                "execution_outcome": "failed",
                "goal_outcome": "unknown",
                "finish_reason": "failed",
            }
        )
    session.status = ExecutionSession.Status.FAILED
    session.error_detail = str(error)
    session.final_context = context
    session.finished_at = timezone.now()
    session.save(update_fields=["status", "error_detail", "final_context", "finished_at", "updated_at"])


def _claim_session_for_run(session_id: int) -> None:
    with transaction.atomic():
        session = ExecutionSession.objects.select_for_update().get(pk=session_id)
        if session.status != ExecutionSession.Status.PENDING:
            raise ValidationError(
                f"Execution session must be pending before it can run; current status is '{session.status}'."
            )
        if session.step_runs.exists():
            raise ValidationError("Execution session already has step runs.")

        session.status = ExecutionSession.Status.RUNNING
        if not session.started_at:
            session.started_at = timezone.now()
        session.final_context = dict(session.initial_context or {})
        session.error_detail = ""
        session.finished_at = None
        session.save(
            update_fields=[
                "status",
                "started_at",
                "final_context",
                "error_detail",
                "finished_at",
                "updated_at",
            ]
        )


def _bounded_iteration_count(runtime_config: dict) -> int:
    raw_value = runtime_config.get("max_iterations", 3)
    try:
        max_iterations = int(raw_value)
    except (TypeError, ValueError):
        max_iterations = 3
    return min(max(max_iterations, 1), 25)


def _bounded_runtime_int(
    runtime_config: dict,
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(runtime_config.get(key, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


def _keep_latest(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    return items[-limit:]


def _bounded_prompt_value(
    value,
    *,
    max_chars: int,
    kind: str,
    reference: dict | None = None,
):
    copied = copy.deepcopy(value)
    serialized = json.dumps(copied, default=str, sort_keys=True)
    if len(serialized) <= max_chars:
        return copied

    marker = {
        "truncated": True,
        "kind": kind,
        "original_chars": len(serialized),
    }
    for key, item in (reference or {}).items():
        if item is not None and isinstance(item, (str, int, float, bool)):
            marker[key] = item if not isinstance(item, str) else item[:120]

    preview_size = max_chars
    while preview_size > 0:
        marker["preview"] = serialized[:preview_size]
        marker_size = len(json.dumps(marker, default=str, sort_keys=True))
        if marker_size <= max_chars:
            return marker
        preview_size -= max(marker_size - max_chars, 1)

    marker.pop("preview", None)
    if len(json.dumps(marker, default=str, sort_keys=True)) <= max_chars:
        return marker
    return {"truncated": True}


def _bounded_game_observation(observation, max_chars: int) -> dict:
    if not isinstance(observation, dict):
        return _bounded_prompt_value(
            observation,
            max_chars=max_chars,
            kind="game_observation",
        )
    reference_keys = (
        "action",
        "status",
        "complete",
        "action_run_id",
        "action_status",
        "waiting_reason",
        "resolution_status",
    )
    return _bounded_prompt_value(
        observation,
        max_chars=max_chars,
        kind="game_observation",
        reference={key: observation.get(key) for key in reference_keys},
    )


def _trim_memory_entry_text(entry, max_chars: int) -> dict:
    if not isinstance(entry, dict):
        return {"summary": str(entry)[:max_chars]}
    trimmed = dict(entry)
    for key in ("summary", "action_output_summary"):
        if key in trimmed:
            trimmed[key] = str(trimmed.get(key) or "")[:max_chars]
    return trimmed


def _available_action_names(runtime_config: dict) -> list[str]:
    names = {"think", *runtime_config.get("finish_actions", DEFAULT_GAME_FINISH_ACTIONS)}
    for action in runtime_config.get("available_actions", []):
        if isinstance(action, str):
            names.add(action)
        elif isinstance(action, dict) and action.get("name"):
            names.add(str(action["name"]))
    return sorted(str(name).strip().lower() for name in names if str(name).strip())


def _game_response_contract(runtime_config: dict) -> dict:
    return {
        "format": "Return only one JSON object. Do not wrap it in Markdown or add prose outside JSON.",
        "required_keys": ["action", "message", "complete", "final_answer"],
        "optional_keys": ["action_input"],
        "actions": _available_action_names(runtime_config),
        "schema": {
            "action": "string; one of actions. Use 'think' to continue or 'finish' to complete.",
            "action_input": "object; arguments for the selected action. Required when the action needs input.",
            "message": "string; short explanation of the decision or observation.",
            "complete": "boolean; true only when the goal is done.",
            "final_answer": "string; empty until complete, then the final answer/result.",
        },
        "example_continue": {
            "action": "search_knowledge",
            "action_input": {"query": "relevant context"},
            "message": "I need more context before deciding.",
            "complete": False,
            "final_answer": "",
        },
        "example_finish": {
            "action": "finish",
            "action_input": {},
            "message": "The goal is complete.",
            "complete": True,
            "final_answer": "Concise final result for the user or calling workflow.",
        },
    }


def _restore_game_reserved_payload(prepared_payload: dict, source_payload: dict) -> dict:
    restored_payload = dict(prepared_payload)
    for key in GAME_RESERVED_PAYLOAD_KEYS:
        restored_payload[key] = source_payload.get(key)
    return restored_payload


def _update_game_context(
    context: dict,
    *,
    goal_text: str,
    memory: list,
    observations: list,
    final_answer: str = "",
    finish_reason: str = "",
    **extra,
) -> None:
    context.update(
        {
            "goal": goal_text,
            "memory": memory,
            "observations": observations,
            "final_answer": final_answer,
            "finish_reason": finish_reason,
            **extra,
        }
    )


def _decode_model_content(output_payload: dict):
    content = ((output_payload or {}).get("llm") or {}).get("content", "")
    if isinstance(content, dict):
        return content, []
    if not isinstance(content, str):
        return content, ["llm.content must be a JSON object string."]

    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped), []
    except (TypeError, ValueError):
        return {"message": content}, ["llm.content must be valid JSON."]


def _extract_json_object_text(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and last > first:
        return stripped[first:last + 1]
    return stripped


def _merge_final_output_object(context: dict) -> None:
    value = context.get("final_output")
    if not value:
        return
    if isinstance(value, dict):
        payload = value
    elif isinstance(value, str):
        try:
            payload = json.loads(_extract_json_object_text(value))
        except (TypeError, ValueError) as exc:
            context["final_output_parse_error"] = str(exc)
            return
    else:
        return
    if not isinstance(payload, dict):
        return
    for key, payload_value in payload.items():
        if key != "final_output":
            context[key] = payload_value


def _raise_final_output_parse_error_if_required(context: dict, contract: dict) -> None:
    parse_error = context.get("final_output_parse_error")
    if not parse_error:
        return
    required = set((contract or {}).get("required") or [])
    if required and not required.issubset(context.keys()):
        raise ValidationError(f"final_output must be valid JSON: {parse_error}")


def _validate_game_decision(decision: dict, runtime_config: dict) -> list[str]:
    errors = []
    if not isinstance(decision, dict):
        return ["GAME decision must be a JSON object."]

    for key in ("action", "message", "complete", "final_answer"):
        if key not in decision:
            errors.append(f"missing '{key}'")

    action = str(decision.get("action") or decision.get("next_action") or "").strip().lower()
    if action and action not in _available_action_names(runtime_config):
        errors.append(f"unsupported action '{action}'")
    if "complete" in decision and not isinstance(decision.get("complete"), bool):
        errors.append("'complete' must be boolean")
    if "message" in decision and not isinstance(decision.get("message"), str):
        errors.append("'message' must be string")
    if "final_answer" in decision and not isinstance(decision.get("final_answer"), str):
        errors.append("'final_answer' must be string")
    return errors


def _game_observation(output_payload: dict, runtime_config: dict) -> dict:
    decoded, parse_errors = _decode_model_content(output_payload)
    if isinstance(decoded, dict):
        decision = decoded.get("decision") if isinstance(decoded.get("decision"), dict) else decoded
    else:
        decision = {"message": decoded}

    contract_errors = parse_errors + _validate_game_decision(decision, runtime_config)
    if runtime_config.get("strict_response_contract") and contract_errors:
        raise ValidationError(f"GAME response contract failed: {', '.join(contract_errors)}")

    finish_actions = set(runtime_config.get("finish_actions") or DEFAULT_GAME_FINISH_ACTIONS)
    stop_statuses = set(runtime_config.get("stop_statuses") or DEFAULT_GAME_STOP_STATUSES)
    action = str(decision.get("action") or decision.get("next_action") or "").strip().lower()
    status = str(decision.get("status") or decision.get("state") or "").strip().lower()
    explicit_complete = bool(decision.get("complete"))
    legacy_complete = bool(decision.get("completed") or decision.get("done"))

    if runtime_config.get("strict_response_contract"):
        is_complete = explicit_complete
    else:
        is_complete = explicit_complete or legacy_complete or action in finish_actions or status in stop_statuses
    final_answer = (
        decision.get("final_answer")
        or decision.get("answer")
        or decision.get("result")
        or decision.get("message")
        or ""
    )
    return {
        "decision": decision,
        "action": action,
        "status": status,
        "complete": is_complete,
        "final_answer": final_answer,
        "contract_valid": not contract_errors,
        "contract_errors": contract_errors,
    }


def _dispatch_observation_action(
    session: ExecutionSession,
    observation: dict,
    entry_agent,
    iteration: int,
    step_run=None,
) -> GameActionRun | None:
    from ai_hub.services.game_action_dispatcher import execute_game_action

    action = observation.get("action") or ""
    if not action or action == "think":
        return None
    decision = observation.get("decision") or {}
    action_input = decision.get("action_input") if isinstance(decision.get("action_input"), dict) else {}
    try:
        action_run = execute_game_action(
            session=session,
            step_run=step_run,
            action_name=action,
            action_input=action_input,
        )
        action_output = dict(action_run.output_payload or {})
        observation["action_run_id"] = action_run.pk
        observation["action_output"] = action_output
        if action_output.get("complete", False):
            observation["complete"] = True
            if action_output.get("final_answer"):
                observation["final_answer"] = action_output["final_answer"]
        if action_run.status == GameActionRun.Status.WAITING_APPROVAL:
            observation["waiting_reason"] = "needs_approval"
            observation["action_status"] = action_run.status
        return action_run
    except ValidationError as exc:
        observation["action_error"] = str(exc)
        return None


def _run_game_session(
    session: ExecutionSession,
    context: dict,
    *,
    allow_legacy_game_action_tools: bool = False,
    use_action_dispatcher: bool = False,
    start_order: int = 1,
) -> None:
    runtime_config = _session_runtime_config(session)
    agent_tool_runtime = _resolve_agent_tool_runtime(session, runtime_config)
    use_dispatcher = (
        use_action_dispatcher
        or bool(runtime_config.get("use_action_dispatcher"))
        or bool(runtime_config.get("game_action_dispatch_enabled"))
        or bool(context.get("game_action_dispatch_enabled"))
    )
    if session.runtime_mode == ExecutionSession.RuntimeMode.HYBRID:
        raise ValidationError("GAME Hybrid continuation is not enabled yet. Use sync or async mode.")
    entry_agent = resolve_game_entry_agent(session)
    if not entry_agent:
        raise ValidationError("GAME sessions require an entry agent or a pipeline with at least one step.")
    if not entry_agent.is_active:
        raise ValidationError("GAME entry agent must be active before it can run.")
    if session.goal_id:
        from ai_hub.services.game_feature_flags import require_game_feature
        from ai_hub.services.game_policy import validate_goal_execution_policy

        require_game_feature("AI_HUB_GAME_GOALS_ENABLED")
        validate_goal_execution_policy(session.goal.workspace, session.goal, session)
    if start_order == 1 and session.step_runs.exists():
        raise ValidationError("Execution session already has step runs.")

    max_iterations = _bounded_iteration_count(runtime_config)
    goal_text = session.goal_text or runtime_config.get("goal") or ""
    if not goal_text:
        raise ValidationError("GAME sessions require goal_text or runtime_config.goal.")

    session.status = ExecutionSession.Status.RUNNING
    session.final_context = context
    session.save(update_fields=["status", "started_at", "final_context", "updated_at"])

    if start_order > 1:
        # Resume: preserve memory and observations from prior run
        memory = list(context.get("memory") or [])
        observations = list(context.get("observations") or [])
    else:
        memory = list(context.get("memory") or runtime_config.get("memory") or [])
        observations = []
    memory_max_entries = _bounded_runtime_int(
        runtime_config,
        "game_memory_max_entries",
        DEFAULT_GAME_MEMORY_MAX_ENTRIES,
        minimum=1,
        maximum=100,
    )
    observations_max_entries = _bounded_runtime_int(
        runtime_config,
        "game_observations_max_entries",
        DEFAULT_GAME_OBSERVATIONS_MAX_ENTRIES,
        minimum=1,
        maximum=50,
    )
    observation_max_chars = _bounded_runtime_int(
        runtime_config,
        "game_observation_max_chars",
        DEFAULT_GAME_OBSERVATION_MAX_CHARS,
        minimum=128,
        maximum=20000,
    )
    previous_response_max_chars = _bounded_runtime_int(
        runtime_config,
        "game_previous_response_max_chars",
        DEFAULT_GAME_PREVIOUS_RESPONSE_MAX_CHARS,
        minimum=128,
        maximum=20000,
    )
    memory_entry_max_chars = _bounded_runtime_int(
        runtime_config,
        "game_memory_entry_max_chars",
        DEFAULT_GAME_MEMORY_ENTRY_MAX_CHARS,
        minimum=32,
        maximum=4000,
    )
    memory = _keep_latest(
        [_trim_memory_entry_text(entry, memory_entry_max_chars) for entry in memory],
        memory_max_entries,
    )
    observations = _keep_latest(
        [
            _bounded_game_observation(observation, observation_max_chars)
            for observation in observations
        ],
        observations_max_entries,
    )
    scoped_memory = dict(context.get("scoped_memory") or {})
    if session.goal_id:
        from ai_hub.services.game_feature_flags import is_game_feature_enabled
        from ai_hub.services.game_memory import build_goal_memory_context

        if is_game_feature_enabled("AI_HUB_GAME_MEMORY_ENABLED"):
            scoped_memory = build_goal_memory_context(
                workspace=session.goal.workspace,
                goal=session.goal,
                session=session,
                max_chars=runtime_config.get("game_memory_max_chars", 4000),
            )
        else:
            scoped_memory = {"entries": [], "disabled": True}
    previous_response = {}
    final_answer = ""
    finish_reason = ""
    response_contract = _game_response_contract(runtime_config)
    _update_game_context(
        context,
        goal_text=goal_text,
        memory=memory,
        observations=observations,
        scoped_memory=scoped_memory,
        game_action_dispatch_enabled=use_dispatcher,
        response_contract=response_contract,
    )

    from ai_hub.services.game_policy import check_budget_before_iteration, BudgetExhaustedError

    for iteration in range(start_order, start_order + max_iterations):
        try:
            check_budget_before_iteration(session)
        except BudgetExhaustedError:
            finish_reason = "budget_exhausted"
            break

        step_run = _create_game_step_run(session, entry_agent, iteration)
        started = time.perf_counter()
        payload = {
            **context,
            "goal": goal_text,
            "goal_text": goal_text,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "memory": memory,
            "scoped_memory": scoped_memory,
            "observations": observations,
            "previous_response": previous_response,
            "available_actions": runtime_config.get("available_actions", []),
            "game_policy": runtime_config.get("policy", {}),
            "game_response_contract": response_contract,
        }

        try:
            prepared_payload = prepare_agent_payload(
                entry_agent,
                payload,
                runtime_config.get("input_mapping") or {},
                workspace=_session_workspace(session),
            )
            prepared_payload = _restore_game_reserved_payload(prepared_payload, payload)
            request_payload = copy.deepcopy(prepared_payload)
            tool_policy = (
                TOOL_POLICY_ALL
                if (
                    agent_tool_runtime == AGENT_TOOL_RUNTIME_LEGACY
                    and allow_legacy_game_action_tools
                )
                else TOOL_POLICY_GAME_CONTEXT_ONLY
            )
            output_payload = _execute_session_agent(
                session=session,
                step_run=step_run,
                agent=entry_agent,
                payload=prepared_payload,
                agent_tool_runtime=agent_tool_runtime,
                tool_policy=tool_policy,
            )
            observation = _game_observation(output_payload, runtime_config)
            action_run = None
            if use_dispatcher and not observation["complete"]:
                action_run = _dispatch_observation_action(
                    session, observation, entry_agent, iteration, step_run
                )
            observations.append(
                _bounded_game_observation(observation, observation_max_chars)
            )
            observations = _keep_latest(
                observations,
                observations_max_entries,
            )
            memory_entry = {
                "iteration": iteration,
                "action": observation.get("action"),
                "status": observation.get("status"),
                "summary": str(
                    observation.get("final_answer")
                    or observation.get("decision", {}).get("message", "")
                )[:memory_entry_max_chars],
            }
            if "action_output" in observation:
                action_out = observation["action_output"]
                memory_entry["action_output_summary"] = str(
                    action_out.get("knowledge_context")
                    or action_out.get("content")
                    or action_out.get("final_answer")
                    or ""
                )[:memory_entry_max_chars]
            memory.append(memory_entry)
            memory = _keep_latest(memory, memory_max_entries)
            previous_response = _bounded_prompt_value(
                output_payload,
                max_chars=previous_response_max_chars,
                kind="previous_response",
                reference={"agent": output_payload.get("agent")},
            )
            if (
                session.goal_id
                and action_run is not None
                and action_run.status == GameActionRun.Status.SUCCESS
                and action_run.action_name == "record_memory"
            ):
                scoped_memory = build_goal_memory_context(
                    workspace=session.goal.workspace,
                    goal=session.goal,
                    session=session,
                    max_chars=runtime_config.get("game_memory_max_chars", 4000),
                )
            _update_game_context(
                context,
                goal_text=goal_text,
                memory=memory,
                observations=observations,
                scoped_memory=scoped_memory,
                final_answer=observation.get("final_answer", ""),
                finish_reason="agent_finished" if observation["complete"] else "",
                response_contract=response_contract,
            )

            step_run.status = ExecutionStepRun.Status.SUCCESS
            step_run.request_payload = request_payload
            step_run.response_payload = output_payload
            step_run.observation_payload = observation
            if observation["complete"]:
                final_answer = observation.get("final_answer", "")
                finish_reason = "agent_finished"
                break
            if action_run and action_run.status == GameActionRun.Status.WAITING_APPROVAL:
                finish_reason = "needs_approval"
                break
            if session.runtime_mode == ExecutionSession.RuntimeMode.HYBRID:
                finish_reason = "waiting_async_continuation"
                break
        except Exception as exc:
            step_run.status = ExecutionStepRun.Status.FAILED
            step_run.error_detail = str(exc)
            step_run.request_payload = locals().get("request_payload", locals().get("prepared_payload", payload))
            _update_game_context(
                context,
                goal_text=goal_text,
                memory=memory,
                observations=observations,
                scoped_memory=scoped_memory,
                final_answer=final_answer,
                finish_reason="failed",
                failed_iteration=iteration,
                last_error=str(exc),
                response_contract=response_contract,
            )
            raise
        finally:
            step_run.latency_ms = int((time.perf_counter() - started) * 1000)
            step_run.save(
                update_fields=[
                    "status",
                    "request_payload",
                    "response_payload",
                    "observation_payload",
                    "error_detail",
                    "latency_ms",
                ]
            )

    if finish_reason in {"waiting_async_continuation", "needs_approval", "needs_information"}:
        session.status = ExecutionSession.Status.WAITING_ASYNC
        session.finished_at = None
        execution_outcome = "waiting"
        goal_outcome = "unknown"
    else:
        session.status = ExecutionSession.Status.SUCCESS
        session.finished_at = timezone.now()
        if not finish_reason:
            finish_reason = "max_iterations"
        execution_outcome = "completed"
        goal_outcome = "achieved" if finish_reason == "agent_finished" else "incomplete"

    _update_game_context(
        context,
        goal_text=goal_text,
        memory=memory,
        observations=observations,
        scoped_memory=scoped_memory,
        final_answer=final_answer,
        finish_reason=finish_reason,
        execution_outcome=execution_outcome,
        goal_outcome=goal_outcome,
        waiting_reason=(
            finish_reason
            if finish_reason in {"needs_approval", "needs_information"}
            else ""
        ),
        response_contract=response_contract,
    )
    session.final_context = context
    session.error_detail = ""
    session.save(update_fields=["status", "final_context", "error_detail", "finished_at", "updated_at"])


def run_game_session_resume(
    session_id: int,
    *,
    next_order: int,
    use_action_dispatcher: bool = False,
) -> int:
    """Re-enter a GAME session that was paused. Session must already be in RUNNING state."""
    session = (
        ExecutionSession.objects.select_related("pipeline", "entry_agent", "goal", "goal__workspace")
        .prefetch_related(
            "pipeline__steps__agent__tools",
            "pipeline__steps__agent__knowledge_collections__documents",
        )
        .get(pk=session_id)
    )
    if session.status != ExecutionSession.Status.RUNNING:
        raise ValidationError(
            f"Session #{session_id} must be RUNNING before resume "
            f"(current: '{session.status}')."
        )
    context = dict(session.final_context or {})
    try:
        if session.runtime_kind == ExecutionSession.RuntimeKind.GAME:
            _run_game_session(
                session,
                context,
                start_order=next_order,
                use_action_dispatcher=use_action_dispatcher,
            )
        else:
            raise ValidationError("Only GAME sessions can be resumed.")
    except Exception as exc:
        _mark_session_failed(session, context, exc)

    if session.goal_id:
        from ai_hub.services.game_goal_outcomes import apply_session_outcome_to_goal
        apply_session_outcome_to_goal(session)

    return session.id


def _run_orchestrator_session(session: ExecutionSession, context: dict) -> None:
    runtime_config = _session_runtime_config(session)
    agent_tool_runtime = _resolve_agent_tool_runtime(session, runtime_config)
    if not session.pipeline_id:
        raise ValidationError("Execution sessions require a pipeline before they can run.")
    if not session.pipeline.is_active:
        raise ValidationError("Pipeline must be active before it can run.")
    if session.step_runs.exists():
        raise ValidationError("Execution session already has step runs.")

    session.status = ExecutionSession.Status.RUNNING
    session.final_context = context
    session.save(update_fields=["status", "started_at", "final_context", "updated_at"])

    validate_payload(context, session.pipeline.global_input_contract or {}, "Pipeline input")
    steps = list(session.pipeline.steps.order_by("order"))
    if not steps:
        raise ValidationError("Pipeline has no steps.")

    waiting_async = False
    for index, step in enumerate(steps):
        step_run = _create_step_run(session, step)
        started = time.perf_counter()
        logical_payload = {}
        logical_payload_ready = False
        payload = {}
        try:
            logical_payload = apply_mapping(context, step.input_mapping or {})
            logical_payload_ready = True
            payload = prepare_agent_payload(
                step.agent,
                copy.deepcopy(logical_payload),
                {},
            )
            output_payload = _execute_session_agent(
                session=session,
                step_run=step_run,
                agent=step.agent,
                payload=payload,
                agent_tool_runtime=agent_tool_runtime,
                tool_policy=TOOL_POLICY_ALL,
            )
            mapped_output = _apply_step_output_mapping(
                output_payload,
                step.output_mapping or {},
            )
            context.update(mapped_output)
            step_run.status = ExecutionStepRun.Status.SUCCESS
            step_run.request_payload = payload
            step_run.response_payload = output_payload
        except Exception as exc:
            step_run.status = ExecutionStepRun.Status.FAILED
            step_run.error_detail = str(exc)
            step_run.request_payload = payload
            if step.on_error == step.OnError.CONTINUE:
                pass
            elif (
                step.on_error == step.OnError.FALLBACK_AGENT
                and step.fallback_agent
                and logical_payload_ready
            ):
                primary_error = exc
                step_run.agent = step.fallback_agent
                step_run.request_payload = {}
                try:
                    fallback_payload = prepare_agent_payload(
                        step.fallback_agent,
                        copy.deepcopy(logical_payload),
                        {},
                    )
                    step_run.request_payload = fallback_payload
                    fallback_output = _execute_session_agent(
                        session=session,
                        step_run=step_run,
                        agent=step.fallback_agent,
                        payload=fallback_payload,
                        agent_tool_runtime=agent_tool_runtime,
                        tool_policy=TOOL_POLICY_ALL,
                    )
                    mapped_output = _apply_step_output_mapping(
                        fallback_output,
                        step.output_mapping or {},
                    )
                except Exception as fallback_error:
                    step_run.status = ExecutionStepRun.Status.FAILED
                    step_run.error_detail = str(fallback_error)
                    step_run.response_payload = {
                        "fallback_for": step.agent.name,
                        "fallback_recovery": _fallback_recovery_metadata(
                            primary_agent=step.agent,
                            primary_error=primary_error,
                            fallback_agent=step.fallback_agent,
                            fallback_status="failed",
                            fallback_error=fallback_error,
                            final_outcome="failed",
                        ),
                    }
                    raise
                context.update(mapped_output)
                step_run.status = ExecutionStepRun.Status.SUCCESS
                step_run.error_detail = ""
                step_run.response_payload = {
                    **fallback_output,
                    "fallback_for": step.agent.name,
                    "fallback_recovery": _fallback_recovery_metadata(
                        primary_agent=step.agent,
                        primary_error=primary_error,
                        fallback_agent=step.fallback_agent,
                        fallback_status="success",
                        final_outcome="recovered",
                    ),
                }
            else:
                raise
        finally:
            step_run.latency_ms = int((time.perf_counter() - started) * 1000)
            step_run.save(
                update_fields=[
                    "agent",
                    "status",
                    "request_payload",
                    "response_payload",
                    "error_detail",
                    "latency_ms",
                ]
            )

        if session.runtime_mode == ExecutionSession.RuntimeMode.HYBRID and index == 0:
            waiting_async = len(steps) > 1
            break

    if waiting_async:
        session.status = ExecutionSession.Status.WAITING_ASYNC
        session.finished_at = None
    else:
        _merge_final_output_object(context)
        _raise_final_output_parse_error_if_required(context, session.pipeline.global_output_contract or {})
        validate_payload(context, session.pipeline.global_output_contract or {}, "Pipeline output")
        session.status = ExecutionSession.Status.SUCCESS
        session.finished_at = timezone.now()
    session.final_context = context
    session.error_detail = ""
    session.save(update_fields=["status", "final_context", "error_detail", "finished_at", "updated_at"])


def run_execution_session(
    session_id: int,
    *,
    allow_legacy_game_action_tools: bool = False,
    use_action_dispatcher: bool = False,
) -> int:
    # The request handler or background worker owns its connection lifecycle.
    # Closing here breaks caller-owned atomic blocks and PostgreSQL transactions.
    _claim_session_for_run(session_id)
    session = (
        ExecutionSession.objects.select_related("pipeline", "entry_agent", "goal", "goal__workspace")
        .prefetch_related(
            "pipeline__steps__agent__tools",
            "pipeline__steps__agent__knowledge_collections__documents",
            "pipeline__steps__fallback_agent__tools",
            "pipeline__steps__fallback_agent__knowledge_collections__documents",
        )
        .get(pk=session_id)
    )
    context = dict(session.initial_context or {})

    try:
        if session.runtime_kind == ExecutionSession.RuntimeKind.GAME:
            _run_game_session(
                session,
                context,
                allow_legacy_game_action_tools=allow_legacy_game_action_tools,
                use_action_dispatcher=use_action_dispatcher,
            )
        else:
            _run_orchestrator_session(session, context)
    except Exception as exc:
        _mark_session_failed(session, context, exc)

    if session.goal_id:
        from ai_hub.services.game_goal_outcomes import apply_session_outcome_to_goal

        apply_session_outcome_to_goal(session)

    return session.id
