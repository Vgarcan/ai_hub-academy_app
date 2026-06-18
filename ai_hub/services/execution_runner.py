import copy
import json
import time

from django.core.exceptions import ValidationError
from django.db import close_old_connections, transaction
from django.utils import timezone

from ai_hub.models import ExecutionSession, ExecutionStepRun
from ai_hub.services.agent_runtime import apply_mapping, execute_agent, prepare_agent_payload
from ai_hub.services.contracts import validate_payload


DEFAULT_GAME_FINISH_ACTIONS = ["finish", "final", "complete", "stop"]
DEFAULT_GAME_STOP_STATUSES = ["success", "complete", "completed", "done", "final"]
GAME_RESERVED_PAYLOAD_KEYS = (
    "goal",
    "goal_text",
    "iteration",
    "max_iterations",
    "memory",
    "observations",
    "previous_response",
    "available_actions",
    "game_policy",
    "game_response_contract",
)


def _create_step_run(session: ExecutionSession, step) -> ExecutionStepRun:
    return ExecutionStepRun.objects.create(
        session=session,
        order=step.order,
        pipeline_step=step,
        agent=step.agent,
        action_name="agent_call",
        status=ExecutionStepRun.Status.RUNNING,
    )


def _create_game_step_run(session: ExecutionSession, agent, order: int) -> ExecutionStepRun:
    return ExecutionStepRun.objects.create(
        session=session,
        order=order,
        agent=agent,
        action_name="game_iteration",
        status=ExecutionStepRun.Status.RUNNING,
    )


def _mark_session_failed(session: ExecutionSession, context: dict, error: Exception) -> None:
    session.status = ExecutionSession.Status.FAILED
    session.error_detail = str(error)
    session.final_context = context
    session.finished_at = timezone.now()
    session.save(update_fields=["status", "error_detail", "final_context", "finished_at", "updated_at"])


def _claim_session_for_run(session_id: int) -> None:
    with transaction.atomic():
        session = ExecutionSession.objects.select_for_update().get(pk=session_id)
        if session.status == ExecutionSession.Status.RUNNING:
            raise ValidationError("Execution session is already running.")
        if session.status == ExecutionSession.Status.WAITING_ASYNC:
            raise ValidationError("Execution session is waiting for async continuation.")
        if session.status == ExecutionSession.Status.SUCCESS:
            raise ValidationError("Execution session has already completed.")
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
        "actions": _available_action_names(runtime_config),
        "schema": {
            "action": "string; one of actions. Use 'think' to continue or 'finish' to complete.",
            "message": "string; short explanation of the decision or observation.",
            "complete": "boolean; true only when the goal is done.",
            "final_answer": "string; empty until complete, then the final answer/result.",
        },
        "example_continue": {
            "action": "think",
            "message": "I need one more iteration to compare the available context.",
            "complete": False,
            "final_answer": "",
        },
        "example_finish": {
            "action": "finish",
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


def _resolve_game_entry_agent(session: ExecutionSession):
    if session.entry_agent_id:
        return session.entry_agent
    if session.pipeline_id:
        first_step = session.pipeline.steps.select_related("agent").order_by("order").first()
        if first_step:
            return first_step.agent
    return None


def _run_game_session(session: ExecutionSession, context: dict) -> None:
    runtime_config = dict(session.runtime_config or {})
    entry_agent = _resolve_game_entry_agent(session)
    if not entry_agent:
        raise ValidationError("GAME sessions require an entry agent or a pipeline with at least one step.")
    if not entry_agent.is_active:
        raise ValidationError("GAME entry agent must be active before it can run.")
    if session.step_runs.exists():
        raise ValidationError("Execution session already has step runs.")

    max_iterations = _bounded_iteration_count(runtime_config)
    goal_text = session.goal_text or runtime_config.get("goal") or ""
    if not goal_text:
        raise ValidationError("GAME sessions require goal_text or runtime_config.goal.")

    session.status = ExecutionSession.Status.RUNNING
    session.final_context = context
    session.save(update_fields=["status", "started_at", "final_context", "updated_at"])

    memory = list(context.get("memory") or runtime_config.get("memory") or [])
    observations = []
    previous_response = {}
    final_answer = ""
    finish_reason = ""
    response_contract = _game_response_contract(runtime_config)
    _update_game_context(
        context,
        goal_text=goal_text,
        memory=memory,
        observations=observations,
        response_contract=response_contract,
    )

    for iteration in range(1, max_iterations + 1):
        step_run = _create_game_step_run(session, entry_agent, iteration)
        started = time.perf_counter()
        payload = {
            **context,
            "goal": goal_text,
            "goal_text": goal_text,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "memory": memory,
            "observations": observations,
            "previous_response": previous_response,
            "available_actions": runtime_config.get("available_actions", []),
            "game_policy": runtime_config.get("policy", {}),
            "game_response_contract": response_contract,
        }

        try:
            prepared_payload = prepare_agent_payload(entry_agent, payload, runtime_config.get("input_mapping") or {})
            prepared_payload = _restore_game_reserved_payload(prepared_payload, payload)
            request_payload = copy.deepcopy(prepared_payload)
            output_payload = execute_agent(entry_agent, prepared_payload)
            observation = _game_observation(output_payload, runtime_config)
            observations.append(observation)
            memory.append(
                {
                    "iteration": iteration,
                    "action": observation.get("action"),
                    "status": observation.get("status"),
                    "summary": observation.get("final_answer") or observation.get("decision", {}).get("message", ""),
                }
            )
            previous_response = output_payload
            _update_game_context(
                context,
                goal_text=goal_text,
                memory=memory,
                observations=observations,
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

    if finish_reason == "waiting_async_continuation":
        session.status = ExecutionSession.Status.WAITING_ASYNC
        session.finished_at = None
    else:
        session.status = ExecutionSession.Status.SUCCESS
        session.finished_at = timezone.now()
        if not finish_reason:
            finish_reason = "max_iterations"

    _update_game_context(
        context,
        goal_text=goal_text,
        memory=memory,
        observations=observations,
        final_answer=final_answer,
        finish_reason=finish_reason,
        response_contract=response_contract,
    )
    session.final_context = context
    session.error_detail = ""
    session.save(update_fields=["status", "final_context", "error_detail", "finished_at", "updated_at"])


def _run_orchestrator_session(session: ExecutionSession, context: dict) -> None:
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
        try:
            payload = prepare_agent_payload(step.agent, context, step.input_mapping or {})
            output_payload = execute_agent(step.agent, payload)
            context.update(apply_mapping(output_payload, step.output_mapping or {}))
            step_run.status = ExecutionStepRun.Status.SUCCESS
            step_run.request_payload = payload
            step_run.response_payload = output_payload
        except Exception as exc:
            step_run.status = ExecutionStepRun.Status.FAILED
            step_run.error_detail = str(exc)
            step_run.request_payload = locals().get("payload", {})
            if step.on_error == step.OnError.CONTINUE:
                pass
            elif step.on_error == step.OnError.FALLBACK_AGENT and step.fallback_agent:
                fallback_payload = dict(step_run.request_payload)
                fallback_payload["knowledge_context"] = prepare_agent_payload(
                    step.fallback_agent,
                    context,
                    {},
                )["knowledge_context"]
                output_payload = execute_agent(step.fallback_agent, fallback_payload)
                context.update(apply_mapping(output_payload, step.output_mapping or {}))
                step_run.status = ExecutionStepRun.Status.SUCCESS
                step_run.response_payload = {
                    "fallback_for": step.agent.name,
                    **output_payload,
                }
            else:
                raise
        finally:
            step_run.latency_ms = int((time.perf_counter() - started) * 1000)
            step_run.save(
                update_fields=[
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


def run_execution_session(session_id: int) -> int:
    close_old_connections()
    try:
        _claim_session_for_run(session_id)
        session = (
            ExecutionSession.objects.select_related("pipeline", "entry_agent")
            .prefetch_related(
                "pipeline__steps__agent__tools",
                "pipeline__steps__agent__knowledge_collections__documents",
            )
            .get(pk=session_id)
        )
        context = dict(session.initial_context or {})

        try:
            if session.runtime_kind == ExecutionSession.RuntimeKind.GAME:
                _run_game_session(session, context)
            else:
                _run_orchestrator_session(session, context)
        except Exception as exc:
            _mark_session_failed(session, context, exc)

        return session.id
    finally:
        close_old_connections()
