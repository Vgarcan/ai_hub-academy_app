import hashlib
import json
import time
from decimal import Decimal

from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from ai_hub.models import ExecutionSession, GameActionDefinition, GameActionRun
from ai_hub.services.contracts import validate_payload
from ai_hub.services.game_feature_flags import require_game_feature


def _resolve_action_definition(action_name: str) -> GameActionDefinition:
    safe_name = str(action_name or "").strip()[:120]
    try:
        return GameActionDefinition.objects.get(name=safe_name, is_active=True)
    except GameActionDefinition.DoesNotExist:
        raise ValidationError(f"Unknown or inactive GAME action '{safe_name}'.")


def _validate_action_input(action_definition: GameActionDefinition, raw_input) -> dict:
    if not isinstance(raw_input, dict):
        raise ValidationError(
            f"Action '{action_definition.name}' input must be a JSON object."
        )
    validate_payload(
        raw_input,
        action_definition.input_contract or {},
        f"Action '{action_definition.name}' input",
    )
    return raw_input


def _validate_action_output(action_definition: GameActionDefinition, output: dict) -> None:
    if not action_definition.output_contract:
        return
    validate_payload(
        output,
        action_definition.output_contract,
        f"Action '{action_definition.name}' output",
    )


def _build_idempotency_key(
    session_id: int,
    step_run_id,
    action_id: int,
    input_payload: dict,
) -> str:
    canonical = json.dumps(
        {
            "session_id": session_id,
            "step_run_id": step_run_id,
            "action_id": action_id,
            "input": input_payload,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---- internal handlers -------------------------------------------------------

def _handle_finish_goal(action_run: GameActionRun, workspace, goal, payload: dict) -> dict:
    return {
        "action_name": "finish_goal",
        "final_answer": str(payload.get("final_answer") or ""),
        "message": str(payload.get("message") or "Goal completed."),
        "complete": True,
    }


def _handle_update_goal_status(action_run: GameActionRun, workspace, goal, payload: dict) -> dict:
    if goal is None:
        raise ValidationError("update_goal_status requires a goal linked to the session.")

    from ai_hub.models import GameGoal
    from ai_hub.services.game_goals import transition_goal_status

    target = str(payload.get("target_status") or "").strip().lower()
    allowed = {
        GameGoal.Status.WAITING_INFO,
        GameGoal.Status.WAITING_APPROVAL,
        GameGoal.Status.BLOCKED,
    }
    if target not in {s.value for s in allowed}:
        raise ValidationError(
            f"update_goal_status only supports: "
            f"{', '.join(s.value for s in allowed)}. Got: '{target}'."
        )

    previous_status = goal.status
    goal.refresh_from_db()
    updated_goal = transition_goal_status(goal, target)
    return {
        "action_name": "update_goal_status",
        "previous_status": previous_status,
        "new_status": updated_goal.status,
        "goal_id": goal.pk,
    }


def _handle_record_memory(action_run: GameActionRun, workspace, goal, payload: dict) -> dict:
    from ai_hub.models import GameMemoryEntry
    from ai_hub.services.game_memory import record_memory

    scope_type = str(payload.get("scope_type") or GameMemoryEntry.ScopeType.GOAL)
    content = str(payload.get("content") or "").strip()
    if not content:
        raise ValidationError("record_memory requires non-empty 'content'.")

    try:
        importance_score = Decimal(str(payload.get("importance_score", "0.50")))
    except Exception:
        importance_score = Decimal("0.50")

    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("source", "action_run")
    metadata.setdefault("source_id", action_run.pk)
    metadata.setdefault("model_derived", True)

    entry = record_memory(
        scope_type=scope_type,
        workspace=workspace,
        content=content,
        goal=goal,
        session=action_run.session,
        metadata=metadata,
        importance_score=importance_score,
    )
    return {
        "action_name": "record_memory",
        "memory_entry_id": entry.pk,
        "scope_type": scope_type,
        "content_preview": content[:200],
    }


# ---- context tool handlers ---------------------------------------------------

def _handle_search_knowledge(action_run: GameActionRun, workspace, goal, payload: dict) -> dict:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValidationError("search_knowledge requires a non-empty 'query'.")

    entry_agent = action_run.session.entry_agent
    if not entry_agent:
        return {
            "action_name": "search_knowledge",
            "query": query,
            "knowledge_context": [],
            "matched_documents": 0,
        }

    query_lower = query.lower()
    results = []
    max_chars = 4000
    remaining = max_chars

    collections = entry_agent.knowledge_collections.filter(is_active=True).prefetch_related("documents")
    for collection in collections:
        for doc in collection.documents.filter(status="active").order_by("title"):
            title_match = query_lower in doc.title.lower()
            tag_match = any(query_lower in str(t).lower() for t in (doc.tags or []))
            content_text = doc.curated_text or ""
            content_match = query_lower in content_text.lower()
            if not (title_match or tag_match or content_match):
                continue
            snippet = content_text[:remaining] if remaining > 0 else ""
            if not snippet and not title_match and not tag_match:
                continue
            results.append({
                "document_id": doc.pk,
                "title": doc.title,
                "collection": collection.name,
                "snippet": snippet,
            })
            remaining -= len(snippet)
            if remaining <= 0:
                break
        if remaining <= 0:
            break

    return {
        "action_name": "search_knowledge",
        "query": query,
        "knowledge_context": results,
        "matched_documents": len(results),
    }


def _handle_read_document(action_run: GameActionRun, workspace, goal, payload: dict) -> dict:
    raw_id = payload.get("document_id")
    try:
        document_id = int(raw_id)
    except (TypeError, ValueError):
        raise ValidationError("read_document requires a numeric 'document_id'.")

    from ai_hub.models import KnowledgeDocument

    entry_agent = action_run.session.entry_agent
    allowed_collection_ids: set = set()
    if entry_agent:
        allowed_collection_ids = set(
            entry_agent.knowledge_collections.filter(is_active=True).values_list("id", flat=True)
        )

    try:
        doc = KnowledgeDocument.objects.get(
            pk=document_id,
            collection_id__in=allowed_collection_ids,
            status=KnowledgeDocument.Status.ACTIVE,
        )
    except KnowledgeDocument.DoesNotExist:
        raise ValidationError(
            f"Document {document_id} not found or not accessible in this session's knowledge collections."
        )

    return {
        "action_name": "read_document",
        "document_id": doc.pk,
        "title": doc.title,
        "content": doc.curated_text[:8000],
    }


# ---- sub-agent handlers ------------------------------------------------------

def _handle_delegate_to_agent(action_run: GameActionRun, workspace, goal, payload: dict) -> dict:
    from ai_hub.services.game_delegation import run_delegated_agent

    if goal is None:
        raise ValidationError("delegate_to_agent requires a goal-linked session.")
    if workspace is None:
        raise ValidationError("delegate_to_agent requires a workspace-linked session.")

    agent_name = str(payload.get("agent_name") or "").strip()
    task = str(payload.get("task") or "").strip()
    expected_result = str(payload.get("expected_result") or "")

    if not agent_name:
        raise ValidationError("delegate_to_agent requires 'agent_name'.")
    if not task:
        raise ValidationError("delegate_to_agent requires 'task'.")

    return run_delegated_agent(
        session=action_run.session,
        action_run=action_run,
        workspace=workspace,
        goal=goal,
        target_agent_name=agent_name,
        task=task,
        expected_result=expected_result,
    )


# ---- handler registry --------------------------------------------------------

_INTERNAL_HANDLERS = {
    "finish_goal": _handle_finish_goal,
    "update_goal_status": _handle_update_goal_status,
    "record_memory": _handle_record_memory,
}
_CONTEXT_TOOL_HANDLERS = {
    "search_knowledge": _handle_search_knowledge,
    "read_document": _handle_read_document,
}
_SUB_AGENT_HANDLERS = {
    "delegate_to_agent": _handle_delegate_to_agent,
}
_HANDLER_REGISTRY = {
    GameActionDefinition.ActionType.INTERNAL: _INTERNAL_HANDLERS,
    GameActionDefinition.ActionType.CONTEXT_TOOL: _CONTEXT_TOOL_HANDLERS,
    GameActionDefinition.ActionType.SUB_AGENT: _SUB_AGENT_HANDLERS,
}


def dispatch_game_action(
    *,
    action_run: GameActionRun,
    workspace,
    goal,
    payload: dict,
) -> dict:
    """Execute one action and return the output dict. Caller manages audit record lifecycle."""
    action_definition = action_run.action

    type_handlers = _HANDLER_REGISTRY.get(action_definition.action_type)
    if type_handlers is None:
        raise ValidationError(
            f"Action type '{action_definition.action_type}' is not implemented in this phase."
        )

    handler_fn = type_handlers.get(action_definition.name)
    if handler_fn is None:
        raise ValidationError(
            f"No handler registered for action '{action_definition.name}' "
            f"(type: {action_definition.action_type})."
        )

    output = handler_fn(action_run, workspace, goal, payload)
    _validate_action_output(action_definition, output)
    return output


def execute_game_action(
    *,
    session: ExecutionSession,
    step_run=None,
    action_name: str,
    action_input: dict,
    workspace=None,
    goal=None,
) -> GameActionRun:
    """Audit, validate, and execute exactly one idempotent GAME action."""
    require_game_feature("AI_HUB_GAME_ACTION_DISPATCH_ENABLED")
    safe_name = str(action_name or "").strip()[:120]
    action_definition = _resolve_action_definition(safe_name)

    if workspace is None and session.goal_id:
        workspace = session.goal.workspace
    if goal is None and session.goal_id:
        goal = session.goal

    step_run_id = step_run.pk if step_run is not None else None
    audit_input = action_input if isinstance(action_input, dict) else {"_invalid_input": action_input}
    idempotency_key = _build_idempotency_key(
        session.pk, step_run_id, action_definition.pk, audit_input
    )

    existing = GameActionRun.objects.filter(idempotency_key=idempotency_key).first()
    if existing:
        if existing.status in {
            GameActionRun.Status.SUCCESS,
            GameActionRun.Status.WAITING_APPROVAL,
        }:
            return existing
        if existing.status in {GameActionRun.Status.RUNNING, GameActionRun.Status.PENDING}:
            raise ValidationError(f"A run for action '{safe_name}' is already in progress.")
        raise ValidationError(
            f"An equivalent action run already ended with status '{existing.status}'. "
            "Automatic identical retries are not permitted; retry from a new step or change the input."
        )

    iteration = (
        step_run.order
        if step_run is not None
        else GameActionRun.objects.filter(session=session).count() + 1
    )

    try:
        with transaction.atomic():
            action_run = GameActionRun.objects.create(
                session=session,
                step_run=step_run,
                action=action_definition,
                idempotency_key=idempotency_key,
                action_name=safe_name,
                iteration=iteration,
                status=GameActionRun.Status.PENDING,
                input_payload=audit_input,
            )
    except Exception as exc:
        concurrent = GameActionRun.objects.filter(idempotency_key=idempotency_key).first()
        if concurrent is None:
            raise
        if concurrent.status in {
            GameActionRun.Status.SUCCESS,
            GameActionRun.Status.WAITING_APPROVAL,
        }:
            return concurrent
        raise ValidationError(
            f"A run for action '{safe_name}' already exists with status "
            f"'{concurrent.status}'."
        ) from exc

    policy_requires_approval = False
    try:
        validated_input = _validate_action_input(action_definition, action_input)
        action_run.input_payload = validated_input
        action_run.save(update_fields=["input_payload"])
        if workspace is not None:
            from ai_hub.services.game_policy import (
                ApprovalRequiredByPolicyError,
                check_budget_before_action,
                validate_action_policy,
            )

            try:
                validate_action_policy(workspace, goal, action_definition, validated_input)
            except ApprovalRequiredByPolicyError:
                policy_requires_approval = True
            check_budget_before_action(
                session,
                action_definition,
                action_run=action_run,
            )
    except Exception as exc:
        action_run.status = GameActionRun.Status.FAILED
        action_run.error_detail = str(exc)
        action_run.finished_at = timezone.now()
        action_run.save(update_fields=["status", "error_detail", "finished_at"])
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(str(exc)) from exc

    # Actions that require human approval (by definition or by workspace policy) create an
    # approval request and pause the session.
    if action_definition.requires_approval or policy_requires_approval:
        from ai_hub.models import GameActionApprovalRequest
        from ai_hub.services.game_resume import pause_session

        goal = goal or (session.goal if session.goal_id else None)
        if goal is None:
            exc = ValidationError(
                f"Action '{safe_name}' requires approval but the session has no linked goal."
            )
            action_run.status = GameActionRun.Status.FAILED
            action_run.error_detail = str(exc)
            action_run.finished_at = timezone.now()
            action_run.save(update_fields=["status", "error_detail", "finished_at"])
            raise exc

        try:
            with transaction.atomic():
                action_run.status = GameActionRun.Status.WAITING_APPROVAL
                action_run.save(update_fields=["status"])
                GameActionApprovalRequest.objects.create(
                    action_run=action_run,
                    goal=goal,
                    requested_payload=dict(validated_input),
                )
                pause_session(
                    session=session,
                    goal=goal,
                    reason_code="needs_approval",
                    detail=f"Action '{safe_name}' requires human approval.",
                    payload={"action_run_id": action_run.pk, "action_name": safe_name},
                )
        except Exception as exc:
            action_run.refresh_from_db()
            action_run.status = GameActionRun.Status.FAILED
            action_run.error_detail = str(exc)
            action_run.finished_at = timezone.now()
            action_run.save(update_fields=["status", "error_detail", "finished_at"])
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(str(exc)) from exc
        return action_run

    action_run.status = GameActionRun.Status.RUNNING
    action_run.started_at = timezone.now()
    action_run.save(update_fields=["status", "started_at"])
    start = time.perf_counter()
    try:
        with transaction.atomic():
            output = dispatch_game_action(
                action_run=action_run,
                workspace=workspace,
                goal=goal,
                payload=validated_input,
            )
        action_run.status = GameActionRun.Status.SUCCESS
        action_run.output_payload = output
        action_run.observation_payload = {
            "action_name": safe_name,
            "action_type": action_definition.action_type,
            "complete": output.get("complete", False),
        }
        action_run.finished_at = timezone.now()
        action_run.latency_ms = int((time.perf_counter() - start) * 1000)
        action_run.save(
            update_fields=[
                "status", "output_payload", "observation_payload",
                "finished_at", "latency_ms",
            ]
        )
    except ValidationError as exc:
        action_run.status = GameActionRun.Status.FAILED
        action_run.error_detail = str(exc)
        action_run.finished_at = timezone.now()
        action_run.latency_ms = int((time.perf_counter() - start) * 1000)
        action_run.save(update_fields=["status", "error_detail", "finished_at", "latency_ms"])
        raise
    except Exception as exc:
        action_run.status = GameActionRun.Status.FAILED
        action_run.error_detail = str(exc)
        action_run.finished_at = timezone.now()
        action_run.latency_ms = int((time.perf_counter() - start) * 1000)
        action_run.save(update_fields=["status", "error_detail", "finished_at", "latency_ms"])
        raise ValidationError(str(exc)) from exc

    return action_run
