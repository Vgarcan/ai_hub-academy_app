import hashlib
import json

from django.core.exceptions import ValidationError
from django.db import transaction

from ai_hub.models import ExecutionSession, GameGoal
from ai_hub.services.game_goals import transition_goal_status


def _goal_result_payload(session: ExecutionSession) -> dict:
    final_context = dict(session.final_context or {})
    explicit_result = final_context.get("result")
    if isinstance(explicit_result, dict):
        return explicit_result
    payload = {"session_id": session.pk}
    if final_context.get("final_answer"):
        payload["final_answer"] = final_context["final_answer"]
    return payload


def _outcome_fingerprint(session: ExecutionSession) -> str:
    payload = {
        "session_status": session.status,
        "execution_outcome": (session.final_context or {}).get("execution_outcome"),
        "goal_outcome": (session.final_context or {}).get("goal_outcome"),
        "finish_reason": (session.final_context or {}).get("finish_reason"),
        "waiting_reason": (session.final_context or {}).get("waiting_reason"),
        "result": (session.final_context or {}).get("result"),
        "final_answer": (session.final_context or {}).get("final_answer"),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@transaction.atomic
def apply_session_outcome_to_goal(session: ExecutionSession) -> GameGoal | None:
    locked_session = (
        ExecutionSession.objects.select_for_update()
        .get(pk=session.pk)
    )
    if not locked_session.goal_id:
        return None

    fingerprint = _outcome_fingerprint(locked_session)
    if locked_session.goal_outcome_fingerprint == fingerprint:
        return locked_session.goal

    goal = GameGoal.objects.select_for_update().get(pk=locked_session.goal_id)
    context = dict(locked_session.final_context or {})
    execution_outcome = context.get("execution_outcome")
    goal_outcome = context.get("goal_outcome")
    finish_reason = context.get("waiting_reason") or context.get("finish_reason")

    target_status = None
    result = None
    if goal_outcome == "achieved":
        target_status = GameGoal.Status.COMPLETED
        result = _goal_result_payload(locked_session)
    elif goal_outcome == "incomplete" and finish_reason in {"max_iterations", "budget_exhausted"}:
        target_status = GameGoal.Status.PARTIAL
        result = _goal_result_payload(locked_session)
    elif execution_outcome == "waiting" and finish_reason == "needs_information":
        target_status = GameGoal.Status.WAITING_INFO
    elif execution_outcome == "waiting" and finish_reason == "needs_approval":
        target_status = GameGoal.Status.WAITING_APPROVAL
    elif execution_outcome == "failed":
        target_status = GameGoal.Status.FAILED

    if target_status is None:
        return goal
    if goal.status != target_status:
        if goal.status != GameGoal.Status.RUNNING:
            raise ValidationError(
                f"Cannot apply session #{locked_session.pk} outcome to goal in status '{goal.status}'."
            )
        goal = transition_goal_status(
            goal,
            target_status,
            reason=f"execution session #{locked_session.pk}: {finish_reason or execution_outcome or goal_outcome}",
            result=result,
        )

    locked_session.goal_outcome_fingerprint = fingerprint
    locked_session.save(update_fields=["goal_outcome_fingerprint", "updated_at"])
    return goal


def reconcile_goal_outcomes(*, limit: int | None = None) -> dict:
    sessions = ExecutionSession.objects.filter(
            goal__isnull=False,
            status__in=[
                ExecutionSession.Status.SUCCESS,
                ExecutionSession.Status.FAILED,
                ExecutionSession.Status.WAITING_ASYNC,
            ],
        ).order_by("created_at")
    if limit is not None:
        sessions = sessions[:limit]
    applied = 0
    skipped = 0
    errors = []
    for session in sessions:
        try:
            before = session.goal_outcome_fingerprint
            apply_session_outcome_to_goal(session)
            session.refresh_from_db(fields=["goal_outcome_fingerprint"])
            if session.goal_outcome_fingerprint != before:
                applied += 1
            else:
                skipped += 1
        except ValidationError as exc:
            errors.append({"session_id": session.pk, "error": exc.messages})
    return {"checked": len(sessions), "applied": applied, "skipped": skipped, "errors": errors}
