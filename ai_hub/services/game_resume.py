import time

from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from ai_hub.services.game_feature_flags import require_game_feature
from ai_hub.models import (
    ExecutionSession,
    GameActionApprovalRequest,
    GameActionRun,
    GameContinuationRequest,
    GameGoal,
)
from ai_hub.services.game_goals import transition_goal_status


_REASON_TO_GOAL_STATUS = {
    GameContinuationRequest.ReasonCode.NEEDS_INFORMATION: GameGoal.Status.WAITING_INFO,
    GameContinuationRequest.ReasonCode.NEEDS_APPROVAL: GameGoal.Status.WAITING_APPROVAL,
    GameContinuationRequest.ReasonCode.EXTERNAL_RESULT_PENDING: GameGoal.Status.WAITING_INFO,
    GameContinuationRequest.ReasonCode.RATE_LIMITED: GameGoal.Status.BLOCKED,
    GameContinuationRequest.ReasonCode.MANUAL_REVIEW_REQUIRED: GameGoal.Status.WAITING_APPROVAL,
}

_WAITING_GOAL_STATUSES = frozenset(
    {
        GameGoal.Status.WAITING_INFO,
        GameGoal.Status.WAITING_APPROVAL,
        GameGoal.Status.BLOCKED,
    }
)


@transaction.atomic
def pause_session(*, session, goal, reason_code, detail="", payload=None):
    """Pause one goal-bound session and create exactly one pending continuation."""
    if reason_code not in GameContinuationRequest.ReasonCode.values:
        raise ValidationError(f"Unknown GAME continuation reason '{reason_code}'.")

    locked_session = ExecutionSession.objects.select_for_update().get(pk=session.pk)
    locked_goal = GameGoal.objects.select_for_update().get(pk=goal.pk)
    if locked_session.goal_id != locked_goal.pk:
        raise ValidationError("Paused session and goal must refer to the same GAME goal.")

    existing = GameContinuationRequest.objects.select_for_update().filter(
        session=locked_session,
        status=GameContinuationRequest.Status.PENDING,
    ).first()
    if existing is not None:
        raise ValidationError(
            f"Session #{locked_session.pk} already has pending continuation request "
            f"#{existing.pk}."
        )

    locked_session.status = ExecutionSession.Status.WAITING_ASYNC
    locked_session.finished_at = None
    locked_session.save(update_fields=["status", "finished_at", "updated_at"])

    target_goal_status = _REASON_TO_GOAL_STATUS[reason_code]
    if locked_goal.status not in _WAITING_GOAL_STATUSES:
        locked_goal = transition_goal_status(
            locked_goal,
            target_goal_status,
            reason=f"paused: {reason_code}",
        )

    return GameContinuationRequest.objects.create(
        session=locked_session,
        goal=locked_goal,
        reason_code=reason_code,
        detail=detail,
        payload=payload or {},
    )


def resume_goal_execution(
    *,
    session_id: int,
    continuation_payload: dict | None = None,
    resolved_by=None,
) -> ExecutionSession:
    """Resolve a valid continuation and resume at the next unused step order."""
    require_game_feature("AI_HUB_GAME_RESUME_ENABLED")
    with transaction.atomic():
        session = (
            ExecutionSession.objects.select_for_update()
            .get(pk=session_id)
        )
        goal = (
            GameGoal.objects.select_for_update().select_related("workspace").get(pk=session.goal_id)
            if session.goal_id
            else None
        )

        if session.status != ExecutionSession.Status.WAITING_ASYNC:
            raise ValidationError(
                f"Session #{session_id} cannot be resumed (current status: '{session.status}')."
            )
        if goal is None:
            raise ValidationError("Cannot resume a session that is not linked to a GAME goal.")
        if goal.status == GameGoal.Status.CANCELLED:
            raise ValidationError(
                f"Cannot resume a session for a cancelled goal (goal #{goal.pk})."
            )

        cont_req = (
            GameContinuationRequest.objects.select_for_update()
            .filter(session=session, status=GameContinuationRequest.Status.PENDING)
            .order_by("-created_at")
            .first()
        )
        if cont_req is None:
            raise ValidationError(
                f"No pending continuation request found for session #{session_id}."
            )

        if cont_req.reason_code == GameContinuationRequest.ReasonCode.NEEDS_APPROVAL:
            action_run_id = (cont_req.payload or {}).get("action_run_id")
            if not action_run_id:
                raise ValidationError("Approval continuation is missing its action run reference.")
            try:
                approval = GameActionApprovalRequest.objects.select_for_update().get(
                    action_run_id=action_run_id,
                    action_run__session=session,
                )
            except GameActionApprovalRequest.DoesNotExist as exc:
                raise ValidationError(
                    "Approval continuation has no matching approval request."
                ) from exc
            if approval.status not in {
                GameActionApprovalRequest.Status.APPROVED,
                GameActionApprovalRequest.Status.REJECTED,
            }:
                raise ValidationError(
                    "Approval request must be approved or rejected before resume "
                    f"(current status: '{approval.status}')."
                )

        existing_context = dict(session.final_context or {})
        if continuation_payload:
            existing_context["continuation_payload"] = continuation_payload
        session.final_context = existing_context
        session.status = ExecutionSession.Status.RUNNING
        session.save(update_fields=["status", "final_context", "updated_at"])

        cont_req.status = GameContinuationRequest.Status.RESOLVED
        cont_req.resolved_at = timezone.now()
        cont_req.resolved_by = resolved_by
        cont_req.save(update_fields=["status", "resolved_at", "resolved_by"])

        if goal.status in _WAITING_GOAL_STATUSES:
            transition_goal_status(
                goal,
                GameGoal.Status.RUNNING,
                reason=f"resumed by continuation #{cont_req.pk}",
            )

        max_order = session.step_runs.aggregate(Max("order"))["order__max"] or 0
        next_order = max_order + 1

    from ai_hub.services.execution_runner import run_game_session_resume

    run_game_session_resume(session_id, next_order=next_order)
    session.refresh_from_db()
    return session


def _append_action_resolution(*, action_run: GameActionRun, status: str, payload: dict) -> None:
    """Persist one action-resolution observation for the resumed parent agent."""
    with transaction.atomic():
        session = ExecutionSession.objects.select_for_update().get(pk=action_run.session_id)
        context = dict(session.final_context or {})
        observations = list(context.get("observations") or [])
        already_recorded = any(
            isinstance(item, dict)
            and item.get("action_run_id") == action_run.pk
            and item.get("resolution_status") == status
            for item in observations
        )
        if not already_recorded:
            observations.append(
                {
                    "action_run_id": action_run.pk,
                    "action": action_run.action_name,
                    "resolution_status": status,
                    **payload,
                }
            )
        context["observations"] = observations
        session.final_context = context
        session.save(update_fields=["final_context", "updated_at"])


def approve_action_run(*, action_run_id: int, reviewed_by, review_note: str = "") -> GameActionRun:
    """Atomically claim one pending approval, then execute its already-audited action."""
    if reviewed_by is None or not reviewed_by.has_perm("ai_hub.approve_game_action"):
        raise ValidationError("You do not have permission to approve GAME action requests.")
    # Approval ultimately dispatches the action; refuse the whole operation when the
    # dispatch kill-switch is off so we never leave an action approved-but-unexecuted.
    require_game_feature("AI_HUB_GAME_ACTION_DISPATCH_ENABLED")

    session_id = GameActionRun.objects.only("session_id").get(pk=action_run_id).session_id
    expired = False
    with transaction.atomic():
        locked_session = ExecutionSession.objects.select_for_update().get(pk=session_id)
        action_run = (
            GameActionRun.objects.select_for_update()
            .select_related("action")
            .get(pk=action_run_id)
        )
        approval_req = GameActionApprovalRequest.objects.select_for_update().get(
            action_run=action_run
        )
        if action_run.status != GameActionRun.Status.WAITING_APPROVAL:
            raise ValidationError(
                f"Action run #{action_run_id} is not awaiting approval "
                f"(current status: '{action_run.status}')."
            )
        if approval_req.status != GameActionApprovalRequest.Status.PENDING:
            raise ValidationError(
                f"Approval request is not pending (status: '{approval_req.status}')."
            )

        if approval_req.expires_at and approval_req.expires_at < timezone.now():
            approval_req.status = GameActionApprovalRequest.Status.EXPIRED
            approval_req.save(update_fields=["status"])
            expired = True
        else:
            approval_req.status = GameActionApprovalRequest.Status.APPROVED
            approval_req.reviewed_by = reviewed_by
            approval_req.review_note = review_note or ""
            approval_req.reviewed_at = timezone.now()
            approval_req.save(
                update_fields=["status", "reviewed_by", "review_note", "reviewed_at"]
            )
            action_run.status = GameActionRun.Status.RUNNING
            action_run.started_at = timezone.now()
            action_run.save(update_fields=["status", "started_at"])

    if expired:
        raise ValidationError("This approval request has expired. Submit a new request to proceed.")

    session = locked_session
    goal = GameGoal.objects.select_related("workspace").get(pk=session.goal_id) if session.goal_id else None
    workspace = goal.workspace if goal else None
    start = time.perf_counter()
    try:
        if workspace is not None:
            from ai_hub.services.game_policy import (
                ApprovalRequiredByPolicyError,
                validate_action_policy,
            )

            try:
                validate_action_policy(
                    workspace,
                    goal,
                    action_run.action,
                    dict(action_run.input_payload or {}),
                )
            except ApprovalRequiredByPolicyError:
                pass

        from ai_hub.services.game_action_dispatcher import dispatch_game_action

        with transaction.atomic():
            output = dispatch_game_action(
                action_run=action_run,
                workspace=workspace,
                goal=goal,
                payload=dict(action_run.input_payload or {}),
            )
        action_run.status = GameActionRun.Status.SUCCESS
        action_run.output_payload = output
        action_run.observation_payload = {
            "action_name": action_run.action_name,
            "action_type": action_run.action.action_type,
            "complete": output.get("complete", False),
        }
        action_run.error_detail = ""
        action_run.finished_at = timezone.now()
        action_run.latency_ms = int((time.perf_counter() - start) * 1000)
        action_run.save(
            update_fields=[
                "status",
                "output_payload",
                "observation_payload",
                "error_detail",
                "finished_at",
                "latency_ms",
            ]
        )
        _append_action_resolution(
            action_run=action_run,
            status="approved",
            payload={
                "action_output": output,
                "message": "Approved action executed successfully.",
            },
        )
    except Exception as exc:
        action_run.status = GameActionRun.Status.FAILED
        action_run.error_detail = str(exc)
        action_run.finished_at = timezone.now()
        action_run.latency_ms = int((time.perf_counter() - start) * 1000)
        action_run.save(
            update_fields=["status", "error_detail", "finished_at", "latency_ms"]
        )
        _append_action_resolution(
            action_run=action_run,
            status="failed_after_approval",
            payload={"action_error": str(exc)},
        )
        raise

    return action_run


@transaction.atomic
def reject_action_run(
    *, action_run_id: int, reviewed_by, review_note: str = ""
) -> tuple[GameActionRun, dict]:
    """Reject one pending action and persist the rejection for the resumed agent."""
    if reviewed_by is None or not reviewed_by.has_perm("ai_hub.approve_game_action"):
        raise ValidationError("You do not have permission to reject GAME action requests.")

    session_id = GameActionRun.objects.only("session_id").get(pk=action_run_id).session_id
    ExecutionSession.objects.select_for_update().get(pk=session_id)
    action_run = (
        GameActionRun.objects.select_for_update()
        .select_related("action")
        .get(pk=action_run_id)
    )
    if action_run.status != GameActionRun.Status.WAITING_APPROVAL:
        raise ValidationError(
            f"Action run #{action_run_id} is not awaiting approval "
            f"(current status: '{action_run.status}')."
        )

    approval_req = GameActionApprovalRequest.objects.select_for_update().get(
        action_run=action_run
    )
    if approval_req.status != GameActionApprovalRequest.Status.PENDING:
        raise ValidationError(
            f"Approval request is not pending (status: '{approval_req.status}')."
        )

    approval_req.status = GameActionApprovalRequest.Status.REJECTED
    approval_req.reviewed_by = reviewed_by
    approval_req.review_note = review_note or ""
    approval_req.reviewed_at = timezone.now()
    approval_req.save(
        update_fields=["status", "reviewed_by", "review_note", "reviewed_at"]
    )

    action_run.status = GameActionRun.Status.REJECTED
    action_run.error_detail = f"Rejected by {reviewed_by}: {review_note or 'No reason given.'}"
    action_run.finished_at = timezone.now()
    action_run.save(update_fields=["status", "error_detail", "finished_at"])

    rejection_observation = {
        "action_name": action_run.action_name,
        "status": "rejected",
        "review_note": review_note or "",
        "message": (
            f"Action '{action_run.action_name}' was rejected by a reviewer. "
            "Choose an alternative action or finish with the information available."
        ),
    }
    _append_action_resolution(
        action_run=action_run,
        status="rejected",
        payload=rejection_observation,
    )
    return action_run, rejection_observation
