import time
from dataclasses import dataclass

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

APPROVAL_EXPIRED_CODE = "APPROVAL_EXPIRED"
APPROVAL_EXPIRED_MESSAGE = (
    f"{APPROVAL_EXPIRED_CODE}: This approval request has expired. "
    "The requested action was not executed."
)


@dataclass(frozen=True)
class _ExpiryResolution:
    action_run_id: int
    session_id: int
    next_order: int | None
    should_resume: bool


@dataclass
class _LockedApprovalLifecycle:
    session: ExecutionSession
    action_run: GameActionRun
    approval: GameActionApprovalRequest
    goal: GameGoal | None
    workspace: object | None
    continuation: GameContinuationRequest | None


def _matching_approval_continuation(
    *,
    session: ExecutionSession,
    action_run_id: int,
) -> GameContinuationRequest | None:
    """Lock the continuation belonging to one approval lifecycle.

    Real dispatcher-created continuations carry ``action_run_id``. The
    payload-less fallback preserves compatibility with older/manual rows while
    the database constraint still guarantees at most one pending continuation
    per session.
    """
    continuations = list(
        GameContinuationRequest.objects.select_for_update()
        .filter(
            session=session,
            reason_code=GameContinuationRequest.ReasonCode.NEEDS_APPROVAL,
            status__in=[
                GameContinuationRequest.Status.PENDING,
                GameContinuationRequest.Status.EXPIRED,
            ],
        )
        .order_by("-created_at")
    )
    for continuation in continuations:
        if (continuation.payload or {}).get("action_run_id") == action_run_id:
            return continuation
    for continuation in continuations:
        if (
            continuation.status == GameContinuationRequest.Status.PENDING
            and not (continuation.payload or {}).get("action_run_id")
        ):
            return continuation
    return None


def _lock_approval_lifecycle(action_run_id: int) -> _LockedApprovalLifecycle:
    """Lock one approval lifecycle in the shared authoritative order.

    Lock order:
    ExecutionSession -> GameActionRun -> GameActionDefinition ->
    GameActionApprovalRequest -> GameGoal -> GameWorkspace ->
    GameContinuationRequest.
    """
    session_id = (
        GameActionRun.objects.only("session_id")
        .get(pk=action_run_id)
        .session_id
    )
    session = ExecutionSession.objects.select_for_update().get(pk=session_id)
    action_run = GameActionRun.objects.select_for_update().get(pk=action_run_id)

    from ai_hub.models import GameActionDefinition, GameWorkspace

    action_run.action = GameActionDefinition.objects.select_for_update().get(
        pk=action_run.action_id
    )
    approval = GameActionApprovalRequest.objects.select_for_update().get(
        action_run=action_run
    )
    goal = (
        GameGoal.objects.select_for_update().get(pk=session.goal_id)
        if session.goal_id
        else None
    )
    workspace = (
        GameWorkspace.objects.select_for_update().get(pk=goal.workspace_id)
        if goal is not None
        else None
    )
    continuation = _matching_approval_continuation(
        session=session,
        action_run_id=action_run.pk,
    )
    return _LockedApprovalLifecycle(
        session=session,
        action_run=action_run,
        approval=approval,
        goal=goal,
        workspace=workspace,
        continuation=continuation,
    )


def _append_action_resolution_to_session(
    *,
    session: ExecutionSession,
    action_run: GameActionRun,
    status: str,
    payload: dict,
) -> None:
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


def _finalize_expired_approval_locked(
    lifecycle: _LockedApprovalLifecycle,
    *,
    now=None,
) -> _ExpiryResolution | None:
    """Atomically close an expired lifecycle without executing its action."""
    approval = lifecycle.approval
    action_run = lifecycle.action_run
    session = lifecycle.session
    goal = lifecycle.goal
    continuation = lifecycle.continuation
    resolution_time = now or timezone.now()

    is_expired = approval.status == GameActionApprovalRequest.Status.EXPIRED
    deadline_passed = (
        approval.status == GameActionApprovalRequest.Status.PENDING
        and approval.expires_at is not None
        and approval.expires_at <= resolution_time
    )
    if not (is_expired or deadline_passed):
        return None

    if action_run.status not in {
        GameActionRun.Status.WAITING_APPROVAL,
        GameActionRun.Status.FAILED,
    }:
        raise ValidationError(
            "Expired approval cannot be reconciled because its action run has "
            f"terminal status '{action_run.status}'."
        )
    if (
        session.status == ExecutionSession.Status.WAITING_ASYNC
        and continuation is None
    ):
        raise ValidationError(
            "Expired approval cannot be reconciled without its matching "
            "approval continuation."
        )

    if approval.status == GameActionApprovalRequest.Status.PENDING:
        approval.status = GameActionApprovalRequest.Status.EXPIRED
        approval.save(update_fields=["status"])

    if action_run.status == GameActionRun.Status.WAITING_APPROVAL:
        action_run.status = GameActionRun.Status.FAILED
        action_run.error_detail = APPROVAL_EXPIRED_MESSAGE
        action_run.finished_at = resolution_time
        action_run.observation_payload = {
            "action_name": action_run.action_name,
            "action_type": action_run.action.action_type,
            "resolution_status": "approval_expired",
            "complete": False,
        }
        action_run.save(
            update_fields=[
                "status",
                "error_detail",
                "finished_at",
                "observation_payload",
            ]
        )

    continuation_closes_lifecycle = continuation is not None and continuation.status in {
        GameContinuationRequest.Status.PENDING,
        GameContinuationRequest.Status.EXPIRED,
    }
    if (
        continuation is not None
        and continuation.status == GameContinuationRequest.Status.PENDING
    ):
        continuation.status = GameContinuationRequest.Status.EXPIRED
        continuation.resolved_at = resolution_time
        continuation.save(update_fields=["status", "resolved_at"])

    _append_action_resolution_to_session(
        session=session,
        action_run=action_run,
        status="approval_expired",
        payload={
            "message": (
                "The requested action was not executed because approval expired. "
                "Choose a new action, request fresh approval, ask for information, "
                "or finish with the information available."
            ),
        },
    )

    has_other_pending_continuation = GameContinuationRequest.objects.filter(
        session=session,
        status=GameContinuationRequest.Status.PENDING,
    ).exclude(
        pk=continuation.pk if continuation is not None else None
    ).exists()
    should_resume = (
        continuation_closes_lifecycle
        and not has_other_pending_continuation
        and session.status == ExecutionSession.Status.WAITING_ASYNC
        and goal is not None
        and goal.status != GameGoal.Status.CANCELLED
    )
    next_order = None
    session_update_fields = ["final_context", "updated_at"]
    if should_resume:
        session.status = ExecutionSession.Status.RUNNING
        session.finished_at = None
        session_update_fields.extend(["status", "finished_at"])
        if goal.status in _WAITING_GOAL_STATUSES:
            transition_goal_status(
                goal,
                GameGoal.Status.RUNNING,
                reason=f"approval #{approval.pk} expired",
            )
        next_order = (session.step_runs.aggregate(Max("order"))["order__max"] or 0) + 1
    elif (
        session.status == ExecutionSession.Status.WAITING_ASYNC
        and goal is not None
        and goal.status == GameGoal.Status.CANCELLED
    ):
        session.status = ExecutionSession.Status.CANCELLED
        session.finished_at = resolution_time
        session_update_fields.extend(["status", "finished_at"])
    session.save(update_fields=list(dict.fromkeys(session_update_fields)))

    return _ExpiryResolution(
        action_run_id=action_run.pk,
        session_id=session.pk,
        next_order=next_order,
        should_resume=should_resume,
    )


def _resume_expired_approval(resolution: _ExpiryResolution) -> None:
    if not resolution.should_resume or resolution.next_order is None:
        return
    from ai_hub.services.execution_runner import run_game_session_resume

    run_game_session_resume(
        resolution.session_id,
        next_order=resolution.next_order,
    )


def finalize_expired_approval(
    *,
    action_run_id: int,
) -> _ExpiryResolution | None:
    """Finalize one expired approval and resume its parent outside DB locks."""
    with transaction.atomic():
        lifecycle = _lock_approval_lifecycle(action_run_id)
        resolution = _finalize_expired_approval_locked(lifecycle)
    if resolution is not None:
        _resume_expired_approval(resolution)
    return resolution


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
    expiry_resolution = None
    next_order = None
    with transaction.atomic():
        session = ExecutionSession.objects.select_for_update().get(pk=session_id)
        if session.status != ExecutionSession.Status.WAITING_ASYNC:
            raise ValidationError(
                f"Session #{session_id} cannot be resumed (current status: '{session.status}')."
            )

        # The Session lock serializes pause/approve/reject/expiry/resume. Read a
        # hint first so approval continuations can then acquire their remaining
        # rows in the shared lifecycle lock order.
        cont_req_hint = (
            GameContinuationRequest.objects
            .filter(session=session, status=GameContinuationRequest.Status.PENDING)
            .order_by("-created_at")
            .first()
        )
        if cont_req_hint is None:
            # Reconcile a historical partially closed expiry where the
            # continuation was expired but the parent Session stayed waiting.
            cont_req_hint = (
                GameContinuationRequest.objects.filter(
                    session=session,
                    status=GameContinuationRequest.Status.EXPIRED,
                    reason_code=GameContinuationRequest.ReasonCode.NEEDS_APPROVAL,
                )
                .order_by("-created_at")
                .first()
            )
        if cont_req_hint is None:
            raise ValidationError(
                f"No pending continuation request found for session #{session_id}."
            )

        if (
            cont_req_hint.reason_code
            == GameContinuationRequest.ReasonCode.NEEDS_APPROVAL
        ):
            action_run_id = (cont_req_hint.payload or {}).get("action_run_id")
            if not action_run_id:
                raise ValidationError(
                    "Approval continuation is missing its action run reference."
                )
            try:
                lifecycle = _lock_approval_lifecycle(action_run_id)
            except (
                GameActionRun.DoesNotExist,
                GameActionApprovalRequest.DoesNotExist,
            ) as exc:
                raise ValidationError(
                    "Approval continuation has no matching approval request."
                ) from exc
            if lifecycle.session.pk != session.pk:
                raise ValidationError(
                    "Approval continuation does not belong to this GAME session."
                )
            if (
                lifecycle.continuation is None
                or lifecycle.continuation.pk != cont_req_hint.pk
            ):
                raise ValidationError(
                    "Approval continuation does not match its action run."
                )

            expiry_resolution = _finalize_expired_approval_locked(lifecycle)
            if expiry_resolution is None:
                approval = lifecycle.approval
                if approval.status not in {
                    GameActionApprovalRequest.Status.APPROVED,
                    GameActionApprovalRequest.Status.REJECTED,
                }:
                    raise ValidationError(
                        "Approval request must be approved or rejected before resume "
                        f"(current status: '{approval.status}')."
                    )
                goal = lifecycle.goal
                cont_req = lifecycle.continuation
                session = lifecycle.session
            else:
                goal = lifecycle.goal
                cont_req = lifecycle.continuation
        else:
            goal = (
                GameGoal.objects.select_for_update()
                .select_related("workspace")
                .get(pk=session.goal_id)
                if session.goal_id
                else None
            )
            cont_req = GameContinuationRequest.objects.select_for_update().get(
                pk=cont_req_hint.pk
            )

        if expiry_resolution is None:
            if goal is None:
                raise ValidationError(
                    "Cannot resume a session that is not linked to a GAME goal."
                )
            if goal.status == GameGoal.Status.CANCELLED:
                raise ValidationError(
                    f"Cannot resume a session for a cancelled goal (goal #{goal.pk})."
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

    if expiry_resolution is not None:
        _resume_expired_approval(expiry_resolution)
    else:
        from ai_hub.services.execution_runner import run_game_session_resume

        run_game_session_resume(session_id, next_order=next_order)
    session.refresh_from_db()
    return session


def _append_action_resolution(*, action_run: GameActionRun, status: str, payload: dict) -> None:
    """Persist one action-resolution observation for the resumed parent agent."""
    with transaction.atomic():
        session = ExecutionSession.objects.select_for_update().get(pk=action_run.session_id)
        _append_action_resolution_to_session(
            session=session,
            action_run=action_run,
            status=status,
            payload=payload,
        )
        session.save(update_fields=["final_context", "updated_at"])


def _reject_dispatch_time_approval_drift(
    *,
    action_run_id: int,
    reviewed_by,
    review_note: str,
    error_detail: str,
) -> GameActionRun:
    """Fail a previously approved run back to the resumable reapproval state."""
    with transaction.atomic():
        action_run = GameActionRun.objects.select_for_update().get(pk=action_run_id)
        approval_req = GameActionApprovalRequest.objects.select_for_update().get(
            action_run=action_run
        )
        reviewed_at = timezone.now()
        approval_req.status = GameActionApprovalRequest.Status.REJECTED
        approval_req.reviewed_by = reviewed_by
        approval_req.review_note = review_note or ""
        approval_req.reviewed_at = reviewed_at
        approval_req.save(
            update_fields=[
                "status",
                "reviewed_by",
                "review_note",
                "reviewed_at",
            ]
        )
        action_run.status = GameActionRun.Status.REJECTED
        action_run.error_detail = error_detail
        action_run.finished_at = reviewed_at
        action_run.save(
            update_fields=["status", "error_detail", "finished_at"]
        )
    return action_run


def approve_action_run(*, action_run_id: int, reviewed_by, review_note: str = "") -> GameActionRun:
    """Atomically claim one pending approval, then execute its already-audited action."""
    if reviewed_by is None or not reviewed_by.has_perm("ai_hub.approve_game_action"):
        raise ValidationError("You do not have permission to approve GAME action requests.")

    expiry_resolution = None
    reapproval_required = False
    reapproval_message = (
        "APPROVAL_REAPPROVAL_REQUIRED: the reviewed execution intent or its current "
        "authorization changed. Submit a fresh action request."
    )
    with transaction.atomic():
        lifecycle = _lock_approval_lifecycle(action_run_id)
        locked_session = lifecycle.session
        action_run = lifecycle.action_run
        approval_req = lifecycle.approval
        goal = lifecycle.goal
        workspace = lifecycle.workspace
        expiry_resolution = _finalize_expired_approval_locked(lifecycle)

        if expiry_resolution is None:
            # Approval ultimately dispatches the action; refuse the operation
            # when the kill-switch is off so no action is left approved but
            # unexecuted. Expiry closure itself never dispatches and remains
            # available even while dispatch is disabled.
            require_game_feature("AI_HUB_GAME_ACTION_DISPATCH_ENABLED")
            if action_run.status != GameActionRun.Status.WAITING_APPROVAL:
                raise ValidationError(
                    f"Action run #{action_run_id} is not awaiting approval "
                    f"(current status: '{action_run.status}')."
                )
            if approval_req.status != GameActionApprovalRequest.Status.PENDING:
                raise ValidationError(
                    f"Approval request is not pending (status: '{approval_req.status}')."
                )

            try:
                from ai_hub.services.game_action_dispatcher import (
                    build_game_action_approval_intent,
                )
                from ai_hub.services.game_policy import check_budget_before_action

                _, current_fingerprint = build_game_action_approval_intent(
                    session=locked_session,
                    workspace=workspace,
                    goal=goal,
                    action_definition=action_run.action,
                    payload=dict(action_run.input_payload or {}),
                )
                if (
                    not approval_req.execution_intent_fingerprint
                    or current_fingerprint
                    != approval_req.execution_intent_fingerprint
                ):
                    raise ValidationError(reapproval_message)
                check_budget_before_action(
                    locked_session,
                    action_run.action,
                    action_run=action_run,
                )
            except ValidationError:
                reviewed_at = timezone.now()
                approval_req.status = GameActionApprovalRequest.Status.REJECTED
                approval_req.reviewed_by = reviewed_by
                approval_req.review_note = review_note or ""
                approval_req.reviewed_at = reviewed_at
                approval_req.save(
                    update_fields=[
                        "status",
                        "reviewed_by",
                        "review_note",
                        "reviewed_at",
                    ]
                )
                action_run.status = GameActionRun.Status.REJECTED
                action_run.error_detail = reapproval_message
                action_run.finished_at = reviewed_at
                action_run.save(
                    update_fields=["status", "error_detail", "finished_at"]
                )
                reapproval_required = True
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

    if expiry_resolution is not None:
        _resume_expired_approval(expiry_resolution)
        raise ValidationError(APPROVAL_EXPIRED_MESSAGE)
    if reapproval_required:
        _append_action_resolution(
            action_run=action_run,
            status="reapproval_required",
            payload={
                "message": (
                    "The requested action was not executed because its reviewed "
                    "intent or current authorization changed."
                ),
            },
        )
        raise ValidationError(reapproval_message)

    session = locked_session
    start = time.perf_counter()
    try:
        from ai_hub.services.game_action_dispatcher import dispatch_game_action

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
        from ai_hub.services.game_action_dispatcher import REAPPROVAL_REQUIRED_CODE

        if (
            isinstance(exc, ValidationError)
            and REAPPROVAL_REQUIRED_CODE in str(exc)
        ):
            action_run = _reject_dispatch_time_approval_drift(
                action_run_id=action_run_id,
                reviewed_by=reviewed_by,
                review_note=review_note,
                error_detail=str(exc),
            )
            _append_action_resolution(
                action_run=action_run,
                status="reapproval_required",
                payload={
                    "message": (
                        "The requested action was not executed because its reviewed "
                        "intent or current authorization changed at dispatch."
                    ),
                },
            )
            raise
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


def reject_action_run(
    *, action_run_id: int, reviewed_by, review_note: str = ""
) -> tuple[GameActionRun, dict]:
    """Reject one pending action and persist the rejection for the resumed agent."""
    if reviewed_by is None or not reviewed_by.has_perm("ai_hub.approve_game_action"):
        raise ValidationError("You do not have permission to reject GAME action requests.")

    expiry_resolution = None
    rejection_observation = None
    with transaction.atomic():
        lifecycle = _lock_approval_lifecycle(action_run_id)
        action_run = lifecycle.action_run
        approval_req = lifecycle.approval
        expiry_resolution = _finalize_expired_approval_locked(lifecycle)

        if expiry_resolution is None:
            if action_run.status != GameActionRun.Status.WAITING_APPROVAL:
                raise ValidationError(
                    f"Action run #{action_run_id} is not awaiting approval "
                    f"(current status: '{action_run.status}')."
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
            action_run.error_detail = (
                f"Rejected by {reviewed_by}: {review_note or 'No reason given.'}"
            )
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
            _append_action_resolution_to_session(
                session=lifecycle.session,
                action_run=action_run,
                status="rejected",
                payload=rejection_observation,
            )
            lifecycle.session.save(update_fields=["final_context", "updated_at"])

    if expiry_resolution is not None:
        _resume_expired_approval(expiry_resolution)
        raise ValidationError(APPROVAL_EXPIRED_MESSAGE)

    return action_run, rejection_observation
