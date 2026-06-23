from decimal import Decimal

from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from ai_hub.models import GameGoal, GameWorkspace
from ai_hub.services.game_feature_flags import require_game_feature


ALLOWED_GOAL_TRANSITIONS = {
    GameGoal.Status.DRAFT: {GameGoal.Status.QUEUED, GameGoal.Status.CANCELLED},
    GameGoal.Status.QUEUED: {GameGoal.Status.RUNNING, GameGoal.Status.BLOCKED, GameGoal.Status.CANCELLED},
    GameGoal.Status.RUNNING: {
        GameGoal.Status.WAITING_INFO,
        GameGoal.Status.WAITING_APPROVAL,
        GameGoal.Status.BLOCKED,
        GameGoal.Status.COMPLETED,
        GameGoal.Status.PARTIAL,
        GameGoal.Status.FAILED,
        GameGoal.Status.CANCELLED,
    },
    GameGoal.Status.WAITING_INFO: {GameGoal.Status.QUEUED, GameGoal.Status.RUNNING, GameGoal.Status.CANCELLED},
    GameGoal.Status.WAITING_APPROVAL: {GameGoal.Status.QUEUED, GameGoal.Status.RUNNING, GameGoal.Status.CANCELLED},
    GameGoal.Status.BLOCKED: {GameGoal.Status.QUEUED, GameGoal.Status.RUNNING, GameGoal.Status.CANCELLED},
    GameGoal.Status.PARTIAL: {GameGoal.Status.QUEUED, GameGoal.Status.COMPLETED, GameGoal.Status.CANCELLED},
    GameGoal.Status.FAILED: {GameGoal.Status.QUEUED, GameGoal.Status.CANCELLED},
    GameGoal.Status.COMPLETED: set(),
    GameGoal.Status.CANCELLED: set(),
}


@transaction.atomic
def create_goal(
    *,
    workspace: GameWorkspace,
    title: str,
    description: str,
    status: str = GameGoal.Status.QUEUED,
    base_priority: int = 50,
    due_at=None,
    success_criteria: dict | None = None,
    context: dict | None = None,
) -> GameGoal:
    require_game_feature("AI_HUB_GAME_GOALS_ENABLED")
    if status not in {GameGoal.Status.DRAFT, GameGoal.Status.QUEUED}:
        raise ValidationError("New GAME goals must start as draft or queued.")
    goal = GameGoal(
        workspace=workspace,
        title=title,
        description=description,
        status=status,
        base_priority=base_priority,
        due_at=due_at,
        success_criteria=success_criteria or {},
        context=context or {},
    )
    goal.full_clean()
    goal.save()
    return goal


@transaction.atomic
def transition_goal_status(goal: GameGoal, target_status: str, *, reason: str = "", result=None) -> GameGoal:
    locked_goal = GameGoal.objects.select_for_update().get(pk=goal.pk)
    if target_status not in GameGoal.Status.values:
        raise ValidationError(f"Unknown GAME goal status '{target_status}'.")
    if target_status == locked_goal.status:
        return locked_goal
    if target_status not in ALLOWED_GOAL_TRANSITIONS[locked_goal.status]:
        raise ValidationError(f"Cannot transition GAME goal from '{locked_goal.status}' to '{target_status}'.")

    previous_status = locked_goal.status
    locked_goal.status = target_status
    if result is not None:
        locked_goal.result = result
    transition_time = timezone.now()
    if target_status == GameGoal.Status.QUEUED:
        locked_goal.queued_at = transition_time
        locked_goal.result = {}
    locked_goal.transition_metadata = {
        "from": previous_status,
        "to": target_status,
        "reason": reason,
        "at": transition_time.isoformat(),
    }
    locked_goal.full_clean()
    locked_goal.save(
        update_fields=["status", "result", "queued_at", "transition_metadata", "updated_at"]
    )
    return locked_goal


@transaction.atomic
def reopen_goal(goal: GameGoal, *, reason: str = "") -> GameGoal:
    locked_goal = GameGoal.objects.select_for_update().get(pk=goal.pk)
    if locked_goal.status not in {GameGoal.Status.COMPLETED, GameGoal.Status.CANCELLED}:
        raise ValidationError("Only completed or cancelled GAME goals require explicit reopening.")

    previous_status = locked_goal.status
    locked_goal.status = GameGoal.Status.QUEUED
    transition_time = timezone.now()
    locked_goal.queued_at = transition_time
    locked_goal.result = {}
    locked_goal.transition_metadata = {
        "from": previous_status,
        "to": GameGoal.Status.QUEUED,
        "reason": reason,
        "at": transition_time.isoformat(),
        "reopened": True,
    }
    locked_goal.full_clean()
    locked_goal.save(
        update_fields=["status", "result", "queued_at", "transition_metadata", "updated_at"]
    )
    return locked_goal


@transaction.atomic
def update_goal_priority(goal: GameGoal, calculated_priority) -> GameGoal:
    locked_goal = GameGoal.objects.select_for_update().get(pk=goal.pk)
    locked_goal.calculated_priority = Decimal(str(calculated_priority))
    locked_goal.full_clean()
    locked_goal.save(update_fields=["calculated_priority", "updated_at"])
    return locked_goal


def find_orphaned_running_goals(*, workspace=None, older_than=None):
    """Return RUNNING goals that have no active execution session.

    A goal can stay stuck in RUNNING when its session(s) never reach a terminal
    state — e.g. interrupted runs, or stub sessions created by integration suites.
    A goal with any active session (pending/running/waiting_async) is never a
    candidate.

    Args:
        workspace: optional GameWorkspace to scope the sweep.
        older_than: optional timedelta; only goals whose ``updated_at`` predates
            ``now - older_than`` are returned.

    Returns a list of candidate GameGoal instances (no mutation).
    """
    from django.db.models import Exists, OuterRef
    from ai_hub.models import ExecutionSession

    active_session = ExecutionSession.objects.filter(
        goal_id=OuterRef("pk"),
        status__in=(
            ExecutionSession.Status.PENDING,
            ExecutionSession.Status.RUNNING,
            ExecutionSession.Status.WAITING_ASYNC,
        ),
    )
    qs = (
        GameGoal.objects.filter(status=GameGoal.Status.RUNNING)
        .annotate(has_active_session=Exists(active_session))
        .filter(has_active_session=False)
    )
    if workspace is not None:
        qs = qs.filter(workspace=workspace)
    if older_than is not None:
        qs = qs.filter(updated_at__lt=timezone.now() - older_than)
    return list(qs)


def cancel_orphaned_running_goals(*, workspace=None, older_than=None):
    """Cancel RUNNING goals that have no active execution session.

    See :func:`find_orphaned_running_goals` for the orphan definition. Each goal is
    transitioned RUNNING → CANCELLED through ``transition_goal_status`` so the
    lifecycle lock and transition metadata stay consistent. Best-effort: a goal
    that gained an active session between selection and transition is still
    cancelled, which is acceptable for a maintenance sweep.

    Returns the list of goals that were cancelled.
    """
    cancelled = []
    for goal in find_orphaned_running_goals(workspace=workspace, older_than=older_than):
        cancelled.append(
            transition_goal_status(
                goal, GameGoal.Status.CANCELLED, reason="orphaned running goal cleanup"
            )
        )
    return cancelled
