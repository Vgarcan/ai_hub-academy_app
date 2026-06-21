from django.db import transaction
from django.db.models import Exists, OuterRef
from django.utils import timezone

from ai_hub.models import GameGoal, GameGoalDependency, GameWorkspace
from ai_hub.services.game_feature_flags import require_game_feature
from ai_hub.services.game_goals import transition_goal_status
from ai_hub.services.game_priority import calculate_goal_priority


def _eligible_goals(workspace_id: int):
    unresolved_required_dependency = (
        GameGoalDependency.objects.filter(goal_id=OuterRef("pk"), is_required=True)
        .exclude(depends_on__status=GameGoal.Status.COMPLETED)
    )
    return (
        GameGoal.objects.filter(workspace_id=workspace_id, status=GameGoal.Status.QUEUED)
        .annotate(has_unresolved_required_dependency=Exists(unresolved_required_dependency))
        .filter(has_unresolved_required_dependency=False)
    )


def _ranked_goals(goals, *, now):
    scored = [(calculate_goal_priority(goal, now=now), goal) for goal in goals]
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].due_at is None,
            item[1].due_at or now,
            item[1].created_at,
            item[1].pk,
        )
    )
    return scored


@transaction.atomic
def refresh_workspace_goal_priorities(workspace_id: int, *, now=None) -> list[GameGoal]:
    now = now or timezone.now()
    goals = list(GameGoal.objects.select_for_update().filter(workspace_id=workspace_id))
    for goal in goals:
        goal.calculated_priority = calculate_goal_priority(goal, now=now)
        goal.updated_at = now
    if goals:
        GameGoal.objects.bulk_update(goals, ["calculated_priority", "updated_at"])
    return goals


def get_next_eligible_goal(workspace_id: int, *, now=None) -> GameGoal | None:
    now = now or timezone.now()
    if not GameWorkspace.objects.filter(pk=workspace_id, is_active=True).exists():
        return None
    ranked = _ranked_goals(list(_eligible_goals(workspace_id)), now=now)
    if not ranked:
        return None
    score, goal = ranked[0]
    goal.calculated_priority = score
    return goal


@transaction.atomic
def claim_next_goal(workspace_id: int, *, actor=None, now=None) -> GameGoal | None:
    require_game_feature("AI_HUB_GAME_SCHEDULER_ENABLED")
    now = now or timezone.now()
    workspace = GameWorkspace.objects.select_for_update().filter(pk=workspace_id, is_active=True).first()
    if not workspace:
        return None

    candidates = list(_eligible_goals(workspace_id).select_for_update())
    ranked = _ranked_goals(candidates, now=now)
    if not ranked:
        return None

    for score, candidate in ranked:
        candidate.calculated_priority = score
        candidate.updated_at = now
    GameGoal.objects.bulk_update(
        [candidate for _, candidate in ranked],
        ["calculated_priority", "updated_at"],
    )

    goal = ranked[0][1]
    actor_label = str(actor) if actor is not None else "scheduler"
    return transition_goal_status(goal, GameGoal.Status.RUNNING, reason=f"claimed by {actor_label}")
