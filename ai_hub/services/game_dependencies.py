from django.core.exceptions import ValidationError
from django.db import transaction

from ai_hub.models import GameGoal, GameGoalDependency, GameWorkspace


@transaction.atomic
def add_goal_dependency(
    goal: GameGoal,
    depends_on: GameGoal,
    *,
    is_required: bool = True,
    note: str = "",
) -> GameGoalDependency:
    if goal.pk == depends_on.pk:
        raise ValidationError("A goal cannot depend on itself.")
    workspace_ids = {goal.workspace_id, depends_on.workspace_id}
    list(GameWorkspace.objects.select_for_update().filter(pk__in=workspace_ids).order_by("pk"))
    if len(workspace_ids) != 1:
        raise ValidationError("Goal dependencies must belong to the same workspace.")
    locked_goals = {
        item.pk: item
        for item in GameGoal.objects.select_for_update().filter(pk__in={goal.pk, depends_on.pk})
    }
    if len(locked_goals) != 2:
        raise ValidationError("Both GAME goals must exist before adding a dependency.")

    dependency = GameGoalDependency(
        goal=locked_goals[goal.pk],
        depends_on=locked_goals[depends_on.pk],
        is_required=is_required,
        note=note,
    )
    dependency.full_clean()
    dependency.save()
    return dependency


@transaction.atomic
def remove_goal_dependency(goal: GameGoal, depends_on: GameGoal) -> bool:
    deleted, _ = GameGoalDependency.objects.filter(goal=goal, depends_on=depends_on).delete()
    return bool(deleted)


def get_goal_blockers(goal: GameGoal) -> list[dict]:
    dependencies = goal.dependencies.filter(is_required=True).select_related("depends_on")
    return [
        {
            "goal_id": dependency.depends_on_id,
            "title": dependency.depends_on.title,
            "status": dependency.depends_on.status,
        }
        for dependency in dependencies
        if dependency.depends_on.status != GameGoal.Status.COMPLETED
    ]
