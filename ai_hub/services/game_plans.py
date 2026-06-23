from django.core.exceptions import ValidationError
from django.db import transaction

from ai_hub.models import GameGoal, GameGoalPlan, GameGoalPlanStep


@transaction.atomic
def create_plan(*, goal: GameGoal, summary: str = "") -> GameGoalPlan:
    """Return the existing plan for a goal or create a new one."""
    try:
        return goal.plan
    except GameGoalPlan.DoesNotExist:
        pass
    plan = GameGoalPlan(goal=goal, summary=summary)
    plan.full_clean()
    plan.save()
    return plan


@transaction.atomic
def add_plan_step(
    *,
    plan: GameGoalPlan,
    title: str,
    order: int,
    description: str = "",
    depends_on_step=None,
) -> GameGoalPlanStep:
    """Add a step to a plan, validating ordering and cross-plan dependency constraints."""
    if depends_on_step is not None:
        if depends_on_step.plan_id != plan.pk:
            raise ValidationError(
                f"Step dependency must belong to the same plan (plan #{plan.pk})."
            )
        if depends_on_step.order >= order:
            raise ValidationError(
                f"Dependency step order ({depends_on_step.order}) must be less than "
                f"the new step's order ({order})."
            )
    step = GameGoalPlanStep(
        plan=plan,
        title=title,
        order=order,
        description=description,
        depends_on_step=depends_on_step,
    )
    step.full_clean()
    step.save()
    return step


@transaction.atomic
def revise_plan(*, plan: GameGoalPlan, summary: str, status: str | None = None) -> GameGoalPlan:
    """Snapshot the current plan and advance its durable version atomically."""
    locked = GameGoalPlan.objects.select_for_update().get(pk=plan.pk)
    history = list(locked.revision_history or [])
    history.append(
        {
            "version": locked.version,
            "summary": locked.summary,
            "status": locked.status,
            "steps": list(
                locked.steps.order_by("order").values(
                    "order", "title", "description", "status", "depends_on_step_id"
                )
            ),
        }
    )
    locked.version += 1
    locked.summary = summary
    if status is not None:
        if status not in GameGoalPlan.Status.values:
            raise ValidationError(f"Unknown GAME plan status '{status}'.")
        locked.status = status
    locked.revision_history = history
    locked.full_clean()
    locked.save(
        update_fields=["version", "summary", "status", "revision_history", "updated_at"]
    )
    return locked
