from datetime import timedelta
from decimal import Decimal

from django.utils import timezone

from ai_hub.models import GameGoal


DUE_TODAY_BONUS = Decimal("30")
DUE_TOMORROW_BONUS = Decimal("15")
OVERDUE_BONUS = Decimal("40")
UNLOCKS_DEPENDENT_BONUS = Decimal("10")
AGED_QUEUE_BONUS = Decimal("5")


def _would_unlock_queued_dependent(goal: GameGoal) -> bool:
    required_relations = goal.required_by.filter(
        is_required=True,
        goal__status=GameGoal.Status.QUEUED,
    ).select_related("goal")
    for relation in required_relations:
        has_other_blocker = (
            relation.goal.dependencies.filter(is_required=True)
            .exclude(depends_on=goal)
            .exclude(depends_on__status=GameGoal.Status.COMPLETED)
            .exists()
        )
        if not has_other_blocker:
            return True
    return False


def calculate_goal_priority(goal: GameGoal, now=None) -> Decimal:
    now = now or timezone.now()
    score = Decimal(goal.base_priority)
    today = timezone.localdate(now)

    if goal.due_at:
        due_date = timezone.localdate(goal.due_at)
        if due_date < today:
            score += OVERDUE_BONUS
        elif due_date == today:
            score += DUE_TODAY_BONUS
        elif due_date == today + timedelta(days=1):
            score += DUE_TOMORROW_BONUS

    if _would_unlock_queued_dependent(goal):
        score += UNLOCKS_DEPENDENT_BONUS

    if goal.status == GameGoal.Status.QUEUED and goal.queued_at < now - timedelta(days=7):
        score += AGED_QUEUE_BONUS

    return score.quantize(Decimal("0.01"))
