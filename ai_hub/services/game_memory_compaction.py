from decimal import Decimal

from django.utils import timezone

from ai_hub.models import GameMemoryEntry


def compact_goal_memory(*, goal, workspace, keep_n: int = 20) -> dict:
    """
    Deterministic compaction for goal-scoped memory.

    Retain:
        - All entries with importance_score >= 0.75 (high-importance)
        - The `keep_n` most recent entries

    Entries outside these sets are expired in-place by setting expires_at=now()
    and annotating metadata with {"compacted": true}. Raw audit telemetry
    (GameActionRun, ExecutionStepRun) is never touched.

    Returns a summary dict describing what was kept and what was expired.
    """
    all_entries = list(
        GameMemoryEntry.objects.filter(
            workspace=workspace,
            goal=goal,
            scope_type__in=[
                GameMemoryEntry.ScopeType.GOAL,
                GameMemoryEntry.ScopeType.ACTION_RESULT,
            ],
        ).order_by("-created_at")
    )

    high_importance_ids = {
        e.pk for e in all_entries if e.importance_score >= Decimal("0.75")
    }
    recent_ids = {e.pk for e in all_entries[:keep_n]}
    retain_ids = high_importance_ids | recent_ids

    to_expire = [e for e in all_entries if e.pk not in retain_ids]
    now = timezone.now()
    expired_count = 0

    for entry in to_expire:
        meta = dict(entry.metadata or {})
        meta["compacted"] = True
        meta["compacted_at"] = now.isoformat()
        entry.metadata = meta
        entry.expires_at = now
        entry.save(update_fields=["metadata", "expires_at"])
        expired_count += 1

    return {
        "total_before": len(all_entries),
        "retained": len(retain_ids),
        "compacted": expired_count,
        "high_importance_preserved": len(high_importance_ids),
        "recent_preserved": len(recent_ids - high_importance_ids),
    }
