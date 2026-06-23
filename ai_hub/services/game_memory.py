from decimal import Decimal

from django.core.exceptions import ValidationError
from django.db.models import Q
from django.utils import timezone

from ai_hub.models import GameMemoryEntry
from ai_hub.services.game_feature_flags import require_game_feature


def record_memory(
    *,
    scope_type: str,
    workspace,
    content: str,
    goal=None,
    session=None,
    metadata=None,
    importance_score=Decimal("0.50"),
    expires_at=None,
) -> GameMemoryEntry:
    """Create and persist a GameMemoryEntry after scope validation."""
    require_game_feature("AI_HUB_GAME_MEMORY_ENABLED")
    # Coerce float/int/str callers to Decimal. A raw float (e.g. 0.9) reaches the
    # DecimalField as 0.9000000000000000222..., which fails the 2-decimal-place
    # validator; routing through str() yields a clean Decimal('0.9').
    if not isinstance(importance_score, Decimal):
        importance_score = Decimal(str(importance_score))
    entry = GameMemoryEntry(
        workspace=workspace,
        goal=goal,
        session=session,
        scope_type=scope_type,
        content=content,
        metadata=metadata or {},
        importance_score=importance_score,
        expires_at=expires_at,
    )
    entry.full_clean()
    entry.save()
    return entry


def build_goal_memory_context(
    *,
    workspace,
    goal=None,
    session=None,
    max_chars: int = 4000,
) -> dict:
    """
    Assemble a memory context dict for a GAME iteration.

    Precedence:
        1. High-importance goal/action_result memory (importance >= 0.75)
        2. Recent goal/action_result memory
        3. Workspace-level memory
        4. Session memory

    Returns metadata about truncation so the caller can surface it.
    """
    require_game_feature("AI_HUB_GAME_MEMORY_ENABLED")
    try:
        max_chars = int(max_chars)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Memory max_chars must be a positive integer.") from exc
    if max_chars <= 0:
        raise ValidationError("Memory max_chars must be a positive integer.")
    if goal is not None and goal.workspace_id != workspace.pk:
        raise ValidationError("Memory goal must belong to the requested workspace.")
    if session is not None and session.goal_id:
        if session.goal.workspace_id != workspace.pk:
            raise ValidationError("Memory session must belong to the requested workspace.")
        if goal is not None and session.goal_id != goal.pk:
            raise ValidationError("Memory session and goal must refer to the same GAME goal.")

    now = timezone.now()
    active_q = Q(expires_at__isnull=True) | Q(expires_at__gt=now)

    selected = []
    chars_used = 0
    truncated_count = 0

    def _try_add(entry):
        nonlocal chars_used, truncated_count
        if chars_used + len(entry.content) > max_chars:
            truncated_count += 1
            return
        selected.append({
            "id": entry.pk,
            "scope_type": entry.scope_type,
            "content": entry.content,
            "importance": str(entry.importance_score),
            "metadata": entry.metadata or {},
        })
        chars_used += len(entry.content)

    # 1. High-importance goal memory
    if goal:
        qs = (
            GameMemoryEntry.objects.filter(
                active_q,
                workspace=workspace,
                goal=goal,
                scope_type__in=[
                    GameMemoryEntry.ScopeType.GOAL,
                    GameMemoryEntry.ScopeType.ACTION_RESULT,
                ],
                importance_score__gte=Decimal("0.75"),
            )
            .order_by("-importance_score", "created_at")[:20]
        )
        for entry in qs:
            _try_add(entry)

    # 2. Recent goal memory (not already selected)
    if goal:
        seen_ids = {e["id"] for e in selected}
        qs = (
            GameMemoryEntry.objects.filter(
                active_q,
                workspace=workspace,
                goal=goal,
                scope_type__in=[
                    GameMemoryEntry.ScopeType.GOAL,
                    GameMemoryEntry.ScopeType.ACTION_RESULT,
                ],
            )
            .exclude(pk__in=seen_ids)
            .order_by("-created_at")[:20]
        )
        for entry in qs:
            _try_add(entry)

    # 3. Workspace memory
    qs = (
        GameMemoryEntry.objects.filter(
            active_q,
            workspace=workspace,
            scope_type=GameMemoryEntry.ScopeType.WORKSPACE,
        )
        .order_by("-importance_score", "created_at")[:10]
    )
    for entry in qs:
        _try_add(entry)

    # 4. Session memory
    if session:
        qs = (
            GameMemoryEntry.objects.filter(
                active_q,
                workspace=workspace,
                session=session,
                scope_type=GameMemoryEntry.ScopeType.SESSION,
            )
            .order_by("-created_at")[:10]
        )
        for entry in qs:
            _try_add(entry)

    return {
        "entries": selected,
        "chars_used": chars_used,
        "truncated": truncated_count > 0,
        "truncated_count": truncated_count,
        "max_chars": max_chars,
    }
