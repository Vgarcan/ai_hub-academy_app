from decimal import Decimal
from datetime import timedelta

from django.db.models import Exists, OuterRef

from ai_hub.models import GameGoal


_SENSITIVE_KEYS = frozenset({
    "api_key",
    "secret",
    "password",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "credentials",
    "private_key",
})


def redact_payload(payload):
    """Recursively redact known-sensitive keys from a dict or list payload."""
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    result = {}
    for k, v in payload.items():
        k_lower = k.lower()
        if any(s == k_lower or s in k_lower for s in _SENSITIVE_KEYS):
            result[k] = "***REDACTED***"
        else:
            result[k] = redact_payload(v)
    return result


def build_scheduler_explanation(goal, *, now=None):
    """Return a human-readable priority breakdown for one goal."""
    from django.utils import timezone
    from ai_hub.services.game_priority import (
        AGED_QUEUE_BONUS,
        DUE_TODAY_BONUS,
        DUE_TOMORROW_BONUS,
        OVERDUE_BONUS,
        UNLOCKS_DEPENDENT_BONUS,
        _would_unlock_queued_dependent,
    )

    now = now or timezone.now()
    today = timezone.localdate(now)
    base = Decimal(goal.base_priority)
    bonuses = []

    if goal.due_at:
        due_date = timezone.localdate(goal.due_at)
        if due_date < today:
            bonuses.append({"reason": "Overdue", "value": OVERDUE_BONUS})
        elif due_date == today:
            bonuses.append({"reason": "Due today", "value": DUE_TODAY_BONUS})
        elif due_date == today + timedelta(days=1):
            bonuses.append({"reason": "Due tomorrow", "value": DUE_TOMORROW_BONUS})

    if _would_unlock_queued_dependent(goal):
        bonuses.append({"reason": "Unlocks dependent goals", "value": UNLOCKS_DEPENDENT_BONUS})

    if goal.status == GameGoal.Status.QUEUED and goal.queued_at < now - timedelta(days=7):
        bonuses.append({"reason": "Aged queue (> 7 days)", "value": AGED_QUEUE_BONUS})

    total = base + sum(b["value"] for b in bonuses)
    return {"base_priority": base, "bonuses": bonuses, "total": total}


def build_workspace_dashboard_context(workspace):
    """Aggregate operational data scoped to one workspace."""
    from django.db.models import Count
    from django.utils import timezone
    from ai_hub.models import (
        ExecutionSession,
        GameActionApprovalRequest,
        GameActionRun,
        GameGoalDependency,
        GameWorkspaceAction,
        GameWorkspaceAgent,
    )

    now = timezone.now()
    goals_qs = GameGoal.objects.filter(workspace=workspace)

    # Single aggregate query; preserve enum order and drop empty statuses.
    raw_counts = {
        row["status"]: row["count"]
        for row in goals_qs.values("status").annotate(count=Count("id"))
    }
    status_counts = {s: raw_counts[s] for s in GameGoal.Status.values if raw_counts.get(s)}

    unresolved_req = (
        GameGoalDependency.objects.filter(goal_id=OuterRef("pk"), is_required=True)
        .exclude(depends_on__status=GameGoal.Status.COMPLETED)
    )
    eligible_qs = list(
        goals_qs.filter(status=GameGoal.Status.QUEUED)
        .annotate(has_unresolved=Exists(unresolved_req))
        .filter(has_unresolved=False)
    )
    # Score each eligible goal exactly once and reuse the explanation for both
    # ranking and display (the explanation total equals the scheduler priority).
    scored = sorted(
        (
            {"goal": g, "explanation": build_scheduler_explanation(g, now=now)}
            for g in eligible_qs
        ),
        key=lambda item: -item["explanation"]["total"],
    )
    top_eligible = scored[:5]

    blocked_goals = list(
        goals_qs.filter(status=GameGoal.Status.QUEUED)
        .annotate(has_unresolved=Exists(unresolved_req))
        .filter(has_unresolved=True)
    )

    pending_approvals = list(
        GameActionApprovalRequest.objects.filter(
            goal__workspace=workspace,
            status=GameActionApprovalRequest.Status.PENDING,
        )
        .select_related("action_run", "goal")
        .order_by("-created_at")[:10]
    )

    recent_sessions = list(
        ExecutionSession.objects.filter(goal__workspace=workspace)
        .select_related("entry_agent", "goal")
        .order_by("-created_at")[:10]
    )

    recent_action_runs = list(
        GameActionRun.objects.filter(session__goal__workspace=workspace)
        .select_related("action", "session")
        .order_by("-started_at")[:10]
    )

    policy = workspace.default_policy or {}
    budget = policy.get("budget", {})

    enabled_agents = list(
        GameWorkspaceAgent.objects.filter(workspace=workspace, is_enabled=True).select_related("agent")
    )
    enabled_actions = list(
        GameWorkspaceAction.objects.filter(workspace=workspace, is_enabled=True).select_related("action")
    )

    return {
        "workspace": workspace,
        "status_counts": status_counts,
        "top_eligible": top_eligible,
        "blocked_goals": blocked_goals,
        "pending_approvals": pending_approvals,
        "recent_sessions": recent_sessions,
        "recent_action_runs": recent_action_runs,
        "budget": budget,
        "policy": policy,
        "enabled_agents": enabled_agents,
        "enabled_actions": enabled_actions,
    }


def build_goal_detail_context(goal):
    """Aggregate operational data scoped to one goal."""
    from django.core.exceptions import ObjectDoesNotExist
    from ai_hub.models import ExecutionSession, GameActionRun, GameMemoryEntry
    from ai_hub.services.game_resume import _WAITING_GOAL_STATUSES

    session_history = list(
        ExecutionSession.objects.filter(goal=goal).select_related("entry_agent").order_by("-created_at")
    )
    action_runs = list(
        GameActionRun.objects.filter(session__goal=goal)
        .select_related("action", "session")
        .order_by("started_at")
    )
    memory_entries = list(
        GameMemoryEntry.objects.filter(goal=goal).order_by("-importance_score", "created_at")[:20]
    )

    plan = None
    try:
        plan = goal.plan
    except ObjectDoesNotExist:
        pass

    waiting_sessions = [s for s in session_history if s.status == ExecutionSession.Status.WAITING_ASYNC]
    is_resumable = goal.status in _WAITING_GOAL_STATUSES and bool(waiting_sessions)

    scheduler_explanation = None
    if goal.status in {GameGoal.Status.QUEUED, GameGoal.Status.RUNNING}:
        scheduler_explanation = build_scheduler_explanation(goal)

    return {
        "session_history": session_history,
        "action_runs": action_runs,
        "memory_entries": memory_entries,
        "plan": plan,
        "is_resumable": is_resumable,
        "scheduler_explanation": scheduler_explanation,
    }
