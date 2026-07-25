from decimal import Decimal
from datetime import timedelta
import re

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


def redact_text(value):
    """Best-effort masking for common secret forms embedded in text/error strings."""
    text = str(value or "")
    text = re.sub(
        r"(?i)(api[_-]?key|password|secret|access[_-]?token|refresh[_-]?token|authorization)"
        r"(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2***REDACTED***",
        text,
    )
    return re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer ***REDACTED***", text)


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


def build_workspace_dashboard_context(workspace, *, user=None):
    """Aggregate operational data scoped to one workspace."""
    from django.db.models import Count
    from django.utils import timezone
    from ai_hub.models import (
        ExecutionSession,
        ExecutionStepRun,
        GameActionApprovalRequest,
        GameActionRun,
        GameDelegationRun,
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
        key=lambda item: (
            -item["explanation"]["total"],
            item["goal"].due_at is None,
            item["goal"].due_at or now,
            item["goal"].created_at,
            item["goal"].pk,
        ),
    )
    top_eligible = scored[:5]

    dependency_blocked = list(
        goals_qs.filter(status=GameGoal.Status.QUEUED)
        .annotate(has_unresolved=Exists(unresolved_req))
        .filter(has_unresolved=True)
    )
    from ai_hub.services.game_dependencies import get_goal_blockers

    blocked_candidates = {
        goal.pk: goal
        for goal in [*dependency_blocked, *goals_qs.filter(status=GameGoal.Status.BLOCKED)]
    }
    blocked_goals = [
        {"goal": goal, "blockers": get_goal_blockers(goal)}
        for goal in blocked_candidates.values()
    ]

    can_review = user is None or user.has_perm("ai_hub.approve_game_action")
    pending_approvals = []
    if can_review:
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

    recent_action_runs = []
    if user is None or user.has_perm("ai_hub.view_gameactionrun"):
        recent_action_runs = list(
            GameActionRun.objects.filter(session__goal__workspace=workspace)
            .select_related("action", "session")
            .order_by("-started_at")[:10]
        )

    policy = workspace.default_policy or {}
    budget = policy.get("budget", {})

    enabled_agents = []
    if user is None or user.has_perm("ai_hub.view_gameworkspaceagent"):
        enabled_agents = list(
            GameWorkspaceAgent.objects.filter(workspace=workspace, is_enabled=True).select_related("agent")
        )
    enabled_actions = []
    if user is None or user.has_perm("ai_hub.view_gameworkspaceaction"):
        enabled_actions = list(
            GameWorkspaceAction.objects.filter(workspace=workspace, is_enabled=True).select_related("action")
        )

    workspace_sessions = ExecutionSession.objects.filter(goal__workspace=workspace)
    budget_consumption = {
        "sessions": workspace_sessions.count(),
        "iterations": ExecutionStepRun.objects.filter(session__goal__workspace=workspace).count(),
        "action_runs": GameActionRun.objects.filter(session__goal__workspace=workspace).count(),
        "delegations": GameDelegationRun.objects.filter(parent_goal__workspace=workspace).count(),
        "configured_limits": budget,
    }

    from ai_hub.services.game_dependencies import get_goal_blockers

    return {
        "workspace": workspace,
        "status_counts": status_counts,
        "top_eligible": top_eligible,
        "blocked_goals": blocked_goals,
        "pending_approvals": pending_approvals,
        "recent_sessions": recent_sessions,
        "recent_action_runs": recent_action_runs,
        "budget": budget,
        "budget_consumption": budget_consumption,
        "policy": policy,
        "enabled_agents": enabled_agents,
        "enabled_actions": enabled_actions,
    }


def build_goal_detail_context(goal, *, user=None):
    """Aggregate operational data scoped to one goal."""
    from django.core.exceptions import ObjectDoesNotExist
    from ai_hub.services.game_dependencies import get_goal_blockers
    from ai_hub.models import (
        ExecutionSession,
        GameActionApprovalRequest,
        GameActionRun,
        GameContinuationRequest,
        GameMemoryEntry,
    )
    from ai_hub.services.game_resume import _WAITING_GOAL_STATUSES

    session_history = []
    if user is None or user.has_perm("ai_hub.view_executionsession"):
        session_history = list(
            ExecutionSession.objects.filter(goal=goal)
            .select_related("entry_agent")
            .order_by("-created_at")
        )
    action_runs = []
    if user is None or user.has_perm("ai_hub.view_gameactionrun"):
        action_runs = list(
            GameActionRun.objects.filter(session__goal=goal)
            .select_related("action", "session")
            .order_by("session__created_at", "iteration", "pk")
        )
    memory_entries = []
    if user is None or user.has_perm("ai_hub.view_gamememoryentry"):
        memory_entries = list(
            GameMemoryEntry.objects.filter(goal=goal)
            .order_by("-importance_score", "created_at")[:20]
        )

    plan = None
    try:
        if user is None or user.has_perm("ai_hub.view_gamegoalplan"):
            plan = goal.plan
    except ObjectDoesNotExist:
        pass

    waiting_sessions = [s for s in session_history if s.status == ExecutionSession.Status.WAITING_ASYNC]
    is_resumable = False
    for waiting_session in waiting_sessions:
        continuation = (
            GameContinuationRequest.objects.filter(
                session=waiting_session,
                status=GameContinuationRequest.Status.PENDING,
            )
            .order_by("-created_at")
            .first()
        )
        if continuation is None:
            continue
        if continuation.reason_code == GameContinuationRequest.ReasonCode.NEEDS_APPROVAL:
            action_run_id = (continuation.payload or {}).get("action_run_id")
            approval_status = GameActionApprovalRequest.objects.filter(
                action_run_id=action_run_id,
                action_run__session=waiting_session,
            ).values_list("status", flat=True).first()
            if approval_status not in {
                GameActionApprovalRequest.Status.APPROVED,
                GameActionApprovalRequest.Status.REJECTED,
            }:
                continue
        is_resumable = goal.status in _WAITING_GOAL_STATUSES
        if is_resumable:
            break

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
        "blockers": get_goal_blockers(goal),
    }


def build_session_timeline(session, *, user=None):
    """Build one permission-aware chronological timeline across GAME audit models."""
    from ai_hub.models import (
        GameActionApprovalRequest,
        GameActionRun,
        GameContinuationRequest,
    )

    events = []
    if user is None or user.has_perm("ai_hub.view_executionsteprun"):
        for step in session.step_runs.select_related("agent").order_by("created_at", "pk"):
            observation = step.observation_payload or {}
            decision = observation.get("decision", {}) if isinstance(observation, dict) else {}
            events.append(
                {
                    "kind": "step",
                    "pk": step.pk,
                    "timestamp": step.created_at,
                    "status": step.status,
                    "agent": step.agent.name if step.agent_id else "",
                    "action": step.action_name,
                    "summary": redact_text(
                        decision.get("message", "") if isinstance(decision, dict) else ""
                    ),
                    "latency_ms": step.latency_ms,
                    "error": redact_text(step.error_detail),
                }
            )
    if user is None or user.has_perm("ai_hub.view_gameactionrun"):
        for run in GameActionRun.objects.filter(session=session).select_related("action"):
            events.append(
                {
                    "kind": "action",
                    "pk": run.pk,
                    "timestamp": run.started_at or session.created_at,
                    "status": run.status,
                    "agent": session.entry_agent.name if session.entry_agent_id else "",
                    "action": run.action_name,
                    "summary": redact_text(
                        (run.output_payload or {}).get("message", "")
                        if isinstance(run.output_payload, dict)
                        else ""
                    ),
                    "latency_ms": run.latency_ms,
                    "error": redact_text(run.error_detail),
                }
            )
    if user is None or user.has_perm("ai_hub.view_gamecontinuationrequest"):
        for continuation in GameContinuationRequest.objects.filter(session=session):
            events.append(
                {
                    "kind": "continuation",
                    "pk": continuation.pk,
                    "timestamp": continuation.created_at,
                    "status": continuation.status,
                    "agent": "",
                    "action": continuation.reason_code,
                    "summary": redact_text(continuation.detail),
                    "latency_ms": None,
                    "error": "",
                }
            )
    can_view_approval = user is None or user.has_perm("ai_hub.approve_game_action")
    if can_view_approval:
        for approval in GameActionApprovalRequest.objects.filter(
            action_run__session=session
        ).select_related("action_run", "reviewed_by"):
            events.append(
                {
                    "kind": "approval",
                    "pk": approval.pk,
                    "timestamp": approval.created_at,
                    "status": approval.status,
                    "agent": str(approval.reviewed_by or ""),
                    "action": approval.action_run.action_name,
                    "summary": redact_text(approval.review_note),
                    "latency_ms": None,
                    "error": "",
                }
            )
    events.sort(key=lambda event: (event["timestamp"], event["kind"], event["pk"]))
    return events
