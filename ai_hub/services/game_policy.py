from django.core.exceptions import ValidationError
from django.utils import timezone


class PolicyViolationError(ValidationError):
    pass


class ApprovalRequiredByPolicyError(ValidationError):
    pass


class BudgetExhaustedError(ValidationError):
    def __init__(self, message, *, counter, limit):
        super().__init__(message)
        self.counter = counter
        self.limit = limit


_POSITIVE_INT_BUDGET_KEYS = frozenset({
    "max_iterations_per_session",
    "max_action_runs_per_session",
    "max_sub_agent_runs_per_goal",
    "max_runtime_seconds",
})

_POSITIVE_FLOAT_BUDGET_KEYS = frozenset({
    "max_total_tokens",
    "max_total_cost_usd",
})


def validate_workspace_policy(policy: dict) -> None:
    """Validate the structure of a workspace policy dict. Raises ValidationError on violations.

    Unknown top-level keys are accepted intentionally — future extensions add keys without
    breaking existing workspaces.
    """
    if not isinstance(policy, dict):
        raise ValidationError("Workspace policy must be a JSON object.")

    budget = policy.get("budget", {})
    if not isinstance(budget, dict):
        raise ValidationError("Policy 'budget' must be a JSON object.")

    for key, value in budget.items():
        if key in _POSITIVE_INT_BUDGET_KEYS:
            try:
                int_val = int(value)
            except (TypeError, ValueError):
                raise ValidationError(
                    f"Policy budget.{key} must be a positive integer, got: {value!r}."
                )
            if int_val <= 0:
                raise ValidationError(
                    f"Policy budget.{key} must be a positive integer, got: {int_val}."
                )
        elif key in _POSITIVE_FLOAT_BUDGET_KEYS:
            try:
                float_val = float(value)
            except (TypeError, ValueError):
                raise ValidationError(
                    f"Policy budget.{key} must be a positive number, got: {value!r}."
                )
            if float_val <= 0:
                raise ValidationError(
                    f"Policy budget.{key} must be a positive number, got: {float_val}."
                )

    safety = policy.get("safety", {})
    if not isinstance(safety, dict):
        raise ValidationError("Policy 'safety' must be a JSON object.")

    bool_safety_keys = {
        "allow_external_writes",
        "allow_self_delegation",
        "require_approval_for_medium_risk",
        "require_approval_for_high_risk",
    }
    for key, value in safety.items():
        if key in bool_safety_keys and not isinstance(value, bool):
            raise ValidationError(
                f"Policy safety.{key} must be a boolean (true/false), got: {value!r}."
            )

    allowed_actions = policy.get("allowed_actions")
    if allowed_actions is not None:
        if not isinstance(allowed_actions, list) or any(
            not isinstance(name, str) or not name.strip() for name in allowed_actions
        ):
            raise ValidationError("Policy 'allowed_actions' must be a list of non-empty action names.")


def validate_goal_execution_policy(workspace, goal, session) -> None:
    """Verify workspace constraints before a goal execution session begins.

    Checks policy structure and the workspace-agent allow-list. Raises PolicyViolationError
    if the entry agent is explicitly disabled for this workspace.
    """
    from ai_hub.models import GameWorkspaceAgent

    policy = workspace.default_policy or {}
    if policy:
        validate_workspace_policy(policy)

    if not session.entry_agent_id:
        return

    agent_entries = GameWorkspaceAgent.objects.filter(workspace=workspace)
    if not agent_entries.exists():
        return

    entry = agent_entries.filter(agent_id=session.entry_agent_id).first()
    if entry is None or not entry.is_enabled:
        raise PolicyViolationError(
            f"Agent '{session.entry_agent.name}' is not enabled for workspace '{workspace.name}'."
        )


def validate_action_policy(workspace, goal, action, payload) -> None:
    """Check workspace-level constraints before running an action.

    Raises PolicyViolationError if the action is blocked.
    Raises ApprovalRequiredByPolicyError if the action needs human approval due to policy.
    Both are subclasses of ValidationError so callers that only handle ValidationError still work.
    """
    from ai_hub.models import GameWorkspaceAction

    policy = workspace.default_policy or {}
    validate_workspace_policy(policy)
    if action.risk_level not in {"low", "medium", "high"}:
        raise PolicyViolationError(
            f"Action '{action.name}' has unknown risk level '{action.risk_level}'."
        )

    workspace_entries = GameWorkspaceAction.objects.filter(workspace=workspace)
    ws_action_entry = workspace_entries.filter(action=action).first()
    if workspace_entries.exists() and (ws_action_entry is None or not ws_action_entry.is_enabled):
        raise PolicyViolationError(
            f"Action '{action.name}' is not enabled for workspace '{workspace.name}'."
        )

    allowed_actions = policy.get("allowed_actions")
    if allowed_actions is not None:
        if isinstance(allowed_actions, list) and action.name not in allowed_actions:
            raise PolicyViolationError(
                f"Action '{action.name}' is not in the workspace allowed_actions policy."
            )

    safety = policy.get("safety", {})
    # Closed by default. Enabling writes must always be an explicit workspace decision.
    allow_external_writes = safety.get("allow_external_writes", False)

    if action.risk_level == "high" and not allow_external_writes:
        raise PolicyViolationError(
            f"Action '{action.name}' (high risk) is blocked by workspace safety policy: "
            "allow_external_writes is disabled."
        )

    if ws_action_entry is not None and ws_action_entry.requires_approval_override is not None:
        if ws_action_entry.requires_approval_override and not action.requires_approval:
            raise ApprovalRequiredByPolicyError(
                f"Action '{action.name}' requires approval per workspace allow-list override."
            )
        return

    require_high = safety.get("require_approval_for_high_risk", False)
    require_medium = safety.get("require_approval_for_medium_risk", False)

    if action.risk_level == "high" and require_high and not action.requires_approval:
        raise ApprovalRequiredByPolicyError(
            f"Action '{action.name}' (high risk) requires approval per workspace safety policy."
        )
    elif action.risk_level == "medium" and require_medium and not action.requires_approval:
        raise ApprovalRequiredByPolicyError(
            f"Action '{action.name}' (medium risk) requires approval per workspace safety policy."
        )


def check_budget_before_iteration(session) -> None:
    """Raise BudgetExhaustedError if the session has consumed its iteration or runtime budget."""
    workspace = _get_workspace(session)
    if workspace is None:
        return

    policy = workspace.default_policy or {}
    budget = policy.get("budget", {})

    max_iterations = budget.get("max_iterations_per_session")
    if max_iterations is not None:
        current_count = session.step_runs.count()
        if current_count >= int(max_iterations):
            raise BudgetExhaustedError(
                f"Session has reached its iteration budget "
                f"({current_count} of {max_iterations} iterations used).",
                counter="iterations",
                limit=int(max_iterations),
            )

    max_runtime = budget.get("max_runtime_seconds")
    if max_runtime is not None and session.started_at:
        elapsed = (timezone.now() - session.started_at).total_seconds()
        if elapsed >= float(max_runtime):
            raise BudgetExhaustedError(
                f"Session has exceeded its runtime budget "
                f"({elapsed:.0f}s elapsed, limit {max_runtime}s).",
                counter="wall_clock_seconds",
                limit=float(max_runtime),
            )


def check_budget_before_action(session, action, *, action_run=None) -> None:
    """Raise BudgetExhaustedError if the session has consumed its action-run budget.

    Token and cost metrics are not yet enforced because provider support is uneven.
    Other defined budgets (e.g. max_action_runs_per_session) are always enforced.
    """
    workspace = _get_workspace(session)
    if workspace is None:
        return

    policy = workspace.default_policy or {}
    budget = policy.get("budget", {})

    max_action_runs = budget.get("max_action_runs_per_session")
    if max_action_runs is not None:
        from ai_hub.models import GameActionRun
        runs = GameActionRun.objects.filter(session=session)
        if action_run is not None and action_run.pk:
            runs = runs.exclude(pk=action_run.pk)
        current_count = runs.count()
        if current_count >= int(max_action_runs):
            raise BudgetExhaustedError(
                f"Session has reached its action run budget "
                f"({current_count} of {max_action_runs} action runs used).",
                counter="action_runs",
                limit=int(max_action_runs),
            )


def get_session_workspace(session):
    try:
        if session.goal_id:
            return session.goal.workspace
        delegation = session.delegation_run
        return delegation.parent_goal.workspace
    except Exception:
        return None


_get_workspace = get_session_workspace


def validate_agent_for_workspace(workspace, agent) -> None:
    """Raise PolicyViolationError if the agent is not enabled for this workspace.

    When no GameWorkspaceAgent entries exist the allow-list is open (legacy compat).
    When entries exist only explicitly-enabled agents are permitted.
    """
    from ai_hub.models import GameWorkspaceAgent

    agent_entries = GameWorkspaceAgent.objects.filter(workspace=workspace)
    if not agent_entries.exists():
        return
    entry = agent_entries.filter(agent=agent).first()
    if entry is None or not entry.is_enabled:
        raise PolicyViolationError(
            f"Agent '{agent.name}' is not enabled for workspace '{workspace.name}'."
        )


def check_delegation_depth(session) -> None:
    """Raise PolicyViolationError if the session is itself a delegated session (depth > 1)."""
    from ai_hub.models import GameDelegationRun

    if GameDelegationRun.objects.filter(delegated_session=session).exists():
        raise PolicyViolationError(
            "Delegation depth limit exceeded: a delegated agent cannot further delegate "
            "(max depth: 1)."
        )


def check_sub_agent_budget(goal, workspace) -> None:
    """Raise BudgetExhaustedError if the goal has consumed its sub-agent run budget."""
    from ai_hub.models import GameDelegationRun

    policy = workspace.default_policy or {}
    budget = policy.get("budget", {})

    max_sub_agents = budget.get("max_sub_agent_runs_per_goal")
    if max_sub_agents is not None:
        current_count = GameDelegationRun.objects.filter(parent_goal=goal).count()
        if current_count >= int(max_sub_agents):
            raise BudgetExhaustedError(
                f"Goal has reached its sub-agent run budget "
                f"({current_count} of {max_sub_agents} sub-agent runs used).",
                counter="sub_agent_runs",
                limit=int(max_sub_agents),
            )
