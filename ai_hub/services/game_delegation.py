from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    GameActionDefinition,
    GameActionRun,
    GameDelegationRun,
    GameGoal,
    GameWorkspaceAction,
)
from ai_hub.services.game_feature_flags import require_game_feature


_DELEGATED_READ_ACTIONS = frozenset({"search_knowledge", "read_document"})


def _delegated_available_actions(workspace) -> list[str]:
    """Return the smallest policy-approved read-only action set for a child session."""
    actions = GameActionDefinition.objects.filter(
        is_active=True,
        action_type=GameActionDefinition.ActionType.CONTEXT_TOOL,
        name__in=_DELEGATED_READ_ACTIONS,
    )
    workspace_entries = GameWorkspaceAction.objects.filter(workspace=workspace)
    if workspace_entries.exists():
        actions = actions.filter(
            workspace_entries__workspace=workspace,
            workspace_entries__is_enabled=True,
        )
    return list(actions.order_by("name").values_list("name", flat=True))


def run_delegated_agent(
    *,
    session: ExecutionSession,
    action_run: GameActionRun,
    workspace,
    goal: GameGoal,
    target_agent_name: str,
    task: str,
    expected_result: str = "",
) -> dict:
    """Reserve, run, and audit one depth-one least-privilege delegation."""
    require_game_feature("AI_HUB_GAME_DELEGATION_ENABLED")
    from ai_hub.services.execution_runner import run_execution_session
    from ai_hub.services.game_policy import (
        check_delegation_depth,
        check_sub_agent_budget,
        validate_agent_for_workspace,
    )

    if session.goal_id != goal.pk or goal.workspace_id != workspace.pk:
        raise ValidationError("Delegation session, goal, and workspace must refer to the same scope.")
    if action_run.session_id != session.pk:
        raise ValidationError("Delegation action run must belong to the parent session.")
    if action_run.action.action_type != GameActionDefinition.ActionType.SUB_AGENT:
        raise ValidationError("Delegation requires a sub-agent action definition.")

    try:
        target_agent = AgentProfile.objects.get(name=target_agent_name, is_active=True)
    except AgentProfile.DoesNotExist as exc:
        raise ValidationError(
            f"Delegation target agent '{target_agent_name}' not found or inactive."
        ) from exc

    check_delegation_depth(session)
    validate_agent_for_workspace(workspace, target_agent)

    allow_self = (workspace.default_policy or {}).get("safety", {}).get(
        "allow_self_delegation", False
    )
    if target_agent.pk == session.entry_agent_id and not allow_self:
        raise ValidationError("Self-delegation is disabled by workspace policy.")

    # Reserve the delegation budget under the goal lock, but never hold the lock
    # while a model/provider call is running.
    with transaction.atomic():
        locked_goal = (
            GameGoal.objects.select_for_update().select_related("workspace").get(pk=goal.pk)
        )
        check_sub_agent_budget(locked_goal, locked_goal.workspace)
        delegation_run = GameDelegationRun(
            parent_action_run=action_run,
            parent_goal=locked_goal,
            target_agent=target_agent,
            task=task,
            expected_result=expected_result,
            status=GameDelegationRun.Status.RUNNING,
        )
        delegation_run.full_clean()
        delegation_run.save()

    delegated_session = None
    try:
        narrowed_context = {
            "goal_text": task,
            "expected_result": expected_result,
            "parent_goal_id": goal.pk,
            "parent_session_id": session.pk,
            "parent_action_run_id": action_run.pk,
            "delegation_depth": 1,
        }
        delegated_session = ExecutionSession(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=target_agent,
            goal_text=task,
            runtime_config={
                "max_iterations": 3,
                "use_action_dispatcher": True,
                "strict_response_contract": True,
                "available_actions": _delegated_available_actions(workspace),
            },
            initial_context=narrowed_context,
            status=ExecutionSession.Status.PENDING,
            source_label=f"delegation:goal-{goal.pk}:run-{action_run.pk}",
        )
        delegated_session.full_clean()
        delegated_session.save()

        delegation_run.delegated_session = delegated_session
        delegation_run.full_clean()
        delegation_run.save(update_fields=["delegated_session"])

        run_execution_session(delegated_session.pk, use_action_dispatcher=True)
        delegated_session.refresh_from_db()
        if delegated_session.status != ExecutionSession.Status.SUCCESS:
            raise ValidationError(
                f"Delegated agent '{target_agent_name}' failed: "
                f"{delegated_session.error_detail or 'unknown error'}"
            )

        final_context = delegated_session.final_context or {}
        result_summary = str(final_context.get("final_answer", ""))[:2000]
        delegation_run.status = GameDelegationRun.Status.SUCCESS
        delegation_run.result_summary = result_summary
        delegation_run.finished_at = timezone.now()
        delegation_run.full_clean()
        delegation_run.save(
            update_fields=["status", "result_summary", "finished_at"]
        )
    except Exception as exc:
        delegation_run.status = GameDelegationRun.Status.FAILED
        delegation_run.result_summary = str(exc)[:2000]
        delegation_run.finished_at = timezone.now()
        delegation_run.save(
            update_fields=["status", "result_summary", "finished_at"]
        )
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(
            f"Delegated agent '{target_agent_name}' failed: {exc}"
        ) from exc

    return {
        "action_name": "delegate_to_agent",
        "target_agent": target_agent_name,
        "task": task,
        "delegation_run_id": delegation_run.pk,
        "delegated_session_id": delegated_session.pk,
        "status": GameDelegationRun.Status.SUCCESS,
        "result_summary": result_summary,
        "complete": final_context.get("goal_outcome") == "achieved",
        "final_answer": result_summary,
    }
