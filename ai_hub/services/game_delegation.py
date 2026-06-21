from django.core.exceptions import ValidationError
from django.utils import timezone
from ai_hub.services.game_feature_flags import require_game_feature

from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    GameActionRun,
    GameDelegationRun,
    GameGoal,
)


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
    """Validate, record, and run a sub-agent delegation synchronously.

    Creates a GameDelegationRun before launching the delegated session so
    budget counts are consistent with in-flight runs. Raises ValidationError
    on policy violation, budget exhaustion, or delegated session failure.
    Returns the result dict on success.
    """
    require_game_feature("AI_HUB_GAME_DELEGATION_ENABLED")
    from ai_hub.services.game_policy import (
        check_delegation_depth,
        check_sub_agent_budget,
        validate_agent_for_workspace,
    )
    from ai_hub.services.execution_runner import run_execution_session

    # 1. Resolve target agent.
    try:
        target_agent = AgentProfile.objects.get(name=target_agent_name, is_active=True)
    except AgentProfile.DoesNotExist:
        raise ValidationError(
            f"Delegation target agent '{target_agent_name}' not found or inactive."
        )

    # 2. Depth: parent session must not itself be a delegated session.
    check_delegation_depth(session)

    # 3. Target agent must be in the workspace allow-list (when configured).
    validate_agent_for_workspace(workspace, target_agent)

    # 4. Sub-agent budget.
    check_sub_agent_budget(goal, workspace)

    # 5. Record delegation run before creating the session so budget count is consistent.
    delegation_run = GameDelegationRun.objects.create(
        parent_action_run=action_run,
        parent_goal=goal,
        target_agent=target_agent,
        task=task,
        expected_result=expected_result,
        status=GameDelegationRun.Status.RUNNING,
    )

    # 6. Build narrowed context — task-specific only, no parent goal memory.
    narrowed_context = {
        "goal_text": task,
        "expected_result": expected_result,
        "parent_goal_id": goal.pk,
        "parent_session_id": session.pk,
        "parent_action_run_id": action_run.pk,
        "delegation_depth": 1,
    }

    # 7. Create delegated session.
    # goal=None keeps the delegated session outside the scheduler's active-goal
    # unique constraint; the parent relationship is tracked via GameDelegationRun.
    delegated_session = ExecutionSession.objects.create(
        runtime_kind=ExecutionSession.RuntimeKind.GAME,
        entry_agent=target_agent,
        goal_text=task,
        runtime_config={"max_iterations": 3, "use_action_dispatcher": True},
        initial_context=narrowed_context,
        status=ExecutionSession.Status.PENDING,
        source_label=f"delegation:goal-{goal.pk}:run-{action_run.pk}",
    )

    delegation_run.delegated_session = delegated_session
    delegation_run.save(update_fields=["delegated_session"])

    # 8. Run synchronously.
    run_execution_session(delegated_session.pk, use_action_dispatcher=True)
    delegated_session.refresh_from_db()

    # 9. Handle failure.
    if delegated_session.status != ExecutionSession.Status.SUCCESS:
        delegation_run.status = GameDelegationRun.Status.FAILED
        delegation_run.finished_at = timezone.now()
        delegation_run.save(update_fields=["status", "finished_at"])
        error_detail = delegated_session.error_detail or "unknown error"
        raise ValidationError(
            f"Delegated agent '{target_agent_name}' failed: {error_detail}"
        )

    # 10. Extract result.
    final_context = delegated_session.final_context or {}
    result_summary = str(final_context.get("final_answer", ""))[:2000]

    delegation_run.status = GameDelegationRun.Status.SUCCESS
    delegation_run.result_summary = result_summary
    delegation_run.finished_at = timezone.now()
    delegation_run.save(update_fields=["status", "result_summary", "finished_at"])

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
