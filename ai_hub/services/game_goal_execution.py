import json

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction

from ai_hub.models import AgentProfile, ExecutionSession, GameGoal
from ai_hub.services.game_goals import (
    ACTIVE_GOAL_SESSION_STATUSES,
    transition_goal_status,
)
from ai_hub.services.game_feature_flags import require_game_feature


def _build_goal_text(goal: GameGoal) -> str:
    success_criteria = json.dumps(goal.success_criteria or {}, ensure_ascii=False, sort_keys=True, indent=2)
    return (
        f"Title: {goal.title}\n\n"
        f"Objective:\n{goal.description}\n\n"
        f"Success criteria:\n{success_criteria}"
    )


@transaction.atomic
def create_goal_execution_session(
    *,
    goal: GameGoal,
    entry_agent: AgentProfile,
    triggered_by=None,
    runtime_config: dict | None = None,
) -> ExecutionSession:
    require_game_feature("AI_HUB_GAME_GOALS_ENABLED")
    locked_goal = GameGoal.objects.select_for_update().select_related("workspace").get(pk=goal.pk)
    if not locked_goal.workspace.is_active:
        raise ValidationError("Cannot start a goal in an inactive GAME workspace.")
    if locked_goal.status not in {GameGoal.Status.QUEUED, GameGoal.Status.RUNNING}:
        raise ValidationError(f"Cannot start a session for a goal with status '{locked_goal.status}'.")
    if not entry_agent.is_active:
        raise ValidationError("GAME entry agent must be active before creating a goal session.")
    if ExecutionSession.objects.filter(
        goal=locked_goal,
        status__in=ACTIVE_GOAL_SESSION_STATUSES,
    ).exists():
        raise ValidationError("This GAME goal already has an active execution session.")

    call_config = runtime_config or {}
    if not isinstance(call_config, dict):
        raise ValidationError("runtime_config must be a JSON object.")
    goal_config = (locked_goal.context or {}).get("runtime_config", {})
    if not isinstance(goal_config, dict):
        raise ValidationError("goal.context.runtime_config must be a JSON object when provided.")
    resolved_runtime_config = {
        **(locked_goal.workspace.default_runtime_config or {}),
        **goal_config,
        **call_config,
    }

    session = ExecutionSession(
        goal=locked_goal,
        entry_agent=entry_agent,
        triggered_by=triggered_by,
        runtime_kind=ExecutionSession.RuntimeKind.GAME,
        runtime_mode=ExecutionSession.RuntimeMode.ASYNC,
        status=ExecutionSession.Status.PENDING,
        goal_text=_build_goal_text(locked_goal),
        runtime_config=resolved_runtime_config,
        initial_context=dict(locked_goal.context or {}),
        source_label=f"GAME goal #{locked_goal.pk}: {locked_goal.title}",
    )
    from ai_hub.services.game_policy import validate_goal_execution_policy

    validate_goal_execution_policy(locked_goal.workspace, locked_goal, session)
    session.full_clean()
    try:
        with transaction.atomic():
            session.save()
    except IntegrityError as exc:
        raise ValidationError("This GAME goal already has an active execution session.") from exc

    if locked_goal.status == GameGoal.Status.QUEUED:
        transition_goal_status(locked_goal, GameGoal.Status.RUNNING, reason=f"execution session #{session.pk} created")
    return session
