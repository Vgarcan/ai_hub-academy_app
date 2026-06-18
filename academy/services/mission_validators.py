"""
Mission validator registry.

Each validator is a function that receives (user, mission) and returns
(bool, feedback_message). Register validators with @register_validator("key").
"""
VALIDATORS = {}


def register_validator(key):
    def decorator(func):
        VALIDATORS[key] = func
        return func
    return decorator


def validate_mission(*, user, mission):
    """Run the validator for the given mission. Returns (passed, feedback)."""
    validator = VALIDATORS.get(mission.validation_key)
    if not validator:
        return False, f"No validator configured for key '{mission.validation_key}'."
    try:
        return validator(user=user, mission=mission)
    except Exception as exc:
        return False, f"Validator error: {exc}"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

@register_validator("visited_control_room")
def validate_visited_control_room(*, user, mission):
    # Orientation mission: always passes — the user marks it manually
    return True, "Welcome to AI Hub! The control room is open. Mission complete."


@register_validator("created_training_provider")
def validate_created_training_provider(*, user, mission):
    from ai_hub.models import ProviderConfig
    exists = ProviderConfig.objects.filter(
        name__icontains="training",
        is_active=True,
    ).exists()
    if exists:
        return True, "Academy Training Provider found. Mission complete!"
    return False, (
        "No active provider named 'Training' found. "
        "Go to Admin → AI Hub → Provider configs and create one with "
        "type 'training' named 'Academy Training Provider'."
    )


@register_validator("created_training_model")
def validate_created_training_model(*, user, mission):
    from ai_hub.models import ModelConfig
    exists = ModelConfig.objects.filter(
        provider__name__icontains="training",
        is_active=True,
    ).exists()
    if exists:
        return True, "Active training model found. Mission complete!"
    return False, (
        "No active model connected to a training provider found. "
        "Go to Admin → AI Hub → Model configs and create one connected "
        "to your Academy Training Provider."
    )


@register_validator("created_first_agent")
def validate_created_first_agent(*, user, mission):
    from ai_hub.models import AgentProfile
    agent = AgentProfile.objects.filter(
        name__icontains="normalizer",
        is_active=True,
    ).first()
    if agent:
        return True, f"Agent '{agent.name}' found and active. Mission complete!"
    return False, (
        "No active agent named 'Input Normalizer' found. "
        "Go to Admin → AI Hub → Agent profiles and create one."
    )


@register_validator("added_agent_contract")
def validate_added_agent_contract(*, user, mission):
    from ai_hub.models import AgentProfile
    agent = AgentProfile.objects.filter(
        name__icontains="normalizer",
        is_active=True,
    ).first()
    if not agent:
        return False, "Complete Mission 2.1 first (create Input Normalizer agent)."
    has_input = bool(agent.input_contract and agent.input_contract.get("required"))
    has_output = bool(agent.output_contract and agent.output_contract.get("required"))
    if has_input and has_output:
        return True, "Input and output contracts found. Mission complete!"
    missing = []
    if not has_input:
        missing.append("input_contract.required")
    if not has_output:
        missing.append("output_contract.required")
    return False, f"Missing: {', '.join(missing)}. Edit the agent and add contract fields."


@register_validator("created_knowledge_collection")
def validate_created_knowledge_collection(*, user, mission):
    from ai_hub.models import KnowledgeCollection
    collection = KnowledgeCollection.objects.filter(is_active=True).first()
    if collection:
        return True, f"Knowledge collection '{collection.name}' found. Mission complete!"
    return False, (
        "No active knowledge collection found. "
        "Go to Admin → AI Hub → Knowledge collections and create one."
    )


@register_validator("created_orchestrator_pipeline")
def validate_created_orchestrator_pipeline(*, user, mission):
    from ai_hub.models import PipelineDefinition
    pipeline = PipelineDefinition.objects.filter(is_active=True).first()
    if not pipeline:
        return False, (
            "No active pipeline found. "
            "Go to Admin → AI Hub → Pipeline definitions and create one with steps."
        )
    step_count = pipeline.steps.count()
    if step_count < 2:
        return False, (
            f"Pipeline '{pipeline.name}' has only {step_count} step(s). "
            "Add at least 2 steps (order 1 and 2)."
        )
    return True, f"Pipeline '{pipeline.name}' with {step_count} steps found. Mission complete!"


@register_validator("ran_successful_execution_session")
def validate_ran_successful_execution_session(*, user, mission):
    from ai_hub.models import ExecutionSession
    qs = ExecutionSession.objects.filter(
        runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR,
        status=ExecutionSession.Status.SUCCESS,
    )
    if user and user.is_authenticated:
        qs = qs.filter(triggered_by=user)
    session = qs.first()
    if session:
        return True, f"Successful orchestrator session #{session.pk} found. Mission complete!"
    return False, (
        "No successful orchestrator execution session found. "
        "Run your pipeline from Admin → AI Hub → Execution sessions."
    )


@register_validator("created_game_session")
def validate_created_game_session(*, user, mission):
    from ai_hub.models import ExecutionSession
    qs = ExecutionSession.objects.filter(
        runtime_kind=ExecutionSession.RuntimeKind.GAME,
        status=ExecutionSession.Status.SUCCESS,
    )
    if user and user.is_authenticated:
        qs = qs.filter(triggered_by=user)
    session = qs.first()
    if session:
        return True, f"Successful GAME session #{session.pk} found. Mission complete!"
    return False, (
        "No successful GAME execution session found. "
        "Create a GAME session from the GAME Workspace in AI Hub."
    )


@register_validator("completed_capstone")
def validate_completed_capstone(*, user, mission):
    from ai_hub.models import ExecutionSession
    from support_demo.models import SupportTicket
    ticket = SupportTicket.objects.filter(
        ai_session__status=ExecutionSession.Status.SUCCESS,
    ).first()
    if ticket:
        return True, (
            f"Ticket '{ticket.title}' has a successful AI session. "
            "Full workflow complete! Mission accomplished!"
        )
    return False, (
        "No support ticket with a successful AI session found. "
        "Go to Support Demo → run triage on a ticket."
    )


@register_validator("inspected_execution_timeline")
def validate_inspected_execution_timeline(*, user, mission):
    # This mission requires manual completion (user reviews the timeline)
    return True, "Execution timeline inspected. Mark as complete when you have reviewed a session."


@register_validator("created_game_goal")
def validate_created_game_goal(*, user, mission):
    from ai_hub.models import ExecutionSession
    qs = ExecutionSession.objects.filter(
        runtime_kind=ExecutionSession.RuntimeKind.GAME,
    ).exclude(goal_text="")
    if user and user.is_authenticated:
        qs = qs.filter(triggered_by=user)
    session = qs.first()
    if session:
        return True, f"GAME session with goal text found. Mission complete!"
    return False, (
        "No GAME session with a goal found. "
        "Create one from the GAME Workspace with a non-empty goal."
    )


@register_validator("fixed_broken_provider")
def validate_fixed_broken_provider(*, user, mission):
    from ai_hub.models import ProviderConfig
    broken = ProviderConfig.objects.filter(is_active=False).count()
    active = ProviderConfig.objects.filter(is_active=True).count()
    if active > 0 and broken == 0:
        return True, "All providers are active. Mission complete!"
    if broken > 0:
        return False, f"There are still {broken} inactive provider(s). Fix them in Admin."
    return False, "Create and fix at least one provider first."


@register_validator("connected_host_object")
def validate_connected_host_object(*, user, mission):
    from support_demo.models import SupportTicket
    ticket = SupportTicket.objects.filter(ai_session__isnull=False).first()
    if ticket:
        return True, f"Ticket '{ticket.title}' is connected to an AI session. Mission complete!"
    return False, (
        "No support ticket connected to an AI session found. "
        "Run triage on a ticket from the Support Demo."
    )
