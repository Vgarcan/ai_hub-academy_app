from ai_hub.models import AgentProfile, ExecutionSession


def resolve_game_entry_agent(session: ExecutionSession) -> AgentProfile | None:
    """Return the single authoritative Agent identity for a GAME session."""
    if session.entry_agent_id:
        return session.entry_agent
    if session.pipeline_id:
        first_step = (
            session.pipeline.steps.select_related("agent")
            .order_by("order")
            .first()
        )
        if first_step is not None:
            return first_step.agent
    return None
