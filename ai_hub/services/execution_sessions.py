from django.contrib.contenttypes.models import ContentType

from ai_hub.models import ExecutionSession


def create_execution_session(
    *,
    source_object=None,
    source_label: str = "",
    pipeline=None,
    entry_agent=None,
    triggered_by=None,
    runtime_kind: str = ExecutionSession.RuntimeKind.ORCHESTRATOR,
    runtime_mode: str = ExecutionSession.RuntimeMode.ASYNC,
    goal_text: str = "",
    runtime_config: dict | None = None,
    initial_context: dict | None = None,
) -> ExecutionSession:
    content_type = None
    object_id = None
    if source_object is not None:
        content_type = ContentType.objects.get_for_model(source_object, for_concrete_model=False)
        object_id = source_object.pk
        if not source_label:
            source_label = str(source_object)

    session = ExecutionSession(
        source_content_type=content_type,
        source_object_id=object_id,
        source_label=source_label,
        pipeline=pipeline,
        entry_agent=entry_agent,
        triggered_by=triggered_by,
        runtime_kind=runtime_kind,
        runtime_mode=runtime_mode,
        goal_text=goal_text,
        runtime_config=runtime_config or {},
        initial_context=initial_context or {},
    )
    session.full_clean()
    session.save()
    return session
