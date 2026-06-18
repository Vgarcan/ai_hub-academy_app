# Integration Guide

## Integration Goal

The host project should use `ai_hub` as an AI platform layer without making `ai_hub` depend on host-domain models.

Good direction:

```text
host app -> adapter/service -> ai_hub execution session -> result -> host persistence
```

Risky direction:

```text
ai_hub imports many host models directly
```

## Host Adapter Pattern

Create a small adapter in the host project.

Example:

```python
from ai_hub.models import ExecutionSession
from ai_hub.services.execution_runner import run_execution_session


def run_document_ai_workflow(*, document, pipeline, user):
    session = ExecutionSession.objects.create(
        runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR,
        runtime_mode=ExecutionSession.RuntimeMode.SYNC,
        status=ExecutionSession.Status.PENDING,
        pipeline=pipeline,
        triggered_by=user,
        source_label=f"Document #{document.pk}",
        initial_context={
            "document_id": document.pk,
            "title": document.title,
            "content": document.body,
        },
    )
    run_execution_session(session.id)
    session.refresh_from_db()
    return session
```

The host project decides how to persist final business results.

## Generic Source Linking

`ExecutionSession` supports generic source fields:

- `source_content_type`
- `source_object_id`
- `source_object`
- `source_label`

Use them when a session should point back to a host object without adding a hard dependency.

## Public UI Integration

Public UI should stay in the host project.

Recommended pattern:

1. User submits product data.
2. Host project creates or updates a domain object.
3. Host project creates an `ExecutionSession`.
4. Worker or staff action runs the session.
5. Host project renders the final result.

## Product Persistence

`ai_hub` stores execution state.

The host project stores business meaning.

Examples:

| Host Domain | Host Persistence |
| --- | --- |
| Support | drafted response, ticket classification |
| Documents | extracted fields, summary, risk rating |
| Finance | invoice review, approval recommendation, audit note |
| Commerce | recommendation, fraud review, customer note |

Do not store final product records inside `ai_hub` unless they are reusable across domains.

## Recovering From Partial AI Output

Sometimes a final model step may produce malformed or truncated JSON.

`ai_hub` records the failure and keeps the final context.

A host adapter may recover when earlier intermediate outputs are valid enough to persist the domain result.

Recommended pattern:

```text
final_output invalid -> inspect intermediate drafts -> hydrate required domain payload -> persist -> mark host result complete
```

Rules:

- only recover when required domain payloads are present,
- keep `final_output_parse_error` in context for auditability,
- do not hide the fact that the model response was malformed,
- keep recovery code in the host adapter,
- add tests for recovered and unrecoverable cases.

## Admin Integration

Recommended host admin actions:

1. Select domain objects.
2. Create execution sessions.
3. Run sessions.
4. Link back to the execution session change page.
5. Store final domain objects after success.

## Tool Integration

Tool definitions live in `ai_hub`.

Tool implementations may live in:

- `ai_hub`, if reusable,
- the host project, if domain-specific.

If a tool reads host data, expose it through a small allowlisted adapter.

## Knowledge Integration

Use `KnowledgeCollection` for reusable knowledge.

Use host-specific context in `initial_context`.

Avoid mixing private user data into global knowledge documents unless that is intentional and safe.

## Migration From A Product-Specific AI App

Recommended sequence:

1. Copy provider and model concepts.
2. Copy agents.
3. Convert prompts into `AgentProfile`.
4. Convert fixed workflows into `PipelineDefinition`.
5. Convert direct runs into `ExecutionSession`.
6. Keep host-specific persistence outside `ai_hub`.
7. Remove old code only after repeated successful runs.
