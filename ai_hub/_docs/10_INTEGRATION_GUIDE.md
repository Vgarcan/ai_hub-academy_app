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

## GAME Integration

The example above is an Orchestrator session. A host can also drive GAME goals.

**Prerequisite — feature flags.** Every GAME service entry point is gated by a flag (see `03_CONFIGURATION.md`). The reusable default is fail-closed, so a host that calls a GAME service while its flag is disabled receives a `ValidationError`. Enable the capabilities you use, for example:

```text
AI_HUB_GAME_GOALS_ENABLED=True
AI_HUB_GAME_SCHEDULER_ENABLED=True
AI_HUB_GAME_ACTION_DISPATCH_ENABLED=True
AI_HUB_GAME_MEMORY_ENABLED=True
```

**Typical GAME adapter flow:**

```python
from ai_hub.services.game_goals import create_goal
from ai_hub.services.game_goal_execution import create_goal_execution_session
from ai_hub.services.execution_runner import run_execution_session


def run_support_triage_goal(*, workspace, agent, ticket):
    goal = create_goal(
        workspace=workspace,
        title=f"Triage ticket #{ticket.pk}",
        description="Classify the ticket and recommend the next response.",
        context={"ticket_text": ticket.body},
    )
    session = create_goal_execution_session(goal=goal, entry_agent=agent)
    run_execution_session(session.id)
    session.refresh_from_db()
    return session
```

The host stores the business outcome; `ai_hub` keeps the goal, session, and audit trail.

**Operational maintenance.** A goal can stay stuck in `running` if a session never reaches a terminal state. The `cleanup_orphaned_goals` management command cancels such goals (those with no active session). Schedule it in long-running environments or run it with `--dry-run` to inspect first.

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

Recommended host flow:

1. Create a reusable `ToolDefinition` with explicit `operation_mode`, `risk_level`, contracts, and config.
2. Put related tools into a `Toolbox`.
3. Assign the toolbox to agents through `AgentToolboxAssignment`.
4. Use `AgentToolGrant` only for explicit exceptions or restrictions.
5. Keep callable implementations in `AI_HUB_ALLOWED_TOOL_CALLABLES`.
6. For external writes, require workspace policy and/or approval before execution.

For GAME, create a `GameActionDefinition` only when the tool should be available
as a governed selected action. Link that action to the reusable
`ToolDefinition`; GAME enforces the action policy plus the Tool's effective
resolved permission/approval requirement through `GameActionRun`. The reusable
execution is recorded in `ToolExecutionRun` when the unified runtime flag is
enabled.

The built-in Orchestrator and GAME runners use the deliberate resolved manifest
by default and let the agent request one tool at a time. Session and step audit
links are supplied automatically. Host code should normally create and run an
`ExecutionSession` instead of calling `execute_agent_deliberate()` or
`execute_tool()` directly. `execute_tool()` is intentionally a low-level
contract/kind executor: a host that calls it directly owns capability and
approval authorization. The bundled runners and GAME dispatcher provide those
governed boundaries.

If a host must roll back an existing flow temporarily, set the session
`runtime_config.agent_tool_runtime` to `legacy_preexecute`. Do not use that mode
for new integrations. Approval-requiring generic tools need a host-owned
checkpoint/resume implementation; for GAME, use a governed selected action
instead. See `15_RUNTIME_STATUS.md`.

## Knowledge Integration

Use `KnowledgeCollection` for reusable knowledge.

Use host-specific context in `initial_context`.

Avoid mixing private user data into global knowledge documents unless that is intentional and safe.

Split reusable knowledge into `KnowledgeDocumentChunk` records. Attaching an
active collection is enough to expose the built-in read-only retrieval tools;
you do not need to assign the starter Knowledge toolbox merely to make those
canonical adapters reachable. Explicit deny grants and GAME workspace policy
remain authoritative.

Retrieval tools list libraries, browse bounded document/chunk indexes, search
chunks, read selected sections and return citation metadata. The runtime binds
the executing agent identity, so callers must not expose `agent_id` as a model
choice. This keeps large documents out of prompts until the agent asks for
relevant evidence.

### Authoring Knowledge From A Host (T2)

An independent Django host that owns its own content store can publish chunk
sets into Knowledge with one public, operation-specific service:

```python
from ai_hub.services.knowledge_authoring import AuthoredChunk, author_explicit_chunk_set
from ai_hub.services.knowledge_mutation import (
    ExpectedKnowledgeState, KnowledgeMutationPrincipal, build_snapshot,
)

expected = ExpectedKnowledgeState.from_snapshot(build_snapshot(document))
result = author_explicit_chunk_set(
    document.pk,
    [AuthoredChunk(content="...", section_title="Intro"), AuthoredChunk(content="...")],
    application_scope=my_scope,
    expected=expected,
    principal=KnowledgeMutationPrincipal.system("my-host-sync"),
    reason_code="content_sync",
)
```

What it does:

- Replaces the document's chunk set with exactly the supplied ordered set,
  declares it `EXPLICIT`, verifies the stored set against the request (indexes
  `1..N`, fingerprint `c1`) and records one content-free, principal-bound
  lifecycle event, all in one transaction. Any failure, including audit-event
  creation, rolls back every change.
- Preserves a chunk's id when its canonical content and section title are
  unchanged (CRLF/NFC differences are not edits). Duplicates match
  first-in-first-out. Changed content gets a new id; an id is never repurposed
  for different text, so an old citation fails visibly instead of drifting.
- Returns an `AuthoringResult` with `outcome` (`changed` or `unchanged`), the
  document id, chunk count, kept/inserted/removed/moved counts and the resulting
  `c1` fingerprint. It carries no event id. A genuine no-op writes nothing and
  records no event. Even a no-op is decided inside the governed boundary, against
  the locked state verified with `expected`, so `unchanged` is never returned on
  the strength of an unlocked read.

Host obligations:

- Call it only from trusted server-side host code. It is not a tool, an Admin
  action or a lifecycle command, and a model must never be able to reach it.
- Pass the `ApplicationScope` the host owns. The document must belong to it and
  the scope must be active, otherwise a uniform "not found or not accessible"
  error is raised. The check is repeated inside the transaction.
- Capture `expected` from the state the host planned against. A stale value
  raises `KnowledgeMutationConflict`, even for a request that would be a no-op.
  The host decides whether to re-plan.
- Create the document shell yourself (`DRAFT`, no chunks, no generator facts).
  First authoring is accepted only for an empty, unprovenanced `UNKNOWN`
  `DRAFT` document; later authoring only for a valid `EXPLICIT` document.
  `DERIVED` documents and populated `UNKNOWN` documents are refused (use the
  operator lifecycle verbs for those).
- Publication stays host-owned. The service never changes `status`. Authoring
  into a `DRAFT` document is invisible to retrieval; the host activates the
  document afterwards, ideally in the same outer transaction.
- The service does not run embeddings. Kept chunks keep their vectors; new and
  changed chunks have none until the host runs `knowledge_embedding_index`.

Known limitations of this first version:

- Chunk `metadata` is not an input. Documents whose chunks already carry
  non-empty metadata are refused rather than silently rewritten.
- Documents with a non-blank `curated_text` or a `source_file` are refused: the
  chunks would otherwise claim to be synchronised with a source representation
  the service does not maintain. GAME's Path 2 reads only `curated_text`, so
  content authored this way is not visible through that path.
- Limits (500 chunks, 20,000 characters per chunk, 2,000,000 characters per
  call, and the model's title length) are bounded v1 operational limits owned
  by this service. They are not scalability evidence.
- Validated on Windows/SQLite only. `select_for_update` serialisation under
  concurrent writers and database-level length enforcement on PostgreSQL are
  pending Linux validation.

## Migration From A Product-Specific AI App

Recommended sequence:

1. Copy provider and model concepts.
2. Copy agents.
3. Convert prompts into `AgentProfile`.
4. Convert fixed workflows into `PipelineDefinition`.
5. Convert direct runs into `ExecutionSession`.
6. Keep host-specific persistence outside `ai_hub`.
7. Remove old code only after repeated successful runs.
