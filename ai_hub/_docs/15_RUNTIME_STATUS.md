# Runtime Status: Current, Legacy, Target

This document separates shipped behavior from compatibility paths and intended
architecture. It is the canonical status reference for Tools, Knowledge, GAME
memory and execution modes.

The source of truth remains code plus tests. Update this page whenever a runtime
path changes.

## Status Vocabulary

- **CURRENT** — implemented, exercised by the current runtime or directly usable
  through a documented service.
- **LEGACY** — supported for compatibility, but not the intended long-term
  architecture.
- **TARGET** — the approved migration direction; it is not current behavior
  until the runner and tests prove it.
- **NOT IMPLEMENTED** — planned or discussed only.

## Tools

### CURRENT

- `resolve_agent_tools()` resolves toolbox assignments, grants, legacy direct
  tools and workspace policy into a redacted manifest.
- `execute_agent_deliberate()` implements and tests a bounded model-selected tool
  loop with `ToolExecutionRun` audit.
- Orchestrator and GAME call that resolved runtime by default. The setting
  `AI_HUB_DEFAULT_AGENT_TOOL_RUNTIME=resolved` supplies the host default and a
  session may override it through `runtime_config.agent_tool_runtime`.
- Tool runs created by a normal runner call are linked to both
  `ExecutionSession` and `ExecutionStepRun`.
- Complete tool results remain in `ToolExecutionRun`; the model-facing copy is
  capped by `AI_HUB_MAX_TOOL_OBSERVATION_CHARS`.
- Existing model prompts may return a non-tool final response at the runner
  boundary. Structured deliberate final responses are unwrapped back into
  normal `llm.content`, with the raw protocol payload retained for audit.
- GAME exposes only resolved `context_tool` capabilities in the model-call loop.
  Side-effect actions remain in the selected-action dispatcher.
- The Agent Admin manifest and Control Center graph use the same resolver as the
  runner.
- A selected GAME `GameActionDefinition` can wrap a reusable `ToolDefinition`.
  With `AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True`, that action passes through
  resolved permissions and the generic executor while retaining `GameActionRun`
  policy, approval and audit.

### LEGACY

- Direct `AgentProfile.tools` attachments remain a compatibility input to the
  resolver.
- `agent_tool_runtime="legacy_preexecute"` (or the matching host default)
  selects `execute_agent()`, which pre-executes allowed direct tools before the
  model decides what it needs.
- `allow_legacy_game_action_tools=True` has an effect only in that legacy mode
  and remains restricted to trusted host integrations.

### TARGET

- Keep the resolved, governed manifest as the only normal source of runtime
  capability.
- Retire legacy direct attachments and pre-execution after deployed callers are
  migrated.
- Add a persisted generic deliberate checkpoint before enabling
  approval-requiring tools in ordinary Orchestrator calls.
- GAME control actions remain separate from reusable ordinary tools.

### NOT IMPLEMENTED

- Removal of the direct-tools M2M.
- Generic approval/reject/resume for a partially completed deliberate
  Orchestrator call. Ordinary runner manifests therefore exclude tools that
  require approval; GAME selected actions already provide the durable path.
- A signed or plugin-style callable registry.

## Knowledge

### CURRENT

- Knowledge is stored as collections, documents and chunks.
- Retrieval services can list libraries, browse indexes, search chunks, read a
  selected chunk/section and return citation metadata.
- Agent access is restricted to active collections and documents attached to the
  agent.
- The host defaults
  `AI_HUB_LEGACY_EAGER_KNOWLEDGE_CONTEXT_ENABLED=False`. Normal runner calls
  receive a bounded collection/document index and no document body.
- Attaching an active collection automatically resolves six canonical,
  system-owned, read-only retrieval adapters. Deny grants and workspace policy
  remain authoritative.
- The runtime binds the current agent identity before audit/execution and
  ignores model-supplied agent identifiers.
- Prompt indexes and list/browse/search/read results have explicit bounds,
  including a finite lexical search candidate window.
- Build Console text documents and migration `0019` receive an initial chunk
  when none exists.

### LEGACY

- Eager full-document text injection, bounded only by
  `knowledge_max_chars`, is available through an explicit `True` flag.

### TARGET

- A small index in the prompt, followed by model-selected search/read/citation
  calls for relevant chunks.
- Knowledge models, retrieval services and tool adapters remain separate layers.
- Replace the compatibility eager path only after deployed callers have
  migrated.

### NOT IMPLEMENTED

- Core vector or hybrid search.
- Reranking and retrieval query audit.
- Automatic file chunking/embedding pipelines in the reusable Core. The
  one-chunk curated-text fallback is not a semantic ingestion pipeline. Academy
  embedding features are host-specific and do not change Core behavior.

## GAME Memory

### CURRENT

- `GameMemoryEntry` provides durable workspace, goal, session and action-result
  scopes with validation, importance and expiry.
- `build_goal_memory_context()` selects bounded active entries by scope,
  importance and recency.
- The dispatcher maps workspace, goal, session and action-result writes to only
  the links allowed by each scope.
- The runner refreshes `scoped_memory` after a successful `record_memory` action,
  before the next iteration.
- The legacy `memory`, `observations` and `previous_response` payloads are also
  carried between iterations, with configurable entry/character limits.
- Oversized rolling values become bounded previews in prompts and final context;
  raw `ExecutionStepRun` / `GameActionRun` audit payloads remain intact.
- `compact_goal_memory()` is a manual service that keeps high-importance/recent
  entries and expires the rest. It is not automatic summarisation and is not
  invoked by the normal runner.

### LEGACY

- The rolling in-context `memory` and `observations` lists.

### TARGET

- Retire the legacy rolling lists after all callers use an explicit working
  context contract.
- Prefer durable references over previews for every large action-result type.

### NOT IMPLEMENTED

- Episodic summaries or checkpoints.
- Semantic memory retrieval / memory RAG.
- Automatic memory consolidation.
- A `checkpointed` goal/session outcome.

## Execution Modes And Workers

### CURRENT

- Calling `run_execution_session()` executes work inline in the calling process.
- `sync` and `async` are stored runtime modes, but `async` does not enqueue a
  background job by itself.
- Orchestrator Hybrid stops after its first step and stores a waiting session.
- GAME Hybrid is rejected.
- Approval/information continuation uses explicit durable pause/resume services,
  independent from the runtime-mode label.

### TARGET

- A documented worker/queue contract that claims pending sessions, runs them
  once and reports retries/timeouts without duplicate execution.

### NOT IMPLEMENTED

- A bundled queue backend.
- A worker management command for general pending sessions.
- Automatic retry, stalled-session recovery and queue-level timeout policy.

## PostgreSQL And Concurrency

### CURRENT

- Scheduler claims, approvals and delegation budget reservation use database row
  locks.
- Functional tests pass on SQLite.
- Three concurrency tests intentionally skip unless the database provides the
  required PostgreSQL locking semantics.
- The host accepts `DATABASE_URL` or discrete `POSTGRES_*` settings and ships
  Psycopg 3.
- CI runs SQLite against Django 5.2 LTS and the current supported Django line,
  plus a PostgreSQL 16 job. The PostgreSQL job runs the complete suite,
  including all three concurrency tests.

### TARGET

- Keep the PostgreSQL job as a required release/branch-protection check before
  multi-worker deployment.

## Stable Capabilities Outside The Migrations Above

The following are CURRENT and do not depend on the Tools/Knowledge target
migrations:

- durable GAME goals and deterministic priority;
- goal-bound and legacy GAME sessions;
- selected-action dispatch with policy, budgets and idempotency;
- approval, rejection and explicit resume;
- plans and one-level delegation;
- execution, step, action, approval, continuation and tool audit records;
- Admin Control Center, workspaces, Operations Inbox and Build Console.
