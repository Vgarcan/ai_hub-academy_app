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

## Provider Routing

### CURRENT

- `resolve_model_config()` carries the selected
  `ProviderConfig.provider_type` into every normal Agent completion.
- `completion_call()` uses that value as its only adapter-routing decision:
  Training selects the deterministic stub, Ollama selects the native chat
  adapter, and supported cloud/compatible types select LiteLLM.
- Model prefixes, endpoint text and port `11434` do not establish provider
  identity.
- The Ollama adapter accepts existing `ollama/<model>` identifiers by removing
  the prefix locally before `/api/chat`; new configuration can use the exact
  unprefixed identifier reported by `/api/tags`.
- Provider failures are explicit and categorized. Ollama never falls through to
  LiteLLM, and a missing LiteLLM dependency is not reported as a stub success.
- Live Ollama Agent and one-step Orchestrator tests are opt-in through
  `AI_HUB_LIVE_OLLAMA_BASE_URL` plus `AI_HUB_LIVE_OLLAMA_MODEL`; normal CI skips
  them and has no developer endpoint default.
- The Phase 2 live validation passed against a real remote Ollama endpoint with
  `llama3.2:3b`. Endpoint-specific details remain in the private foundation
  report rather than production defaults.

### TARGET

- Keep provider choice centralized at the completion boundary as more adapters
  are added.

## Orchestrator Fallback

### CURRENT

- `PipelineStep.input_mapping` creates one logical input from session context.
  Primary and fallback Agents are independently prepared from that input.
- A configured fallback uses its own contracts, Knowledge, resolved tool
  manifest, identity, ModelConfig and ProviderConfig through the normal Agent
  execution boundary.
- Direct, primary, fallback and GAME Agent calls enforce current `is_active`
  before Knowledge preparation, Tool access or Provider invocation.
- Pipeline activation checks fallback contract presence and statically obvious
  required-input mismatches. Runtime validation independently enforces Agent
  input/output contracts.
- Missing `output_mapping` source paths are failures for both normal and
  fallback output.
- `on_error=fallback_agent` makes one fallback attempt. `stop`, `continue` and
  primary success do not prepare or invoke it.
- Recovered steps use `status=success`, record the fallback as effective Agent,
  leave `error_detail` empty and add structured `fallback_recovery` metadata to
  the response. Double failures use `status=failed` and preserve both causes.
- Categorized provider errors enter normal Orchestrator fallback policy.
  Provider adapters never perform implicit provider switching.
- Deterministic regressions are always runnable. The real Ollama fallback slice
  is opt-in through the existing live environment variables and skips in
  normal CI.

### NOT IMPLEMENTED

- Recursive fallback chains, general retries, backoff, worker recovery and
  resumable generic Orchestrator checkpoints.

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
  policy, approval and audit. Session entry-agent and pipeline-backed GAME paths
  share one effective-agent resolver for capabilities, Knowledge, workspace
  policy and audit identity.
- Unified GAME approval is restrictive across action policy and the effective
  resolved Tool policy. Once either requires approval, a wrapper whose local
  `requires_approval` is false cannot bypass it; execution resumes only through
  the durable GAME approval path.
- Durable GAME approval stores a redacted canonical intent and fingerprint for
  payload, Action/Tool configuration and contracts, effective Agent/permission,
  and relevant workspace policy. Review checks under its row locks; final
  dispatch then re-reads authoritative state after that transaction, resolves
  one immutable Agent/Tool capability snapshot, compares it and executes that
  same snapshot without carrying a database transaction across external work.
  Drift and pre-`0020` rows require a fresh request without side effects.
- Approval expiry is a terminal non-execution outcome. One atomic row-locked
  finalizer marks Approval/continuation `expired`, marks the ActionRun `failed`,
  stores an `approval_expired` observation, returns Session/Goal to `running`,
  and invokes the existing resume runner only after commit. Stale deadlines are
  detected from approve, reject and resume; a later retry requires a new
  governed approval lifecycle.
- HTTP Tools reject `operation_mode=read` with write-capable methods. Their
  `http`/`https` scheme and hostname allow-list are checked before the initial
  request and before every bounded redirect; automatic redirects are disabled.
- Cross-origin redirects remove all headers classified as credential-bearing
  by the shared case/separator-normalized redaction classifier. Same-origin
  headers remain available, ordinary request/correlation headers are preserved,
  and sanitized multi-hop state never restores credentials. The same classifier
  redacts nested approval snapshots and protected Admin/audit views, while the
  approval fingerprint continues to cover raw credential intent.
- HTTP responses are streamed and always closed. `config.max_response_bytes`
  defaults to 1 MiB (clamped to 1 KiB..10 MiB), applies to success/error bodies
  and is distinct from the 4,000-character model-facing preview.

### LEGACY

- Direct `AgentProfile.tools` attachments remain a compatibility input to the
  resolver.
- `agent_tool_runtime="legacy_preexecute"` (or the matching host default)
  selects `execute_agent()`, which pre-executes only direct tools still allowed
  by effective grant/workspace resolution and excludes approval-requiring
  capabilities before the model call.
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
- Lexical candidates include document tags in the bounded database query while
  retaining active document/collection and Agent collection filters.
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

- Scheduler claims, approvals, delegation budget reservation and orphan cleanup
  use database row locks. Goal session creation and orphan cleanup serialize on
  the same Goal row and recheck active sessions before cancellation.
- Functional tests pass on SQLite.
- Four concurrency tests intentionally skip unless the database provides the
  required PostgreSQL locking semantics.
- The host accepts `DATABASE_URL` or discrete `POSTGRES_*` settings and ships
  Psycopg 3.
- CI runs SQLite against Django 5.2 LTS and the current supported Django line,
  plus a PostgreSQL 16 job. The PostgreSQL job runs the complete suite,
  including all four concurrency tests.

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
