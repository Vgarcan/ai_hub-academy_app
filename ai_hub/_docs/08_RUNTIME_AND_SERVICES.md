# Runtime And Services

## Service Layer

`ai_hub` keeps runtime behavior in services.

Important modules:

- `ai_hub.services.contracts`
- `ai_hub.services.provider_registry`
- `ai_hub.services.litellm_client`
- `ai_hub.services.agent_runtime`
- `ai_hub.services.execution_runner`
- `ai_hub.services.execution_sessions`
- `ai_hub.services.build_console`
- `ai_hub.services.health`
- `ai_hub.services.tools_runtime`
- `ai_hub.services.tool_resolution`
- `ai_hub.services.knowledge_ingestion`
- `ai_hub.services.knowledge_retrieval`
- `ai_hub.services.knowledge_tooling`
- `ai_hub.services.admin_control_center`
- `ai_hub.services.game_workspaces`
- `ai_hub.services.game_goals`
- `ai_hub.services.game_dependencies`
- `ai_hub.services.game_priority`
- `ai_hub.services.game_scheduler`
- `ai_hub.services.game_goal_execution`
- `ai_hub.services.game_goal_outcomes`
- `ai_hub.services.game_action_dispatcher`
- `ai_hub.services.game_memory`
- `ai_hub.services.game_memory_compaction`
- `ai_hub.services.game_resume`
- `ai_hub.services.game_policy`
- `ai_hub.services.game_plans`
- `ai_hub.services.game_delegation`
- `ai_hub.services.game_operational_ux`
- `ai_hub.services.game_feature_flags`
- `ai_hub.services.starter_toolboxes`
- `ai_hub.services.starter_demo`

## Contract Validation

Contracts validate required keys and basic JSON types.

Supported type names:

- `string`
- `integer`
- `number`
- `boolean`
- `object`
- `array`

Validation is deliberately simple. It is meant to catch obvious configuration mistakes before model behavior becomes hard to debug.

## Provider Resolution And Routing

The normal completion path is:

```text
AgentProfile
  -> ModelConfig
  -> ProviderConfig
  -> resolve_model_config()
  -> completion_call(provider_type=...)
  -> selected adapter
```

`ProviderConfig.provider_type` is the single routing decision. Training and
Ollama use their dedicated adapters; the supported cloud/compatible provider
types continue through LiteLLM. The completion boundary does not infer provider
identity from a model prefix, URL or port.

The Ollama adapter locally removes a historical `ollama/` prefix before sending
the model identifier to `/api/chat`. This is compatibility normalization, not a
routing rule. Ollama requires an explicit `base_url`; an unreachable provider,
missing model, provider HTTP error or invalid response raises a categorized
provider error and never switches to LiteLLM. A missing LiteLLM dependency also
fails explicitly rather than returning a fake successful response.

## Agent Runtime

The Orchestrator and GAME runners select an agent-call runtime from
`ExecutionSession.runtime_config.agent_tool_runtime`, falling back to
`AI_HUB_DEFAULT_AGENT_TOOL_RUNTIME`. The default is `resolved`:

1. prepare the input payload and knowledge context,
2. resolve toolbox assignments, grants, compatible direct attachments and
   workspace policy,
3. present a redacted manifest to the model,
4. execute at most one model-selected tool per bounded round,
5. audit each call through `ToolExecutionRun`,
6. return the final model output in the existing pipeline/GAME shape.

The output normally contains:

```json
{
  "agent": "agent_name",
  "tools": {},
  "llm": {
    "content": "model output"
  }
}
```

`execute_agent_deliberate()` is the controlled loop behind that default. Direct
callers remain strict: the model must return either a `final` response or one
`tool_call`. At the runner boundary only, a compatibility adapter accepts an
existing non-tool final response. Structured `final.answer` values are
unwrapped into `llm.content`, so existing output mappings and the GAME decision
decoder keep their contracts; the raw tool-protocol response remains in
`tool_protocol_llm`.

`agent_tool_runtime="legacy_preexecute"` selects the old `execute_agent()` shim.
It sees only active direct `AgentProfile.tools`, intersects them with current
grant/workspace resolution, excludes every Tool whose effective policy requires
approval, pre-executes the remaining set and injects all results into one model
call. This exists for rollback while legacy prompts and demos migrate; new
integrations should not select it.

## Tool Resolution And Execution

`resolve_agent_tools()` builds the final capability set for an agent from:

- active toolbox assignments,
- explicit allow and deny grants,
- legacy direct agent tools,
- workspace policy `allowed_tools` / `blocked_tools`,
- side-effect policy for external writes.

The model-facing manifest includes labels, descriptions, risk, operation mode and
schemas, but never exposes callable paths or private config. Governed callers
enforce the resolved permission and effective approval decision before invoking
`execute_tool()`. That low-level executor enforces contracts, callable allow-lists
and kind-specific runtime safety; it is not a replacement for capability
resolution.

`ToolExecutionRun` records deliberate reusable tool calls and links them to the
owning execution session and step. GAME selected actions continue to record
`GameActionRun`; when a GAME action is linked to a `ToolDefinition`, the adapter
can additionally route through the unified tool runtime behind
`AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED`.

The resolver is the single source of normal runtime capability and is also used
by the Agent Admin manifest, Control Center graph and unified GAME wrappers.
Legacy direct attachments are one compatibility input to that resolver.
Approval-requiring tools are not advertised in ordinary runner calls because
there is no resumable generic LLM checkpoint yet. In GAME, expose such work as
a selected action so the dispatcher can persist approval and resume state.
Each successful tool keeps its complete result in `ToolExecutionRun`, while the
copy returned to the model is capped by
`AI_HUB_MAX_TOOL_OBSERVATION_CHARS`.

HTTP Tool configuration is validated in the model/Admin, during capability
resolution and again at execution. `operation_mode=read` accepts only `GET` or
`HEAD`; write-capable methods require a write/execute operation mode and a
matching grant. URLs require an explicit hostname allow-list and an `http` or
`https` scheme. Redirects are followed explicitly rather than by the HTTP
client: every hop is checked against the same allow-list before contact and the
redirect count is bounded by `config.max_redirects` (default 5, maximum 10).
Sensitive authentication/cookie headers are removed when a permitted redirect
changes origin.

## Knowledge Retrieval

`knowledge_retrieval` exposes read-only services:

- `list_knowledge_libraries()`
- `browse_knowledge_index()`
- `search_knowledge()`
- `read_knowledge_chunk()`
- `read_document_section()`
- `cite_knowledge_source()`

The default agent context contains a bounded collection/document index rather
than document text. If an agent has an active collection, the resolver
automatically exposes the six canonical, system-owned, read-only adapters.
These adapters are still filtered by explicit deny grants and workspace policy.

The runtime removes any model-supplied `agent_id`/`agent_name` and binds the
executing agent server-side before validation, audit and execution. This keeps
collection attachment as the authorization boundary and prevents agent
impersonation. List, browse, search and read results have hard output bounds;
read results report when content was truncated. Lexical search evaluates at
most 1,000 matching chunks per call, materializes at most 20,000 characters per
candidate for in-process scoring, and reports both truncation cases.

`AI_HUB_LEGACY_EAGER_KNOWLEDGE_CONTEXT_ENABLED=true` temporarily restores
bounded eager text injection. It is compatibility behavior, not the normal
path. Knowledge models, retrieval services and tool adapters remain separate;
Knowledge records are not converted into `ToolDefinition` rows.

## Orchestrator Runtime

The Orchestrator runtime:

1. loads the pipeline,
2. validates pipeline input,
3. runs ordered steps,
4. prepares each agent payload,
5. executes the agent,
6. applies output mapping,
7. stores step telemetry,
8. merges `final_output` when present,
9. validates final output,
10. marks the session as success or failed.

If a step fails, `on_error` determines whether the session stops, continues, or tries a fallback agent.

## GAME Runtime

The GAME runtime:

1. resolves one effective agent (the session entry agent, otherwise the first
   pipeline-step agent),
2. loads goal and runtime config,
3. checks the workspace agent policy,
4. executes only tools classified as safe context tools,
5. builds legacy and scoped memory plus observations,
6. sends the GAME payload to the agent,
7. parses the JSON decision,
8. dispatches only the selected action when the dispatcher is enabled,
9. stores action and step telemetry,
10. pauses immediately when approval or information is required,
11. continues or finishes within deterministic budgets.

When `strict_response_contract` is enabled, invalid GAME decision JSON fails the session.

GAME has explicit durable pause/resume services, but GAME Hybrid mode remains rejected until its separate automatic-continuation contract is implemented. Sync and async GAME modes remain supported. Orchestrator Hybrid behavior is unchanged.

### GAME outcome semantics

GAME stores execution and goal outcomes separately in `final_context`:

```json
{
  "execution_outcome": "completed",
  "goal_outcome": "incomplete",
  "finish_reason": "max_iterations"
}
```

An agent that explicitly completes its goal produces `completed` / `achieved`. Reaching the iteration cap produces `completed` / `incomplete`; a runtime exception produces `failed` / `unknown`.

### GAME tool safety

Tool definitions use `config.game_tool_category`:

- `context_tool`: read-only and idempotent; GAME may execute it before the model call.
- `action_tool`: never auto-executed by GAME under the default policy.

Missing or invalid categories are treated as action tools. Python context tools also require `read_only=true`; HTTP context tools require GET or HEAD. Python callables must be present in `AI_HUB_ALLOWED_TOOL_CALLABLES`, and HTTP hosts must be explicitly listed in the tool configuration. Orchestrator keeps its legacy selection behavior but still receives these callable and host protections. The legacy action-tool opt-in remains restricted to explicitly trusted integrations; goal-bound GAME actions use the auditable dispatcher.

GAME actions can now optionally link to a `ToolDefinition`. With the unified tool
runtime flag enabled, `action_type=tool` dispatches through the same resolver and
tool executor used by deliberate agents, while still preserving GAME policy,
approval and `GameActionRun` audit. The Tool must be resolved for the effective
GAME agent; model input cannot select or replace that identity. The effective
approval decision is restrictive across the action definition, GAME/workspace
policy and the resolved Tool policy. If any layer requires approval, execution
pauses before `ToolExecutionRun` is created. After review, the reusable audit
record is created with `approval_state=approved`.

### Actions, approval, and idempotency

`execute_game_action()` creates a durable `GameActionRun` before validating a known action's input, policy, and budget. Contract, policy, budget, and handler failures are stored as failed attempts. Equivalent successful or waiting-approval calls return the existing run; terminal failed/rejected attempts return a controlled validation error rather than colliding with the unique idempotency key.

Approval-required actions—including wrappers whose resolved Tool requires
approval—create one `GameActionApprovalRequest`, pause the session, move the
goal to `waiting_approval`, and create one pending continuation. The iteration
loop stops immediately. Approval and rejection are row-locked, persisted as
parent observations, and must be resolved before `resume_goal_execution()` can
continue at the next unused step order.

Real row-lock concurrency is verified by the PostgreSQL CI job. SQLite tests
cover functional behavior but intentionally skip locking semantics.

### Scoped memory

`GameMemoryEntry` separates workspace, goal, session, and action-result scopes.
Service validation rejects cross-workspace or cross-goal combinations, and
database constraints protect the basic required/null field shapes. The
dispatcher maps each scope to only its valid goal/session links.

Goal-bound runner payloads include bounded `scoped_memory` with selection and
truncation metadata. A successful `record_memory` action refreshes it before the
next iteration. Legacy rolling memory/observations remain available with
configurable entry and character caps; raw step/action audit payloads are not
truncated.

### Policies and budgets

Workspace agent/action mappings become closed allow-lists once at least one mapping of that type exists. With no mappings, legacy-compatible behavior remains open. Disabled or absent entries in a configured list are rejected.

External writes are closed by default and require an explicit workspace safety policy. Policy is checked when a goal session is created, again when it runs or resumes, and before each selected action. Iteration, action-count, and runtime budgets are enforced; token and cost limits remain declared but are not yet enforceable for every provider.

Delegated goal-less sessions recover their authoritative workspace from `GameDelegationRun`; they do not bypass action policy or action budgets. Child runtimes use a strict contract and only policy-enabled read-only context actions. Self-delegation is denied unless `safety.allow_self_delegation=true`. Delegation budget reservation locks the parent goal before creating the run.

## Deterministic GAME scheduler

The initial scheduler uses stored fields and fixed rules; it never asks an LLM which goal to run.

Priority starts at `base_priority` and adds:

- 40 for an overdue goal,
- 30 when due today,
- 15 when due tomorrow,
- 10 when completing the goal would unlock a queued dependent,
- 5 after more than seven days in the queue.

Only queued goals in an active workspace with no unresolved required dependency are eligible. Stable tie-breaking uses score, due date, creation time, and primary key.

`get_next_eligible_goal()` is read-only. `claim_next_goal()` locks the workspace and candidate rows, persists candidate scores, and moves exactly one selected goal to `running` through the goal-transition service. It does not create an execution session.

SQLite does not provide realistic `select_for_update()` semantics. Functional
scheduler tests run on SQLite, while the concurrent-claim test executes in the
PostgreSQL CI job.

## Goal-bound GAME sessions

`create_goal_execution_session()` links durable work to a runtime record while preserving legacy sessions that use only `goal_text`.

Creation locks the goal, rejects inactive workspaces and terminal or waiting goals, prevents more than one active session per goal, creates the pending session, and moves a queued goal to `running` in one transaction. A goal already claimed by the scheduler may create its first session directly.

Runtime configuration precedence is:

1. `workspace.default_runtime_config`,
2. `goal.context.runtime_config`, when present,
3. call-time `runtime_config`.

Later values override earlier values. The complete goal context is copied into `session.initial_context`. `goal_text` is generated from title, description, and serialised success criteria for compatibility with the existing GAME runner.

After a goal-bound session stops, `game_goal_outcomes` maps its final context centrally:

- achieved → completed,
- max-iteration incomplete → partial,
- needs information → waiting for information,
- needs approval → waiting for approval,
- execution failure → failed.

Legacy sessions with no `goal` never create or update a goal. Historical terminal sessions remain linked, allowing a reopened or retried goal to accumulate multiple run records.

Each applied outcome stores a fingerprint on the session. Replaying the same historical outcome is a no-op, while a changed outcome after a future resume can be applied again. `reconcile_goal_outcomes()` repairs the detectable crash window where a terminal session was saved before its goal update completed.

## Execution Sessions

`ExecutionSession` stores:

- runtime kind,
- runtime mode,
- status,
- source object reference,
- pipeline or entry agent,
- goal text,
- runtime config,
- initial context,
- final context,
- error detail.

`ExecutionStepRun` stores:

- request payload,
- response payload,
- observation payload,
- latency,
- status,
- error detail.

This is the audit trail for every run.

## Final Output Handling

When context contains `final_output`, the runner attempts to parse it as JSON and merge its top-level keys into final context.

If `final_output` is malformed or truncated:

- `final_output_parse_error` is stored in context,
- the generic runner may mark the session failed if required output keys are missing,
- host adapters may recover domain-specific output from earlier drafts if they have enough data.

## Starter Seeds

`seed_starter_toolboxes()` creates a safe starter catalog: core foundation,
knowledge retrieval, file-intake planning, workspace draft artifacts, and
developer-assistance toolboxes. Only knowledge retrieval tools are executable
read-only callables by default; the rest are prompt macros or approval-oriented
draft helpers.

`seed_starter_demo()` builds on those seeds with a small knowledge library, a
safe GAME workspace, a `submit_for_approval` action linked to its tool
definition, and starter workspace-agent mappings. The matching management
commands are:

```bash
python manage.py seed_ai_hub_starter_toolboxes
python manage.py seed_ai_hub_starter_demo
```

This keeps `ai_hub` generic while allowing host-specific resilience.

## Staff Endpoints

`ai_hub.urls` exposes staff-only endpoints:

```text
POST /ai-hub/internal/execution-session/run/
```

Expected POST body:

```text
session_id=<id>
```

The host project may expose its own endpoints or call services directly.

## Background Workers

The reusable service layer is compatible with background workers.

Recommended worker behavior:

1. Claim one pending session.
2. Execute it with `run_execution_session`.
3. Store status and errors in the session.
4. Never hide model failures.
5. Never rerun a session that already has step runs.

The repository does not currently ship a general session worker, queue backend,
retry policy or stalled-session recovery loop. `runtime_mode=async` is metadata
until a host dispatches `run_execution_session()` from its own worker.

## Host Adapter Responsibility

The host adapter is responsible for:

- building initial context,
- choosing the pipeline or GAME agent,
- calling the runner,
- persisting product-specific results,
- handling product-specific recovery logic.

The runner should stay reusable.

## Admin Control Center Context

`ai_hub.services.admin_control_center.build_control_center_context()` builds the
global Control Center page context. It is a query/aggregation service, not a
runtime executor.

It returns:

- graph nodes and edges for providers, models, knowledge, tools, agents,
  pipelines and steps,
- pipeline scopes used by the graph filter,
- model catalog rows,
- pipeline summaries and step summaries,
- recent sessions,
- status summaries and checklist items,
- `attention_items` for the **Needs attention** inbox.

`attention_items` normalize different operational signals into one UI model:

- provider health warnings and errors,
- configured Ollama models missing from provider health,
- active knowledge collections with no active documents,
- active agents missing input or output contracts,
- active pipelines with no steps,
- failed pipeline step runs grouped by pipeline step, linked to the latest failed
  `ExecutionStepRun`.

Each attention item has a stable local id, severity, source type, relevance,
optional incident timestamp, hover detail and optional Admin URL. The service does
not archive or silence incidents. Archive and silence are UI preferences stored in
the browser by the Control Center JavaScript.

## AI Hub Home Context

`ai_hub.services.admin_control_center.build_ai_hub_home_context()` builds the
cockpit home page (`/admin/ai_hub/`). Like the Control Center context, it is a
read-only query/aggregation service. It returns, under `ai_hub_home`:

- `vitals` — separate `running` and `waiting` counts plus `live` (= running +
  waiting), `needs_attention`, `open_goals`, `sessions` and `failed`,
- `metrics` and per-area pulse counts (orchestrator, game),
- an `action_queue` (the home "needs your attention" preview),
- `recent_sessions`, a `health_summary`, a setup `checklist` with a
  `recommended_action`, and example blueprints,
- `hidden_models` — the catalog of models demoted from the index via
  `AIHubHideFromIndexMixin`, each with a category and the reason it is hidden,
  used to render the "Show supporting tables" toggle.

## Operations Inbox Context

`ai_hub.services.admin_control_center.build_operations_inbox_context()` powers the
Operations Inbox (`/admin/ai_hub/operations/`). It aggregates, cross-workspace,
everything that needs a human into one `operations_inbox` payload with four
categories and a `counts` summary:

- **approvals** — pending `GameActionApprovalRequest` records (unbounded), each
  with inline approve/reject URLs,
- **waiting** — pending `GameContinuationRequest` records (unbounded),
- **failures** — failed `ExecutionSession` records (most recent 25),
- **blocked** — `GameGoal` records in `blocked` status (most recent 25).

Approvals and waiting items are unbounded because they are the actionable gates;
failures and blocked goals are capped.

## Composed change-page overviews

Several admin classes add a `change_view` override that injects a read-only
"at a glance" overview into the change page context via a `_build_*_overview()`
helper: `_build_provider_overview`, `_build_collection_overview`,
`_build_toolbox_overview` (Foundation hubs) and `_build_agent_overview`,
`_build_pipeline_overview`, `_build_workspace_overview` plus the Session Explorer
overview (root entities). These live in `ai_hub/admin.py` rather than
`services/`, but they follow the same read-only aggregation pattern: they
summarize what the entity contains, what uses it, and a small health checklist for
the Overview tab, and never mutate data.
