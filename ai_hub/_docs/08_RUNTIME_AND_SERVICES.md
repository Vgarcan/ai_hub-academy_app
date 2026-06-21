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
- `ai_hub.services.tools_runtime`
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

## Agent Runtime

The agent runtime:

1. prepares the input payload,
2. injects knowledge context,
3. resolves tools where allowed,
4. calls the configured model,
5. returns a structured output payload.

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

1. resolves the entry agent,
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

### Actions, approval, and idempotency

`execute_game_action()` creates a durable `GameActionRun` before validating a known action's input, policy, and budget. Contract, policy, budget, and handler failures are stored as failed attempts. Equivalent successful or waiting-approval calls return the existing run; terminal failed/rejected attempts return a controlled validation error rather than colliding with the unique idempotency key.

Approval-required actions create one `GameActionApprovalRequest`, pause the session, move the goal to `waiting_approval`, and create one pending continuation. The iteration loop stops immediately. Approval and rejection are row-locked, persisted as parent observations, and must be resolved before `resume_goal_execution()` can continue at the next unused step order.

Real row-lock concurrency must be verified on PostgreSQL. SQLite tests cover behavior but not locking semantics.

### Scoped memory

`GameMemoryEntry` separates workspace, goal, session, and action-result scopes. Service validation rejects cross-workspace or cross-goal combinations, and database constraints protect the basic required/null field shapes. Goal-bound runner payloads include bounded `scoped_memory` with selection and truncation metadata; legacy `final_context.memory` remains available for compatibility.

### Policies and budgets

Workspace agent/action mappings become closed allow-lists once at least one mapping of that type exists. With no mappings, legacy-compatible behavior remains open. Disabled or absent entries in a configured list are rejected.

External writes are closed by default and require an explicit workspace safety policy. Policy is checked when a goal session is created, again when it runs or resumes, and before each selected action. Iteration, action-count, and runtime budgets are enforced; token and cost limits remain declared but are not yet enforceable for every provider.

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

SQLite does not provide realistic `select_for_update()` semantics. Functional scheduler tests run on SQLite, while the concurrent-claim test is conditionally executed only on a locking backend such as PostgreSQL.

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

## Host Adapter Responsibility

The host adapter is responsible for:

- building initial context,
- choosing the pipeline or GAME agent,
- calling the runner,
- persisting product-specific results,
- handling product-specific recovery logic.

The runner should stay reusable.
