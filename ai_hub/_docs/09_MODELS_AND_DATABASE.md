# Models And Database

AI Hub keeps reusable AI platform data in `ai_hub_*` tables. Host projects should
keep domain-specific data in their own app tables and link to AI Hub execution
sessions when needed.

The database boundary is one of the most important portability rules:

```text
ai_hub owns reusable AI configuration and execution telemetry.
The host project owns business records and final domain persistence.
```

## Model Groups

AI Hub models fall into six groups.

| Group | Purpose |
| --- | --- |
| Providers and models | Connect to AI services. |
| Agents | Define reusable AI workers. |
| Knowledge and tools | Give agents context and capabilities. |
| Orchestrator | Run fixed multi-agent workflows. |
| GAME domain | Store workspaces, goals and dependency rules. |
| Execution and GAME | Record sessions, steps and autonomous loops. |

## `ProviderConfig`

Defines an AI provider, endpoint or local inference server.

Important fields:

- `name`: friendly admin name.
- `provider_type`: runtime adapter type.
- `base_url`: endpoint for local or compatible providers.
- `api_key_env_var`: environment variable name for the secret.
- `default_timeout`: request timeout in seconds.
- `is_active`: whether the provider can be used.

Use this model for operational configuration only. Do not store raw API keys in
the database.

## `ModelConfig`

Defines a concrete model available through a provider.

Important fields:

- `provider`: parent provider.
- `model_name`: exact provider model identifier. For a `training` (stub)
  provider it must be `training` or start with `training/`; model/Admin
  validation and the Build Console enforce this convention.
- `temperature_default`: default generation temperature (defaults to `0.70`).
- `max_tokens_default`: default output budget.
- `supports_tools`: whether this model/runtime can work with tools.
- `is_active`: whether the model can be used. Validation rejects an active
  model under an inactive provider.

Operational note:

```text
Structured JSON workflows usually need low temperature and generous max tokens.
```

## `ToolDefinition`

Defines a reusable tool capability.

Important fields:

- `name`: friendly tool name.
- `label`: admin-facing and model-facing display label.
- `description`: what the tool does.
- `tool_kind`: runtime tool category.
- `risk_level`: low, medium or high operational risk.
- `operation_mode`: read, draft write, state write, external write or execute.
- `requires_approval`: whether execution should pause for review.
- `is_system_tool`: marks built-in platform tools.
- `input_schema`: expected input shape.
- `output_schema`: expected output shape.
- `config`: tool-specific settings.
- `is_active`: whether the tool can be used.

Tools should be treated as controlled capabilities. A tool attached to an agent
does not mean every model can automatically use it. The runtime still decides
whether that tool kind is available and safe.

For HTTP Tools, `operation_mode` is validated against the real request method:
`read` permits only GET or HEAD. The configuration also requires an
`http`/`https` URL and an explicit hostname allow-list. These are model/runtime
invariants rather than database constraints, so no schema migration is needed.

## `Toolbox`, `ToolboxTool`, `AgentToolboxAssignment`, and `AgentToolGrant`

Toolboxes group related tools for a role or workflow.

Important fields:

- `Toolbox.slug`: stable toolbox identifier.
- `ToolboxTool.display_order`: ordering within the toolbox.
- `ToolboxTool.is_enabled`: whether that toolbox currently exposes the tool.
- `AgentToolboxAssignment.is_enabled`: whether the agent currently receives the toolbox.
- `AgentToolGrant.permission_level`: explicit operation allowance for one agent/tool pair.
- `AgentToolGrant.is_enabled`: enabled means allow/override; disabled means deny.

The resolver combines active toolbox assignments, grants, legacy direct tools and
workspace policy into a final per-agent manifest. Disabled grants win over
allows, and restrictive permission levels can remove broader toolbox access for
that tool.

## `KnowledgeCollection`

Groups related knowledge documents.

Important fields:

- `name`: collection name.
- `description`: what the collection teaches agents.
- `is_active`: whether the collection can be used.

Use collections to attach a coherent body of context to one or more agents.

## `KnowledgeDocument`

Stores curated text and optional uploaded source material.

Important fields:

- `collection`: parent collection.
- `title`: document name.
- `curated_text`: clean source text used to create retrievable chunks; only the
  legacy eager-knowledge mode injects it directly into agent context.
- `source_file`: optional uploaded source.
- `tags`: optional classification.
- `language`: content language.
- `status`: draft, active or archived state.
- `notes`: admin notes.

Operational note:

```text
Keep curated_text concise. Very large documents should be summarized or retrieved selectively.
```

## `KnowledgeDocumentChunk`

Stores one retrievable section of a knowledge document.

Important fields:

- `document`: parent knowledge document.
- `chunk_index`: stable position in the document; platform ingestion uses
  one-based numbering.
- `section_title`: optional human-readable heading.
- `content`: chunk text returned by retrieval tools.
- `token_estimate`: approximate size for prompt budgeting.
- `metadata`: source or extraction details.

Chunks let agents browse, search, read and cite knowledge without injecting the
entire document into every prompt.

## `AgentProfile`

Defines a reusable AI worker.

Important fields:

- `name`: clear agent name.
- `role`: short role description.
- `system_prompt`: core behavior instruction.
- `model_config`: model used by the agent.
- `tools`: allowed tool definitions.
- `knowledge_collections`: attached context.
- `knowledge_max_chars`: legacy eager-text context budget; retrieval-first does
  not inject document bodies.
- `input_contract`: expected input shape.
- `output_contract`: expected output shape.
- `execution_mode`: intended mode or behavior marker.
- `is_active`: whether the agent can run.

Agents are shared across workspaces. The admin should make their usage clear:

- Pipeline only.
- GAME only.
- Both.
- Unused.

## `PipelineDefinition`

Defines a fixed Orchestrator workflow.

Important fields:

- `name`: pipeline name.
- `description`: what the pipeline does.
- `is_active`: whether the pipeline can run.
- `entry_agent`: optional main agent reference.
- `global_input_contract`: expected session input.
- `global_output_contract`: expected final output.

Pipelines should describe stable processes. If the next action depends on agent
decisions, consider GAME instead.

## `PipelineStep`

Defines one ordered step inside a pipeline.

Important fields:

- `pipeline`: parent pipeline.
- `agent`: agent that runs this step.
- `order`: execution order.
- `input_mapping`: context-to-agent mapping.
- `output_mapping`: agent-to-context mapping.
- `on_error`: stop, continue or fallback behavior.
- `fallback_agent`: optional fallback agent.

Rules:

- Step order should be unique within a pipeline.
- Mappings should be explicit.
- Fallback agents should have compatible contracts.

## `GameWorkspace`

Defines the persistent environment in which GAME goals exist.

Important fields:

- `name`: unique workspace name.
- `description`: human-readable purpose.
- `is_active`: whether the workspace may later participate in scheduling.
- `default_policy`: workspace safety, allow-list and budget configuration used
  by GAME policy services.
- `default_runtime_config`: defaults for future goal-bound sessions.

A workspace owns goals but does not schedule or execute them by itself.

## `GameGoal`

Defines durable work independently from any execution session.

Important fields:

- `workspace`: owning GAME environment.
- `title` and `description`: the work to achieve.
- `status`: explicit lifecycle state.
- `base_priority` and `calculated_priority`: stored priority inputs and output.
- `due_at`: optional deadline.
- `queued_at`: start of the current queue wait, reset on retry or reopen.
- `success_criteria`, `context`, and `result`: portable structured data.
- `transition_metadata`: latest controlled lifecycle transition, kept outside agent context.

Status changes must use `transition_goal_status()` or `reopen_goal()`. Completed and cancelled goals require explicit reopening. Priority calculation and claiming remain service-layer responsibilities rather than model behavior.

The deterministic scheduler now calculates and persists `calculated_priority` when priorities are refreshed or a goal is claimed. The stored value is explanatory telemetry; eligibility is checked separately and never inferred from score alone.

## `GameGoalDependency`

Defines a directed relationship from one goal to a prerequisite goal.

Rules:

- Both goals must belong to the same workspace.
- A goal cannot depend on itself.
- Duplicate and circular dependencies are rejected.
- Incomplete required dependencies are reported as blockers.
- Optional dependencies do not block a goal.

Deleting a workspace intentionally cascades to its goals and their dependency records.

## GAME actions, memory, and policy records

`GameActionDefinition` is the allow-listed action registry. It stores action type, optional linked `ToolDefinition`, contracts, adapter configuration, risk level, approval requirement, and active state. Executable Python remains registered in code rather than stored as arbitrary database source.

`GameActionRun` is one durable selected-action attempt. Its unique idempotency key includes session, step, action, and canonical input. Status, payloads, observation, error, latency, and timestamps form the action audit trail. Known contract, policy, budget, and handler failures are retained as failed runs.

`GameMemoryEntry` stores bounded workspace, goal, session, or action-result memory. The database enforces the basic scope shape; service validation additionally checks relationships that span tables, including workspace and goal ownership. Importance is constrained to `0.00`–`1.00`.

`GameContinuationRequest` records why a session paused and permits only one pending continuation per session. `GameActionApprovalRequest` links an approval decision to exactly one action run and records reviewer, note, expiry, and status.

`GameWorkspaceAction` and `GameWorkspaceAgent` configure per-workspace permissions. Once at least one mapping of a type exists, that mapping set is treated as a closed allow-list. Policy services—not model output—decide permissions, approvals, external-write safety, and budgets.

`GameGoalPlan.revision_history` stores JSON snapshots when `revise_plan()` advances the plan version. Plan versions and step orders start at one; step dependencies must point backwards within the same plan, and self/circular dependencies are rejected.

`GameDelegationRun` validates its parent action/goal, delegated session/target-agent relationship and terminal timestamp. Admin treats delegation rows as immutable audit records.

## `ExecutionSession`

Defines one runtime execution.

Important fields:

- `runtime_kind`: orchestrator or GAME.
- `runtime_mode`: sync, async or hybrid (hybrid is supported only by
  Orchestrator; GAME rejects it).
- `status`: pending, running, waiting_async, success, failed or cancelled.
- `pipeline`: pipeline used by Orchestrator sessions.
- `entry_agent`: entry agent used by GAME sessions.
- `goal`: optional durable GAME goal; null for legacy GAME and Orchestrator sessions.
- `goal_text`: GAME goal or user-readable objective.
- `runtime_config`: session runtime options.
- `initial_context`: original execution context.
- `final_context`: final session state.
- `goal_outcome_fingerprint`: idempotency marker for session-to-goal outcome application.
- `error_detail`: failure details.
- `started_at`: run start time.
- `finished_at`: run finish time.

Host projects should link their own records to this model.

The goal relation uses `PROTECT`, so a goal with execution history cannot be deleted accidentally. One goal may have multiple historical sessions, while a conditional database constraint (`aihub_unique_active_goal`) permits only one active session per goal — where "active" means status `pending`, `running` or `waiting_async`.

Example host relation:

```python
class ReportRun(models.Model):
    source_document = models.ForeignKey("documents.Document", on_delete=models.CASCADE)
    ai_session = models.OneToOneField("ai_hub.ExecutionSession", on_delete=models.PROTECT)
    final_report = models.TextField(blank=True)
```

## `ExecutionStepRun`

Defines one observable step, agent call or tool action inside a session.

Important fields:

- `session`: parent session.
- `order`: step or iteration order.
- `pipeline_step`: related pipeline step when applicable.
- `agent`: agent used for this step.
- `action_name`: GAME action or tool/action label.
- `status`: pending, running, success, failed or skipped.
- `request_payload`: payload sent to the runtime.
- `response_payload`: response received from the runtime.
- `observation_payload`: observations or tool results.
- `latency_ms`: measured latency.
- `error_detail`: failure details.

Step runs are the primary audit trail for debugging model behavior.

## `ToolExecutionRun`

Stores one deliberate tool execution through the unified runtime.

Important fields:

- `session`: optional execution session context.
- `step_run`: optional parent step.
- `agent`: agent that requested the tool.
- `tool`: tool definition executed.
- `status`: pending, running, waiting_approval, success, failed or skipped.
- `risk_level`: operational risk captured at execution time.
- `approval_state`: approval outcome when the tool is approval-gated.
- `idempotency_key`: dedupe key so a retried call is not executed twice.
- `input_payload`, `output_payload`, `error_detail`: audit payloads.
- `latency_ms`: measured execution time.

GAME action runs remain the durable audit record for selected GAME actions.
`ToolExecutionRun` records reusable tool runtime calls so non-GAME and GAME
adapter usage share the same execution trail.

## Host Adapter Models

Host projects may add domain-specific models such as:

- `DocumentProcessingRun`.
- `TicketTriageRun`.
- `ReportGeneration`.
- `CustomerReplyDraft`.
- `InvoiceReview`.

These models should not live in AI Hub. They should link to
`ExecutionSession`, store final domain output, and expose whatever the host UI
needs.

## Status Semantics

Recommended status meaning:

| Status | Meaning |
| --- | --- |
| `pending` | Created but not started. |
| `running` | Runtime is currently working. |
| `success` | Finished successfully. |
| `failed` | Finished with an error. |
| `waiting_async` | Paused for external input or asynchronous continuation. |
| `waiting_approval` | Paused pending a human approval (tool/action runs). |
| `cancelled` | Ended deliberately (e.g. orphaned-goal cleanup, manual cancel). |
| `skipped` | Step/tool run not executed (e.g. an unmet condition). |

Host projects can display friendlier text, but should not change the core
meaning.

## JSON Payload Rules

AI Hub deliberately stores many runtime fields as JSON. This makes the platform
flexible and reusable, but it requires discipline.

Recommended rules:

- Keep JSON inspectable by admins.
- Use contracts for expected shape.
- Avoid storing raw secrets.
- Store large files outside JSON fields.
- Keep final domain persistence in the host app.
- Use stable key names so mappings and tests remain reliable.

## Migration Rules

For reusable app migrations:

- Keep migrations domain-neutral.
- Do not import host app models.
- Do not create invoice-, ticket-, report- or customer-specific tables.
- Prefer additive changes when possible.
- Document any destructive reset clearly before applying it.

For host projects:

- Link to `ai_hub.ExecutionSession`.
- Store host-specific result tables in the host app.
- Keep adapter migrations outside AI Hub.

## Data Safety

Recommended production rules:

- Protect execution sessions referenced by host records.
- Avoid deleting provider/model records that old sessions depend on.
- Use inactive status instead of deletion for operational changes.
- Back up execution telemetry before destructive maintenance.
- Purge sensitive request/response payloads only through an explicit retention
  policy.

## Admin Display Rules

The admin should help users understand relationships without forcing them to
read raw JSON first.

Recommended list displays:

- Providers: active status, type, model count.
- Models: provider, active status, tool support.
- Agents: model, pipeline usage, GAME usage, active status.
- Pipelines: active status, step count, recent runs.
- Sessions: status, runtime kind, related pipeline/agent, error summary.
- Step runs: session, order, agent, action, status, latency.

The database may be technical, but the admin should remain readable.
