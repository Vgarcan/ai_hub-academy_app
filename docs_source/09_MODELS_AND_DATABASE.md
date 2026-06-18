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

AI Hub models fall into five groups.

| Group | Purpose |
| --- | --- |
| Providers and models | Connect to AI services. |
| Agents | Define reusable AI workers. |
| Knowledge and tools | Give agents context and capabilities. |
| Orchestrator | Run fixed multi-agent workflows. |
| Execution and GAME | Record sessions, steps and autonomous loops. |

## `ProviderConfig`

Defines an AI provider, endpoint or local inference server.

Important fields:

- `name`: friendly admin name.
- `provider_type`: runtime adapter type.
- `base_url`: endpoint for local or compatible providers.
- `api_key_env_var`: environment variable name for the secret.
- `default_timeout`: request timeout in seconds.
- `config`: provider-specific JSON configuration.
- `is_active`: whether the provider can be used.

Use this model for operational configuration only. Do not store raw API keys in
the database.

## `ModelConfig`

Defines a concrete model available through a provider.

Important fields:

- `provider`: parent provider.
- `display_name`: friendly user-facing name.
- `model_name`: exact provider model identifier.
- `temperature_default`: default generation temperature.
- `max_tokens_default`: default output budget.
- `supports_tools`: whether this model/runtime can work with tools.
- `config`: model-specific JSON configuration.
- `is_active`: whether the model can be used.

Operational note:

```text
Structured JSON workflows usually need low temperature and generous max tokens.
```

## `ToolDefinition`

Defines a reusable tool capability.

Important fields:

- `name`: friendly tool name.
- `tool_kind`: runtime tool category.
- `description`: what the tool does.
- `input_schema`: expected input shape.
- `output_schema`: expected output shape.
- `config`: tool-specific settings.
- `is_active`: whether the tool can be used.

Tools should be treated as controlled capabilities. A tool attached to an agent
does not mean every model can automatically use it. The runtime still decides
whether that tool kind is available and safe.

## `KnowledgeCollection`

Groups related knowledge documents.

Important fields:

- `name`: collection name.
- `description`: what the collection teaches agents.
- `tags`: optional classification.
- `is_active`: whether the collection can be used.

Use collections to attach a coherent body of context to one or more agents.

## `KnowledgeDocument`

Stores curated text and optional uploaded source material.

Important fields:

- `collection`: parent collection.
- `title`: document name.
- `curated_text`: clean text injected into agent context.
- `source_file`: optional uploaded source.
- `source_url`: optional source reference.
- `tags`: optional classification.
- `language`: content language.
- `status`: draft, active or archived state.
- `notes`: admin notes.

Operational note:

```text
Keep curated_text concise. Very large documents should be summarized or retrieved selectively.
```

## `AgentProfile`

Defines a reusable AI worker.

Important fields:

- `name`: clear agent name.
- `role`: short role description.
- `system_prompt`: core behavior instruction.
- `model_config`: model used by the agent.
- `tools`: allowed tool definitions.
- `knowledge_collections`: attached context.
- `knowledge_max_chars`: context budget.
- `input_contract`: expected input shape.
- `output_contract`: expected output shape.
- `execution_mode`: intended mode or behavior marker.
- `config`: agent-specific runtime settings.
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
- `config`: pipeline-specific runtime settings.

Pipelines should describe stable processes. If the next action depends on agent
decisions, consider GAME instead.

## `PipelineStep`

Defines one ordered step inside a pipeline.

Important fields:

- `pipeline`: parent pipeline.
- `agent`: agent that runs this step.
- `order`: execution order.
- `name`: optional step label.
- `input_mapping`: context-to-agent mapping.
- `output_mapping`: agent-to-context mapping.
- `on_error`: stop, continue or fallback behavior.
- `fallback_agent`: optional fallback agent.
- `config`: step-specific settings.

Rules:

- Step order should be unique within a pipeline.
- Mappings should be explicit.
- Fallback agents should have compatible contracts.

## `ExecutionSession`

Defines one runtime execution.

Important fields:

- `runtime_kind`: orchestrator or GAME.
- `runtime_mode`: sync, async or host-specific mode.
- `status`: pending, running, success, failed, waiting or stopped.
- `pipeline`: pipeline used by Orchestrator sessions.
- `entry_agent`: entry agent used by GAME sessions.
- `goal_text`: GAME goal or user-readable objective.
- `runtime_config`: session runtime options.
- `initial_context`: original execution context.
- `final_context`: final session state.
- `error_detail`: failure details.
- `started_at`: run start time.
- `finished_at`: run finish time.

Host projects should link their own records to this model.

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
- `status`: pending, running, success, failed, skipped or waiting.
- `request_payload`: payload sent to the runtime.
- `response_payload`: response received from the runtime.
- `observation_payload`: observations or tool results.
- `latency_ms`: measured latency.
- `error_detail`: failure details.

Step runs are the primary audit trail for debugging model behavior.

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
| `waiting` | Paused for external input or continuation. |
| `stopped` | Ended because a limit or stop condition was reached. |

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
