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
3. builds memory and observations,
4. sends GAME payload to the agent,
5. parses the JSON decision,
6. stores an observation,
7. continues or finishes,
8. stops at max iterations or failure.

When `strict_response_contract` is enabled, invalid GAME decision JSON fails the session.

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
