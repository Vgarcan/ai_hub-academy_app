# Roadmap

This roadmap describes the recommended direction for AI Hub as a reusable Django
AI operations app.

AI Hub should grow carefully. The priority is not to add every AI framework
feature at once. The priority is to make providers, models, agents, knowledge,
tools, Orchestrator workflows and GAME sessions understandable, testable and
portable.

## Current Foundation

AI Hub already has the core platform pieces:

- Provider configuration.
- Model configuration.
- Agent profiles.
- Knowledge collections and documents.
- Tool definitions.
- Orchestrator pipeline definitions.
- Pipeline steps.
- Execution sessions.
- Execution step runs.
- GAME sessions through `ExecutionSession`.
- Admin workspaces.
- Control center.
- Guided admin forms and changelists.
- Runtime services.
- Host adapter pattern.

This is enough to build real AI workflows.

## Product Direction

The product direction is:

```text
Make AI workflow creation possible for non-specialist admin users while keeping the backend reusable for engineers.
```

That means every future feature should pass two questions:

1. Does this make AI Hub easier to operate?
2. Does this keep AI Hub reusable across host projects?

If a feature only helps one host project, it probably belongs in that host app.

## Priority 1: Admin UX Maturity

The admin is the main product surface.

Recommended improvements:

- More guided setup flows.
- Better empty states.
- Clear "next action" panels.
- Inline examples for prompts and contracts.
- Safer JSON editing.
- Responsive improvements for all model pages.
- Workspace-specific dashboards.
- More readable execution error summaries.

Target outcome:

```text
A user with little coding knowledge can configure a small workflow from the admin.
```

## Priority 2: Public Host Adapter API

Create a small stable API for host projects.

Target flow:

```text
host object -> AI Hub session -> execution -> host persistence callback
```

Recommended API functions:

- Create Orchestrator session from context.
- Create GAME session from goal.
- Run session synchronously.
- Queue session asynchronously.
- Read session status.
- Extract final context.
- Attach host metadata.

This should reduce direct model imports in host apps.

## Priority 3: Worker And Queue Contract

Long AI runs should have a clear background execution model.

Recommended work:

- Define pending/running/waiting lifecycle.
- Add a worker command for pending sessions.
- Add retry policy configuration.
- Add timeout handling.
- Add stalled-session detection.
- Add idempotency guards.

Target outcome:

```text
Slow two-minute model runs do not block the user interface and do not create duplicate results.
```

## Priority 4: Tool Runtime Hardening

Tools are powerful and need strict controls.

Recommended work:

- Explicit allowlists per tool kind.
- Safe HTTP tool configuration.
- Safe callable registry.
- Tool execution audit logs.
- Tool timeout configuration.
- Tool error normalization.
- Clear model compatibility rules.

Target outcome:

```text
Agents can use tools only when the app, model and runtime all allow it.
```

## Priority 5: Better Knowledge And RAG

Current knowledge is curated text. The next step is stronger retrieval.

Recommended work:

- Chunking strategy.
- Embedding provider configuration.
- Retrieval query logs.
- Per-agent retrieval limits.
- Knowledge freshness status.
- Source citation metadata.
- Admin preview of injected context.

Target outcome:

```text
Agents receive the right context without forcing admins to paste huge documents into prompts.
```

## Priority 6: GAME Readiness

GAME should become easier to understand and safer to operate.

Recommended work:

- Explicit GAME-ready agent flag.
- GAME session templates.
- Visual GAME timeline.
- Memory preview.
- Stop reason display.
- Continue session action.
- Human approval checkpoints.
- Built-in action registry.

Target outcome:

```text
Users understand why a GAME session continued, waited, stopped or finished.
```

## Priority 7: Import And Export

Reusable apps need portable configuration.

Recommended export objects:

- Providers without secrets.
- Models.
- Agents.
- Knowledge collections.
- Tool definitions.
- Pipeline definitions.
- Pipeline steps.
- GAME templates.

Recommended behavior:

- Export to JSON or YAML.
- Validate before import.
- Show a preview of changes.
- Never export raw API keys.
- Keep host-specific data out of reusable exports.

Target outcome:

```text
A workflow can move from one project to another with minimal manual setup.
```

## Priority 8: Observability And Cost

Operational users need to see health.

Recommended metrics:

- Success rate by workspace.
- Latency by provider/model/agent.
- Failure rate by contract.
- Token usage when available.
- Cost when available.
- Tool usage.
- Waiting/stalled sessions.
- Recent recoveries.

Target outcome:

```text
Admins can tell whether the AI system is healthy without reading logs.
```

## Priority 9: Safer Structured Output

Structured output failures are common with local and long-output models.

Recommended work:

- JSON repair mode.
- Schema-specific retry prompts.
- Final output size warnings.
- Truncation detection.
- Streaming parser support when useful.
- Per-agent response format hints.
- Recovery hooks for host adapters.

Target outcome:

```text
Malformed final JSON becomes an expected operational case, not a confusing failure.
```

## Priority 10: Documentation Examples

Add complete examples for common reusable scenarios:

- Document extraction.
- Support ticket triage.
- Report generation.
- Customer email drafting.
- Moderation workflow.
- Research assistant GAME session.
- Data cleanup workflow.

Each example should include:

- Provider setup.
- Model setup.
- Agent prompts.
- Contracts.
- Pipeline or GAME configuration.
- Host adapter outline.
- Test cases.

## Non-Goals

AI Hub should not become:

- A billing system.
- A public SaaS product by itself.
- A replacement for the host app's domain models.
- A generic CMS.
- An uncontrolled tool execution engine.
- A direct clone of external AI frameworks.

AI Hub can integrate with external frameworks later, but its core value is a
Django-native, admin-first AI operations layer.

## Decision Rules

Use these rules when deciding whether to add something to AI Hub:

- Put reusable AI configuration in AI Hub.
- Put domain-specific persistence in the host app.
- Put UI copy for the AI Hub admin in AI Hub.
- Put public user-facing copy in the host app.
- Add abstractions only when they reduce real duplication.
- Prefer explicit admin guidance over hidden behavior.
- Keep Orchestrator and GAME visually separate.
- Keep providers, models, agents, knowledge and tools shared.
