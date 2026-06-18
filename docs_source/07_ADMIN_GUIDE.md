# Admin Guide

## Purpose

The `ai_hub` Admin is designed as an AI operations backend, not only a raw Django model index.

It should help non-expert staff users understand:

- what to configure first,
- what each object means,
- how Orchestrator differs from GAME,
- where agents, tools, and knowledge are shared,
- how to inspect runs,
- how to debug failures.

## AI Hub Home

Open:

```text
/admin/ai_hub/
```

The home page shows:

- a guided overview,
- setup metrics,
- recommended next action,
- Orchestrator and GAME entry points,
- shared resources,
- examples and blueprints,
- advanced Django records.

Use this page as the first stop for new admins.

## Guided Changelists

Resource list pages include a short explanation and quick actions.

Guided sections exist for:

- Providers,
- Models,
- Tools,
- Knowledge collections,
- Knowledge documents,
- Agents,
- Pipeline steps,
- Execution step runs.

Each page should explain what the resource is and where it fits in the system.

## Guided Forms

`ai_hub` forms include:

- field descriptions,
- placeholders,
- JSON examples,
- prompt examples,
- subtle light/dark styling,
- clearer fieldsets,
- better error presentation.

Examples appear in fields such as:

- provider base URL,
- API key environment variable name,
- model name,
- agent prompt,
- input contract,
- output contract,
- tool schema,
- pipeline mappings,
- GAME goal,
- initial context JSON.

These examples are intentionally small. They should teach shape, not become huge templates.

## Control Center

Open:

```text
/admin/ai_hub/pipelinedefinition/control-center/
```

The control center shows global operational health:

- providers,
- models,
- agents,
- pipelines,
- Orchestrator sessions,
- GAME sessions,
- success rate,
- failures,
- average latency,
- warnings.

It also includes a visual connection graph.

Use the graph to inspect how:

```text
provider -> model -> agent -> knowledge/tools -> pipeline -> step
```

connects across the system.

## Orchestrator Workspace

Open:

```text
/admin/ai_hub/workspaces/orchestrator/
```

Use this when the workflow is a known ordered path.

The workspace shows:

- pipeline metrics,
- recent Orchestrator sessions,
- pipeline cards,
- links to the control center graph.

## GAME Workspace

Open:

```text
/admin/ai_hub/workspaces/game/
```

Use this when one agent receives a goal and decides the next action.

The workspace shows:

- GAME metrics,
- a GAME visual map,
- recent GAME sessions,
- recommended GAME agents.

## Creating A GAME Session

Open:

```text
/admin/ai_hub/executionsession/game/new/
```

Fill:

- Entry agent,
- Goal,
- Max iterations,
- Runtime mode,
- Strict response contract,
- Optional source label,
- Optional initial context JSON.

Recommended first values:

```text
max_iterations = 3
runtime_mode = async
strict_response_contract = true
```

Keep the first goal small. Inspect the timeline before increasing complexity.

## Agent List

The agent list shows workspace usage:

- `Orchestrator`
- `GAME`
- `Both`
- `Unused`

It also shows counts for pipeline usage and GAME sessions.

Use this to avoid losing track of which agents are operational and which are only drafts.

## Execution Sessions

Execution sessions are generic run records.

Use filters to separate:

- Orchestrator sessions,
- GAME sessions,
- status,
- runtime mode,
- pipeline.

The session change page includes a timeline of step runs with:

- step order,
- agent,
- action,
- status,
- latency,
- message,
- error detail.

## Recommended Admin Workflow

1. Configure a provider.
2. Configure a model.
3. Configure one narrow agent.
4. Add knowledge and tools only if needed.
5. Choose Orchestrator or GAME.
6. Run one small test.
7. Inspect the timeline.
8. Inspect the control center.
9. Expand only after the small test is stable.
