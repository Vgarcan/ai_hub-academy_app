# Orchestrator Workspace

## Purpose

The Orchestrator workspace is for deterministic agent flows.

Use it when:

- the steps are known,
- the order matters,
- each agent has a clear responsibility,
- outputs from earlier steps feed later steps,
- you want predictable execution and debugging.

## Mental Model

An Orchestrator pipeline is a recipe:

```text
input context -> step 1 agent -> step 2 agent -> mappings -> final output
```

It is not autonomous. It does not decide which step comes next. The pipeline definition decides the path.

## Main Models

- `PipelineDefinition`
- `PipelineStep`
- `ExecutionSession`
- `ExecutionStepRun`

Host projects may also create their own domain result models.

## Pipeline Definition

A pipeline defines the full recipe.

Common fields:

- name,
- description,
- active flag,
- optional entry agent,
- global input contract,
- global output contract,
- ordered steps.

Example name:

```text
support_triage_v1
```

Example description:

```text
Reads a support ticket, classifies it, checks policy knowledge, then drafts a response.
```

Keep a pipeline inactive while building it. Activation validates that steps are continuous and agents have contracts.

## Pipeline Steps

Each step points to one agent.

Fields:

- `order`
- `agent`
- `input_mapping`
- `output_mapping`
- `on_error`
- `fallback_agent`

Step order should be continuous:

```text
1, 2, 3, 4
```

## Mapping Examples

`input_mapping` maps workflow context into the agent input.

Left side: agent input key.

Right side: context path.

```json
{
  "ticket_text": "ticket_text",
  "knowledge_context": "knowledge_context"
}
```

`output_mapping` writes agent output back into workflow context.

Left side: context key.

Right side: response path.

```json
{
  "triage_result": "agent",
  "model_text": "llm.content"
}
```

Dot paths such as `llm.content` are supported.

## Error Strategy

`on_error` supports:

| Value | Behavior |
| --- | --- |
| `stop` | Stop the session as failed |
| `continue` | Record the error and continue |
| `fallback_agent` | Try a fallback agent |

Use `stop` while testing. Use `continue` only for optional enrichment. Use fallback agents only when another agent can genuinely recover the step.

## Admin UX

Open:

```text
/admin/ai_hub/workspaces/orchestrator/
```

The workspace shows:

- pipeline counts,
- active pipeline counts,
- session counts,
- running or waiting sessions,
- failed sessions,
- recent orchestrator sessions,
- pipeline links.

The control center graph shows how providers, models, agents, knowledge, tools, pipelines, and steps connect.

The graph is intentionally data-driven. Columns only appear when that node type
exists in the current configuration. For example:

- no pipelines means no `Pipeline` column;
- pipelines with no configured steps means a `Pipeline` column can appear without
  a `Step` column;
- active pipelines without steps surface as warnings in **Needs attention**.

Use node selection to inspect one-hop, two-hop or three-hop neighborhoods. When
`Isolate` is enabled, nodes outside the selected neighborhood are hidden. Hovering
a node only shows a brief status preview; it does not isolate the graph.

The graph can be opened in full-screen mode from its toolbar. Selecting a node
opens a small draggable detail pop-up with an **Open record in admin** action.

## Recommended Pattern

Keep each agent narrow.

Good:

```text
source_reader -> facts_extractor -> risk_classifier -> final_writer
```

Risky:

```text
mega_agent_that_does_everything
```

Narrow agents make contracts, failures, timelines, and tests easier.
