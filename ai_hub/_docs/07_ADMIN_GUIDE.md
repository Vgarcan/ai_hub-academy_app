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

The AI Hub admin index also exposes `GAME workspaces`, `GAME goals`, and `GAME goal dependencies`. Use those model pages to define durable work and dependency relationships. They do not schedule or launch sessions automatically.

Goal lists can be filtered by workspace, status, due date, and calculated-priority range. Dependency records must connect goals inside the same workspace and are validated against duplicates and cycles.

### Workspace operational dashboard

Each workspace has a **Dashboard** link in its changelist row. The dashboard shows the following scoped to that workspace:

- goal status counts,
- top eligible goals with an inline scheduler explanation (base priority plus each bonus),
- pending approval requests,
- blocked goals (queued goals with unresolved required dependencies),
- recent execution sessions,
- workspace policy — enabled agents, enabled actions, and budget.

The dashboard requires staff access and the `view_executionsession` permission.

### Goal detail enrichments

The change form for a `GameGoal` includes additional read-only panels below the main fields:

- **Scheduler explanation** — shows base priority, each bonus and its reason, and the calculated total; visible when the goal is queued or running.
- **Resume** — indicates when the goal has a waiting session that can be resumed; only appears when the goal is in a waiting status with at least one `WAITING_ASYNC` session.
- **Session history** — all execution sessions linked to the goal, ordered by most recent.
- **Action runs** — all `GameActionRun` records linked to sessions of the goal, ordered by `started_at`.
- **Goal plan** — the attached `GameGoalPlan` (if any) with its ordered steps and statuses.
- **Memory entries** — the most important `GameMemoryEntry` records for the goal.

### Approval operations

The `GAME action approval requests` changelist supports two bulk actions:

- **Approve selected action requests** — calls `approve_action_run()` for each selected pending request.
- **Reject selected action requests** — calls `reject_action_run()` for each selected pending request.

Both actions require the `ai_hub.approve_game_action` permission. Users without the permission do not see these actions in the dropdown and cannot execute them even via a direct POST.

Approval is gated by `AI_HUB_GAME_ACTION_DISPATCH_ENABLED`: when that kill-switch is off, approving is refused outright so an action is never left approved-but-unexecuted.

### Feature flags and the admin

GAME kill-switch flags (`AI_HUB_GAME_GOALS_ENABLED`, `AI_HUB_GAME_SCHEDULER_ENABLED`, etc.) gate the **service layer** — `create_goal`, `claim_next_goal`, `execute_game_action`, `record_memory`, `resume_goal_execution`, and `run_delegated_agent`.

They do **not** gate direct Django admin model edits. The standard "Add GAME goal" form writes through the model, so it bypasses both the flag and the `create_goal` "must start draft/queued" rule. Likewise, the goal bulk actions (`queue`/`cancel`/`reopen`) call `transition_goal_status`, which is not flag-gated. If you need a hard stop on goal creation in an environment, disable the relevant add/change admin permissions in addition to the feature flag.

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

Goal-bound GAME sessions show their durable goal and can also be filtered by goal workspace. Legacy GAME sessions continue to show their standalone `goal_text` without requiring a goal record.

Execution payload dashboards require both staff access and the `view_executionsession` permission. Running a session through the internal endpoint requires `change_executionsession`. Runtime-generated statuses, final contexts, outcome fingerprints, and step telemetry are read-only in Admin.

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
