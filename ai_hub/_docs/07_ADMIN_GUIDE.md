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

The home page is a **cockpit**, not a raw model index. It is organized as:

- a **header** with the primary entry points: Control Center, Build Console, Orchestrator, GAME and Operations;
- a **vitals strip** — *Live now* (running + waiting sessions), *Needs you* (approvals, info requests, failures), *Goals in flight*, and *Sessions*;
- an inventory line (providers · models · agents · pipelines · GAME sessions);
- a **Needs your attention** queue that links to the Operations Inbox, plus a Recent activity feed;
- a **five-area navigation map**:
  - **0 · Overview & Entry** — Control Center, Build Console;
  - **1 · Foundation** — Connectivity, Knowledge Library, Tool Registry, Agent Composer;
  - **2 · Orchestrator** — the Orchestrator workspace and new pipelines;
  - **3 · GAME** — the GAME workspace and new GAME sessions;
  - **4 · Operations** — Operations Inbox, Session Explorer;
- a **System health** bar and a **setup checklist** with a recommended next action;
- an **All records** panel (the raw Django tables, grouped by stage) with a **"Show supporting tables"** toggle that reveals the supporting/bridge tables demoted from the index (see *Guided Changelists* below).

Use this page as the first stop for new admins.

## Build Console

Open:

```text
/admin/ai_hub/workspaces/build/?kind=game
/admin/ai_hub/workspaces/build/?kind=orchestrator
```

The Build Console is a multi-step guided wizard that creates a complete GAME session or Orchestrator pipeline in one atomic transaction. It covers engine, agent, tools, knowledge and workspace configuration without requiring you to navigate multiple raw admin forms.

Use the Build Console when starting from scratch or testing a new configuration. Use the raw admin forms when editing existing objects.

See [`16_BUILD_CONSOLE.md`](16_BUILD_CONSOLE.md) for the full step-by-step reference.

## Operations Inbox

Open:

```text
/admin/ai_hub/operations/
```

The Operations Inbox is the single cross-workspace queue for everything that needs a human. It aggregates four categories:

- **Approvals** — pending `GameActionApprovalRequest` records, with inline **Approve** / **Reject** buttons (shown only to users holding `ai_hub.approve_game_action`) and an **Open** link;
- **Waiting info** — paused sessions with a pending `GameContinuationRequest`, linking to the goal;
- **Failures** — recently failed execution sessions (most recent 25);
- **Blocked goals** — GAME goals in `blocked` status (most recent 25).

Filter chips at the top narrow the list to one category. Approvals and waiting-info items are unbounded (they are the actionable gates); failures and blocked goals are capped. The page is reached from the home header, the home "Needs your attention" panel, and the Operations area card. It requires staff access and the `view_executionsession` permission.

The Operations Inbox is the action queue; the Control Center (below) is the deeper diagnostics surface.

## Guided Changelists

Resource list pages include a short explanation and quick actions.

Guided sections exist for the primary resources:

- Providers (Connectivity),
- Models,
- Tools and Toolboxes (Tool Registry),
- Knowledge collections and documents (Knowledge Library),
- Agents (Agent Composer).

### Supporting tables hidden from the index

Bridge tables, structural children and runtime/audit records are **demoted from the admin index and sidebar** via `AIHubHideFromIndexMixin` (it returns empty model perms so the model is registered and its URLs work, but it is not listed). These are normally managed through their parent page (e.g. pipeline steps inside the Pipeline Designer, document chunks inside their document) and include: knowledge document chunks, toolbox↔tool and agent↔toolbox/grant links, pipeline steps, goal dependencies/plans/plan-steps, workspace actions/agents, memory entries, and the runtime audit records (execution step runs, tool execution runs, GAME action runs, delegation runs).

They remain fully reachable from the **"Show supporting tables"** toggle on the *All records* panel of the AI Hub home, which lists each one grouped by category (Bridge tables · GAME structural children · Runtime / audit) with a one-line reason it is hidden. If a model "disappeared" from the sidebar, this is where to find it.

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

## Composed change pages

The most important entities open as **composed change pages**: a header, a tab bar, and an at-a-glance **Overview** tab in front of the editable **Configuration** form. Tabs progressively enhance — without JavaScript the panels stack; with it they group, and on a validation error the page lands on the Configuration tab so the errors are visible.

- **Root entities** (drive the flows): Agent Composer (`AgentProfile`), Orchestrator Designer (`PipelineDefinition`), GAME Workspace (`GameWorkspace`), Goal Detail (`GameGoal`), Session Explorer (`ExecutionSession`).
- **Foundation hubs** (the reusable parts): Connectivity (`ProviderConfig`), Knowledge Library (`KnowledgeCollection`), Tool Registry (`Toolbox`).

Each Overview summarizes what the entity contains, what uses it, and a small health checklist; the Configuration tab holds the normal Django form and inlines. Leaf records (e.g. `ModelConfig`, `KnowledgeDocument`, `ToolDefinition`) keep the standard guided form.

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

It also includes the Control Center visual console (built on the shared "Mission Deck" graph engine):

- vitals and inventory,
- a connection graph,
- an operational attention inbox,
- detail tabs for pipelines, models, sessions, critical nodes and checklist.

### Connection graph

Use the graph to inspect how:

```text
provider -> model -> agent -> knowledge/tools -> pipeline -> step
```

connects across the system.

The graph is built from the existing Admin context. It does not call a separate
API endpoint. Nodes appear only for object types that exist in the current
configuration:

- if there are no pipelines, the `Pipeline` column is absent;
- if there are pipelines but no steps, the `Pipeline` column appears but the
  `Step` column is absent;
- if a pipeline is active without steps, the attention inbox raises a warning.

Graph controls:

- **Pipeline scope** filters the graph to one configured pipeline and its related
  providers, models, agents, knowledge and tools.
- **Search** filters nodes by label, kind and detail.
- **Hop depth** controls how far selection focus expands.
- **Isolate** hides nodes outside the selected node's hop neighborhood. It is off
  by default so selection remains easy to understand on first load.
- **Full screen** opens the graph as a page-level overlay. Use the same button or
  `Esc` to exit.
- Type chips show or hide node kinds.

Hovering a node shows a compact status preview. Selecting a node opens a small
draggable pop-up with the same node detail, relation counts, incoming/outgoing
relations and a prominent **Open record in admin** action.

### Needs attention inbox

The **Needs attention** section normalizes configuration warnings and runtime
failures into incident-like attention items. Each item includes:

- severity,
- source type,
- incident date when the underlying record has one,
- detail text,
- hover detail,
- a link to the relevant Admin record when available.

Use **Open incident** to jump to the concrete record:

- provider warnings open the provider;
- missing-model warnings open the model;
- empty knowledge warnings open the collection;
- incomplete-agent-contract warnings open the agent;
- active-pipeline-without-steps warnings open the pipeline;
- failed pipeline-step incidents open the latest failed `ExecutionStepRun`.

The inbox supports:

- filter by severity or source type,
- sort by newest, relevance or severity,
- archived view,
- restore,
- silence for 24 hours.

Archive and silence are non-destructive. They are stored in browser
`localStorage` under `aiHubMissionDeckAttention:v1`. They do not delete or modify
database records, and they are local to that browser/profile.

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
- a GAME decision graph using the same shared graph engine,
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

The primary place to act on approvals is the **Operations Inbox** (`/admin/ai_hub/operations/`), which lists every pending request with inline **Approve** / **Reject** buttons. The approval change page also exposes per-record POST approve/reject controls.

For bulk handling, the `GAME action approval requests` changelist supports two actions:

- **Approve selected action requests** — calls `approve_action_run()` for each selected pending request.
- **Reject selected action requests** — calls `reject_action_run()` for each selected pending request.

All of these paths require the `ai_hub.approve_game_action` permission. Users without it do not see the controls and cannot execute them even via a direct POST.

Approval is gated by `AI_HUB_GAME_ACTION_DISPATCH_ENABLED`: when that kill-switch is off, approving is refused outright so an action is never left approved-but-unexecuted.

### Cleaning up orphaned running goals

A goal can stay stuck in `running` if its execution session never reaches a terminal
state (an interrupted run, or stub sessions created by integration test suites).
These never clean themselves up. Use:

```text
python manage.py cleanup_orphaned_goals
python manage.py cleanup_orphaned_goals --workspace 3
python manage.py cleanup_orphaned_goals --older-than-hours 24
python manage.py cleanup_orphaned_goals --dry-run
```

The command cancels `running` goals that have **no active session**
(pending/running/waiting_async). Goals with an active session are never touched.
Use `--dry-run` first to see what would be cancelled. This is distinct from
`reconcile_goal_outcomes()`, which only repairs goals whose session already reached
a terminal state.

### Feature flags and the admin

GAME kill-switch flags (`AI_HUB_GAME_GOALS_ENABLED`, `AI_HUB_GAME_SCHEDULER_ENABLED`, etc.) gate the **service layer** — `create_goal`, `claim_next_goal`, `execute_game_action`, `record_memory`, `resume_goal_execution`, and `run_delegated_agent`.

The goals flag removes goal creation and lifecycle bulk actions from Admin. Runtime audit records (action runs, continuations, approvals and delegations) are immutable; approval/rejection use dedicated POST controls that call the service layer and preserve an optional review note.

Aggregated GAME views apply permissions to each related collection. Approval panels require `approve_game_action`; session, action, memory and configuration panels require their corresponding view permissions. Session/step/action/continuation/approval payloads use the shared redaction layer rather than raw JSON widgets.

## Creating A GAME Session

**Recommended path**: use the Build Console at `/admin/ai_hub/workspaces/build/?kind=game`. It creates the engine, agent, and session together in one transaction and guides you through all required fields.

**Quick form** (existing agent and model already configured):

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

## Tools And Toolboxes

Use `Tools` for the underlying reusable capability and `Toolboxes` for role-level
bundles. Assign toolboxes to agents first, then use individual grants only for
exceptions.

Recommended setup:

1. Create or seed safe tool definitions.
2. Group them into a toolbox.
3. Assign the toolbox to a specific agent role.
4. Add deny grants for any capability the role should not inherit.
5. Keep external-write tools inactive or approval-gated until workspace policy is ready.

Starter configuration can be created from the command line:

```text
python manage.py seed_ai_hub_starter_toolboxes
python manage.py seed_ai_hub_starter_demo
```

The demo seed creates a small knowledge library, starter agents, a safe GAME
workspace, and an approval-gated `submit_for_approval` action. Re-run with
`--force-update` to refresh seed-owned labels, descriptions and policies.

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

**Starting from scratch**: use the Build Console (step 1 below). It handles steps 2–4 inside the wizard.

**Iterating on existing objects**: skip to step 5 and use the raw admin forms.

1. Open the Build Console and choose a blueprint (GAME or Orchestrator).
2. Configure a provider and model (or reuse an existing one).
3. Configure one narrow agent (or reuse an existing one).
4. Add knowledge and tools only if needed.
5. Complete the workspace-specific steps (goal + runtime for GAME; pipeline + steps for Orchestrator).
6. Run one small test.
7. Inspect the timeline.
8. Inspect the control center.
9. Expand only after the small test is stable.
