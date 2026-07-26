# GAME Workspace

## Purpose

The GAME workspace is for autonomous goal sessions.

Use it when an agent should:

- receive a goal,
- inspect context,
- decide an action,
- update memory,
- continue or finish,
- stop after a configured budget.

## Mental Model

GAME is not a fixed pipeline.

An Orchestrator pipeline says:

```text
Run step 1, then step 2, then step 3.
```

A GAME session says:

```text
Here is the goal. Decide what to do next until the goal is complete or the session stops.
```

## Main Models

- `GameWorkspace`
- `GameGoal`
- `GameGoalDependency`
- `ExecutionSession`
- `ExecutionStepRun`
- `AgentProfile`
- `GameActionDefinition` and `GameActionRun`
- `GameMemoryEntry`
- `GameContinuationRequest` and `GameActionApprovalRequest`
- `GameWorkspaceAction` and `GameWorkspaceAgent`
- `GameGoalPlan` and `GameGoalPlanStep`
- `GameDelegationRun`
- `ToolDefinition`, when a GAME action wraps a reusable tool
- `ToolExecutionRun`, when the unified tool runtime executes that wrapped tool

A workspace contains durable goals. A goal describes what must be achieved; an execution session records one particular run. Creating a workspace or goal does not start work automatically.

## Feature flags

The whole GAME subsystem is gated by feature flags so a host project can adopt
it incrementally. In this bundled host the flags default to the value of
`DEBUG`: enabled in development and disabled in production. The underlying
reusable safety default is **fail-closed**.

| Flag | Gates |
| --- | --- |
| `AI_HUB_GAME_GOALS_ENABLED` | `create_goal` |
| `AI_HUB_GAME_SCHEDULER_ENABLED` | `claim_next_goal` |
| `AI_HUB_GAME_ACTION_DISPATCH_ENABLED` | `execute_game_action` and approval dispatch |
| `AI_HUB_GAME_MEMORY_ENABLED` | `record_memory` |
| `AI_HUB_GAME_RESUME_ENABLED` | `resume_goal_execution` |
| `AI_HUB_GAME_DELEGATION_ENABLED` | `run_delegated_agent` |
| `AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED` | GAME actions linked to reusable `ToolDefinition` records |

Flags primarily gate the service layer. Some Admin entry points also hide
operations, but direct model writes and not every lifecycle helper are uniformly
gated. Enforce database/Admin permissions as well when an environment requires
a hard stop. See `03_CONFIGURATION.md` and `12_TROUBLESHOOTING.md`.

## Workspaces, goals, and dependencies

A GAME workspace provides the environment and defaults for related goals. Each goal has its own lifecycle, priority values, context, success criteria, and result.

Goal statuses are: `draft`, `queued`, `running`, `waiting_info`, `waiting_approval`, `blocked`, `completed`, `partial`, `failed`, and `cancelled`. Status changes are controlled so completed or cancelled work cannot be silently restarted.

Goals may depend on other goals in the same workspace:

- Required unfinished dependencies block later scheduling.
- Optional dependencies are informational and do not block.
- Self-dependencies, duplicates, cross-workspace links, and cycles are rejected.

## Deterministic scheduling

GAME can rank queued goals without asking an AI model to choose what runs next. The score combines base priority with transparent bonuses for deadlines, queue age, and goals that unlock other work.

A goal is eligible only when:

- its workspace is active,
- its status is queued,
- all required dependencies are completed.

Blocked, waiting, failed, completed, and cancelled goals are never selected unless their lifecycle is explicitly changed first. Optional dependencies do not prevent selection.

Reading the next eligible goal does not change data. Claiming work is a separate transactional operation that records current scores and marks one goal as running. Claiming still does not create or run an execution session.

## Goal sessions and history

A claimed or queued goal can create a goal-bound execution session. The session remains a record of one particular run; it is not the goal itself.

```text
Workspace
    Goal
        Execution session 1
        Execution session 2
        Execution session 3
```

Only one active session is allowed per goal. Terminal sessions remain as history, so failed, partial, reopened, or retried goals can have multiple runs over time.

Outcome updates are idempotent: replaying an already applied historical result cannot overwrite a newer attempt. Queue-age priority is measured from the most recent queue or reopen time, rather than the original creation date.

When a run ends, its explicit outcome updates the goal centrally:

- achieved goals become completed,
- iteration-limited goals become partial,
- failed runs mark the goal failed,
- information or approval pauses map to their matching waiting states and can resume at the next unused step.

Legacy GAME sessions remain supported without a durable goal. They continue to use `goal_text`, runtime configuration, and initial context exactly as before.

## Required Session Data

A GAME session needs:

- `runtime_kind=game`
- `entry_agent`
- `goal_text`
- `runtime_config`
- `initial_context`

The guided Admin form is available at:

```text
/admin/ai_hub/executionsession/game/new/
```

## Runtime Config

Common fields:

```json
{
  "max_iterations": 3,
  "strict_response_contract": true,
  "available_actions": [],
  "game_action_dispatch_enabled": true,
  "game_memory_max_chars": 4000,
  "game_memory_max_entries": 20,
  "game_observations_max_entries": 8,
  "game_observation_max_chars": 2000,
  "game_previous_response_max_chars": 2000,
  "game_memory_entry_max_chars": 500,
  "policy": {}
}
```

Keep `max_iterations` low while testing a new agent. Increase only after the session timeline looks correct.

The rolling-context limits cap the legacy prompt state. Old entries are dropped
first; oversized observations and previous responses become bounded previews
with audit references. `ExecutionStepRun` and `GameActionRun` retain the raw
payloads.

Tune `max_iterations` to the task. Classification or binary-decision goals usually need only one or two steps — keep the cap low so an indecisive model is stopped early with a clean `partial` rather than looping. Planning or research goals can use a higher cap.

GAME currently supports sync and async sessions. Explicit approval/information pauses can resume safely without reusing step order numbers. GAME Hybrid remains intentionally unavailable because its automatic-continuation contract is separate from explicit resume.

## Goal Examples

Good goals are specific, bounded, and testable.

Examples:

```text
Review this support ticket and decide the next best response.
```

```text
Read the uploaded research notes and produce a short action plan.
```

```text
Explore the context, identify missing information, and finish with a clear recommendation.
```

## GAME Response Contract

The agent should return one JSON object.

Expected shape:

```json
{
  "action": "finish",
  "message": "The goal is complete.",
  "complete": true,
  "final_answer": "Final result."
}
```

When `strict_response_contract` is enabled, missing keys or invalid JSON fail the session.

## Understanding the result

Session execution and goal completion are reported separately:

- An agent that finishes the goal records a completed execution and an achieved goal.
- Reaching `max_iterations` records a completed execution but an incomplete (partial) goal.
- A runtime error records a failed execution and an unknown goal outcome.

These values appear in the session's final context together with `finish_reason`.

## Tool safety

GAME automatically prepares context only with tools explicitly classified as read-only context tools. Tools that perform actions, and tools without a valid category, are not run automatically.

When the explicit dispatcher is enabled, GAME executes only the action selected in the validated model decision. Known attempts are audited, workspace policy and budgets run before execution, and approval-required work pauses immediately. External writes are disabled unless workspace policy explicitly enables them.

Reusable capabilities should be defined as `ToolDefinition` records and grouped
through toolboxes/grants on the agent. A `GameActionDefinition` may then link to
one of those reusable tools as a governed GAME wrapper. With
`AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=true`, GAME creates a `GameActionRun`,
resolves one effective agent (direct entry agent or first pipeline-step agent),
validates workspace action policy and that agent's Tool access, and combines the
action and effective Tool approval decisions. If either requires review, GAME
pauses before contacting the Tool. An approved execution records the reusable
call in `ToolExecutionRun` with the effective agent and approved audit state.

The normal GAME model-call path now uses the same resolved manifest as
Orchestrator, filtered to explicitly safe `context_tool` capabilities. The
model chooses a context tool; action tools never enter this manifest, even if a
caller passes the old `allow_legacy_game_action_tools` argument. Selected
side-effect actions continue through the dispatcher above because that path owns
GAME policy, budgets, approval, resume and `GameActionRun` audit.

`agent_tool_runtime="legacy_preexecute"` remains a temporary per-session
compatibility mode. Only in that mode does the explicit
`allow_legacy_game_action_tools=True` host opt-in retain its old meaning.

Keep GAME control actions separate from ordinary tools. Actions such as `finish_goal`, `record_memory`, `update_goal_status`, and `delegate_to_agent` remain GAME control flow and should not be replaced by generic tool records.

If workspace agent or action mappings are configured, they operate as closed allow-lists: entries absent from the configured list are not permitted. With no mappings, legacy sessions retain their previous behavior.

## Scoped memory and continuation

Goal-bound payloads include bounded `scoped_memory` selected from the active
workspace, goal, and session. Scope validation rejects memory belonging to
another workspace or goal. A successful `record_memory` action refreshes this
context before the next iteration. Workspace, goal, session and action-result
actions now pass only the links allowed by their scope.

The existing `memory`, `observations` and `previous_response` state remains for
legacy compatibility, but each component has entry/character limits so long
sessions do not carry raw action payloads indefinitely.

`compact_goal_memory()` is currently a manual expiry helper, not automatic
summarisation. The normal runner does not invoke it. Episodic checkpoints and
memory retrieval are not implemented.

An approval pause creates one pending action, approval request, and continuation. Approval or rejection is stored as an observation before resume, so the next agent iteration receives the human decision and action result. A pending approval cannot be resumed prematurely.

The same effective agent identity scopes legacy selected Knowledge actions,
resolved Knowledge adapters, workspace agent allow-lists, self-delegation
checks, Tool audit and the GAME timeline. Agent identifiers supplied in action
input cannot replace that server-resolved identity.

## Goal Plans

A `GameGoalPlan` is an optional structured execution aid attached one-to-one to a goal.

Plans are created by the agent or operator before or during execution and refined as work progresses. They are informational: the agent uses them as a reference, not as an enforced pipeline.

Each plan contains ordered `GameGoalPlanStep` records:

- Each step has a title, description, and status (`pending`, `in_progress`, `completed`, `skipped`, `blocked`).
- Steps can declare an optional same-plan dependency on a prior step.
- Self-dependencies and cross-plan dependencies are rejected.
- Step order must be unique within the plan.

Use plans when the goal is complex enough that the agent benefits from a durable checklist it can update between iterations.

## Sub-agent Delegation

A GAME agent can delegate a sub-task to another agent using the `delegate_to_agent` dispatcher action.

Every delegation creates a `GameDelegationRun` record that tracks:

- the parent goal and parent action run,
- the target agent,
- the task and expected result,
- the resulting delegated session,
- lifecycle status and finish time.

Rules:

- **Depth limit**: a delegated agent cannot further delegate (maximum depth 1).
- **Self-delegation**: denied by default; it requires explicit workspace policy permission.
- **Allow-list**: if the workspace defines `GameWorkspaceAgent` entries, only enabled agents in that list may be targets.
- **Budget**: `max_sub_agent_runs_per_goal` in workspace policy limits total delegations per goal.
- **Least privilege**: delegated sessions receive only explicitly enabled read-only context actions and use a strict response contract.
- **Policy continuity**: every child action resolves policy and budget from its durable delegation record; approval-gated work must return to the parent.

The delegated session is created without a goal link to avoid the active-goal unique constraint. The parent-goal relationship is tracked through `GameDelegationRun.parent_goal`.

If the delegated session does not finish successfully, the delegation raises an error and marks the parent action run as failed.

## GAME-Ready Agents

An agent is a good GAME candidate when its prompt and input contract mention goal-loop concepts such as:

- `goal`
- `iteration`
- `memory`
- `observations`
- `game_response_contract`
- `action`
- `final_answer`

Example prompt fragment:

```text
You are an autonomous planning agent.
Read the goal, inspect memory and observations, choose one next action, and decide whether the goal is complete.
Under uncertainty, commit to a best-effort decision or use an action to request information — do not re-deliberate silently.
Return valid JSON only with action, message, complete, and final_answer.
```

## Admin UX

Open:

```text
/admin/ai_hub/workspaces/game/
```

The workspace shows:

- GAME session counts,
- running or waiting sessions,
- success and failure counts,
- GAME graph,
- recent GAME sessions,
- agents used in GAME,
- agents that look prepared for GAME.

### Build Console

The **Build Console** button in the workspace header opens the guided creation wizard for a new GAME session. Use it when you want to create the full engine → agent → session chain in one transaction, without navigating multiple raw admin forms. See [`16_BUILD_CONSOLE.md`](16_BUILD_CONSOLE.md).

The GAME graph uses the same Mission Deck graph engine as the global Control
Center. It maps goal, agent, decision, action, memory and stop nodes when those
records exist in the selected context. Graph controls support search, node-kind
chips, hop depth, optional isolation, full-screen mode and a draggable node-detail
pop-up with an Admin record link.

Hovering a node is only a lightweight status preview. Selecting a node focuses
the relevant neighborhood and opens the movable detail pop-up.

### Operational dashboard

Each workspace row in the GAME workspaces changelist includes a **Dashboard** link. The dashboard is an operational control centre scoped to one workspace:

- goal status counts (a metric strip per status),
- top eligible goals with an inline scheduler explanation showing base priority, each active bonus, and the calculated total,
- pending approval requests,
- blocked goals — queued goals whose required dependencies are not yet completed,
- recent execution sessions,
- recent action runs,
- workspace policy panel — enabled agents, enabled actions, and budget.

Dashboard access requires staff status, view/change permission for the
`GameWorkspace` Admin and `ai_hub.view_executionsession`.

### Goal detail enrichments

Opening a goal in the admin shows additional read-only panels below the main fields:

- **Scheduler explanation** — visible for queued and running goals; shows base priority, bonuses, and total.
- **Resume indicator** — appears when the goal is in a waiting status and has at least one `WAITING_ASYNC` session.
- **Session history** — all execution sessions linked to the goal.
- **Action runs** — all `GameActionRun` records for the goal's sessions, ordered by `started_at`.
- **Goal plan** — the attached `GameGoalPlan` with ordered steps.
- **Memory entries** — the top-weighted `GameMemoryEntry` records for the goal.

### Goal lifecycle bulk actions

The `GAME goals` changelist offers bulk lifecycle actions (each routed through the validated service layer so transitions stay consistent):

- **Queue selected goals** — move goals to `queued`.
- **Cancel selected goals** — move goals to `cancelled`.
- **Reopen selected goals** — completed/cancelled goals go back to `queued`.
- **Resume selected goal sessions** — resume a goal that has a waiting session.

### Approval operations

The `GAME action approval requests` changelist has two bulk actions that require `ai_hub.approve_game_action`:

- **Approve selected action requests** — calls `approve_action_run()` from the service layer.
- **Reject selected action requests** — calls `reject_action_run()` from the service layer.

Users without the permission do not see these actions and cannot execute them. Approval also respects `AI_HUB_GAME_ACTION_DISPATCH_ENABLED`: when that kill-switch is off, approving is refused so an action is never left approved-but-unexecuted.

### Cleaning up orphaned running goals

A goal can stay stuck in `running` if its session never reaches a terminal state (an interrupted run, or stub sessions left by integration tests). Cancel those with:

```text
python manage.py cleanup_orphaned_goals
python manage.py cleanup_orphaned_goals --workspace <id>
python manage.py cleanup_orphaned_goals --older-than-hours 24
python manage.py cleanup_orphaned_goals --dry-run
```

The command cancels `running` goals that have **no active session** (pending/running/waiting_async); goals with an active session are never touched.

## Recommendation For v1

Do not put GAME inside Orchestrator pipelines for v1.

Keep them visually and operationally separate:

- Pipeline: known steps.
- GAME: goal, decision, memory, and stopping.

Both can reuse the same agents, tools, and knowledge.
