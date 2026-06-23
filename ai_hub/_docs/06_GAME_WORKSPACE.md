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

The whole GAME subsystem is gated by feature flags so a host project can adopt it incrementally. Each flag defaults to **enabled** in this repo's settings, but the underlying safety default in the reusable layer is **fail-closed**: if a flag is unset and disabled, the matching service raises a clear error instead of running.

| Flag | Gates |
| --- | --- |
| `AI_HUB_GAME_GOALS_ENABLED` | `create_goal` |
| `AI_HUB_GAME_SCHEDULER_ENABLED` | `claim_next_goal` |
| `AI_HUB_GAME_ACTION_DISPATCH_ENABLED` | `execute_game_action` and approval dispatch |
| `AI_HUB_GAME_MEMORY_ENABLED` | `record_memory` |
| `AI_HUB_GAME_RESUME_ENABLED` | `resume_goal_execution` |
| `AI_HUB_GAME_DELEGATION_ENABLED` | `run_delegated_agent` |
| `AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED` | GAME actions linked to reusable `ToolDefinition` records |

Flags gate the service layer only. Direct Django admin model edits (for example the "Add GAME goal" form, or the goal bulk actions) write through the model and bypass the flags. To hard-stop creation in an environment, also remove the relevant admin add/change permissions. See `03_CONFIGURATION.md` for the settings and `12_TROUBLESHOOTING.md` for the disabled-feature error.

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
  "policy": {}
}
```

Keep `max_iterations` low while testing a new agent. Increase only after the session timeline looks correct.

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

Reusable capabilities should be defined as `ToolDefinition` records and grouped through toolboxes/grants on the agent. A `GameActionDefinition` may then link to one of those reusable tools as a governed GAME wrapper. With `AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=true`, GAME validates the workspace action policy, creates a `GameActionRun`, resolves the agent's tool access, executes the linked tool through the generic tool runtime, and records the reusable call in `ToolExecutionRun`.

Keep GAME control actions separate from ordinary tools. Actions such as `finish_goal`, `record_memory`, `update_goal_status`, and `delegate_to_agent` remain GAME control flow and should not be replaced by generic tool records.

If workspace agent or action mappings are configured, they operate as closed allow-lists: entries absent from the configured list are not permitted. With no mappings, legacy sessions retain their previous behavior.

## Scoped memory and continuation

Goal-bound payloads include bounded `scoped_memory` selected from the active workspace, goal, and session. Scope validation rejects memory belonging to another workspace or goal. The existing `memory` list remains for legacy compatibility.

An approval pause creates one pending action, approval request, and continuation. Approval or rejection is stored as an observation before resume, so the next agent iteration receives the human decision and action result. A pending approval cannot be resumed prematurely.

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

Dashboard access requires staff status and the `view_executionsession` permission.

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
