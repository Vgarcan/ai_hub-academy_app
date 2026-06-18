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

- `ExecutionSession`
- `ExecutionStepRun`
- `AgentProfile`

A GAME session usually uses one entry agent.

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
  "policy": {}
}
```

Keep `max_iterations` low while testing a new agent. Increase only after the session timeline looks correct.

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

## Recommendation For v1

Do not put GAME inside Orchestrator pipelines for v1.

Keep them visually and operationally separate:

- Pipeline: known steps.
- GAME: goal, decision, memory, and stopping.

Both can reuse the same agents, tools, and knowledge.
