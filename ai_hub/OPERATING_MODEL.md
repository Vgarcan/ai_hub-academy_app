# AI Hub Operating Model

## Purpose

`ai_hub` is a reusable AI operations app for Django projects.

It should be portable across host projects with minimal changes. The app owns the reusable AI platform layer. The host project owns product-specific data, public UI, final business persistence, and domain-specific user experience.

## Ownership Rule

Keep reusable AI primitives inside `ai_hub`.

Examples:

- providers,
- models,
- agents,
- knowledge collections and documents,
- tools,
- contracts,
- Orchestrator pipelines,
- GAME sessions,
- execution sessions,
- step telemetry,
- admin control center,
- guided admin forms.

Keep product-specific behavior outside `ai_hub`.

Examples:

- product database objects,
- public forms and pages,
- final domain records,
- billing,
- product-specific permissions beyond Admin integration,
- product-specific recovery or persistence logic.

## Workspace Model

`ai_hub` has two workspaces:

- Orchestrator: fixed ordered workflows.
- GAME: autonomous goal sessions.

They share providers, models, agents, knowledge, and tools.

They should remain visually separated in Admin because they represent different execution mental models.

## Admin Product Model

Admin is not treated as a raw Django model index.

The intended UX is:

```text
AI Hub Home -> choose workspace or shared resource -> configure -> run -> inspect
```

Admin pages should help non-expert users understand:

- what each object is,
- when to create it,
- what fields mean,
- what example values look like,
- how it connects to the rest of the system,
- where to inspect failures.

The app should keep:

- guided home dashboard,
- workspace pages,
- visual control center,
- section intros,
- guided forms,
- placeholders and examples,
- clear empty states,
- usage badges.

## Integration Model

Host projects should integrate through a small adapter.

Recommended flow:

```text
host object -> host adapter -> ExecutionSession -> ai_hub runtime -> host result persistence
```

Avoid making reusable `ai_hub` services depend directly on host-domain models.

Host adapters may implement domain-specific behavior such as:

- building initial context,
- choosing a pipeline,
- dispatching async work,
- persisting final business objects,
- recovering from domain-specific partial outputs.

## Runtime Rules

- Do not hide model failures.
- Store request and response payloads for auditability.
- Store observation payloads for GAME loops.
- Validate contracts before and after agent calls.
- Keep secrets in environment variables, not database rows.
- Do not rerun sessions that already have step runs.
- Keep GAME max iterations low while testing new agents.
- Prefer small test sessions before expanding workflows.

## Recovery Rule

`ai_hub` should stay generic.

If a host project can recover useful domain output from intermediate drafts after a final model response is malformed or truncated, that recovery belongs in the host adapter.

Example:

```text
ai_hub stores failed final_context -> host adapter hydrates domain payload from drafts -> host persists result
```

This keeps the reusable runner honest while allowing product-specific resilience.

## Documentation Rules

- App-level documentation lives in `ai_hub/_docs/`.
- App-level documentation is written in English.
- Host-specific documentation lives in the host project.
- `ai_hub/README.md` is a quick entry point, not the full manual.
