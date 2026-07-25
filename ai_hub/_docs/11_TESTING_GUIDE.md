# Testing Guide

AI Hub should be tested as both a reusable Django app and an operational admin
product.

The goal is not only "the code runs." The goal is:

```text
Users can configure providers, models, agents, knowledge, tools, pipelines and GAME sessions without breaking the host project.
```

## Basic Commands

Run Django checks:

```bash
python manage.py check
```

Run AI Hub tests:

```bash
python manage.py test ai_hub
```

Run AI Hub plus the host integration app:

```bash
python manage.py test your_host_app ai_hub
```

Run the full project suite:

```bash
python manage.py test
```

If the project uses a virtual environment, run these commands through it (for example `.venv/bin/python manage.py test ai_hub` on Unix, or the equivalent interpreter path on Windows).

## Test Layers

AI Hub needs coverage at several layers.

| Layer | What it proves |
| --- | --- |
| Models | Configuration rules are enforced. |
| Services | Runtime behavior is predictable. |
| Contracts | Bad payloads fail clearly. |
| Admin views | Non-technical users can operate the app. |
| Integration adapters | Host projects can use AI Hub safely. |
| Regression tests | Known failures do not return. |

## Model Tests

Test provider and model rules:

- Active model cannot use inactive provider.
- Inactive model cannot be selected for active execution.
- Provider API key fields store environment variable names, not secrets.
- Model tool support is represented clearly.

Test agent rules:

- Agent requires an active model for execution.
- Agent contracts accept valid payloads.
- Agent contracts reject missing required keys.
- Agent list can identify pipeline usage and GAME usage.

Test pipeline rules:

- Active pipeline requires valid steps.
- Step order is deterministic.
- Fallback agent is optional but compatible when present.
- Input and output mappings are valid dictionaries.

## Contract Tests

Contracts are one of the strongest defenses against confusing AI failures.

Test:

- explicit GAME execution and goal outcomes,
- incomplete goals at the iteration cap,
- GAME Hybrid rejection without affecting Orchestrator Hybrid,
- context-tool execution and action-tool blocking in GAME,
- toolbox resolution, explicit grant allow/deny behavior, and workspace tool policy,
- deliberate tool-call rounds with final-answer and tool-observation paths,
- bounded model observations with complete raw tool-run audit,
- default Orchestrator/GAME integration of toolbox-only tools with session/step-linked audit,
- strict direct deliberate responses plus runner-only plain-final compatibility,
- explicit `legacy_preexecute` rollback behavior and invalid runtime rejection,
- exclusion of unresumable approval tools from ordinary runner manifests,
- Control Center tool edges derived from the same resolved manifest,
- unified GAME tool adapter behavior with the kill-switch disabled and enabled,
- default retrieval-only knowledge context with automatic
  browse/search/read/citation tools,
- server-bound Knowledge identity, cross-agent isolation, deny grants and
  workspace policy,
- bounded Knowledge prompt indexes, lexical candidates and
  list/browse/search/read outputs,
- initial chunk creation for curated Build Console documents,
- starter toolbox and starter demo seeds, including idempotency,
- explicit legacy Orchestrator/GAME pre-execution behavior,
- deterministic GAME priority for a fixed time,
- scheduler eligibility and dependency blocking,
- transactional goal claims on a database with row-lock support,
- goal-bound session creation and duplicate-active-session prevention,
- central session-outcome mapping without changing legacy GAME sessions,
- idempotent outcome replay and reconciliation after interrupted finalisation,
- selected-action dispatch, failed-attempt audit, and controlled idempotency states,
- immediate approval pause with exactly one pending continuation,
- approval/rejection observation delivery after resume,
- closed configured action/agent allow-lists and safe external-write defaults,
- scoped-memory workspace/goal/session isolation and runner injection,
- PostgreSQL-only concurrent approval and scheduler claim serialization,
- staff-only access to dashboards containing execution payloads,
- Django 5.2 LTS and current-Django migration/system-check compatibility,

- Missing required keys.
- Invalid primitive types.
- Invalid nested objects.
- Empty payloads.
- Extra keys when strict mode is expected.
- Large payloads that exceed model output assumptions.

Example failing payload:

```json
{
  "summary": "This is missing the required final_output key."
}
```

Expected result:

```text
The step fails with a clear contract validation error.
```

## Orchestrator Tests

Test successful execution:

- One-step pipeline runs to success.
- Multi-step pipeline passes context through mappings.
- Final context contains expected keys.
- Step run telemetry is created.

Test failure behavior:

- `on_error = stop` stops the session.
- `on_error = continue` records the error and continues safely.
- `on_error = fallback` uses the fallback agent.
- Inactive pipelines are rejected.
- Contract failures are visible in session and step errors.

Test admin behavior:

- Orchestrator workspace renders.
- Pipeline graph renders without overflowing.
- Pipeline detail links work.
- Recent runs are visible.

## GAME Tests

Test successful execution:

- Session starts with an entry agent and goal.
- Iterations are recorded as step runs.
- Memory and observations are persisted.
- Finish action marks the session successful.

Test stop behavior:

- Max iterations stop the session.
- Waiting action pauses the session.
- Invalid GAME response fails clearly.
- Strict mode rejects malformed responses.

Example GAME response:

```json
{
  "action": "finish",
  "message": "The goal is complete.",
  "complete": true,
  "final_answer": "Final result."
}
```

Test admin behavior:

- GAME workspace renders.
- Existing sessions appear in the GAME workspace.
- Create GAME session form includes guidance.
- Session detail links work.

## Runtime Tests

Do not hit real providers in unit tests.

Mock the model call service and return predictable data.

Test:

- Provider adapter selection.
- Timeout handling.
- JSON parsing.
- Tool gating.
- Tool manifest redaction.
- Tool execution audit records.
- Legacy eager Knowledge injection limits.
- Retrieval-first prompt-index and tool-result limits.
- Knowledge retrieval scoping by assigned agent collections.
- Forged Knowledge `agent_id` arguments and explicit capability denies.
- Latency recording.
- Error serialization.

Recommended pattern:

```text
Arrange session configuration.
Mock model response.
Run service.
Assert session status, final context and step telemetry.
```

## Admin UX Tests

AI Hub admin is part of the product, not just a debugging surface.

Test these pages:

- `/admin/ai_hub/`
- `/admin/ai_hub/workspaces/orchestrator/`
- `/admin/ai_hub/workspaces/game/`
- `/admin/ai_hub/workspaces/build/` (Build Console wizard; `?kind=game` and `?kind=orchestrator`)
- `/admin/ai_hub/operations/` (Operations Inbox)
- `/admin/ai_hub/pipelinedefinition/control-center/`
- Provider changelist and form.
- Model changelist and form.
- Agent changelist and form.
- Pipeline changelist and form.
- Execution session changelist and form.

Test expectations:

- Pages return HTTP 200 for staff users.
- Anonymous users are redirected.
- Main action links point to valid admin URLs.
- Guided copy is visible.
- Forms include help text for important fields.
- Layout does not depend on a desktop-only viewport.

Control Center / Mission Deck expectations:

- The page includes `admin_control_center.css` and `admin_control_center.js` with
  the current cache-busting query string.
- The graph renders from the embedded `ai-control-graph-data` JSON.
- Control Center and GAME workspace both use the shared graph hooks:
  `data-mc-graph`, `data-mc-stage-inner`, and `data-mc-edges`.
- Hovering graph nodes should not isolate or hide other nodes.
- Selecting a node opens the draggable detail pop-up and exposes an Admin record
  link when the node has a URL.
- `Isolate` is not checked by default.
- Full-screen mode can be entered by button and exited by button or `Esc`.
- Needs attention rows are rendered from `attention_items` with stable ids,
  severity/source metadata, optional timestamps and Admin links.
- The legacy warning list must not be visible.
- Attention filters, sorting, archive/restore and 24-hour silence are client-side
  behaviors backed by `localStorage`.

## Responsive QA

Manual responsive checks should include:

- 390px mobile width.
- 768px tablet width.
- 1366px desktop width.
- Wide desktop.

Check:

- Cards do not overflow.
- Action buttons wrap cleanly.
- Graph controls remain usable.
- Long error text wraps.
- Needs attention action buttons remain compact on small screens.
- Tables remain readable or degrade gracefully.
- The page does not create accidental horizontal scrolling.

## Host Adapter Tests

Every host project should test its adapter.

Example flow:

```text
Host object -> create ExecutionSession -> run AI Hub -> persist host result -> update host UI status.
```

Test:

- Correct initial context is created.
- Correct pipeline or entry agent is selected.
- Successful sessions create the expected host record.
- Failed sessions display useful host errors.
- Partial output recovery is idempotent when implemented.
- Re-running a session does not duplicate host records.

## Regression Tests For Known Failure Modes

Keep tests for bugs that have already happened.

Recommended regression cases:

- Malformed final JSON does not crash the whole host flow.
- Recoverable partial outputs can be converted into host records once.
- Success sessions do not keep polling forever.
- Static admin assets load under `DEBUG=False`.
- Uploaded media does not break static paths.
- GAME sessions are visible from the GAME workspace.
- Agent usage labels distinguish pipeline and GAME usage.

## Production Smoke Test

After deployment:

1. Open `/admin/ai_hub/`.
2. Open Orchestrator workspace.
3. Open GAME workspace.
4. Open Control Center.
5. Create or inspect one provider.
6. Create or inspect one model.
7. Create or inspect one agent.
8. Run a tiny Orchestrator session.
9. Run a tiny GAME session.
10. Confirm execution sessions and step runs are visible.
11. Confirm host-specific output is stored by the host app.

## GAME Feature Flag Tests

Each GAME subsystem has an env-var kill switch. Test that:

- `create_goal` raises when `AI_HUB_GAME_GOALS_ENABLED=False`.
- `claim_next_goal` raises when `AI_HUB_GAME_SCHEDULER_ENABLED=False`.
- `execute_game_action` raises when `AI_HUB_GAME_ACTION_DISPATCH_ENABLED=False`.
- `record_memory` raises when `AI_HUB_GAME_MEMORY_ENABLED=False`.
- `resume_goal_execution` raises when `AI_HUB_GAME_RESUME_ENABLED=False`.
- `run_delegated_agent` raises when `AI_HUB_GAME_DELEGATION_ENABLED=False`.
- All functions behave normally when flags are True.

Use `@override_settings(FLAG_NAME=False)` in tests. The reusable app fallback is fail-closed. `_core` enables GAME by default only in DEBUG development; production requires explicit environment opt-in for each flag.

Feature tests cover whole boundaries rather than only one public function: goal-bound session creation/execution, memory read and write, action dispatch/approval, scheduler claims, resume and delegated child policy.

## Release Gates

Before enabling a GAME phase in a non-development environment, confirm all of these:

```text
Migrations apply cleanly against the target database.
Focused phase tests are green.
Full ai_hub suite is green (python manage.py test ai_hub).
Admin screens load for a staff user.
Legacy GAME sessions continue to execute.
A manual vertical-slice test succeeds (create workspace → goal → session → finish).
Logs show no unhandled exceptions during the smoke test.
Rollback path is confirmed: the feature flag can be set to False without a deploy.
Delegated policy and budget concurrency are validated on PostgreSQL.
Admin mutation-bypass and payload-redaction regressions are green.
```

## Test Data Rules

- Use small prompts and small payloads.
- Do not require real API keys for unit tests.
- Do not depend on local Ollama models in CI.
- Run the concurrent GAME claim test against PostgreSQL; SQLite intentionally skips it because it cannot prove row-lock semantics.
- Use factories or fixtures for provider/model/agent setup.
- Keep host-specific fixtures outside AI Hub.

The bundled CI has separate SQLite and PostgreSQL 16 jobs. The PostgreSQL job
runs the complete suite, including concurrent scheduler claim, approval review
and delegation-budget reservation. A local PostgreSQL 16.14 run completed all
377 tests without skips during the stabilization audit.

## Before Shipping

Run:

```bash
python manage.py check
python manage.py test ai_hub
python manage.py test
```

If the full suite is slow, at minimum run:

```bash
python manage.py check
python manage.py test ai_hub your_host_app
```
