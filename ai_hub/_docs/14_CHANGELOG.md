# Changelog

This changelog records app-level changes for AI Hub documentation and reusable
platform behavior.

Host-project features should be documented in the host project, not here.

## Unreleased

### Added

#### Tool platform foundation (Tools implementation phases 01-07)

- Expanded `ToolDefinition` metadata with labels, descriptions, risk level,
  operation mode, approval requirement, and system-tool marker.
- Added toolboxes, toolbox-tool entries, agent toolbox assignments, explicit
  agent tool grants, and `ToolExecutionRun` audit records.
- Added `resolve_agent_tools()` for toolbox/grant/legacy/workspace-policy
  resolution with redacted model-facing manifests.
- Added deliberate agent tool execution with bounded tool rounds and reusable
  `execute_tool()` dispatch.
- Added optional GAME action-to-tool adapter behind
  `AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED`.
- Added `KnowledgeDocumentChunk` plus read-only browse, search, read and cite
  retrieval services under agent collection scope.
- Added starter toolbox and starter demo management commands:
  `seed_ai_hub_starter_toolboxes` and `seed_ai_hub_starter_demo`.

#### GAME domain models (Phase 03)

- `GameWorkspace` model — owns a set of goals, holds default runtime config and
  workspace-level policy.
- `GameGoal` model — stores goal title, description, success criteria, context,
  result, status, priority, and due date.  Status transitions are locked through
  service functions; completed and cancelled goals require explicit reopening.
- `GameGoalDependency` model — declares required or optional ordering between
  goals within a workspace.  Self-links, cross-workspace links, duplicates, and
  cycles are rejected at the service/admin layer.
- Migration `0003_gameworkspace_gamegoal_gamegoaldependency_and_more` — additive;
  introduces the three GAME domain tables plus supporting indexes.
- Admin registration for all GAME domain models with workspace, lifecycle,
  due-date, and priority filters.

#### Deterministic scheduler (Phase 04)

- `game_priority` service — deterministic scoring using base priority, time-to-
  deadline, queue age, and dependent-unlock bonus; model validators bound base
  and calculated values to prevent decimal overflow.
- `game_scheduler` service — read-only `get_next_eligible_goal()` and
  transactional `claim_next_goal()`; claim persists candidate scores and marks
  one goal running atomically via `SELECT FOR UPDATE`.
- Stable tie-breaking on score, due date, creation time, and primary key; no LLM
  participates in eligibility or selection.
- `queued_at` timestamp resets on every retry/reopen so queue-age bonus reflects
  actual wait time, not original creation time.

#### Goal-bound sessions and outcomes (Phase 05)

- `ExecutionSession.goal` nullable FK — links a runtime record to a durable GAME
  goal while preserving all legacy null behaviour.
- `create_goal_execution_session()` service — locks the goal, prevents duplicate
  active sessions via a conditional database constraint, and transitions queued
  goals to running atomically.
- Central outcome mapping in `game_goal_outcomes` — maps session finish reasons
  to completed, partial, waiting-information, waiting-approval, or failed goal
  outcomes.
- Session outcome fingerprinting — replaying the same historical outcome is
  idempotent; the fingerprint detects divergent replays.
- `reconcile_goal_outcomes()` repair function — detects and heals interrupted
  finalisation where a session reached a terminal state but its goal was not
  updated.
- Migration `0004_executionsession_goal` — nullable FK; additive.

#### Audit hardening (Phase 05 post-audit)

- Migration `0005` — adds queue timestamps, transition metadata column,
  outcome-fingerprint column, priority bounds check constraint, and a conditional
  unique index allowing at most one pending/running/waiting session per goal.
- `transition_metadata` field on `GameGoal` — stores internal lifecycle events
  separately from `context` so lifecycle metadata never reaches the agent.

#### Feature flags and release gates (Phase 12)

- `game_feature_flags` service — `require_game_feature(flag_name)` raises
  `ValidationError` when the named Django setting is explicitly `False`; defaults
  to `True` when the setting is absent.
- Six GAME kill-switch flags in `_core/settings.py`, each overrideable via env var
  and defaulting to `True` (all phases tested):
  - `AI_HUB_GAME_GOALS_ENABLED` — gates `create_goal`.
  - `AI_HUB_GAME_SCHEDULER_ENABLED` — gates `claim_next_goal`.
  - `AI_HUB_GAME_ACTION_DISPATCH_ENABLED` — gates `execute_game_action`.
  - `AI_HUB_GAME_MEMORY_ENABLED` — gates `record_memory`.
  - `AI_HUB_GAME_RESUME_ENABLED` — gates `resume_goal_execution`.
  - `AI_HUB_GAME_DELEGATION_ENABLED` — gates `run_delegated_agent`.
- `.one-env.example` updated with a commented GAME flags section.
- `11_TESTING_GUIDE.md` updated with feature-flag test patterns and release-gate
  checklist.
- No new migrations — Phase 12 adds no model changes.

#### Admin and operational UX (Phase 11)

- `game_operational_ux` service — three pure query functions used by the admin:
  - `build_workspace_dashboard_context(workspace)` — aggregates goal status counts,
    top eligible goals with scheduler explanations, pending approvals, blocked goals,
    recent sessions, recent action runs, budget, policy, and workspace agent/action
    mappings scoped to one workspace.
  - `build_goal_detail_context(goal)` — aggregates session history, action runs,
    memory entries, the attached plan, an `is_resumable` flag, and a scheduler
    explanation scoped to one goal.
  - `build_scheduler_explanation(goal)` — returns `{base_priority, bonuses, total}`
    as a human-readable breakdown of how the scheduler scores a goal.
  - `redact_payload(payload)` — recursively replaces known-sensitive keys
    (`api_key`, `secret`, `password`, `token`, `access_token`, `refresh_token`,
    `authorization`, `credentials`, `private_key`) with `"***REDACTED***"` in
    arbitrarily nested dicts and lists.
- Workspace changelist: **Dashboard** link per row opens an operational view at
  `<workspace_id>/dashboard/`; requires `view_executionsession` permission.
- Goal change form enriched with scheduler explanation, resume indicator, session
  history, action runs, goal plan, and memory entry panels rendered below the
  standard field sets.
- `GameActionApprovalRequestAdmin`: `approve_selected_actions` and
  `reject_selected_actions` bulk actions call `approve_action_run()` and
  `reject_action_run()` from the service layer; both actions are hidden from and
  blocked for users without `ai_hub.approve_game_action`.
- `GameGoalAdmin` bulk actions: `queue_selected_goals`, `cancel_selected_goals`,
  `reopen_selected_goals`, and `resume_selected_goals` for lifecycle and resume
  operations directly from the changelist.
- No new migrations — Phase 11 adds no model changes.

#### Plans and multi-agent delegation (Phase 10)

- `GameGoalPlan` model (4.15) — optional structured execution aid linked one-to-one
  with a goal; tracks version, summary, and active/completed/abandoned status.
- `GameGoalPlanStep` model (4.16) — ordered steps within a plan with optional
  same-plan dependency chain; self-links, cross-plan dependencies, and
  out-of-order dependencies are rejected.  `ai_hub_unique_plan_step_order`
  constraint enforces uniqueness within a plan.
- `GameDelegationRun` model (4.17) — durable record for every sub-agent
  delegation: links the parent action run, parent goal, target agent, and
  delegated session; stores task, expected result, result summary, lifecycle
  status, and timing.
- `game_plans` service — `create_plan()` and `add_plan_step()` with full model
  validation and cross-plan dependency guards.
- `game_delegation` service — `run_delegated_agent()` validates delegation depth,
  workspace agent allow-list, and sub-agent budget before creating a delegation
  run and running a narrowed goal-less session synchronously.
- `game_policy` additions — `validate_agent_for_workspace()` (closed allow-list
  per workspace), `check_delegation_depth()` (maximum depth 1), and
  `check_sub_agent_budget()` (`max_sub_agent_runs_per_goal` budget key).
- Dispatcher extended with `_handle_delegate_to_agent` under `ActionType.SUB_AGENT`.
- Delegated sessions are created without a goal FK to avoid the active-goal
  unique constraint; the parent-goal relationship is tracked exclusively through
  `GameDelegationRun.parent_goal`.
- Migration `0012_phase10_plans_delegation` — additive; creates the three new
  tables plus `aihub_delegation_goal_stat_idx` covering index.
- Admin: `GameGoalPlanAdmin` (with `GameGoalPlanStepInline`), `GameGoalPlanStepAdmin`,
  and `GameDelegationRunAdmin` (all fields read-only) registered.

#### GAME actions, memory, continuation, and policy (Phases 06–09)

- Added the selected-action dispatcher and durable action attempts with contracts,
  telemetry, output observations, and stable idempotency keys.
- Added scoped workspace/goal/session/action-result memory with bounded runner
  context and deterministic compaction.
- Added durable continuation and approval requests, permission-controlled review,
  rejection observations, and next-order resume.
- Added workspace action/agent mappings plus deterministic iteration, action, and
  runtime budgets. External writes are closed by default.
- Migration `0011_stabilize_phases_06_09` aligns action reverse relations, bounds
  memory importance, normalises legacy scope data, and enforces one pending
  continuation per session plus valid memory field shapes.
- Post-audit stabilization makes approval stop the iteration loop immediately,
  rechecks agent policy at creation and execution, persists known rejected action
  attempts, serialises concurrent reviews, and feeds approval/rejection results to
  the resumed parent agent.

#### Security hardening

- Python tools now require membership in `AI_HUB_ALLOWED_TOOL_CALLABLES`; no
  arbitrary dotted-path callable can be imported from a database record.
- HTTP tools now require an explicit `allowed_hosts` entry, must use GET or HEAD
  for context-tool classification, and have timeouts clamped to 1–60 seconds.
- Dashboard views require staff status plus `view_executionsession` permission.
- Execution endpoint requires `change_executionsession` in addition to staff
  status.
- Workspace admin pages additionally require `view_executionsession`.
- Anonymous assistant execution is disabled by default; questions are bounded at
  4 000 characters; anonymous access is an explicit opt-in that must sit behind
  external rate limiting.
- Production mode refuses to start without a real `SECRET_KEY`; secure redirect,
  session, and CSRF cookies default on when `DEBUG=False`.
- `setup_dev.py` generates the initial admin password unless
  `DJANGO_SUPERUSER_PASSWORD` is explicitly supplied.

#### Documentation (Phase 00–01)

- Full English documentation set for AI Hub as a reusable Django app.
- App-level `README.md` with current state, workspaces, admin entry points,
  runtime model and host integration rules.
- `_docs/README.md` documentation index with reading order and ownership rules.
- `OPERATING_MODEL.md` defining the reusable-app boundary.
- Project overview for non-domain-specific AI operations.
- Installation guide covering app setup, URLs, migrations, static files,
  provider setup, model setup and smoke tests.
- Core concepts guide for providers, models, agents, knowledge, tools,
  contracts, mappings, Orchestrator, GAME, sessions and host adapters.
- Configuration guide with practical setup examples for providers, models,
  agents, knowledge and tools.
- Orchestrator workspace guide for fixed multi-agent workflows.
- GAME workspace guide for autonomous goal sessions.
- Admin guide for AI Hub Home, Control Center, workspaces, guided forms and
  changelists.
- Runtime and services guide covering service responsibilities, contracts,
  execution sessions, step runs, final output handling and worker expectations.
- Models and database guide defining reusable AI Hub tables and host adapter
  boundaries.
- Integration guide explaining how host projects should create sessions and
  persist domain-specific results.
- Testing guide covering models, contracts, Orchestrator, GAME, admin UX,
  responsive QA, host adapters and known regression cases.
- Troubleshooting guide for admin styling, static files, provider/model issues,
  pipeline and GAME issues, invalid final JSON, polling, tools and warnings.
- Roadmap for admin UX maturity, adapter API, workers, tools, knowledge/RAG,
  GAME readiness, import/export, observability and structured output safety.

### Changed

#### Runtime behaviour

- GAME final context now distinguishes execution outcome, goal outcome, and finish
  reason as three separate fields; previously all were collapsed into one.
- GAME Hybrid mode creation and execution are now rejected until resumable
  continuation is implemented; attempts return a clear error rather than running
  in an undefined state.
- GAME auto-executes only tools with an explicit `context_tool` classification;
  tools in missing or action categories are blocked by default.  Orchestrator
  retains its legacy broad-tool behaviour.
- Host doc-sync tooling may use a temporary explicit legacy-action opt-in
  (`allow_legacy_game_action_tools=True`) until a proper dispatcher definition
  exists. (Doc-sync is a host-project command, not part of reusable `ai_hub`.)
- Runtime audit records (lifecycle status, final context, outcome fingerprints,
  timing, and step audit fields) are now read-only in Admin.  Completed sessions
  also lock their runtime inputs.
- Python context tools additionally require `read_only=True`; HTTP context tools
  additionally require GET or HEAD; both constraints close a prior classification
  loophole.

#### Compatibility and environment

- Minimum Python raised from 3.10 to **3.12**.
- Supported Django range narrowed to **>=5.2,<7.0** (5.2 LTS through 6.x);
  Django 5.0 was removed because it fails internally under Python 3.14.
- Setup and documentation now consistently reference `.one-env.example` (removed
  stale `.env.example` references).
- Nonexistent project-level static source path removed; Django system checks are
  clean.

#### Documentation

- Documentation now treats AI Hub as the single reusable AI app.
- Documentation uses `ai_hub` for the Python/Django package and `AI Hub` for the
  visible product name.
- Documentation describes Orchestrator and GAME as two visual workspaces sharing
  the same technical foundation.
- Documentation clarifies that host-specific persistence belongs in the host app.
- Documentation frames the Django Admin as the primary product surface for
  configuration and operations.
- Documentation explains that tools are attached to agents and gated by runtime
  support, model compatibility and safety rules.
- Documentation explains that malformed structured output is an operational case
  to test, monitor and recover from when possible.

### Fixed

#### Post-Phase-12 stabilization

- Delegated sessions now recover workspace policy from their durable delegation record, use strict least-privilege read actions, and cannot execute approval-gated child actions.
- Self-delegation is denied by default; delegation budget reservation is serialized under the parent-goal lock.
- Plan step ordering/cycles are validated from every model/Admin path; plans gain revision snapshots and database constraints for versions, orders and self-links.
- Approval, continuation and delegation audit records are immutable in Admin. Dedicated CSRF-protected approval/rejection POST controls preserve review notes and call service functions.
- Aggregated Admin views enforce related-model permissions; session, step, action, continuation and approval payloads share recursive redaction.
- The execution timeline combines steps, action runs, continuations and approvals. Workspace/goal views expose usage counters, concrete blockers and resolvability based on a valid pending continuation.
- GAME feature fallbacks are fail-closed; development enables them through DEBUG while production requires explicit opt-in. Goal execution and memory retrieval are gated in addition to creation/write paths.
- Migration `0013_stabilize_phases_10_12` adds plan revision history and normalises legacy data before plan/delegation constraints.

#### Post-implementation audit (Phases 11–12)

- `redact_payload` is now actually applied: `GameActionRunAdmin` renders
  `input_payload`, `output_payload`, and `observation_payload` through a redacted
  display so known-sensitive keys are masked instead of shown raw.
- Goal change-form session cards rendered an empty title because of a broken
  `|add` filter chain; the label now falls back to `Session #<pk>` correctly.
- `require_game_feature` raises `ValueError` on an unknown flag name so a typo can
  no longer silently leave a feature ungated.
- `approve_action_run` now checks `AI_HUB_GAME_ACTION_DISPATCH_ENABLED` before
  approving, closing a gap where an approved action could dispatch while the
  action kill-switch was off.
- Documented that GAME feature flags gate the service layer only; direct Django
  admin model edits (e.g. "Add GAME goal", goal bulk transitions) bypass them.
- Workspace dashboard performance: `status_counts` now uses a single aggregate
  query instead of one count per status, and `top_eligible` scores each goal once
  (reusing the explanation for ranking and display) instead of computing priority
  twice per goal.
- Workspace dashboard now renders the `recent_action_runs` panel that was
  previously computed but never displayed.
- `build_goal_detail_context` catches `ObjectDoesNotExist` for a missing goal plan
  instead of a bare `except Exception`.

#### GAME execution-test report follow-up

- `record_memory()` now coerces `importance_score` to `Decimal` via `Decimal(str(value))`,
  so float callers (e.g. `0.9`) no longer hit the DecimalField 2-decimal-place
  validator.
- `ModelConfig.clean()` enforces the training-stub naming convention: a
  Training-provider model must be named `training` or start with `training/`,
  turning a previously silent runtime fall-through into a clear config-time error.
  Admin help text documents the convention.
- `ModelConfig.temperature_default` default changed from the float `0.70` to
  `Decimal("0.70")`; the float default failed `full_clean()` on any instance that
  did not set temperature explicitly. Migration `0014_alter_modelconfig_temperature_default`
  (metadata-only, no data change).
- Orphaned RUNNING goal cleanup: `find_orphaned_running_goals()` /
  `cancel_orphaned_running_goals()` in `game_goals`, plus the
  `cleanup_orphaned_goals` management command (supports `--workspace`,
  `--older-than-hours`, `--dry-run`). A goal stuck in RUNNING with no active
  session (interrupted runs, integration-suite stubs) is cancelled; goals with an
  active session are never touched.

### Removed

- References to legacy AI app switching as a supported architecture.
- Any recommendation to keep domain-specific run models inside AI Hub.
- Any assumption that AI Hub belongs to only one host-project domain.

### Notes

- AI Hub documentation is intentionally written in English.
- Host-specific workflows, UI copy and persistence rules should be documented
  outside `ai_hub`.
- The current recommended product model is:

```text
Two workspaces, one shared AI foundation.
```

- Supported runtime baseline after Phases 00–05 audit:

```text
Python  >= 3.12
Django  >= 5.2 LTS, < 7.0
SQLite  — local development and single-worker behaviour only
PostgreSQL — required before running multiple scheduler workers concurrently
```

- GAME domain models (`GameWorkspace`, `GameGoal`, `GameGoalDependency`) and the
  scheduler are available once migrations `0001`–`0005` are applied.  No Phase 06
  features (action dispatcher, `GameActionDefinition`, `GameActionRun`) are
  present in this release.
- `reconcile_goal_outcomes()` exists as a service but is not yet a scheduled job.
  Operators should call it manually or via a management command before running
  concurrent background workers.
