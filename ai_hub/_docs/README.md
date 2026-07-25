# AI Hub Documentation

This folder contains the app-level documentation for `ai_hub`.

The docs are written from the point of view of `ai_hub` as a reusable Django app. Host-specific workflows should be documented in the host project, unless they illustrate a generic integration pattern.

## Reading Order

1. [`01_PROJECT_OVERVIEW.md`](01_PROJECT_OVERVIEW.md) - what `ai_hub` is, what it owns, and how it should be reused.
2. [`02_INSTALLATION.md`](02_INSTALLATION.md) - app installation, URLs, migrations, admin access, and static files.
3. [`03_CONFIGURATION.md`](03_CONFIGURATION.md) - providers, models, agents, knowledge, tools, and first setup.
4. [`04_CORE_CONCEPTS.md`](04_CORE_CONCEPTS.md) - the vocabulary used throughout the app.
5. [`05_ORCHESTRATOR_WORKSPACE.md`](05_ORCHESTRATOR_WORKSPACE.md) - fixed ordered workflows.
6. [`06_GAME_WORKSPACE.md`](06_GAME_WORKSPACE.md) - autonomous goal sessions.
7. [`07_ADMIN_GUIDE.md`](07_ADMIN_GUIDE.md) - guided Admin UX, forms, control center, and everyday operation.
8. [`08_RUNTIME_AND_SERVICES.md`](08_RUNTIME_AND_SERVICES.md) - service layer and execution behavior.
9. [`09_MODELS_AND_DATABASE.md`](09_MODELS_AND_DATABASE.md) - model reference and database relationships.
10. [`10_INTEGRATION_GUIDE.md`](10_INTEGRATION_GUIDE.md) - how a host project connects to `ai_hub`.
11. [`11_TESTING_GUIDE.md`](11_TESTING_GUIDE.md) - recommended test commands and coverage strategy.
12. [`12_TROUBLESHOOTING.md`](12_TROUBLESHOOTING.md) - common failures and fixes.
13. [`13_ROADMAP.md`](13_ROADMAP.md) - recommended next product and engineering work.
14. [`14_CHANGELOG.md`](14_CHANGELOG.md) - notable app and documentation changes.
15. [`15_RUNTIME_STATUS.md`](15_RUNTIME_STATUS.md) - canonical CURRENT / LEGACY / TARGET / NOT IMPLEMENTED runtime status.
16. [`16_BUILD_CONSOLE.md`](16_BUILD_CONSOLE.md) - Build Console wizard: step-by-step guide for the guided GAME and Orchestrator creation wizard.

## Current Product Shape

`ai_hub` currently provides:

- a **cockpit app home** at `/admin/ai_hub/` organized into five areas (Overview & Entry, Foundation, Orchestrator, GAME, Operations), with vitals, a "needs your attention" queue, recent activity and a setup checklist,
- a **Build Console wizard** at `/admin/ai_hub/workspaces/build/` for guided GAME session and Orchestrator pipeline creation in one atomic transaction,
- an **Operations Inbox** at `/admin/ai_hub/operations/` — one cross-workspace queue for pending approvals (inline approve/reject), sessions waiting for information, failed sessions and blocked goals,
- a visual Orchestrator workspace for fixed ordered pipelines,
- a full GAME subsystem: durable goals, deterministic scheduling, action dispatch, scoped memory, pause/approval/resume, policies and budgets, plans, and sub-agent delegation,
- per-capability GAME feature flags (fail-closed in the reusable layer),
- operational GAME dashboards, goal-detail enrichment, and lifecycle/approval bulk actions,
- a **Control Center** with a shared graph engine, full-screen graph
  mode, draggable node details and a non-destructive attention inbox,
- **composed change pages** (tabbed Overview + Configuration) for the root entities and Foundation hubs,
- guided forms with examples and placeholders, with supporting/bridge tables demoted from the index behind a "Show supporting tables" toggle,
- reusable provider/model/agent/tool/knowledge configuration,
- generic execution sessions and step telemetry,
- staff-only execution endpoints.

## Documentation Rules

- Write app documentation in English.
- Keep reusable platform documentation in `ai_hub/_docs/`.
- Keep host-product instructions outside `ai_hub` unless they explain a generic adapter pattern.
- Do not document private credentials or secrets.
- Prefer examples that can be copied into another Django project.
