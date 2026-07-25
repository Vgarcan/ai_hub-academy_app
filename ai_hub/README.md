# AI Hub

`ai_hub` is a reusable, admin-first AI operations layer for Django projects.

It owns the portable AI platform — providers, models, agents, knowledge, tools, contracts, two execution workspaces, and audit/telemetry. The host project owns its domain data, public UI, and final business persistence. See [`OPERATING_MODEL.md`](OPERATING_MODEL.md) for the ownership boundary.

This file is a quick entry point. The full manual lives in [`_docs/`](_docs/README.md).

## Two workspaces, one shared foundation

- **Orchestrator** — fixed, ordered pipelines (run step 1, then 2, then 3).
- **GAME** — autonomous goal sessions (here is a goal; decide what to do next until it is complete or the session stops).

Both reuse the same providers, models, agents, knowledge, and tools.

The GAME subsystem includes durable goals, a deterministic scheduler, an audited action dispatcher with allow-lists, scoped memory, pause/approval/resume, policies and budgets, goal plans, and one-level sub-agent delegation. Each GAME capability is gated by an `AI_HUB_GAME_*_ENABLED` feature flag (fail-closed in the reusable layer).

Runtime migrations are intentionally explicit. See
[`_docs/15_RUNTIME_STATUS.md`](_docs/15_RUNTIME_STATUS.md) before changing Tools,
Knowledge, memory or worker behavior; it separates CURRENT, LEGACY, TARGET and
NOT IMPLEMENTED paths.

## Quick start

```bash
python manage.py migrate ai_hub
python manage.py createsuperuser
python manage.py test ai_hub
```

Then open the guided home:

```text
/admin/ai_hub/
```

Start small: one provider, one model, one narrow agent, one short run. The built-in `training` (stub) provider needs no API key — name its model `training` or `training/...`.

## Documentation

Read in order from [`_docs/`](_docs/README.md):

1. `01_PROJECT_OVERVIEW` — what it is and what it owns
2. `02_INSTALLATION` — install, migrate, admin, static files
3. `03_CONFIGURATION` — providers, models, agents, tools, GAME feature flags
4. `04_CORE_CONCEPTS` — vocabulary
5. `05_ORCHESTRATOR_WORKSPACE` — fixed pipelines
6. `06_GAME_WORKSPACE` — autonomous goals
7. `07_ADMIN_GUIDE` — guided admin and everyday operation
8. `08_RUNTIME_AND_SERVICES` — service layer and execution
9. `09_MODELS_AND_DATABASE` — model reference
10. `10_INTEGRATION_GUIDE` — connecting a host project
11. `11_TESTING_GUIDE` — test commands and strategy
12. `12_TROUBLESHOOTING` — common failures and fixes
13. `13_ROADMAP` — recommended next work
14. `14_CHANGELOG` — notable changes
15. `15_RUNTIME_STATUS` — current, legacy, target, and not-implemented runtime paths
16. `16_BUILD_CONSOLE` — guided GAME/Orchestrator creation wizard

## Integration in one line

```text
host object -> host adapter -> ExecutionSession -> ai_hub runtime -> host result persistence
```

Keep reusable AI primitives in `ai_hub`; keep domain-specific persistence in the host adapter.
