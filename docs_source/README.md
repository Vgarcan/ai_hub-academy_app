# AI Hub Academy Documentation

This folder (`docs_source/`) holds the **academy host project's own** documentation.

The reusable platform docs are **not duplicated here** — they live in the single
source of truth at [`ai_hub/_docs/`](../ai_hub/_docs/README.md). The
`import_academy_docs` command and Documentation Sync agent publish the platform
docs from `ai_hub/_docs/` plus the Academy docs from this folder, so the site
shows the full set without two copies drifting apart.

## What lives here

- [`15_ACADEMY_FEATURES.md`](15_ACADEMY_FEATURES.md) — academy-specific features: documentation browser, AI assistant, tutorials, lab exercises, management commands, URL reference, and academy troubleshooting.
- This `README.md` — the academy doc index.

## Platform docs (published from `ai_hub/_docs/`)

1. `01_PROJECT_OVERVIEW` — what `ai_hub` is and what it owns
2. `02_INSTALLATION` — install, migrate, admin, static files
3. `03_CONFIGURATION` — providers, models, agents, tools, GAME feature flags
4. `04_CORE_CONCEPTS` — vocabulary
5. `05_ORCHESTRATOR_WORKSPACE` — fixed ordered pipelines
6. `06_GAME_WORKSPACE` — autonomous goal sessions
7. `07_ADMIN_GUIDE` — guided admin and everyday operation
8. `08_RUNTIME_AND_SERVICES` — service layer and execution
9. `09_MODELS_AND_DATABASE` — model reference
10. `10_INTEGRATION_GUIDE` — connecting a host project
11. `11_TESTING_GUIDE` — test commands and strategy
12. `12_TROUBLESHOOTING` — common failures and fixes
13. `13_ROADMAP` — recommended next work
14. `14_CHANGELOG` — notable changes
15. `15_RUNTIME_STATUS` — current, legacy, target and not-implemented paths
16. `16_BUILD_CONSOLE` — guided GAME/Orchestrator creation wizard

## Documentation Rules

- Reusable platform docs are canonical in `ai_hub/_docs/`; edit them there, not here.
- Academy-specific docs live in this folder (numbered `15`+).
- Write documentation in English.
- Do not document private credentials or secrets.
