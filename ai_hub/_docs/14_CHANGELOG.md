# Changelog

This changelog records app-level changes for AI Hub documentation and reusable
platform behavior.

Host-project features should be documented in the host project, not here.

## Unreleased

### Added

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
