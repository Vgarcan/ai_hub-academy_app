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

**Before you configure Agents, Knowledge collections or GAME workspaces, the host
project must have an `ApplicationScope`.** Every one of them belongs to exactly
one scope, and Core never creates or guesses a scope for you. Create yours in
the Admin (`/admin/ai_hub/applicationscope/add/`) or from your own host setup
command; see *Create An Application Scope* in
[`_docs/02_INSTALLATION.md`](_docs/02_INSTALLATION.md#create-an-application-scope).
(`bootstrap_academy_scope` belongs to the bundled Academy host and provisions
only Academy's own scope; it is not a generic Core installer.) AI Hub is
installed as a Django app inside the host project, as described there.

For retrieval, embeddings and the optional hybrid Knowledge search, start at
[`_docs/RAG_RETRIEVAL_STATUS.md`](_docs/RAG_RETRIEVAL_STATUS.md).

Start small: one provider, one model, one narrow agent, one short run. The built-in `training` (stub) provider needs no API key — name its model `training` or `training/...`.

## Documentation

The canonical documentation index, with the reading order of every document
(including the RAG references), is [`_docs/README.md`](_docs/README.md). Start with
`01_PROJECT_OVERVIEW` and `02_INSTALLATION`, and read
[`15_RUNTIME_STATUS`](_docs/15_RUNTIME_STATUS.md) before changing Tools, Knowledge,
memory, prompt authority or worker behavior.

## Integration in one line

```text
host object -> host adapter -> ExecutionSession -> ai_hub runtime -> host result persistence
```

Keep reusable AI primitives in `ai_hub`; keep domain-specific persistence in the host adapter.
