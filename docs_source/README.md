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

The reusable platform documentation is indexed once, in
[`ai_hub/_docs/README.md`](../ai_hub/_docs/README.md). This page does not repeat
that list, so it cannot drift from it.

## Documentation Rules

- Reusable platform docs are canonical in `ai_hub/_docs/`; edit them there, not here.
- Academy-specific docs live in this folder (numbered `15`+).
- Write documentation in English.
- Do not document private credentials or secrets.
