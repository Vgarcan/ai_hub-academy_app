# Academy Features

This document covers the academy-specific features of AI Hub Academy — the host project that ships with `ai_hub`.

These features live in the `academy` Django app and use `ai_hub` for all AI execution. They are not part of the reusable `ai_hub` platform.

---

## Features Overview

| Feature | URL | Description |
| --- | --- | --- |
| Home / Landing | `/` | Academy home page |
| Documentation browser | `/docs/` | Browse and search imported Markdown docs |
| Documentation assistant | `/assistant/` | AI chatbot powered by semantic search |
| Tutorial list | `/tutorials/` | All available tutorial modules |
| Tutorial module | `/tutorials/<module>/` | Missions inside a module |
| Tutorial mission | `/tutorials/<module>/<mission>/` | Interactive step with Admin validation |
| Lab exercise | `/tutorials/<module>/lab/<exercise>/` | AI-evaluated coding or concept exercise |
| Progress dashboard | `/progress/` | Per-user mission completion overview |

---

## Documentation Browser

Imports the reusable Markdown files from `ai_hub/_docs/` plus the
Academy-specific files from `docs_source/`, and stores them as
`DocumentationPage` and `DocumentationChunk` records.

**Key models:**

- `DocumentationSource` — groups pages by source directory or topic.
- `DocumentationPage` — one imported Markdown file. Has a `slug` used for the URL.
- `DocumentationChunk` — a section of a page (heading + body). It can store an
  optional embedding vector for semantic search; the command default is
  `bge-m3:latest`.

**Management commands:**

```bash
# Synchronize both configured Markdown roots
python manage.py import_academy_docs

# Generate semantic embeddings (requires Ollama with bge-m3:latest)
python manage.py embed_docs --model bge-m3:latest

# Force re-embed even for chunks that already have an embedding
python manage.py embed_docs --force
```

**Search:** `/docs/search/` supports both keyword and semantic search depending on whether embeddings exist.

---

## Documentation Assistant (Chat)

An AI chatbot at `/assistant/` that answers questions about the imported documentation.

Submitting AI questions requires authentication by default, and questions are limited to 4,000 characters. Anonymous execution can be enabled with `ACADEMY_ASSISTANT_ALLOW_ANONYMOUS=True` only when deployment-level rate limiting and provider-cost controls are in place.

**How it works:**

1. User submits a question via `/assistant/ask/`.
2. The service embeds the question using Ollama `bge-m3:latest` and runs a semantic search over `DocumentationChunk` records.
3. If embeddings are missing or Ollama is unreachable, it falls back to keyword filtering on `search_text` and `heading`.
4. When the configured `AI Hub Documentation Assistant` is available, retrieved
   chunks are passed to a GAME `ExecutionSession`.
5. The generated answer is stored in `DocumentationChatMessage` alongside the
   retrieved chunks. If the agent is absent or execution fails, the message uses
   a bounded extracted-text fallback and is not presented as a model-generated
   run.

**Key models:**

- `DocumentationChatSession` — one conversation thread per user.
- `DocumentationChatMessage` — one message (user or AI). AI messages have a `retrieved_chunks` M2M relation to the chunks that informed the answer.

**Key services:**

- `academy.services.documentation_assistant.answer_documentation_question()` — orchestrates the full search + agent call.
- `academy.services.documentation_search.search_documentation()` — semantic or keyword search over chunks.
- `academy.services.embeddings.get_embedding()` — calls Ollama `/api/embed` to generate a vector.

**Agent:** `seed_academy_training_data` creates a deterministic Training version
of `AI Hub Documentation Assistant`. Running `seed_ollama_agents` upgrades that
seed-owned Training agent to Ollama and attaches the `search_documentation`
context tool; existing custom non-Training agents are preserved unless
`--force-update` is used. The assistant uses GAME with the default cap of three
iterations.

---

## Safe Markdown Rendering

Documentation pages, documentation search results, tutorial instructions and
assistant answers are all rendered through one template filter,
`academy.templatetags.markdownify.markdownify`. Its input is not trusted:
Python-Markdown passes raw HTML through, and assistant answers are model output.

The filter converts Markdown to HTML and then sanitizes it with
[`nh3`](https://pypi.org/project/nh3/) (Rust `ammonia` bindings, pinned to
`>=0.3,<0.4` in `requirements.txt`) before marking it safe. Everything outside
an explicit allow-list is removed:

| Allowed | Notes |
| --- | --- |
| `p br hr h1–h6 ul ol li strong em b i code pre blockquote` | Ordinary prose, lists, quotes and code. |
| `table thead tbody tr th td` | Table alignment (`text-align: left\|right\|center`) is the only permitted style. |
| `a` with `href`, `title` | Schemes `http`, `https`, `mailto` and relative URLs only. Links get `rel="noopener noreferrer"`. |
| `img` with `src`, `alt`, `title` | `src` must be a local path starting with a single `/`. Remote images are dropped. |
| `id` on `h1–h6` | Anchors generated by the `toc` extension, limited to `[A-Za-z0-9_-]`. |
| `class="language-…"` on `code` | Syntax-highlight hint from fenced code blocks. |

`<script>` and `<style>` are removed together with their content. Event-handler
attributes, `javascript:`/`data:`/`vbscript:` URLs (including encoded
variants), SVG, MathML, iframes, forms and other active content do not survive.
HTML-like text inside inline or fenced code is escaped and shown as text.

Every Markdown file in both documentation roots renders with the same element
structure before and after sanitizing, apart from the added link `rel`.
`academy/test_markdown_rendering.py` enforces this together with the hostile
cases above.

**What this does and does not cover.** It prevents a browser from executing
HTML that arrives through these pages. It does not protect an LLM from
instructions hidden in retrieved text (prompt injection), which is a separate
concern handled at the retrieval layer and not by this filter. A page reader can
still be shown misleading *text*, and a heading `id` could in principle collide
with an `id` used by the page's own scripts.

**Change record (2026-11-01).** HOST-SEC-2: `markdownify` previously marked
unsanitized Python-Markdown output safe. It now sanitizes with the allow-list
above. No other behavior of the documentation browser or assistant changed.

---

## Tutorials

Interactive step-by-step missions that validate real Django Admin state. Each mission checks whether the user has performed a specific configuration action (e.g. created a provider, run a GAME session).

**Key models:**

- `TutorialModule` — a group of related missions (e.g. "Providers", "GAME Workspace").
- `TutorialMission` — one step with a title, instructions, and a validator key.
- `UserMissionProgress` — tracks whether a user has completed each mission.

**Validation:** Each mission has a `validator_key` that maps to a Python function in `academy.services.mission_validators`. The `/tutorials/<module>/<mission>/check/` endpoint calls the validator and marks the mission complete if it passes.

**Management command:**

```bash
# Seed all tutorial modules and missions
python manage.py seed_academy_training_data
```

---

## Lab Exercises

Coding and concept exercises attached to tutorial modules. Users submit free-text answers that are evaluated by an AI agent using a rubric.

**Key models:**

- `LabExercise` — one exercise with a prompt, rubric, and expected answer outline.
- `LabAttempt` — one user submission. Stores the raw answer, categorical score
  (`pass`, `partial`, `fail` or `pending`) and AI feedback.

**AI evaluation:** `academy.services.lab_evaluator.evaluate_lab_answer()` creates an `ExecutionSession` with the `Lab Exercise Evaluator` agent. The agent returns JSON with `score`, `feedback`, and `follow_up_question`. If no agent is configured, the attempt is saved with a `pending` score.

> **Known limitation.** AI scoring cannot currently be enabled through a documented
> path. `seed_lab_exercises --create-evaluator` fails to create the evaluator agent
> because it is created without an ApplicationScope
> ([issue #26](https://github.com/Vgarcan/ai_hub-academy_app/issues/26)), and the
> Admin form for agents does not yet offer the scope field
> ([issue #25](https://github.com/Vgarcan/ai_hub-academy_app/issues/25)). Until
> they are fixed, lab attempts are saved with a `pending` score; the exercises
> themselves work without an evaluator.

**Management command:**

```bash
# Seed lab exercises for all tutorial modules
python manage.py seed_lab_exercises
```

---

## Management Commands Reference

| Command | What it does | Requires Ollama |
| --- | --- | --- |
| `bootstrap_academy_scope` | Create Academy's own `academy` ApplicationScope when it is safe to, or confirm the existing one. Never adopts another application's scope; never reactivates, renames or reassigns anything | No |
| `seed_academy_training_data` | Create training provider, models, agents, tutorial modules and missions (needs Academy's ApplicationScope) | No |
| `import_academy_docs` | Synchronize `ai_hub/_docs/` and `docs_source/` into the database | No |
| `embed_docs` | Generate semantic embeddings for all active documentation chunks | Yes (`bge-m3:latest`) |
| `embed_docs --force` | Re-generate embeddings even for chunks that already have one | Yes (`bge-m3:latest`) |
| `seed_lab_exercises` | Seed lab exercises. Do not use `--create-evaluator` yet: it currently fails to create the evaluator agent ([issue #26](https://github.com/Vgarcan/ai_hub-academy_app/issues/26)) | No |
| `seed_ollama_agents` | Set up Ollama provider, LLM model, embedding model, and GAME agents | Yes (`qwen3:8b`, `bge-m3:latest`) |
| `run_doc_sync` | Synchronize both documentation roots through the Documentation Sync GAME agent (smoke test) | Yes (`qwen3:8b`) |

### Full Setup Order

```bash
python manage.py migrate
python manage.py bootstrap_academy_scope      # the seeds need an ApplicationScope
python manage.py seed_academy_training_data   # tutorials, training provider
python manage.py import_academy_docs          # docs into database
python manage.py seed_ollama_agents           # Ollama provider + agents
python manage.py embed_docs --model bge-m3:latest  # semantic search
python manage.py seed_lab_exercises          # labs (AI scoring: see the known limitation above)
python manage.py run_doc_sync                 # smoke test
```

`setup_dev.py` runs `migrate` and the first three commands automatically, then
creates the `admin` user. Every step is safe to repeat. Ollama is required only for
the Ollama-backed agent and embedding commands; labs can be seeded without it.

**Why a scope, and whose.** AI Hub Core keeps every agent, Knowledge collection and
GAME workspace inside an `ApplicationScope` and never creates or guesses one. A fresh
database has none, so Academy provides its own explicitly with
`bootstrap_academy_scope`. Academy owns one named scope, `academy` (or the slug in
`ACADEMY_APPLICATION_SCOPE_SLUG`), and **never adopts another application's scope**
merely because it is the only one active. The decision is shared by the bootstrap
and every seed (`seed_academy_training_data`, `seed_ollama_agents`), and each refuses
before writing anything:

- no scope at all: the `academy` scope is created, with no embedding egress;
- the `academy` scope exists and is active: it is used unchanged, whether or not other
  scopes are active;
- the `academy` scope is inactive: stop; it is not reactivated;
- another application's scope is the only active one: stop; it is not adopted, and a
  second active scope is not added beside it. Install Academy in a dedicated database
  (the default SQLite file is one), or, for a scope that really is Academy's, such
  as `legacy-default` after an upgrade from before scopes existed, declare it with
  `ACADEMY_APPLICATION_SCOPE_SLUG=legacy-default` in `.env`. A declared scope must
  already exist and be active; it is never created or modified;
- Academy-seeded agents or collections that already live in a different scope: stop;
  they are never reused or moved across the boundary.

Existing data is never reassigned, renamed, deleted or reactivated. Core's Build
Console and GAME quick-start resolve a single active scope and refuse while several
are active, which is a further reason to keep Academy in its own database.

---

## URL Reference

```
/                                                  → Landing page
/docs/                                             → Documentation index
/docs/search/                                      → Documentation search
/docs/<slug>/                                      → Single documentation page
/tutorials/                                        → Tutorial module list
/tutorials/<module_slug>/                          → Module detail + mission list
/tutorials/<module_slug>/<mission_slug>/           → Mission detail + validation
/tutorials/<module_slug>/<mission_slug>/check/     → Mission validation endpoint (POST)
/progress/                                         → User progress dashboard
/tutorials/<module_slug>/lab/<exercise_slug>/      → Lab exercise page
/tutorials/<module_slug>/lab/<exercise_slug>/submit/ → Lab submission endpoint (POST)
/assistant/                                        → Documentation assistant chat UI
/assistant/ask/                                    → Chat API endpoint (POST)
```

---

## Troubleshooting (Academy)

### Documentation Assistant Gives Vague Or Off-Topic Answers

The documentation assistant supports two search modes:

1. **Semantic search** — uses vector embeddings (cosine similarity) to find content even when the user's words differ from the docs. Requires Ollama running with `bge-m3:latest` and at least one `DocumentationChunk` with a non-null embedding.
2. **Keyword search** — plain Django ORM text filtering on `search_text` and `heading`. Active when embeddings are missing or Ollama is unreachable.

If the assistant gives answers that miss the point, semantic search is probably inactive. Verify:

```bash
python manage.py shell -c "
from academy.models import DocumentationChunk
chunks = DocumentationChunk.objects.filter(
    is_active=True,
    page__is_active=True,
    page__source__is_active=True,
)
total = chunks.count()
embedded = chunks.exclude(embedding__isnull=True).count()
print(f'{embedded}/{total} chunks have embeddings')
"
```

If the count shows `0/N`, generate embeddings:

```bash
ollama list            # confirm bge-m3 is available
ollama pull bge-m3     # pull if missing
python manage.py embed_docs --model bge-m3:latest
```

Unchanged imports preserve existing embeddings. Run `embed_docs` after importing
new or changed pages; only chunks without embeddings are processed. Use
`embed_docs --force` only when the embedding model or already-embedded content
must be refreshed.

### Documentation Chat Falls Back To Keyword Search At Runtime

Even with embeddings present, the assistant falls back to keyword search if Ollama is unreachable at query time (used to embed the user's question). Check that Ollama is running and the base URL in the provider config is correct:

```bash
curl http://<ollama-base-url>/api/tags
```

If this returns a list of models, Ollama is reachable. If it times out or refuses, start Ollama or correct the URL.
