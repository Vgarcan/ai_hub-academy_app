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

Imports Markdown files from `docs_source/` and stores them as `DocumentationPage` and `DocumentationChunk` records.

**Key models:**

- `DocumentationSource` — groups pages by source directory or topic.
- `DocumentationPage` — one imported Markdown file. Has a `slug` used for the URL.
- `DocumentationChunk` — a section of a page (heading + body). Stores a `bge-m3` embedding vector for semantic search.

**Management commands:**

```bash
# Import or re-import all Markdown files from docs_source/
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
4. Retrieved chunks are passed to an `ExecutionSession` that runs the `AI Hub Documentation Assistant` GAME agent.
5. The agent generates an answer and the response is stored in `DocumentationChatMessage` alongside the retrieved chunks.

**Key models:**

- `DocumentationChatSession` — one conversation thread per user.
- `DocumentationChatMessage` — one message (user or AI). AI messages have a `retrieved_chunks` M2M relation to the chunks that informed the answer.

**Key services:**

- `academy.services.documentation_assistant.answer_documentation_question()` — orchestrates the full search + agent call.
- `academy.services.documentation_search.search_documentation()` — semantic or keyword search over chunks.
- `academy.services.embeddings.get_embedding()` — calls Ollama `/api/embed` to generate a vector.

**Agent:** `AI Hub Documentation Assistant` — seeded by `seed_ollama_agents`. Uses GAME runtime with up to 3 iterations and the `search_documentation` tool.

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
- `LabAttempt` — one user submission. Stores the raw answer, AI score (0–100), and AI feedback.

**AI evaluation:** `academy.services.lab_evaluator.evaluate_lab_answer()` creates an `ExecutionSession` with the `Lab Exercise Evaluator` agent. The agent returns JSON with `score`, `feedback`, and `follow_up_question`. If no agent is configured, the attempt is saved with a `pending` score.

**Management command:**

```bash
# Seed lab exercises for all tutorial modules
python manage.py seed_lab_exercises
```

---

## Management Commands Reference

| Command | What it does | Requires Ollama |
| --- | --- | --- |
| `seed_academy_training_data` | Create training provider, models, agents, tutorial modules and missions | No |
| `import_academy_docs` | Import or re-import Markdown files from `docs_source/` into the database | No |
| `embed_docs` | Generate semantic embeddings for all active documentation chunks | Yes (`bge-m3:latest`) |
| `embed_docs --force` | Re-generate embeddings even for chunks that already have one | Yes (`bge-m3:latest`) |
| `seed_lab_exercises` | Seed lab exercises for tutorial modules | No |
| `seed_ollama_agents` | Set up Ollama provider, LLM model, embedding model, and GAME agents | Yes (`qwen3:8b`, `bge-m3:latest`) |
| `run_doc_sync` | Run the Documentation Sync GAME agent once (smoke test) | Yes (`qwen3:8b`) |

### Full Setup Order

```bash
python manage.py seed_academy_training_data   # tutorials, training provider
python manage.py import_academy_docs          # docs into database
python manage.py embed_docs --model bge-m3:latest  # semantic search
python manage.py seed_lab_exercises           # lab exercises
python manage.py seed_ollama_agents           # Ollama provider + agents
python manage.py run_doc_sync                 # smoke test
```

`setup_dev.py` runs the first two automatically. The rest require Ollama.

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
