# Installation

This guide explains how to install `ai_hub` as a reusable Django app in any host
project.

`ai_hub` is intentionally self-contained. A host project installs the app, runs
its migrations, configures providers and models in Django Admin, then connects
domain-specific workflows through a thin adapter.

## Requirements

- Python 3.12 or newer.
- Django 5 or newer.
- SQLite for local development, or PostgreSQL for production.
- A Django admin user.
- Static files configured for Django Admin.
- Provider credentials stored in environment variables.

The app can work with local models, hosted providers, or custom provider
adapters. Provider credentials should never be stored directly in the database.

## Add The App

Copy or install the `ai_hub` package into the Django project.

Example layout:

```text
my_project/
    manage.py
    my_project/
        settings.py
        urls.py
    ai_hub/
        apps.py
        models.py
        admin.py
```

Add the app to `INSTALLED_APPS`.

```python
INSTALLED_APPS = [
    # ...
    "ai_hub",
]
```

The technical app label is `ai_hub`. The visible admin name is `AI Hub`.

## Include URLs

Add the internal URLs if the host project needs staff-only execution endpoints.

```python
from django.urls import include, path

urlpatterns = [
    # ...
    path("ai-hub/", include("ai_hub.urls")),
]
```

The most important URL is the internal execution endpoint:

```text
/ai-hub/internal/execution-session/run/
```

Use it when the host project wants a staff-only HTTP trigger for execution
sessions. If the host project calls services directly from Python, including
these URLs is optional.

## Run Migrations

```bash
python manage.py migrate ai_hub
```

After migrating, the database should contain reusable AI tables such as:

- `ai_hub_providerconfig`
- `ai_hub_modelconfig`
- `ai_hub_agentprofile`
- `ai_hub_pipelinedefinition`
- `ai_hub_executionsession`
- `ai_hub_executionsteprun`

## Create An Admin User

```bash
python manage.py createsuperuser
```

Then open:

```text
/admin/ai_hub/
```

This is the main AI Hub entry point. Users should normally begin here instead of
jumping directly into raw model lists.

## Configure Static Files

The admin experience depends on app-specific CSS and JavaScript. In production
or any `DEBUG=False` environment, run:

```bash
python manage.py collectstatic
```

Confirm that these files are served:

```text
ai_hub/admin/ai_hub_admin.css
ai_hub/admin/ai_hub_admin.js
```

If the page renders without the guided UI styling, the most likely cause is a
static files deployment issue.

## Configure The First Provider

Open:

```text
/admin/ai_hub/providerconfig/add/
```

Recommended first fields:

- `Name`: `Local Ollama` or `OpenAI Production`.
- `Provider type`: choose the matching provider adapter.
- `Base URL`: local or remote endpoint when required.
- `API key env var`: the environment variable name, not the secret itself.
- `Default timeout`: use a realistic value for slow models.
- `Is active`: enabled only when the provider is ready.

Example:

```text
Name: Local Ollama
Provider type: Ollama
Base URL: http://localhost:11434
Default timeout: 180
Is active: yes
```

## Configure The First Model

Open:

```text
/admin/ai_hub/modelconfig/add/
```

Recommended fields:

- `Provider`: the active provider.
- `Display name`: a friendly name for users.
- `Model name`: the exact provider model identifier.
- `Temperature default`: low for structured outputs, higher for creative text.
- `Max tokens default`: large enough for the expected output contract.
- `Supports tools`: only enable this when the model/runtime can use tools.
- `Is active`: enabled only when the model is installed and reachable.

Example:

```text
Display name: Qwen 3 8B
Model name: qwen3:8b
Temperature default: 0.2
Max tokens default: 12000
Supports tools: no
```

## Configure The First Agent

Open:

```text
/admin/ai_hub/agentprofile/add/
```

Minimum useful setup:

- Choose an active model.
- Give the agent a clear role.
- Write a focused system prompt.
- Add an output contract if the result must be structured JSON.
- Attach knowledge collections only when the agent needs curated context.
- Attach tools only when the runtime can execute them safely.

Good agent names are concrete:

```text
Input Normalizer
Evidence Extractor
Report Writer
Support Triage Agent
GAME Planning Agent
```

## Choose A Workspace

AI Hub has two execution workspaces.

Use **Orchestrator** when the workflow has known steps:

```text
input -> extract -> analyze -> write -> final output
```

Use **GAME** when one agent should work autonomously toward a goal:

```text
goal -> decision loop -> observations -> memory -> finish or stop
```

Both workspaces use the same providers, models, agents, knowledge and tools.
The separation is a UX and operating-model separation, not a separate database.

## First Orchestrator Smoke Test

1. Create one active provider.
2. Create one active model.
3. Create one active agent.
4. Create a pipeline with that agent as the entry agent.
5. Add one pipeline step.
6. Create an execution session with `runtime_kind = orchestrator`.
7. Run it from the admin or internal endpoint.
8. Confirm that step logs appear under `ExecutionStepRun`.

## First GAME Smoke Test

1. Create one active provider.
2. Create one active model.
3. Create one active GAME-oriented agent.
4. Open the GAME workspace.
5. Create a GAME session with a small goal.
6. Run the session.
7. Confirm that iterations appear as step runs.
8. Confirm that the session ends as success, failed, waiting or stopped.

## Checks

Run:

```bash
python manage.py check
python manage.py test ai_hub
```

For a host project that already integrates with AI Hub, also run that app's
tests:

```bash
python manage.py test your_host_app ai_hub
```

## Academy App Setup

After installing `ai_hub` and running migrations, the Academy app requires additional setup steps to populate its content and enable all features.

Run these commands in order:

```bash
# 1. Seed tutorial modules, missions, training provider, models, and agents
python manage.py seed_academy_training_data

# 2. Import Markdown documentation from docs_source/ into the database
python manage.py import_academy_docs

# 3. Generate semantic embeddings for the documentation assistant
#    Requires Ollama running with bge-m3:latest
python manage.py embed_docs --model bge-m3:latest

# 4. (Optional) Seed lab exercises for tutorial modules
python manage.py seed_lab_exercises

# 5. (Optional) Set up Ollama provider and GAME agents for real AI responses
#    Skip this if you want to stay on the Training provider stub
python manage.py seed_ollama_agents --base-url http://localhost:11434

# 6. (Optional) Run the Documentation Sync GAME agent to verify agent connectivity
python manage.py run_doc_sync
```

The `setup_dev.py` script runs steps 1 and 2 automatically. Steps 3–6 require Ollama to be running and configured.

### Ollama Models Required

The academy uses two distinct Ollama models:

| Model | Purpose | Pull command |
| --- | --- | --- |
| `qwen3:8b` | LLM for the documentation assistant and sync agents | `ollama pull qwen3:8b` |
| `bge-m3:latest` | Embedding model for semantic search in the chat | `ollama pull bge-m3` |

Both must be available in Ollama before running `seed_ollama_agents` and `embed_docs`. The documentation assistant falls back to keyword search if embeddings are missing or Ollama is unreachable.

### Re-running After Documentation Updates

When the content in `docs_source/` changes, run both commands again to keep the database in sync:

```bash
python manage.py import_academy_docs
python manage.py embed_docs --force
```

`--force` re-generates embeddings even for chunks that already have one.

## Production Checklist

- `ai_hub` is installed directly in `INSTALLED_APPS`.
- Migrations are applied.
- Admin static files are collected and served.
- Provider secrets live in environment variables.
- Provider timeouts match real model latency.
- Models are marked active only when available.
- Agents have clear prompts and contracts.
- Pipelines have ordered steps.
- GAME sessions have a goal and an entry agent.
- Host-specific persistence lives in the host app, not inside `ai_hub`.
