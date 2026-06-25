# Build Console

## What it is

The Build Console is a multi-step guided wizard for creating GAME sessions and Orchestrator pipelines from scratch. It walks you through the full object-creation chain — engine, agent, tools, knowledge, and workspace-specific configuration — and submits everything as one atomic database transaction.

Use it when bootstrapping a new setup or running a quick test configuration. Use the raw admin forms when you need fine-grained control over an existing object (e.g., editing a pipeline step's `input_mapping`, adjusting knowledge weights, or changing an agent's prompt without creating a new one).

## How to access it

| Entry point | URL |
| --- | --- |
| AI Hub home | `/admin/ai_hub/` → **Build Console** button |
| GAME workspace | `/admin/ai_hub/workspaces/game/` → **Build Console** (primary action) |
| Orchestrator workspace | `/admin/ai_hub/workspaces/orchestrator/` → **Build Console** (primary action) |

Direct URLs:

```text
/admin/ai_hub/workspaces/build/?kind=game
/admin/ai_hub/workspaces/build/?kind=orchestrator
```

## Blueprint selector

The wizard opens on a blueprint selector. Clicking **GAME** (rose accent) or **Orchestrator** (green accent) switches the rail and all step panels to that blueprint. The selected blueprint drives what gets created on submit.

## Shared steps: Engine and Agent

Both blueprints begin with the same two steps.

### Step 0 — Engine

Choose how the AI model is configured.

**Reuse existing** (default): select a `ModelConfig` from the dropdown. No new provider or model objects are created. Use this when a model is already tested and working.

**Create new**: provide:

- provider name,
- provider base URL,
- API key environment variable name (not the key value itself — the name of the env var where the key is stored),
- model name.

The wizard calls `get_or_create` on `ProviderConfig` (by name) and on `ModelConfig` (by name under that provider). If matching records already exist, they are reused without modification. No duplicates are created.

Training stub: if you want to wire a training provider for local tests, name the model `training` or `training/<name>`. See `ModelConfig.clean()` for the convention.

### Step 1 — Agent

Choose how the agent is configured.

**Reuse existing**: select an existing `AgentProfile` by name. Tools and knowledge selected in this step are added to the reused agent (get_or_create for toolboxes; existing toolboxes and knowledge are not removed).

**Create new**: provide:

- name,
- prompt,
- optional input contract,
- optional output contract.

A new `AgentProfile` is created and linked to the engine selected in Step 0.

#### Tools

Expand the **Tools** subsection. Check any toolboxes to assign. Each checked toolbox creates an `AgentToolboxAssignment` record using `get_or_create`, so re-running the wizard with the same agent and toolboxes does not produce duplicate assignments.

Prefer toolbox assignments over per-agent tool grants. See `07_ADMIN_GUIDE.md` for the recommended setup order.

#### Knowledge

Expand the **Knowledge** subsection. Choose a mode:

| Mode | What happens |
| --- | --- |
| **No knowledge** (default) | Nothing is created or attached. |
| **Reuse collection** | An existing `KnowledgeCollection` is linked to the agent via `agent.knowledge_collections.add()`. |
| **Create collection** | A new `KnowledgeCollection` is created. Add document rows — each row becomes one `KnowledgeDocument`. |

Important: when creating documents, the wizard forces `status=ACTIVE` on every document. Draft documents are invisible to agents at runtime. Do not change this unless you intend the documents to be hidden.

## GAME-specific steps

### Step 2 — Goal

- **Goal text**: the natural-language goal the agent will receive. Keep the first goal small and specific. See `06_GAME_WORKSPACE.md` for good goal examples.
- **Source label**: optional human-readable name shown in session lists.
- **Flavor**:
  - **Simple** (default): creates a standalone `ExecutionSession` with `runtime_kind=GAME`. No `GameWorkspace` or `GameGoal` record is created. Best for quick experiments.
  - **Advanced**: requires `AI_HUB_GAME_GOALS_ENABLED=True`. Creates a `GameWorkspace` (get_or_create by name), a `GameGoal` linked to that workspace, and a goal-bound `ExecutionSession`. Use this when you want durable goals, scheduling, dependencies, and the full operational dashboard.

### Step 3 — Governance

(Advanced flavor only. Skipped for Simple.)

- **Workspace name**: used for `get_or_create` on `GameWorkspace`. If a workspace with this name already exists, the new goal is added to it.
- **Safety checkboxes**: `strict_response_contract`, `game_action_dispatch_enabled`.
- **Budget**: max actions per session, stored in `runtime_config.policy.max_budget_actions`.

### Step 4 — Runtime

- **Max iterations**: the `max_iterations` key in `runtime_config`. Keep at 3–5 for initial tests. Increase only after the session timeline looks correct.
- **Runtime mode**: `async` (default) or `sync`.
- **Strict response contract**: if checked, missing keys or invalid JSON from the agent fail the session immediately.
- **Initial context**: optional JSON string prepopulated into the session's starting context.

### Step 5 — Review

Shows a live manifest of all choices. Verify the engine, agent, goal and runtime fields before submitting. Click **Create GAME session** to submit.

## Orchestrator-specific steps

### Step 2 — Pipeline

- **Pipeline name**: human-readable name for the new `PipelineDefinition`. Use a naming convention such as `support_triage_v1`.
- **Description**: optional text stored on the definition.

Existing inactive pipelines are listed below for reference. To modify an existing pipeline, use the raw admin instead.

### Step 3 — Steps

Add one or more `PipelineStep` records.

- Click **Add step** to append a row.
- Each row has an order field and an agent selector.
- Keep order values continuous: `1, 2, 3, 4 …`.
- Remove a row before submit to prevent that step from being created.

The wizard creates the steps in order. Step `input_mapping` and `output_mapping` are not configured here — edit the `PipelineStep` records directly in admin after creation.

### Step 4 — Contracts

- **Input contract**: JSON schema or free-form keys expected at pipeline entry.
- **Output contract**: JSON schema or keys the pipeline should produce.

Both are stored as strings on `PipelineDefinition`. Leave blank if contracts are not yet defined.

### Step 5 — Activate

Optional checkbox to set `PipelineDefinition.is_active=True` immediately on creation.

A gate warning is shown: activating validates that steps are continuous and each step's agent has a valid contract. Leave this unchecked and activate later from the Orchestrator admin once all agents are verified.

### Step 6 — Review

Shows the manifest and step list. Click **Create pipeline** to submit.

## What gets created

### GAME Simple

```text
ProviderConfig (get_or_create by name)
  └─ ModelConfig (get_or_create by name)
       └─ AgentProfile (reuse or create)
            ├─ AgentToolboxAssignment × N (get_or_create per selected toolbox)
            └─ KnowledgeCollection (if create mode)
                 └─ KnowledgeDocument × N  (status=ACTIVE, one per row)
ExecutionSession (runtime_kind=GAME, status=PENDING)
```

### GAME Advanced

```text
ProviderConfig (get_or_create by name)
  └─ ModelConfig (get_or_create by name)
       └─ AgentProfile (reuse or create)
            ├─ AgentToolboxAssignment × N (get_or_create per selected toolbox)
            └─ KnowledgeCollection (if create mode)
                 └─ KnowledgeDocument × N  (status=ACTIVE)
GameWorkspace (get_or_create by name)
  └─ GameGoal (create)
       └─ ExecutionSession (runtime_kind=GAME, status=PENDING, goal=GameGoal)
```

### Orchestrator

```text
ProviderConfig (get_or_create by name)
  └─ ModelConfig (get_or_create by name)
       └─ AgentProfile (reuse or create)
            ├─ AgentToolboxAssignment × N (get_or_create per selected toolbox)
            └─ KnowledgeCollection (if create mode)
                 └─ KnowledgeDocument × N  (status=ACTIVE)
PipelineDefinition (create, is_active from Step 5 checkbox)
  └─ PipelineStep × N (create per step row, in order)
```

## Atomic transaction

All objects are created inside a single `transaction.atomic()` block. If any step raises a validation error, a sentinel exception rolls back the entire transaction — including any partially-created `ProviderConfig`, `ModelConfig`, `AgentProfile`, or `KnowledgeCollection`. Nothing is left in a partial state.

Errors are collected and displayed at the top of the page. The form is redisplayed for correction.

## After creation

**GAME**: you are redirected to the new `ExecutionSession` in admin. The session has `status=PENDING`. Use the **Run session** action or the execution endpoint to start it.

**Orchestrator**: you are redirected to the new `PipelineDefinition` in admin. Review the steps. Configure `input_mapping` and `output_mapping` on each `PipelineStep`. Activate the pipeline when agents and contracts are ready.

## What the wizard does not do

- **Provider credentials**: the wizard stores an env var name, not a key value. Set the actual key in your environment before testing.
- **GAME workspace policy and actions**: `GameActionDefinition` records, workspace-level budgets, and action allow-lists must be configured in the raw admin after the session is created.
- **Step mappings**: `input_mapping` and `output_mapping` on `PipelineStep` are not set by the wizard. Edit each step in admin after creation.
- **Knowledge files**: the wizard creates text-content `KnowledgeDocument` records only. It does not process uploaded files or generate document chunks. Use `KnowledgeDocumentChunk` records and the retrieval service for RAG.
- **Editing existing objects**: the wizard only creates new objects (or reuses via get_or_create). To change an existing agent's prompt, provider base URL, or pipeline step order, edit those records directly in admin.
