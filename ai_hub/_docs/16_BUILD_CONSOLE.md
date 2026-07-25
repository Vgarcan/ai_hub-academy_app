# Build Console

## What it is

The Build Console is a multi-step guided wizard for creating GAME sessions and Orchestrator pipelines from scratch. It walks you through the full object-creation chain — engine, agent, tools, knowledge, and workspace-specific configuration — and submits everything as one atomic database transaction.

Use it when bootstrapping a new setup or running a quick test configuration. Use the raw admin forms when you need fine-grained control over an existing object (e.g., editing a pipeline step's `input_mapping`, changing knowledge-document metadata, or changing an agent's prompt without creating a new one).

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

### Step 1 — Engine

Choose how the AI model is configured.

**Reuse existing** (default): select an active `ModelConfig` whose provider is
also active. No new provider or model objects are created. Use this when a model
is already tested and working.

**Create new**: provide:

- provider name,
- provider type,
- model name,
- temperature.

The wizard calls `get_or_create` on `ProviderConfig` (by name) and on `ModelConfig` (by name under that provider). If matching records already exist, they are reused without modification. No duplicates are created.

Training stub: if you want to wire a training provider for local tests, name the model `training` or `training/<name>`. See `ModelConfig.clean()` for the convention.

### Step 2 — Agent

Choose how the agent is configured.

**Reuse existing**: select an active `AgentProfile` whose model and provider are
also active. Tools and knowledge selected in this step are added to the reused
agent (get_or_create for toolboxes; existing toolboxes and knowledge are not
removed).

**Create new**: provide:

- name,
- role,
- prompt,
- input and output contracts for Orchestrator agents.

A new `AgentProfile` is created and linked to the engine selected in Step 1.

#### Tools

Expand the **Tools** subsection. Check any toolboxes to assign. Each checked toolbox creates an `AgentToolboxAssignment` record using `get_or_create`, so re-running the wizard with the same agent and toolboxes does not produce duplicate assignments.

Prefer toolbox assignments over per-agent tool grants. See `07_ADMIN_GUIDE.md` for the recommended setup order.

The assignment feeds the governed resolver used by the default
Orchestrator/GAME runtime. Toolbox-only capabilities are therefore available to
the model without a duplicate direct assignment. GAME model-call manifests
contain context tools only; define a governed GAME action for selected
side-effect work. See `15_RUNTIME_STATUS.md`.

#### Knowledge

Expand the **Knowledge** subsection. Choose a mode:

| Mode | What happens |
| --- | --- |
| **No knowledge** (default) | Nothing is created or attached. |
| **Reuse collection** | An existing `KnowledgeCollection` is linked to the agent via `agent.knowledge_collections.add()`. |
| **Create collection** | A new `KnowledgeCollection` and one active `KnowledgeDocument` are created with one initial retrievable chunk. |

Important: when creating documents, the wizard forces `status=ACTIVE` and
creates one initial chunk from the curated text. Draft documents are invisible
to agents at runtime. The initial chunk makes the text immediately usable by
retrieval; replace it with deliberate section chunks for larger documents.
Collection name, document title and non-empty curated text are required in this
mode.

## GAME-specific steps

### Step 3 — Goal

- **Goal text**: the natural-language goal the agent will receive. Keep the first goal small and specific. See `06_GAME_WORKSPACE.md` for good goal examples.
- **Source label**: optional human-readable name shown in session lists.
- **Flavor**:
  - **Simple** (default): creates a standalone `ExecutionSession` with `runtime_kind=GAME`. No `GameWorkspace` or `GameGoal` record is created. Best for quick experiments.
  - **Advanced**: requires `AI_HUB_GAME_GOALS_ENABLED=True`. Creates a `GameWorkspace` (get_or_create by name), a `GameGoal` linked to that workspace, and a goal-bound `ExecutionSession`. Use this when you want durable goals, scheduling, dependencies, and the full operational dashboard.

### Step 4 — Governance

(Advanced flavor only. Skipped for Simple.)

These choices are written to the **workspace's `default_policy`**, not to the session's `runtime_config`.

- **Workspace name**: used for `get_or_create` on `GameWorkspace`. If a workspace with this name already exists, the new goal is added to it. **Caveat:** the policy below is only applied when the workspace is *newly created* — `get_or_create` does not overwrite the policy of an existing workspace.
- **Safety checkboxes**: `safety_require_approval_medium` and `safety_require_approval_high` → stored as `default_policy.safety.require_approval_for_medium_risk` and `…require_approval_for_high_risk`. External writes are always locked (`safety.allow_external_writes = False`).
- **Budget**: the `budget_max_actions` field → `default_policy.budget.max_action_runs_per_session` (defaults to 2, minimum 1). `default_policy.budget.max_iterations_per_session` is set from the Max iterations field.
- `default_policy.allowed_actions` is seeded with `["submit_for_approval"]`.

### Step 5 — Runtime

- **Max iterations**: the `max_iterations` key in `runtime_config`. Keep at 3–5 for initial tests. Increase only after the session timeline looks correct.
- **Runtime mode**: `async` (default) or `sync`. *(Simple flavor only.)*
- **Strict response contract**: if checked, missing keys or invalid JSON from the agent fail the session immediately. Stored in `runtime_config.strict_response_contract`.
- **Initial context**: optional JSON string prepopulated into the session's starting context. *(Simple flavor only.)*

> **Advanced flavor note:** the goal-bound session is built by `create_goal_execution_session()`, which always runs **async** and derives the starting context from `goal.context`. The **Runtime mode** and **Initial context** fields are therefore ignored in Advanced mode — only **Max iterations** and **Strict response contract** (carried in `runtime_config`) take effect.

### Step 6 — Review

Shows a live manifest of all choices. Verify the engine, agent, goal and runtime fields before submitting. Click **Create GAME session** to submit.

## Orchestrator-specific steps

### Step 3 — Pipeline

- **Pipeline name**: human-readable name for the new `PipelineDefinition`. Use a naming convention such as `support_triage_v1`.
- **Description**: optional text stored on the definition.

Existing inactive pipelines are listed below for reference. To modify an existing pipeline, use the raw admin instead.

### Step 4 — Steps

Add `PipelineStep` records. You may leave the pipeline without steps while it is
a draft, but activation requires at least one step.

- Click **Add step** to append a row.
- Each row has an agent selector, an error strategy (`stop` or `continue`) and
  expandable input/output mapping editors.
- Row position determines continuous order: `1, 2, 3, 4 …`.
- Remove a row before submit to prevent that step from being created.

The wizard validates mapping JSON as an object and stores it on each
`PipelineStep`. Use the Pipeline Designer afterward for `fallback_agent` or
other fine-grained edits not offered by the wizard.

### Step 5 — Contracts

- **Input contract**: JSON object describing the keys expected at pipeline entry.
- **Output contract**: JSON object describing the keys the pipeline should produce.

Both are parsed as JSON and stored on `PipelineDefinition.global_input_contract` and `PipelineDefinition.global_output_contract` (JSON fields, default `{}`). Leave blank if contracts are not yet defined.

### Step 6 — Activate

Optional checkbox to set `PipelineDefinition.is_active=True` immediately on creation.

A gate warning is shown: activating validates that steps are continuous and each step's agent has a valid contract. Leave this unchecked and activate later from the Orchestrator admin once all agents are verified.

### Step 7 — Review

Shows the manifest and step list. Click **Create pipeline** to submit.

## What gets created

### GAME Simple

```text
ProviderConfig (get_or_create by name)
  └─ ModelConfig (get_or_create by name)
       └─ AgentProfile (reuse or create)
            ├─ AgentToolboxAssignment × N (get_or_create per selected toolbox)
            └─ KnowledgeCollection (if create mode)
                 └─ KnowledgeDocument (status=ACTIVE)
                      └─ KnowledgeDocumentChunk (initial curated-text chunk)
ExecutionSession (runtime_kind=GAME, status=PENDING)
```

### GAME Advanced

```text
ProviderConfig (get_or_create by name)
  └─ ModelConfig (get_or_create by name)
       └─ AgentProfile (reuse or create)
            ├─ AgentToolboxAssignment × N (get_or_create per selected toolbox)
            └─ KnowledgeCollection (if create mode)
                 └─ KnowledgeDocument (status=ACTIVE)
                      └─ KnowledgeDocumentChunk (initial curated-text chunk)
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
                 └─ KnowledgeDocument (status=ACTIVE)
                      └─ KnowledgeDocumentChunk (initial curated-text chunk)
PipelineDefinition (create, is_active from Step 6 checkbox)
  └─ PipelineStep × N (create per step row, in order)
```

## Atomic transaction

All objects are created inside a single `transaction.atomic()` block. Field or
resource validation errors raise a sentinel that rolls back the entire
transaction — including any partially-created `ProviderConfig`, `ModelConfig`,
`AgentProfile`, or `KnowledgeCollection`. Nothing from an invalid submission is
left in a partial state.

An immediate-activation request is the deliberate exception: if activation
validation fails, the fully built pipeline is kept inactive and the page shows a
warning explaining what must be corrected.

Errors are collected and displayed at the top of the page. Invalid object JSON
in contracts, mappings or initial context is rejected instead of being silently
replaced with `{}`. The form is redisplayed for correction.

## After creation

**GAME**: you are redirected to the new `ExecutionSession` in admin. The session has `status=PENDING`. Use the **Run session** action or the execution endpoint to start it.

**Orchestrator**: you are redirected to the new `PipelineDefinition` in admin.
Review the steps and mappings. Activate the pipeline when agents and contracts
are ready.

## What the wizard does not do

- **Provider endpoint or credentials**: the wizard does not collect a base URL or API-key environment-variable name. Configure those on the provider record before testing a hosted or remote provider.
- **Additional GAME policy and actions**: Advanced mode seeds a conservative
  workspace policy (approval settings, iteration/action budgets, external writes
  off and `submit_for_approval` allowed) only when the workspace is new. It does
  not create `GameActionDefinition` records or workspace agent/action mappings;
  configure those in raw Admin.
- **Fallback step strategy**: the wizard offers `stop` and `continue`. Configure
  `fallback_agent` from the Pipeline Designer when needed.
- **Knowledge files**: the wizard accepts curated text, not uploaded-file
  ingestion. It creates one initial chunk per new text document, but does not
  perform semantic sectioning, embeddings or file parsing. Create deliberate
  `KnowledgeDocumentChunk` records for larger documents.
- **Editing existing objects**: the wizard only creates new objects (or reuses via get_or_create). To change an existing agent's prompt, provider base URL, or pipeline step order, edit those records directly in admin.
