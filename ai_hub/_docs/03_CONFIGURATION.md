# Configuration

## Configuration Order

Start small and configure in this order:

1. Provider.
2. Model.
3. Agent.
4. Knowledge and tools, if needed.
5. Orchestrator pipeline or GAME session.
6. One small test run.
7. Review the execution timeline.

Do not begin with a large workflow. The fastest path to a stable AI system is one provider, one model, one narrow agent, and one small run.

## Database Configuration

The host defaults to SQLite:

```text
DATABASE_ENGINE=sqlite
```

Select PostgreSQL with a standard `DATABASE_URL`, or set
`DATABASE_ENGINE=postgresql` and `POSTGRES_DB`, `POSTGRES_USER`,
`POSTGRES_PASSWORD`, `POSTGRES_HOST` and `POSTGRES_PORT`. A URL takes
precedence over the discrete variables. Unsupported engines, missing database
names, invalid ports and invalid connection settings fail during Django startup.

Psycopg 3 is included. PostgreSQL 14+ is supported by Django 5.2; CI exercises
PostgreSQL 16. Keep `DB_CONN_MAX_AGE=0` until the deployment has an intentional
connection/pooling policy.

## Provider Secrets

`ai_hub` stores only the environment variable name for provider credentials.

Environment:

```text
OPENAI_API_KEY=...
```

Admin field:

```text
api_key_env_var = OPENAI_API_KEY
```

Never paste the secret value into the database.

## Provider Configuration

Create a `ProviderConfig` for each AI service account or local endpoint.

Common fields:

- `name`
- `provider_type`
- `base_url`
- `api_key_env_var`
- `default_timeout`
- `is_active`

Examples:

```text
name = Ollama LAN
provider_type = ollama
base_url = http://localhost:11434
api_key_env_var =
default_timeout = 60
```

```text
name = OpenAI Production
provider_type = openai
base_url =
api_key_env_var = OPENAI_API_KEY
default_timeout = 60
```

Use `base_url` for local or custom providers. Leave it blank for standard provider endpoints.

## Model Configuration

Create a `ModelConfig` for each model that agents may use.

Common fields:

- `provider`
- `model_name`
- `temperature_default`
- `max_tokens_default`
- `supports_tools`
- `is_active`

Example for the bundled Ollama adapter:

```text
model_name = ollama/qwen3:8b
temperature_default = 0.30
max_tokens_default = 4000
supports_tools = false
```

Use the exact identifier expected by the selected provider adapter; hosted
provider model catalogs change independently of AI Hub. Use lower temperatures
for structured extraction and contract-heavy JSON. Use higher temperatures for
style, rewriting, and reflective text. If you leave `temperature_default` unset
it defaults to `0.70`.

### Training (stub) provider

The built-in `training` provider is a deterministic stub that returns canned responses without calling any external API — useful for local development and tests with no API key. Its router only recognises a model whose `model_name` is exactly `training` or starts with `training/` (for example `training/assistant`). Model/Admin validation and the Build Console reject other names on a training provider, because they would otherwise fall through to the real client and fail at runtime. Direct ORM code must call `full_clean()` before saving configuration records.

## Agent Configuration

Create an `AgentProfile` for each specialist.

Common fields:

- `name`
- `role`
- `model_config`
- `tools`
- `knowledge_collections`
- `knowledge_max_chars`
- `system_prompt`
- `input_contract`
- `output_contract`
- `execution_mode`
- `is_active`

Example prompt:

```text
You are support_ticket_triage.
Read the user request and return valid JSON only.
Classify the ticket as billing, technical, account, or other.
Do not write the final customer reply.
```

Example input contract:

```json
{
  "required": ["ticket_text"],
  "properties": {
    "ticket_text": {"type": "string"},
    "knowledge_context": {"type": "object"}
  }
}
```

Example output contract:

```json
{
  "required": ["category", "priority", "reason"],
  "properties": {
    "category": {"type": "string"},
    "priority": {"type": "string"},
    "reason": {"type": "string"}
  }
}
```

Agents can be reused in Orchestrator, GAME, both, or neither.

## Knowledge Configuration

Use `KnowledgeCollection` to group documents by purpose.

Examples:

```text
Safety rules
Product docs
Interpretation style guide
Legal disclaimers
```

Use `KnowledgeDocument` for curated text, source files, tags, language, and status.

Only active documents from active collections are exposed to agent payloads.
The default retrieval-first mode injects a bounded collection/document index,
not document bodies. Attaching an active collection also makes the six built-in
read-only retrieval adapters available to the agent. Explicit deny grants and
workspace allow/block policy still apply.

For temporary compatibility, the eager flag can inject document text up to the
agent character budget. New integrations should leave that flag disabled.
Documents need chunks to be searchable and readable; the Build Console and
migration `0019_retrieval_first_foundation` create one initial chunk for curated
text that has not been chunked yet.

Recommended `tags` shape:

```json
["refunds", "policy", "support"]
```

## Tool Configuration

The preferred configuration groups tools into `Toolbox` records, assigns those
toolboxes to agents, and uses `AgentToolGrant` for explicit allow/deny
exceptions. Workspace policy can further filter the resolved set.

The Orchestrator and GAME runners use the resolved deliberate runtime by
default. The resolver combines toolbox assignments, grants, compatible legacy
direct attachments, and workspace policy. The model chooses one advertised
tool at a time; successful calls create `ToolExecutionRun` records linked to the
session and step.

For a temporary compatibility rollback, set
`runtime_config.agent_tool_runtime="legacy_preexecute"` on a session or set
`AI_HUB_DEFAULT_AGENT_TOOL_RUNTIME=legacy_preexecute` for the host. That mode
sees only direct `AgentProfile.tools` attachments and executes the allowed set
before the model call. It is not the target architecture.

Common fields:

- `name`
- `tool_kind`
- `input_schema`
- `output_schema`
- `config`
- `is_active`

Example input schema:

```json
{
  "required": ["customer_id"],
  "properties": {
    "customer_id": {"type": "string"}
  }
}
```

Example config:

```json
{
  "url": "https://api.example.com/customers/{customer_id}",
  "method": "GET",
  "allowed_hosts": ["api.example.com"],
  "timeout": 10
}
```

Do not store secrets in tool config. Use environment variables or host-project adapters.

Python-callable tools are code-execution capabilities. Their dotted callable path must also appear in the host setting `AI_HUB_ALLOWED_TOOL_CALLABLES`. A Python callable classified as a GAME `context_tool` additionally requires `config.read_only=true`; HTTP context tools must use GET or HEAD and an explicitly allowed host.

## GAME Feature Flags

The GAME subsystem is gated by per-capability flags read from Django settings (overridable via environment variables). They let a host project enable GAME incrementally and act as kill-switches.

| Setting | Gates |
| --- | --- |
| `AI_HUB_GAME_GOALS_ENABLED` | Goal creation |
| `AI_HUB_GAME_SCHEDULER_ENABLED` | Claiming the next eligible goal |
| `AI_HUB_GAME_ACTION_DISPATCH_ENABLED` | Action execution and approval dispatch |
| `AI_HUB_GAME_MEMORY_ENABLED` | Recording scoped memory |
| `AI_HUB_GAME_RESUME_ENABLED` | Resuming a waiting session |
| `AI_HUB_GAME_DELEGATION_ENABLED` | Sub-agent delegation |

Behaviour:

- In this repo's settings each flag defaults to the value of `DEBUG`; development is open, production is fail-closed unless explicitly enabled.
- The reusable safety default is **fail-closed**: when a flag is disabled, the matching service raises a `ValidationError` rather than running.
- Flags primarily gate the **service layer**. Some Admin entry points also hide
  operations when a flag is disabled, but direct model writes and not every
  lifecycle helper are covered uniformly. Treat the service boundary and
  database permissions as the security boundary; do not rely on a hidden Admin
  button alone.

Set a flag to `False` to disable a capability without a code change:

```text
AI_HUB_GAME_DELEGATION_ENABLED=False
```

## Tool Runtime And Retrieval Flags

| Setting | Default | Use |
| --- | --- | --- |
| `AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED` | `False` | Enables GAME actions linked to reusable `ToolDefinition` records. |
| `AI_HUB_DEFAULT_AGENT_TOOL_RUNTIME` | `resolved` | Selects the normal Orchestrator/GAME agent-call path. Allowed values: `resolved` and temporary `legacy_preexecute`. |
| `AI_HUB_LEGACY_EAGER_KNOWLEDGE_CONTEXT_ENABLED` | `False` | Compatibility switch for bounded eager document injection. Retrieval-first indexes and tools are the default. |
| `AI_HUB_MAX_TOOL_ROUNDS_PER_AGENT_CALL` | `3` | Caps deliberate tool-call loops per agent call. |
| `AI_HUB_MAX_TOOL_OBSERVATION_CHARS` | `12000` | Caps each tool result copied into the model prompt; the complete result remains in `ToolExecutionRun`. |

`AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED` is a separate GAME selected-action
kill-switch; it does not choose the normal agent-call runtime. A session-level
`agent_tool_runtime` value overrides the host default, which makes legacy
rollback explicit and testable. Explicitly disable legacy eager knowledge when
testing retrieval-first behavior; enable it only while migrating a caller that
still depends on injected document bodies.

## Provider Health Check Policy

The Control Center and the Academy dashboard status page can run a live health
probe against Ollama providers (`GET <base_url>/api/tags`). Because the provider
`base_url` is admin-controlled, this probe is a small SSRF surface and is
governed by a shared trusted-endpoint policy:

| Setting | Default | Use |
| --- | --- | --- |
| `AI_HUB_PROVIDER_HEALTH_ALLOWED_HOSTS` | empty (permissive) | Comma-separated host allow-list for the live health probe. |

Behavior:

- The probe always requires an `http`/`https` base URL with a real host; anything
  else (e.g. `file://`) is refused without making a request.
- When the allow-list is **empty** (the default), any valid http(s) host is
  allowed — this keeps local development against `http://localhost:11434` working.
- When the allow-list is **non-empty**, only those hosts are probed; a provider
  pointing elsewhere is reported as a health warning and **no outbound request is
  made**. Set it in production, e.g.
  `AI_HUB_PROVIDER_HEALTH_ALLOWED_HOSTS="localhost,127.0.0.1,ollama.internal"`.

## Runtime Modes

Execution sessions support:

| Mode | Use |
| --- | --- |
| `sync` | Local quick tests |
| `async` | Marks work intended for asynchronous operation; the bundled runner still executes inline when called |
| `hybrid` | Orchestrator only: run the first step and wait for continuation |

`runtime_mode=async` does not enqueue work. A host must provide a worker/queue
and call `run_execution_session()` from it. The bundled project does not yet
ship that general worker contract. GAME rejects `hybrid`; its approval and
information pauses use the explicit continuation services instead.
