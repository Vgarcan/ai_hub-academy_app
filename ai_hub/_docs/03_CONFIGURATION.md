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

Examples:

```text
model_name = gpt-4.1-mini
temperature_default = 0.20
max_tokens_default = 1200
supports_tools = true
```

```text
model_name = ollama/qwen3:8b
temperature_default = 0.30
max_tokens_default = 4000
supports_tools = false
```

Use lower temperatures for structured extraction and contract-heavy JSON. Use higher temperatures for style, rewriting, and reflective text. If you leave `temperature_default` unset it defaults to `0.70`.

### Training (stub) provider

The built-in `training` provider is a deterministic stub that returns canned responses without calling any external API — useful for local development and tests with no API key. Its router only recognises a model whose `model_name` is exactly `training` or starts with `training/` (for example `training/assistant`). Any other name on a training provider is rejected at save time with a clear validation error, because it would otherwise fall through to the real client and fail at runtime.

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

Only active documents from active collections are injected into agent payloads.

Recommended `tags` shape:

```json
["refunds", "policy", "support"]
```

## Tool Configuration

Tools are attached to agents.

They are not attached to a workspace. If an agent is used in GAME and has tools, the GAME runtime treats those tools as capabilities of that agent.

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
- Flags gate the **service layer** only. The Django admin writes through the model directly, so the "Add GAME goal" form and the goal bulk actions are not blocked by a disabled flag. To hard-stop an operation in an environment, also remove the relevant admin permissions.

Set a flag to `False` to disable a capability without a code change:

```text
AI_HUB_GAME_DELEGATION_ENABLED=False
```

## Tool Runtime And Retrieval Flags

| Setting | Default | Use |
| --- | --- | --- |
| `AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED` | `False` | Enables GAME actions linked to reusable `ToolDefinition` records. |
| `AI_HUB_LEGACY_EAGER_KNOWLEDGE_CONTEXT_ENABLED` | `True` | Keeps legacy full-document knowledge injection. Set `False` to use retrieval-first indexes and tools. |
| `AI_HUB_MAX_TOOL_ROUNDS_PER_AGENT_CALL` | `3` | Caps deliberate tool-call loops per agent call. |

For automated testing of the new tool platform, explicitly set the unified tool
runtime flag when testing GAME reusable-tool actions, and explicitly disable
legacy eager knowledge when testing retrieval-first behavior.

## Runtime Modes

Execution sessions support:

| Mode | Use |
| --- | --- |
| `sync` | Local quick tests |
| `async` | Normal long-running AI work |
| `hybrid` | First step now, continuation later |

Async is usually the best default for user-facing AI features.
