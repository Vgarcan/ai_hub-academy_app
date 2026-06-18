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

Use lower temperatures for structured extraction and contract-heavy JSON. Use higher temperatures for style, rewriting, and reflective text.

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
  "timeout": 10
}
```

Do not store secrets in tool config. Use environment variables or host-project adapters.

## Runtime Modes

Execution sessions support:

| Mode | Use |
| --- | --- |
| `sync` | Local quick tests |
| `async` | Normal long-running AI work |
| `hybrid` | First step now, continuation later |

Async is usually the best default for user-facing AI features.
