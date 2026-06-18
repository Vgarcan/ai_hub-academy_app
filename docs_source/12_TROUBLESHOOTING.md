# Troubleshooting

## API Key Not Working

Symptoms: provider shows as active but all sessions fail immediately, or the execution session logs show an authentication error.

Check:

- `.env` contains the key with no surrounding quotes and no trailing spaces,
- the server was **restarted** after editing `.env` — changes are not hot-reloaded,
- `api_key_env_var` in the provider config matches the variable name in `.env` exactly (case-sensitive),
- the key has not expired or been revoked in your provider dashboard.

Quick check — print the resolved value from the Django shell:

```bash
python manage.py shell -c "import os; print(repr(os.getenv('OPENAI_API_KEY')))"
```

If it prints `None`, the variable is not loaded. Confirm `.env` is in the project root (same folder as `manage.py`).

## Switching from Training to a Real Provider

After install the Training provider is active and all agents use it. To switch to a real LLM:

1. Add your API key to `.env` and restart the server.
2. Create a new **Provider config** in Admin → AI Hub → Provider configs (`openai`, `anthropic`, or `ollama`).
3. Create a new **Model config** in Admin → AI Hub → Model configs linked to that provider.
4. Edit each **Agent profile** and change its model config to the new one.

Existing ExecutionSessions are unaffected — they keep the provider they ran with.
You do not need to delete or disable the Training provider; it is safe to leave active alongside a real one.

## Ollama Not Reachable

If the provider fails with a connection error and you are using Ollama:

- Confirm Ollama is running: open a terminal and run `ollama list` — if it hangs, start Ollama first.
- Confirm the `base_url` in the provider config matches the actual Ollama address (default: `http://localhost:11434`).
- Verify the model is installed: `ollama list` should show the exact model name you configured.
- If running inside Docker or WSL, `localhost` may not resolve correctly — use the host machine IP instead.

## Admin CSS Looks Broken

For `DEBUG=False`, collect and serve static files.

```bash
python manage.py collectstatic
```

Also confirm your web server serves Django admin static files.

If the browser still shows old styles, hard refresh the page. Admin assets often use cache-busting query strings such as `guided-v8`.

## Admin Form Looks Unguided

Confirm the model admin uses the `ai_hub` styled change form or mixins.

Expected behavior:

- fieldsets are grouped,
- form rows have subtle panels,
- fields include help text,
- text fields include placeholders,
- JSON fields include JSON examples,
- errors are visible and readable.

If a new model admin is added, connect it to the shared form style.

## Provider Fails Before Model Call

Check:

- provider is active,
- model is active,
- `api_key_env_var` exists in the process environment,
- `base_url` is correct for local providers,
- local provider is reachable from the Django process.

For Ollama, confirm:

```text
GET <base_url>/api/tags
```

returns installed models.

## Configured Model Is Missing

The control center may warn when a configured model is not reported by the provider.

Check:

- model name spelling,
- provider prefix,
- local model installation,
- active/inactive state.

Examples:

```text
ollama/qwen3:8b
qwen3:8b
gpt-4.1-mini
```

The exact accepted form depends on the provider adapter.

## Agent Does Not Appear In GAME Workspace

The GAME workspace highlights agents that are already used in GAME sessions or look GAME-ready.

To make an agent easy to identify as GAME-ready, include goal-loop keys in the input contract:

```json
{
  "required": ["goal", "iteration", "memory", "game_response_contract"]
}
```

Also mention goal, action, memory, observation, completion, and final answer in the prompt.

## GAME Session Fails With Contract Error

If strict response contracts are enabled, the agent must return valid JSON with:

- `action`
- `message`
- `complete`
- `final_answer`

Example:

```json
{
  "action": "finish",
  "message": "The goal is complete.",
  "complete": true,
  "final_answer": "Final answer."
}
```

If this fails often, reduce prompt complexity and keep the first goal smaller.

## Pipeline Cannot Be Activated

Check:

- it has at least one step,
- step order is continuous,
- each step agent is active,
- each agent has input and output contracts,
- fallback agents are active when required.

## Pipeline Output Fails With Invalid JSON

Common message:

```text
final_output must be valid JSON
```

Possible causes:

- final model output was truncated,
- final model included commentary or markdown,
- final model generated malformed JSON,
- `max_tokens_default` was too low,
- final step is trying to combine too much data.

Recommended fixes:

1. Inspect the final step in the execution timeline.
2. Check `final_context.final_output`.
3. Check `final_context.final_output_parse_error`.
4. Increase max tokens for the final model if truncation is likely.
5. Make the final step produce a smaller payload.
6. Move domain-specific recovery into the host adapter if intermediate drafts are valid.

## Session Already Has Step Runs

The runtime protects against running the same session twice.

Create a new session if you want to rerun the same goal or source payload.

## Run Is Waiting

`WAITING_ASYNC` means the session intentionally stopped for continuation.

Use a worker or continuation process if your runtime mode expects asynchronous completion.

## Tools Do Not Run

Check:

- tool is active,
- tool is attached to the agent,
- runtime supports the tool kind,
- model supports tool usage if the tool requires model-native tool calls,
- tool config is allowlisted and valid.

## Control Center Shows Warnings

Warnings are intended to surface configuration drift.

Common causes:

- provider cannot be reached,
- configured model is not reported by provider,
- agent has missing or inactive dependencies,
- pipeline contains inactive or misconfigured steps.

## Documentation Assistant Gives Vague Or Off-Topic Answers

The documentation assistant supports two search modes:

1. **Semantic search** — uses vector embeddings (cosine similarity) to find content even when the user's words differ from the docs. Requires Ollama running with `bge-m3:latest` and at least one `DocumentationChunk` with a non-null embedding.
2. **Keyword search** — plain Django ORM text filtering on `search_text` and `heading`. Active when embeddings are missing or Ollama is unreachable.

If the assistant gives answers that miss the point, semantic search is probably inactive. Verify:

```bash
python manage.py shell -c "
from academy.models import DocumentationChunk
total = DocumentationChunk.objects.count()
embedded = DocumentationChunk.objects.exclude(embedding__isnull=True).count()
print(f'{embedded}/{total} chunks have embeddings')
"
```

If the count shows `0/N`, generate embeddings:

```bash
# Confirm bge-m3 is available in Ollama
ollama list

# Pull if missing
ollama pull bge-m3

# Generate embeddings for all active chunks
python manage.py embed_docs --model bge-m3:latest
```

Run `embed_docs` once after `import_academy_docs` and again after any bulk update to the documentation source files. To force a full refresh of existing embeddings:

```bash
python manage.py embed_docs --force
```

## Documentation Chat Falls Back To Keyword Search At Runtime

Even with embeddings present, the assistant falls back to keyword search if Ollama is unreachable at query time (used to embed the user's question for comparison).

Check that Ollama is running and the base URL in the provider config is correct:

```bash
curl http://<ollama-base-url>/api/tags
```

If this returns a list of models, Ollama is reachable. If it times out or refuses the connection, start Ollama or correct the URL in the provider config.
