# Troubleshooting

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
