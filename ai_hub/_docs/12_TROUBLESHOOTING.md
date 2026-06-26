# Troubleshooting

## Admin CSS Looks Broken

For `DEBUG=False`, collect and serve static files.

```bash
python manage.py collectstatic
```

Also confirm your web server serves Django admin static files.

If the browser still shows old styles, hard refresh the page. Admin assets often
use cache-busting query strings such as `guided-v8` or `missiondeck-v10`.

If a development server keeps serving older Admin behavior after the files change,
restart `runserver`. Django usually reloads templates and Python modules, but a
stale autoreloader process can still leave you looking at an older in-memory
version.

## A Model Disappeared From The Admin Sidebar

Supporting tables — bridge/link records, structural children and runtime/audit
records — are intentionally **demoted from the admin index and sidebar** via
`AIHubHideFromIndexMixin`. They are still registered and their URLs still work;
they are just not listed. This affects, among others: knowledge document chunks,
toolbox↔tool and agent↔toolbox/grant links, pipeline steps, goal
dependencies/plans/plan-steps, workspace actions/agents, memory entries, and the
runtime audit records (execution step runs, tool execution runs, GAME action
runs, delegation runs).

Where to find them:

- Open the AI Hub home (`/admin/ai_hub/`) and use the **"Show supporting tables"**
  toggle on the *All records* panel. It lists each hidden model, grouped by
  category, with a one-line reason and a direct link to its changelist.
- Or manage them from their parent page (e.g. pipeline steps inside the Pipeline
  Designer, document chunks inside their document).

This is by design, not a permissions bug. If you genuinely need a record promoted
back to the sidebar, remove `AIHubHideFromIndexMixin` from that model's admin.

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

## Training Provider Model Is Rejected On Save

The `training` provider is a deterministic stub. Its model name must be exactly `training` or start with `training/` (for example `training/assistant`). Saving a training-provider model with any other name fails with:

```text
Training-provider models must be named 'training' or start with 'training/'
```

This is intentional: any other name would not be routed to the stub and would fail at runtime against the real client. Rename the model to follow the convention.

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

## GAME Feature Is Disabled

A GAME service can raise:

```text
GAME feature 'AI_HUB_GAME_..._ENABLED' is disabled. Set ... =True in Django settings to enable it.
```

This means the matching feature flag is off (the reusable default is fail-closed). Enable the flag in settings or the environment — see `03_CONFIGURATION.md`. Remember flags gate the service layer only; the admin add/change forms write directly to the model.

## Goal Is Stuck In Running

A goal stays in `running` if its session never reached a terminal state (an interrupted run, or stub sessions left by tests). Cancel orphaned goals — those with no active session — with:

```bash
python manage.py cleanup_orphaned_goals --dry-run
python manage.py cleanup_orphaned_goals
```

A goal with an active session (pending/running/waiting_async) is never touched.

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

## Control Center Attention Inbox Looks Wrong

The **Needs attention** inbox should show enriched attention items, not the older
plain warning list. Expected behavior:

- `Open` count matches visible unsilenced/unarchived incidents.
- Filtering by `Errors` hides warnings when there are no error incidents.
- Sorting by newest, relevance or severity reorders the enriched incident rows.
- `Archived` shows only locally archived incidents.
- Hovering a row shows more detail.
- `Open incident` links to the relevant Admin record when available.

If filters or archive buttons do nothing:

1. Confirm the page loaded the latest `missiondeck-v...` cache-busting value.
2. Hard refresh the browser.
3. Restart `runserver` if local development still shows stale counts.
4. Inspect the rendered row markup. Current rows include `data-mc-attn-item`.
   Legacy rows do not, and should be hidden.

Archive and silence are intentionally local browser state. They are stored in
`localStorage` under:

```text
aiHubMissionDeckAttention:v1
```

Clearing site data removes archived/silenced state and makes all current
incidents visible again. This never deletes the underlying provider, model,
agent, knowledge, pipeline or execution records.
