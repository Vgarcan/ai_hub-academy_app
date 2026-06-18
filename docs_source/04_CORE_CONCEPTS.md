# Core Concepts

AI Hub is built around a small set of reusable concepts. These concepts should
remain domain-neutral so the app can move from one project to another.

The easiest mental model is:

```text
Providers give access to models.
Models power agents.
Agents use knowledge and tools.
Agents run through either Orchestrator or GAME.
Execution sessions record what happened.
The host app owns domain-specific persistence.
```

## Provider

A provider is an AI service account, endpoint or local model server.

Examples:

- OpenAI.
- Ollama.
- Anthropic.
- DeepSeek.
- A private OpenAI-compatible endpoint.
- A custom internal inference service.

Providers answer the question:

```text
Where will model calls be sent?
```

Provider records should store operational configuration, not secrets. Store the
name of an environment variable in the database and keep the actual API key in
the environment.

## Model

A model is a concrete model choice inside a provider.

Models answer the question:

```text
Which model should this agent use?
```

Useful model configuration includes:

- Exact provider model name.
- Friendly display name.
- Default temperature.
- Default max tokens.
- Tool support.
- Active/inactive status.

Use low temperature for structured outputs, contracts and final JSON. Use higher
temperature only when the agent's job benefits from creative variation.

## Agent

An agent is a focused AI worker.

Agents answer the question:

```text
Who is doing this part of the work?
```

An agent combines:

- A model.
- A role.
- A system prompt.
- Optional knowledge collections.
- Optional tools.
- Optional input and output contracts.
- Runtime settings.

Good agents are specific. Avoid creating a generic "do everything" agent when a
short chain of focused agents would be easier to debug.

Examples:

```text
Source Extractor
Risk Classifier
JSON Repair Agent
Report Writer
GAME Planner
Support Reply Drafting Agent
```

## Knowledge

Knowledge is reusable curated context.

Knowledge answers the question:

```text
What should the agent know before it answers?
```

AI Hub stores knowledge as collections and documents. A collection groups related
documents. A document stores curated text, optional source files, tags, language
and notes.

Knowledge is not the same as user input. User input belongs to the execution
context. Knowledge is stable context that can be reused by many agents and
sessions.

## Tool

A tool is a capability that an agent may use.

Tools answer the question:

```text
What can this agent do besides produce text?
```

Examples:

- Search a curated database.
- Fetch a document.
- Call an HTTP endpoint.
- Run a restricted internal function.
- Transform a file.

Tool use depends on several gates:

- The tool must exist and be active.
- The agent must have the tool attached.
- The runtime must support that tool kind.
- The selected model/provider must be compatible with the tool strategy.
- The host project must allow the side effect.

Tools remain associated with agents, not workspaces. A GAME agent and an
Orchestrator agent can both use tools when the runtime allows it.

## Contract

A contract is a JSON-schema-like rule set used to validate payloads.

Contracts answer the question:

```text
What shape should this input or output have?
```

Example:

```json
{
  "required": ["source", "goal"],
  "properties": {
    "source": {"type": "string"},
    "goal": {"type": "string"}
  }
}
```

Contracts make AI failures visible. Instead of silently accepting incomplete
output, AI Hub can fail the step with a useful error.

Recommended contract rules:

- Keep required fields explicit.
- Use stable key names.
- Prefer small nested structures over huge ambiguous blobs.
- Give final output agents enough token budget.
- Add repair/retry behavior in the host adapter only when it is safe.

## Mapping

Mappings move data between the shared execution context and an agent payload.

Mappings answer the question:

```text
What does this step receive, and where does its result go?
```

`input_mapping` builds the input payload for an agent.

`output_mapping` writes selected output values back into session context.

Example workflow:

```text
initial_context.source_text
    -> extractor input
extractor.response.entities
    -> final_context.extracted_entities
```

Mappings are a major reason Orchestrator workflows are reusable. They let a
pipeline define how agents cooperate without hardcoding domain behavior into the
agent model.

## Orchestrator

The Orchestrator workspace is for fixed workflows where the steps are known in
advance.

Orchestrator answers the question:

```text
What sequence of agents should run?
```

Example:

```text
input -> normalize -> extract -> analyze -> write -> final output
```

Use Orchestrator when:

- The process has predictable steps.
- Each agent has a specific responsibility.
- You want repeatable behavior.
- You want simple operational debugging.
- You need to map outputs from one step to the next.

For v1, do not put a full GAME loop inside an Orchestrator pipeline. Keep the
mental models separate unless a future design explicitly supports hybrid runs.

## GAME

GAME is for autonomous goal sessions.

GAME answers the question:

```text
How should one agent decide what to do next until the goal is finished or stopped?
```

In a GAME session, an entry agent receives:

- The goal.
- Current iteration.
- Memory.
- Observations.
- Available actions.
- Runtime limits.
- A strict response contract.

The agent can decide to continue, use an allowed action, wait, finish or stop.

Use GAME when:

- The exact steps are not known in advance.
- The agent must inspect state before deciding.
- The process benefits from memory.
- The session needs a stop contract.
- A human may continue or inspect the session later.

GAME is not automatically better than Orchestrator. It is more flexible, and
therefore requires clearer limits.

## Execution Session

An execution session is the top-level runtime record.

Sessions answer the question:

```text
What happened during this AI run?
```

Sessions store:

- Runtime kind.
- Runtime mode.
- Status.
- Pipeline or entry agent.
- Goal text.
- Initial context.
- Final context.
- Error detail.
- Timing.

The host project should link its own domain records to `ExecutionSession` instead
of duplicating the AI runtime.

## Execution Step Run

A step run is one observable action inside a session.

Step runs answer the question:

```text
What did this agent or tool do at this point in the execution?
```

Step runs store:

- Agent.
- Pipeline step or action name.
- Request payload.
- Response payload.
- Observation payload.
- Status.
- Error detail.
- Latency.

This is the audit trail. When something fails, start with the session and then
inspect the step runs.

## Host Adapter

A host adapter is project-specific glue code outside AI Hub.

The adapter answers the question:

```text
How does this project turn its own domain object into an AI Hub session and then store the result?
```

The adapter owns:

- Domain input.
- Language rules.
- UI status messages.
- Domain-specific output persistence.
- Recovery behavior for partial model output.
- Links between domain records and AI Hub sessions.

Examples:

```text
Ticket -> AI Hub session -> TriageResult
Document -> AI Hub session -> ExtractedReport
Invoice -> AI Hub session -> InvoiceReview
```

The reusable app should not import the host app.

## Workspace Separation

AI Hub has two workspaces but one shared foundation.

Shared foundation:

- Providers.
- Models.
- Agents.
- Knowledge.
- Tools.

Workspace-specific runtime:

- Orchestrator uses pipelines and ordered steps.
- GAME uses goal sessions and decision loops.

This separation keeps the admin understandable for non-technical users while
keeping the technical model compact.
