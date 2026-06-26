"""
Seed the database with training provider, models, agents, knowledge, and tutorial missions.

Run once after migrate to set up the demo environment:
    python manage.py seed_academy_training_data
    python manage.py seed_academy_training_data --force-update
"""
from django.core.management.base import BaseCommand

from ai_hub.models import (
    AgentProfile,
    KnowledgeCollection,
    ModelConfig,
    ProviderConfig,
)


def _get_or_create_provider():
    provider, created = ProviderConfig.objects.get_or_create(
        name="Academy Training Provider",
        defaults={
            "provider_type": "training",
            "base_url": "",
            "api_key_env_var": "",
            "default_timeout": 30,
            "is_active": True,
        },
    )
    if created:
        print("  Created: Academy Training Provider")
    return provider


def _get_or_create_model(provider):
    model, created = ModelConfig.objects.get_or_create(
        provider=provider,
        model_name="training/assistant",
        defaults={
            "temperature_default": "0.20",
            "max_tokens_default": 2000,
            "supports_tools": False,
            "is_active": True,
        },
    )
    if created:
        print("  Created: Academy Training Model (training/assistant)")
    return model


def _get_or_create_doc_assistant(model):
    agent, created = AgentProfile.objects.get_or_create(
        name="AI Hub Documentation Assistant",
        defaults={
            "role": "Documentation answering agent",
            "system_prompt": (
                "You are the AI Hub Documentation Assistant.\n\n"
                "Answer questions about AI Hub using only the provided documentation context.\n\n"
                "Rules:\n"
                "1. Do not invent undocumented features.\n"
                "2. If the documentation does not contain the answer, say so clearly.\n"
                "3. Explain concepts with practical Django examples when useful.\n"
                "4. Keep answers clear for users learning the platform.\n"
                "5. Mention the source page titles used.\n"
                "6. Return your final answer as a GAME complete=true response."
            ),
            "model_config": model,
            "is_active": True,
        },
    )
    if created:
        print("  Created: AI Hub Documentation Assistant")
    return agent


def _get_or_create_normalizer(model):
    agent, created = AgentProfile.objects.get_or_create(
        name="Input Normalizer",
        defaults={
            "role": "Normalise and validate incoming data",
            "system_prompt": (
                "You are an Input Normalizer. Your job is to clean and validate incoming data "
                "before it reaches downstream agents.\n\n"
                "For a support ticket:\n"
                "- Remove HTML tags\n"
                "- Trim whitespace\n"
                "- Extract the ticket title and body\n\n"
                "Return a JSON object with: normalized_title, normalized_body, word_count."
            ),
            "model_config": model,
            "input_contract": {"required": ["ticket_title", "ticket_text"]},
            # Agent contracts validate the agent's wrapper payload {agent, llm, tools},
            # not the model's business output. The business keys (normalized_*) are
            # surfaced via the pipeline step output_mapping + the pipeline's
            # global_output_contract — see the "Make The Agent Specific" mission.
            "output_contract": {"required": ["agent", "llm", "tools"]},
            "is_active": True,
        },
    )
    if created:
        print("  Created: Input Normalizer")
    else:
        _repair_agent_output_contract(agent, {"required": ["agent", "llm", "tools"]})
    return agent


# Agents seeded before the contract fix carried business keys in their
# output_contract (e.g. normalized_title), which validate against the wrapper
# payload {agent, llm, tools} and therefore always fail. Repair existing agents
# in place so re-running the seed unbreaks the golden path.
def _repair_agent_output_contract(agent, correct_contract):
    if agent.output_contract != correct_contract:
        agent.output_contract = correct_contract
        agent.save(update_fields=["output_contract"])
        print(f"  Repaired output_contract: {agent.name}")


def _get_or_create_classifier(model):
    agent, created = AgentProfile.objects.get_or_create(
        name="Ticket Classifier",
        defaults={
            "role": "Classify support tickets by category and priority",
            "system_prompt": (
                "You are a Support Ticket Classifier.\n\n"
                "Given a support ticket, classify it into:\n"
                "- category: one of Technical Issue, Billing, Feature Request, General Enquiry, Bug Report\n"
                "- priority: one of Critical, High, Medium, Low\n"
                "- reason: a one-sentence explanation of your classification\n\n"
                "Return a JSON object with exactly these three keys."
            ),
            "model_config": model,
            "input_contract": {"required": ["ticket_title", "ticket_text"]},
            # See Input Normalizer: agent output_contract describes the wrapper
            # payload. Business keys (category/priority/reason) are declared on the
            # pipeline's global_output_contract, not here.
            "output_contract": {"required": ["agent", "llm", "tools"]},
            "is_active": True,
        },
    )
    if created:
        print("  Created: Ticket Classifier")
    else:
        _repair_agent_output_contract(agent, {"required": ["agent", "llm", "tools"]})
    return agent


def _get_or_create_knowledge():
    collection, created = KnowledgeCollection.objects.get_or_create(
        name="AI Hub Official Docs",
        defaults={
            "description": "Official AI Hub documentation imported from Markdown source files.",
            "is_active": True,
        },
    )
    if created:
        print("  Created: AI Hub Official Docs knowledge collection")
    return collection


def _seed_tutorial_missions(force_update=False):
    from academy.models import TutorialMission, TutorialModule

    # Migrate the legacy slug so existing progress is preserved and no orphan
    # mission is left behind when this mission is re-seeded under its new slug.
    TutorialMission.objects.filter(
        slug="inspect-the-mission-deck"
    ).exclude(
        slug="inspect-the-control-center"
    ).update(slug="inspect-the-control-center")

    modules_data = [
        {
            "order": 0,
            "title": "Orientation",
            "slug": "orientation",
            "description": "Get familiar with the AI Hub interface.",
            "missions": [
                {
                    "order": 1,
                    "title": "Enter The Control Room",
                    "slug": "enter-the-control-room",
                    "goal": "Visit AI Hub Home, Control Center, Orchestrator Workspace and GAME Workspace.",
                    "validation_key": "visited_control_room",
                    "instructions": (
                        "## Welcome to AI Hub!\n\n"
                        "Your first mission is to explore the AI Hub Admin interface.\n\n"
                        "### Steps\n\n"
                        "1. Open [AI Hub Admin](/admin/ai_hub/)\n"
                        "2. Click **Workspaces → Orchestrator** to see the pipeline control center\n"
                        "3. Click **Workspaces → GAME** to see the autonomous agent workspace\n"
                        "4. Return here and click **Check Mission**\n\n"
                        "### What you will find\n\n"
                        "- The **Orchestrator** workspace manages fixed sequential pipelines\n"
                        "- The **GAME** workspace manages autonomous goal-driven sessions\n"
                        "- Both workspaces share the same providers, models and agents\n"
                    ),
                },
            ],
        },
        {
            "order": 1,
            "title": "Providers and Models",
            "slug": "providers-and-models",
            "description": "Set up the AI service accounts and model configurations.",
            "missions": [
                {
                    "order": 1,
                    "title": "Power Up The Lab",
                    "slug": "power-up-the-lab",
                    "goal": "Create an active ProviderConfig named 'Academy Training Provider'.",
                    "validation_key": "created_training_provider",
                    "instructions": (
                        "## Create Your First Provider\n\n"
                        "A **Provider** is an AI service account — OpenAI, Anthropic, Ollama, or Training.\n\n"
                        "### Steps\n\n"
                        "1. Go to [Admin → AI Hub → Provider configs](/admin/ai_hub/providerconfig/add/)\n"
                        "2. Set **Name** to `Academy Training Provider`\n"
                        "3. Set **Provider type** to `Training (stub)`\n"
                        "4. Leave **API key env var** empty (training providers don't need keys)\n"
                        "5. Set **Is active** to checked\n"
                        "6. Save\n\n"
                        "![The Add Provider config form: Name, Provider type, Is active, Base url, "
                        "Default timeout and the Api key env var field]"
                        "(/static/academy/img/tutorials/provider-form.png)\n\n"
                        "### Why Training Mode?\n\n"
                        "The Training provider returns deterministic responses without calling any external API. "
                        "This lets you complete all missions without spending API credits.\n"
                    ),
                },
                {
                    "order": 2,
                    "title": "Install Your First Engine",
                    "slug": "install-your-first-engine",
                    "goal": "Create an active ModelConfig connected to Academy Training Provider.",
                    "validation_key": "created_training_model",
                    "instructions": (
                        "## Create Your First Model\n\n"
                        "A **Model** is a specific AI model tied to a provider.\n\n"
                        "### Steps\n\n"
                        "1. Go to [Admin → AI Hub → Model configs](/admin/ai_hub/modelconfig/add/)\n"
                        "2. Set **Provider** to `Academy Training Provider`\n"
                        "3. Set **Model name** to `training/assistant`\n"
                        "4. Set **Temperature default** to `0.2`\n"
                        "5. Set **Max tokens default** to `2000`\n"
                        "6. Set **Is active** to checked\n"
                        "7. Save\n\n"
                        "![The Add Model config form: Provider, Model name, Temperature default, "
                        "Max tokens default and Is active]"
                        "(/static/academy/img/tutorials/model-form.png)\n\n"
                        "### Notes\n\n"
                        "The model name `training/assistant` is intercepted by the training stub "
                        "and returns deterministic responses.\n"
                    ),
                },
            ],
        },
        {
            "order": 2,
            "title": "Agents",
            "slug": "agents",
            "description": "Create and configure AI agents with clear roles.",
            "missions": [
                {
                    "order": 1,
                    "title": "Create Your First Worker",
                    "slug": "create-your-first-worker",
                    "goal": "Create an active AgentProfile named 'Input Normalizer'.",
                    "validation_key": "created_first_agent",
                    "instructions": (
                        "## Create Your First Agent\n\n"
                        "An **Agent Profile** is a named AI worker with a role, system prompt and model.\n\n"
                        "### Steps\n\n"
                        "1. Go to [Admin → AI Hub → Agent profiles](/admin/ai_hub/agentprofile/add/)\n"
                        "2. Set **Name** to `Input Normalizer`\n"
                        "3. Set **Role** to `Normalise and validate incoming data`\n"
                        "4. Set **Model config** to your `Academy Training Model`\n"
                        "5. Write a system prompt explaining what the agent should do\n"
                        "6. Set **Is active** to checked\n"
                        "7. Save\n\n"
                        "![The Add Agent profile form: Agent identity (Name, Role, Is active), "
                        "Model config and the System prompt field]"
                        "(/static/academy/img/tutorials/agent-form.png)\n\n"
                        "### Good agent names are concrete\n\n"
                        "- ✅ Input Normalizer\n"
                        "- ✅ Ticket Classifier\n"
                        "- ✅ Evidence Extractor\n"
                        "- ❌ Agent 1\n"
                        "- ❌ My AI\n"
                    ),
                },
                {
                    "order": 2,
                    "title": "Make The Agent Specific",
                    "slug": "make-the-agent-specific",
                    "goal": "Add input and output contracts to your Input Normalizer agent.",
                    "validation_key": "added_agent_contract",
                    "instructions": (
                        "## Add Contracts\n\n"
                        "**Contracts** define what data an agent expects as input and what shape its "
                        "result payload must have.\n\n"
                        "### Steps\n\n"
                        "1. Open your `Input Normalizer` agent in Admin\n"
                        "2. Set **Input contract** to:\n"
                        "   ```json\n"
                        '   {"required": ["ticket_title", "ticket_text"]}\n'
                        "   ```\n"
                        "3. Set **Output contract** to:\n"
                        "   ```json\n"
                        '   {"required": ["agent", "llm", "tools"]}\n'
                        "   ```\n"
                        "4. Save\n\n"
                        "![The Contracts section of the agent form, with the Input contract and "
                        "Output contract fields]"
                        "(/static/academy/img/tutorials/agent-contracts.png)\n\n"
                        "### Input vs output contracts — read this carefully\n\n"
                        "- The **input contract** checks the data going *into* the agent. If a pipeline "
                        "step hands this agent a payload without `ticket_title` or `ticket_text`, the "
                        "session fails immediately with a clear message.\n"
                        "- The **output contract** checks the agent's *result wrapper*, which always has "
                        "the keys `agent`, `llm` and `tools`. The model's own answer lives **inside** "
                        "`llm.content` as JSON.\n"
                        "- So the agent's business output (like `normalized_title`) is **not** declared "
                        "here. You expose it later, at the pipeline level, using a step **output mapping** "
                        "and the pipeline **global output contract** — you will do exactly that in the "
                        "Orchestrator module.\n\n"
                        "### Why contracts matter\n\n"
                        "- Input contracts enforce data quality at every pipeline step\n"
                        "- A session fails immediately if a required input field is missing\n"
                        "- This prevents silent failures deep in multi-agent pipelines\n"
                    ),
                },
            ],
        },
        {
            "order": 3,
            "title": "Knowledge",
            "slug": "knowledge",
            "description": "Give agents access to curated knowledge.",
            "missions": [
                {
                    "order": 1,
                    "title": "Give The Agent A Manual",
                    "slug": "give-the-agent-a-manual",
                    "goal": "Create a KnowledgeCollection with at least one active document.",
                    "validation_key": "created_knowledge_collection",
                    "instructions": (
                        "## Create a Knowledge Collection\n\n"
                        "A **Knowledge Collection** is a curated set of documents injected into the agent's context.\n\n"
                        "### Steps\n\n"
                        "1. Go to [Admin → AI Hub → Knowledge collections](/admin/ai_hub/knowledgecollection/add/)\n"
                        "2. Create a collection named `AI Hub Official Docs`\n"
                        "3. Add a Knowledge Document with curated text about AI Hub concepts\n"
                        "4. Set the document status to `Active`\n"
                        "5. Save\n\n"
                        "![The Add Knowledge collection form: Name, Description and Is active]"
                        "(/static/academy/img/tutorials/knowledge-form.png)\n\n"
                        "### Tip\n\n"
                        "The Documentation section of this Academy already has the official docs imported. "
                        "You can copy relevant text from there into a knowledge document.\n"
                    ),
                },
            ],
        },
        {
            "order": 4,
            "title": "Orchestrator",
            "slug": "orchestrator",
            "description": "Build and run fixed sequential pipelines.",
            "missions": [
                {
                    "order": 1,
                    "title": "Build Your First Conveyor Belt",
                    "slug": "build-your-first-conveyor-belt",
                    "goal": "Create an active pipeline with at least 2 steps.",
                    "validation_key": "created_orchestrator_pipeline",
                    "instructions": (
                        "## Create a Pipeline\n\n"
                        "A **Pipeline** is a sequence of agent steps that run in order.\n\n"
                        "### Steps\n\n"
                        "1. Go to [Admin → AI Hub → Pipeline definitions](/admin/ai_hub/pipelinedefinition/add/)\n"
                        "2. Name it `Ticket Triage Pipeline`\n"
                        "3. Save (don't activate yet)\n"
                        "4. Add **Step 1**: agent = `Input Normalizer`, order = 1\n"
                        "5. Add **Step 2**: agent = `Ticket Classifier`, order = 2\n"
                        "6. Now activate the pipeline and save\n\n"
                        "### Input mapping\n\n"
                        "Leave input/output mappings empty for now. When empty, AI Hub passes the full "
                        "context from one step to the next.\n"
                    ),
                },
                {
                    "order": 2,
                    "title": "Run The Pipeline",
                    "slug": "run-the-pipeline",
                    "goal": "Create and run an ExecutionSession that completes successfully.",
                    "validation_key": "ran_successful_execution_session",
                    "instructions": (
                        "## Run Your Pipeline\n\n"
                        "An **Execution Session** is a single run of a pipeline with specific input data.\n\n"
                        "### Steps\n\n"
                        "1. Go to [Admin → AI Hub → Execution sessions](/admin/ai_hub/executionsession/add/)\n"
                        "2. Set **Pipeline** to `Ticket Triage Pipeline`\n"
                        "3. Set **Runtime kind** to `Orchestrator`\n"
                        "4. Set **Initial context** to:\n"
                        "   ```json\n"
                        '   {"ticket_title": "Login broken", "ticket_text": "I cannot log in since the update."}\n'
                        "   ```\n"
                        "5. Save, then click **Run Session** from the session detail page\n"
                        "6. Refresh and check the status is `success`\n"
                    ),
                },
            ],
        },
        {
            "order": 5,
            "title": "GAME Workspace",
            "slug": "game-workspace",
            "description": "Run autonomous goal-driven AI sessions.",
            "missions": [
                {
                    "order": 1,
                    "title": "Launch An Autonomous Session",
                    "slug": "launch-an-autonomous-session",
                    "goal": "Create a GAME session with a goal and entry agent.",
                    "validation_key": "created_game_goal",
                    "instructions": (
                        "## The GAME Workspace\n\n"
                        "**GAME** (Goal-Agent-Memory-Execute) is AI Hub's autonomous execution mode. "
                        "Instead of a fixed pipeline, one agent runs in a loop until it achieves a goal.\n\n"
                        "### Steps\n\n"
                        "1. Go to [Admin → AI Hub → GAME Workspace](/admin/ai_hub/workspaces/game/)\n"
                        "2. Create a new GAME session\n"
                        "3. Set **Runtime kind** to `GAME`\n"
                        "4. Set **Entry agent** to `AI Hub Documentation Assistant`\n"
                        "5. Set **Goal text** to: `Explain what AI Hub is in 3 bullet points`\n"
                        "6. Set **Runtime config** to: `{\"max_iterations\": 3}`\n"
                        "7. Save and run\n"
                    ),
                },
                {
                    "order": 2,
                    "title": "Finish The Goal",
                    "slug": "finish-the-goal",
                    "goal": "Obtain a successful GAME session with a final_answer in the context.",
                    "validation_key": "created_game_session",
                    "instructions": (
                        "## Complete a GAME Session\n\n"
                        "A GAME session ends when the agent sets `complete: true` in its response.\n\n"
                        "### Steps\n\n"
                        "1. Run your GAME session from Mission 7.1\n"
                        "2. Check that the status is `success`\n"
                        "3. Open the session and expand **Final context**\n"
                        "4. Verify that `final_answer` contains text\n\n"
                        "### How GAME works\n\n"
                        "Each iteration, the agent receives:\n"
                        "- The original goal\n"
                        "- Memory from previous iterations\n"
                        "- Available actions (think, finish)\n\n"
                        "The session ends when the agent returns `complete: true`.\n"
                    ),
                },
            ],
        },
        {
            "order": 6,
            "title": "Execution Timeline",
            "slug": "execution-timeline",
            "description": "Learn to read and interpret AI execution telemetry.",
            "missions": [
                {
                    "order": 1,
                    "title": "Read The Black Box",
                    "slug": "read-the-black-box",
                    "goal": "Inspect the step runs of a completed ExecutionSession.",
                    "validation_key": "inspected_execution_timeline",
                    "instructions": (
                        "## Reading Execution Telemetry\n\n"
                        "Every AI Hub run leaves a complete audit trail.\n\n"
                        "### Steps\n\n"
                        "1. Open [Admin → AI Hub → Execution sessions](/admin/ai_hub/executionsession/)\n"
                        "2. Click on a successful session to open the **Session Explorer**\n"
                        "3. Open the **Timeline** tab to see each step run\n"
                        "4. Expand a step run and read:\n"
                        "   - **Request payload**: what was sent to the model\n"
                        "   - **Response payload**: what the model returned\n"
                        "   - **Latency ms**: how long the call took\n"
                        "5. Return here and click **Check Mission**\n\n"
                        "### Why this matters\n\n"
                        "Full telemetry at every step lets you:\n"
                        "- Debug unexpected outputs\n"
                        "- Audit AI decisions for compliance\n"
                        "- Measure cost and latency per step\n"
                        "- Reproduce exact conditions for a run\n"
                    ),
                },
            ],
        },
        {
            "order": 7,
            "title": "Host Integration",
            "slug": "host-integration",
            "description": "Connect a domain object to an AI session.",
            "missions": [
                {
                    "order": 1,
                    "title": "Connect A Host Object",
                    "slug": "connect-a-host-object",
                    "goal": "Run triage on a SupportTicket so it has an ai_session.",
                    "validation_key": "connected_host_object",
                    "instructions": (
                        "## Connect Real Objects to AI Hub\n\n"
                        "AI Hub is designed to integrate with your domain models. "
                        "The `support_demo` app shows this pattern with support tickets.\n\n"
                        "### Steps\n\n"
                        "1. Open [Admin → Support Demo → Support tickets](/admin/support_demo/supportticket/)\n"
                        "2. Tick a ticket's checkbox, then choose the **Run AI triage on selected tickets** action and click **Go**\n"
                        "3. Refresh and check that the ticket now shows an AI session link\n"
                        "4. Click the session link to see the full execution audit trail\n\n"
                        "![The Support tickets list with a ticket checked and the Run AI triage on "
                        "selected tickets action selected, next to the Run button]"
                        "(/static/academy/img/tutorials/triage-action.png)\n\n"
                        "### The adapter pattern\n\n"
                        "The `support_demo/services/ai_hub_adapter.py` file shows the clean integration:\n"
                        "- Create an ExecutionSession with domain data as `initial_context`\n"
                        "- Run the session\n"
                        "- Read the `final_context` to extract structured output\n"
                        "- Save the output back to domain models\n"
                    ),
                },
            ],
        },
        {
            "order": 8,
            "title": "Capstone",
            "slug": "capstone",
            "description": "Build the complete ticket triage workflow end-to-end.",
            "missions": [
                {
                    "order": 1,
                    "title": "Complete Workflow",
                    "slug": "complete-workflow",
                    "goal": "Run a full ticket triage with provider, model, agents, pipeline and execution all working together.",
                    "validation_key": "completed_capstone",
                    "instructions": (
                        "## Capstone: Full Ticket Triage Workflow\n\n"
                        "This mission validates that your complete AI Hub setup works end-to-end.\n\n"
                        "### Checklist\n\n"
                        "- [ ] Academy Training Provider is active\n"
                        "- [ ] Academy Training Model is active\n"
                        "- [ ] Input Normalizer agent is active with contracts\n"
                        "- [ ] Ticket Classifier agent is active with contracts\n"
                        "- [ ] Ticket Triage Pipeline is active with 2 steps\n"
                        "- [ ] At least one SupportTicket exists in Support Demo\n"
                        "- [ ] Triage action has been run on the ticket\n"
                        "- [ ] Ticket shows a successful AI session\n\n"
                        "### Steps\n\n"
                        "1. Go to [Support Demo → Tickets](/admin/support_demo/supportticket/)\n"
                        "2. Create a ticket (or use the seeded demo tickets)\n"
                        "3. Tick its checkbox, choose **Run AI triage on selected tickets** and click **Go**\n"
                        "4. Verify the session completed with `success`\n"
                        "5. Click **Check Mission**\n"
                    ),
                },
            ],
        },
    ]

    interface_mission = {
        "order": 2,
        "title": "Inspect The Control Center",
        "slug": "inspect-the-control-center",
        "goal": "Use the connection graph, node pop-up, full screen mode and Needs attention inbox.",
        "validation_key": "visited_control_room",
        "instructions": (
            "## Inspect the Control Center\n\n"
            "The Control Center is designed for answering one question quickly: "
            "`what is connected, what is healthy, and where do I click next?`\n\n"
            "### Steps\n\n"
            "1. Open the [Control Center](/admin/ai_hub/pipelinedefinition/control-center/)\n"
            "2. In **Connection graph**, hover over a node and read the compact status preview\n"
            "3. Select a node and use the draggable detail pop-up\n"
            "4. Click **Open record in admin** from the pop-up\n"
            "5. Turn **Isolate** on only after selecting a node; it hides nodes outside the selected hop range\n"
            "6. Use **Full screen**, then exit with the same button or `Esc`\n"
            "7. In **Needs attention**, try the view tabs, severity/source filter and sort menu\n\n"
            "![The Control Center: connection graph on top, the All pipelines menu and "
            "Full screen / Isolate controls, and the Needs attention inbox below]"
            "(/static/academy/img/tutorials/control-center.png)\n\n"
            "### Operator rules\n\n"
            "- Hover gives brief information only; it should not hide the rest of the graph\n"
            "- Selection opens the detail pop-up and can drive hop isolation\n"
            "- **Open record in admin** is the fastest path from a visual signal to the real object\n"
            "- Archive, restore and **Silence 24h** are local browser controls; they do not delete database records\n"
            "- The Pipeline and Step graph columns appear only when pipeline and step objects exist\n"
        ),
    }

    mission_overrides = {
        "enter-the-control-room": {
            "instructions": (
                "## Welcome to AI Hub!\n\n"
                "Your first mission is to learn the shape of the AI Hub Admin cockpit.\n\n"
                "### Steps\n\n"
                "1. Open [AI Hub Admin](/admin/ai_hub/) — the cockpit home, organised into five areas "
                "(Overview & Entry, Foundation, Orchestrator, GAME, Operations)\n\n"
                "![AI Hub cockpit home: the vitals strip, the inventory line, the Needs your "
                "attention panel and the five-area navigation map]"
                "(/static/academy/img/tutorials/home-cockpit.png)\n\n"
                "2. Open the [Control Center](/admin/ai_hub/pipelinedefinition/control-center/)\n"
                "3. Open the [Build Console](/admin/ai_hub/workspaces/build/) — the guided wizard that creates a whole setup at once\n\n"
                "![Build Console: the numbered wizard steps (Engine, Agent, Goal, Governance, "
                "Runtime, Review) that create a whole setup in one transaction]"
                "(/static/academy/img/tutorials/build-console.png)\n\n"
                "4. Open [Orchestrator Workspace](/admin/ai_hub/workspaces/orchestrator/)\n"
                "5. Open [GAME Workspace](/admin/ai_hub/workspaces/game/)\n"
                "6. Open the [Operations Inbox](/admin/ai_hub/operations/) — the queue for everything that needs a human\n"
                "7. Return here and click **Check Mission**\n\n"
                "### What you will find\n\n"
                "- The **cockpit home** shows live vitals, what needs a human, and the five-area navigation map\n"
                "- The **Control Center** is the deep-diagnostics surface: system health, the connection graph and attention items\n"
                "- The **Build Console** creates provider → model → agent → workspace in one atomic step\n"
                "- The **Orchestrator** workspace manages fixed sequential pipelines\n"
                "- The **GAME** workspace manages autonomous goal-driven sessions\n"
                "- The **Operations Inbox** collects approvals, paused sessions, failures and blocked goals in one place\n"
                "- Both runtimes share the same providers, models, agents, knowledge and tools\n"
            ),
        },
        "build-your-first-conveyor-belt": {
            "instructions": (
                "## Create a Pipeline\n\n"
                "A **Pipeline** is a sequence of agent steps that run in order. This is where the "
                "business output you could not put on the agent (back in *Make The Agent Specific*) "
                "is finally surfaced.\n\n"
                "### Steps\n\n"
                "1. Go to [Admin - AI Hub - Pipeline definitions](/admin/ai_hub/pipelinedefinition/add/)\n"
                "2. Name it `Ticket Triage Pipeline`\n"
                "3. Save it as draft while you add steps\n"
                "4. Add **Step 1**: agent = `Input Normalizer`, order = 1. In its **Output mapping** set:\n"
                "   ```json\n"
                '   {"final_output": "llm.content"}\n'
                "   ```\n"
                "5. Add **Step 2**: agent = `Ticket Classifier`, order = 2, with the same **Output mapping**:\n"
                "   ```json\n"
                '   {"final_output": "llm.content"}\n'
                "   ```\n"
                "6. Set the pipeline **Global output contract** to:\n"
                "   ```json\n"
                '   {"required": ["category", "priority"]}\n'
                "   ```\n"
                "7. Activate the pipeline and save\n"
                "8. Open the [Control Center](/admin/ai_hub/pipelinedefinition/control-center/) and pick this pipeline in the **All pipelines** menu\n\n"
                "![The Pipeline definition form with its two ordered steps inline and the "
                "global output contract]"
                "(/static/academy/img/tutorials/pipeline-form.png)\n\n"
                "### How the output gets out of the agent\n\n"
                "- Each agent returns a wrapper payload; the model's JSON answer is the string in `llm.content`\n"
                "- The step **output mapping** `{\"final_output\": \"llm.content\"}` copies that JSON into the "
                "run context, where AI Hub merges its keys (`category`, `priority`, `reason`) into the context\n"
                "- The pipeline **global output contract** then checks those merged business keys exist\n"
                "- This is why the agent output contract only described the wrapper, not the business keys\n\n"
                "### Reading the graph\n\n"
                "- Pipeline columns appear only after pipeline records exist\n"
                "- Step columns appear only after step records exist\n"
                "- An active pipeline with no steps appears in **Needs attention** as a warning\n"
                "- Use node selection plus **Isolate** to focus on one pipeline's neighborhood\n"
            ),
        },
        "run-the-pipeline": {
            "instructions": (
                "## Run Your Pipeline\n\n"
                "An **Execution Session** is a single run of a pipeline with specific input data.\n\n"
                "### Steps\n\n"
                "1. Go to [Admin - AI Hub - Execution sessions](/admin/ai_hub/executionsession/add/)\n"
                "2. Set **Pipeline** to `Ticket Triage Pipeline`\n"
                "3. Set **Runtime kind** to `Orchestrator`\n"
                "4. Set **Initial context** to:\n"
                "   ```json\n"
                '   {"ticket_title": "Login broken", "ticket_text": "I cannot log in since the update."}\n'
                "   ```\n"
                "5. Save, then click **Run Session** from the session detail page\n"
                "6. Refresh and check the status is `success`\n"
                "7. Return to Control Center and check whether **Needs attention** changed\n\n"
                "![The Session Explorer Overview showing status SUCCESS, the step and event "
                "counts and zero failed steps]"
                "(/static/academy/img/tutorials/session-success.png)\n\n"
                "### Operator habit\n\n"
                "Use **Open incident** on attention items and **Open record in admin** on graph nodes "
                "to move from dashboard signal to the exact Admin record.\n"
            ),
        },
        "launch-an-autonomous-session": {
            "instructions": (
                "## The GAME Workspace\n\n"
                "**GAME** (Goal-Agent-Memory-Execute) is AI Hub's autonomous execution mode. "
                "Instead of a fixed pipeline, one agent runs in a loop until it achieves a goal.\n\n"
                "### Steps\n\n"
                "1. Go to [Admin - AI Hub - GAME Workspace](/admin/ai_hub/workspaces/game/)\n"
                "2. Create a new GAME session\n"
                "3. Set **Runtime kind** to `GAME`\n"
                "4. Set **Entry agent** to `AI Hub Documentation Assistant`\n"
                "5. Set **Goal text** to: `Explain what AI Hub is in 3 bullet points`\n"
                "6. Set **Runtime config** to: `{\"max_iterations\": 3}`\n"
                "7. Save and run\n"
                "8. Use the GAME graph to inspect the entry agent, model, knowledge and tool links\n\n"
                "![The Create GAME session form: Entry agent, Goal, Max iterations and Runtime fields]"
                "(/static/academy/img/tutorials/game-new.png)\n\n"
                "### Interface note\n\n"
                "The GAME graph uses the same interaction model as the Control Center graph: "
                "hover for a brief preview, select for the draggable detail pop-up and use full screen when the graph needs room.\n"
            ),
        },
        "finish-the-goal": {
            "instructions": (
                "## Complete a GAME Session\n\n"
                "A GAME session ends when the agent sets `complete: true` in its response.\n\n"
                "### Steps\n\n"
                "1. Run your GAME session from the previous mission\n"
                "2. Check that the status is `success`\n"
                "3. Open the session and expand **Final context**\n"
                "4. Verify that `final_answer` contains text\n"
                "5. Open the related records from the GAME graph when you need to verify configuration\n\n"
                "### How GAME works\n\n"
                "Each iteration, the agent receives:\n"
                "- The original goal\n"
                "- Memory from previous iterations\n"
                "- Available actions and tools\n\n"
                "The session ends when the agent returns `complete: true`.\n"
            ),
        },
        "read-the-black-box": {
            "instructions": (
                "## Reading Execution Telemetry\n\n"
                "Every AI Hub run leaves a complete audit trail.\n\n"
                "### Steps\n\n"
                "1. Open [Admin - AI Hub - Execution sessions](/admin/ai_hub/executionsession/)\n"
                "2. Click on a successful or failed session to open the **Session Explorer**\n"
                "3. Open the **Timeline** tab to see each step run in order\n"
                "4. Expand a step run and read:\n"
                "   - **Request payload**: what was sent to the model\n"
                "   - **Response payload**: what the model returned\n"
                "   - **Latency ms**: how long the call took\n"
                "5. Return to Control Center and compare this with **Needs attention**\n\n"
                "![The Timeline tab of the Session Explorer, listing each step run with its "
                "status, action, time and latency, and an Open button]"
                "(/static/academy/img/tutorials/timeline-tab.png)\n\n"
                "### Why this matters\n\n"
                "Failed step runs surface as attention incidents with dates and an **Open incident** action. "
                "Use the incident link to jump to the latest failed step run, then use telemetry to diagnose the cause.\n"
            ),
        },
        "complete-workflow": {
            "instructions": (
                "## Capstone: Full Ticket Triage Workflow\n\n"
                "This mission validates that your complete AI Hub setup works end-to-end.\n\n"
                "### Checklist\n\n"
                "- [ ] Academy Training Provider is active\n"
                "- [ ] Academy Training Model is active\n"
                "- [ ] Input Normalizer agent is active with contracts\n"
                "- [ ] Ticket Classifier agent is active with contracts\n"
                "- [ ] Ticket Triage Pipeline is active with 2 steps\n"
                "- [ ] The Control Center graph shows provider, model, agent, pipeline and step links\n"
                "- [ ] Any **Needs attention** item has been opened, understood, archived or silenced intentionally\n"
                "- [ ] At least one SupportTicket exists in Support Demo\n"
                "- [ ] Triage action has been run on the ticket\n"
                "- [ ] Ticket shows a successful AI session\n\n"
                "### Steps\n\n"
                "1. Go to [Support Demo - Tickets](/admin/support_demo/supportticket/)\n"
                "2. Create a ticket or use the seeded demo tickets\n"
                "3. Tick its checkbox, choose **Run AI triage on selected tickets** and click **Go**\n"
                "4. Verify the session completed with `success`\n"
                "5. Open Control Center and confirm the graph and attention inbox match the workflow state\n"
                "6. Click **Check Mission**\n\n"
                "![A triaged support ticket: Status is Triaged, the AI session is linked, and the "
                "Ticket Analysis shows the category, priority and reason the pipeline produced]"
                "(/static/academy/img/tutorials/ticket-triaged.png)\n"
            ),
        },
    }

    modules_data[0]["missions"].append(interface_mission)
    for mod_data in modules_data:
        for index, mission in enumerate(mod_data["missions"]):
            override = mission_overrides.get(mission["slug"])
            if override:
                updated = mission.copy()
                updated.update(override)
                mod_data["missions"][index] = updated

    created_count = 0
    updated_count = 0
    for mod_data in modules_data:
        module, module_created = TutorialModule.objects.get_or_create(
            slug=mod_data["slug"],
            defaults={
                "title": mod_data["title"],
                "order": mod_data["order"],
                "description": mod_data.get("description", ""),
                "is_active": True,
            },
        )
        if not module_created and force_update:
            module_fields = {
                "title": mod_data["title"],
                "order": mod_data["order"],
                "description": mod_data.get("description", ""),
                "is_active": True,
            }
            changed = False
            for field, value in module_fields.items():
                if getattr(module, field) != value:
                    setattr(module, field, value)
                    changed = True
            if changed:
                module.save(update_fields=list(module_fields.keys()))
                updated_count += 1

        for m_data in mod_data["missions"]:
            mission, created = TutorialMission.objects.get_or_create(
                slug=m_data["slug"],
                defaults={
                    "module": module,
                    "title": m_data["title"],
                    "order": m_data["order"],
                    "goal": m_data["goal"],
                    "instructions_markdown": m_data["instructions"],
                    "validation_key": m_data["validation_key"],
                    "is_active": True,
                },
            )
            if created:
                created_count += 1
                print(f"  Created mission: {m_data['title']}")
            elif force_update:
                mission_fields = {
                    "module": module,
                    "title": m_data["title"],
                    "order": m_data["order"],
                    "goal": m_data["goal"],
                    "instructions_markdown": m_data["instructions"],
                    "validation_key": m_data["validation_key"],
                    "is_active": True,
                }
                changed = False
                for field, value in mission_fields.items():
                    if getattr(mission, field) != value:
                        setattr(mission, field, value)
                        changed = True
                if changed:
                    mission.save(update_fields=list(mission_fields.keys()))
                    updated_count += 1
                    print(f"  Updated mission: {m_data['title']}")
    return created_count, updated_count


class Command(BaseCommand):
    help = "Seed training provider, models, agents, knowledge and tutorial missions."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force-update",
            action="store_true",
            help="Update existing tutorial modules and missions with the latest seeded content.",
        )

    def handle(self, *args, **options):
        self.stdout.write("Seeding Academy training data...")
        provider = _get_or_create_provider()
        model = _get_or_create_model(provider)
        _get_or_create_doc_assistant(model)
        _get_or_create_normalizer(model)
        _get_or_create_classifier(model)
        _get_or_create_knowledge()
        created, updated = _seed_tutorial_missions(force_update=options["force_update"])
        self.stdout.write(self.style.SUCCESS(
            "Done. Training provider, models, agents and tutorial content seeded "
            f"({created} created, {updated} updated)."
        ))
