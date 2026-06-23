"""
Seed Lab exercises for AI Hub Academy tutorial modules.

Usage:
    python manage.py seed_lab_exercises
    python manage.py seed_lab_exercises --force           # overwrite existing
    python manage.py seed_lab_exercises --create-evaluator  # also create/update the AI evaluator agent
"""
from django.core.management.base import BaseCommand


EVALUATOR_SYSTEM_PROMPT = """You are an educational assistant that evaluates student answers for AI Hub Academy lab exercises.

You receive the following in your context:
- exercise_title: the exercise name
- exercise_prompt: the challenge the student was given
- exercise_context: background information shown to the student
- evaluation_rubric: the criteria you must use to score the answer
- student_answer: what the student wrote

You MUST respond with ONLY this exact JSON (no prose, no markdown fence, nothing else):
{
  "score": "pass",
  "feedback": "Detailed, constructive feedback — what was good, what was missing, how to improve. Be specific. 2-4 sentences.",
  "follow_up_question": "A question to deepen understanding. Leave empty string if not needed."
}

Valid score values: "pass", "partial", "fail"
- pass: student clearly meets all rubric criteria
- partial: student shows some understanding but is missing key elements
- fail: answer is incorrect, too vague, or shows a fundamental misunderstanding

Always be encouraging and educational, never harsh."""


# Keyed by module ORDER number → list of exercise dicts
EXERCISES = {
    0: [  # Orientation and Mission Deck
        {
            "title": "Mission Deck Interface Audit",
            "slug": "mission-deck-interface-audit",
            "order": 1,
            "difficulty": "beginner",
            "requires_api": False,
            "prompt": (
                "Open the AI Hub Control Center and write a short operator note for a teammate "
                "who has never used the new interface. "
                "Your note must explain: "
                "1) how hover differs from selecting a graph node, "
                "2) how the draggable node pop-up helps you reach the real Admin record, "
                "3) when to use Isolate and hop depth, "
                "4) how to enter and exit full screen, "
                "5) how to use Needs attention filters, sorting, archive/restore and Silence 24h safely."
            ),
            "context": (
                "The Control Center includes the Mission Deck graph and the Needs attention inbox. "
                "Hovering a graph node only shows a compact preview. Selecting a node opens a draggable "
                "detail pop-up with relation counts and an Open record in admin action. "
                "Isolate is off by default and only hides nodes outside the selected hop range after a node is selected. "
                "Full screen gives the graph more space and can be exited with the same button or Esc. "
                "Archive and Silence 24h are stored locally in the browser; they do not delete or modify database records."
            ),
            "evaluation_rubric": (
                "Score PASS if the answer clearly covers all five required points and states that archive/silence "
                "are non-destructive local browser actions. It must mention Open record in admin as the main route "
                "from graph detail to the real Admin object. "
                "Score PARTIAL if it covers 3-4 points but misses either full screen or non-destructive attention handling. "
                "Score FAIL if it confuses hover with selection, says Isolate is automatic, or suggests archive/silence "
                "delete database records."
            ),
            "hint": (
                "Think like an operator: first inspect lightly, then select, then open the exact record only when needed."
            ),
        },
    ],

    1: [  # Provider & Model Configuration
        {
            "title": "Provider Config Design",
            "slug": "provider-config-design",
            "order": 1,
            "difficulty": "beginner",
            "requires_api": True,
            "prompt": (
                "A team wants to use AI Hub with the Anthropic Claude API. "
                "They don't want API keys visible in source code or the database. "
                "Describe how you would configure the ProviderConfig in AI Hub Admin: "
                "which fields you would fill in, what value goes in `api_key_env_var`, "
                "and why the system is designed to store the key this way."
            ),
            "context": (
                "AI Hub never stores actual API keys in the database. "
                "Instead, `api_key_env_var` holds the ENVIRONMENT VARIABLE NAME — for example "
                "'ANTHROPIC_API_KEY'. At runtime, the system calls `os.environ.get('ANTHROPIC_API_KEY')` "
                "to read the actual key. The key itself lives only in the `.env` file (which is gitignored) "
                "and in the production environment's secrets manager."
            ),
            "evaluation_rubric": (
                "Score PASS if the student: (1) says to put the env var NAME not the key value "
                "in api_key_env_var (e.g. 'ANTHROPIC_API_KEY'), (2) explains the security reason "
                "(key never in git, never in DB), (3) mentions setting the provider_type correctly. "
                "Score PARTIAL if they get 2 of 3 points. "
                "Score FAIL if they suggest putting the raw API key in the database field or in code."
            ),
            "hint": (
                "Think about what the `.env` file is for and why it's listed in `.gitignore`. "
                "The field stores a *name*, not a *value*."
            ),
        },
        {
            "title": "Model Selection Scenario",
            "slug": "model-selection-scenario",
            "order": 2,
            "difficulty": "intermediate",
            "requires_api": True,
            "prompt": (
                "Your team has access to two models: a powerful cloud API (accurate, costs money per call) "
                "and a local Ollama model (fast, free, less capable). "
                "You need to serve two agents:\n"
                "A) A classifier that processes 500 support emails/day (simple categorization task)\n"
                "B) A contract analysis agent used 10 times/month (needs precise reasoning)\n\n"
                "Which model would you assign to each agent in AI Hub? "
                "Explain your reasoning, including the cost/quality tradeoff."
            ),
            "context": (
                "Each AgentProfile in AI Hub points to exactly one ModelConfig. "
                "You can create multiple ModelConfigs pointing to different providers and models. "
                "Different agents in the same system can use different models."
            ),
            "evaluation_rubric": (
                "Score PASS if student: assigns local model to A (high volume, simple task) and cloud model "
                "to B (low volume, complex task), with clear reasoning about cost vs quality. "
                "Score PARTIAL if they choose correctly but reasoning is thin or missing. "
                "Score FAIL if they use the expensive cloud model for everything without considering cost, "
                "or the local model for everything without considering quality requirements."
            ),
            "hint": "Calculate: 500 calls/day × 30 days = 15,000 API calls/month for agent A. Does that change your decision?",
        },
    ],

    2: [  # Agent Profiles
        {
            "title": "Write a System Prompt",
            "slug": "write-system-prompt",
            "order": 1,
            "difficulty": "beginner",
            "requires_api": True,
            "prompt": (
                "Write a complete system prompt for an agent that classifies customer complaints. "
                "The agent receives a customer message and must respond with ONLY valid JSON in this exact format:\n"
                '{"category": "BILLING|TECHNICAL|SHIPPING|OTHER", "confidence": "HIGH|MEDIUM|LOW", '
                '"summary": "one sentence summary of the complaint"}\n\n'
                "Your system prompt must make the output format completely unambiguous."
            ),
            "context": (
                "A system prompt defines the agent's role, behavior, and output requirements. "
                "For structured JSON output, you must be extremely explicit — some models will still "
                "add prose or markdown fences around JSON unless explicitly forbidden. "
                "The best system prompts give a concrete example of valid output."
            ),
            "evaluation_rubric": (
                "Score PASS if the system prompt: (1) clearly defines the agent's role as a complaint classifier, "
                "(2) lists all 4 categories with exact casing, (3) explicitly forbids any output except the JSON object "
                "(no prose, no fences), (4) defines all 3 output fields including their allowed values. "
                "Score PARTIAL if 3 of 4 criteria are met. "
                "Score FAIL if the prompt is vague, missing categories, or doesn't enforce pure JSON output."
            ),
            "hint": (
                "A strong instruction might be: 'You MUST respond with ONLY a JSON object. "
                "No text before it, no text after it, no markdown code fences.' "
                "Then give an example output."
            ),
        },
        {
            "title": "Design Input/Output Contracts",
            "slug": "design-io-contracts",
            "order": 2,
            "difficulty": "intermediate",
            "requires_api": True,
            "prompt": (
                "Write the JSON Schema `input_contract` and `output_contract` for the complaint classifier "
                "from the previous exercise. "
                "Each contract must be a valid JSON Schema object with `type`, `properties`, and `required` fields. "
                "The input receives a customer message; the output returns category, confidence, and summary."
            ),
            "context": (
                "AI Hub validates agent I/O against these contracts before and after each call. "
                "Invalid messages are rejected immediately. Both use JSON Schema v7 syntax:\n"
                '{"type": "object", "properties": {"field": {"type": "string"}}, "required": ["field"]}\n'
                "Enum fields use: {\"type\": \"string\", \"enum\": [\"VALUE1\", \"VALUE2\"]}"
            ),
            "evaluation_rubric": (
                "Score PASS if: both contracts are valid JSON Schema, "
                "input_contract has 'message' as a string in 'required', "
                "output_contract has 'category' with enum ['BILLING','TECHNICAL','SHIPPING','OTHER'], "
                "'confidence' with enum ['HIGH','MEDIUM','LOW'], 'summary' as string, "
                "and all 3 in 'required'. "
                "Score PARTIAL if one contract is correct and the other has minor issues (e.g., missing required array). "
                "Score FAIL if the JSON is invalid, enums don't match the exercise, or required arrays are missing."
            ),
            "hint": 'Start the output_contract with: {"type": "object", "properties": {"category": {"type": "string", "enum": [...]}',
        },
        {
            "title": "Debug: Agent Returns Free Text",
            "slug": "debug-free-text-output",
            "order": 3,
            "difficulty": "advanced",
            "requires_api": True,
            "prompt": (
                "Your agent has a valid output_contract requiring JSON, "
                "but it keeps returning plain English sentences instead. "
                "The system prompt says 'Return your answer in JSON format'. "
                "List 3 specific, actionable steps you would take to fix this, most likely cause first. "
                "For each step, explain WHY it would fix the issue."
            ),
            "context": (
                "Common causes of structured output failures:\n"
                "1. System prompt instruction is too vague — models need an explicit format example\n"
                "2. The model doesn't support JSON/tool-call mode well\n"
                "3. Provider JSON mode not configured (some providers need a specific API parameter)\n"
                "4. Temperature too high — higher temperatures increase randomness, including format randomness\n"
                "Note: The output_contract validates the output AFTER it's received, but it doesn't force the model to produce JSON."
            ),
            "evaluation_rubric": (
                "Score PASS if student correctly identifies at least 2 of these root causes with specific fixes: "
                "(a) Strengthen system prompt — add explicit example and 'ONLY output JSON' instruction; "
                "(b) Switch to a model with better structured output support; "
                "(c) Check if provider supports JSON mode and enable it; "
                "(d) Lower the model temperature. "
                "Each fix must include an explanation of WHY it helps. "
                "Score PARTIAL if they identify 1 correct fix with good reasoning. "
                "Score FAIL if suggestions don't address the actual output format problem (e.g., 'just accept the text response')."
            ),
            "hint": "The contract validates output but doesn't control HOW the model generates it. The fix has to happen before the call, not after.",
        },
    ],

    3: [  # Knowledge Collections
        {
            "title": "Design a Knowledge Collection",
            "slug": "design-knowledge-collection",
            "order": 1,
            "difficulty": "beginner",
            "requires_api": True,
            "prompt": (
                "You're building an internal help desk bot for a 50-person software company. "
                "Design a KnowledgeCollection:\n"
                "1. List at least 4 document types you would include\n"
                "2. Explain how you would organize them\n"
                "3. Explain why semantic search (bge-m3 embeddings) is better than keyword search "
                "for this use case. Give one concrete example of a query that would work with semantic search "
                "but fail with keyword search."
            ),
            "context": (
                "A KnowledgeCollection groups related documents for the AI to search. "
                "The `embed_docs` command generates vector embeddings for each document chunk. "
                "Semantic search finds relevant content based on meaning — not just exact word matches. "
                "Example: the query 'my login is broken' semantically matches 'authentication failure troubleshooting' "
                "even though none of the words overlap."
            ),
            "evaluation_rubric": (
                "Score PASS if student: (1) lists 4+ relevant document types appropriate for a software company "
                "(e.g., FAQs, runbooks, API docs, release notes, onboarding guides, policies), "
                "(2) explains an organization strategy (by topic, by product area, by recency), "
                "(3) clearly explains semantic search with a concrete example showing query vs document vocabulary mismatch. "
                "Score PARTIAL if they list documents and mention embeddings but skip the semantic vs keyword comparison. "
                "Score FAIL if they list only generic document types with no reasoning about organization or search quality."
            ),
            "hint": "Think about the vocabulary gap: users say 'broken' and 'not working', but docs say 'error' and 'failure'. How does semantic search bridge this?",
        },
    ],

    4: [  # Orchestrator Pipelines
        {
            "title": "Design a 3-Step Pipeline",
            "slug": "design-pipeline",
            "order": 1,
            "difficulty": "intermediate",
            "requires_api": True,
            "prompt": (
                "Design a 3-step orchestrator pipeline for processing customer support emails:\n"
                "Step 1: Classify urgency (LOW / MEDIUM / HIGH)\n"
                "Step 2: Generate a reply draft\n"
                "Step 3: Translate to Spanish if the original was in English\n\n"
                "For each step, define:\n"
                "- Agent name and purpose\n"
                "- Key fields in the output contract\n"
                "- What data it receives from the previous step and how it uses it"
            ),
            "context": (
                "A PipelineDefinition has ordered PipelineSteps. "
                "Each step's output JSON is automatically passed as input context to the next step. "
                "All agents must have compatible contracts — Step N's output field names must match "
                "what Step N+1 expects in its input contract. "
                "Mismatched field names cause validation errors."
            ),
            "evaluation_rubric": (
                "Score PASS if: all 3 steps are clearly defined with distinct purposes, "
                "data flows logically between steps (urgency level from step 1 is used by step 2; "
                "draft + original language from steps 1-2 flow to step 3), "
                "and output contract fields are specified for each step. "
                "Score PARTIAL if 2 of 3 steps are complete and data flow is mostly correct. "
                "Score FAIL if the data flow is broken (step 2 doesn't receive step 1's output, etc.) "
                "or steps are too vague to be implementable."
            ),
            "hint": "Think about what data step 3 needs from step 2 (the draft) AND from step 1 (urgency). Both need to flow through.",
        },
        {
            "title": "Debug Pipeline Contract Mismatch",
            "slug": "debug-pipeline-contract",
            "order": 2,
            "difficulty": "advanced",
            "requires_api": True,
            "prompt": (
                "Step 2 of your pipeline fails with: "
                "'input_contract validation error: required field urgency_level is missing'. "
                "Examining the logs, you see step 1 returned: "
                '{"classification": "HIGH", "confidence": 0.95}\n'
                "Step 2's input_contract requires: "
                '{"urgency_level": {"type": "string"}}\n\n'
                "What is the root cause? Describe the TWO valid ways to fix this without disabling contracts."
            ),
            "context": (
                "The orchestrator passes the previous step's complete output JSON directly as "
                "the next step's input. If the field name in step 1's output doesn't match "
                "the field name in step 2's input_contract, validation fails immediately. "
                "The fix must align the field names — either by changing what step 1 outputs "
                "or by changing what step 2 expects."
            ),
            "evaluation_rubric": (
                "Score PASS if student correctly identifies: (1) the root cause is a field name mismatch "
                "('classification' vs 'urgency_level'), AND provides both fixes: "
                "(A) change step 1's output_contract to use 'urgency_level' instead of 'classification', "
                "(B) change step 2's input_contract to accept 'classification' instead of 'urgency_level'. "
                "Score PARTIAL if they identify the mismatch but only describe one fix. "
                "Score FAIL if they suggest disabling contract validation or renaming the field in step 2's system prompt only."
            ),
            "hint": "The contracts define the 'API contract' between pipeline steps. Both sides of the contract must match.",
        },
        {
            "title": "Explain Missing Graph Columns",
            "slug": "explain-missing-graph-columns",
            "order": 3,
            "difficulty": "beginner",
            "requires_api": False,
            "prompt": (
                "A teammate opens the Control Center graph and says: "
                "'I can see providers, models and agents, but I do not see Pipeline or Step columns. Is this broken?' "
                "Write the explanation you would give them. Include when Pipeline and Step columns appear, "
                "what warning should appear for an active pipeline with no steps, and what Admin records they should check."
            ),
            "context": (
                "The Mission Deck graph is built from actual AI Hub Admin records. "
                "If no PipelineDefinition records exist, the Pipeline column is absent. "
                "If pipelines exist but have no PipelineStep records, the Step column is absent. "
                "If a pipeline is active with no steps, Needs attention should show a warning that links to the pipeline record."
            ),
            "evaluation_rubric": (
                "Score PASS if the student says this is expected when the records do not exist yet, "
                "correctly distinguishes PipelineDefinition from PipelineStep, mentions the active-pipeline-without-steps warning, "
                "and tells the teammate to check Pipeline definitions and Pipeline steps in Admin. "
                "Score PARTIAL if they explain missing records but miss the Needs attention warning. "
                "Score FAIL if they say the graph should always show empty columns or that the only fix is refreshing cookies/cache."
            ),
            "hint": "The graph is data-driven: no record usually means no column.",
        },
    ],

    5: [  # GAME Loop
        {
            "title": "Explain GAME With an Analogy",
            "slug": "explain-game-analogy",
            "order": 1,
            "difficulty": "beginner",
            "requires_api": True,
            "prompt": (
                "Explain the GAME loop using an analogy from everyday life. "
                "Do NOT use examples from the AI Hub documentation — create your own. "
                "Your analogy must map each letter:\n"
                "G = Goals\n"
                "A = Actions\n"
                "M = Memory\n"
                "E = Environment\n\n"
                "Then explain: why does the AI loop iterate multiple times instead of answering in one shot?"
            ),
            "context": (
                "GAME is the core execution model in AI Hub:\n"
                "Goals = what the AI needs to achieve (set once at the start)\n"
                "Actions = what it does each iteration (search, call a tool, write a result)\n"
                "Memory = context it accumulates across iterations\n"
                "Environment = tools, data sources, and context available to it\n"
                "The loop continues until the AI returns {\"complete\": true} in its JSON response. "
                "Max iterations is set in runtime_config to prevent infinite loops."
            ),
            "evaluation_rubric": (
                "Score PASS if: (1) all 4 letters correctly defined, (2) analogy is original and each component "
                "is clearly mapped (not just named), (3) student explains why iteration is needed — "
                "complex goals require multiple steps, each action may reveal information that guides the next action. "
                "Score PARTIAL if analogy is clear but one letter is weakly mapped or the iteration explanation is missing. "
                "Score FAIL if any letter is incorrectly defined or the analogy confuses rather than clarifies."
            ),
            "hint": "Try a cooking, navigation, debugging, or investigation scenario. The key is: the agent doesn't know everything upfront — it discovers as it goes.",
        },
        {
            "title": "Troubleshoot: Infinite GAME Loop",
            "slug": "troubleshoot-game-loop",
            "order": 2,
            "difficulty": "intermediate",
            "requires_api": True,
            "prompt": (
                "Your GAME agent is stuck. Every iteration returns:\n"
                '{"complete": false, "action": "search", "message": "Looking for more information..."}\n'
                "It has run 50 iterations without finishing. "
                "List 3 root causes and the specific fix for each. "
                "Be precise — 'rewrite the code' is not a useful answer."
            ),
            "context": (
                "The GAME loop terminates only when complete: true is returned. "
                "Common infinite loop causes:\n"
                "1. Goal is too vague — the agent can always find 'more information'\n"
                "2. System prompt never tells the agent WHEN to set complete: true\n"
                "3. max_iterations not set in runtime_config (or set too high)\n"
                "4. A tool returns new results every call, so the agent never feels 'done'\n"
                "5. The goal's completion criteria cannot be verified by the agent"
            ),
            "evaluation_rubric": (
                "Score PASS if student identifies at least 2 of these with a specific fix: "
                "(a) Vague goal → rewrite goal_text to include a specific completion criterion; "
                "(b) Missing completion instruction → add 'set complete: true when you have answered the goal' to system prompt; "
                "(c) No iteration cap → set max_iterations in runtime_config; "
                "(d) Tool always returns results → add a MAX_RESULTS limit or tell agent to stop after N results. "
                "Each fix must be actionable. "
                "Score PARTIAL for 1 correct cause+fix pair with good reasoning. "
                "Score FAIL if suggestions are vague or don't address loop termination."
            ),
            "hint": "The loop terminates on a boolean flag. Ask yourself: what tells the agent to SET that flag to true?",
        },
        {
            "title": "Write a GAME Goal",
            "slug": "write-game-goal",
            "order": 3,
            "difficulty": "intermediate",
            "requires_api": True,
            "prompt": (
                "Write a `goal_text` string for a GAME agent that must:\n"
                "1. Query the last 10 support tickets from the database\n"
                "2. Identify the 3 most common issue types\n"
                "3. Generate a 3-paragraph summary report for the operations team\n\n"
                "Your goal must be specific enough that the AI knows EXACTLY when it's done. "
                "Then explain the 4 elements of a well-written GAME goal."
            ),
            "context": (
                "The goal_text is the AI's primary directive — it reads it at every iteration to stay on track. "
                "Vague goals like 'analyze support data' lead to unfocused, incomplete results. "
                "A good goal specifies: (1) data source and scope, (2) the analysis task, "
                "(3) the expected output format and destination, (4) a clear completion criterion."
            ),
            "evaluation_rubric": (
                "Score PASS if goal_text includes all 4 elements: "
                "exact data scope (last 10 support tickets), "
                "specific task (identify top 3 issue types), "
                "output format (3-paragraph summary report), "
                "clear completion condition (e.g., 'when the report is written and final_answer contains the report'). "
                "The explanation of 4 elements must be correct. "
                "Score PARTIAL if 2-3 elements are present in the goal_text or the explanation misses one element. "
                "Score FAIL if the goal is vague, doesn't define a clear output, or the 4-element explanation is missing."
            ),
            "hint": "Write the goal as if you're briefing someone who has never spoken to you before and will have NO follow-up questions.",
        },
    ],

    7: [  # Troubleshooting (module order 7, or fallback by title)
        {
            "title": "Diagnose a 401 Provider Error",
            "slug": "diagnose-provider-401",
            "order": 1,
            "difficulty": "beginner",
            "requires_api": True,
            "prompt": (
                "Your provider returns 401 Unauthorized on every API call. "
                "The base_url is correct and the model exists. "
                "Walk through your diagnostic steps in order of most likely cause. "
                "For each step: describe what you check, how you check it, and what the fix is."
            ),
            "context": (
                "401 Unauthorized means the API key is missing or invalid. "
                "In AI Hub, keys are NEVER stored directly — only the env var name. "
                "Common causes in order of likelihood:\n"
                "1. Typo in api_key_env_var field (e.g., 'ANTHRPOIC_API_KEY' instead of 'ANTHROPIC_API_KEY')\n"
                "2. .env file exists but the server wasn't restarted after adding the key\n"
                "3. .env file not in the right directory or not loaded\n"
                "4. The actual key has been revoked or rotated by the vendor\n"
                "5. Wrong provider_type selected (training providers don't need a key)"
            ),
            "evaluation_rubric": (
                "Score PASS if student's diagnostic steps include, roughly in order: "
                "(1) verify the exact name in api_key_env_var matches the .env key name, "
                "(2) confirm .env is loaded and server was restarted, "
                "(3) test the key is valid by trying it directly (e.g., curl), "
                "(4) check the provider_type isn't incorrectly set to training. "
                "Score PARTIAL if they identify the env var and .env steps but miss server restart or key validity. "
                "Score FAIL if they suggest storing the raw API key in the Admin UI or the database."
            ),
            "hint": "Run: python -c \"import os; from dotenv import load_dotenv; load_dotenv(); print(os.environ.get('YOUR_KEY_NAME'))\" to verify the key is loading.",
        },
    ],
}


class Command(BaseCommand):
    help = "Seed Lab exercises for Academy tutorial modules."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite existing exercises with the same slug.",
        )
        parser.add_argument(
            "--create-evaluator",
            action="store_true",
            help=(
                "Also create/update the 'Lab Exercise Evaluator' AgentProfile "
                "using the first available non-training model config."
            ),
        )

    def handle(self, *args, **options):
        from academy.models import LabExercise, TutorialModule

        force = options["force"]
        created_count = 0
        skipped_count = 0

        for module_order, exercises in EXERCISES.items():
            module = TutorialModule.objects.filter(order=module_order, is_active=True).first()

            # Fallback: troubleshooting module might not be order 7 — search by title
            if not module and module_order == 7:
                module = TutorialModule.objects.filter(
                    title__icontains="troubleshoot",
                    is_active=True,
                ).first()

            if not module:
                self.stdout.write(
                    self.style.WARNING(
                        f"  Module order={module_order} not found — skipping {len(exercises)} exercise(s). "
                        "Create the module first in Admin."
                    )
                )
                continue

            self.stdout.write(f"Module: {module.title} (order={module.order})")

            for ex_data in exercises:
                slug = ex_data["slug"]
                existing = LabExercise.objects.filter(slug=slug).first()

                if existing and not force:
                    self.stdout.write(f"  [skip] {ex_data['title']} (slug exists, use --force to overwrite)")
                    skipped_count += 1
                    continue

                defaults = {
                    "module": module,
                    "title": ex_data["title"],
                    "order": ex_data["order"],
                    "difficulty": ex_data["difficulty"],
                    "requires_api": ex_data.get("requires_api", True),
                    "prompt": ex_data["prompt"].strip(),
                    "context": ex_data.get("context", "").strip(),
                    "evaluation_rubric": ex_data["evaluation_rubric"].strip(),
                    "hint": ex_data.get("hint", "").strip(),
                    "is_active": True,
                }

                exercise, was_created = LabExercise.objects.update_or_create(
                    slug=slug,
                    defaults=defaults,
                )
                action = "Created" if was_created else "Updated"
                self.stdout.write(self.style.SUCCESS(f"  [{action}] {exercise.title}"))
                created_count += 1

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. {created_count} exercise(s) created/updated, {skipped_count} skipped."
        ))

        if options["create_evaluator"]:
            self._create_evaluator_agent()
        else:
            self.stdout.write(
                "\nNext step: to enable AI evaluation, create an AgentProfile named "
                "'Lab Exercise Evaluator' in Admin > AI Hub > Agent profiles.\n"
                "Recommended system prompt:\n"
                f"{EVALUATOR_SYSTEM_PROMPT}\n"
                "Or re-run with --create-evaluator to do it automatically."
            )

    def _create_evaluator_agent(self):
        try:
            from ai_hub.models import AgentProfile, ModelConfig, ProviderConfig

            model = (
                ModelConfig.objects.filter(is_active=True)
                .exclude(provider__provider_type=ProviderConfig.ProviderType.TRAINING)
                .first()
            )
            if not model:
                self.stdout.write(self.style.WARNING(
                    "No non-training model found. "
                    "Create a model config first, then re-run with --create-evaluator."
                ))
                return

            agent, created = AgentProfile.objects.update_or_create(
                name="Lab Exercise Evaluator",
                defaults={
                    "role": "Evaluates student answers for Academy Lab exercises. Do not modify the system prompt.",
                    "model_config": model,
                    "system_prompt": EVALUATOR_SYSTEM_PROMPT,
                    "is_active": True,
                },
            )
            action = "Created" if created else "Updated"
            self.stdout.write(self.style.SUCCESS(
                f"{action} 'Lab Exercise Evaluator' agent (model: {model})."
            ))
        except ImportError:
            self.stdout.write(self.style.WARNING("ai_hub app not available — skipping evaluator creation."))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f"Could not create evaluator agent: {exc}"))
