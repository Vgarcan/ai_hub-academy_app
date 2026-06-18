"""
Seed the Ollama provider, model configs, tools, and GAME agents for the documentation POC.

The Ollama base URL is read from the OLLAMA_BASE_URL environment variable (set it in .env).
You can also pass it directly with --base-url.

Usage:
    python manage.py seed_ollama_agents
    python manage.py seed_ollama_agents --base-url http://localhost:11434
    python manage.py seed_ollama_agents --force-update
"""
import os

from django.core.management.base import BaseCommand

from ai_hub.models import AgentProfile, ModelConfig, ProviderConfig, ToolDefinition


DOC_SYNC_SYSTEM_PROMPT = """\
You are the Documentation Sync Agent for AI Hub Academy.

Your goal: ensure the documentation database is synchronized with the Markdown source files.

You have one tool called `sync_all_docs`.  It scans every .md file in docs_source/,
compares content with the database, and syncs anything that changed or is missing.

The result of the tool is available in:
  tool_results.sync_all_docs.checked     — total files found
  tool_results.sync_all_docs.synced      — files that were created or updated
  tool_results.sync_all_docs.unchanged   — files already in sync
  tool_results.sync_all_docs.details     — per-file breakdown
  tool_results.sync_all_docs.message     — human-readable summary
  tool_results.sync_all_docs.error       — set only if something went wrong

Instructions:
1. Read the tool_results to see what was synced.
2. If there is an error, include it in your final_answer.
3. If synced > 0, list which files were updated.
4. If unchanged == checked, confirm everything is already up to date.
5. Set complete=true and write a concise summary in final_answer.

Return ONLY a JSON object — no prose, no markdown, no code fences:
{
  "action": "finish",
  "message": "one-line note on what you observed",
  "complete": true,
  "final_answer": "Human-readable sync report."
}

One iteration is enough — the tool does all the work.
"""

DOC_ASSISTANT_SYSTEM_PROMPT = """\
You are the AI Hub Documentation Assistant.

Your goal: answer the user's question by searching the AI Hub documentation database.

You have one tool called `search_documentation`.
- First call: searches using the user's original question (from context.goal_text).
- Later calls: searches using the `search_query` field you include in your response.

The results are in:
  tool_results.search_documentation.query    — the query that was used
  tool_results.search_documentation.results  — list of {page, section, content, relevance_rank}
  tool_results.search_documentation.total    — number of results
  tool_results.search_documentation.message  — human-readable status

Process:
1. Read the tool_results carefully.
2. If you have enough information to answer, set complete=true and write the answer.
3. If you need more specific information, set complete=false and include a "search_query" field
   with a refined search term.  The tool will use it in the next iteration.
4. After at most 3 searches, give the best answer you have.

Return ONLY a JSON object:
{
  "action": "think",
  "message": "brief note on what you found or why you need another search",
  "search_query": "next search query — ONLY include when action is think",
  "complete": false,
  "final_answer": ""
}

OR when you are done:
{
  "action": "finish",
  "message": "I have found all the relevant information.",
  "complete": true,
  "final_answer": "Complete answer here.  Cite [page / section] inline."
}

Rules:
- Only answer from the documentation returned in tool_results.
- Cite sources as [Page Title / Section].
- If no relevant docs found after searching, say so clearly.
- Be concise but complete.
"""


class Command(BaseCommand):
    help = "Seed Ollama provider, model configs, tools, and GAME agents for the documentation POC."

    def add_arguments(self, parser):
        parser.add_argument(
            "--base-url",
            default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            help=(
                "Ollama base URL. "
                "Defaults to OLLAMA_BASE_URL env var, then http://localhost:11434."
            ),
        )
        parser.add_argument(
            "--sync-model",
            default="ollama/qwen3:8b",
            help="Model for the doc sync agent (default: ollama/qwen3:8b).",
        )
        parser.add_argument(
            "--chat-model",
            default="ollama/qwen3:8b",
            help="Model for the documentation assistant (default: ollama/qwen3:8b).",
        )
        parser.add_argument(
            "--force-update",
            action="store_true",
            help="Update system prompts and settings on existing records.",
        )

    def handle(self, *args, **options):
        base_url = options["base_url"]
        sync_model_name = options["sync_model"]
        chat_model_name = options["chat_model"]
        force = options["force_update"]

        # ── Provider ──────────────────────────────────────────────────────────
        provider, created = ProviderConfig.objects.get_or_create(
            name="Ollama Local",
            defaults={
                "provider_type": ProviderConfig.ProviderType.OLLAMA,
                "base_url": base_url,
                "api_key_env_var": "",
                "default_timeout": 120,
                "is_active": True,
            },
        )
        if not created and force:
            provider.base_url = base_url
            provider.is_active = True
            provider.save(update_fields=["base_url", "is_active", "updated_at"])
        self.stdout.write(f"{'Created' if created else 'Found'} provider: {provider.name} at {provider.base_url}")

        # ── Models ────────────────────────────────────────────────────────────
        sync_model, created = ModelConfig.objects.get_or_create(
            provider=provider,
            model_name=sync_model_name,
            defaults={
                "temperature_default": "0.20",
                "max_tokens_default": 1500,
                "supports_tools": False,
                "is_active": True,
            },
        )
        self.stdout.write(f"{'Created' if created else 'Found'} model: {sync_model}")

        chat_model, created = ModelConfig.objects.get_or_create(
            provider=provider,
            model_name=chat_model_name,
            defaults={
                "temperature_default": "0.60",
                "max_tokens_default": 2000,
                "supports_tools": False,
                "is_active": True,
            },
        )
        if sync_model_name == chat_model_name:
            chat_model = sync_model
        self.stdout.write(f"{'Created' if created else 'Found'} model: {chat_model}")

        # ── Tools ─────────────────────────────────────────────────────────────
        sync_tool, created = ToolDefinition.objects.get_or_create(
            name="sync_all_docs",
            defaults={
                "tool_kind": ToolDefinition.ToolKind.PYTHON_CALLABLE,
                "config": {
                    "callable": "academy.tools.doc_sync.sync_all_docs",
                    "source_name": "AI Hub Official Docs",
                },
                "input_schema": {},
                "output_schema": {},
                "is_active": True,
            },
        )
        self.stdout.write(f"{'Created' if created else 'Found'} tool: {sync_tool.name}")

        search_tool, created = ToolDefinition.objects.get_or_create(
            name="search_documentation",
            defaults={
                "tool_kind": ToolDefinition.ToolKind.PYTHON_CALLABLE,
                "config": {
                    "callable": "academy.tools.doc_search.search_docs",
                    "limit": 6,
                },
                "input_schema": {},
                "output_schema": {},
                "is_active": True,
            },
        )
        self.stdout.write(f"{'Created' if created else 'Found'} tool: {search_tool.name}")

        # ── Agents ────────────────────────────────────────────────────────────
        sync_agent, created = AgentProfile.objects.get_or_create(
            name="Documentation Sync Agent",
            defaults={
                "role": "Autonomous documentation database sync agent",
                "system_prompt": DOC_SYNC_SYSTEM_PROMPT,
                "model_config": sync_model,
                "is_active": True,
            },
        )
        if not created and force:
            sync_agent.system_prompt = DOC_SYNC_SYSTEM_PROMPT
            sync_agent.model_config = sync_model
            sync_agent.is_active = True
            sync_agent.save(update_fields=["system_prompt", "model_config", "is_active", "updated_at"])
        sync_agent.tools.set([sync_tool])
        self.stdout.write(f"{'Created' if created else 'Found'} agent: {sync_agent.name}")

        assistant_agent, created = AgentProfile.objects.get_or_create(
            name="AI Hub Documentation Assistant",
            defaults={
                "role": "Documentation Q&A assistant",
                "system_prompt": DOC_ASSISTANT_SYSTEM_PROMPT,
                "model_config": chat_model,
                "is_active": True,
            },
        )
        if not created and force:
            assistant_agent.system_prompt = DOC_ASSISTANT_SYSTEM_PROMPT
            assistant_agent.model_config = chat_model
            assistant_agent.is_active = True
            assistant_agent.save(update_fields=["system_prompt", "model_config", "is_active", "updated_at"])
        assistant_agent.tools.set([search_tool])
        self.stdout.write(f"{'Created' if created else 'Found'} agent: {assistant_agent.name}")

        self.stdout.write(self.style.SUCCESS(
            "\nOllama POC agents ready.\n"
            "Next steps:\n"
            "  python manage.py run_doc_sync          # run the doc-sync GAME agent once\n"
            "  python manage.py runserver             # then open /chat/ to test the assistant\n"
        ))
