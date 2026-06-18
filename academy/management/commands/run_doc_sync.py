"""
Run the Documentation Sync GAME agent once.

This command creates an ExecutionSession for the "Documentation Sync Agent",
runs the GAME loop, and prints the result.  Use it to test the agent manually
or to trigger a sync from a cron job / startup script.

Usage:
    python manage.py run_doc_sync
    python manage.py run_doc_sync --max-iterations 3
"""
from django.core.management.base import BaseCommand, CommandError

from ai_hub.models import AgentProfile, ExecutionSession
from ai_hub.services.execution_runner import run_execution_session


class Command(BaseCommand):
    help = "Run the Documentation Sync GAME agent to sync docs_source/ → database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=3,
            help="Maximum GAME iterations (default: 3).",
        )

    def handle(self, *args, **options):
        raw = options["max_iterations"]
        max_iter = max(1, min(raw, 10))
        if raw != max_iter:
            self.stdout.write(self.style.WARNING(
                f"--max-iterations {raw} clamped to {max_iter} (allowed range: 1–10)."
            ))

        agent = AgentProfile.objects.filter(name="Documentation Sync Agent", is_active=True).first()
        if not agent:
            raise CommandError(
                "Documentation Sync Agent not found. "
                "Run: python manage.py seed_ollama_agents"
            )

        self.stdout.write(f"Starting doc sync with agent: {agent.name} (model: {agent.model_config})")

        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=agent,
            goal_text=(
                "Synchronize the documentation database with the Markdown source files. "
                "Check all .md files, update any that have changed, and report what was done."
            ),
            initial_context={},
            runtime_config={"max_iterations": max_iter},
            source_label="run_doc_sync management command",
        )

        self.stdout.write(f"Session pk={session.pk} created. Running GAME loop …")
        run_execution_session(session.pk)

        session.refresh_from_db()
        self.stdout.write(f"Status: {session.status}")

        final_ctx = session.final_context or {}
        final_answer = final_ctx.get("final_answer", "")
        if final_answer:
            self.stdout.write(self.style.SUCCESS(f"\nSync result:\n{final_answer}"))

        step_count = session.step_runs.count()
        self.stdout.write(f"Completed in {step_count} GAME iteration(s). Session pk={session.pk}")
        self.stdout.write("View full audit trail in Django admin > Execution sessions.")

        if session.status == ExecutionSession.Status.FAILED:
            raise CommandError(f"Sync session failed: {session.error_detail or 'no detail'}")
