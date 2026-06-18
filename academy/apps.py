import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AcademyConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "academy"
    verbose_name = "Academy"

    def ready(self):
        from django.db.models.signals import post_migrate
        post_migrate.connect(_on_post_migrate, sender=self)


def _on_post_migrate(sender, **kwargs):
    """
    After migrations complete, run a lightweight doc-sync check.

    We use post_migrate (not ready()) so the DB is guaranteed to exist and
    all migrations are applied before we touch any models.
    """
    try:
        from ai_hub.models import AgentProfile
        if not AgentProfile.objects.filter(name="Documentation Sync Agent", is_active=True).exists():
            return  # seed not run yet — skip silently
        _trigger_doc_sync()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto doc-sync skipped: %s", exc)


def _trigger_doc_sync():
    from ai_hub.models import ExecutionSession
    from ai_hub.services.execution_runner import run_execution_session
    from ai_hub.models import AgentProfile

    agent = AgentProfile.objects.filter(name="Documentation Sync Agent", is_active=True).first()
    if not agent:
        return

    session = ExecutionSession.objects.create(
        runtime_kind=ExecutionSession.RuntimeKind.GAME,
        entry_agent=agent,
        goal_text=(
            "Synchronize the documentation database with the Markdown source files. "
            "Check all .md files, update any that have changed, and report what was done."
        ),
        initial_context={},
        runtime_config={"max_iterations": 3},
        source_label="startup auto-sync",
    )
    run_execution_session(session.pk)
    session.refresh_from_db()
    logger.info(
        "Doc sync completed (session pk=%s, status=%s).",
        session.pk,
        session.status,
    )
