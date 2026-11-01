"""INSTALL-1: a fresh Academy development installation must work, repeatably.

A real Windows installation failed at `seed_academy_training_data` with "No
active application scope exists": the seeds need an ApplicationScope, AI Hub Core
refuses to create or guess one, and nothing in setup provided it. The failed run
also left a half-seeded database behind.

Everything here drives the real management commands, and the real `setup_dev.py`
steps against a disposable SQLite file. Nothing mocks a command to claim it ran.
AI Hub Core's own refusal is asserted unchanged: this suite proves the *host*
supplies the scope, not that Core relaxed.
"""
import os
import re
import sqlite3
import sys
from contextlib import closing, contextmanager
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.core.management import call_command, get_commands
from django.core.management.base import CommandError
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from academy.models import (
    DocumentationChunk,
    DocumentationPage,
    TutorialMission,
    TutorialModule,
)
from academy.services.scope_bootstrap import (
    ACADEMY_SCOPE_NAME,
    ACADEMY_SCOPE_SLUG,
    ScopeBootstrapError,
    ensure_academy_scope,
)
from ai_hub.models import (
    AgentProfile,
    ApplicationScope,
    KnowledgeCollection,
    ModelConfig,
    ProviderConfig,
)
from ai_hub.services.application_scope import require_single_active_scope

User = get_user_model()
PROJECT_ROOT = Path(settings.BASE_DIR)


def run(command, *args):
    out = StringIO()
    with patch("builtins.print"):  # the seeds report progress with print()
        call_command(command, *args, stdout=out, stderr=StringIO())
    return out.getvalue()


def counts():
    """Every table setup writes to, as one comparable snapshot."""
    return {
        "scopes": ApplicationScope.objects.count(),
        "providers": ProviderConfig.objects.count(),
        "models": ModelConfig.objects.count(),
        "agents": AgentProfile.objects.count(),
        "collections": KnowledgeCollection.objects.count(),
        "modules": TutorialModule.objects.count(),
        "missions": TutorialMission.objects.count(),
        "pages": DocumentationPage.objects.count(),
        "chunks": DocumentationChunk.objects.count(),
    }


def install_sequence():
    """The README's manual sequence, minus the superuser and the server."""
    run("bootstrap_academy_scope")
    run("seed_academy_training_data")
    run("import_academy_docs")


class ScopeBootstrapServiceTests(TestCase):
    def test_core_still_refuses_to_create_or_guess_a_scope(self):
        """The Core rule this whole fix must leave alone."""
        self.assertEqual(ApplicationScope.objects.count(), 0)
        with self.assertRaises(ValidationError):
            require_single_active_scope()
        self.assertEqual(ApplicationScope.objects.count(), 0)

    def test_fresh_database_gets_one_explicit_stable_scope(self):
        result = ensure_academy_scope()

        self.assertTrue(result.created)
        scope = ApplicationScope.objects.get()
        self.assertEqual((scope.slug, scope.name), (ACADEMY_SCOPE_SLUG, ACADEMY_SCOPE_NAME))
        self.assertTrue(scope.is_active)
        self.assertEqual(require_single_active_scope(), scope)

    def test_created_scope_grants_no_egress_and_no_retrieval_configuration(self):
        scope = ensure_academy_scope().scope
        self.assertFalse(scope.allow_external_embedding_corpus_egress)
        self.assertFalse(scope.allow_external_embedding_query_egress)
        self.assertIsNone(scope.retrieval_embedding_model_config)

    def test_repeating_is_idempotent(self):
        first = ensure_academy_scope()
        second = ensure_academy_scope()

        self.assertFalse(second.created)
        self.assertEqual(first.scope.pk, second.scope.pk)
        self.assertEqual(ApplicationScope.objects.count(), 1)

    def test_an_existing_single_active_scope_is_used_unchanged(self):
        existing = ApplicationScope.objects.create(
            slug="legacy-default", name="Legacy default", description="kept"
        )
        before = ApplicationScope.objects.values().get(pk=existing.pk)

        result = ensure_academy_scope()

        self.assertFalse(result.created)
        self.assertEqual(result.scope.pk, existing.pk)
        self.assertEqual(ApplicationScope.objects.count(), 1)
        self.assertEqual(ApplicationScope.objects.values().get(pk=existing.pk), before)

    def test_existing_agents_and_collections_are_never_reassigned(self):
        existing = ApplicationScope.objects.create(slug="host-app", name="Host app")
        provider = ProviderConfig.objects.create(name="p", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="m")
        agent = AgentProfile.objects.create(
            name="a", role="r", model_config=model, application_scope=existing
        )
        collection = KnowledgeCollection.objects.create(name="c", application_scope=existing)

        ensure_academy_scope()

        agent.refresh_from_db()
        collection.refresh_from_db()
        self.assertEqual(agent.application_scope_id, existing.pk)
        self.assertEqual(collection.application_scope_id, existing.pk)

    def test_several_active_scopes_are_never_resolved_by_guessing(self):
        ApplicationScope.objects.create(slug="alpha", name="Alpha")
        ApplicationScope.objects.create(slug="beta", name="Beta")

        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()

        message = str(raised.exception)
        self.assertIn("'alpha'", message)
        self.assertIn("'beta'", message)
        self.assertIn("will not choose", message)
        self.assertEqual(ApplicationScope.objects.count(), 2)
        self.assertFalse(ApplicationScope.objects.filter(slug=ACADEMY_SCOPE_SLUG).exists())

    def test_an_inactive_academy_scope_is_not_reactivated(self):
        ApplicationScope.objects.create(
            slug=ACADEMY_SCOPE_SLUG, name=ACADEMY_SCOPE_NAME, is_active=False
        )

        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()

        self.assertIn("will not reactivate", str(raised.exception))
        self.assertFalse(ApplicationScope.objects.get(slug=ACADEMY_SCOPE_SLUG).is_active)

    def test_an_inactive_scope_with_the_academy_name_is_not_renamed_or_reactivated(self):
        ApplicationScope.objects.create(slug="other", name=ACADEMY_SCOPE_NAME, is_active=False)

        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()

        self.assertIn("will not rename or reactivate", str(raised.exception))
        other = ApplicationScope.objects.get(slug="other")
        self.assertFalse(other.is_active)
        self.assertEqual(other.name, ACADEMY_SCOPE_NAME)
        self.assertEqual(ApplicationScope.objects.count(), 1)

    def test_unrelated_inactive_scopes_stay_inactive(self):
        ApplicationScope.objects.create(slug="retired", name="Retired", is_active=False)

        result = ensure_academy_scope()

        self.assertTrue(result.created)
        self.assertFalse(ApplicationScope.objects.get(slug="retired").is_active)
        self.assertEqual(require_single_active_scope().slug, ACADEMY_SCOPE_SLUG)


class SeedFailsBeforeWritingTests(TestCase):
    """The reported failure, and the half-seeded database it used to leave."""

    def test_seed_on_a_database_with_no_scope_writes_nothing_and_says_what_to_run(self):
        with self.assertRaises(CommandError) as raised:
            run("seed_academy_training_data")

        message = str(raised.exception)
        self.assertIn("No active application scope exists", message)
        self.assertIn("python manage.py bootstrap_academy_scope", message)
        self.assertEqual(ProviderConfig.objects.count(), 0)
        self.assertEqual(ModelConfig.objects.count(), 0)
        self.assertEqual(ApplicationScope.objects.count(), 0)

    def test_ollama_seed_also_writes_nothing_without_a_scope(self):
        with self.assertRaises(CommandError):
            run("seed_ollama_agents")
        self.assertEqual(ProviderConfig.objects.count(), 0)

    def test_seed_never_creates_a_scope_itself(self):
        with self.assertRaises(CommandError):
            run("seed_academy_training_data")
        self.assertEqual(ApplicationScope.objects.count(), 0)

    def test_seed_refuses_ambiguity_and_does_not_point_at_a_guess(self):
        ApplicationScope.objects.create(slug="alpha", name="Alpha")
        ApplicationScope.objects.create(slug="beta", name="Beta")

        with self.assertRaises(CommandError) as raised:
            run("seed_academy_training_data")

        message = str(raised.exception)
        self.assertIn("2 active application scopes", message)
        # Bootstrap would refuse as well, so the seed must not send the operator
        # to it: the decision belongs to them, in the Admin.
        self.assertNotIn("bootstrap_academy_scope", message)
        self.assertIn("Admin > Application scopes", message)
        self.assertEqual(ProviderConfig.objects.count(), 0)

    def test_bootstrap_command_reports_a_conflict_as_a_command_error(self):
        ApplicationScope.objects.create(slug="alpha", name="Alpha")
        ApplicationScope.objects.create(slug="beta", name="Beta")
        with self.assertRaises(CommandError):
            run("bootstrap_academy_scope")


class InstallationSequenceTests(TestCase):
    """The documented manual sequence, run for real."""

    def test_fresh_installation_completes(self):
        install_sequence()

        snapshot = counts()
        self.assertEqual(snapshot["scopes"], 1)
        self.assertGreaterEqual(snapshot["agents"], 3)
        self.assertGreater(snapshot["modules"], 0)
        self.assertGreater(snapshot["missions"], 0)
        self.assertGreater(snapshot["pages"], 10)
        self.assertGreater(snapshot["chunks"], 100)
        scope = ApplicationScope.objects.get()
        self.assertFalse(
            AgentProfile.objects.exclude(application_scope=scope).exists()
        )
        self.assertFalse(
            KnowledgeCollection.objects.exclude(application_scope=scope).exists()
        )

    def test_repeating_the_sequence_changes_nothing(self):
        install_sequence()
        first = counts()
        ids = set(AgentProfile.objects.values_list("pk", flat=True))

        install_sequence()

        self.assertEqual(counts(), first)
        self.assertEqual(set(AgentProfile.objects.values_list("pk", flat=True)), ids)

    def test_repeating_does_not_overwrite_user_configuration(self):
        install_sequence()
        agent = AgentProfile.objects.get(name="AI Hub Documentation Assistant")
        agent.system_prompt = "My own prompt."
        agent.save(update_fields=["system_prompt"])
        provider = ProviderConfig.objects.get(name="Academy Training Provider")
        provider.default_timeout = 77
        provider.save(update_fields=["default_timeout"])

        install_sequence()

        agent.refresh_from_db()
        provider.refresh_from_db()
        self.assertEqual(agent.system_prompt, "My own prompt.")
        self.assertEqual(provider.default_timeout, 77)

    def test_recovery_from_the_reported_partial_installation(self):
        """Provider and model exist (as the failed seed left them); no scope."""
        provider = ProviderConfig.objects.create(
            name="Academy Training Provider", provider_type="training", default_timeout=30
        )
        ModelConfig.objects.create(provider=provider, model_name="training/assistant")

        with self.assertRaises(CommandError):
            run("seed_academy_training_data")
        self.assertEqual(AgentProfile.objects.count(), 0)

        install_sequence()

        self.assertEqual(ProviderConfig.objects.filter(name="Academy Training Provider").count(), 1)
        self.assertEqual(ModelConfig.objects.filter(model_name="training/assistant").count(), 1)
        self.assertEqual(counts()["scopes"], 1)
        self.assertGreaterEqual(counts()["agents"], 3)

    def test_an_existing_compatible_scope_receives_the_seeded_resources(self):
        existing = ApplicationScope.objects.create(slug="legacy-default", name="Legacy default")

        install_sequence()

        self.assertEqual(ApplicationScope.objects.count(), 1)
        self.assertFalse(ApplicationScope.objects.filter(slug=ACADEMY_SCOPE_SLUG).exists())
        self.assertFalse(AgentProfile.objects.exclude(application_scope=existing).exists())

    def test_ambiguous_scopes_stop_the_installation_before_any_seeding(self):
        ApplicationScope.objects.create(slug="alpha", name="Alpha")
        ApplicationScope.objects.create(slug="beta", name="Beta")

        with self.assertRaises(CommandError):
            install_sequence()

        self.assertEqual(counts()["providers"], 0)
        self.assertEqual(counts()["pages"], 0)
        self.assertEqual(counts()["scopes"], 2)


class InstalledApplicationSmokeTests(TestCase):
    """After the documented sequence the application actually serves pages."""

    @classmethod
    def setUpTestData(cls):
        install_sequence()
        cls.admin_password = "smoke-test-only-password"
        User.objects.create_superuser("admin", "admin@example.com", cls.admin_password)

    def test_public_pages_render_without_a_provider_or_a_model(self):
        module = TutorialModule.objects.filter(is_active=True).first()
        mission = module.missions.first()
        page = DocumentationPage.objects.filter(is_active=True).first()
        for name, args in (
            ("academy:landing", []),
            ("academy:docs_list", []),
            ("academy:docs_detail", [page.slug]),
            ("academy:tutorial_list", []),
            ("academy:tutorial_module", [module.slug]),
            ("academy:tutorial_mission", [module.slug, mission.slug]),
            ("academy:assistant", []),
        ):
            with self.subTest(page=name):
                self.assertEqual(self.client.get(reverse(name, args=args)).status_code, 200)
        self.assertEqual(
            self.client.get(reverse("academy:docs_search") + "?q=provider").status_code, 200
        )

    def test_protected_pages_require_login_and_work_for_the_admin(self):
        self.assertEqual(self.client.get("/dashboard/").status_code, 302)
        self.assertEqual(self.client.get(reverse("academy:progress")).status_code, 302)
        self.assertEqual(self.client.get("/admin/login/").status_code, 200)

        self.assertTrue(self.client.login(username="admin", password=self.admin_password))
        for url in ("/admin/", "/admin/ai_hub/", "/dashboard/", reverse("academy:progress")):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 200)


@contextmanager
def disposable_database():
    """An environment where setup_dev's commands see only a throwaway SQLite file."""
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "install-smoke.sqlite3"
        env = {
            "DATABASE_URL": "",
            "DATABASE_ENGINE": "sqlite",
            "SQLITE_NAME": str(db_path),
        }
        previous_cwd = Path.cwd()
        os.chdir(PROJECT_ROOT)
        try:
            with patch.dict(os.environ, env):
                yield db_path
        finally:
            os.chdir(previous_cwd)


def table_count(db_path, table):
    # `with sqlite3.connect()` only ends a transaction; it leaves the file open,
    # which Windows then refuses to delete with the temporary directory.
    with closing(sqlite3.connect(db_path)) as connection:
        return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class SetupScriptEndToEndTests(SimpleTestCase):
    """`setup_dev.py`'s own steps, as real subprocesses, on a disposable database.

    The virtual environment, `pip install` and `.env` steps are not run: they
    would touch the working tree and the network, and CI has just installed the
    dependencies itself. Everything that writes to the database is.
    """

    def run_database_steps(self, setup_dev):
        python = Path(sys.executable)
        setup_dev.run_migrations(python)
        setup_dev.bootstrap_scope(python)
        setup_dev.seed_data(python)
        setup_dev.import_docs(python)
        with patch.object(setup_dev, "ADMIN_PASSWORD", "smoke-test-only-password"):
            return setup_dev.create_admin(python)

    def test_fresh_then_repeated_setup(self):
        import setup_dev

        with disposable_database() as db_path, patch("builtins.print"):
            first_password = self.run_database_steps(setup_dev)
            self.assertEqual(first_password, "smoke-test-only-password")
            snapshot = {
                table: table_count(db_path, table)
                for table in (
                    "ai_hub_applicationscope",
                    "ai_hub_agentprofile",
                    "ai_hub_providerconfig",
                    "academy_tutorialmission",
                    "academy_documentationpage",
                    "auth_user",
                )
            }
            self.assertEqual(snapshot["ai_hub_applicationscope"], 1)
            self.assertGreaterEqual(snapshot["ai_hub_agentprofile"], 3)
            self.assertEqual(snapshot["auth_user"], 1)

            with patch.object(setup_dev, "_warn") as warned:
                self.assertIsNone(self.run_database_steps(setup_dev))  # admin already exists
            warned.assert_not_called()  # a repeat must not claim a failure

            for table, expected in snapshot.items():
                with self.subTest(table=table):
                    self.assertEqual(table_count(db_path, table), expected)

    def test_django_duplicate_username_wording_is_recognized(self):
        import setup_dev

        for wording in (
            "CommandError: Error: That username is already taken.",
            "A user with that username already exists.",
        ):
            with self.subTest(wording=wording):
                with patch.object(setup_dev, "_manage") as manage, patch.object(
                    setup_dev, "_warn"
                ) as warned, patch("builtins.print"):
                    manage.return_value = type(
                        "R", (), {"returncode": 1, "stdout": "", "stderr": wording}
                    )()
                    self.assertIsNone(setup_dev.create_admin(Path("python")))
                warned.assert_not_called()

    def test_scope_is_provided_before_the_seed_runs(self):
        import setup_dev

        order = []
        steps = (
            "check_python", "create_venv", "install_deps", "setup_env",
            "run_migrations", "bootstrap_scope", "seed_data", "import_docs",
            "create_admin",
        )
        patches = [
            patch.object(setup_dev, name, side_effect=lambda *a, _n=name: order.append(_n))
            for name in steps
        ]
        with disposable_database(), patch("builtins.print"):
            for p in patches:
                p.start()
            try:
                with patch.object(setup_dev, "print_summary"):
                    setup_dev.main()
            finally:
                for p in patches:
                    p.stop()

        self.assertEqual(order, list(steps))
        self.assertLess(order.index("bootstrap_scope"), order.index("seed_data"))

    def test_a_failed_step_tells_the_user_it_is_safe_to_repeat(self):
        import setup_dev

        with patch("sys.stderr", new_callable=StringIO) as stderr:
            with self.assertRaises(SystemExit):
                setup_dev._run([sys.executable, "-c", "import sys; sys.exit(3)"])
        self.assertIn("Setup is safe to repeat", stderr.getvalue())
        self.assertIn("python setup_dev.py", stderr.getvalue())

    def test_a_broken_existing_venv_is_refused_and_not_removed(self):
        import setup_dev

        with TemporaryDirectory() as tmp:
            fake_python = Path(tmp) / "python.exe"
            fake_python.write_text("not an interpreter", encoding="utf-8")
            self.assertIsNone(setup_dev._interpreter_version(fake_python))
            with patch("sys.stderr", new_callable=StringIO) as stderr:
                with self.assertRaises(SystemExit):
                    setup_dev.check_existing_venv(fake_python)
            self.assertIn("will not remove it for you", stderr.getvalue())
            self.assertTrue(fake_python.exists())

    def test_a_too_old_existing_venv_is_refused(self):
        import setup_dev

        with patch.object(setup_dev, "_interpreter_version", return_value=(3, 10, 4)):
            with patch("sys.stderr", new_callable=StringIO) as stderr:
                with self.assertRaises(SystemExit):
                    setup_dev.check_existing_venv(Path("python"))
        self.assertIn("uses Python 3.10", stderr.getvalue())

    def test_a_working_existing_venv_is_accepted(self):
        import setup_dev

        version = setup_dev._interpreter_version(Path(sys.executable))
        self.assertEqual(version[:2], sys.version_info[:2])
        with patch("builtins.print"):
            setup_dev.check_existing_venv(Path(sys.executable))


class ReadmeIsAnOperationalContractTests(SimpleTestCase):
    """Commands the README tells people to run must exist, in a working order."""

    def readme(self):
        return (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    def test_every_documented_manage_command_exists(self):
        documented = set(re.findall(r"python manage\.py ([a-z_]+)", self.readme()))
        builtin_or_known = documented & {"runserver", "shell", "test", "migrate"}
        self.assertTrue(builtin_or_known)
        missing = sorted(documented - set(get_commands()))
        self.assertEqual(missing, [], f"README names commands that do not exist: {missing}")

    def test_manual_setup_provides_the_scope_before_seeding(self):
        text = self.readme()
        manual = text[text.index("## Manual setup"):]
        order = [
            manual.index("python manage.py migrate"),
            manual.index("python manage.py bootstrap_academy_scope"),
            manual.index("python manage.py seed_academy_training_data"),
            manual.index("python manage.py import_academy_docs"),
        ]
        self.assertEqual(order, sorted(order))

    def test_readme_documents_recovery_from_the_missing_scope_error(self):
        text = self.readme()
        self.assertIn("No active application scope exists", text)
        self.assertIn("bootstrap_academy_scope", text)
