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
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from academy.models import (
    DocumentationChunk,
    DocumentationPage,
    TutorialMission,
    TutorialModule,
)
from academy.services.scope_bootstrap import (
    ACADEMY_AGENT_NAMES,
    ACADEMY_COLLECTION_NAMES,
    ACADEMY_SCOPE_NAME,
    ACADEMY_SCOPE_SLUG,
    SETTING_NAME,
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


FOREIGN_APP = "customer-portal"


def foreign_scope(slug=FOREIGN_APP, *, is_active=True):
    """Another host application's scope: not Academy's, and not Academy's to use."""
    return ApplicationScope.objects.create(
        slug=slug, name=slug.replace("-", " ").title(), is_active=is_active
    )


def academy_scope_row():
    return ApplicationScope.objects.get(slug=ACADEMY_SCOPE_SLUG)


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
        self.assertEqual(result.other_active_slugs, ())

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

    def test_an_existing_academy_scope_is_reused_unchanged(self):
        existing = ApplicationScope.objects.create(
            slug=ACADEMY_SCOPE_SLUG, name="Renamed by an operator", description="kept"
        )
        before = ApplicationScope.objects.values().get(pk=existing.pk)

        result = ensure_academy_scope()

        self.assertFalse(result.created)
        self.assertEqual(result.scope.pk, existing.pk)
        self.assertEqual(ApplicationScope.objects.values().get(pk=existing.pk), before)

    def test_a_single_foreign_scope_is_not_adopted(self):
        """The previously unguarded case.

        Core's own helper happily returns `customer-portal` as THE scope, and the
        old Academy bootstrap accepted it, so Academy's agents and collection
        were created inside another application's boundary.
        """
        foreign_scope()
        self.assertEqual(require_single_active_scope().slug, FOREIGN_APP)  # the trap

        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()

        message = str(raised.exception)
        self.assertIn(f"'{FOREIGN_APP}'", message)
        self.assertIn("will not adopt", message)
        self.assertIn("dedicated development database", message)
        self.assertIn(f"{SETTING_NAME}=", message)
        self.assertFalse(ApplicationScope.objects.filter(slug=ACADEMY_SCOPE_SLUG).exists())
        self.assertEqual(ApplicationScope.objects.count(), 1)

    def test_academy_does_not_add_a_second_active_scope_next_to_a_foreign_one(self):
        foreign_scope()
        with self.assertRaises(ScopeBootstrapError):
            ensure_academy_scope()
        self.assertEqual(ApplicationScope.objects.filter(is_active=True).count(), 1)

    @override_settings(**{SETTING_NAME: "legacy-default"})
    def test_an_explicitly_declared_legacy_scope_is_used_unchanged(self):
        legacy = ApplicationScope.objects.create(
            slug="legacy-default", name="Legacy default", description="kept"
        )
        before = ApplicationScope.objects.values().get(pk=legacy.pk)

        result = ensure_academy_scope()

        self.assertFalse(result.created)
        self.assertEqual(result.scope.pk, legacy.pk)
        self.assertEqual(ApplicationScope.objects.count(), 1)
        self.assertEqual(ApplicationScope.objects.values().get(pk=legacy.pk), before)

    @override_settings(**{SETTING_NAME: "legacy-default"})
    def test_a_declared_scope_that_does_not_exist_is_never_created(self):
        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()
        self.assertIn("never created for you", str(raised.exception))
        self.assertEqual(ApplicationScope.objects.count(), 0)

    @override_settings(**{SETTING_NAME: "not a slug!"})
    def test_an_invalid_declared_slug_is_refused(self):
        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()
        self.assertIn("not a valid scope slug", str(raised.exception))

    @override_settings(**{SETTING_NAME: FOREIGN_APP})
    def test_declaring_a_foreign_scope_is_an_explicit_decision_and_is_honored(self):
        """The operator's word is the ownership evidence; Academy does not second-guess it."""
        foreign = foreign_scope()
        self.assertEqual(ensure_academy_scope().scope.pk, foreign.pk)

    def test_academy_scope_is_used_even_when_another_scope_is_also_active(self):
        academy = ApplicationScope.objects.create(slug=ACADEMY_SCOPE_SLUG, name=ACADEMY_SCOPE_NAME)
        foreign_scope()

        result = ensure_academy_scope()

        self.assertEqual(result.scope.pk, academy.pk)  # explicit identity, not a guess
        self.assertEqual(result.other_active_slugs, (FOREIGN_APP,))
        self.assertEqual(ApplicationScope.objects.filter(is_active=True).count(), 2)

    def test_several_foreign_active_scopes_are_never_resolved_by_guessing(self):
        foreign_scope("alpha")
        foreign_scope("beta")

        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()

        self.assertIn("'alpha', 'beta'", str(raised.exception))
        self.assertEqual(ApplicationScope.objects.count(), 2)
        self.assertFalse(ApplicationScope.objects.filter(slug=ACADEMY_SCOPE_SLUG).exists())

    def test_an_inactive_academy_scope_is_not_reactivated(self):
        ApplicationScope.objects.create(
            slug=ACADEMY_SCOPE_SLUG, name=ACADEMY_SCOPE_NAME, is_active=False
        )

        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()

        self.assertIn("will not reactivate", str(raised.exception))
        self.assertFalse(academy_scope_row().is_active)

    def test_an_inactive_academy_scope_is_not_replaced_by_a_foreign_active_one(self):
        ApplicationScope.objects.create(
            slug=ACADEMY_SCOPE_SLUG, name=ACADEMY_SCOPE_NAME, is_active=False
        )
        foreign_scope()
        with self.assertRaises(ScopeBootstrapError):
            ensure_academy_scope()
        self.assertFalse(academy_scope_row().is_active)

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
        foreign_scope("retired", is_active=False)

        result = ensure_academy_scope()

        self.assertTrue(result.created)
        self.assertFalse(ApplicationScope.objects.get(slug="retired").is_active)

    def test_academy_resources_in_a_foreign_scope_are_a_conflict_not_a_reuse(self):
        academy = ApplicationScope.objects.create(slug=ACADEMY_SCOPE_SLUG, name=ACADEMY_SCOPE_NAME)
        other = foreign_scope()
        provider = ProviderConfig.objects.create(name="p", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="m")
        agent = AgentProfile.objects.create(
            name="Input Normalizer", role="r", model_config=model, application_scope=other
        )

        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()

        self.assertIn("Input Normalizer", str(raised.exception))
        self.assertIn(f"'{FOREIGN_APP}'", str(raised.exception))
        agent.refresh_from_db()
        self.assertEqual(agent.application_scope_id, other.pk)  # never reassigned
        self.assertNotEqual(agent.application_scope_id, academy.pk)

    def test_the_refusal_names_where_academy_data_already_lives(self):
        """Evidence is offered as a hint to the operator, never acted on."""
        legacy = ApplicationScope.objects.create(slug="legacy-default", name="Legacy default")
        provider = ProviderConfig.objects.create(name="p", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="m")
        AgentProfile.objects.create(
            name="AI Hub Documentation Assistant", role="r",
            model_config=model, application_scope=legacy,
        )

        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()

        message = str(raised.exception)
        self.assertIn("AI Hub Documentation Assistant in 'legacy-default'", message)
        self.assertIn(f"{SETTING_NAME}=legacy-default", message)
        self.assertFalse(ApplicationScope.objects.filter(slug=ACADEMY_SCOPE_SLUG).exists())

    def inactive_legacy_scope_holding(self, *, agent=None, collection=None):
        """`legacy-default` is INACTIVE and holds Academy-named resources; no `academy`."""
        legacy = ApplicationScope.objects.create(
            slug="legacy-default", name="Legacy default", is_active=False
        )
        if agent:
            provider = ProviderConfig.objects.create(name="p", provider_type="training")
            model = ModelConfig.objects.create(provider=provider, model_name="m")
            AgentProfile.objects.create(
                name=agent, role="r", model_config=model, application_scope=legacy
            )
        if collection:
            KnowledgeCollection.objects.create(name=collection, application_scope=legacy)
        return legacy

    def test_bootstrap_does_not_create_academy_beside_academy_resources_in_an_inactive_scope(self):
        """No scope is active, so the old code created `academy`, then the seed refused."""
        legacy = self.inactive_legacy_scope_holding(agent="Input Normalizer")
        self.assertFalse(ApplicationScope.objects.filter(is_active=True).exists())

        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()

        message = str(raised.exception)
        self.assertIn("Input Normalizer", message)
        self.assertIn("'legacy-default'", message)
        self.assertIn(f"{SETTING_NAME}=", message)
        self.assertFalse(ApplicationScope.objects.filter(slug=ACADEMY_SCOPE_SLUG).exists())
        self.assertEqual(ApplicationScope.objects.count(), 1)
        legacy.refresh_from_db()
        self.assertFalse(legacy.is_active)  # never reactivated
        self.assertEqual(
            AgentProfile.objects.get(name="Input Normalizer").application_scope_id, legacy.pk
        )  # never reassigned

    def test_the_same_holds_for_an_academy_collection_in_an_inactive_scope(self):
        self.inactive_legacy_scope_holding(collection="AI Hub Official Docs")

        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()

        self.assertIn("AI Hub Official Docs", str(raised.exception))
        self.assertFalse(ApplicationScope.objects.filter(slug=ACADEMY_SCOPE_SLUG).exists())

    def test_bootstrap_command_leaves_no_scope_behind_for_the_seed_to_trip_over(self):
        self.inactive_legacy_scope_holding(agent="Input Normalizer")

        with self.assertRaises(CommandError):
            run("bootstrap_academy_scope")
        self.assertEqual(ApplicationScope.objects.count(), 1)

        with self.assertRaises(CommandError) as raised:  # the seed agrees, and writes nothing
            run("seed_academy_training_data")
        self.assertIn("Input Normalizer", str(raised.exception))
        self.assertEqual(ProviderConfig.objects.count(), 1)  # only the fixture's
        self.assertEqual(ApplicationScope.objects.count(), 1)

    def test_install_sequence_stops_at_the_bootstrap_with_nothing_created(self):
        self.inactive_legacy_scope_holding(agent="Input Normalizer")
        before = counts()

        with self.assertRaises(CommandError):
            install_sequence()

        self.assertEqual(counts(), before)

    @override_settings(**{SETTING_NAME: "legacy-default"})
    def test_declaring_that_inactive_scope_is_still_refused_not_reactivated(self):
        """Declared ownership is honored, but an inactive scope is never reactivated."""
        legacy = self.inactive_legacy_scope_holding(agent="Input Normalizer")

        with self.assertRaises(ScopeBootstrapError) as raised:
            ensure_academy_scope()

        self.assertIn("will not reactivate", str(raised.exception))
        legacy.refresh_from_db()
        self.assertFalse(legacy.is_active)

    def test_an_unrelated_inactive_scope_with_no_academy_resources_still_allows_creation(self):
        """The conflict check must not block a legitimate fresh install."""
        legacy = ApplicationScope.objects.create(
            slug="legacy-default", name="Legacy default", is_active=False
        )
        provider = ProviderConfig.objects.create(name="p", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="m")
        AgentProfile.objects.create(
            name="Somebody Else's Agent", role="r", model_config=model, application_scope=legacy
        )

        result = ensure_academy_scope()

        self.assertTrue(result.created)
        legacy.refresh_from_db()
        self.assertFalse(legacy.is_active)

    def test_every_seeded_academy_name_is_covered_by_the_conflict_check(self):
        """The conflict check lists names by hand; keep it in step with the seeds."""
        ensure_academy_scope()
        run("seed_academy_training_data")
        run("seed_ollama_agents")

        self.assertLessEqual(
            set(AgentProfile.objects.values_list("name", flat=True)),
            set(ACADEMY_AGENT_NAMES),
        )
        self.assertLessEqual(
            set(KnowledgeCollection.objects.values_list("name", flat=True)),
            set(ACADEMY_COLLECTION_NAMES),
        )


class SeedFailsBeforeWritingTests(TestCase):
    """The reported failure, and the half-seeded database it used to leave."""

    def assertNothingWritten(self):
        self.assertEqual(ProviderConfig.objects.count(), 0)
        self.assertEqual(ModelConfig.objects.count(), 0)
        self.assertEqual(AgentProfile.objects.count(), 0)
        self.assertEqual(KnowledgeCollection.objects.count(), 0)

    def test_seed_on_a_database_with_no_scope_writes_nothing_and_says_what_to_run(self):
        with self.assertRaises(CommandError) as raised:
            run("seed_academy_training_data")

        message = str(raised.exception)
        self.assertIn("does not exist yet", message)
        self.assertIn("python manage.py bootstrap_academy_scope", message)
        self.assertIn("Nothing was written by this seed", message)
        self.assertNothingWritten()
        self.assertEqual(ApplicationScope.objects.count(), 0)

    def test_ollama_seed_also_writes_nothing_without_a_scope(self):
        with self.assertRaises(CommandError):
            run("seed_ollama_agents")
        self.assertNothingWritten()

    def test_seed_never_creates_a_scope_itself(self):
        with self.assertRaises(CommandError):
            run("seed_academy_training_data")
        self.assertEqual(ApplicationScope.objects.count(), 0)

    def test_seed_refuses_a_single_foreign_scope_before_writing(self):
        """Without Academy ownership, the lone active scope must not receive the seeds."""
        foreign_scope()

        for command in ("seed_academy_training_data", "seed_ollama_agents"):
            with self.subTest(command=command):
                with self.assertRaises(CommandError) as raised:
                    run(command)
                self.assertIn(f"'{FOREIGN_APP}'", str(raised.exception))
                self.assertIn("will not adopt", str(raised.exception))
        self.assertNothingWritten()
        self.assertEqual(ApplicationScope.objects.count(), 1)

    def test_seed_refuses_several_foreign_scopes_before_writing(self):
        foreign_scope("alpha")
        foreign_scope("beta")

        with self.assertRaises(CommandError) as raised:
            run("seed_academy_training_data")

        self.assertIn("'alpha', 'beta'", str(raised.exception))
        self.assertNothingWritten()

    def test_seed_refuses_an_inactive_academy_scope_before_writing(self):
        ApplicationScope.objects.create(
            slug=ACADEMY_SCOPE_SLUG, name=ACADEMY_SCOPE_NAME, is_active=False
        )
        with self.assertRaises(CommandError) as raised:
            run("seed_academy_training_data")
        self.assertIn("will not reactivate", str(raised.exception))
        self.assertNothingWritten()
        self.assertFalse(academy_scope_row().is_active)

    def test_seed_refuses_academy_resources_held_by_another_scope_before_writing(self):
        ApplicationScope.objects.create(slug=ACADEMY_SCOPE_SLUG, name=ACADEMY_SCOPE_NAME)
        other = foreign_scope()
        provider = ProviderConfig.objects.create(name="p", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="m")
        AgentProfile.objects.create(
            name="Ticket Classifier", role="r", model_config=model, application_scope=other
        )

        with self.assertRaises(CommandError) as raised:
            run("seed_academy_training_data")

        self.assertIn("Ticket Classifier", str(raised.exception))
        self.assertEqual(ProviderConfig.objects.count(), 1)  # only the fixture's
        self.assertEqual(AgentProfile.objects.count(), 1)

    def test_bootstrap_command_reports_a_conflict_as_a_command_error(self):
        foreign_scope()
        with self.assertRaises(CommandError):
            run("bootstrap_academy_scope")
        self.assertFalse(ApplicationScope.objects.filter(slug=ACADEMY_SCOPE_SLUG).exists())

    def test_bootstrap_command_output_for_each_outcome(self):
        self.assertIn("Created application scope 'academy'", run("bootstrap_academy_scope"))
        self.assertIn("Using Academy's application scope", run("bootstrap_academy_scope"))
        foreign_scope()
        output = run("bootstrap_academy_scope")
        self.assertIn("Other application scopes are also active", output)
        self.assertIn(f"'{FOREIGN_APP}'", output)


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
        scope = academy_scope_row()
        self.assertFalse(AgentProfile.objects.exclude(application_scope=scope).exists())
        self.assertFalse(KnowledgeCollection.objects.exclude(application_scope=scope).exists())

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
        scope = academy_scope_row()
        scope.description = "Edited by an operator"
        scope.save(update_fields=["description"])

        install_sequence()

        agent.refresh_from_db()
        provider.refresh_from_db()
        scope.refresh_from_db()
        self.assertEqual(agent.system_prompt, "My own prompt.")
        self.assertEqual(provider.default_timeout, 77)
        self.assertEqual(scope.description, "Edited by an operator")

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

    @override_settings(**{SETTING_NAME: "legacy-default"})
    def test_a_declared_legacy_scope_receives_the_seeded_resources(self):
        legacy = ApplicationScope.objects.create(slug="legacy-default", name="Legacy default")

        install_sequence()

        self.assertEqual(ApplicationScope.objects.count(), 1)
        self.assertFalse(ApplicationScope.objects.filter(slug=ACADEMY_SCOPE_SLUG).exists())
        self.assertGreaterEqual(counts()["agents"], 3)
        self.assertFalse(AgentProfile.objects.exclude(application_scope=legacy).exists())
        self.assertFalse(KnowledgeCollection.objects.exclude(application_scope=legacy).exists())

    def test_an_undeclared_legacy_scope_holding_academy_data_is_not_silently_continued(self):
        """Existing Academy data in `legacy-default` needs the explicit declaration."""
        legacy = ApplicationScope.objects.create(slug="legacy-default", name="Legacy default")
        provider = ProviderConfig.objects.create(name="p", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="m")
        AgentProfile.objects.create(
            name="AI Hub Documentation Assistant", role="r",
            model_config=model, application_scope=legacy,
        )
        before = counts()

        with self.assertRaises(CommandError) as raised:
            install_sequence()

        self.assertIn(f"{SETTING_NAME}=legacy-default", str(raised.exception))
        self.assertEqual(counts(), before)

    def test_a_foreign_scope_stops_the_installation_before_any_seeding(self):
        foreign_scope()

        with self.assertRaises(CommandError):
            install_sequence()

        snapshot = counts()
        self.assertEqual(snapshot["providers"], 0)
        self.assertEqual(snapshot["agents"], 0)
        self.assertEqual(snapshot["pages"], 0)
        self.assertEqual(snapshot["scopes"], 1)

    def test_ambiguous_scopes_stop_the_installation_before_any_seeding(self):
        foreign_scope("alpha")
        foreign_scope("beta")

        with self.assertRaises(CommandError):
            install_sequence()

        self.assertEqual(counts()["providers"], 0)
        self.assertEqual(counts()["pages"], 0)
        self.assertEqual(counts()["scopes"], 2)

    def test_academy_installs_into_its_own_scope_beside_a_foreign_one(self):
        """The foreign scope pre-dates Academy's; Academy never touches its data."""
        academy = ApplicationScope.objects.create(slug=ACADEMY_SCOPE_SLUG, name=ACADEMY_SCOPE_NAME)
        foreign_scope()

        install_sequence()

        self.assertFalse(AgentProfile.objects.exclude(application_scope=academy).exists())
        self.assertFalse(KnowledgeCollection.objects.exclude(application_scope=academy).exists())


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

    def test_setup_refuses_a_foreign_scope_end_to_end_and_writes_nothing(self):
        """Real subprocesses: another application's scope is never adopted by setup."""
        import setup_dev

        with disposable_database() as db_path:
            python = Path(sys.executable)
            with patch("builtins.print"):
                setup_dev.run_migrations(python)
                setup_dev._manage(
                    python, "shell", "-c",
                    "from ai_hub.models import ApplicationScope as S;"
                    "S.objects.create(slug='customer-portal', name='Customer Portal')",
                )

            with patch("sys.stderr", new_callable=StringIO) as stderr:
                with self.assertRaises(SystemExit):
                    setup_dev.bootstrap_scope(python)
                with self.assertRaises(SystemExit):
                    setup_dev.seed_data(python)

            output = stderr.getvalue()
            self.assertIn("will not adopt", output)
            self.assertIn("customer-portal", output)
            self.assertEqual(table_count(db_path, "ai_hub_applicationscope"), 1)
            self.assertEqual(table_count(db_path, "ai_hub_agentprofile"), 0)
            self.assertEqual(table_count(db_path, "ai_hub_providerconfig"), 0)
            self.assertEqual(table_count(db_path, "ai_hub_knowledgecollection"), 0)

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
        self.assertIn("does not exist yet", text)
        self.assertIn("bootstrap_academy_scope", text)

    def documents(self):
        return {
            "README.md": self.readme(),
            "docs_source/15_ACADEMY_FEATURES.md": (
                PROJECT_ROOT / "docs_source" / "15_ACADEMY_FEATURES.md"
            ).read_text(encoding="utf-8"),
        }

    def test_docs_do_not_tell_people_to_disable_other_applications(self):
        for name, text in self.documents().items():
            with self.subTest(document=name):
                self.assertNotIn("do not belong to this installation", text)
                self.assertNotIn("leaving exactly one", text)
                self.assertNotIn("clear *Is active* on", text)

    def test_docs_describe_the_safe_alternatives_and_the_explicit_declaration(self):
        for name, text in self.documents().items():
            with self.subTest(document=name):
                self.assertIn("dedicated", text)
                self.assertIn("ACADEMY_APPLICATION_SCOPE_SLUG", text)
                self.assertIn("legacy-default", text)
                flattened = " ".join(text.split())
                self.assertIn("never adopts", flattened)
                self.assertIn("another application", flattened)

    def test_the_lab_evaluator_step_is_documented_as_all_or_nothing(self):
        text = " ".join(self.documents()["docs_source/15_ACADEMY_FEATURES.md"].split())
        self.assertIn("seed_lab_exercises --create-evaluator", text)
        self.assertIn("all-or-nothing", text)
        self.assertIn("non-zero status", text)
        self.assertNotIn("Known limitation", text)
        self.assertNotIn("Do not use `--create-evaluator` yet", text)
        self.assertNotIn("issues/26", text)

    def test_markdown_tables_in_the_academy_docs_are_not_torn_apart_by_pipes(self):
        """GitHub splits a table row on EVERY unescaped `|`, even inside `code`.

        `<slug|id>` in a cell once split four command rows into extra columns. A
        row whose cell count differs from its header's renders broken.
        """
        cell_split = re.compile(r"(?<!\\)\|")
        separator = re.compile(r"^\|[\s:|-]+\|?\s*$")

        def cells(line):
            return len(cell_split.split(line.strip().strip("|")))

        paths = [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "DEMO_SCRIPT.md",
            *sorted((PROJECT_ROOT / "docs_source").glob("*.md")),
        ]
        checked = 0
        for path in paths:
            lines = path.read_text(encoding="utf-8").splitlines()
            in_fence = False
            index = 0
            while index < len(lines):
                line = lines[index]
                if line.lstrip().startswith("```"):
                    in_fence = not in_fence
                is_table_start = (
                    not in_fence
                    and line.startswith("|")
                    and index + 1 < len(lines)
                    and separator.match(lines[index + 1])
                )
                if not is_table_start:
                    index += 1
                    continue
                width = cells(line)
                index += 2
                while index < len(lines) and lines[index].startswith("|"):
                    checked += 1
                    self.assertEqual(
                        cells(lines[index]), width,
                        f"{path.name}:{index + 1} has {cells(lines[index])} cells "
                        f"but its table header has {width}: {lines[index][:90]}",
                    )
                    index += 1
        self.assertGreater(checked, 30)

    def test_env_template_documents_the_ownership_declaration_as_optional(self):
        template = (PROJECT_ROOT / ".one-env.example").read_text(encoding="utf-8")
        self.assertIn("# ACADEMY_APPLICATION_SCOPE_SLUG=academy", template)  # commented out
