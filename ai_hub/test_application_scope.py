"""S-14 — ApplicationScope ownership foundation.

Scope of these tests, deliberately narrow:

  * the root model behaves as a root model;
  * the three root-owned resources REQUIRE an owner;
  * `PROTECT` means a scope that still owns resources cannot be deleted;
  * migration 0023 adopts a pre-existing corpus without losing a row;
  * there is NO runtime default;
  * existing behaviour inside one scope is unchanged.

What these tests deliberately do NOT assert: that scope ownership prevents
cross-scope Knowledge access. It does not, and one test below characterizes
that as an expected S-14 limitation. Enforcement is S-15's effective
authorization boundary.
"""

from decimal import Decimal

from django.core.exceptions import ValidationError
from django.db import IntegrityError, connection, transaction
from django.db.migrations.executor import MigrationExecutor
from django.test import TestCase, TransactionTestCase

from ai_hub.models import (
    AgentProfile,
    ApplicationScope,
    GameWorkspace,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    ProviderConfig,
)
from ai_hub.services import knowledge_retrieval
from ai_hub.services.application_scope import require_single_active_scope
from ai_hub.test_application_scope_helpers import test_scope


MIGRATION_BEFORE = ("ai_hub", "0022_knowledge_lifecycle_event")
MIGRATION_AFTER = ("ai_hub", "0023_application_scope")
LEGACY_SLUG = "legacy-default"


def make_model_config():
    provider = ProviderConfig.objects.create(
        name="Scope Provider", provider_type=ProviderConfig.ProviderType.TRAINING
    )
    return ModelConfig.objects.create(
        provider=provider, model_name="training",
        temperature_default=Decimal("0.10"),
    )


# ---------------------------------------------------------------------------
# The root model
# ---------------------------------------------------------------------------

class ApplicationScopeModelTests(TestCase):
    def test_a_scope_can_be_created_and_is_active_by_default(self):
        scope = ApplicationScope.objects.create(name="Alpha", slug="alpha")
        self.assertTrue(scope.is_active)
        self.assertIsNotNone(scope.created_at)
        self.assertIsNotNone(scope.updated_at)

    def test_str_is_the_name(self):
        scope = ApplicationScope.objects.create(name="Alpha", slug="alpha")
        self.assertEqual(str(scope), "Alpha")

    def test_slug_is_globally_unique(self):
        ApplicationScope.objects.create(name="Alpha", slug="alpha")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ApplicationScope.objects.create(name="Other", slug="alpha")

    def test_name_is_globally_unique(self):
        ApplicationScope.objects.create(name="Alpha", slug="alpha")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ApplicationScope.objects.create(name="Alpha", slug="other")

    def test_a_scope_can_be_deactivated_without_being_deleted(self):
        scope = ApplicationScope.objects.create(name="Alpha", slug="alpha")
        scope.is_active = False
        scope.save(update_fields=["is_active"])
        scope.refresh_from_db()
        self.assertFalse(scope.is_active)
        self.assertTrue(ApplicationScope.objects.filter(pk=scope.pk).exists())

    def test_blank_name_and_slug_are_rejected_by_validation(self):
        for field, kwargs in (
            ("name", {"name": "   ", "slug": "alpha"}),
            ("slug", {"name": "Alpha", "slug": ""}),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValidationError) as raised:
                    ApplicationScope(**kwargs).full_clean()
                self.assertIn(field, raised.exception.error_dict)

    def test_the_scope_carries_no_host_or_domain_concept(self):
        """Core must never learn what a scope MEANS."""
        field_names = {field.name for field in ApplicationScope._meta.get_fields()}
        for forbidden in (
            "client", "project", "academy", "advisory", "tenant",
            "pdd", "assessment", "customer", "organisation", "organization",
        ):
            self.assertNotIn(forbidden, field_names)


class FirstScopeCreationPathTests(TestCase):
    """A fresh installation must have an explicit operator path to its FIRST scope.

    This is not cosmetic. Core creates no scope implicitly, there is no bootstrap
    and no runtime default, so on a brand-new database EVERY root-owned creation
    path refuses until an operator makes a scope by hand. If the admin
    registration were ever dropped, a fresh install would be unable to create a
    knowledge collection, an agent or a workspace at all.
    """

    def test_application_scope_is_registered_in_the_admin(self):
        from django.contrib import admin as django_admin

        self.assertIn(ApplicationScope, django_admin.site._registry)

    def test_the_admin_add_url_reverses(self):
        from django.urls import reverse

        self.assertTrue(reverse("admin:ai_hub_applicationscope_add"))
        self.assertTrue(reverse("admin:ai_hub_applicationscope_changelist"))

    def test_the_admin_can_create_the_very_first_scope(self):
        """End to end on an empty database, through the real admin form."""
        from django.contrib.auth import get_user_model

        self.assertEqual(ApplicationScope.objects.count(), 0)
        with self.assertRaises(ValidationError):
            require_single_active_scope()

        operator = get_user_model().objects.create_superuser(
            username="scope-operator", email="", password="x" * 14
        )
        self.client.force_login(operator)

        response = self.client.post(
            "/admin/ai_hub/applicationscope/add/",
            {"name": "First App", "slug": "first-app", "description": "",
             "is_active": "on"},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)

        scope = ApplicationScope.objects.get(slug="first-app")
        self.assertTrue(scope.is_active)
        # ... and the compatibility bridge now resolves, so the surfaces that
        # depend on it come to life exactly once a scope exists.
        self.assertEqual(require_single_active_scope(), scope)

    def test_the_admin_page_is_visible_in_the_index(self):
        """Not hidden: an operator has to be able to FIND it."""
        from django.contrib import admin as django_admin

        model_admin = django_admin.site._registry[ApplicationScope]
        self.assertNotIn(
            "AIHubHideFromIndexMixin",
            [base.__name__ for base in type(model_admin).__mro__],
        )


# ---------------------------------------------------------------------------
# Root-owned resources
# ---------------------------------------------------------------------------

class RootOwnershipTests(TestCase):
    def setUp(self):
        self.scope = ApplicationScope.objects.create(name="Alpha", slug="alpha")
        self.model_config = make_model_config()

    def test_all_three_root_resources_can_be_owned(self):
        collection = KnowledgeCollection.objects.create(
            name="C", application_scope=self.scope
        )
        agent = AgentProfile.objects.create(
            name="A", role="r", model_config=self.model_config,
            application_scope=self.scope,
        )
        workspace = GameWorkspace.objects.create(
            name="W", application_scope=self.scope
        )
        for resource in (collection, agent, workspace):
            with self.subTest(resource=type(resource).__name__):
                self.assertEqual(resource.application_scope_id, self.scope.pk)

    def test_ownership_is_mandatory_for_all_three(self):
        cases = (
            (KnowledgeCollection, {"name": "C"}),
            (AgentProfile, {"name": "A", "role": "r",
                            "model_config": self.model_config}),
            (GameWorkspace, {"name": "W"}),
        )
        for model, kwargs in cases:
            with self.subTest(model=model.__name__):
                with self.assertRaises(IntegrityError):
                    with transaction.atomic():
                        model.objects.create(**kwargs)

    def test_a_scope_owning_resources_cannot_be_deleted(self):
        """PROTECT: deleting a security boundary is never a side effect."""
        cases = (
            (KnowledgeCollection, {"name": "C"}),
            (AgentProfile, {"name": "A", "role": "r",
                            "model_config": self.model_config}),
            (GameWorkspace, {"name": "W"}),
        )
        for model, kwargs in cases:
            with self.subTest(model=model.__name__):
                scope = ApplicationScope.objects.create(
                    name=f"S{model.__name__}", slug=f"s-{model.__name__.lower()}"
                )
                model.objects.create(application_scope=scope, **kwargs)
                from django.db.models import ProtectedError

                with self.assertRaises(ProtectedError):
                    with transaction.atomic():
                        scope.delete()
                self.assertTrue(
                    ApplicationScope.objects.filter(pk=scope.pk).exists()
                )

    def test_an_empty_scope_can_be_deleted(self):
        scope = ApplicationScope.objects.create(name="Empty", slug="empty")
        scope.delete()
        self.assertFalse(ApplicationScope.objects.filter(slug="empty").exists())

    def test_related_names_resolve_from_the_scope(self):
        KnowledgeCollection.objects.create(name="C", application_scope=self.scope)
        AgentProfile.objects.create(
            name="A", role="r", model_config=self.model_config,
            application_scope=self.scope,
        )
        GameWorkspace.objects.create(name="W", application_scope=self.scope)

        self.assertEqual(self.scope.knowledge_collections.count(), 1)
        self.assertEqual(self.scope.agent_profiles.count(), 1)
        self.assertEqual(self.scope.game_workspaces.count(), 1)

    def test_the_required_S14_invariant_is_answerable_without_inference(self):
        """Which scope owns this? - answered by a field, not by a name."""
        collection = KnowledgeCollection.objects.create(
            name="Ownable", application_scope=self.scope
        )
        agent = AgentProfile.objects.create(
            name="Ownable Agent", role="r", model_config=self.model_config,
            application_scope=self.scope,
        )
        workspace = GameWorkspace.objects.create(
            name="Ownable Workspace", application_scope=self.scope
        )
        for resource in (collection, agent, workspace):
            with self.subTest(resource=type(resource).__name__):
                fresh = type(resource).objects.get(pk=resource.pk)
                self.assertEqual(fresh.application_scope, self.scope)


# ---------------------------------------------------------------------------
# No runtime default
# ---------------------------------------------------------------------------

class NoRuntimeDefaultTests(TestCase):
    """The legacy scope is a migration artifact, never a fallback."""

    def setUp(self):
        self.model_config = make_model_config()

    def test_no_field_declares_a_default_scope(self):
        for model in (KnowledgeCollection, AgentProfile, GameWorkspace):
            with self.subTest(model=model.__name__):
                field = model._meta.get_field("application_scope")
                self.assertFalse(field.has_default())
                self.assertFalse(field.null)
                self.assertFalse(field.blank)

    def test_a_legacy_scope_present_does_not_get_used_implicitly(self):
        ApplicationScope.objects.create(name="Legacy default", slug=LEGACY_SLUG)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                KnowledgeCollection.objects.create(name="Unowned")

    def test_no_production_module_resolves_the_legacy_scope_by_name(self):
        """The compatibility scope must not be reachable at runtime by slug."""
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parent
        offenders = []
        for path in root.rglob("*.py"):
            rel = path.relative_to(root).as_posix()
            if rel.startswith(("migrations/", "test")) or rel == "tests.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and node.value == LEGACY_SLUG:
                    offenders.append(rel)
        self.assertEqual(offenders, [], f"legacy slug referenced at runtime: {offenders}")

    def test_the_scope_helper_refuses_rather_than_guessing(self):
        with self.assertRaises(ValidationError):
            require_single_active_scope()

        first = ApplicationScope.objects.create(name="One", slug="one")
        self.assertEqual(require_single_active_scope(), first)

        ApplicationScope.objects.create(name="Two", slug="two")
        with self.assertRaises(ValidationError) as raised:
            require_single_active_scope()
        self.assertIn("cannot be inferred", str(raised.exception))

    def test_the_helper_ignores_inactive_scopes(self):
        active = ApplicationScope.objects.create(name="On", slug="on")
        ApplicationScope.objects.create(name="Off", slug="off", is_active=False)
        self.assertEqual(require_single_active_scope(), active)


# ---------------------------------------------------------------------------
# Global uniqueness is deliberately RETAINED
# ---------------------------------------------------------------------------

class GlobalUniquenessRetainedTests(TestCase):
    """Stricter than isolation requires, on purpose, for S-14 only.

    Name-based resolution paths still assume a global namespace. Relaxing this
    to UniqueConstraint(application_scope, name) before those paths are
    scope-aware would create ambiguous identity resolution.
    """

    def setUp(self):
        self.alpha = ApplicationScope.objects.create(name="Alpha", slug="alpha")
        self.beta = ApplicationScope.objects.create(name="Beta", slug="beta")
        self.model_config = make_model_config()

    def test_collection_names_remain_globally_unique_across_scopes(self):
        KnowledgeCollection.objects.create(name="Shared", application_scope=self.alpha)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                KnowledgeCollection.objects.create(
                    name="Shared", application_scope=self.beta
                )

    def test_agent_names_remain_globally_unique_across_scopes(self):
        AgentProfile.objects.create(
            name="Shared", role="r", model_config=self.model_config,
            application_scope=self.alpha,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                AgentProfile.objects.create(
                    name="Shared", role="r", model_config=self.model_config,
                    application_scope=self.beta,
                )

    def test_workspace_names_remain_globally_unique_across_scopes(self):
        GameWorkspace.objects.create(name="Shared", application_scope=self.alpha)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                GameWorkspace.objects.create(
                    name="Shared", application_scope=self.beta
                )

    def test_no_scoped_unique_constraint_was_introduced_yet(self):
        for model in (KnowledgeCollection, AgentProfile, GameWorkspace):
            with self.subTest(model=model.__name__):
                scoped = [
                    constraint for constraint in model._meta.constraints
                    if "application_scope" in getattr(constraint, "fields", ())
                ]
                self.assertEqual(scoped, [])
                self.assertTrue(model._meta.get_field("name").unique)


# ---------------------------------------------------------------------------
# Characterization: what S-14 does NOT fix
# ---------------------------------------------------------------------------

class CrossScopeCharacterizationTests(TestCase):
    """EXPECTED S-14 LIMITATION - blocked by S-15, not by S-14.

    Ownership is now a fact. Enforcement is not. This test exists so the gap is
    recorded as a deliberate, measured limitation rather than discovered later
    as a surprise, and so S-15 has a test to invert.
    """

    def setUp(self):
        self.scope_a = ApplicationScope.objects.create(name="App A", slug="app-a")
        self.scope_b = ApplicationScope.objects.create(name="App B", slug="app-b")
        model_config = make_model_config()

        self.collection_a = KnowledgeCollection.objects.create(
            name="A Knowledge", application_scope=self.scope_a
        )
        self.collection_b = KnowledgeCollection.objects.create(
            name="B Knowledge", application_scope=self.scope_b
        )
        for collection, marker in (
            (self.collection_a, "ALPHA"), (self.collection_b, "BETA"),
        ):
            document = KnowledgeDocument.objects.create(
                collection=collection, title=f"{marker} doc",
                curated_text=f"{marker} widgets",
                status=KnowledgeDocument.Status.ACTIVE,
            )
            KnowledgeDocumentChunk.objects.create(
                document=document, chunk_index=1,
                section_title=f"{marker} s1", content=f"{marker} widgets payload",
            )

        self.agent_a = AgentProfile.objects.create(
            name="A Agent", role="r", model_config=model_config,
            application_scope=self.scope_a,
        )
        self.agent_a.knowledge_collections.add(self.collection_a)

    def test_cross_scope_m2m_assignment_is_still_possible_in_S14(self):
        self.agent_a.knowledge_collections.add(self.collection_b)
        self.agent_a.full_clean()   # no validation objects to it, by design
        self.assertIn(
            self.collection_b,
            self.agent_a.knowledge_collections.all(),
        )

    def test_cross_scope_retrieval_is_still_reachable_in_S14(self):
        self.agent_a.knowledge_collections.add(self.collection_b)
        result = knowledge_retrieval.search_knowledge(
            self.agent_a, query="widgets", limit=10
        )
        collections = {row["collection"] for row in result["results"]}
        self.assertIn("B Knowledge", collections)

    def test_ownership_facts_are_nonetheless_unambiguous(self):
        """S-14's actual deliverable: S-15 has something to enforce."""
        self.agent_a.knowledge_collections.add(self.collection_b)
        self.assertEqual(self.agent_a.application_scope, self.scope_a)
        self.assertEqual(self.collection_b.application_scope, self.scope_b)
        self.assertNotEqual(
            self.agent_a.application_scope_id,
            self.collection_b.application_scope_id,
        )


# ---------------------------------------------------------------------------
# Existing behaviour inside ONE scope is unchanged
# ---------------------------------------------------------------------------

class SingleScopeBehaviourUnchangedTests(TestCase):
    def setUp(self):
        self.scope = test_scope("Unchanged", slug="unchanged")
        model_config = make_model_config()
        self.collection = KnowledgeCollection.objects.create(
            name="Unchanged Knowledge", application_scope=self.scope
        )
        self.other = KnowledgeCollection.objects.create(
            name="Unassigned Knowledge", application_scope=self.scope
        )
        for collection, marker in (
            (self.collection, "ASSIGNED"), (self.other, "UNASSIGNED"),
        ):
            document = KnowledgeDocument.objects.create(
                collection=collection, title=f"{marker} doc",
                curated_text=f"{marker} widgets",
                status=KnowledgeDocument.Status.ACTIVE,
            )
            KnowledgeDocumentChunk.objects.create(
                document=document, chunk_index=1,
                section_title=f"{marker} s1", content=f"{marker} widgets payload",
            )
        self.agent = AgentProfile.objects.create(
            name="Unchanged Agent", role="r", model_config=model_config,
            application_scope=self.scope,
        )
        self.agent.knowledge_collections.add(self.collection)

    def test_assignment_still_grants_access(self):
        result = knowledge_retrieval.search_knowledge(
            self.agent, query="widgets", limit=10
        )
        self.assertEqual(
            {row["collection"] for row in result["results"]},
            {"Unchanged Knowledge"},
        )

    def test_unassigned_collections_in_the_same_scope_remain_invisible(self):
        """Scope ownership did not widen anything."""
        chunks = knowledge_retrieval._accessible_chunks(self.agent)
        self.assertEqual(chunks.count(), 1)
        self.assertEqual(KnowledgeDocumentChunk.objects.count(), 2)

    def test_the_out_of_scope_collection_error_is_still_uniform(self):
        for label, collection_id in (
            ("unassigned but existing", self.other.pk),
            ("nonexistent", 999_999),
        ):
            with self.subTest(case=label):
                with self.assertRaises(ValidationError) as raised:
                    knowledge_retrieval.search_knowledge(
                        self.agent, query="widgets", collection_id=collection_id
                    )
                self.assertEqual(
                    raised.exception.messages[0],
                    "Knowledge collection is not accessible to this agent.",
                )

    def test_lifecycle_fields_are_untouched_by_S14(self):
        document = self.collection.documents.get()
        self.assertEqual(
            document.chunk_authority_mode,
            KnowledgeDocument.ChunkAuthorityMode.UNKNOWN,
        )
        self.assertEqual(document.generation_input_fingerprint, "")
        self.assertIsNone(document.generator_version)


# ---------------------------------------------------------------------------
# Migration 0023
# ---------------------------------------------------------------------------

class ApplicationScopeMigrationTests(TransactionTestCase):
    """Forward and reverse behaviour of the staged 0023 migration.

    TransactionTestCase because a real schema migration runs here.
    """

    available_apps = None

    def _migrate(self, target):
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate([target])
        executor.loader.build_graph()
        return executor

    def tearDown(self):
        # Restore the graph leaf so later tests never inherit a partial schema.
        leaf = MigrationExecutor(connection).loader.graph.leaf_nodes("ai_hub")[0]
        self._migrate(leaf)

    def _seed_pre_s14_corpus(self, apps_state):
        """Create rows through the HISTORICAL 0022 model state."""
        ProviderConfigHistorical = apps_state.get_model("ai_hub", "ProviderConfig")
        ModelConfigHistorical = apps_state.get_model("ai_hub", "ModelConfig")
        KnowledgeCollectionHistorical = apps_state.get_model(
            "ai_hub", "KnowledgeCollection"
        )
        AgentProfileHistorical = apps_state.get_model("ai_hub", "AgentProfile")
        GameWorkspaceHistorical = apps_state.get_model("ai_hub", "GameWorkspace")

        provider = ProviderConfigHistorical.objects.create(
            name="Legacy Provider", provider_type="training"
        )
        model_config = ModelConfigHistorical.objects.create(
            provider=provider, model_name="training"
        )
        for index in range(3):
            KnowledgeCollectionHistorical.objects.create(name=f"Legacy Coll {index}")
        for index in range(2):
            AgentProfileHistorical.objects.create(
                name=f"Legacy Agent {index}", role="r", model_config=model_config
            )
        GameWorkspaceHistorical.objects.create(name="Legacy Workspace")

    def test_forward_migration_adopts_every_pre_existing_row(self):
        executor = self._migrate(MIGRATION_BEFORE)
        self._seed_pre_s14_corpus(executor.loader.project_state(MIGRATION_BEFORE).apps)

        executor = self._migrate(MIGRATION_AFTER)
        state = executor.loader.project_state(MIGRATION_AFTER).apps

        Scope = state.get_model("ai_hub", "ApplicationScope")
        scopes = list(Scope.objects.all())
        self.assertEqual(len(scopes), 1, "exactly one deterministic legacy scope")
        self.assertEqual(scopes[0].slug, LEGACY_SLUG)

        counts = {}
        for name, expected in (
            ("KnowledgeCollection", 3),
            ("AgentProfile", 2),
            ("GameWorkspace", 1),
        ):
            model = state.get_model("ai_hub", name)
            counts[name] = model.objects.count()
            self.assertEqual(counts[name], expected, f"{name} rows preserved")
            self.assertEqual(
                model.objects.filter(application_scope__isnull=True).count(), 0,
                f"every {name} has an owner",
            )
            self.assertEqual(
                model.objects.filter(application_scope=scopes[0]).count(), expected
            )

    def test_a_fresh_database_gets_no_legacy_scope(self):
        """A new install should not inherit a compatibility artifact."""
        self._migrate(MIGRATION_BEFORE)
        executor = self._migrate(MIGRATION_AFTER)
        state = executor.loader.project_state(MIGRATION_AFTER).apps
        Scope = state.get_model("ai_hub", "ApplicationScope")
        self.assertEqual(Scope.objects.count(), 0)

    def test_reverse_migration_restores_the_pre_s14_state_without_data_loss(self):
        executor = self._migrate(MIGRATION_BEFORE)
        self._seed_pre_s14_corpus(executor.loader.project_state(MIGRATION_BEFORE).apps)
        self._migrate(MIGRATION_AFTER)

        executor = self._migrate(MIGRATION_BEFORE)
        state = executor.loader.project_state(MIGRATION_BEFORE).apps
        for name, expected in (
            ("KnowledgeCollection", 3),
            ("AgentProfile", 2),
            ("GameWorkspace", 1),
        ):
            model = state.get_model("ai_hub", name)
            self.assertEqual(model.objects.count(), expected, f"{name} survived")
        self.assertFalse(
            any(
                table.endswith("applicationscope")
                for table in connection.introspection.table_names()
            )
        )
