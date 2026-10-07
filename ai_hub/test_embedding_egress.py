"""S-17 — the embedding egress policy boundary.

The question this module guards: may this application's corpus or query text be
sent to this provider at all? It is answered BEFORE any content exists at the
boundary, and it is answered independently of whether an Agent may read the
Knowledge in the first place (that is S-15's `EffectiveKnowledgeScope`).

Nothing here embeds anything. No provider is called. No vector exists.
"""

import inspect
from decimal import Decimal
from unittest import mock

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import TestCase

from ai_hub.models import (
    AgentProfile,
    ApplicationScope,
    KnowledgeCollection,
    ModelConfig,
    ProviderConfig,
    ProviderGrant,
)
from ai_hub.services import embedding_egress
from ai_hub.services.embedding_egress import (
    PAYLOAD_CORPUS,
    PAYLOAD_QUERY,
    EmbeddingAccessDecision,
    ReasonCode,
    resolve_embedding_access,
)


LOCALITY = ProviderConfig.DeclaredLocality
POLICY = KnowledgeCollection.ExternalEmbeddingEgressPolicy


class EgressFixtureMixin:
    def make_scope(self, name="App A", slug="app-a", **kwargs):
        return ApplicationScope.objects.create(name=name, slug=slug, **kwargs)

    def make_provider(self, name, provider_type, locality, *, base_url="", **kwargs):
        return ProviderConfig.objects.create(
            name=name, provider_type=provider_type,
            declared_locality=locality, base_url=base_url, **kwargs
        )

    def make_collection(self, scope, name="A Knowledge", **kwargs):
        return KnowledgeCollection.objects.create(
            name=name, application_scope=scope, **kwargs
        )

    def grant(self, scope, provider, *, allow=True):
        return ProviderGrant.objects.create(
            application_scope=scope, provider=provider, allow_embeddings=allow
        )

    def decide(self, scope, provider, collection, kind):
        return resolve_embedding_access(
            scope, provider, collection=collection, payload_kind=kind
        )


# ---------------------------------------------------------------------------
# Declared locality: an operator FACT, never an inference
# ---------------------------------------------------------------------------

class DeclaredLocalityTests(EgressFixtureMixin, TestCase):
    def test_the_default_is_unknown_and_fails_closed(self):
        provider = ProviderConfig.objects.create(name="P", provider_type="openai")
        self.assertEqual(provider.declared_locality, LOCALITY.UNKNOWN)

    def test_ollama_may_be_declared_external(self):
        """Self-hosted vendor labels say nothing about the trust boundary."""
        scope = self.make_scope()
        provider = self.make_provider(
            "Hosted Ollama", "ollama", LOCALITY.EXTERNAL,
            base_url="https://ollama.example.com",
        )
        collection = self.make_collection(scope)
        self.grant(scope, provider)

        decision = self.decide(scope, provider, collection, PAYLOAD_CORPUS)
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.requires_external_egress)
        self.assertEqual(decision.reason_code, ReasonCode.SCOPE_DENIES_EXTERNAL_CORPUS)

    def test_openai_may_be_declared_local(self):
        """A vendor name is not a location either."""
        scope = self.make_scope()
        provider = self.make_provider(
            "Internal OpenAI Gateway", "openai", LOCALITY.LOCAL,
            base_url="https://gateway.internal.example.com",
        )
        collection = self.make_collection(scope)
        self.grant(scope, provider)

        decision = self.decide(scope, provider, collection, PAYLOAD_CORPUS)
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_external_egress)
        self.assertEqual(decision.reason_code, ReasonCode.ALLOWED_LOCAL)

    def test_a_remote_looking_url_declared_local_is_local(self):
        scope = self.make_scope()
        provider = self.make_provider(
            "Remote Looking", "other", LOCALITY.LOCAL,
            base_url="https://api.somewhere-far-away.example.net/v1",
        )
        collection = self.make_collection(scope)
        self.grant(scope, provider)
        self.assertTrue(
            self.decide(scope, provider, collection, PAYLOAD_CORPUS).allowed
        )

    def test_a_localhost_looking_url_declared_external_is_external(self):
        scope = self.make_scope()
        provider = self.make_provider(
            "Localhost Looking", "ollama", LOCALITY.EXTERNAL,
            base_url="http://127.0.0.1:11434",
        )
        collection = self.make_collection(scope)
        self.grant(scope, provider)

        decision = self.decide(scope, provider, collection, PAYLOAD_CORPUS)
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.requires_external_egress)

    def test_the_service_performs_no_network_or_dns_inference(self):
        """Source-level: no hostname parsing, no DNS, no address classification."""
        source = inspect.getsource(embedding_egress)
        for forbidden in (
            "socket", "gethostbyname", "urlparse", "ipaddress", "127.0.0.1",
            "localhost", "resolve(", "requests.", "httpx",
        ):
            self.assertNotIn(forbidden, source, f"locality inference via {forbidden!r}")

    def test_the_service_never_reads_provider_type_or_base_url(self):
        source = inspect.getsource(embedding_egress)
        for forbidden in ("provider_type", "base_url", "api_key"):
            self.assertNotIn(
                f".{forbidden}", source,
                f"{forbidden} must not participate in the decision",
            )


# ---------------------------------------------------------------------------
# ApplicationScope egress fields and the query ⊆ corpus invariant
# ---------------------------------------------------------------------------

class ScopeEgressPolicyTests(EgressFixtureMixin, TestCase):
    def test_a_fresh_scope_denies_both(self):
        scope = self.make_scope()
        self.assertFalse(scope.allow_external_embedding_corpus_egress)
        self.assertFalse(scope.allow_external_embedding_query_egress)

    def test_model_validation_refuses_query_broader_than_corpus(self):
        scope = ApplicationScope(
            name="Bad", slug="bad",
            allow_external_embedding_corpus_egress=False,
            allow_external_embedding_query_egress=True,
        )
        with self.assertRaises(ValidationError) as raised:
            scope.full_clean()
        self.assertIn(
            "allow_external_embedding_query_egress", raised.exception.error_dict
        )

    def test_the_database_constraint_refuses_it_too(self):
        """Admin and full_clean() are bypassable; the constraint is not."""
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ApplicationScope.objects.create(
                    name="Raw", slug="raw",
                    allow_external_embedding_corpus_egress=False,
                    allow_external_embedding_query_egress=True,
                )

    def test_the_constraint_also_refuses_a_raw_update(self):
        scope = self.make_scope()
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                ApplicationScope.objects.filter(pk=scope.pk).update(
                    allow_external_embedding_query_egress=True
                )

    def test_the_three_valid_combinations_are_accepted(self):
        cases = (
            ("deny-deny", False, False),
            ("corpus-only", True, False),
            ("both", True, True),
        )
        for index, (label, corpus, query) in enumerate(cases):
            with self.subTest(case=label):
                scope = ApplicationScope(
                    name=f"S{index}", slug=f"s{index}",
                    allow_external_embedding_corpus_egress=corpus,
                    allow_external_embedding_query_egress=query,
                )
                scope.full_clean()
                scope.save()


# ---------------------------------------------------------------------------
# ProviderGrant
# ---------------------------------------------------------------------------

class ProviderGrantTests(EgressFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.make_scope()
        self.provider = self.make_provider("Local P", "ollama", LOCALITY.LOCAL)
        self.collection = self.make_collection(self.scope)

    def test_no_grant_denies(self):
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_CORPUS
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.NO_PROVIDER_GRANT)

    def test_a_grant_with_embeddings_disabled_denies(self):
        self.grant(self.scope, self.provider, allow=False)
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_CORPUS
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(
            decision.reason_code, ReasonCode.GRANT_DOES_NOT_ALLOW_EMBEDDINGS
        )

    def test_a_grant_with_embeddings_enabled_is_eligible(self):
        self.grant(self.scope, self.provider, allow=True)
        self.assertTrue(
            self.decide(self.scope, self.provider, self.collection, PAYLOAD_CORPUS).allowed
        )

    def test_allow_embeddings_defaults_to_false(self):
        grant = ProviderGrant.objects.create(
            application_scope=self.scope, provider=self.provider
        )
        self.assertFalse(grant.allow_embeddings)

    def test_the_scope_provider_pair_is_unique(self):
        self.grant(self.scope, self.provider)
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                self.grant(self.scope, self.provider)

    def test_a_grant_for_another_scope_authorizes_nothing_here(self):
        other = self.make_scope("App B", "app-b")
        self.grant(other, self.provider, allow=True)
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_CORPUS
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.NO_PROVIDER_GRANT)

    def test_deleting_a_scope_cascades_its_grants(self):
        other = self.make_scope("App C", "app-c")
        self.grant(other, self.provider)
        other.delete()
        self.assertFalse(
            ProviderGrant.objects.filter(application_scope_id=other.pk).exists()
        )

    def test_deleting_a_granted_provider_is_protected(self):
        """Authorization records must not vanish under a provider deletion."""
        from django.db.models import ProtectedError

        self.grant(self.scope, self.provider)
        with self.assertRaises(ProtectedError):
            with transaction.atomic():
                self.provider.delete()

    def test_str_states_whether_it_grants_anything(self):
        allowed = self.grant(self.scope, self.provider, allow=True)
        self.assertIn("embeddings allowed", str(allowed))
        other = self.make_scope("App D", "app-d")
        denied = self.grant(other, self.provider, allow=False)
        self.assertIn("no embedding use", str(denied))


# ---------------------------------------------------------------------------
# LOCAL providers
# ---------------------------------------------------------------------------

class LocalProviderTests(EgressFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.make_scope()          # both egress flags False
        self.provider = self.make_provider("Local", "openai", LOCALITY.LOCAL)
        self.collection = self.make_collection(self.scope)
        self.grant(self.scope, self.provider)

    def test_local_corpus_is_allowed_without_external_flags(self):
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_CORPUS
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.ALLOWED_LOCAL)
        self.assertFalse(self.scope.allow_external_embedding_corpus_egress)

    def test_local_query_is_allowed_without_external_flags(self):
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_QUERY
        )
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_external_egress)

    def test_local_still_requires_the_collection_to_belong_to_the_scope(self):
        other = self.make_scope("App B", "app-b")
        foreign = self.make_collection(other, "B Knowledge")
        decision = self.decide(self.scope, self.provider, foreign, PAYLOAD_CORPUS)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.COLLECTION_FOREIGN_SCOPE)

    def test_local_still_requires_an_active_collection(self):
        self.collection.is_active = False
        self.collection.save(update_fields=["is_active"])
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_CORPUS
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.COLLECTION_INACTIVE)

    def test_a_collection_deny_does_not_block_a_local_provider(self):
        """`deny` is about EXTERNAL egress; nothing external is happening."""
        self.collection.external_embedding_egress_policy = POLICY.DENY
        self.collection.save(update_fields=["external_embedding_egress_policy"])
        self.assertTrue(
            self.decide(self.scope, self.provider, self.collection, PAYLOAD_CORPUS).allowed
        )


# ---------------------------------------------------------------------------
# EXTERNAL providers
# ---------------------------------------------------------------------------

class ExternalCorpusTests(EgressFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.make_scope()
        self.provider = self.make_provider("External", "openai", LOCALITY.EXTERNAL)
        self.collection = self.make_collection(self.scope)
        self.grant(self.scope, self.provider)

    def _allow_corpus(self):
        self.scope.allow_external_embedding_corpus_egress = True
        self.scope.save(update_fields=["allow_external_embedding_corpus_egress"])

    def test_scope_corpus_false_denies(self):
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_CORPUS
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.SCOPE_DENIES_EXTERNAL_CORPUS)

    def test_scope_corpus_true_with_inherit_allows(self):
        self._allow_corpus()
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_CORPUS
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.ALLOWED_EXTERNAL)
        self.assertTrue(decision.requires_external_egress)

    def test_collection_deny_defeats_a_scope_level_allow(self):
        self._allow_corpus()
        self.collection.external_embedding_egress_policy = POLICY.DENY
        self.collection.save(update_fields=["external_embedding_egress_policy"])
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_CORPUS
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.COLLECTION_DENIES_EXTERNAL)

    def test_a_missing_grant_denies_even_with_egress_enabled(self):
        self._allow_corpus()
        ProviderGrant.objects.all().delete()
        self.assertEqual(
            self.decide(self.scope, self.provider, self.collection, PAYLOAD_CORPUS)
            .reason_code,
            ReasonCode.NO_PROVIDER_GRANT,
        )

    def test_an_inactive_scope_denies(self):
        self._allow_corpus()
        self.scope.is_active = False
        self.scope.save(update_fields=["is_active"])
        self.assertEqual(
            self.decide(self.scope, self.provider, self.collection, PAYLOAD_CORPUS)
            .reason_code,
            ReasonCode.SCOPE_INACTIVE,
        )

    def test_an_inactive_provider_denies(self):
        self._allow_corpus()
        self.provider.is_active = False
        self.provider.save(update_fields=["is_active"])
        self.assertEqual(
            self.decide(self.scope, self.provider, self.collection, PAYLOAD_CORPUS)
            .reason_code,
            ReasonCode.PROVIDER_INACTIVE,
        )

    def test_an_inactive_collection_denies(self):
        self._allow_corpus()
        self.collection.is_active = False
        self.collection.save(update_fields=["is_active"])
        self.assertEqual(
            self.decide(self.scope, self.provider, self.collection, PAYLOAD_CORPUS)
            .reason_code,
            ReasonCode.COLLECTION_INACTIVE,
        )

    def test_a_foreign_collection_denies(self):
        self._allow_corpus()
        other = self.make_scope("App B", "app-b")
        foreign = self.make_collection(other, "B Knowledge")
        self.assertEqual(
            self.decide(self.scope, self.provider, foreign, PAYLOAD_CORPUS).reason_code,
            ReasonCode.COLLECTION_FOREIGN_SCOPE,
        )


class ExternalQueryTests(EgressFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.make_scope(
            allow_external_embedding_corpus_egress=True
        )
        self.provider = self.make_provider("External", "openai", LOCALITY.EXTERNAL)
        self.collection = self.make_collection(self.scope)
        self.grant(self.scope, self.provider)

    def test_corpus_true_query_false_denies_the_query(self):
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_QUERY
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.SCOPE_DENIES_EXTERNAL_QUERY)

    def test_corpus_true_query_false_still_allows_the_corpus(self):
        self.assertTrue(
            self.decide(self.scope, self.provider, self.collection, PAYLOAD_CORPUS).allowed
        )

    def test_corpus_true_query_true_allows_the_query(self):
        self.scope.allow_external_embedding_query_egress = True
        self.scope.save(update_fields=["allow_external_embedding_query_egress"])
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_QUERY
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.ALLOWED_EXTERNAL)

    def test_collection_deny_defeats_query_egress_too(self):
        self.scope.allow_external_embedding_query_egress = True
        self.scope.save(update_fields=["allow_external_embedding_query_egress"])
        self.collection.external_embedding_egress_policy = POLICY.DENY
        self.collection.save(update_fields=["external_embedding_egress_policy"])
        self.assertEqual(
            self.decide(self.scope, self.provider, self.collection, PAYLOAD_QUERY)
            .reason_code,
            ReasonCode.COLLECTION_DENIES_EXTERNAL,
        )

    def test_the_per_collection_query_subset_invariant_holds(self):
        """external_query_allowed ⊆ external_corpus_allowed, per collection."""
        self.scope.allow_external_embedding_query_egress = True
        self.scope.save(update_fields=["allow_external_embedding_query_egress"])
        for policy in (POLICY.INHERIT, POLICY.DENY):
            with self.subTest(policy=policy):
                self.collection.external_embedding_egress_policy = policy
                self.collection.save(
                    update_fields=["external_embedding_egress_policy"]
                )
                corpus = self.decide(
                    self.scope, self.provider, self.collection, PAYLOAD_CORPUS
                ).allowed
                query = self.decide(
                    self.scope, self.provider, self.collection, PAYLOAD_QUERY
                ).allowed
                self.assertTrue(corpus or not query, "query allowed where corpus is not")


# ---------------------------------------------------------------------------
# UNKNOWN locality beats every permission
# ---------------------------------------------------------------------------

class UnknownLocalityTests(EgressFixtureMixin, TestCase):
    def test_unknown_denies_even_when_everything_else_permits(self):
        scope = self.make_scope(
            allow_external_embedding_corpus_egress=True,
            allow_external_embedding_query_egress=True,
        )
        provider = self.make_provider("Unclassified", "openai", LOCALITY.UNKNOWN)
        collection = self.make_collection(scope)   # inherit
        self.grant(scope, provider, allow=True)

        for kind in (PAYLOAD_CORPUS, PAYLOAD_QUERY):
            with self.subTest(payload_kind=kind):
                decision = self.decide(scope, provider, collection, kind)
                self.assertFalse(decision.allowed)
                self.assertEqual(
                    decision.reason_code, ReasonCode.PROVIDER_LOCALITY_UNDECLARED
                )

    def test_unknown_is_never_auto_upgraded_by_provider_type(self):
        scope = self.make_scope(allow_external_embedding_corpus_egress=True)
        collection = self.make_collection(scope)
        for provider_type in ("ollama", "openai", "anthropic", "training", "other"):
            with self.subTest(provider_type=provider_type):
                provider = self.make_provider(
                    f"P {provider_type}", provider_type, LOCALITY.UNKNOWN
                )
                self.grant(scope, provider, allow=True)
                self.assertFalse(
                    self.decide(scope, provider, collection, PAYLOAD_CORPUS).allowed
                )


# ---------------------------------------------------------------------------
# The decision object and the content boundary
# ---------------------------------------------------------------------------

class DecisionContractTests(EgressFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.make_scope()
        self.provider = self.make_provider("Local", "ollama", LOCALITY.LOCAL)
        self.collection = self.make_collection(self.scope)
        self.grant(self.scope, self.provider)

    def test_the_decision_is_immutable_and_not_a_model(self):
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_CORPUS
        )
        with self.assertRaises(Exception):
            decision.allowed = False
        self.assertFalse(hasattr(decision, "save"))
        self.assertFalse(hasattr(decision, "_meta"))

    def test_the_resolver_accepts_no_content_parameters(self):
        """Policy is decided BEFORE content reaches the boundary."""
        parameters = set(inspect.signature(resolve_embedding_access).parameters)
        self.assertEqual(
            parameters,
            {"application_scope", "provider", "collection", "payload_kind"},
        )
        for forbidden in ("text", "content", "query", "prompt", "input", "chunk"):
            self.assertNotIn(forbidden, parameters)

    def test_the_decision_carries_no_content_fields(self):
        decision = self.decide(
            self.scope, self.provider, self.collection, PAYLOAD_CORPUS
        )
        for field in decision.__dataclass_fields__:
            for forbidden in ("text", "content", "prompt", "body", "credential", "key"):
                self.assertNotIn(forbidden, field)

    def test_only_two_payload_kinds_are_accepted(self):
        for kind in ("", None, "prompt", "arbitrary", "CORPUS", "embedding"):
            with self.subTest(payload_kind=kind):
                decision = self.decide(
                    self.scope, self.provider, self.collection, kind
                )
                self.assertFalse(decision.allowed)
                self.assertEqual(
                    decision.reason_code, ReasonCode.UNKNOWN_PAYLOAD_KIND
                )

    def test_resolving_makes_no_provider_or_network_call(self):
        with mock.patch(
            "ai_hub.services.provider_registry.resolve_model_config"
        ) as resolve_model, mock.patch(
            "ai_hub.services.litellm_client.completion_call"
        ) as completion:
            self.decide(self.scope, self.provider, self.collection, PAYLOAD_CORPUS)
            self.decide(self.scope, self.provider, self.collection, PAYLOAD_QUERY)
        resolve_model.assert_not_called()
        completion.assert_not_called()

    def test_the_service_imports_no_provider_client(self):
        source = inspect.getsource(embedding_egress)
        for forbidden in ("litellm", "completion_call", "provider_registry", "openai"):
            self.assertNotIn(forbidden, source)


# ---------------------------------------------------------------------------
# Separation from Knowledge authorization
# ---------------------------------------------------------------------------

class AuthorizationSeparationTests(EgressFixtureMixin, TestCase):
    """Provider authorization never grants Knowledge authorization, or vice versa."""

    def setUp(self):
        self.scope = self.make_scope()
        provider_config = ProviderConfig.objects.create(
            name="Model Host", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        self.model_config = ModelConfig.objects.create(
            provider=provider_config, model_name="training",
            temperature_default=Decimal("0.10"),
        )
        self.embedding_provider = self.make_provider(
            "Embed Local", "ollama", LOCALITY.LOCAL
        )
        self.collection = self.make_collection(self.scope)
        self.agent = AgentProfile.objects.create(
            name="A Agent", role="r", model_config=self.model_config,
            application_scope=self.scope,
        )

    def test_a_provider_grant_does_not_grant_knowledge_access(self):
        from ai_hub.services.knowledge_authorization import (
            resolve_effective_knowledge_scope,
        )

        self.grant(self.scope, self.embedding_provider, allow=True)
        # The agent was never assigned the collection.
        knowledge = resolve_effective_knowledge_scope(self.agent)
        self.assertNotIn(self.collection.pk, knowledge.collection_ids)
        # ...while the egress policy independently says yes.
        self.assertTrue(
            self.decide(
                self.scope, self.embedding_provider, self.collection, PAYLOAD_CORPUS
            ).allowed
        )

    def test_knowledge_access_does_not_grant_provider_egress(self):
        from ai_hub.services.knowledge_authorization import (
            resolve_effective_knowledge_scope,
        )

        self.agent.knowledge_collections.add(self.collection)
        knowledge = resolve_effective_knowledge_scope(self.agent)
        self.assertIn(self.collection.pk, knowledge.collection_ids)
        # ...while no grant exists, so egress is refused.
        decision = self.decide(
            self.scope, self.embedding_provider, self.collection, PAYLOAD_CORPUS
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.NO_PROVIDER_GRANT)

    def test_the_two_results_are_different_types(self):
        from ai_hub.services.knowledge_authorization import EffectiveKnowledgeScope

        self.assertIsNot(EffectiveKnowledgeScope, EmbeddingAccessDecision)
        self.assertNotEqual(
            set(EffectiveKnowledgeScope.__dataclass_fields__),
            set(EmbeddingAccessDecision.__dataclass_fields__),
        )


# ---------------------------------------------------------------------------
# Existing completion runtime must be unaffected
# ---------------------------------------------------------------------------

class CompletionRuntimeCompatibilityTests(EgressFixtureMixin, TestCase):
    """An unclassified provider must keep working for chat exactly as before."""

    def test_resolve_model_config_ignores_declared_locality(self):
        from ai_hub.services.provider_registry import resolve_model_config

        provider = ProviderConfig.objects.create(
            name="Training", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        self.assertEqual(provider.declared_locality, LOCALITY.UNKNOWN)
        model_config = ModelConfig.objects.create(
            provider=provider, model_name="training"
        )
        resolved = resolve_model_config(model_config)
        self.assertEqual(resolved["model"], "training")
        self.assertNotIn("declared_locality", resolved)
        self.assertNotIn("egress", " ".join(resolved))

    def test_no_completion_module_consults_the_egress_policy(self):
        import ai_hub.services.agent_runtime as agent_runtime
        import ai_hub.services.execution_runner as execution_runner
        import ai_hub.services.provider_registry as provider_registry

        for module in (provider_registry, agent_runtime, execution_runner):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("embedding_egress", source)
                self.assertNotIn("resolve_embedding_access", source)
                self.assertNotIn("ProviderGrant", source)

    def test_an_unknown_locality_provider_still_runs_an_agent(self):
        from ai_hub.services.agent_runtime import prepare_agent_payload

        provider = ProviderConfig.objects.create(
            name="Chat", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        model_config = ModelConfig.objects.create(
            provider=provider, model_name="training"
        )
        scope = self.make_scope()
        agent = AgentProfile.objects.create(
            name="Chatty", role="r", model_config=model_config,
            application_scope=scope,
        )
        payload = prepare_agent_payload(agent, {}, {})
        self.assertIn("knowledge_context", payload)


# ---------------------------------------------------------------------------
# Admin reachability
# ---------------------------------------------------------------------------

class EgressAdminTests(TestCase):
    def test_provider_grant_is_registered(self):
        from django.contrib import admin as django_admin

        self.assertIn(ProviderGrant, django_admin.site._registry)

    def test_the_provider_grant_admin_urls_reverse(self):
        from django.urls import reverse

        self.assertTrue(reverse("admin:ai_hub_providergrant_add"))
        self.assertTrue(reverse("admin:ai_hub_providergrant_changelist"))

    def test_declared_locality_is_operator_editable(self):
        from django.contrib import admin as django_admin

        model_admin = django_admin.site._registry[ProviderConfig]
        self.assertNotIn(
            "declared_locality", model_admin.get_readonly_fields(None)
        )
        self.assertIn("declared_locality", model_admin.list_display)

    def test_scope_egress_flags_are_reachable(self):
        from django.contrib import admin as django_admin

        model_admin = django_admin.site._registry[ApplicationScope]
        for field in (
            "allow_external_embedding_corpus_egress",
            "allow_external_embedding_query_egress",
        ):
            with self.subTest(field=field):
                self.assertIn(field, model_admin.list_display)
                self.assertNotIn(field, model_admin.get_readonly_fields(None))

    def test_collection_narrowing_policy_is_reachable(self):
        from django.contrib import admin as django_admin

        model_admin = django_admin.site._registry[KnowledgeCollection]
        self.assertIn("external_embedding_egress_policy", model_admin.list_display)

    def test_the_grant_admin_exposes_no_credentials(self):
        from django.contrib import admin as django_admin

        model_admin = django_admin.site._registry[ProviderGrant]
        rendered = " ".join(model_admin.fields) + " ".join(model_admin.list_display)
        for forbidden in ("api_key", "credential", "secret", "token"):
            self.assertNotIn(forbidden, rendered)
