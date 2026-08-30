"""S-18 — the embedding capability contract and the `e1` fingerprint.

`e1` answers one question: do two vectors belong to the same compatible
retrieval space? It is not permission, not routing and not an operational
setting, and the matrices below exist to keep it that way.

Nothing here generates an embedding, stores a vector, or calls a provider.
"""

import ast
import inspect
import re
import unicodedata
from decimal import Decimal
from unittest import mock

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import TestCase

from ai_hub.models import (
    ApplicationScope,
    EmbeddingModelConfig,
    KnowledgeCollection,
    ModelConfig,
    ProviderConfig,
    ProviderGrant,
)
from ai_hub.services import embedding_contract
from ai_hub.services.embedding_contract import (
    E1_PREFIX,
    EmbeddingContractError,
    ResolvedEmbeddingContract,
    embedding_contract_fingerprint,
    embedding_contract_payload,
    resolve_embedding_contract,
)


def _tree(module):
    return ast.parse(inspect.getsource(module))


def imported_names(module) -> set:
    """Every module and symbol this module actually imports."""
    names = set()
    for node in ast.walk(_tree(module)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.update(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.update(node.module.split("."))
            for alias in node.names:
                names.add(alias.name)
    return names


def referenced_names(module) -> set:
    """Every identifier and attribute the CODE touches - never prose.

    Docstrings and comments are absent by construction: only `ast.Name` and
    `ast.Attribute` nodes are collected, so a module that DESCRIBES a forbidden
    call in its docstring does not trip an assertion about making one.
    """
    names = set()
    for node in ast.walk(_tree(module)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def code_strings(module) -> set:
    """String literals used as VALUES, excluding docstrings."""
    tree = _tree(module)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }


METRIC = EmbeddingModelConfig.DistanceMetric
NORMALIZATION = EmbeddingModelConfig.Normalization
LOCALITY = ProviderConfig.DeclaredLocality

E1_PATTERN = re.compile(r"^e1:sha256:[0-9a-f]{64}$")

#: One fixed canonical configuration, pinned end to end.
GOLDEN_FACTS = {
    "provider_type": "openai",
    "model_name": "text-embedding-3-small",
    "model_revision": "v1",
    "vector_dimension": 1536,
    "distance_metric": "cosine",
    "normalization": "l2",
}

#: The expected fingerprint for GOLDEN_FACTS, as a LITERAL.
#:
#: Deliberately hard-coded rather than computed. An expected value derived from
#: the implementation under test is not a pin: it moves whenever the
#: implementation moves, so a change to the canonical field set, key naming,
#: Unicode normalization, JSON serialization, separators, encoding or hash
#: format would silently take the expectation with it and the test would stay
#: green.
#:
#: Once vectors exist, `e1` is persisted compatibility evidence. Any change that
#: alters this string invalidates every vector ever produced under the old
#: contract, so it must fail here first and force a conscious decision about
#: whether the `e1` contract is being versioned - never be absorbed silently.
#:
#: Regenerating this literal is therefore an explicit act. It is not derived at
#: import time, not built in `setUp`, and uses no helper from the production
#: fingerprint module.
GOLDEN_E1 = "e1:sha256:5f0adca9eb56c78eb98ec424562227a95aa413246df9f8d45bd59474609c134e"


class ContractFixtureMixin:
    def make_provider(self, name="P", provider_type="openai", **kwargs):
        return ProviderConfig.objects.create(
            name=name, provider_type=provider_type, **kwargs
        )

    def make_config(self, provider=None, *, name="embed-1", **kwargs):
        fields = {
            "model_name": "text-embedding-3-small",
            "model_revision": "v1",
            "vector_dimension": 1536,
            "distance_metric": METRIC.COSINE,
            "normalization": NORMALIZATION.L2,
            "max_input_chars": 8000,
            "request_timeout_seconds": 60,
        }
        fields.update(kwargs)
        return EmbeddingModelConfig.objects.create(
            name=name, provider=provider or self.make_provider(), **fields
        )


# ---------------------------------------------------------------------------
# Model contract and validation
# ---------------------------------------------------------------------------

class EmbeddingModelConfigTests(ContractFixtureMixin, TestCase):
    def test_a_valid_active_config_on_an_active_provider_validates(self):
        config = self.make_config()
        config.full_clean()
        self.assertTrue(config.is_active)

    def test_it_is_a_separate_model_from_ModelConfig(self):
        embedding_fields = {f.name for f in EmbeddingModelConfig._meta.get_fields()}
        completion_fields = {f.name for f in ModelConfig._meta.get_fields()}
        # Completion-only concepts must not appear on the embedding contract...
        for field in ("temperature_default", "max_tokens_default", "supports_tools"):
            self.assertNotIn(field, embedding_fields)
        # ...and embedding-only concepts must not have been bolted onto ModelConfig.
        for field in (
            "vector_dimension", "distance_metric", "normalization",
            "model_revision", "max_input_chars",
        ):
            self.assertNotIn(field, completion_fields)

    def test_it_carries_no_application_ownership(self):
        """Global definition. Permission lives in ProviderGrant + egress policy."""
        field_names = {f.name for f in EmbeddingModelConfig._meta.get_fields()}
        for forbidden in (
            "application_scope", "knowledge_collection", "agent", "workspace",
        ):
            self.assertNotIn(forbidden, field_names)

    def test_locality_is_not_duplicated_here(self):
        field_names = {f.name for f in EmbeddingModelConfig._meta.get_fields()}
        for forbidden in ("declared_locality", "locality", "is_local", "is_external"):
            self.assertNotIn(forbidden, field_names)

    def test_the_provider_relationship_is_protected(self):
        from django.db.models import ProtectedError

        provider = self.make_provider("Protected")
        self.make_config(provider, name="protected-embed")
        with self.assertRaises(ProtectedError):
            with transaction.atomic():
                provider.delete()

    def test_blank_required_text_is_rejected(self):
        for field in ("name", "model_name", "model_revision"):
            with self.subTest(field=field):
                config = EmbeddingModelConfig(
                    name="x", provider=self.make_provider(f"P {field}"),
                    model_name="m", model_revision="v1", vector_dimension=8,
                    distance_metric=METRIC.COSINE, normalization=NORMALIZATION.NONE,
                )
                setattr(config, field, "   ")
                with self.assertRaises(ValidationError) as raised:
                    config.full_clean()
                self.assertIn(field, raised.exception.error_dict)

    def test_non_positive_integers_are_rejected(self):
        for field in (
            "vector_dimension", "max_input_chars", "request_timeout_seconds",
        ):
            with self.subTest(field=field):
                config = EmbeddingModelConfig(
                    name=f"x-{field}", provider=self.make_provider(f"Pi {field}"),
                    model_name="m", model_revision="v1", vector_dimension=8,
                    distance_metric=METRIC.COSINE, normalization=NORMALIZATION.NONE,
                )
                setattr(config, field, 0)
                with self.assertRaises(ValidationError) as raised:
                    config.full_clean()
                self.assertIn(field, raised.exception.error_dict)

    def test_unknown_distance_metric_is_rejected(self):
        config = self.make_config(name="bad-metric")
        config.distance_metric = "pgvector_cosine"
        with self.assertRaises(ValidationError) as raised:
            config.full_clean()
        self.assertIn("distance_metric", raised.exception.error_dict)

    def test_unknown_normalization_is_rejected(self):
        config = self.make_config(name="bad-norm")
        config.normalization = "unit"
        with self.assertRaises(ValidationError) as raised:
            config.full_clean()
        self.assertIn("normalization", raised.exception.error_dict)

    def test_activating_against_an_inactive_provider_is_rejected(self):
        provider = self.make_provider("Idle", is_active=False)
        config = EmbeddingModelConfig(
            name="idle-embed", provider=provider, model_name="m",
            model_revision="v1", vector_dimension=8,
            distance_metric=METRIC.COSINE, normalization=NORMALIZATION.NONE,
        )
        with self.assertRaises(ValidationError) as raised:
            config.full_clean()
        self.assertIn("is_active", raised.exception.error_dict)

    def test_the_name_is_globally_unique(self):
        self.make_config(name="shared")
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                self.make_config(self.make_provider("P2"), name="shared")

    def test_no_backend_specific_metric_members_exist(self):
        for forbidden in ("pgvector_cosine", "l2_ops", "vector_ip_ops"):
            self.assertNotIn(forbidden, METRIC.values)

    def test_the_metric_and_normalization_choices_are_exactly_as_contracted(self):
        self.assertEqual(set(METRIC.values), {"cosine", "dot_product", "euclidean"})
        self.assertEqual(set(NORMALIZATION.values), {"none", "l2"})


# ---------------------------------------------------------------------------
# e1 format, determinism, and the golden pin
# ---------------------------------------------------------------------------

class E1FormatTests(TestCase):
    def test_the_format_is_self_describing(self):
        fingerprint = embedding_contract_fingerprint(**GOLDEN_FACTS)
        self.assertTrue(E1_PATTERN.match(fingerprint), fingerprint)
        self.assertTrue(fingerprint.startswith(E1_PREFIX))

    def test_it_is_deterministic_across_repeated_calls(self):
        results = {embedding_contract_fingerprint(**GOLDEN_FACTS) for _ in range(25)}
        self.assertEqual(len(results), 1)

    def test_the_golden_fingerprint_is_pinned(self):
        """If canonical serialization changes, this test MUST fail.

        `GOLDEN_E1` is a literal in this module's source. Nothing in the
        production fingerprint implementation contributes to it, so the two
        sides of this assertion cannot drift together.
        """
        self.assertEqual(
            embedding_contract_fingerprint(**GOLDEN_FACTS),
            GOLDEN_E1,
        )

    def test_the_golden_literal_is_well_formed(self):
        """The pin itself must be a valid e1, independent of any computation."""
        self.assertTrue(E1_PATTERN.match(GOLDEN_E1), GOLDEN_E1)
        self.assertEqual(len(GOLDEN_E1), len(E1_PREFIX) + 64)
        self.assertEqual(GOLDEN_E1, GOLDEN_E1.lower())

    def test_the_golden_literal_is_a_constant_not_a_computation(self):
        """Guards the pin against being quietly made self-referential again.

        Reads THIS module's own source and asserts `GOLDEN_E1` is assigned a
        plain string constant - not a call, not a name, not an f-string.
        """
        module_tree = ast.parse(inspect.getsource(__import__(__name__, fromlist=["x"])))
        assignments = [
            node
            for node in ast.walk(module_tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "GOLDEN_E1"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1, "GOLDEN_E1 must be assigned exactly once")
        value = assignments[0].value
        self.assertIsInstance(
            value, ast.Constant,
            "GOLDEN_E1 must be a literal, never derived from the code under test",
        )
        self.assertIsInstance(value.value, str)
        self.assertEqual(value.value, GOLDEN_E1)

    def test_the_canonical_payload_is_exactly_the_contracted_shape(self):
        payload = embedding_contract_payload(**GOLDEN_FACTS)
        self.assertEqual(
            set(payload),
            {
                "contract", "provider_type", "model_name", "model_revision",
                "vector_dimension", "distance_metric", "normalization",
            },
        )
        self.assertEqual(payload["contract"], "e1")
        self.assertIsInstance(payload["vector_dimension"], int)

    def test_key_order_does_not_affect_the_fingerprint(self):
        """Insertion order must not be the contract."""
        reversed_facts = dict(reversed(list(GOLDEN_FACTS.items())))
        self.assertEqual(
            embedding_contract_fingerprint(**reversed_facts),
            embedding_contract_fingerprint(**GOLDEN_FACTS),
        )

    def test_the_payload_carries_no_secret_or_routing_material(self):
        payload = embedding_contract_payload(**GOLDEN_FACTS)
        rendered = " ".join(f"{k}{v}" for k, v in payload.items()).lower()
        for forbidden in (
            "api_key", "credential", "token", "secret", "base_url",
            "locality", "grant", "timeout",
        ):
            self.assertNotIn(forbidden, rendered)
            self.assertNotIn(forbidden, set(payload))


# ---------------------------------------------------------------------------
# Included fields: each one MUST invalidate
# ---------------------------------------------------------------------------

class E1IncludedFieldTests(TestCase):
    def test_each_included_field_changes_the_fingerprint(self):
        baseline = embedding_contract_fingerprint(**GOLDEN_FACTS)
        variations = {
            "provider_type": "ollama",
            "model_name": "text-embedding-3-large",
            "model_revision": "v2",
            "vector_dimension": 3072,
            "distance_metric": "dot_product",
            "normalization": "none",
        }
        for field, value in variations.items():
            with self.subTest(field=field):
                facts = dict(GOLDEN_FACTS, **{field: value})
                self.assertNotEqual(
                    embedding_contract_fingerprint(**facts), baseline,
                    f"changing {field} must change e1",
                )

    def test_all_six_variations_are_mutually_distinct(self):
        """No two different contracts may collide onto one fingerprint."""
        variations = [
            dict(GOLDEN_FACTS),
            dict(GOLDEN_FACTS, provider_type="ollama"),
            dict(GOLDEN_FACTS, model_name="other"),
            dict(GOLDEN_FACTS, model_revision="v2"),
            dict(GOLDEN_FACTS, vector_dimension=768),
            dict(GOLDEN_FACTS, distance_metric="euclidean"),
            dict(GOLDEN_FACTS, normalization="none"),
        ]
        fingerprints = {embedding_contract_fingerprint(**facts) for facts in variations}
        self.assertEqual(len(fingerprints), len(variations))

    def test_provider_type_namespaces_the_contract(self):
        """Same model string on two families is NOT assumed interchangeable."""
        openai = embedding_contract_fingerprint(**dict(GOLDEN_FACTS, provider_type="openai"))
        ollama = embedding_contract_fingerprint(**dict(GOLDEN_FACTS, provider_type="ollama"))
        self.assertNotEqual(openai, ollama)

    def test_model_identifiers_are_not_case_folded(self):
        """Vendor strings are opaque; casing may be meaningful."""
        self.assertNotEqual(
            embedding_contract_fingerprint(**dict(GOLDEN_FACTS, model_name="TEXT-EMBEDDING-3-SMALL")),
            embedding_contract_fingerprint(**GOLDEN_FACTS),
        )


# ---------------------------------------------------------------------------
# Excluded fields: none may invalidate
# ---------------------------------------------------------------------------

class E1ExcludedFieldTests(ContractFixtureMixin, TestCase):
    """Operational, routing and authorization changes must not move `e1`."""

    def setUp(self):
        self.provider = self.make_provider(
            "Primary", "openai", base_url="https://a.example.com",
            api_key_env_var="A_KEY", declared_locality=LOCALITY.LOCAL,
        )
        self.config = self.make_config(self.provider, name="stable")
        self.baseline = resolve_embedding_contract(self.config).e1

    def _e1(self):
        self.config.refresh_from_db()
        return resolve_embedding_contract(self.config).e1

    def test_config_operational_changes_do_not_move_e1(self):
        cases = {
            "name": "renamed-config",
            "max_input_chars": 32000,
            "request_timeout_seconds": 5,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                setattr(self.config, field, value)
                self.config.save(update_fields=[field])
                self.assertEqual(self._e1(), self.baseline)

    def test_provider_routing_and_credential_changes_do_not_move_e1(self):
        cases = {
            "name": "Renamed Provider",
            "base_url": "https://b.example.com",
            "api_key_env_var": "B_KEY",
            "default_timeout": 900,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                setattr(self.provider, field, value)
                self.provider.save(update_fields=[field])
                self.config.refresh_from_db()
                self.assertEqual(self._e1(), self.baseline)

    def test_declared_locality_does_not_move_e1(self):
        """Permission changes; vector identity does not."""
        for locality in (LOCALITY.EXTERNAL, LOCALITY.UNKNOWN, LOCALITY.LOCAL):
            with self.subTest(locality=locality):
                self.provider.declared_locality = locality
                self.provider.save(update_fields=["declared_locality"])
                self.config.refresh_from_db()
                self.assertEqual(self._e1(), self.baseline)

    def test_authorization_changes_do_not_move_e1(self):
        scope = ApplicationScope.objects.create(name="App", slug="app")
        collection = KnowledgeCollection.objects.create(
            name="C", application_scope=scope
        )
        grant = ProviderGrant.objects.create(
            application_scope=scope, provider=self.provider, allow_embeddings=False
        )
        self.assertEqual(self._e1(), self.baseline)

        grant.allow_embeddings = True
        grant.save(update_fields=["allow_embeddings"])
        self.assertEqual(self._e1(), self.baseline)

        scope.allow_external_embedding_corpus_egress = True
        scope.save(update_fields=["allow_external_embedding_corpus_egress"])
        scope.allow_external_embedding_query_egress = True
        scope.save(update_fields=["allow_external_embedding_query_egress"])
        self.assertEqual(self._e1(), self.baseline)

        collection.external_embedding_egress_policy = (
            KnowledgeCollection.ExternalEmbeddingEgressPolicy.DENY
        )
        collection.save(update_fields=["external_embedding_egress_policy"])
        self.assertEqual(self._e1(), self.baseline)

    def test_active_state_does_not_move_e1(self):
        """Tested against the pure function - the resolver fails closed instead."""
        facts = dict(
            provider_type=self.provider.provider_type,
            model_name=self.config.model_name,
            model_revision=self.config.model_revision,
            vector_dimension=self.config.vector_dimension,
            distance_metric=self.config.distance_metric,
            normalization=self.config.normalization,
        )
        before = embedding_contract_fingerprint(**facts)
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(is_active=False)
        ProviderConfig.objects.filter(pk=self.provider.pk).update(is_active=False)
        self.assertEqual(embedding_contract_fingerprint(**facts), before)
        self.assertEqual(before, self.baseline)

        # ...and the runtime resolver still refuses, rather than being weakened.
        self.config.refresh_from_db()
        with self.assertRaises(EmbeddingContractError):
            resolve_embedding_contract(self.config)

    def test_two_configs_sharing_semantics_share_e1(self):
        """Operationally different, semantically identical - correct to match."""
        twin = self.make_config(
            self.provider, name="long-timeout",
            request_timeout_seconds=600, max_input_chars=64000,
        )
        self.assertEqual(
            resolve_embedding_contract(twin).e1,
            resolve_embedding_contract(self.config).e1,
        )
        self.assertNotEqual(twin.pk, self.config.pk)

    def test_no_uniqueness_constraint_assumes_one_row_per_contract(self):
        constrained = {
            frozenset(getattr(c, "fields", ()))
            for c in EmbeddingModelConfig._meta.constraints
        }
        semantic = frozenset({
            "provider", "model_name", "model_revision", "vector_dimension",
            "distance_metric", "normalization",
        })
        self.assertNotIn(semantic, constrained)


# ---------------------------------------------------------------------------
# Unicode canonicalization
# ---------------------------------------------------------------------------

class E1UnicodeTests(TestCase):
    def test_nfc_and_nfd_forms_produce_the_same_fingerprint(self):
        for field in ("model_name", "model_revision"):
            with self.subTest(field=field):
                composed = unicodedata.normalize("NFC", "café-embed")
                decomposed = unicodedata.normalize("NFD", "café-embed")
                self.assertNotEqual(composed, decomposed)   # different bytes
                self.assertEqual(
                    embedding_contract_fingerprint(**dict(GOLDEN_FACTS, **{field: composed})),
                    embedding_contract_fingerprint(**dict(GOLDEN_FACTS, **{field: decomposed})),
                )

    def test_normalization_does_not_strip_or_fold(self):
        """NFC is a representation fix, not a semantic rewrite."""
        baseline = embedding_contract_fingerprint(**GOLDEN_FACTS)
        for variant in (" text-embedding-3-small", "text-embedding-3-small "):
            with self.subTest(variant=repr(variant)):
                self.assertNotEqual(
                    embedding_contract_fingerprint(
                        **dict(GOLDEN_FACTS, model_name=variant)
                    ),
                    baseline,
                )


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------

class ResolveEmbeddingContractTests(ContractFixtureMixin, TestCase):
    def setUp(self):
        self.provider = self.make_provider("Res", "openai")
        self.config = self.make_config(self.provider, name="res-embed")

    def test_a_valid_config_resolves_completely(self):
        resolved = resolve_embedding_contract(self.config)
        self.assertIsInstance(resolved, ResolvedEmbeddingContract)
        self.assertEqual(resolved.embedding_model_config_id, self.config.pk)
        self.assertEqual(resolved.provider_id, self.provider.pk)
        self.assertEqual(resolved.provider_type, "openai")
        self.assertEqual(resolved.vector_dimension, 1536)
        self.assertEqual(resolved.distance_metric, METRIC.COSINE)
        self.assertEqual(resolved.normalization, NORMALIZATION.L2)
        self.assertTrue(E1_PATTERN.match(resolved.e1))

    def test_the_result_is_immutable_and_not_a_model(self):
        resolved = resolve_embedding_contract(self.config)
        with self.assertRaises(Exception):
            resolved.e1 = "tampered"
        self.assertFalse(hasattr(resolved, "save"))
        self.assertFalse(hasattr(resolved, "_meta"))

    def test_it_reports_but_does_not_own_declared_locality(self):
        self.provider.declared_locality = LOCALITY.EXTERNAL
        self.provider.save(update_fields=["declared_locality"])
        self.config.refresh_from_db()
        self.assertEqual(
            resolve_embedding_contract(self.config).declared_locality,
            LOCALITY.EXTERNAL,
        )

    def test_unknown_locality_still_yields_a_valid_contract(self):
        """The contract exists; S-17 separately refuses its USE."""
        self.provider.declared_locality = LOCALITY.UNKNOWN
        self.provider.save(update_fields=["declared_locality"])
        self.config.refresh_from_db()
        resolved = resolve_embedding_contract(self.config)
        self.assertEqual(resolved.declared_locality, LOCALITY.UNKNOWN)
        self.assertTrue(E1_PATTERN.match(resolved.e1))

    # -- fail closed --------------------------------------------------------

    def test_missing_or_unsaved_config_is_refused(self):
        for candidate in (None, EmbeddingModelConfig()):
            with self.subTest(candidate=type(candidate).__name__):
                with self.assertRaises(EmbeddingContractError):
                    resolve_embedding_contract(candidate)

    def test_an_inactive_config_is_refused(self):
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(is_active=False)
        self.config.refresh_from_db()
        with self.assertRaises(EmbeddingContractError):
            resolve_embedding_contract(self.config)

    def test_an_inactive_provider_is_refused(self):
        ProviderConfig.objects.filter(pk=self.provider.pk).update(is_active=False)
        self.config.refresh_from_db()
        with self.assertRaises(EmbeddingContractError):
            resolve_embedding_contract(self.config)

    def test_raw_orm_malformed_state_is_refused_at_runtime(self):
        """full_clean() is bypassable; this is the runtime boundary."""
        cases = (
            {"model_name": "   "},
            {"model_revision": ""},
            {"vector_dimension": 0},
            {"max_input_chars": 0},
            {"request_timeout_seconds": 0},
            {"distance_metric": "pgvector_cosine"},
            {"normalization": "unit"},
        )
        for update in cases:
            with self.subTest(update=update):
                original = {
                    field: getattr(self.config, field) for field in update
                }
                EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(**update)
                self.config.refresh_from_db()
                with self.assertRaises(EmbeddingContractError):
                    resolve_embedding_contract(self.config)
                EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(**original)
                self.config.refresh_from_db()

    def test_it_never_falls_back_to_another_config_or_provider(self):
        other_provider = self.make_provider("Fallback", "ollama")
        self.make_config(other_provider, name="fallback-embed")
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(is_active=False)
        self.config.refresh_from_db()
        with self.assertRaises(EmbeddingContractError):
            resolve_embedding_contract(self.config)

    def test_there_is_no_default_embedding_model_helper(self):
        """Nothing selects a configuration on the caller's behalf."""
        referenced = referenced_names(embedding_contract)
        for forbidden in (
            "first", "last", "filter", "all", "get_default",
            "require_single_active_scope", "objects",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, referenced)
        for forbidden in ("default_embedding_model", "get_default_embedding"):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(hasattr(embedding_contract, forbidden))


# ---------------------------------------------------------------------------
# No secrets, no provider call, no execution
# ---------------------------------------------------------------------------

class ContractBoundaryTests(ContractFixtureMixin, TestCase):
    def setUp(self):
        self.provider = self.make_provider(
            "Boundary", "openai", api_key_env_var="SECRET_KEY_VAR",
            base_url="https://api.example.com",
        )
        self.config = self.make_config(self.provider, name="boundary-embed")

    def test_the_resolved_contract_carries_no_secret_or_content_field(self):
        """`vector_dimension` is a contract FACT; a stored vector is not."""
        fields = set(ResolvedEmbeddingContract.__dataclass_fields__)
        for forbidden in (
            "api_key", "credential", "token", "secret", "password",
            "text", "content", "prompt", "query",
            "vector_value", "vector_data", "vector_bytes", "embedding_value",
            "embedding", "values",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, fields)
        # The only vector-shaped field is the DIMENSION, an integer.
        self.assertEqual(
            {f for f in fields if "vector" in f}, {"vector_dimension"}
        )

    def test_no_credential_value_appears_in_the_resolved_contract(self):
        resolved = resolve_embedding_contract(self.config)
        rendered = repr(resolved)
        for forbidden in ("SECRET_KEY_VAR", "api.example.com"):
            self.assertNotIn(forbidden, rendered)

    def test_the_service_imports_no_provider_or_http_client(self):
        imported = imported_names(embedding_contract)
        for forbidden in (
            "litellm", "litellm_client", "completion_call", "requests", "httpx",
            "openai", "ollama", "urllib", "socket", "os", "environ", "getenv",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, imported)
        self.assertEqual(
            imported & {"hashlib", "json", "unicodedata"},
            {"hashlib", "json", "unicodedata"},
            "the module should need only stdlib serialization helpers",
        )

    def test_resolving_makes_no_provider_call(self):
        with mock.patch(
            "ai_hub.services.litellm_client.completion_call"
        ) as completion, mock.patch(
            "ai_hub.services.provider_registry.resolve_model_config"
        ) as resolver:
            for _ in range(3):
                resolve_embedding_contract(self.config)
        completion.assert_not_called()
        resolver.assert_not_called()

    def test_no_embedding_generation_function_exists(self):
        for forbidden in (
            "embed_text", "generate_embedding", "embedding_call", "create_vector",
            "embed", "encode",
        ):
            self.assertFalse(
                hasattr(embedding_contract, forbidden),
                f"{forbidden}() must not exist in S-18",
            )

    def test_the_service_implements_no_distance_computation(self):
        """The metric names are configuration facts, not implementations."""
        imported = imported_names(embedding_contract)
        for forbidden in ("numpy", "scipy", "math", "faiss", "torch"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, imported)
        referenced = referenced_names(embedding_contract)
        for forbidden in (
            "dot", "sqrt", "cosine_similarity", "nearest_neighbours",
            "similarity", "search", "query_vector",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, referenced)

    def test_no_k1_is_implemented_yet(self):
        """The docstring may DESCRIBE the future relation; nothing computes it."""
        self.assertNotIn("k1:", code_strings(embedding_contract))
        for forbidden in ("k1_fingerprint", "chunk_text_fingerprint", "k1"):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(hasattr(embedding_contract, forbidden))

    def test_it_does_not_touch_the_knowledge_lifecycle_contracts(self):
        imported = imported_names(embedding_contract)
        referenced = referenced_names(embedding_contract)
        constants = code_strings(embedding_contract)
        for forbidden in ("knowledge_lifecycle", "KnowledgeLifecycleEvent"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, imported)
        for forbidden in ("chunk_authority_mode", "generation_input_fingerprint"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, referenced)
        for forbidden in ("i1:", "c1:"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, constants)


# ---------------------------------------------------------------------------
# Separation from the other boundaries
# ---------------------------------------------------------------------------

class BoundarySeparationTests(ContractFixtureMixin, TestCase):
    def setUp(self):
        self.scope = ApplicationScope.objects.create(name="App", slug="app")
        self.collection = KnowledgeCollection.objects.create(
            name="C", application_scope=self.scope
        )
        self.provider = self.make_provider(
            "Sep", "openai", declared_locality=LOCALITY.LOCAL
        )
        self.config = self.make_config(self.provider, name="sep-embed")

    def test_a_contract_alone_does_not_authorize_provider_use(self):
        from ai_hub.services.embedding_egress import (
            PAYLOAD_CORPUS,
            ReasonCode,
            resolve_embedding_access,
        )

        resolve_embedding_contract(self.config)      # contract is fine
        decision = resolve_embedding_access(
            self.scope, self.provider,
            collection=self.collection, payload_kind=PAYLOAD_CORPUS,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.NO_PROVIDER_GRANT)

    def test_a_grant_alone_does_not_produce_a_usable_contract(self):
        from ai_hub.services.embedding_egress import (
            PAYLOAD_CORPUS,
            resolve_embedding_access,
        )

        ProviderGrant.objects.create(
            application_scope=self.scope, provider=self.provider,
            allow_embeddings=True,
        )
        self.assertTrue(
            resolve_embedding_access(
                self.scope, self.provider,
                collection=self.collection, payload_kind=PAYLOAD_CORPUS,
            ).allowed
        )
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(is_active=False)
        self.config.refresh_from_db()
        with self.assertRaises(EmbeddingContractError):
            resolve_embedding_contract(self.config)

    def test_the_three_results_are_distinct_types(self):
        from ai_hub.services.embedding_egress import EmbeddingAccessDecision
        from ai_hub.services.knowledge_authorization import EffectiveKnowledgeScope

        types = {EffectiveKnowledgeScope, EmbeddingAccessDecision, ResolvedEmbeddingContract}
        self.assertEqual(len(types), 3)
        for other in (EffectiveKnowledgeScope, EmbeddingAccessDecision):
            self.assertNotEqual(
                set(other.__dataclass_fields__),
                set(ResolvedEmbeddingContract.__dataclass_fields__),
            )

    def test_the_contract_service_performs_no_authorization_lookup(self):
        """The docstring names the other boundaries; the CODE never calls them."""
        imported = imported_names(embedding_contract)
        referenced = referenced_names(embedding_contract)
        for forbidden in (
            "ProviderGrant", "resolve_embedding_access", "EffectiveKnowledgeScope",
            "resolve_effective_knowledge_scope", "embedding_egress",
            "knowledge_authorization",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, imported)
                self.assertNotIn(forbidden, referenced)
        for forbidden in (
            "allow_external_embedding_corpus_egress",
            "allow_external_embedding_query_egress",
            "allow_embeddings",
            "external_embedding_egress_policy",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, referenced)


# ---------------------------------------------------------------------------
# Query shape
# ---------------------------------------------------------------------------

class ContractQueryShapeTests(ContractFixtureMixin, TestCase):
    def setUp(self):
        self.provider = self.make_provider("Q", "openai")
        self.config = self.make_config(self.provider, name="q-embed")

    def test_resolution_needs_no_query_when_the_provider_is_already_loaded(self):
        config = EmbeddingModelConfig.objects.select_related("provider").get(
            pk=self.config.pk
        )
        with self.assertNumQueries(0):
            resolve_embedding_contract(config)

    def test_it_queries_no_corpus_agent_or_authorization_table(self):
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        config = EmbeddingModelConfig.objects.get(pk=self.config.pk)
        with CaptureQueriesContext(connection) as captured:
            resolve_embedding_contract(config)
        rendered = " ".join(query["sql"] for query in captured)
        for table in (
            "knowledgedocument", "knowledgedocumentchunk", "agentprofile",
            "providergrant", "applicationscope", "knowledgecollection",
        ):
            self.assertNotIn(table, rendered.lower())


# ---------------------------------------------------------------------------
# Existing completion runtime unaffected
# ---------------------------------------------------------------------------

class CompletionCompatibilityTests(ContractFixtureMixin, TestCase):
    def test_completion_resolution_is_unchanged(self):
        from ai_hub.services.provider_registry import resolve_model_config

        provider = ProviderConfig.objects.create(
            name="Training", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        model_config = ModelConfig.objects.create(
            provider=provider, model_name="training",
            temperature_default=Decimal("0.20"),
        )
        resolved = resolve_model_config(model_config)
        self.assertEqual(resolved["model"], "training")
        self.assertEqual(resolved["temperature"], 0.20)
        for absent in ("vector_dimension", "distance_metric", "normalization", "e1"):
            self.assertNotIn(absent, resolved)

    def test_an_embedding_config_does_not_affect_completion_selection(self):
        from ai_hub.services.provider_registry import resolve_model_config

        provider = ProviderConfig.objects.create(
            name="Shared", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        model_config = ModelConfig.objects.create(
            provider=provider, model_name="training"
        )
        before = resolve_model_config(model_config)
        self.make_config(provider, name="shared-embed")
        self.assertEqual(resolve_model_config(model_config), before)

    def test_no_completion_module_imports_the_contract_service(self):
        import ai_hub.services.agent_runtime as agent_runtime
        import ai_hub.services.execution_runner as execution_runner
        import ai_hub.services.provider_registry as provider_registry

        for module in (provider_registry, agent_runtime, execution_runner):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("embedding_contract", source)
                self.assertNotIn("EmbeddingModelConfig", source)


# ---------------------------------------------------------------------------
# Admin reachability
# ---------------------------------------------------------------------------

class EmbeddingAdminTests(ContractFixtureMixin, TestCase):
    def setUp(self):
        from django.contrib import admin as django_admin

        self.registry = django_admin.site._registry

    def test_the_model_is_registered(self):
        self.assertIn(EmbeddingModelConfig, self.registry)

    def test_the_admin_urls_reverse(self):
        from django.urls import reverse

        self.assertTrue(reverse("admin:ai_hub_embeddingmodelconfig_add"))
        self.assertTrue(reverse("admin:ai_hub_embeddingmodelconfig_changelist"))

    def test_every_contracted_field_is_operator_visible(self):
        model_admin = self.registry[EmbeddingModelConfig]
        for field in (
            "name", "provider", "model_name", "model_revision", "vector_dimension",
            "distance_metric", "normalization", "max_input_chars",
            "request_timeout_seconds", "is_active",
        ):
            with self.subTest(field=field):
                self.assertIn(field, model_admin.fields)
                self.assertNotIn(field, model_admin.get_readonly_fields(None))

    def test_the_e1_display_delegates_to_the_canonical_implementation(self):
        model_admin = self.registry[EmbeddingModelConfig]
        config = self.make_config(name="admin-embed")
        rendered = model_admin.embedding_contract_fingerprint(config)
        self.assertEqual(rendered, resolve_embedding_contract(config).e1)
        self.assertIn("embedding_contract_fingerprint", model_admin.readonly_fields)

    def test_the_admin_does_not_reimplement_the_fingerprint(self):
        """It must delegate, so a serialization change can never diverge."""
        import ai_hub.admin as admin_module

        imported = imported_names(admin_module)
        for forbidden in ("hashlib", "sha256"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, imported)
        self.assertIn("embedding_contract_fingerprint", imported)

    def test_the_admin_exposes_no_credentials(self):
        model_admin = self.registry[EmbeddingModelConfig]
        rendered = " ".join(model_admin.fields) + " ".join(model_admin.list_display)
        for forbidden in ("api_key", "credential", "secret", "token"):
            self.assertNotIn(forbidden, rendered)
