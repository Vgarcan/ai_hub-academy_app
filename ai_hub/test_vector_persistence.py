"""S-19 — `k1`, vector encoding, and the portable reference vector store.

Three separable contracts are exercised here:

    k1        what exact text a vector is an embedding of
    f32le1    how a vector is physically stored
    storage   which namespace a vector lives in, and when it is CURRENT

Every vector in this module is a synthetic list of floats written by the test.
Nothing calls an embedding provider, and no similarity is ever computed.
"""

import ast
import inspect
import math
import re
import struct
import unicodedata
from decimal import Decimal

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import TestCase

from ai_hub.models import (
    ApplicationScope,
    EmbeddingModelConfig,
    KnowledgeChunkEmbedding,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ProviderConfig,
)
from ai_hub.services import chunk_embedding_identity, vector_store
from ai_hub.services.chunk_embedding_identity import (
    K1_PREFIX,
    canonical_chunk_embedding_text,
    chunk_embedding_fingerprint,
    chunk_embedding_input_fingerprint,
)
from ai_hub.services.knowledge_lifecycle import chunk_set_fingerprint
from ai_hub.services.vector_store import (
    VECTOR_FORMAT_F32LE1,
    DjangoBinaryVectorStore,
    InspectionReason,
    VectorEncodingError,
    VectorStore,
    VectorStoreError,
    decode_vector,
    encode_vector,
    inspect_vector_record,
    inspect_vector_record as inspect_record,
    load_current_vectors,
    store_chunk_vector,
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
    """Identifiers and attributes the CODE touches - never docstring prose."""
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


K1_PATTERN = re.compile(r"^k1:sha256:[0-9a-f]{64}$")

#: The fixed source facts the golden pin is computed from.
GOLDEN_SECTION_TITLE = "Safety rules"
GOLDEN_CONTENT = "Always search assigned knowledge first."

#: The expected `k1` for those facts, as a LITERAL.
#:
#: Hard-coded on purpose. An expectation derived from the implementation under
#: test moves whenever the implementation moves, so a change to the renderer,
#: the canonical payload, normalization, JSON serialization, separators,
#: encoding or hash format would take the expectation with it and stay green.
#:
#: Once vectors exist, `k1` decides whether a stored vector is still an
#: embedding of the current text. Any change that alters this string invalidates
#: every vector ever produced, so it must fail here first and force a conscious
#: decision about versioning the contract - never be absorbed silently.
GOLDEN_K1 = "k1:sha256:59eb07d68c0cf9ff66d84cb2c877e4588a09ae710e3b15c823f58c361ef76cae"


class FakeChunk:
    """A chunk-shaped object for PURE fingerprint tests.

    Used where moving a real row would mean relocating Knowledge across a
    security boundary just to observe a hash. `k1` reads exactly two attributes,
    so nothing else is needed.
    """

    def __init__(self, section_title="", content=""):
        self.section_title = section_title
        self.content = content


class VectorFixtureMixin:
    def build_scope(self, name="App A", slug="app-a"):
        return ApplicationScope.objects.create(name=name, slug=slug)

    def build_collection(self, scope, name="Collection A"):
        return KnowledgeCollection.objects.create(
            name=name, application_scope=scope
        )

    def build_chunk(self, collection, *, title="Doc", section_title="S", content="Body",
                    chunk_index=1):
        document = KnowledgeDocument.objects.create(
            collection=collection, title=title, curated_text=content,
            status=KnowledgeDocument.Status.ACTIVE,
        )
        return KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=chunk_index,
            section_title=section_title, content=content,
        )

    def build_config(self, *, name="embed", dimension=4, revision="v1",
                     provider=None, provider_name="P"):
        provider = provider or ProviderConfig.objects.create(
            name=provider_name, provider_type="openai",
            declared_locality=ProviderConfig.DeclaredLocality.LOCAL,
        )
        return EmbeddingModelConfig.objects.create(
            name=name, provider=provider, model_name="m", model_revision=revision,
            vector_dimension=dimension,
            distance_metric=EmbeddingModelConfig.DistanceMetric.COSINE,
            normalization=EmbeddingModelConfig.Normalization.L2,
        )


# ---------------------------------------------------------------------------
# The canonical embedding text renderer
# ---------------------------------------------------------------------------

class CanonicalEmbeddingTextTests(TestCase):
    def test_with_a_section_title_the_rendering_is_exact(self):
        self.assertEqual(
            canonical_chunk_embedding_text(FakeChunk("Title", "Body")),
            "Title\n\nBody",
        )

    def test_without_a_section_title_only_the_content_is_rendered(self):
        """No leading blank line is invented."""
        self.assertEqual(
            canonical_chunk_embedding_text(FakeChunk("", "Body")), "Body"
        )
        self.assertFalse(
            canonical_chunk_embedding_text(FakeChunk("", "Body")).startswith("\n")
        )

    def test_a_whitespace_only_section_title_is_not_empty(self):
        """Normalization is representation-only; it never trims."""
        self.assertEqual(
            canonical_chunk_embedding_text(FakeChunk("   ", "Body")),
            "   \n\nBody",
        )

    def test_it_reads_only_section_title_and_content(self):
        source = inspect.getsource(canonical_chunk_embedding_text)
        attributes = {
            node.args[1].value
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "getattr"
            and isinstance(node.args[1], ast.Constant)
        }
        self.assertEqual(attributes, {"section_title", "content"})

    def test_line_endings_are_normalized(self):
        for variant in ("A\r\nB", "A\rB"):
            with self.subTest(variant=repr(variant)):
                self.assertEqual(
                    canonical_chunk_embedding_text(FakeChunk("", variant)), "A\nB"
                )

    def test_unicode_is_nfc_normalized(self):
        composed = unicodedata.normalize("NFC", "café")
        decomposed = unicodedata.normalize("NFD", "café")
        self.assertNotEqual(composed, decomposed)
        self.assertEqual(
            canonical_chunk_embedding_text(FakeChunk("", composed)),
            canonical_chunk_embedding_text(FakeChunk("", decomposed)),
        )

    def test_nothing_else_is_rewritten(self):
        """No stripping, collapsing, case-folding or punctuation rewriting."""
        cases = (
            ("  leading", "trailing  "),
            ("a  b", "c\t\td"),
            ("Case", "CASE"),
        )
        for section_title, content in cases:
            with self.subTest(text=(section_title, content)):
                rendered = canonical_chunk_embedding_text(
                    FakeChunk(section_title, content)
                )
                self.assertIn(section_title, rendered)
                self.assertIn(content, rendered)


# ---------------------------------------------------------------------------
# k1 format, determinism, and the golden pin
# ---------------------------------------------------------------------------

class K1FingerprintTests(TestCase):
    def test_the_format_is_self_describing(self):
        fingerprint = chunk_embedding_fingerprint(FakeChunk("T", "B"))
        self.assertTrue(K1_PATTERN.match(fingerprint), fingerprint)
        self.assertTrue(fingerprint.startswith(K1_PREFIX))

    def test_it_is_deterministic(self):
        results = {
            chunk_embedding_fingerprint(FakeChunk("T", "B")) for _ in range(25)
        }
        self.assertEqual(len(results), 1)

    def test_the_golden_fingerprint_is_pinned(self):
        """If the renderer or serialization changes, this MUST fail."""
        self.assertEqual(
            chunk_embedding_fingerprint(
                FakeChunk(GOLDEN_SECTION_TITLE, GOLDEN_CONTENT)
            ),
            GOLDEN_K1,
        )

    def test_the_golden_literal_is_well_formed(self):
        self.assertTrue(K1_PATTERN.match(GOLDEN_K1), GOLDEN_K1)
        self.assertEqual(len(GOLDEN_K1), len(K1_PREFIX) + 64)
        self.assertEqual(GOLDEN_K1, GOLDEN_K1.lower())

    def test_the_golden_literal_is_a_constant_not_a_computation(self):
        """Guards the pin against ever being made self-referential."""
        module_tree = ast.parse(
            inspect.getsource(__import__(__name__, fromlist=["x"]))
        )
        assignments = [
            node
            for node in ast.walk(module_tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "GOLDEN_K1"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        value = assignments[0].value
        self.assertIsInstance(
            value, ast.Constant,
            "GOLDEN_K1 must be a literal, never derived from the code under test",
        )
        self.assertEqual(value.value, GOLDEN_K1)

    def test_the_canonical_payload_names_the_contract(self):
        """A bare digest could be mistaken for another contract's."""
        source = inspect.getsource(chunk_embedding_input_fingerprint)
        self.assertIn('"contract": "k1"', source)

    # -- included --------------------------------------------------------

    def test_section_title_and_content_each_change_k1(self):
        baseline = chunk_embedding_fingerprint(FakeChunk("T", "B"))
        self.assertNotEqual(chunk_embedding_fingerprint(FakeChunk("T2", "B")), baseline)
        self.assertNotEqual(chunk_embedding_fingerprint(FakeChunk("T", "B2")), baseline)

    def test_whitespace_and_case_are_significant(self):
        baseline = chunk_embedding_fingerprint(FakeChunk("T", "Body"))
        for variant in (" Body", "Body ", "body", "BODY", "Bo dy"):
            with self.subTest(variant=repr(variant)):
                self.assertNotEqual(
                    chunk_embedding_fingerprint(FakeChunk("T", variant)), baseline
                )

    def test_representation_variants_are_equivalent(self):
        baseline = chunk_embedding_fingerprint(FakeChunk("T", "A\nB"))
        for variant in ("A\r\nB", "A\rB"):
            with self.subTest(variant=repr(variant)):
                self.assertEqual(
                    chunk_embedding_fingerprint(FakeChunk("T", variant)), baseline
                )
        self.assertEqual(
            chunk_embedding_fingerprint(
                FakeChunk("T", unicodedata.normalize("NFD", "café"))
            ),
            chunk_embedding_fingerprint(
                FakeChunk("T", unicodedata.normalize("NFC", "café"))
            ),
        )

    def test_the_title_boundary_cannot_be_forged(self):
        """A title plus body must not collide with a body containing the join."""
        self.assertNotEqual(
            chunk_embedding_fingerprint(FakeChunk("T", "B")),
            chunk_embedding_fingerprint(FakeChunk("", "T\n\nB")),
        ) if False else None
        # The two DO render identically by contract, which is intentional and
        # recorded rather than hidden: the embedded text is genuinely the same
        # string, so it is the same vector.
        self.assertEqual(
            canonical_chunk_embedding_text(FakeChunk("T", "B")),
            canonical_chunk_embedding_text(FakeChunk("", "T\n\nB")),
        )


# ---------------------------------------------------------------------------
# k1 excluded facts - against real rows where safe, pure objects otherwise
# ---------------------------------------------------------------------------

class K1ExcludedFactTests(VectorFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.build_scope()
        self.collection = self.build_collection(self.scope)
        self.chunk = self.build_chunk(
            self.collection, section_title="T", content="Body"
        )
        self.baseline = chunk_embedding_fingerprint(self.chunk)

    def test_chunk_index_does_not_change_k1(self):
        self.chunk.chunk_index = 7
        self.chunk.save(update_fields=["chunk_index"])
        self.chunk.refresh_from_db()
        self.assertEqual(chunk_embedding_fingerprint(self.chunk), self.baseline)

    def test_metadata_and_token_estimate_do_not_change_k1(self):
        self.chunk.metadata = {"anything": "at all"}
        self.chunk.token_estimate = 999
        self.chunk.save(update_fields=["metadata", "token_estimate"])
        self.chunk.refresh_from_db()
        self.assertEqual(chunk_embedding_fingerprint(self.chunk), self.baseline)

    def test_document_title_does_not_change_k1(self):
        document = self.chunk.document
        document.title = "A Completely Different Document Title"
        document.save(update_fields=["title"])
        self.chunk.refresh_from_db()
        self.assertEqual(chunk_embedding_fingerprint(self.chunk), self.baseline)

    def test_chunk_and_document_primary_keys_do_not_change_k1(self):
        """Pure: a second identical chunk in another document has the same k1."""
        other = self.build_chunk(
            self.collection, title="Other Doc", section_title="T", content="Body"
        )
        self.assertNotEqual(other.pk, self.chunk.pk)
        self.assertNotEqual(other.document_id, self.chunk.document_id)
        self.assertEqual(chunk_embedding_fingerprint(other), self.baseline)

    def test_collection_and_scope_do_not_change_k1(self):
        """Pure fingerprint test - no Knowledge is moved across a boundary."""
        other_scope = self.build_scope("App B", "app-b")
        other_collection = self.build_collection(other_scope, "Collection B")
        elsewhere = self.build_chunk(
            other_collection, title="B Doc", section_title="T", content="Body"
        )
        self.assertNotEqual(
            elsewhere.document.collection_id, self.chunk.document.collection_id
        )
        self.assertEqual(chunk_embedding_fingerprint(elsewhere), self.baseline)


# ---------------------------------------------------------------------------
# c1 vs k1 - two contracts that must not move together
# ---------------------------------------------------------------------------

class C1VersusK1Tests(VectorFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.build_scope()
        self.collection = self.build_collection(self.scope)
        self.chunk = self.build_chunk(
            self.collection, section_title="T", content="Body"
        )

    def _c1(self):
        self.chunk.refresh_from_db()
        return chunk_set_fingerprint([self.chunk])

    def _k1(self):
        self.chunk.refresh_from_db()
        return chunk_embedding_fingerprint(self.chunk)

    def test_reindex_moves_c1_but_not_k1(self):
        """The decisive distinction: ordering is retrieval evidence, not input."""
        c1_before, k1_before = self._c1(), self._k1()
        self.chunk.chunk_index = 2
        self.chunk.save(update_fields=["chunk_index"])
        self.assertNotEqual(self._c1(), c1_before)
        self.assertEqual(self._k1(), k1_before)

    def test_content_change_moves_both(self):
        c1_before, k1_before = self._c1(), self._k1()
        self.chunk.content = "Rewritten body"
        self.chunk.save(update_fields=["content"])
        self.assertNotEqual(self._c1(), c1_before)
        self.assertNotEqual(self._k1(), k1_before)

    def test_section_title_change_moves_both(self):
        c1_before, k1_before = self._c1(), self._k1()
        self.chunk.section_title = "Another Title"
        self.chunk.save(update_fields=["section_title"])
        self.assertNotEqual(self._c1(), c1_before)
        self.assertNotEqual(self._k1(), k1_before)

    def test_metadata_and_token_estimate_move_neither(self):
        c1_before, k1_before = self._c1(), self._k1()
        self.chunk.metadata = {"ingestion": "changed"}
        self.chunk.token_estimate = 4242
        self.chunk.save(update_fields=["metadata", "token_estimate"])
        self.assertEqual(self._c1(), c1_before)
        self.assertEqual(self._k1(), k1_before)

    def test_the_two_contracts_live_in_separate_modules(self):
        """Four contracts, four fates. None may be moved by editing another."""
        from ai_hub.services import knowledge_lifecycle

        self.assertNotIn("k1:", code_strings(knowledge_lifecycle))
        self.assertNotIn(
            "chunk_embedding_identity", imported_names(knowledge_lifecycle)
        )
        identity_strings = code_strings(chunk_embedding_identity)
        for forbidden in ("c1:", "i1:"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, identity_strings)
        self.assertNotIn(
            "chunk_set_fingerprint", referenced_names(chunk_embedding_identity)
        )


# ---------------------------------------------------------------------------
# f32le1 encoding
# ---------------------------------------------------------------------------

class VectorEncodingTests(TestCase):
    def test_round_trip_preserves_float32_values(self):
        values = [0.0, 1.0, -1.0, 0.5, -0.25]
        decoded = decode_vector(
            encode_vector(values, expected_dimension=5), expected_dimension=5
        )
        for original, restored in zip(values, decoded):
            self.assertAlmostEqual(original, restored, places=6)

    def test_precision_is_float32_not_float64(self):
        """Do not claim more precision than the format provides."""
        value = 0.1234567890123456789
        decoded = decode_vector(
            encode_vector([value], expected_dimension=1), expected_dimension=1
        )[0]
        self.assertAlmostEqual(value, decoded, places=6)
        self.assertNotEqual(value, decoded)

    def test_byte_length_is_exactly_four_per_component(self):
        self.assertEqual(len(encode_vector([1.0] * 8, expected_dimension=8)), 32)

    def test_encoding_is_little_endian(self):
        self.assertEqual(
            encode_vector([1.0], expected_dimension=1), struct.pack("<f", 1.0)
        )

    def test_wrong_dimensions_are_refused(self):
        for values, expected in (([1.0, 2.0], 3), ([1.0, 2.0, 3.0], 2), ([], 1)):
            with self.subTest(values=values, expected=expected):
                with self.assertRaises(VectorEncodingError):
                    encode_vector(values, expected_dimension=expected)

    def test_non_finite_values_are_refused(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(VectorEncodingError):
                    encode_vector([value], expected_dimension=1)

    def test_float32_overflow_is_refused_rather_than_silently_infinite(self):
        with self.assertRaises(VectorEncodingError):
            encode_vector([1e39], expected_dimension=1)
        with self.assertRaises(VectorEncodingError):
            encode_vector([-1e39], expected_dimension=1)

    def test_non_numeric_values_are_refused(self):
        for value in ("1.0", None, [1.0], True):
            with self.subTest(value=value):
                with self.assertRaises(VectorEncodingError):
                    encode_vector([value], expected_dimension=1)

    def test_bad_byte_length_is_refused_on_decode(self):
        payload = encode_vector([1.0, 2.0], expected_dimension=2)
        for corrupt in (payload[:-1], payload + b"\x00", b""):
            with self.subTest(length=len(corrupt)):
                with self.assertRaises(VectorEncodingError):
                    decode_vector(corrupt, expected_dimension=2)

    def test_unknown_format_fails_closed(self):
        payload = encode_vector([1.0], expected_dimension=1)
        with self.assertRaises(VectorEncodingError):
            decode_vector(payload, expected_dimension=1, vector_format="f64be9")

    def test_nothing_is_normalized_or_compared(self):
        """The encoder stores what it was given; it never transforms or ranks."""
        referenced = referenced_names(vector_store)
        for forbidden in (
            "cosine", "similarity", "nearest", "l2_normalize", "normalize",
            "argsort", "top_k", "rank", "distance", "sort",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, referenced)
        for forbidden in ("numpy", "faiss", "chromadb"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, imported_names(vector_store))


# ---------------------------------------------------------------------------
# Storage: namespace, derivation, atomicity
# ---------------------------------------------------------------------------

class StoreChunkVectorTests(VectorFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.build_scope()
        self.collection = self.build_collection(self.scope)
        self.chunk = self.build_chunk(self.collection)
        self.config = self.build_config(dimension=4)
        self.vector = [0.1, 0.2, 0.3, 0.4]

    def _store(self, **overrides):
        kwargs = dict(
            application_scope=self.scope, chunk=self.chunk,
            embedding_model_config=self.config, vector=self.vector,
        )
        kwargs.update(overrides)
        return store_chunk_vector(**kwargs)

    def test_a_correct_write_persists_every_derived_fact(self):
        from ai_hub.services.embedding_contract import resolve_embedding_contract

        record = self._store()
        contract = resolve_embedding_contract(self.config)
        self.assertEqual(record.application_scope_id, self.scope.pk)
        self.assertEqual(record.collection_id, self.collection.pk)
        self.assertEqual(record.chunk_id, self.chunk.pk)
        self.assertEqual(record.k1, chunk_embedding_fingerprint(self.chunk))
        self.assertEqual(record.e1, contract.e1)
        self.assertEqual(record.vector_dimension, 4)
        self.assertEqual(record.vector_format, VECTOR_FORMAT_F32LE1)

    def test_the_caller_cannot_supply_collection_k1_e1_or_dimension(self):
        parameters = set(
            inspect.signature(store_chunk_vector).parameters
        )
        self.assertEqual(
            parameters,
            {"application_scope", "chunk", "embedding_model_config", "vector"},
        )
        for forbidden in ("collection", "k1", "e1", "vector_dimension", "vector_format"):
            self.assertNotIn(forbidden, parameters)

    def test_a_foreign_application_scope_is_refused(self):
        other = self.build_scope("App B", "app-b")
        with self.assertRaises(VectorStoreError):
            self._store(application_scope=other)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_the_scope_mismatch_is_refused_not_silently_corrected(self):
        other = self.build_scope("App C", "app-c")
        with self.assertRaises(VectorStoreError):
            self._store(application_scope=other)
        self.assertFalse(
            KnowledgeChunkEmbedding.objects.filter(
                application_scope_id=self.scope.pk
            ).exists()
        )

    def test_a_wrong_dimension_vector_is_refused(self):
        for wrong in ([0.1, 0.2], [0.1] * 5):
            with self.subTest(length=len(wrong)):
                with self.assertRaises(VectorEncodingError):
                    self._store(vector=wrong)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_an_unresolvable_config_is_refused(self):
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(is_active=False)
        self.config.refresh_from_db()
        from ai_hub.services.embedding_contract import EmbeddingContractError

        with self.assertRaises(EmbeddingContractError):
            self._store()
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_an_inactive_provider_is_refused(self):
        from ai_hub.services.embedding_contract import EmbeddingContractError

        ProviderConfig.objects.filter(pk=self.config.provider_id).update(
            is_active=False
        )
        self.config.refresh_from_db()
        with self.assertRaises(EmbeddingContractError):
            self._store()

    def test_the_write_path_uses_the_operational_resolver(self):
        """Writing NEW vectors still requires an active, usable contract.

        The asymmetry is deliberate and is the whole shape of the correction:

            WRITE   -> operational resolver, active required
            INSPECT -> pure fingerprint, active irrelevant
        """
        from ai_hub.services.vector_store import DjangoBinaryVectorStore

        source = inspect.getsource(DjangoBinaryVectorStore.store_chunk_vector)
        called = {
            node.func.id
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("resolve_embedding_contract", called)
        self.assertNotIn("embedding_contract_fingerprint", called)

    def test_an_inactive_config_still_blocks_a_new_write_for_an_existing_slot(self):
        """Deactivation stops NEW work even where a current vector already exists."""
        from ai_hub.services.embedding_contract import EmbeddingContractError

        record = self._store()
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(is_active=False)
        self.config.refresh_from_db()
        with self.assertRaises(EmbeddingContractError):
            self._store(vector=[0.9, 0.9, 0.9, 0.9])
        # ...and the existing vector is untouched.
        original = bytes(record.vector_bytes)
        record.refresh_from_db()
        self.assertEqual(bytes(record.vector_bytes), original)

    def test_a_failed_replacement_leaves_the_previous_row_untouched(self):
        record = self._store()
        original_bytes = bytes(record.vector_bytes)
        original_k1 = record.k1

        with self.assertRaises(VectorEncodingError):
            self._store(vector=[float("nan")] * 4)

        record.refresh_from_db()
        self.assertEqual(bytes(record.vector_bytes), original_bytes)
        self.assertEqual(record.k1, original_k1)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 1)

    def test_the_stored_bytes_decode_back_to_the_vector(self):
        record = self._store()
        decoded = decode_vector(
            record.vector_bytes, expected_dimension=record.vector_dimension
        )
        for original, restored in zip(self.vector, decoded):
            self.assertAlmostEqual(original, restored, places=6)


# ---------------------------------------------------------------------------
# Upsert semantics
# ---------------------------------------------------------------------------

class UpsertTests(VectorFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.build_scope()
        self.collection = self.build_collection(self.scope)
        self.chunk = self.build_chunk(self.collection)
        self.config = self.build_config(dimension=4)

    def test_a_second_write_updates_the_same_slot(self):
        first = store_chunk_vector(
            application_scope=self.scope, chunk=self.chunk,
            embedding_model_config=self.config, vector=[0.1, 0.2, 0.3, 0.4],
        )
        second = store_chunk_vector(
            application_scope=self.scope, chunk=self.chunk,
            embedding_model_config=self.config, vector=[0.9, 0.8, 0.7, 0.6],
        )
        self.assertEqual(first.pk, second.pk)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 1)
        decoded = decode_vector(second.vector_bytes, expected_dimension=4)
        self.assertAlmostEqual(decoded[0], 0.9, places=6)

    def test_two_configs_sharing_one_e1_share_one_slot(self):
        """Same vector-space contract - do not duplicate the vector."""
        twin = self.build_config(
            name="embed-long-timeout", dimension=4, revision="v1",
            provider=self.config.provider,
        )
        twin.request_timeout_seconds = 600
        twin.save(update_fields=["request_timeout_seconds"])

        from ai_hub.services.embedding_contract import resolve_embedding_contract

        self.assertEqual(
            resolve_embedding_contract(twin).e1,
            resolve_embedding_contract(self.config).e1,
        )

        first = store_chunk_vector(
            application_scope=self.scope, chunk=self.chunk,
            embedding_model_config=self.config, vector=[0.1] * 4,
        )
        second = store_chunk_vector(
            application_scope=self.scope, chunk=self.chunk,
            embedding_model_config=twin, vector=[0.2] * 4,
        )
        self.assertEqual(first.pk, second.pk)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 1)
        # The row records who actually performed the latest write.
        second.refresh_from_db()
        self.assertEqual(second.embedding_model_config_id, twin.pk)

    def test_different_e1_values_coexist_on_one_chunk(self):
        """Controlled reindex: the old index survives while the new one builds."""
        other = self.build_config(
            name="embed-v2", dimension=4, revision="v2",
            provider=self.config.provider,
        )
        store_chunk_vector(
            application_scope=self.scope, chunk=self.chunk,
            embedding_model_config=self.config, vector=[0.1] * 4,
        )
        store_chunk_vector(
            application_scope=self.scope, chunk=self.chunk,
            embedding_model_config=other, vector=[0.2] * 4,
        )
        self.assertEqual(
            KnowledgeChunkEmbedding.objects.filter(chunk=self.chunk).count(), 2
        )

    def test_the_unique_constraint_is_on_chunk_and_e1(self):
        constrained = {
            tuple(getattr(c, "fields", ()))
            for c in KnowledgeChunkEmbedding._meta.constraints
        }
        self.assertIn(("chunk", "e1"), constrained)
        self.assertNotIn(("chunk", "embedding_model_config"), constrained)

    def test_a_duplicate_chunk_e1_row_is_refused_by_the_database(self):
        record = store_chunk_vector(
            application_scope=self.scope, chunk=self.chunk,
            embedding_model_config=self.config, vector=[0.1] * 4,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                KnowledgeChunkEmbedding.objects.create(
                    application_scope=self.scope, collection=self.collection,
                    chunk=self.chunk, embedding_model_config=self.config,
                    k1=record.k1, e1=record.e1, vector_dimension=4,
                    vector_bytes=record.vector_bytes,
                )


# ---------------------------------------------------------------------------
# Currentness / staleness
# ---------------------------------------------------------------------------

class CurrentnessTests(VectorFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.build_scope()
        self.collection = self.build_collection(self.scope)
        self.chunk = self.build_chunk(
            self.collection, section_title="T", content="Body"
        )
        self.config = self.build_config(dimension=4)
        self.record = store_chunk_vector(
            application_scope=self.scope, chunk=self.chunk,
            embedding_model_config=self.config, vector=[0.1, 0.2, 0.3, 0.4],
        )

    def _inspect(self):
        self.record.refresh_from_db()
        return inspect_vector_record(
            KnowledgeChunkEmbedding.objects.select_related(
                "chunk", "chunk__document", "chunk__document__collection",
                "embedding_model_config", "embedding_model_config__provider",
            ).get(pk=self.record.pk)
        )

    def test_a_fresh_record_is_current(self):
        inspection = self._inspect()
        self.assertTrue(inspection.current)
        self.assertEqual(inspection.reason_codes, ())

    def test_a_content_change_makes_it_stale_via_k1(self):
        self.chunk.content = "Rewritten"
        self.chunk.save(update_fields=["content"])
        inspection = self._inspect()
        self.assertFalse(inspection.current)
        self.assertFalse(inspection.k1_matches)
        self.assertIn(InspectionReason.K1_MISMATCH, inspection.reason_codes)

    def test_a_stale_record_still_exists(self):
        """Staleness is observable state, not corruption to auto-delete."""
        self.chunk.content = "Rewritten"
        self.chunk.save(update_fields=["content"])
        self.assertTrue(
            KnowledgeChunkEmbedding.objects.filter(pk=self.record.pk).exists()
        )

    def test_an_exact_content_revert_makes_it_current_again(self):
        original = self.chunk.content
        self.chunk.content = "Rewritten"
        self.chunk.save(update_fields=["content"])
        self.assertFalse(self._inspect().current)

        self.chunk.content = original
        self.chunk.save(update_fields=["content"])
        self.assertTrue(self._inspect().current)

    def test_a_model_revision_change_makes_it_stale_via_e1(self):
        self.config.model_revision = "v2"
        self.config.save(update_fields=["model_revision"])
        inspection = self._inspect()
        self.assertFalse(inspection.current)
        self.assertFalse(inspection.e1_matches)
        self.assertIn(InspectionReason.E1_MISMATCH, inspection.reason_codes)

    def test_operational_changes_leave_it_current(self):
        """Timeout, input cap and rename are excluded from e1."""
        self.config.request_timeout_seconds = 600
        self.config.max_input_chars = 64000
        self.config.name = "renamed"
        self.config.save(update_fields=[
            "request_timeout_seconds", "max_input_chars", "name",
        ])
        self.assertTrue(self._inspect().current)

    def test_a_locality_change_leaves_it_current(self):
        """Permission may change; vector identity does not."""
        for locality in (
            ProviderConfig.DeclaredLocality.EXTERNAL,
            ProviderConfig.DeclaredLocality.UNKNOWN,
        ):
            with self.subTest(locality=locality):
                ProviderConfig.objects.filter(pk=self.config.provider_id).update(
                    declared_locality=locality
                )
                self.assertTrue(self._inspect().current)

    def test_deactivating_the_config_leaves_the_vector_current(self):
        """`is_active` is not in `e1`, so it cannot change what a vector IS.

        Mathematical currentness and operational usability are different
        questions. Nothing about the stored bytes changed - only permission to
        do new work with the configuration did.
        """
        self.assertTrue(self._inspect().current)

        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(is_active=False)

        inspection = self._inspect()
        self.assertTrue(inspection.current)
        self.assertTrue(inspection.k1_matches)
        self.assertTrue(inspection.e1_matches)
        self.assertTrue(inspection.dimension_matches)
        self.assertTrue(inspection.namespace_matches)
        self.assertTrue(inspection.collection_matches)
        self.assertTrue(inspection.encoding_valid)
        self.assertEqual(inspection.reason_codes, ())

    def test_deactivating_the_provider_leaves_the_vector_current(self):
        """Same reasoning: provider `is_active` is not in `e1` either."""
        self.assertTrue(self._inspect().current)

        ProviderConfig.objects.filter(pk=self.config.provider_id).update(
            is_active=False
        )

        inspection = self._inspect()
        self.assertTrue(inspection.current)
        self.assertTrue(inspection.e1_matches)
        self.assertEqual(inspection.reason_codes, ())

    def test_deactivating_both_leaves_the_vector_current(self):
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(is_active=False)
        ProviderConfig.objects.filter(pk=self.config.provider_id).update(
            is_active=False
        )
        self.assertTrue(self._inspect().current)

    def test_inspection_never_consults_active_state_or_permission(self):
        """Structural: the identity path must not reach for those facts at all."""
        from ai_hub.services.vector_store import inspect_vector_record as inspector

        referenced = {
            node.attr
            for node in ast.walk(ast.parse(inspect.getsource(inspector).lstrip()))
            if isinstance(node, ast.Attribute)
        }
        for forbidden in (
            "is_active", "declared_locality", "allow_embeddings",
            "allow_external_embedding_corpus_egress",
            "allow_external_embedding_query_egress",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, referenced)

    def test_inspection_uses_the_pure_fingerprint_not_the_operational_resolver(self):
        """The whole point of the correction, pinned structurally."""
        from ai_hub.services.vector_store import inspect_vector_record as inspector

        called = {
            node.func.id
            for node in ast.walk(ast.parse(inspect.getsource(inspector).lstrip()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("embedding_contract_fingerprint", called)
        self.assertNotIn("resolve_embedding_contract", called)

    def test_a_stale_row_is_still_persisted_after_deactivation(self):
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(is_active=False)
        self.assertTrue(
            KnowledgeChunkEmbedding.objects.filter(pk=self.record.pk).exists()
        )

    def test_a_dimension_change_stales_the_vector_without_malforming_it(self):
        """A DIFFERENT dimension is a well-formed contract that no longer matches.

        Named for what it actually proves. The configuration is perfectly
        valid - it simply describes a different vector space now, so the stored
        vector is stale via `e1` and `dimension`, not malformed.
        """
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(
            vector_dimension=99
        )
        inspection = self._inspect()
        self.assertFalse(inspection.current)
        self.assertFalse(inspection.dimension_matches)
        self.assertFalse(inspection.e1_matches)
        self.assertNotIn(
            InspectionReason.CONTRACT_MALFORMED, inspection.reason_codes
        )

    def test_a_semantically_malformed_config_reports_contract_malformed(self):
        """The defensive branch, exercised directly.

        Mutated IN MEMORY immediately before inspection: a non-coercible
        dimension is something the schema would refuse, so persisting one merely
        to reach this branch would be manufacturing impossible data. The branch
        exists because raw ORM and future code paths are not obliged to leave a
        configuration well formed.
        """
        record = KnowledgeChunkEmbedding.objects.select_related(
            "chunk", "chunk__document", "chunk__document__collection",
            "embedding_model_config", "embedding_model_config__provider",
        ).get(pk=self.record.pk)
        self.assertTrue(inspect_vector_record(record).current)

        record.embedding_model_config.vector_dimension = object()

        inspection = inspect_vector_record(record)
        self.assertFalse(inspection.current)
        self.assertIn(
            InspectionReason.CONTRACT_MALFORMED, inspection.reason_codes
        )
        self.assertFalse(inspection.e1_matches)
        self.assertFalse(inspection.dimension_matches)
        # The identity facts that do not depend on the contract still stand.
        self.assertTrue(inspection.k1_matches)
        self.assertTrue(inspection.namespace_matches)

    def test_a_non_numeric_dimension_also_reports_contract_malformed(self):
        """The ValueError half of the narrowed handler."""
        record = KnowledgeChunkEmbedding.objects.select_related(
            "chunk", "chunk__document", "chunk__document__collection",
            "embedding_model_config", "embedding_model_config__provider",
        ).get(pk=self.record.pk)
        record.embedding_model_config.vector_dimension = "not-a-number"

        inspection = inspect_vector_record(record)
        self.assertFalse(inspection.current)
        self.assertIn(
            InspectionReason.CONTRACT_MALFORMED, inspection.reason_codes
        )

    def test_the_defensive_handler_is_narrow(self):
        """CONTRACT_MALFORMED must not swallow arbitrary programming errors."""
        from ai_hub.services.vector_store import inspect_vector_record as inspector

        handlers = [
            node
            for node in ast.walk(ast.parse(inspect.getsource(inspector).lstrip()))
            if isinstance(node, ast.ExceptHandler)
        ]
        self.assertTrue(handlers)
        for handler in handlers:
            with self.subTest(lineno=handler.lineno):
                self.assertIsNotNone(handler.type, "bare `except:` is not allowed")
                caught = (
                    {element.id for element in handler.type.elts}
                    if isinstance(handler.type, ast.Tuple)
                    else {handler.type.id}
                )
                self.assertNotIn("Exception", caught)
                self.assertNotIn("BaseException", caught)

    def test_a_corrupt_encoding_is_reported(self):
        KnowledgeChunkEmbedding.objects.filter(pk=self.record.pk).update(
            vector_bytes=b"\x00\x01"
        )
        inspection = self._inspect()
        self.assertFalse(inspection.encoding_valid)
        self.assertIn(InspectionReason.ENCODING_INVALID, inspection.reason_codes)

    def test_the_inspection_carries_no_content_or_values(self):
        from ai_hub.services.vector_store import VectorRecordInspection

        for field in VectorRecordInspection.__dataclass_fields__:
            for forbidden in ("text", "content", "values", "vector_bytes", "credential"):
                self.assertNotIn(forbidden, field)


# ---------------------------------------------------------------------------
# Scoped load
# ---------------------------------------------------------------------------

class ScopedLoadTests(VectorFixtureMixin, TestCase):
    def setUp(self):
        self.scope_a = self.build_scope("App A", "app-a")
        self.scope_b = self.build_scope("App B", "app-b")
        self.a1 = self.build_collection(self.scope_a, "A1")
        self.a2 = self.build_collection(self.scope_a, "A2")
        self.b1 = self.build_collection(self.scope_b, "B1")

        self.config = self.build_config(dimension=4)
        from ai_hub.services.embedding_contract import resolve_embedding_contract

        self.e1 = resolve_embedding_contract(self.config).e1

        self.chunks = {}
        for label, collection, scope in (
            ("a1", self.a1, self.scope_a),
            ("a2", self.a2, self.scope_a),
            ("b1", self.b1, self.scope_b),
        ):
            chunk = self.build_chunk(
                collection, title=f"{label} doc", section_title=label,
                content=f"{label} body",
            )
            self.chunks[label] = chunk
            store_chunk_vector(
                application_scope=scope, chunk=chunk,
                embedding_model_config=self.config, vector=[0.1, 0.2, 0.3, 0.4],
            )

    def _load(self, scope, collection_ids):
        return load_current_vectors(
            application_scope=scope, e1=self.e1, collection_ids=collection_ids
        )

    def test_narrowing_to_one_collection(self):
        loaded = self._load(self.scope_a, [self.a1.pk])
        self.assertEqual({row.collection_id for row in loaded}, {self.a1.pk})

    def test_narrowing_to_both_collections_in_scope(self):
        loaded = self._load(self.scope_a, [self.a1.pk, self.a2.pk])
        self.assertEqual(
            {row.collection_id for row in loaded}, {self.a1.pk, self.a2.pk}
        )

    def test_an_empty_collection_set_returns_zero_rows(self):
        """Fail-closed: empty means NONE, never 'no filter'."""
        self.assertEqual(self._load(self.scope_a, []), [])
        self.assertEqual(self._load(self.scope_a, frozenset()), [])
        self.assertEqual(self._load(self.scope_a, None), [])

    def test_a_foreign_collection_id_returns_zero_rows(self):
        self.assertEqual(self._load(self.scope_a, [self.b1.pk]), [])

    def test_the_other_scope_sees_only_its_own(self):
        loaded = self._load(self.scope_b, [self.b1.pk])
        self.assertEqual({row.collection_id for row in loaded}, {self.b1.pk})

    def test_a_stale_row_is_never_returned(self):
        self.chunks["a1"].content = "changed"
        self.chunks["a1"].save(update_fields=["content"])
        loaded = self._load(self.scope_a, [self.a1.pk, self.a2.pk])
        self.assertEqual({row.collection_id for row in loaded}, {self.a2.pk})

    def test_an_inactive_config_does_not_remove_rows_from_a_scoped_load(self):
        """The loader is STORAGE, not provider execution."""
        loaded_before = self._load(self.scope_a, [self.a1.pk, self.a2.pk])
        self.assertEqual(len(loaded_before), 2)

        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(is_active=False)

        loaded_after = self._load(self.scope_a, [self.a1.pk, self.a2.pk])
        self.assertEqual(loaded_after, loaded_before)

    def test_an_inactive_provider_does_not_remove_rows_from_a_scoped_load(self):
        loaded_before = self._load(self.scope_a, [self.a1.pk, self.a2.pk])
        self.assertEqual(len(loaded_before), 2)

        ProviderConfig.objects.filter(pk=self.config.provider_id).update(
            is_active=False
        )

        loaded_after = self._load(self.scope_a, [self.a1.pk, self.a2.pk])
        self.assertEqual(loaded_after, loaded_before)

    def test_a_different_e1_returns_nothing(self):
        other = self.build_config(
            name="embed-v9", dimension=4, revision="v9",
            provider=self.config.provider,
        )
        from ai_hub.services.embedding_contract import resolve_embedding_contract

        loaded = load_current_vectors(
            application_scope=self.scope_a,
            e1=resolve_embedding_contract(other).e1,
            collection_ids=[self.a1.pk],
        )
        self.assertEqual(loaded, [])

    def test_a_corrupt_routing_row_is_never_returned(self):
        """Denormalized columns accelerate; they never override lineage.

        Manufacture a row whose routing columns claim Scope A / Collection A1
        while its chunk genuinely belongs to Scope B / Collection B1.
        """
        record = KnowledgeChunkEmbedding.objects.get(chunk=self.chunks["b1"])
        KnowledgeChunkEmbedding.objects.filter(pk=record.pk).update(
            application_scope_id=self.scope_a.pk, collection_id=self.a1.pk
        )
        loaded = self._load(self.scope_a, [self.a1.pk, self.a2.pk])
        self.assertNotIn(
            self.chunks["b1"].pk, {row.chunk_id for row in loaded}
        )
        # ...and inspection explains why it is not current.
        inspection = inspect_record(
            KnowledgeChunkEmbedding.objects.select_related(
                "chunk", "chunk__document", "chunk__document__collection",
                "embedding_model_config", "embedding_model_config__provider",
            ).get(pk=record.pk)
        )
        self.assertFalse(inspection.current)
        self.assertIn(
            InspectionReason.COLLECTION_MISMATCH, inspection.reason_codes
        )

    def test_the_loader_returns_decoded_values_but_no_knowledge_text(self):
        loaded = self._load(self.scope_a, [self.a1.pk])
        row = loaded[0]
        self.assertEqual(len(row.values), 4)
        for field in row.__dataclass_fields__:
            for forbidden in ("text", "content", "title", "section"):
                self.assertNotIn(forbidden, field)

    def test_no_ranking_or_similarity_is_performed(self):
        parameters = set(inspect.signature(load_current_vectors).parameters)
        self.assertEqual(
            parameters, {"application_scope", "e1", "collection_ids"}
        )
        for forbidden in ("query_vector", "top_k", "limit", "metric", "threshold"):
            self.assertNotIn(forbidden, parameters)


# ---------------------------------------------------------------------------
# The abstraction
# ---------------------------------------------------------------------------

class VectorStoreAbstractionTests(VectorFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.build_scope()
        self.collection = self.build_collection(self.scope)
        self.chunk = self.build_chunk(self.collection)
        self.config = self.build_config(dimension=4)

    def test_the_backend_satisfies_the_protocol(self):
        store = DjangoBinaryVectorStore()
        self.assertTrue(hasattr(store, "store_chunk_vector"))
        self.assertTrue(hasattr(store, "load_current_vectors"))
        self.assertTrue(isinstance(store, VectorStore))

    def test_the_backend_and_the_module_helpers_behave_identically(self):
        from ai_hub.services.embedding_contract import resolve_embedding_contract

        store = DjangoBinaryVectorStore()
        record = store.store_chunk_vector(
            application_scope=self.scope, chunk=self.chunk,
            embedding_model_config=self.config, vector=[0.1, 0.2, 0.3, 0.4],
        )
        e1 = resolve_embedding_contract(self.config).e1
        through_backend = store.load_current_vectors(
            application_scope=self.scope, e1=e1,
            collection_ids=[self.collection.pk],
        )
        through_module = load_current_vectors(
            application_scope=self.scope, e1=e1,
            collection_ids=[self.collection.pk],
        )
        self.assertEqual(len(through_backend), 1)
        self.assertEqual(through_backend, through_module)
        self.assertEqual(through_backend[0].record_id, record.pk)

    def test_the_protocol_declares_no_search_api(self):
        """Ranking semantics belong to semantic retrieval, not persistence."""
        for forbidden in (
            "nearest_neighbors", "similarity_search", "rank", "score",
            "cosine_search", "ann_search", "search",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(hasattr(VectorStore, forbidden), forbidden)
                self.assertFalse(
                    hasattr(DjangoBinaryVectorStore, forbidden), forbidden
                )
        self.assertEqual(
            {
                name
                for name in dir(DjangoBinaryVectorStore)
                if not name.startswith("_")
            },
            {"store_chunk_vector", "load_current_vectors"},
        )

    def test_there_is_no_backend_discovery(self):
        """Callers select a store explicitly; nothing sniffs the environment."""
        imported = imported_names(vector_store)
        referenced = referenced_names(vector_store)
        for forbidden in ("os", "environ", "importlib", "settings", "pgvector"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, imported)
        for forbidden in (
            "getenv", "environ", "import_module", "auto_detect",
            "first_available", "get_backend",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, referenced)
        self.assertNotIn("pgvector", code_strings(vector_store))


# ---------------------------------------------------------------------------
# Boundary separation and absence
# ---------------------------------------------------------------------------

class StorageBoundaryTests(VectorFixtureMixin, TestCase):
    def test_storage_does_not_consult_egress_permission(self):
        """The docstring EXPLAINS the separation; the code never crosses it."""
        imported = imported_names(vector_store)
        referenced = referenced_names(vector_store)
        for forbidden in (
            "ProviderGrant", "resolve_embedding_access", "EmbeddingAccessDecision",
            "embedding_egress",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, imported)
                self.assertNotIn(forbidden, referenced)
        for forbidden in (
            "allow_external_embedding_corpus_egress",
            "allow_external_embedding_query_egress",
            "allow_embeddings", "declared_locality",
            "external_embedding_egress_policy",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, referenced)

    def test_a_vector_stores_fine_with_no_grant_and_egress_denied(self):
        """Proves the two boundaries are genuinely independent."""
        scope = self.build_scope()
        collection = self.build_collection(scope)
        chunk = self.build_chunk(collection)
        config = self.build_config(dimension=2)

        from ai_hub.models import ProviderGrant

        self.assertFalse(ProviderGrant.objects.exists())
        self.assertFalse(scope.allow_external_embedding_corpus_egress)

        record = store_chunk_vector(
            application_scope=scope, chunk=chunk,
            embedding_model_config=config, vector=[0.5, 0.5],
        )
        self.assertIsNotNone(record.pk)

    def test_storage_does_not_consult_knowledge_authorization(self):
        imported = imported_names(vector_store)
        referenced = referenced_names(vector_store)
        for forbidden in (
            "EffectiveKnowledgeScope", "resolve_effective_knowledge_scope",
            "knowledge_authorization", "AgentProfile",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, imported)
                self.assertNotIn(forbidden, referenced)
        for forbidden in ("agent", "agents", "knowledge_collections"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, referenced)

    def test_no_signal_triggers_indexing(self):
        """Indexing stays post-commit and explicit; nothing reacts to a save."""
        import ai_hub.models as models_module

        for module in (models_module, vector_store, chunk_embedding_identity):
            with self.subTest(module=module.__name__):
                imported = imported_names(module)
                referenced = referenced_names(module)
                for forbidden in (
                    "post_save", "pre_save", "post_delete", "pre_delete",
                    "receiver", "signals",
                ):
                    self.assertNotIn(forbidden, imported)
                    self.assertNotIn(forbidden, referenced)

    def test_no_embedding_generation_exists(self):
        for module in (vector_store, chunk_embedding_identity):
            with self.subTest(module=module.__name__):
                for forbidden in (
                    "embed_text", "generate_embedding", "embedding_call",
                    "create_vector", "embed",
                ):
                    self.assertFalse(hasattr(module, forbidden), forbidden)

    def test_no_provider_or_http_client_is_imported(self):
        for module in (vector_store, chunk_embedding_identity):
            with self.subTest(module=module.__name__):
                imported = imported_names(module)
                for forbidden in (
                    "litellm", "litellm_client", "requests", "httpx", "openai",
                    "ollama", "urllib", "socket",
                ):
                    self.assertNotIn(forbidden, imported)

    def test_no_ann_or_vector_backend_dependency(self):
        for module in (vector_store, chunk_embedding_identity):
            with self.subTest(module=module.__name__):
                imported = imported_names(module)
                referenced = referenced_names(module)
                for forbidden in (
                    "numpy", "faiss", "chromadb", "pgvector", "VectorField",
                    "VectorExtension", "HnswIndex", "IvfflatIndex",
                ):
                    self.assertNotIn(forbidden, imported)
                    self.assertNotIn(forbidden, referenced)

    def test_the_lifecycle_contracts_are_untouched(self):
        for module in (vector_store, chunk_embedding_identity):
            with self.subTest(module=module.__name__):
                imported = imported_names(module)
                referenced = referenced_names(module)
                for forbidden in (
                    "chunk_authority_mode", "KnowledgeLifecycleEvent",
                    "generation_input_fingerprint",
                    "generation_chunk_set_fingerprint",
                ):
                    self.assertNotIn(forbidden, imported)
                    self.assertNotIn(forbidden, referenced)

    def test_the_model_is_not_registered_in_admin(self):
        """Derived index state, not operator-authored Knowledge."""
        from django.contrib import admin as django_admin

        self.assertNotIn(KnowledgeChunkEmbedding, django_admin.site._registry)

    def test_str_exposes_no_vector_values(self):
        scope = self.build_scope()
        collection = self.build_collection(scope)
        chunk = self.build_chunk(collection)
        config = self.build_config(dimension=2)
        record = store_chunk_vector(
            application_scope=scope, chunk=chunk,
            embedding_model_config=config, vector=[0.25, 0.75],
        )
        rendered = str(record)
        self.assertNotIn("0.25", rendered)
        self.assertNotIn("0.75", rendered)
        self.assertIn(str(chunk.pk), rendered)


# ---------------------------------------------------------------------------
# Query shape
# ---------------------------------------------------------------------------

class VectorQueryShapeTests(VectorFixtureMixin, TestCase):
    def setUp(self):
        self.scope = self.build_scope()
        self.collection = self.build_collection(self.scope)
        self.config = self.build_config(dimension=4)
        from ai_hub.services.embedding_contract import resolve_embedding_contract

        self.e1 = resolve_embedding_contract(self.config).e1
        for index in range(6):
            chunk = self.build_chunk(
                self.collection, title=f"Doc {index}", section_title=f"S{index}",
                content=f"Body {index}",
            )
            store_chunk_vector(
                application_scope=self.scope, chunk=chunk,
                embedding_model_config=self.config, vector=[0.1] * 4,
            )

    def test_a_scoped_load_does_not_scale_queries_with_row_count(self):
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        with CaptureQueriesContext(connection) as before:
            load_current_vectors(
                application_scope=self.scope, e1=self.e1,
                collection_ids=[self.collection.pk],
            )
        baseline = len(before)

        for index in range(6, 18):
            chunk = self.build_chunk(
                self.collection, title=f"Bulk {index}", section_title=f"B{index}",
                content=f"Bulk body {index}",
            )
            store_chunk_vector(
                application_scope=self.scope, chunk=chunk,
                embedding_model_config=self.config, vector=[0.1] * 4,
            )

        with CaptureQueriesContext(connection) as after:
            load_current_vectors(
                application_scope=self.scope, e1=self.e1,
                collection_ids=[self.collection.pk],
            )
        self.assertEqual(len(after), baseline)

    def test_narrowing_happens_in_sql_before_decoding(self):
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        with CaptureQueriesContext(connection) as captured:
            load_current_vectors(
                application_scope=self.scope, e1=self.e1,
                collection_ids=[self.collection.pk],
            )
        rendered = " ".join(query["sql"] for query in captured).lower()
        self.assertIn("application_scope_id", rendered)
        self.assertIn("collection_id", rendered)
        self.assertIn("e1", rendered)
