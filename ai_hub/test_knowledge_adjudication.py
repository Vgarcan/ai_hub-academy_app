"""Tests for operator adjudication of UNKNOWN Knowledge authority (Slice 10).

RC-002 step 3d. The first real lifecycle operations, and the first consumers of
the Slice 9 governed mutation boundary.

The two decisions are tested asymmetrically because they ARE asymmetric:
EXPLICIT is a declaration Core cannot verify, DERIVED is a reproducibility claim
Core can and must verify — forward only, never as a guess about history.
"""
import dataclasses
from unittest import mock

from django.db import connection, models
from django.test import TestCase

from ai_hub.models import (
    AgentProfile,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    KnowledgeLifecycleEvent,
    ToolDefinition,
)
from ai_hub.services.knowledge_adjudication import (
    ADOPTION_GENERATOR,
    OPERATION_ADJUDICATE_UNKNOWN_AS_EXPLICIT,
    OPERATION_ADOPT_UNKNOWN_AS_DERIVED,
    AuthorityNotUnknownError,
    ChunkSetNotReproducibleError,
    EmptyGeneratorOutputError,
    KnowledgeAdjudicationError,
    MissingChunkSetError,
    UnexpectedProvenanceError,
    adjudicate_unknown_as_explicit,
    adopt_unknown_as_derived,
)
from ai_hub.services.knowledge_ingestion import ensure_initial_knowledge_chunk
from ai_hub.services.knowledge_lifecycle import (
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK,
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION,
    UnsupportedGeneratorError,
    chunk_set_fingerprint,
    curated_text_single_chunk_projection,
    document_chunk_set_fingerprint,
    generator_output_projection,
)
from ai_hub.services.knowledge_mutation import (
    ExpectedKnowledgeState,
    KnowledgeMutationConflict,
    KnowledgeMutationPrincipal,
    build_snapshot,
)
from ai_hub.services.knowledge_preflight import run_knowledge_preflight
from ai_hub.services.starter_toolboxes import seed_starter_toolboxes

MODES = KnowledgeDocument.ChunkAuthorityMode


def a_principal():
    return KnowledgeMutationPrincipal.human("operator-1")


def expected_for(document):
    return ExpectedKnowledgeState.from_snapshot(build_snapshot(document))


def make_document(collection, title="Doc", *, curated_text="body", chunks=(), **kwargs):
    document = KnowledgeDocument.objects.create(
        collection=collection,
        title=title,
        curated_text=curated_text,
        status=kwargs.pop("status", KnowledgeDocument.Status.ACTIVE),
        **kwargs,
    )
    for index, content in chunks:
        KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=index,
            section_title=f"Section {index}", content=content,
        )
    return document


def make_reproducible_document(collection, title="Reproducible", *, curated_text="derived body"):
    """A document whose chunk set the real generator actually produced."""
    document = make_document(collection, title, curated_text=curated_text)
    ensure_initial_knowledge_chunk(document)
    return document


def chunk_rows(document):
    return list(
        document.chunks.order_by("chunk_index").values(
            "pk", "chunk_index", "section_title", "content", "token_estimate",
            "metadata", "created_at", "updated_at",
        )
    )


# ---------------------------------------------------------------------------
# The projection must model the REAL generator, or every adoption is wrong
# ---------------------------------------------------------------------------

class ProjectionMatchesRealGeneratorTests(TestCase):
    """`curated_text_single_chunk_projection` is a MODEL of
    `ensure_initial_knowledge_chunk`. The real writer is deliberately untouched,
    so nothing but these tests keeps the two from drifting apart — and a drift
    would silently make adoption decide the wrong thing.
    """

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Projection")

    def _assert_projection_matches_writer(self, title, curated_text):
        document = make_document(self.collection, title, curated_text=curated_text)
        projected = curated_text_single_chunk_projection(document)

        created = ensure_initial_knowledge_chunk(document)
        actual = list(
            document.chunks.order_by("chunk_index").values(
                "chunk_index", "section_title", "content"
            )
        )
        if created is None:
            self.assertEqual(projected, [])
        self.assertEqual(projected, actual)
        self.assertEqual(chunk_set_fingerprint(projected), document_chunk_set_fingerprint(document))

    def test_projection_equals_what_the_writer_creates(self):
        self._assert_projection_matches_writer("A Title", "some curated body")

    def test_projection_matches_for_padded_curated_text(self):
        self._assert_projection_matches_writer("Padded", "\n\n  padded body  \t\n")

    def test_projection_matches_for_multiline_curated_text(self):
        self._assert_projection_matches_writer("Multi", "line one\nline two\n\nline four")

    def test_projection_matches_for_unicode_title_and_body(self):
        self._assert_projection_matches_writer("Título ünicode", "cuerpo — con guiones")

    def test_projection_is_empty_exactly_when_the_writer_creates_nothing(self):
        for curated_text in ("", "   ", "\n\t \n"):
            with self.subTest(curated_text=repr(curated_text)):
                self._assert_projection_matches_writer("Blank", curated_text)

    def test_projection_uses_the_title_verbatim_as_section_title(self):
        """The title becomes `section_title` unstripped — same as the writer."""
        document = make_document(self.collection, "  Spaced Title  ", curated_text="body")
        self.assertEqual(
            curated_text_single_chunk_projection(document)[0]["section_title"],
            "  Spaced Title  ",
        )
        ensure_initial_knowledge_chunk(document)
        self.assertEqual(document.chunks.get().section_title, "  Spaced Title  ")

    def test_projection_writes_nothing(self):
        document = make_document(self.collection, "Pure", curated_text="body")
        curated_text_single_chunk_projection(document)
        self.assertEqual(document.chunks.count(), 0)

    def test_projection_omits_fields_the_chunk_set_contract_excludes(self):
        """`c1` covers index/section/content only, so projecting `token_estimate`
        or `metadata` would impose a stricter identity than the contract."""
        document = make_document(self.collection, "Fields", curated_text="body")
        self.assertEqual(
            set(curated_text_single_chunk_projection(document)[0]),
            {"chunk_index", "section_title", "content"},
        )

    def test_an_unknown_generator_identity_has_no_projection(self):
        document = make_document(self.collection, "Unknown Gen", curated_text="body")
        with self.assertRaises(UnsupportedGeneratorError):
            generator_output_projection("some_future_generator", document)

    def test_the_supported_identity_dispatches(self):
        document = make_document(self.collection, "Dispatch", curated_text="body")
        self.assertEqual(
            generator_output_projection(GENERATOR_CURATED_TEXT_SINGLE_CHUNK, document),
            curated_text_single_chunk_projection(document),
        )


# ---------------------------------------------------------------------------
# UNKNOWN -> EXPLICIT : a declaration
# ---------------------------------------------------------------------------

class AdjudicateExplicitTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Explicit")
        self.document = make_document(
            self.collection, "Hand Authored", curated_text="source",
            chunks=((1, "hand written chunk"),),
        )

    def test_successful_declaration_sets_only_the_authority_mode(self):
        before_chunks = chunk_rows(self.document)
        result = adjudicate_unknown_as_explicit(
            self.document.pk, expected=expected_for(self.document),
            principal=a_principal(), reason_code="hand_authored",
        )

        self.assertEqual(result.chunk_authority_mode, MODES.EXPLICIT)
        # No generation provenance is invented — EXPLICIT has none by definition.
        self.assertEqual(result.generation_input_fingerprint, "")
        self.assertEqual(result.generation_chunk_set_fingerprint, "")
        self.assertEqual(result.generator_identity, "")
        self.assertIsNone(result.generator_version)
        self.assertEqual(chunk_rows(self.document), before_chunks)

    def test_declaration_records_one_lifecycle_event(self):
        adjudicate_unknown_as_explicit(
            self.document.pk, expected=expected_for(self.document),
            principal=KnowledgeMutationPrincipal.human("alice-id"),
            reason_code="hand_authored",
        )
        event = KnowledgeLifecycleEvent.objects.get()
        self.assertEqual(event.operation, OPERATION_ADJUDICATE_UNKNOWN_AS_EXPLICIT)
        self.assertEqual(event.reason_code, "hand_authored")
        self.assertEqual(event.previous_authority_mode, MODES.UNKNOWN)
        self.assertEqual(event.new_authority_mode, MODES.EXPLICIT)
        self.assertEqual(event.principal_kind, "human")
        self.assertEqual(event.principal_identifier, "alice-id")
        self.assertEqual(event.document_id, self.document.pk)
        self.assertEqual(event.previous_chunk_count, event.new_chunk_count)

    def test_curated_text_differing_from_chunks_is_irrelevant(self):
        """D-L5: EXPLICIT chunks differing from `curated_text` is not staleness."""
        document = make_document(
            self.collection, "Diverged", curated_text="a completely different source",
            chunks=((1, "the authored chunk"),),
        )
        result = adjudicate_unknown_as_explicit(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )
        self.assertEqual(result.chunk_authority_mode, MODES.EXPLICIT)

    def test_a_document_with_no_chunks_is_refused(self):
        document = make_document(self.collection, "No Artifact", curated_text="source")
        with self.assertRaises(MissingChunkSetError):
            adjudicate_unknown_as_explicit(
                document.pk, expected=expected_for(document), principal=a_principal(),
            )
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_unusable_chunk_text_does_not_block_the_authority_decision(self):
        """Authority and structural readiness are orthogonal axes (Slice 8)."""
        document = make_document(
            self.collection, "Blank Chunk", curated_text="source", chunks=((1, "   "),),
        )
        result = adjudicate_unknown_as_explicit(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )
        self.assertEqual(result.chunk_authority_mode, MODES.EXPLICIT)
        row = _preflight_row(document.pk)
        self.assertEqual(row["lifecycle_state"], "EXPLICIT_AUTHORITY")
        self.assertEqual(row["structural_state"], "UNUSABLE_CHUNKS")
        self.assertFalse(row["canonically_retrievable"])

    def test_a_non_active_document_can_still_be_adjudicated(self):
        document = make_document(
            self.collection, "Archived", curated_text="s", chunks=((1, "c"),),
            status=KnowledgeDocument.Status.ARCHIVED,
        )
        result = adjudicate_unknown_as_explicit(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )
        self.assertEqual(result.chunk_authority_mode, MODES.EXPLICIT)


# ---------------------------------------------------------------------------
# UNKNOWN -> DERIVED : forward adoption only
# ---------------------------------------------------------------------------

class AdoptDerivedTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Adopt")
        self.document = make_reproducible_document(self.collection)

    def test_reproducible_chunk_set_is_adopted(self):
        result = adopt_unknown_as_derived(
            self.document.pk, expected=expected_for(self.document),
            principal=a_principal(), reason_code="reproduces_exactly",
        )
        self.assertEqual(result.chunk_authority_mode, MODES.DERIVED)
        self.assertTrue(result.generation_input_fingerprint.startswith("i1:"))
        self.assertTrue(result.generation_chunk_set_fingerprint.startswith("c1:"))
        self.assertEqual(result.generator_identity, ADOPTION_GENERATOR)
        self.assertEqual(result.generator_version, GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION)

    def test_adoption_does_not_touch_a_single_chunk(self):
        """The whole point: adoption records, it never regenerates."""
        before = chunk_rows(self.document)
        adopt_unknown_as_derived(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        self.assertEqual(chunk_rows(self.document), before)

    def test_adoption_does_not_touch_the_source_either(self):
        before = KnowledgeDocument.objects.filter(pk=self.document.pk).values(
            "title", "curated_text", "status", "notes", "tags"
        )[0]
        adopt_unknown_as_derived(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        after = KnowledgeDocument.objects.filter(pk=self.document.pk).values(
            "title", "curated_text", "status", "notes", "tags"
        )[0]
        self.assertEqual(before, after)

    def test_the_recorded_fingerprints_describe_the_state_at_adoption(self):
        adopt_unknown_as_derived(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        self.document.refresh_from_db()
        snapshot = build_snapshot(self.document)
        self.assertEqual(
            self.document.generation_chunk_set_fingerprint,
            snapshot.observed_chunk_set_fingerprint,
        )
        self.assertEqual(
            self.document.generation_input_fingerprint, snapshot.observed_input_fingerprint
        )

    def test_an_adopted_document_is_immediately_derived_current_in_preflight(self):
        """Adopted under `c1`, judged under `c1` — one notion of sameness."""
        adopt_unknown_as_derived(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        row = _preflight_row(self.document.pk)
        self.assertEqual(row["authority_mode"], MODES.DERIVED)
        self.assertEqual(row["lifecycle_state"], "DERIVED_CURRENT")
        self.assertEqual(row["issues"], [])

    def test_adoption_records_one_lifecycle_event_marking_the_moment(self):
        adopt_unknown_as_derived(
            self.document.pk, expected=expected_for(self.document),
            principal=KnowledgeMutationPrincipal.system("adoption-job"),
        )
        event = KnowledgeLifecycleEvent.objects.get()
        self.assertEqual(event.operation, OPERATION_ADOPT_UNKNOWN_AS_DERIVED)
        self.assertEqual(event.previous_authority_mode, MODES.UNKNOWN)
        self.assertEqual(event.new_authority_mode, MODES.DERIVED)
        self.assertEqual(event.principal_kind, "system")
        self.assertIsNotNone(event.created_at)

    def test_the_event_does_not_fabricate_earlier_provenance(self):
        """It marks when DERIVED became KNOWN, not when chunks were generated."""
        adopt_unknown_as_derived(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        event = KnowledgeLifecycleEvent.objects.get()
        # Before the decision Core knew nothing, and the event says exactly that.
        self.assertEqual(event.previous_generation_input_fingerprint, "")
        self.assertEqual(event.previous_generation_chunk_set_fingerprint, "")
        self.assertEqual(event.previous_generator_identity, "")
        self.assertIsNone(event.previous_generator_version)
        # And the chunk set it adopted is recorded as already present beforehand.
        self.assertEqual(
            event.previous_observed_chunk_set_fingerprint,
            event.new_observed_chunk_set_fingerprint,
        )

    # -- rejections ---------------------------------------------------------

    def _assert_rejected(self, document, exception=ChunkSetNotReproducibleError):
        before_chunks = chunk_rows(document)
        with self.assertRaises(exception) as caught:
            adopt_unknown_as_derived(
                document.pk, expected=expected_for(document), principal=a_principal(),
            )
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(document.generator_identity, "")
        self.assertEqual(chunk_rows(document), before_chunks)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)
        return caught.exception

    def test_hand_edited_chunk_content_is_rejected_not_repaired(self):
        document = make_reproducible_document(self.collection, "Edited")
        chunk = document.chunks.get()
        chunk.content = "a human improved this"
        chunk.save(update_fields=["content"])

        error = self._assert_rejected(document)
        self.assertIn("were NOT modified", str(error))
        self.assertNotEqual(error.expected_fingerprint, error.actual_fingerprint)

    def test_a_renamed_document_is_rejected(self):
        """The title becomes `section_title`, so a rename breaks reproduction."""
        document = make_reproducible_document(self.collection, "Original Name")
        document.title = "Renamed Later"
        document.save(update_fields=["title"])
        self._assert_rejected(document)

    def test_a_changed_source_is_rejected(self):
        document = make_reproducible_document(self.collection, "Changed Source")
        document.curated_text = "the source moved on without the chunks"
        document.save(update_fields=["curated_text"])
        self._assert_rejected(document)

    def test_a_multi_chunk_document_is_rejected(self):
        """The supported generator emits exactly one chunk."""
        document = make_reproducible_document(self.collection, "Multi")
        KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=2, section_title="Extra", content="second",
        )
        self._assert_rejected(document)

    def test_a_differently_segmented_chunk_set_is_rejected(self):
        document = make_document(
            self.collection, "Segmented", curated_text="part one part two",
            chunks=((1, "part one"), (2, "part two")),
        )
        self._assert_rejected(document)

    def test_a_document_with_no_chunks_is_rejected(self):
        document = make_document(self.collection, "Chunkless", curated_text="a source")
        self._assert_rejected(document)

    def test_an_empty_document_is_rejected_rather_than_trivially_matching(self):
        """Empty projection vs empty chunk set would otherwise 'match'."""
        document = make_document(self.collection, "Empty", curated_text="")
        error = self._assert_rejected(document, EmptyGeneratorOutputError)
        self.assertIn("no chunks at all", str(error))

    def test_a_whitespace_only_source_is_rejected(self):
        document = make_document(self.collection, "Blank Source", curated_text="   \n\t ")
        self._assert_rejected(document, EmptyGeneratorOutputError)

    def test_rejection_carries_both_fingerprints_for_the_operator(self):
        document = make_reproducible_document(self.collection, "Diagnostics")
        document.curated_text = "changed"
        document.save(update_fields=["curated_text"])
        error = self._assert_rejected(document)
        self.assertEqual(error.document_id, document.pk)
        self.assertTrue(error.expected_fingerprint.startswith("c1:"))
        self.assertTrue(error.actual_fingerprint.startswith("c1:"))

    # -- deliberate tolerances ---------------------------------------------

    def test_outer_whitespace_in_curated_text_still_adopts(self):
        """The generator strips, so `i1`/`c1` do too. Not a difference."""
        document = make_reproducible_document(self.collection, "Padded", curated_text="body")
        document.curated_text = "\n\n  body  \t\n"
        document.save(update_fields=["curated_text"])
        result = adopt_unknown_as_derived(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )
        self.assertEqual(result.chunk_authority_mode, MODES.DERIVED)

    def test_metadata_and_token_estimate_differences_still_adopt(self):
        """`c1` deliberately excludes both, and adoption uses `c1` so that an
        adopted document is judged later under the same contract."""
        document = make_reproducible_document(self.collection, "Metadata")
        chunk = document.chunks.get()
        chunk.metadata = {"ingestion": "something_else"}
        chunk.token_estimate = 9999
        chunk.save(update_fields=["metadata", "token_estimate"])
        result = adopt_unknown_as_derived(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )
        self.assertEqual(result.chunk_authority_mode, MODES.DERIVED)

    def test_a_chunk_set_with_no_ingestion_marker_can_still_adopt(self):
        """Slice 4: marker absence is uninformative. Reproduction is the test."""
        document = make_document(self.collection, "No Marker", curated_text="marker free")
        KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1,
            section_title="No Marker", content="marker free", metadata={},
        )
        result = adopt_unknown_as_derived(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )
        self.assertEqual(result.chunk_authority_mode, MODES.DERIVED)

    def test_an_ingestion_marker_alone_does_not_authorize_adoption(self):
        """...and marker presence is equally uninformative."""
        document = make_document(self.collection, "Marked", curated_text="source text")
        KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1, section_title="Marked",
            content="unrelated content", metadata={"ingestion": "initial_curated_text"},
        )
        self._assert_rejected(document)


# ---------------------------------------------------------------------------
# Shared preconditions
# ---------------------------------------------------------------------------

class AdjudicationPreconditionTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Preconditions")

    def _both_operations(self, document):
        return (
            (adjudicate_unknown_as_explicit, expected_for(document)),
            (adopt_unknown_as_derived, expected_for(document)),
        )

    def test_a_non_unknown_document_is_refused_by_both_operations(self):
        for mode in (MODES.EXPLICIT, MODES.DERIVED):
            with self.subTest(mode=mode):
                document = make_reproducible_document(self.collection, f"Already {mode}")
                KnowledgeDocument.objects.filter(pk=document.pk).update(
                    chunk_authority_mode=mode
                )
                document.refresh_from_db()
                for operation, expected in self._both_operations(document):
                    with self.subTest(operation=operation.__name__):
                        with self.assertRaises(AuthorityNotUnknownError):
                            operation(document.pk, expected=expected, principal=a_principal())
                self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_an_unknown_document_carrying_provenance_is_refused_by_both(self):
        """UNKNOWN + provenance is a KP010 inconsistency, not a legacy row."""
        for field, value in (
            ("generation_input_fingerprint", "i1:" + "0" * 64),
            ("generation_chunk_set_fingerprint", "c1:" + "0" * 64),
            ("generator_identity", "curated_text_single_chunk"),
            ("generator_version", 1),
        ):
            with self.subTest(field=field):
                document = make_reproducible_document(self.collection, f"Provenance {field}")
                KnowledgeDocument.objects.filter(pk=document.pk).update(**{field: value})
                document.refresh_from_db()
                for operation, expected in self._both_operations(document):
                    with self.subTest(operation=operation.__name__):
                        with self.assertRaises(UnexpectedProvenanceError) as caught:
                            operation(document.pk, expected=expected, principal=a_principal())
                        self.assertIn(field, str(caught.exception))
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_every_refusal_shares_one_base_class(self):
        for error in (
            AuthorityNotUnknownError, MissingChunkSetError,
            UnexpectedProvenanceError, ChunkSetNotReproducibleError,
            EmptyGeneratorOutputError,
        ):
            with self.subTest(error=error.__name__):
                self.assertTrue(issubclass(error, KnowledgeAdjudicationError))


# ---------------------------------------------------------------------------
# The Slice 9 boundary is reused, not reimplemented
# ---------------------------------------------------------------------------

class AdjudicationUsesTheGovernedBoundaryTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Boundary")
        self.document = make_reproducible_document(self.collection)

    def test_stale_review_is_rejected_by_the_shared_compare_and_swap(self):
        expected = expected_for(self.document)
        chunk = self.document.chunks.get()
        chunk.content = "changed after the operator reviewed it"
        chunk.save(update_fields=["content"])

        for operation in (adjudicate_unknown_as_explicit, adopt_unknown_as_derived):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(KnowledgeMutationConflict):
                    operation(self.document.pk, expected=expected, principal=a_principal())
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_a_collection_move_between_review_and_commit_is_rejected(self):
        """Neither `i1` nor `c1` moves, but the authorization boundary did."""
        expected = expected_for(self.document)
        other = KnowledgeCollection.objects.create(name="Elsewhere")
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(collection=other)

        with self.assertRaises(KnowledgeMutationConflict) as caught:
            adopt_unknown_as_derived(
                self.document.pk, expected=expected, principal=a_principal()
            )
        self.assertEqual(caught.exception.field, "collection_id")

    def test_adjudication_does_not_reimplement_transaction_or_audit_logic(self):
        """One governed-write path, not two.

        Checked against the AST rather than the raw text, so prose in a
        docstring cannot pass or fail it — only real code references count.
        """
        import ast
        import pathlib

        tree = ast.parse(
            (
                pathlib.Path(__file__).resolve().parent
                / "services" / "knowledge_adjudication.py"
            ).read_text(encoding="utf-8")
        )
        referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        referenced |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }

        for reimplementation in (
            "atomic", "select_for_update", "KnowledgeLifecycleEvent",
            "verify_expected_state", "build_snapshot",
        ):
            with self.subTest(symbol=reimplementation):
                self.assertNotIn(reimplementation, referenced)
                self.assertNotIn(reimplementation, imported)

        self.assertIn("_governed_knowledge_mutation", imported)

    def test_a_principal_is_required(self):
        from ai_hub.services.knowledge_mutation import KnowledgeMutationPrincipalError

        for operation in (adjudicate_unknown_as_explicit, adopt_unknown_as_derived):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(KnowledgeMutationPrincipalError):
                    operation(
                        self.document.pk, expected=expected_for(self.document),
                        principal=None,
                    )

    def test_the_operation_slugs_satisfy_the_foundation_contract(self):
        import re

        for slug in (
            OPERATION_ADJUDICATE_UNKNOWN_AS_EXPLICIT, OPERATION_ADOPT_UNKNOWN_AS_DERIVED,
        ):
            with self.subTest(slug=slug):
                self.assertRegex(slug, r"^[a-z][a-z0-9_]{2,63}$")

    def test_an_event_failure_rolls_the_adjudication_back(self):
        with mock.patch(
            "ai_hub.services.knowledge_mutation.KnowledgeLifecycleEvent.objects.create",
            side_effect=RuntimeError("audit write failed"),
        ):
            with self.assertRaises(RuntimeError):
                adopt_unknown_as_derived(
                    self.document.pk, expected=expected_for(self.document),
                    principal=a_principal(),
                )
        self.document.refresh_from_db()
        self.assertEqual(self.document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)


# ---------------------------------------------------------------------------
# Security boundary
# ---------------------------------------------------------------------------

class AdjudicationSecurityBoundaryTests(TestCase):
    """The scanned module lists below are asserted to EXIST, never skipped.

    A silent `continue` on a missing path would let a rename or deletion quietly
    shrink the security guard to nothing while the suite stayed green - the test
    would keep passing precisely because it had stopped checking anything.
    """

    # Modules a model can reach, directly or transitively. None of them may
    # mention adjudication. Extend only with repository evidence.
    AGENT_FACING_MODULES = (
        "services/knowledge_retrieval.py",
        "services/game_action_dispatcher.py",
        "services/agent_runtime.py",
        "services/tools_runtime.py",
        "services/tool_resolution.py",
        "services/knowledge_ingestion.py",
        "services/knowledge_preflight.py",
    )

    # Operator-facing write surfaces deliberately NOT wired up in this slice.
    DEFERRED_SURFACE_MODULES = (
        "admin.py",
        "services/build_console.py",
    )

    ADJUDICATION_SYMBOLS = (
        "knowledge_adjudication",
        "adopt_unknown_as_derived",
        "adjudicate_unknown_as_explicit",
    )

    def _module_path(self, relative):
        import pathlib

        path = pathlib.Path(__file__).resolve().parent / relative
        self.assertTrue(
            path.exists(),
            f"{relative} does not exist, so this security test would silently "
            "stop checking it. Update the module list deliberately rather than "
            "letting the guard lapse.",
        )
        return path

    def test_every_scanned_module_exists(self):
        """Guards the guard: the lists must not rot into no-ops."""
        for relative in self.AGENT_FACING_MODULES + self.DEFERRED_SURFACE_MODULES:
            with self.subTest(module=relative):
                self._module_path(relative)

    def test_no_seeded_tool_definition_exposes_adjudication(self):
        seed_starter_toolboxes()
        self.assertGreater(ToolDefinition.objects.count(), 0)
        for tool in ToolDefinition.objects.all():
            with self.subTest(tool=tool.name):
                blob = " ".join(
                    str(part) for part in (tool.name, tool.label, tool.description, tool.config)
                ).lower()
                for forbidden in ("adjudicat", "adopt_unknown", "knowledge_adjudication", "authority"):
                    self.assertNotIn(forbidden, blob)

    def test_no_agent_facing_module_imports_adjudication(self):
        for relative in self.AGENT_FACING_MODULES:
            with self.subTest(module=relative):
                source = self._module_path(relative).read_text(encoding="utf-8")
                for symbol in self.ADJUDICATION_SYMBOLS:
                    self.assertNotIn(symbol, source)

    def test_no_admin_or_build_console_integration_exists_yet(self):
        """Deliberately deferred: an adjudication surface is its own design."""
        for relative in self.DEFERRED_SURFACE_MODULES:
            with self.subTest(module=relative):
                source = self._module_path(relative).read_text(encoding="utf-8")
                for symbol in self.ADJUDICATION_SYMBOLS:
                    self.assertNotIn(symbol, source)

    def test_the_principal_cannot_be_an_agent(self):
        principal = a_principal()
        self.assertNotIsInstance(principal, AgentProfile)
        self.assertEqual({f.name for f in dataclasses.fields(principal)}, {"kind", "identifier"})


# ---------------------------------------------------------------------------
# Nothing else moved
# ---------------------------------------------------------------------------

class AdjudicationLeavesEverythingElseAloneTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Unchanged")
        self.document = make_reproducible_document(self.collection)

    def test_no_new_migration_is_required(self):
        from django.db.migrations.autodetector import MigrationAutodetector
        from django.db.migrations.loader import MigrationLoader
        from django.db.migrations.questioner import NonInteractiveMigrationQuestioner
        from django.db.migrations.state import ProjectState

        loader = MigrationLoader(None, ignore_no_migrations=True)
        autodetector = MigrationAutodetector(
            loader.project_state(), ProjectState.from_apps(__import__("django").apps.apps),
            NonInteractiveMigrationQuestioner(specified_apps=set(), dry_run=True),
        )
        self.assertEqual(autodetector.changes(graph=loader.graph).get("ai_hub", []), [])

    def test_the_compatibility_writer_still_produces_unknown(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Fresh", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        ensure_initial_knowledge_chunk(document)
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(document.generator_identity, "")
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_preflight_stays_read_only_around_an_adjudication(self):
        adopt_unknown_as_derived(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        first = run_knowledge_preflight()
        events_before = KnowledgeLifecycleEvent.objects.count()
        second = run_knowledge_preflight()
        self.assertEqual(first, second)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), events_before)

    def test_adjudication_does_not_change_retrievability(self):
        before = _preflight_row(self.document.pk)["canonically_retrievable"]
        adopt_unknown_as_derived(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        self.assertEqual(_preflight_row(self.document.pk)["canonically_retrievable"], before)

    def test_no_repair_regeneration_or_backfill_verb_exists(self):
        from ai_hub.services import knowledge_adjudication

        for forbidden in ("repair", "regenerate", "backfill", "rechunk", "migrate_all", "bulk"):
            with self.subTest(name=forbidden):
                self.assertFalse(
                    any(name.startswith(forbidden) for name in dir(knowledge_adjudication)),
                    f"{forbidden!r} would be step 3e",
                )

    def test_the_public_surface_is_exactly_two_operations(self):
        import inspect

        from ai_hub.services import knowledge_adjudication

        functions = {
            name for name, obj in vars(knowledge_adjudication).items()
            if not name.startswith("_")
            and inspect.isfunction(obj)
            and obj.__module__ == knowledge_adjudication.__name__
        }
        self.assertEqual(
            functions, {"adjudicate_unknown_as_explicit", "adopt_unknown_as_derived"}
        )


def _preflight_row(document_id):
    for row in run_knowledge_preflight()["documents"]:
        if row["document_id"] == document_id:
            return row
    raise AssertionError(f"document {document_id} missing from preflight report")
