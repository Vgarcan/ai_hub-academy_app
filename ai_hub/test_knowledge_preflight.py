"""Tests for the lifecycle-aware read-only Knowledge preflight (V2, Slice 8).

Covers the three orthogonal axes (structural state, authority mode, lifecycle
state), canonical retrievability, the twelve issue codes, lifecycle consistency
detection, generator support, query behaviour, the operator security boundary,
and — most importantly — the READ-ONLY invariant.
"""
import json
from io import StringIO

from django.core.management import call_command
from django.test import TestCase

from ai_hub.models import (
    AgentProfile,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    ProviderConfig,
    ToolDefinition,
)
from ai_hub.services.knowledge_lifecycle import (
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK,
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION,
    chunk_set_fingerprint,
    curated_text_single_chunk_input_fingerprint,
)
from ai_hub.services.knowledge_preflight import (
    DERIVED_CHUNKS_MODIFIED,
    DERIVED_GENERATOR_UNSUPPORTED,
    DERIVED_GENERATOR_VERSION_AHEAD,
    DERIVED_INPUT_CHANGED,
    DERIVED_PROVENANCE_INCOMPLETE,
    EMPTY_ACTIVE,
    GENERATOR_OUTDATED,
    INCONSISTENCY_DERIVED_WITHOUT_CHUNK_SET_FINGERPRINT,
    INCONSISTENCY_DERIVED_WITHOUT_GENERATOR_IDENTITY,
    INCONSISTENCY_DERIVED_WITHOUT_GENERATOR_VERSION,
    INCONSISTENCY_DERIVED_WITHOUT_INPUT_FINGERPRINT,
    INCONSISTENCY_NON_DERIVED_WITH_PROVENANCE,
    ISSUE_CODES,
    KP001_SOURCE_WITHOUT_CHUNKS,
    KP002_EMPTY_ACTIVE,
    KP003_UNUSABLE_CHUNK_CONTENT,
    KP004_UNKNOWN_AUTHORITY,
    KP005_SOURCE_FILE_WITHOUT_CHUNKS,
    KP006_CHUNK_INDEX_ANOMALY,
    KP007_DERIVED_CHUNKS_MODIFIED,
    KP008_GENERATOR_OUTDATED,
    KP009_DERIVED_INPUT_CHANGED,
    KP010_LIFECYCLE_FACT_INCONSISTENCY,
    KP011_GENERATOR_UNSUPPORTED,
    KP012_GENERATOR_VERSION_AHEAD,
    LIFECYCLE_STATES,
    NON_ACTIVE,
    PREFLIGHT_CONTRACT_VERSION,
    READY_CANONICAL,
    DERIVED_CURRENT,
    EXPLICIT_AUTHORITY,
    SEVERITY_INFO,
    SOURCE_WITHOUT_CHUNKS,
    STRUCTURAL_STATES,
    UNKNOWN_AUTHORITY,
    UNUSABLE_CHUNKS,
    VERSION_AHEAD,
    VERSION_CURRENT,
    VERSION_OUTDATED,
    run_knowledge_preflight,
    summarize_preflight,
)
from ai_hub.test_application_scope_helpers import test_scope

MODES = KnowledgeDocument.ChunkAuthorityMode


def make_document(collection, title, *, curated_text="", status=None, chunks=(), source_file="", **lifecycle):
    document = KnowledgeDocument.objects.create(
        collection=collection,
        title=title,
        curated_text=curated_text,
        source_file=source_file,
        status=status or KnowledgeDocument.Status.ACTIVE,
        **lifecycle,
    )
    for index, content in chunks:
        KnowledgeDocumentChunk.objects.create(
            document=document,
            chunk_index=index,
            section_title=f"Section {index}",
            content=content,
        )
    return document


def make_derived_document(collection, title, *, curated_text, chunks, version=None):
    """A coherent DERIVED document whose recorded fingerprints actually match."""
    document = make_document(collection, title, curated_text=curated_text, chunks=chunks)
    record_generation(document, version=version)
    return document


def record_generation(document, *, version=None):
    """Stamp a document with fingerprints matching its CURRENT state."""
    document.chunk_authority_mode = MODES.DERIVED
    document.generation_input_fingerprint = curated_text_single_chunk_input_fingerprint(
        title=document.title, curated_text=document.curated_text
    )
    document.generation_chunk_set_fingerprint = chunk_set_fingerprint(
        document.chunks.order_by("chunk_index")
    )
    document.generator_identity = GENERATOR_CURATED_TEXT_SINGLE_CHUNK
    document.generator_version = (
        GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION if version is None else version
    )
    document.save()
    return document


def document_row(report, document_id):
    for row in report["documents"]:
        if row["document_id"] == document_id:
            return row
    raise AssertionError(f"document {document_id} missing from report")


class PreflightStructuralStateTests(TestCase):
    """Axis 1 — retrieval evidence only, carrying no authority claim."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Preflight Structural")
        cls.ready = make_document(cls.collection, "Ready", chunks=((1, "canonical body"),))
        cls.with_source = make_document(
            cls.collection, "Ready With Source",
            curated_text="source body", chunks=((1, "chunk body"),),
        )
        cls.source_only = make_document(
            cls.collection, "Source Without Chunks", curated_text="orphan source",
        )
        cls.empty_active = make_document(cls.collection, "Empty Active")
        cls.unusable = make_document(
            cls.collection, "Unusable Chunks", chunks=((1, ""), (2, "   \n\t  ")),
        )
        cls.draft = make_document(
            cls.collection, "Draft Source", curated_text="draft body",
            status=KnowledgeDocument.Status.DRAFT,
        )
        cls.archived = make_document(
            cls.collection, "Archived Source", curated_text="archived body",
            status=KnowledgeDocument.Status.ARCHIVED,
        )

    def test_each_shape_gets_its_expected_structural_state(self):
        report = run_knowledge_preflight()
        expected = {
            self.ready.pk: READY_CANONICAL,
            self.with_source.pk: READY_CANONICAL,
            self.source_only.pk: SOURCE_WITHOUT_CHUNKS,
            self.empty_active.pk: EMPTY_ACTIVE,
            self.unusable.pk: UNUSABLE_CHUNKS,
            self.draft.pk: NON_ACTIVE,
            self.archived.pk: NON_ACTIVE,
        }
        for document_id, state in expected.items():
            with self.subTest(document=document_id):
                self.assertEqual(document_row(report, document_id)["structural_state"], state)

    def test_structural_state_carries_no_authority_claim(self):
        """V1 conflated 'has source and chunks' with 'authority unprovable'.

        The two documents below differ only in source presence and now share a
        structural state, because authority is its own axis.
        """
        report = run_knowledge_preflight()
        self.assertEqual(
            document_row(report, self.ready.pk)["structural_state"],
            document_row(report, self.with_source.pk)["structural_state"],
        )
        self.assertFalse(document_row(report, self.ready.pk)["has_source_text"])
        self.assertTrue(document_row(report, self.with_source.pk)["has_source_text"])

    def test_exactly_one_structural_state_per_document(self):
        report = run_knowledge_preflight()
        self.assertEqual(
            sum(report["summary"]["by_structural_state"].values()),
            report["summary"]["documents_in_scope"],
        )
        for row in report["documents"]:
            self.assertIn(row["structural_state"], STRUCTURAL_STATES)

    def test_non_active_is_not_a_repair_problem(self):
        report = run_knowledge_preflight()
        for document in (self.draft, self.archived):
            with self.subTest(document=document.title):
                row = document_row(report, document.pk)
                self.assertEqual(row["structural_state"], NON_ACTIVE)
                self.assertEqual(row["issues"], [])
                self.assertFalse(row["canonically_retrievable"])

    def test_structural_issue_codes_keep_their_v1_meanings(self):
        report = run_knowledge_preflight()
        self.assertIn(
            KP001_SOURCE_WITHOUT_CHUNKS, document_row(report, self.source_only.pk)["issues"]
        )
        self.assertIn(
            KP002_EMPTY_ACTIVE, document_row(report, self.empty_active.pk)["issues"]
        )
        self.assertIn(
            KP003_UNUSABLE_CHUNK_CONTENT, document_row(report, self.unusable.pk)["issues"]
        )

    def test_source_file_without_chunks_raises_kp005(self):
        document = make_document(
            self.collection, "Uploaded", source_file="ai_hub/knowledge/manual.txt",
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertIn(KP005_SOURCE_FILE_WITHOUT_CHUNKS, row["issues"])
        self.assertEqual(row["structural_state"], SOURCE_WITHOUT_CHUNKS)

    def test_chunk_index_anomalies_raise_kp006(self):
        gapped = make_document(self.collection, "Gapped", chunks=((1, "a"), (3, "b")))
        zero = make_document(self.collection, "Zero", chunks=((0, "a"), (1, "b")))
        report = run_knowledge_preflight()
        self.assertIn(KP006_CHUNK_INDEX_ANOMALY, document_row(report, gapped.pk)["issues"])
        self.assertIn(KP006_CHUNK_INDEX_ANOMALY, document_row(report, zero.pk)["issues"])


class PreflightAuthorityAndLifecycleTests(TestCase):
    """Axes 2 and 3 — the persisted claim and whether it still holds."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Preflight Lifecycle")

    # -- UNKNOWN ----------------------------------------------------------

    def test_unknown_with_usable_chunks_is_retrievable_but_authority_unknown(self):
        document = make_document(
            self.collection, "Unknown Chunks", chunks=((1, "body"),)
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["authority_mode"], MODES.UNKNOWN)
        self.assertEqual(row["lifecycle_state"], UNKNOWN_AUTHORITY)
        self.assertEqual(row["structural_state"], READY_CANONICAL)
        self.assertTrue(row["canonically_retrievable"])
        self.assertIn(KP004_UNKNOWN_AUTHORITY, row["issues"])

    def test_unknown_is_never_inferred_away(self):
        """Matching content, an ingestion marker and a single chunk change nothing."""
        document = make_document(self.collection, "Looks Derived", curated_text="body")
        KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1, section_title="Looks Derived",
            content="body", metadata={"ingestion": "initial_curated_text"},
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["authority_mode"], MODES.UNKNOWN)
        self.assertEqual(row["lifecycle_state"], UNKNOWN_AUTHORITY)
        self.assertTrue(row["provenance"]["known_derivation_origin"])

    def test_unknown_without_chunks_does_not_raise_kp004(self):
        """No chunk set means no authority question - it is a structural one."""
        document = make_document(self.collection, "No Chunks", curated_text="body")
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertNotIn(KP004_UNKNOWN_AUTHORITY, row["issues"])
        self.assertIn(KP001_SOURCE_WITHOUT_CHUNKS, row["issues"])

    def test_unknown_with_unexpected_provenance_is_flagged_not_reclassified(self):
        document = make_document(
            self.collection, "Unknown With Provenance", chunks=((1, "body"),),
            generation_input_fingerprint="i1:" + "0" * 64,
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["authority_mode"], MODES.UNKNOWN)
        self.assertEqual(row["lifecycle_state"], UNKNOWN_AUTHORITY)
        self.assertIn(KP010_LIFECYCLE_FACT_INCONSISTENCY, row["issues"])
        self.assertIn(
            INCONSISTENCY_NON_DERIVED_WITH_PROVENANCE, row["lifecycle"]["inconsistencies"]
        )

    # -- EXPLICIT ---------------------------------------------------------

    def test_explicit_with_usable_chunks_is_ready(self):
        document = make_document(
            self.collection, "Explicit", chunks=((1, "authored body"),),
            chunk_authority_mode=MODES.EXPLICIT,
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], EXPLICIT_AUTHORITY)
        self.assertTrue(row["canonically_retrievable"])
        self.assertNotIn(KP004_UNKNOWN_AUTHORITY, row["issues"])

    def test_explicit_with_differing_curated_text_is_NOT_stale(self):
        """curated_text is background material for an EXPLICIT set."""
        document = make_document(
            self.collection, "Explicit Divergent",
            curated_text="completely different source narrative",
            chunks=((1, "authored body"),),
            chunk_authority_mode=MODES.EXPLICIT,
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], EXPLICIT_AUTHORITY)
        self.assertNotIn(KP009_DERIVED_INPUT_CHANGED, row["issues"])
        self.assertNotIn(KP007_DERIVED_CHUNKS_MODIFIED, row["issues"])
        # No derivation evaluation happened at all.
        self.assertIsNone(row["lifecycle"]["input_matches"])
        self.assertIsNone(row["lifecycle"]["chunk_set_matches"])

    def test_explicit_with_unexpected_provenance_is_flagged(self):
        document = make_document(
            self.collection, "Explicit With Provenance", chunks=((1, "body"),),
            chunk_authority_mode=MODES.EXPLICIT,
            generator_identity=GENERATOR_CURATED_TEXT_SINGLE_CHUNK,
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], EXPLICIT_AUTHORITY)
        self.assertIn(KP010_LIFECYCLE_FACT_INCONSISTENCY, row["issues"])
        self.assertIn(
            INCONSISTENCY_NON_DERIVED_WITH_PROVENANCE, row["lifecycle"]["inconsistencies"]
        )

    # -- DERIVED healthy ---------------------------------------------------

    def test_healthy_derived_document_is_ready(self):
        document = make_derived_document(
            self.collection, "Healthy", curated_text="body", chunks=((1, "body"),),
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["authority_mode"], MODES.DERIVED)
        self.assertEqual(row["lifecycle_state"], DERIVED_CURRENT)
        self.assertTrue(row["lifecycle"]["input_matches"])
        self.assertTrue(row["lifecycle"]["chunk_set_matches"])
        self.assertTrue(row["lifecycle"]["generator_supported"])
        self.assertEqual(row["lifecycle"]["generator_version_relation"], VERSION_CURRENT)
        self.assertTrue(row["lifecycle"]["provenance_complete"])
        self.assertEqual(row["issues"], [])

    # -- DERIVED provenance incomplete ------------------------------------

    def test_derived_missing_each_provenance_fact_is_incomplete(self):
        cases = {
            "generation_input_fingerprint": INCONSISTENCY_DERIVED_WITHOUT_INPUT_FINGERPRINT,
            "generation_chunk_set_fingerprint": INCONSISTENCY_DERIVED_WITHOUT_CHUNK_SET_FINGERPRINT,
            "generator_identity": INCONSISTENCY_DERIVED_WITHOUT_GENERATOR_IDENTITY,
            "generator_version": INCONSISTENCY_DERIVED_WITHOUT_GENERATOR_VERSION,
        }
        for field, reason in cases.items():
            with self.subTest(missing=field):
                document = make_derived_document(
                    self.collection, f"Missing {field}", curated_text="body",
                    chunks=((1, "body"),),
                )
                setattr(document, field, None if field == "generator_version" else "")
                document.save(update_fields=[field])
                row = document_row(run_knowledge_preflight(), document.pk)
                self.assertEqual(row["lifecycle_state"], DERIVED_PROVENANCE_INCOMPLETE)
                self.assertFalse(row["lifecycle"]["provenance_complete"])
                self.assertIn(KP010_LIFECYCLE_FACT_INCONSISTENCY, row["issues"])
                self.assertIn(reason, row["lifecycle"]["inconsistencies"])

    # -- DERIVED unsupported generator -------------------------------------

    def test_derived_with_unsupported_generator_is_unverifiable_not_mismatched(self):
        document = make_derived_document(
            self.collection, "Future Parser", curated_text="body", chunks=((1, "body"),),
        )
        document.generator_identity = "some_future_parser"
        document.save(update_fields=["generator_identity"])
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], DERIVED_GENERATOR_UNSUPPORTED)
        self.assertIn(KP011_GENERATOR_UNSUPPORTED, row["issues"])
        self.assertFalse(row["lifecycle"]["generator_supported"])
        # No claim either way about the inputs.
        self.assertIsNone(row["lifecycle"]["input_matches"])
        self.assertIsNone(row["lifecycle"]["current_input_fingerprint"])
        self.assertNotIn(KP009_DERIVED_INPUT_CHANGED, row["issues"])
        # The chunk-set contract is generator-independent, so it still applies.
        self.assertTrue(row["lifecycle"]["chunk_set_matches"])


class PreflightTitleChangeRegressionTests(TestCase):
    """MANDATORY: the Slice 7 correction, end to end through the preflight."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Title Change")

    def test_changing_only_the_title_is_detected_as_input_change(self):
        """`title` becomes `section_title`, so it IS a generation input.

        A curated_text-only fingerprint would have reported DERIVED_CURRENT here.
        """
        document = make_derived_document(
            self.collection, "Original Title",
            curated_text="unchanged body", chunks=((1, "unchanged body"),),
        )
        self.assertEqual(
            document_row(run_knowledge_preflight(), document.pk)["lifecycle_state"],
            DERIVED_CURRENT,
        )

        document.title = "Renamed Title"
        document.save(update_fields=["title"])

        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], DERIVED_INPUT_CHANGED)
        self.assertIn(KP009_DERIVED_INPUT_CHANGED, row["issues"])
        self.assertFalse(row["lifecycle"]["input_matches"])
        # Chunks were not touched.
        self.assertTrue(row["lifecycle"]["chunk_set_matches"])
        self.assertNotIn(KP007_DERIVED_CHUNKS_MODIFIED, row["issues"])
        # Retrieval is unaffected.
        self.assertTrue(row["canonically_retrievable"])

    def test_changing_only_curated_text_is_detected_as_input_change(self):
        document = make_derived_document(
            self.collection, "Stable Title", curated_text="original body",
            chunks=((1, "original body"),),
        )
        document.curated_text = "rewritten body"
        document.save(update_fields=["curated_text"])
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], DERIVED_INPUT_CHANGED)
        self.assertFalse(row["lifecycle"]["input_matches"])

    def test_curated_text_outer_whitespace_only_is_not_an_input_change(self):
        document = make_derived_document(
            self.collection, "Whitespace", curated_text="body", chunks=((1, "body"),),
        )
        document.curated_text = "   body   "
        document.save(update_fields=["curated_text"])
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], DERIVED_CURRENT)
        self.assertTrue(row["lifecycle"]["input_matches"])


class PreflightChunkTamperRegressionTests(TestCase):
    """MANDATORY: an ungoverned chunk edit must never read as DERIVED_CURRENT."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Chunk Tamper")

    def test_raw_orm_content_edit_is_detected(self):
        document = make_derived_document(
            self.collection, "Tampered", curated_text="body", chunks=((1, "body"),),
        )
        chunk = document.chunks.get()
        chunk.content = "body edited by hand outside any governed path"
        chunk.save(update_fields=["content"])

        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], DERIVED_CHUNKS_MODIFIED)
        self.assertNotEqual(row["lifecycle_state"], DERIVED_CURRENT)
        self.assertIn(KP007_DERIVED_CHUNKS_MODIFIED, row["issues"])
        self.assertFalse(row["lifecycle"]["chunk_set_matches"])
        # Lifecycle integrity is broken; retrieval is not.
        self.assertTrue(row["canonically_retrievable"])
        self.assertEqual(
            ISSUE_CODES[KP007_DERIVED_CHUNKS_MODIFIED]["lifecycle_impact"],
            "blocks_safe_regeneration",
        )
        self.assertEqual(ISSUE_CODES[KP007_DERIVED_CHUNKS_MODIFIED]["retrieval_impact"], "none")

    def test_metadata_only_change_is_NOT_a_chunk_set_mismatch(self):
        document = make_derived_document(
            self.collection, "Metadata Only", curated_text="body", chunks=((1, "body"),),
        )
        chunk = document.chunks.get()
        chunk.metadata = {"ingestion": "something", "extra": True}
        chunk.save(update_fields=["metadata"])
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], DERIVED_CURRENT)
        self.assertTrue(row["lifecycle"]["chunk_set_matches"])

    def test_chunks_modified_dominates_input_changed(self):
        """Slice 6's precedence rule, asserted."""
        document = make_derived_document(
            self.collection, "Both Changed", curated_text="body", chunks=((1, "body"),),
        )
        chunk = document.chunks.get()
        chunk.content = "hand edited"
        chunk.save(update_fields=["content"])
        document.title = "Renamed Too"
        document.save(update_fields=["title"])

        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], DERIVED_CHUNKS_MODIFIED)
        # Both facts are still reported - the dominant state hides nothing.
        self.assertIn(KP007_DERIVED_CHUNKS_MODIFIED, row["issues"])
        self.assertIn(KP009_DERIVED_INPUT_CHANGED, row["issues"])
        self.assertFalse(row["lifecycle"]["chunk_set_matches"])
        self.assertFalse(row["lifecycle"]["input_matches"])


class PreflightGeneratorOutdatedRegressionTests(TestCase):
    """MANDATORY: a version bump is a rechunk recommendation, not staleness."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Generator Outdated")

    def test_older_generator_version_is_outdated_not_stale(self):
        document = make_derived_document(
            self.collection, "Old Generator", curated_text="body",
            chunks=((1, "body"),), version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION - 1,
        )
        row = document_row(run_knowledge_preflight(), document.pk)

        self.assertEqual(row["lifecycle_state"], GENERATOR_OUTDATED)
        self.assertIn(KP008_GENERATOR_OUTDATED, row["issues"])
        # Explicitly NOT source/input staleness.
        self.assertNotIn(KP009_DERIVED_INPUT_CHANGED, row["issues"])
        self.assertTrue(row["lifecycle"]["input_matches"])
        self.assertTrue(row["lifecycle"]["chunk_set_matches"])
        self.assertEqual(row["lifecycle"]["generator_version_relation"], VERSION_OUTDATED)
        self.assertEqual(
            ISSUE_CODES[KP008_GENERATOR_OUTDATED]["lifecycle_impact"],
            "rechunk_recommended",
        )

    def test_input_change_dominates_generator_outdated(self):
        document = make_derived_document(
            self.collection, "Old And Changed", curated_text="body",
            chunks=((1, "body"),), version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION - 1,
        )
        document.curated_text = "changed"
        document.save(update_fields=["curated_text"])
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], DERIVED_INPUT_CHANGED)
        self.assertIn(KP008_GENERATOR_OUTDATED, row["issues"])


class PreflightGeneratorVersionDirectionTests(TestCase):
    """A version difference is not one fact: behind and ahead are opposites.

    Behind, this Core knows a newer segmentation policy and can still verify the
    recorded inputs, so the finding is advisory. Ahead, the row was written by
    code this Core does not have; its input contract may differ, so an input
    verdict computed with today's contract would be meaningless. None of these
    cases involves an actual source-input change - `curated_text` and `title`
    are left exactly as generated throughout.
    """

    CURRENT = GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Version Direction")

    def _document(self, title, version):
        return make_derived_document(
            self.collection, title, curated_text="body",
            chunks=((1, "body"),), version=version,
        )

    def test_stored_equals_current_is_derived_current(self):
        document = self._document("Equal", self.CURRENT)
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], DERIVED_CURRENT)
        self.assertEqual(row["lifecycle"]["generator_version_relation"], VERSION_CURRENT)
        self.assertEqual(row["issues"], [])

    def test_stored_below_current_is_outdated_only(self):
        document = self._document("Behind", self.CURRENT - 1)
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], GENERATOR_OUTDATED)
        self.assertEqual(row["lifecycle"]["generator_version_relation"], VERSION_OUTDATED)
        self.assertEqual(row["issues"], [KP008_GENERATOR_OUTDATED])
        # Behind is still verifiable: the input axis is answered, not skipped.
        self.assertTrue(row["lifecycle"]["input_matches"])

    def test_stored_above_current_is_version_ahead_not_outdated(self):
        document = self._document("Ahead", self.CURRENT + 1)
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], DERIVED_GENERATOR_VERSION_AHEAD)
        self.assertEqual(row["lifecycle"]["generator_version_relation"], VERSION_AHEAD)
        self.assertIn(KP012_GENERATOR_VERSION_AHEAD, row["issues"])
        self.assertNotIn(KP008_GENERATOR_OUTDATED, row["issues"])

    def test_version_ahead_leaves_the_input_axis_unverifiable(self):
        """Never fabricate an input verdict under an unknown newer contract."""
        document = self._document("Ahead Unverifiable", self.CURRENT + 5)
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertIsNone(row["lifecycle"]["input_matches"])
        self.assertIsNone(row["lifecycle"]["current_input_fingerprint"])
        self.assertNotIn(KP009_DERIVED_INPUT_CHANGED, row["issues"])
        # The chunk-set contract is generator-independent, so it still applies.
        self.assertTrue(row["lifecycle"]["chunk_set_matches"])

    def test_version_ahead_blocks_safe_regeneration(self):
        self.assertEqual(
            ISSUE_CODES[KP012_GENERATOR_VERSION_AHEAD]["lifecycle_impact"],
            "blocks_safe_regeneration",
        )
        self.assertEqual(
            ISSUE_CODES[KP012_GENERATOR_VERSION_AHEAD]["retrieval_impact"], "none"
        )

    def test_version_ahead_does_not_affect_retrieval(self):
        document = self._document("Ahead Retrievable", self.CURRENT + 1)
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertTrue(row["canonically_retrievable"])
        self.assertEqual(row["structural_state"], READY_CANONICAL)

    def test_version_ahead_is_distinct_from_an_unknown_identity(self):
        """A known identity at a newer version is NOT an unknown identity."""
        ahead = self._document("Known Ahead", self.CURRENT + 1)
        unknown = self._document("Unknown Identity", self.CURRENT)
        unknown.generator_identity = "some_future_generator"
        unknown.save(update_fields=["generator_identity"])

        report = run_knowledge_preflight()
        ahead_row = document_row(report, ahead.pk)
        unknown_row = document_row(report, unknown.pk)

        self.assertNotEqual(ahead_row["lifecycle_state"], unknown_row["lifecycle_state"])
        self.assertEqual(unknown_row["lifecycle_state"], DERIVED_GENERATOR_UNSUPPORTED)
        self.assertIn(KP011_GENERATOR_UNSUPPORTED, unknown_row["issues"])
        self.assertNotIn(KP012_GENERATOR_VERSION_AHEAD, unknown_row["issues"])
        self.assertNotIn(KP011_GENERATOR_UNSUPPORTED, ahead_row["issues"])
        # The identity IS supported; only the version is unknown.
        self.assertTrue(ahead_row["lifecycle"]["generator_supported"])
        self.assertFalse(unknown_row["lifecycle"]["generator_supported"])
        self.assertIsNone(unknown_row["lifecycle"]["generator_version_relation"])

    def test_chunk_tampering_still_dominates_version_ahead(self):
        document = self._document("Ahead And Tampered", self.CURRENT + 1)
        chunk = document.chunks.get(chunk_index=1)
        chunk.content = "edited outside a governed path"
        chunk.save(update_fields=["content"])
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], DERIVED_CHUNKS_MODIFIED)
        self.assertIn(KP012_GENERATOR_VERSION_AHEAD, row["issues"])

    def test_each_direction_is_counted_separately_in_the_census(self):
        self._document("Census Equal", self.CURRENT)
        self._document("Census Behind", self.CURRENT - 1)
        self._document("Census Ahead", self.CURRENT + 1)
        lifecycle = run_knowledge_preflight()["lifecycle"]
        self.assertEqual(lifecycle["derived_current"], 1)
        self.assertEqual(lifecycle["generator_outdated"], 1)
        self.assertEqual(lifecycle["derived_generator_version_ahead"], 1)


class PreflightLifecycleNamingContractTests(TestCase):
    """Lifecycle state names describe a CLAIM's standing, never readiness.

    `EXPLICIT_AUTHORITY` and `DERIVED_CURRENT` were once `READY_EXPLICIT` and
    `READY_DERIVED`, which invited the reading that a lifecycle verdict implied a
    document was retrievable. It does not: retrievability is a separate axis and
    the two can disagree in both directions.
    """

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Lifecycle Naming")

    def test_no_lifecycle_state_name_claims_readiness(self):
        for name in LIFECYCLE_STATES:
            with self.subTest(state=name):
                self.assertNotIn("READY", name)

    def test_explicit_authority_coexists_with_unusable_chunks(self):
        document = make_document(
            self.collection, "Explicit But Empty", chunks=((1, "   "),),
            chunk_authority_mode=MODES.EXPLICIT,
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], EXPLICIT_AUTHORITY)
        self.assertEqual(row["structural_state"], UNUSABLE_CHUNKS)
        self.assertFalse(row["canonically_retrievable"])

    def test_explicit_authority_survives_a_non_active_status(self):
        document = make_document(
            self.collection, "Explicit Archived", chunks=((1, "body"),),
            status=KnowledgeDocument.Status.ARCHIVED,
            chunk_authority_mode=MODES.EXPLICIT,
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertEqual(row["lifecycle_state"], EXPLICIT_AUTHORITY)
        self.assertFalse(row["canonically_retrievable"])

    def test_derived_current_is_independent_of_retrievability(self):
        retrievable = make_derived_document(
            self.collection, "Derived Active", curated_text="body", chunks=((1, "body"),),
        )
        archived = make_derived_document(
            self.collection, "Derived Archived", curated_text="body", chunks=((1, "body"),),
        )
        KnowledgeDocument.objects.filter(pk=archived.pk).update(
            status=KnowledgeDocument.Status.ARCHIVED
        )

        report = run_knowledge_preflight()
        for document in (retrievable, archived):
            with self.subTest(document=document.title):
                self.assertEqual(
                    document_row(report, document.pk)["lifecycle_state"], DERIVED_CURRENT
                )
        self.assertTrue(document_row(report, retrievable.pk)["canonically_retrievable"])
        self.assertFalse(document_row(report, archived.pk)["canonically_retrievable"])


class PreflightIssueCodeContractTests(TestCase):
    def test_twelve_documented_codes_with_precise_impact_dimensions(self):
        self.assertEqual(len(ISSUE_CODES), 12)
        for code, spec in ISSUE_CODES.items():
            with self.subTest(code=code):
                self.assertIn(spec["severity"], {"blocker", "warning", "info"})
                self.assertIn(
                    spec["retrieval_impact"],
                    {"blocks_retrieval", "degrades_retrieval", "none"},
                )
                self.assertIn(
                    spec["lifecycle_impact"],
                    {
                        "blocks_safe_regeneration",
                        "regeneration_recommended",
                        "rechunk_recommended",
                        "advisory",
                        "none",
                    },
                )
                self.assertTrue(spec["meaning"])

    def test_v1_codes_keep_their_severities(self):
        self.assertEqual(ISSUE_CODES[KP001_SOURCE_WITHOUT_CHUNKS]["severity"], "blocker")
        self.assertEqual(ISSUE_CODES[KP002_EMPTY_ACTIVE]["severity"], "warning")
        self.assertEqual(ISSUE_CODES[KP003_UNUSABLE_CHUNK_CONTENT]["severity"], "blocker")
        self.assertEqual(ISSUE_CODES[KP004_UNKNOWN_AUTHORITY]["severity"], SEVERITY_INFO)
        self.assertEqual(ISSUE_CODES[KP005_SOURCE_FILE_WITHOUT_CHUNKS]["severity"], "blocker")
        self.assertEqual(ISSUE_CODES[KP006_CHUNK_INDEX_ANOMALY]["severity"], "warning")

    def test_kp004_remains_informational_and_not_a_defect(self):
        meaning = ISSUE_CODES[KP004_UNKNOWN_AUTHORITY]["meaning"].lower()
        self.assertIn("not a defect", meaning)
        self.assertEqual(ISSUE_CODES[KP004_UNKNOWN_AUTHORITY]["retrieval_impact"], "none")

    def test_lifecycle_codes_never_block_retrieval(self):
        """A lifecycle warning must not imply a retrieval problem."""
        for code in (
            KP007_DERIVED_CHUNKS_MODIFIED,
            KP008_GENERATOR_OUTDATED,
            KP009_DERIVED_INPUT_CHANGED,
            KP010_LIFECYCLE_FACT_INCONSISTENCY,
            KP011_GENERATOR_UNSUPPORTED,
        ):
            with self.subTest(code=code):
                self.assertEqual(ISSUE_CODES[code]["retrieval_impact"], "none")

    def test_every_emitted_issue_carries_both_impact_dimensions(self):
        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Emitted Codes")
        make_document(collection, "Src", curated_text="body")
        make_document(collection, "Chunks", chunks=((1, "body"),))
        report = run_knowledge_preflight()
        self.assertTrue(report["issues"])
        for issue in report["issues"]:
            with self.subTest(code=issue["code"]):
                self.assertIn(issue["code"], ISSUE_CODES)
                self.assertEqual(issue["severity"], ISSUE_CODES[issue["code"]]["severity"])
                self.assertEqual(
                    issue["retrieval_impact"], ISSUE_CODES[issue["code"]]["retrieval_impact"]
                )
                self.assertEqual(
                    issue["lifecycle_impact"], ISSUE_CODES[issue["code"]]["lifecycle_impact"]
                )


class PreflightSummaryCensusTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Census")
        make_document(cls.collection, "Unknown", chunks=((1, "body"),))
        make_document(
            cls.collection, "Explicit", chunks=((1, "body"),),
            chunk_authority_mode=MODES.EXPLICIT,
        )
        make_derived_document(cls.collection, "Healthy", curated_text="b", chunks=((1, "b"),))
        stale = make_derived_document(
            cls.collection, "Stale", curated_text="b", chunks=((1, "b"),)
        )
        stale.curated_text = "changed"
        stale.save(update_fields=["curated_text"])
        tampered = make_derived_document(
            cls.collection, "Tampered", curated_text="b", chunks=((1, "b"),)
        )
        chunk = tampered.chunks.get()
        chunk.content = "edited"
        chunk.save(update_fields=["content"])
        make_derived_document(
            cls.collection, "Outdated", curated_text="b", chunks=((1, "b"),), version=0,
        )

    def test_authority_census(self):
        summary = run_knowledge_preflight()["summary"]
        self.assertEqual(summary["by_authority_mode"][MODES.UNKNOWN], 1)
        self.assertEqual(summary["by_authority_mode"][MODES.EXPLICIT], 1)
        self.assertEqual(summary["by_authority_mode"][MODES.DERIVED], 4)

    def test_lifecycle_census_answers_the_operator_questions(self):
        lifecycle = run_knowledge_preflight()["lifecycle"]
        self.assertEqual(lifecycle["unknown"], 1)
        self.assertEqual(lifecycle["explicit"], 1)
        self.assertEqual(lifecycle["derived"], 4)
        self.assertEqual(lifecycle["derived_current"], 1)
        self.assertEqual(lifecycle["explicit_authority"], 1)
        self.assertEqual(lifecycle["derived_input_changed"], 1)
        self.assertEqual(lifecycle["derived_chunks_modified"], 1)
        self.assertEqual(lifecycle["generator_outdated"], 1)
        self.assertEqual(lifecycle["derived_provenance_incomplete"], 0)
        self.assertEqual(lifecycle["derived_generator_unsupported"], 0)

    def test_lifecycle_states_sum_to_documents_in_scope(self):
        summary = run_knowledge_preflight()["summary"]
        self.assertEqual(
            sum(summary["by_lifecycle_state"].values()), summary["documents_in_scope"]
        )
        for name in summary["by_lifecycle_state"]:
            self.assertIn(name, LIFECYCLE_STATES)

    def test_truncation_does_not_corrupt_the_lifecycle_census(self):
        """Slice 5 property preserved: summary covers full scope."""
        full = run_knowledge_preflight()
        truncated = run_knowledge_preflight(document_limit=1)
        self.assertEqual(len(truncated["documents"]), 1)
        self.assertTrue(truncated["scope"]["documents_truncated"])
        self.assertEqual(truncated["summary"], full["summary"])
        self.assertEqual(truncated["lifecycle"], full["lifecycle"])

    def test_report_declares_its_contract_version(self):
        self.assertEqual(
            run_knowledge_preflight()["contract_version"], PREFLIGHT_CONTRACT_VERSION
        )
        self.assertEqual(PREFLIGHT_CONTRACT_VERSION, 2)

    def test_convergence_reports_facts_not_a_verdict(self):
        convergence = run_knowledge_preflight()["convergence"]
        self.assertIn("unknown_authority", convergence)
        self.assertNotIn("canonical_transition_safe", convergence)
        self.assertNotIn("safe", json.dumps(convergence))


class PreflightScopeAndBoundsTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.first = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Alpha Collection")
        cls.second = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Beta Collection")
        cls.inactive = KnowledgeCollection.objects.create(
            application_scope=test_scope(),
            name="Gamma Collection", is_active=False,
        )
        make_document(cls.first, "Alpha Doc", curated_text="a")
        make_document(cls.second, "Beta Doc", curated_text="b")
        make_document(cls.inactive, "Gamma Doc", chunks=((1, "gamma body"),))

    def test_collection_scoping(self):
        report = run_knowledge_preflight(collection_ids=[self.first.pk])
        self.assertEqual(report["summary"]["documents_in_scope"], 1)
        self.assertEqual(report["summary"]["documents_total"], 3)

    def test_status_scoping(self):
        make_document(
            self.first, "Alpha Draft", curated_text="d",
            status=KnowledgeDocument.Status.DRAFT,
        )
        report = run_knowledge_preflight(statuses=[KnowledgeDocument.Status.DRAFT])
        self.assertEqual(report["summary"]["documents_in_scope"], 1)
        self.assertEqual(report["documents"][0]["structural_state"], NON_ACTIVE)

    def test_collection_grouping_carries_all_three_axes(self):
        report = run_knowledge_preflight()
        self.assertEqual(len(report["collections"]), 3)
        for row in report["collections"]:
            with self.subTest(collection=row["name"]):
                self.assertEqual(row["documents"], 1)
                self.assertIn("by_structural_state", row)
                self.assertIn("by_authority_mode", row)
                self.assertIn("by_lifecycle_state", row)

    def test_inactive_collection_document_is_not_retrievable(self):
        """Retrievability stays separate from structural and lifecycle state."""
        report = run_knowledge_preflight(collection_ids=[self.inactive.pk])
        row = report["documents"][0]
        self.assertEqual(row["structural_state"], READY_CANONICAL)
        self.assertFalse(row["collection_is_active"])
        self.assertFalse(row["canonically_retrievable"])

    def test_document_limit_is_clamped(self):
        self.assertEqual(
            run_knowledge_preflight(document_limit=0)["scope"]["document_limit"], 1
        )
        self.assertEqual(
            run_knowledge_preflight(document_limit="nonsense")["scope"]["document_limit"],
            run_knowledge_preflight()["scope"]["document_limit"],
        )

    def test_report_contains_no_chunk_bodies(self):
        """Bounded output survives lifecycle awareness."""
        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Bodies")
        make_derived_document(
            collection, "Long", curated_text="x",
            chunks=((1, "unmistakable-chunk-body-marker " * 200),),
        )
        report = run_knowledge_preflight()
        self.assertNotIn("unmistakable-chunk-body-marker", json.dumps(report))

    def test_empty_corpus_produces_a_valid_report(self):
        KnowledgeDocument.objects.all().delete()
        report = run_knowledge_preflight()
        self.assertEqual(report["summary"]["documents_in_scope"], 0)
        self.assertEqual(report["documents"], [])
        self.assertEqual(report["issues"], [])


class PreflightReadOnlyInvariantTests(TestCase):
    """THE core invariant: preflight changes nothing, ever — lifecycle included."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="ReadOnly V2")
        make_document(cls.collection, "Unknown", chunks=((1, "body"),))
        make_document(
            cls.collection, "Explicit", chunks=((1, "body"),),
            chunk_authority_mode=MODES.EXPLICIT,
        )
        make_derived_document(cls.collection, "Healthy", curated_text="b", chunks=((1, "b"),))
        stale = make_derived_document(
            cls.collection, "Stale", curated_text="b", chunks=((1, "b"),)
        )
        stale.title = "Stale Renamed"
        stale.save(update_fields=["title"])
        tampered = make_derived_document(
            cls.collection, "Tampered", curated_text="b", chunks=((1, "b"),)
        )
        chunk = tampered.chunks.get()
        chunk.content = "edited"
        chunk.save(update_fields=["content"])
        incomplete = make_derived_document(
            cls.collection, "Incomplete", curated_text="b", chunks=((1, "b"),)
        )
        incomplete.generator_identity = ""
        incomplete.save(update_fields=["generator_identity"])
        make_document(cls.collection, "Source Only", curated_text="orphan")
        make_document(cls.collection, "Empty")

    @staticmethod
    def _snapshot():
        return {
            "documents": list(
                KnowledgeDocument.objects.order_by("pk").values(
                    "pk", "collection_id", "title", "curated_text", "source_file",
                    "tags", "language", "status", "notes",
                    "chunk_authority_mode", "generation_input_fingerprint",
                    "generation_chunk_set_fingerprint", "generator_identity",
                    "generator_version", "created_at", "updated_at",
                )
            ),
            "chunks": list(
                KnowledgeDocumentChunk.objects.order_by("pk").values(
                    "pk", "document_id", "chunk_index", "section_title", "content",
                    "token_estimate", "metadata", "created_at", "updated_at",
                )
            ),
            "collections": list(
                KnowledgeCollection.objects.order_by("pk").values(
                    "pk", "name", "description", "is_active", "created_at", "updated_at",
                )
            ),
        }

    def test_preflight_does_not_modify_any_row_including_lifecycle_fields(self):
        before = self._snapshot()
        run_knowledge_preflight()
        run_knowledge_preflight(collection_ids=[self.collection.pk])
        run_knowledge_preflight(statuses=[KnowledgeDocument.Status.ACTIVE])
        self.assertEqual(before, self._snapshot())

    def test_it_never_repairs_an_inconsistent_document(self):
        """Observing a contradiction must not fix it."""
        document = KnowledgeDocument.objects.get(title="Incomplete")
        run_knowledge_preflight()
        document.refresh_from_db()
        self.assertEqual(document.generator_identity, "")
        self.assertEqual(document.chunk_authority_mode, MODES.DERIVED)

    def test_it_never_fills_a_missing_fingerprint(self):
        document = KnowledgeDocument.objects.get(title="Unknown")
        run_knowledge_preflight()
        document.refresh_from_db()
        self.assertEqual(document.generation_input_fingerprint, "")
        self.assertEqual(document.generation_chunk_set_fingerprint, "")

    def test_it_never_reclassifies_authority(self):
        document = KnowledgeDocument.objects.get(title="Tampered")
        run_knowledge_preflight()
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, MODES.DERIVED)

    def test_row_counts_are_unchanged(self):
        counts = (
            KnowledgeCollection.objects.count(),
            KnowledgeDocument.objects.count(),
            KnowledgeDocumentChunk.objects.count(),
        )
        run_knowledge_preflight()
        self.assertEqual(
            counts,
            (
                KnowledgeCollection.objects.count(),
                KnowledgeDocument.objects.count(),
                KnowledgeDocumentChunk.objects.count(),
            ),
        )

    def test_timestamps_and_metadata_are_untouched(self):
        chunk = KnowledgeDocumentChunk.objects.first()
        document = chunk.document
        before = (
            chunk.updated_at, chunk.created_at, chunk.metadata,
            document.updated_at, document.created_at,
        )
        run_knowledge_preflight()
        chunk.refresh_from_db()
        document.refresh_from_db()
        self.assertEqual(
            before,
            (
                chunk.updated_at, chunk.created_at, chunk.metadata,
                document.updated_at, document.created_at,
            ),
        )

    def test_repeated_runs_return_an_identical_report(self):
        self.assertEqual(run_knowledge_preflight(), run_knowledge_preflight())

    def test_the_report_carries_no_timestamp(self):
        serialized = json.dumps(run_knowledge_preflight(), default=str)
        for forbidden in ("generated_at", "timestamp", "run_at"):
            self.assertNotIn(forbidden, serialized)

    def test_the_service_exposes_no_write_capability(self):
        from ai_hub.services import knowledge_preflight

        for forbidden in (
            "repair", "fix", "backfill", "regenerate", "adjudicate", "apply", "write",
        ):
            matches = [
                name for name in dir(knowledge_preflight)
                if forbidden in name.lower() and not name.startswith("__")
            ]
            with self.subTest(term=forbidden):
                self.assertEqual(matches, [])


class PreflightQueryEfficiencyTests(TestCase):
    """Query count must not grow with the corpus."""

    BASE_QUERIES = 3  # documents + chunk probe pass + total document count
    DERIVED_QUERIES = 4  # + one full-content pass, only when something claims DERIVED

    def _build(self, document_count, *, derived=False):
        collection = KnowledgeCollection.objects.create(
            application_scope=test_scope(),
            name=f"Scale {document_count} {derived}",
        )
        for index in range(document_count):
            document = make_document(
                collection, f"Doc {index:04d}", curated_text="source body",
                chunks=((1, "chunk body"),),
            )
            if derived:
                record_generation(document)
        return collection

    def test_base_query_count_is_constant(self):
        self._build(5)
        with self.assertNumQueries(self.BASE_QUERIES):
            run_knowledge_preflight()

    def test_base_query_count_is_unchanged_for_a_larger_corpus(self):
        self._build(120)
        with self.assertNumQueries(self.BASE_QUERIES):
            report = run_knowledge_preflight()
        self.assertEqual(report["summary"]["documents_in_scope"], 120)

    def test_derived_documents_add_exactly_one_query(self):
        """The full-content pass runs once, not per document."""
        self._build(5, derived=True)
        with self.assertNumQueries(self.DERIVED_QUERIES):
            run_knowledge_preflight()

    def test_derived_query_count_is_unchanged_for_a_larger_corpus(self):
        self._build(120, derived=True)
        with self.assertNumQueries(self.DERIVED_QUERIES):
            report = run_knowledge_preflight()
        self.assertEqual(report["lifecycle"]["derived_current"], 120)

    def test_truncating_the_detail_list_does_not_change_query_count(self):
        self._build(120)
        with self.assertNumQueries(self.BASE_QUERIES):
            report = run_knowledge_preflight(document_limit=10)
        self.assertEqual(len(report["documents"]), 10)
        self.assertEqual(report["summary"]["documents_in_scope"], 120)


class PreflightSecurityBoundaryTests(TestCase):
    """Operator diagnostic, not an Agent-facing Knowledge tool."""

    def test_preflight_is_not_registered_as_a_tool_definition(self):
        self.assertFalse(
            ToolDefinition.objects.filter(
                config__callable__icontains="knowledge_preflight"
            ).exists()
        )
        self.assertFalse(ToolDefinition.objects.filter(name__icontains="preflight").exists())

    def test_preflight_is_not_one_of_the_canonical_knowledge_tools(self):
        from ai_hub.services.knowledge_tooling import KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES

        for callable_path in KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES.values():
            self.assertNotIn("preflight", callable_path)
            self.assertNotIn("lifecycle", callable_path)

    def test_preflight_is_not_auto_resolved_into_an_agent_manifest(self):
        from ai_hub.services.tool_resolution import resolve_agent_tools

        provider = ProviderConfig.objects.create(
            name="preflight-provider", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        agent = AgentProfile.objects.create(
            application_scope=test_scope(),
            name="preflight-agent", role="Boundary", model_config=model,
        )
        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Preflight Boundary")
        agent.knowledge_collections.add(collection)

        names = resolve_agent_tools(agent).tool_names()
        self.assertTrue(all("preflight" not in name for name in names))

    def test_preflight_takes_no_agent_argument(self):
        import inspect

        signature = inspect.signature(run_knowledge_preflight)
        self.assertNotIn("agent", signature.parameters)
        self.assertEqual(
            set(signature.parameters), {"collection_ids", "statuses", "document_limit"}
        )


class PreflightCommandTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Command V2")
        make_document(cls.collection, "Unknown", chunks=((1, "body"),))
        make_derived_document(cls.collection, "Healthy", curated_text="b", chunks=((1, "b"),))

    def _run(self, *args):
        out = StringIO()
        call_command("knowledge_preflight", *args, stdout=out)
        return out.getvalue()

    def test_command_reports_all_three_axes_and_changes_nothing(self):
        before = PreflightReadOnlyInvariantTests._snapshot()
        output = self._run()
        self.assertIn("Preflight contract v2", output)
        self.assertIn("Structural state:", output)
        self.assertIn("Authority mode:", output)
        self.assertIn("Lifecycle state:", output)
        self.assertIn("read-only", output)
        self.assertEqual(before, PreflightReadOnlyInvariantTests._snapshot())

    def test_command_emits_valid_json_with_v2_facts(self):
        payload = json.loads(self._run("--json"))
        self.assertEqual(payload["contract_version"], 2)
        self.assertIn("lifecycle", payload)
        self.assertIn("by_authority_mode", payload["summary"])
        self.assertIn("by_lifecycle_state", payload["summary"])
        self.assertIn("lifecycle", payload["documents"][0])

    def test_command_renders_a_version_ahead_document_end_to_end(self):
        """The newest lifecycle state must survive the whole operator path."""
        document = make_derived_document(
            self.collection, "Ahead", curated_text="b", chunks=((1, "b"),),
            version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION + 1,
        )
        before = PreflightReadOnlyInvariantTests._snapshot()
        self.assertIn(DERIVED_GENERATOR_VERSION_AHEAD, self._run())
        payload = json.loads(self._run("--json"))
        self.assertEqual(
            document_row(payload, document.pk)["lifecycle_state"],
            DERIVED_GENERATOR_VERSION_AHEAD,
        )
        self.assertEqual(payload["lifecycle"]["derived_generator_version_ahead"], 1)
        self.assertEqual(before, PreflightReadOnlyInvariantTests._snapshot())

    def test_command_shows_impact_dimensions_for_issues(self):
        output = self._run()
        self.assertIn("retrieval:", output)
        self.assertIn("lifecycle:", output)

    def test_command_has_no_write_flags(self):
        from django.core.management.base import CommandError

        for flag in (
            "--fix", "--write", "--repair", "--backfill", "--apply",
            "--regenerate", "--adjudicate",
        ):
            with self.subTest(flag=flag):
                with self.assertRaisesMessage(CommandError, "unrecognized arguments"):
                    call_command(
                        "knowledge_preflight", flag, stdout=StringIO(), stderr=StringIO()
                    )


class PreflightSummaryRenderingTests(TestCase):
    def test_summary_lists_every_axis(self):
        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Render V2")
        make_document(collection, "Ready", chunks=((1, "body"),))
        rendered = "\n".join(summarize_preflight(run_knowledge_preflight()))
        for name in STRUCTURAL_STATES:
            self.assertIn(name, rendered)
        for value, _label in MODES.choices:
            self.assertIn(value, rendered)
