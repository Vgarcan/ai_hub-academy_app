"""Evidence available today for a Knowledge lifecycle contract (Slice 4).

PURPOSE
-------
ADR-N8's direction is decided: `KnowledgeDocument.curated_text` is a SOURCE /
AUTHORING representation and `KnowledgeDocumentChunk` is the RETRIEVAL
representation. Chunks are either DERIVED from a source or EXPLICITLY authored,
and explicit chunks must never be overwritten automatically.

To act on that distinction, AI Hub has to be able to TELL THE TWO APART. This
module measures whether the evidence that exists today is sufficient.

It answers, with measurements rather than assumptions:

* what provenance each production chunk-creation path actually writes;
* whether that provenance survives later edits (does a marker still mean what
  it claimed?);
* whether the absence of a marker means anything;
* whether structural heuristics (chunk count, section title, content equality)
  can substitute for a marker, and where each one breaks;
* whether timestamps can answer "did the source change?".

Slice 3 established that nothing links the two representations. This module is
narrower: given that nothing links them, WHAT CAN BE INFERRED AFTER THE FACT.

Scope guard: every test below asserts CURRENT behavior. Nothing here tests a
contract that production does not implement, and no test is expected to fail.

NO PRODUCTION CODE IS CHANGED BY THIS MODULE.
"""
from django.test import TestCase

from ai_hub.models import (
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
)
from ai_hub.services.knowledge_ingestion import ensure_initial_knowledge_chunk
from ai_hub.test_application_scope_helpers import test_scope


# The provenance markers production actually writes, gathered by inspection and
# pinned here so a new creation path shows up as a test change.
RUNTIME_FALLBACK_MARKER = {"ingestion": "initial_curated_text"}
MIGRATION_0019_MARKER = {"ingestion": "initial_curated_text_backfill"}
STARTER_DEMO_MARKER = {"example": True}


class ProvenanceMarkerInventoryTests(TestCase):
    """What each production chunk-creation path records about its origin."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Marker Inventory")

    def _document(self, curated_text="source body", title="Marker Document"):
        return KnowledgeDocument.objects.create(
            collection=self.collection,
            title=title,
            curated_text=curated_text,
            status=KnowledgeDocument.Status.ACTIVE,
        )

    def test_runtime_fallback_records_a_derived_marker(self):
        """Build Console `create` mode is the only automatic runtime caller."""
        chunk = ensure_initial_knowledge_chunk(self._document())
        self.assertEqual(chunk.metadata, RUNTIME_FALLBACK_MARKER)

    def test_the_two_derived_markers_are_different_strings(self):
        """A classifier must know BOTH spellings, not one.

        The runtime fallback writes `initial_curated_text`; migration 0019 wrote
        `initial_curated_text_backfill`. Matching only one silently
        misclassifies every document that predates the migration.
        """
        self.assertNotEqual(
            RUNTIME_FALLBACK_MARKER["ingestion"], MIGRATION_0019_MARKER["ingestion"]
        )
        self.assertTrue(
            MIGRATION_0019_MARKER["ingestion"].startswith(
                RUNTIME_FALLBACK_MARKER["ingestion"]
            )
        )

    def test_a_directly_created_chunk_records_nothing(self):
        """The default. Admin, host adapters, fixtures and imports all land here.

        `metadata` defaults to an empty dict, so 'no marker' is the most common
        state and carries no information at all.
        """
        document = self._document()
        chunk = KnowledgeDocumentChunk.objects.create(
            document=document,
            chunk_index=1,
            section_title="Hand authored",
            content="explicitly authored body",
        )
        self.assertEqual(chunk.metadata, {})

    def test_marker_absence_cannot_distinguish_explicit_from_unknown(self):
        """Two chunks with different origins are byte-identical in provenance.

        One is deliberately curated; one is a legacy row of unknown origin.
        Nothing persisted separates them.
        """
        document = self._document()
        deliberate = KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1, section_title="A", content="body a"
        )
        unknown_legacy = KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=2, section_title="B", content="body b"
        )
        self.assertEqual(deliberate.metadata, unknown_legacy.metadata)
        self.assertEqual(deliberate.metadata, {})

    def test_seeded_chunks_are_derived_but_carry_no_derived_marker(self):
        """`starter_demo` builds a chunk FROM curated_text with `{"example": True}`.

        It is derived in fact and unmarked as derived in the data, so a
        marker-based classifier would call it explicit and refuse to refresh it.
        """
        self.assertNotIn("ingestion", STARTER_DEMO_MARKER)


class ProvenanceMarkerDurabilityTests(TestCase):
    """Does a marker still describe the chunk after the chunk is edited?

    This is the central question for ADR-N8: a marker records how a row was
    CREATED, and nothing revisits it.
    """

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Marker Durability")

    def _derived_document(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection,
            title="Durability Document",
            curated_text="original source body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        return document, ensure_initial_knowledge_chunk(document)

    def test_editing_a_derived_chunk_leaves_it_claiming_to_be_derived(self):
        """A hand-edited chunk keeps the derived marker.

        After this edit the row is, in substance, an explicitly curated chunk.
        Its provenance still says it was generated from curated_text. Any
        automatic refresh trusting the marker would destroy the operator's work
        - which is precisely what the explicit-chunks invariant forbids.
        """
        document, chunk = self._derived_document()

        chunk.content = "operator rewrote this by hand"
        chunk.save(update_fields=["content", "updated_at"])

        chunk.refresh_from_db()
        self.assertEqual(chunk.metadata, RUNTIME_FALLBACK_MARKER)
        self.assertNotEqual(chunk.content, document.curated_text)

    def test_editing_the_source_leaves_the_chunk_claiming_to_be_current(self):
        """The mirror image: the source moves and the marker does not notice."""
        document, chunk = self._derived_document()

        document.curated_text = "source has since been rewritten"
        document.save(update_fields=["curated_text", "updated_at"])

        chunk.refresh_from_db()
        self.assertEqual(chunk.metadata, RUNTIME_FALLBACK_MARKER)
        self.assertEqual(chunk.content, "original source body")

    def test_the_marker_alone_cannot_separate_those_two_situations(self):
        """Same marker, same inequality, opposite correct remedies.

        Case 1 (stale derived): the chunk should be regenerated.
        Case 2 (hand-edited):   the chunk must be preserved.
        Provenance is identical in both, so a marker-only classifier would
        apply the same action to both and be wrong half the time.
        """
        stale_document, stale_chunk = self._derived_document()
        stale_document.curated_text = "moved on"
        stale_document.save(update_fields=["curated_text", "updated_at"])

        edited_document, edited_chunk = self._derived_document()
        edited_chunk.content = "hand edited"
        edited_chunk.save(update_fields=["content", "updated_at"])

        stale_chunk.refresh_from_db()
        edited_chunk.refresh_from_db()

        self.assertEqual(stale_chunk.metadata, edited_chunk.metadata)
        self.assertNotEqual(stale_chunk.content, stale_document.curated_text)
        self.assertNotEqual(edited_chunk.content, edited_document.curated_text)


class StructuralHeuristicTests(TestCase):
    """Could structure substitute for a marker? Measured, including where it breaks."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Heuristics")

    def _derived(self, title="Heuristic Document", body="heuristic source body"):
        document = KnowledgeDocument.objects.create(
            collection=self.collection,
            title=title,
            curated_text=body,
            status=KnowledgeDocument.Status.ACTIVE,
        )
        return document, ensure_initial_knowledge_chunk(document)

    def test_a_derived_chunk_set_is_always_exactly_one_chunk(self):
        """Both derived paths produce a single chunk at index 1, whatever the size."""
        document, chunk = self._derived(body="word " * 5000)
        self.assertEqual(document.chunks.count(), 1)
        self.assertEqual(chunk.chunk_index, 1)

    def test_multiple_chunks_therefore_imply_explicit_authoring_today(self):
        """A useful one-way signal: nothing in production generates 2+ chunks.

        One-way only - a single chunk proves nothing, because an operator may
        legitimately author exactly one.
        """
        document = KnowledgeDocument.objects.create(
            collection=self.collection,
            title="Multi Chunk Document",
            curated_text="source",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        for index in range(1, 4):
            KnowledgeDocumentChunk.objects.create(
                document=document,
                chunk_index=index,
                section_title=f"Section {index}",
                content=f"body {index}",
            )
        self.assertGreater(document.chunks.count(), 1)
        self.assertEqual(
            list(document.chunks.values_list("metadata", flat=True)), [{}, {}, {}]
        )

    def test_a_derived_chunk_section_title_equals_the_document_title(self):
        document, chunk = self._derived(title="Exact Title Match")
        self.assertEqual(chunk.section_title, document.title)

    def test_renaming_the_document_breaks_the_section_title_heuristic(self):
        """SURPRISING - RECORDED, NOT FIXED.

        Renaming a document silently invalidates the only structural signal that
        links a derived chunk back to its origin, without touching the chunk.
        """
        document, chunk = self._derived(title="Original Title")
        document.title = "Renamed Title"
        document.save(update_fields=["title", "updated_at"])

        chunk.refresh_from_db()
        self.assertEqual(chunk.section_title, "Original Title")
        self.assertNotEqual(chunk.section_title, document.title)

    def test_content_equality_holds_at_creation_and_is_direction_blind(self):
        """`chunk.content == curated_text.strip()` detects divergence only.

        It cannot say which side moved, so it cannot choose between
        'regenerate the chunk' and 'preserve the chunk'.
        """
        document, chunk = self._derived(body="  equality body  ")
        self.assertEqual(chunk.content, document.curated_text.strip())

        document.curated_text = "changed"
        document.save(update_fields=["curated_text", "updated_at"])
        chunk.refresh_from_db()
        self.assertNotEqual(chunk.content, document.curated_text.strip())

    def test_multi_chunk_documents_never_satisfy_content_equality(self):
        """A legitimately chunked document always 'looks' divergent.

        Any equality-based staleness check must special-case chunk sets, or it
        will flag every properly chunked document as broken.
        """
        document = KnowledgeDocument.objects.create(
            collection=self.collection,
            title="Chunked Manual",
            curated_text="section one body\n\nsection two body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        for index, body in enumerate(["section one body", "section two body"], start=1):
            KnowledgeDocumentChunk.objects.create(
                document=document,
                chunk_index=index,
                section_title=f"Section {index}",
                content=body,
            )
        joined = "\n\n".join(
            chunk.content for chunk in document.chunks.order_by("chunk_index")
        )
        self.assertEqual(joined, document.curated_text)
        for chunk in document.chunks.all():
            self.assertNotEqual(chunk.content, document.curated_text)


class TimestampEvidenceTests(TestCase):
    """Can timestamps answer 'did the source change?' Measured: no."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Timestamps")

    def _derived(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection,
            title="Timestamp Document",
            curated_text="timestamp source body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        return document, ensure_initial_knowledge_chunk(document)

    def test_document_updated_at_moves_for_changes_unrelated_to_the_source(self):
        """SURPRISING - RECORDED, NOT FIXED.

        `updated_at` is `auto_now`, so a status, tag or title change advances it
        exactly like a `curated_text` rewrite. A 'document newer than chunk'
        rule would therefore report stale chunks for documents whose source
        never moved.
        """
        document, chunk = self._derived()
        original = document.updated_at

        document.status = KnowledgeDocument.Status.DRAFT
        document.save()
        document.refresh_from_db()

        self.assertGreater(document.updated_at, original)
        self.assertGreater(document.updated_at, chunk.updated_at)
        # The source itself is untouched and the chunk still matches it.
        self.assertEqual(chunk.content, document.curated_text)

    def test_a_targeted_save_can_avoid_advancing_updated_at_at_all(self):
        """The inverse failure: a source edit that leaves no timestamp trace.

        `save(update_fields=[...])` without `updated_at` skips the auto_now
        write, so `curated_text` can change while `updated_at` stands still.
        """
        document, chunk = self._derived()
        original = document.updated_at

        document.curated_text = "source changed with no timestamp trace"
        document.save(update_fields=["curated_text"])
        document.refresh_from_db()

        self.assertEqual(document.updated_at, original)
        self.assertNotEqual(chunk.content, document.curated_text)

    def test_timestamps_are_therefore_unusable_in_both_directions(self):
        """Both false positives and false negatives are reachable.

        Together with the two tests above: `updated_at` can advance without the
        source changing, and the source can change without `updated_at`
        advancing. It cannot support a staleness rule.
        """
        document, chunk = self._derived()
        self.assertEqual(
            KnowledgeDocument._meta.get_field("updated_at").auto_now, True
        )
        self.assertEqual(
            KnowledgeDocumentChunk._meta.get_field("updated_at").auto_now, True
        )


class NoPersistedLifecycleStateTests(TestCase):
    """What the schema offers today for lifecycle reasoning: nothing."""

    def test_no_lifecycle_or_version_field_exists_on_either_model(self):
        document_fields = {field.name for field in KnowledgeDocument._meta.get_fields()}
        chunk_fields = {field.name for field in KnowledgeDocumentChunk._meta.get_fields()}

        for candidate in (
            "chunking_mode",
            "content_mode",
            "authority_mode",
            "source_mode",
            "is_derived",
            "source_hash",
            "content_hash",
            "source_version",
            "chunk_generation",
            "chunker_version",
            "parser_version",
            "retrieval_ready",
        ):
            with self.subTest(field=candidate):
                self.assertNotIn(candidate, document_fields)
                self.assertNotIn(candidate, chunk_fields)

    def test_the_only_lifecycle_carrier_available_is_free_form_chunk_metadata(self):
        """`metadata` is an unvalidated JSON object beyond being a dict."""
        field = KnowledgeDocumentChunk._meta.get_field("metadata")
        self.assertEqual(field.get_internal_type(), "JSONField")

        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Metadata Shape")
        document = KnowledgeDocument.objects.create(
            collection=collection, title="Shape", status=KnowledgeDocument.Status.ACTIVE
        )
        chunk = KnowledgeDocumentChunk(
            document=document, chunk_index=1, section_title="S", content="body"
        )
        chunk.metadata = {"anything": ["at", "all"], "nested": {"is": "accepted"}}
        chunk.full_clean()
        chunk.save()

        chunk.refresh_from_db()
        self.assertEqual(chunk.metadata["nested"], {"is": "accepted"})

    def test_document_status_carries_no_retrieval_readiness_meaning(self):
        """ACTIVE is a business status. It does not imply retrievability.

        Re-stated here as a lifecycle fact because the future contract depends
        on it; the retrieval consequences were measured in Slice 3.
        """
        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Readiness")
        empty_active = KnowledgeDocument.objects.create(
            collection=collection,
            title="Active But Empty",
            curated_text="",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        empty_active.full_clean()
        self.assertEqual(empty_active.status, KnowledgeDocument.Status.ACTIVE)
        self.assertEqual(empty_active.chunks.count(), 0)
        self.assertEqual(empty_active.curated_text, "")
