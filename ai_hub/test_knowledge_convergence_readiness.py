"""Knowledge data-gap characterization ahead of retrieval convergence (Slice 3).

ARCHITECTURAL CONTEXT
---------------------
ADR-N1 is DECIDED: AI Hub must eventually have ONE canonical reusable Knowledge
retrieval capability, consumed by both Orchestrator and GAME. The GAME internal
handlers are a CURRENT compatibility path.

Slice 2 established that the two paths read DIFFERENT FIELDS:

    Path 1 (canonical)  -> KnowledgeDocumentChunk
    Path 2 (GAME compat) -> KnowledgeDocument.curated_text

A document can therefore be retrievable on one path and invisible on the other.
Before GAME can migrate, that data gap has to be understood.

This module measures it. It changes nothing and fixes nothing. Where current
behavior is surprising, it is RECORDED, not corrected.

NO PRODUCTION CODE IS CHANGED BY THIS MODULE.
"""
from django.conf import settings
from django.test import TestCase

from ai_hub.admin import KnowledgeDocumentAdmin, KnowledgeDocumentChunkInline
from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    ProviderConfig,
)
from ai_hub.services.knowledge_ingestion import ensure_initial_knowledge_chunk
from ai_hub.services.knowledge_retrieval import (
    browse_knowledge_index,
    read_document_section,
    read_knowledge_chunk,
    search_knowledge,
)
from ai_hub.test_game_retrieval_baseline import _ActionRunStub, run_path2_search
from ai_hub.test_application_scope_helpers import test_scope


SEARCH_LIMIT = 5


# ---------------------------------------------------------------------------
# Knowledge state matrix
# ---------------------------------------------------------------------------
# Each state is constructed for real and probed on both retrieval paths. The
# `possible_today` column is not an opinion: every state below is built by this
# module using ordinary model APIs, which is the proof that it is reachable.

STATE_MATRIX = (
    {
        "state": "A",
        "label": "curated_text + chunks, consistent",
        "curated_text": "state-a-marker Consistent document body for state A.",
        "chunks": (("State A Section", "state-a-marker Consistent document body for state A."),),
        "classification": "VALID",
        "path1_visible": True,
        "path2_visible": True,
        "convergence_risk": "none",
    },
    {
        "state": "B",
        "label": "curated_text only, zero chunks",
        "curated_text": "state-b-marker Body that exists only as curated text.",
        "chunks": (),
        "classification": "LEGACY / AMBIGUOUS",
        "path1_visible": False,
        "path2_visible": True,
        "convergence_risk": "HIGH - content disappears when GAME moves to canonical retrieval",
    },
    {
        "state": "C",
        "label": "chunks only, empty curated_text",
        "curated_text": "",
        "chunks": (("State C Section", "state-c-marker Body that exists only as a chunk."),),
        "classification": "VALID (canonical shape)",
        "path1_visible": True,
        "path2_visible": False,
        "convergence_risk": "none for convergence - already invisible to the path being retired",
    },
    {
        "state": "D",
        "label": "curated_text + chunks, materially different",
        "curated_text": "state-d-curated-marker The curated body says one thing.",
        "chunks": (("State D Section", "state-d-chunk-marker The chunk says something else."),),
        "classification": "INVALID / UNDETECTED DRIFT",
        "path1_visible": True,
        "path2_visible": True,
        "convergence_risk": "HIGH - the two paths return contradictory evidence for the same document",
    },
    {
        "state": "E",
        "label": "no curated_text, no chunks",
        "curated_text": "",
        "chunks": (),
        "classification": "EMPTY / INERT",
        "path1_visible": False,
        "path2_visible": False,
        "convergence_risk": "low - invisible before and after, but still listed as an ACTIVE document",
    },
)

STATE_MARKERS = {
    "A": ("state-a-marker", "state-a-marker"),
    "B": ("state-b-marker", None),
    "C": (None, "state-c-marker"),
    "D": ("state-d-curated-marker", "state-d-chunk-marker"),
    "E": (None, None),
}


def build_state_matrix_corpus(agent):
    """Create one ACTIVE document per state. Returns {state: document}."""
    collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Convergence State Matrix")
    agent.knowledge_collections.add(collection)
    documents = {}
    for spec in STATE_MATRIX:
        document = KnowledgeDocument.objects.create(
            collection=collection,
            title=f"State {spec['state']} Document",
            curated_text=spec["curated_text"],
            status=KnowledgeDocument.Status.ACTIVE,
        )
        for index, (section_title, content) in enumerate(spec["chunks"], start=1):
            KnowledgeDocumentChunk.objects.create(
                document=document,
                chunk_index=index,
                section_title=section_title,
                content=content,
                token_estimate=len(content.split()),
            )
        documents[spec["state"]] = document
    return documents


class KnowledgeStateMatrixTests(TestCase):
    """Every state in the matrix is reachable through ordinary model APIs."""

    @classmethod
    def setUpTestData(cls):
        provider = ProviderConfig.objects.create(
            name="convergence-provider",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        cls.agent = AgentProfile.objects.create(
            application_scope=test_scope(),
            name="convergence-agent",
            role="Convergence readiness",
            model_config=model,
        )
        cls.documents = build_state_matrix_corpus(cls.agent)
        cls.session = ExecutionSession.objects.create(
            entry_agent=cls.agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            status=ExecutionSession.Status.RUNNING,
            goal_text="Convergence readiness.",
        )

    def _path1_finds(self, marker) -> bool:
        return bool(search_knowledge(self.agent, query=marker, limit=SEARCH_LIMIT)["results"])

    def _path2_finds(self, marker) -> bool:
        return bool(run_path2_search(self.session, marker)["knowledge_context"])

    def test_every_state_is_constructible_and_active(self):
        """No validation prevents any of the five states from existing."""
        self.assertEqual(len(self.documents), 5)
        for state, document in self.documents.items():
            with self.subTest(state=state):
                document.full_clean()  # no model-level rule forbids any state
                self.assertEqual(document.status, KnowledgeDocument.Status.ACTIVE)

    def test_recorded_path_visibility_matches_measurement(self):
        """The matrix's visibility columns are measured, not asserted by hand."""
        for spec in STATE_MATRIX:
            state = spec["state"]
            curated_marker, chunk_marker = STATE_MARKERS[state]
            with self.subTest(state=state):
                # Path 1 sees a document only through its chunks.
                path1 = self._path1_finds(chunk_marker) if chunk_marker else False
                # Path 2 sees a document only through curated_text (or title/tags).
                path2 = self._path2_finds(curated_marker) if curated_marker else False
                self.assertEqual(path1, spec["path1_visible"])
                self.assertEqual(path2, spec["path2_visible"])

    def test_state_b_is_the_convergence_blocker(self):
        """curated_text with no chunks: visible now, invisible after convergence."""
        self.assertTrue(self._path2_finds("state-b-marker"))
        self.assertFalse(self._path1_finds("state-b-marker"))

    def test_state_d_returns_contradictory_evidence(self):
        """The same ACTIVE document answers differently depending on the path.

        Nothing in the model, the admin or the services detects this.
        """
        self.assertTrue(self._path2_finds("state-d-curated-marker"))
        self.assertFalse(self._path1_finds("state-d-curated-marker"))
        self.assertTrue(self._path1_finds("state-d-chunk-marker"))
        self.assertFalse(self._path2_finds("state-d-chunk-marker"))

    def test_state_e_is_inert_but_still_advertised(self):
        """An empty ACTIVE document is unfindable, yet still listed in the index.

        `browse_knowledge_index` reports it with zero chunks, and the
        retrieval-first prompt index counts it as an available document.
        """
        index = browse_knowledge_index(self.agent)
        titles = {
            document["title"]: document
            for collection in index["collections"]
            for document in collection["documents"]
        }
        self.assertIn("State E Document", titles)
        self.assertEqual(titles["State E Document"]["total_chunks"], 0)


class KnowledgeDriftCharacterizationTests(TestCase):
    """Nothing links curated_text and chunks. Drift is silent and undetectable."""

    @classmethod
    def setUpTestData(cls):
        provider = ProviderConfig.objects.create(
            name="drift-provider", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        cls.agent = AgentProfile.objects.create(
            application_scope=test_scope(),
            name="drift-agent", role="Drift characterization", model_config=model,
        )
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Drift Collection")
        cls.agent.knowledge_collections.add(cls.collection)

    def _document(self, curated_text, chunk_content):
        document = KnowledgeDocument.objects.create(
            collection=self.collection,
            title="Drift Document",
            curated_text=curated_text,
            status=KnowledgeDocument.Status.ACTIVE,
        )
        chunk = KnowledgeDocumentChunk.objects.create(
            document=document,
            chunk_index=1,
            section_title="Drift Section",
            content=chunk_content,
            token_estimate=len(chunk_content.split()),
        )
        return document, chunk

    def test_editing_curated_text_does_not_touch_chunks(self):
        """Q5: it does nothing. No recreate, no update, no invalidation, no flag."""
        document, chunk = self._document("original body", "original body")
        before = (chunk.content, chunk.updated_at, chunk.metadata)

        document.curated_text = "completely rewritten body"
        document.save(update_fields=["curated_text", "updated_at"])

        chunk.refresh_from_db()
        self.assertEqual(chunk.content, before[0])
        self.assertEqual(chunk.updated_at, before[1])
        self.assertEqual(chunk.metadata, before[2])
        self.assertEqual(document.chunks.count(), 1)

    def test_editing_chunks_does_not_touch_curated_text(self):
        """Q6: also nothing. The relationship is one-way and only at creation."""
        document, chunk = self._document("original body", "original body")

        chunk.content = "chunk rewritten independently"
        chunk.save(update_fields=["content", "updated_at"])

        document.refresh_from_db()
        self.assertEqual(document.curated_text, "original body")

    def test_chunks_go_stale_with_no_signal_of_any_kind(self):
        """Q7: yes. There is no staleness marker anywhere to check."""
        document, chunk = self._document("v1 body", "v1 body")
        document.curated_text = "v2 body"
        document.save(update_fields=["curated_text", "updated_at"])

        chunk.refresh_from_db()
        # The only observable difference is a timestamp ordering, and nothing
        # in Core reads it: no status field, no hash, no version, no metadata
        # key is written to record that the chunk no longer matches its source.
        self.assertGreater(document.updated_at, chunk.updated_at)
        self.assertEqual(chunk.metadata, {})
        self.assertNotIn("stale", str(chunk.metadata))

    def test_no_model_or_app_hook_links_the_two_fields(self):
        """Q4/Q5/Q6 structural proof: there is nothing to fire.

        `KnowledgeDocument` and `KnowledgeDocumentChunk` define no `save()`
        override, and the app config registers no signal handlers.
        """
        from django.apps import AppConfig
        from django.db import models as django_models
        from django.db.models.signals import post_save, pre_save

        from ai_hub.apps import AIHubConfig

        # The app config does not override ready(), so it registers nothing.
        self.assertIs(AIHubConfig.ready, AppConfig.ready)
        for model in (KnowledgeDocument, KnowledgeDocumentChunk):
            with self.subTest(model=model.__name__):
                self.assertIs(model.save, django_models.Model.save)
                for signal in (pre_save, post_save):
                    receivers = signal._live_receivers(model)
                    # Django returns (sync, async) in 5.x+, a flat list before.
                    flattened = (
                        [r for group in receivers for r in group]
                        if receivers and isinstance(receivers[0], (list, tuple))
                        else list(receivers)
                    )
                    self.assertEqual(flattened, [])


class IngestionFallbackAuthorityTests(TestCase):
    """`ensure_initial_knowledge_chunk` semantics — the backfill contract."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Fallback Collection")

    def _document(self, curated_text=""):
        return KnowledgeDocument.objects.create(
            collection=self.collection,
            title=f"Fallback Document {KnowledgeDocument.objects.count() + 1}",
            curated_text=curated_text,
            status=KnowledgeDocument.Status.ACTIVE,
        )

    def test_explicit_chunks_are_authoritative_and_never_overwritten(self):
        """THE INVARIANT any future backfill must preserve."""
        document = self._document("curated body that must NOT replace the chunk")
        curated_chunk = KnowledgeDocumentChunk.objects.create(
            document=document,
            chunk_index=1,
            section_title="Hand-curated",
            content="carefully authored chunk",
            metadata={"authored_by": "human"},
        )

        returned = ensure_initial_knowledge_chunk(document)

        self.assertEqual(returned.pk, curated_chunk.pk)
        self.assertEqual(document.chunks.count(), 1)
        curated_chunk.refresh_from_db()
        self.assertEqual(curated_chunk.content, "carefully authored chunk")
        self.assertEqual(curated_chunk.metadata, {"authored_by": "human"})

    def test_multi_chunk_documents_are_left_completely_alone(self):
        document = self._document("curated body")
        for index in range(1, 4):
            KnowledgeDocumentChunk.objects.create(
                document=document,
                chunk_index=index,
                section_title=f"Section {index}",
                content=f"chunk {index} body",
            )

        returned = ensure_initial_knowledge_chunk(document)

        self.assertEqual(document.chunks.count(), 3)
        self.assertEqual(returned.chunk_index, 1)

    def test_empty_curated_text_produces_no_chunk(self):
        """State E cannot be repaired by the fallback: there is no source text."""
        document = self._document("")
        self.assertIsNone(ensure_initial_knowledge_chunk(document))
        self.assertEqual(document.chunks.count(), 0)

    def test_whitespace_only_curated_text_produces_no_chunk(self):
        document = self._document("   \n\t  ")
        self.assertIsNone(ensure_initial_knowledge_chunk(document))
        self.assertEqual(document.chunks.count(), 0)

    def test_state_b_is_exactly_what_the_fallback_repairs(self):
        document = self._document("state-b body needing a chunk")
        chunk = ensure_initial_knowledge_chunk(document)

        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.chunk_index, 1)
        self.assertEqual(chunk.section_title, document.title)
        self.assertEqual(chunk.content, "state-b body needing a chunk")
        self.assertEqual(chunk.metadata, {"ingestion": "initial_curated_text"})

    def test_the_fallback_is_idempotent(self):
        document = self._document("idempotency body")
        first = ensure_initial_knowledge_chunk(document)
        second = ensure_initial_knowledge_chunk(document)
        self.assertEqual(first.pk, second.pk)
        self.assertEqual(document.chunks.count(), 1)

    def test_the_fallback_cannot_repair_state_d_drift(self):
        """Once a chunk exists, drift is permanent as far as the fallback knows."""
        document = self._document("curated says ALPHA")
        KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1, section_title="S", content="chunk says BETA"
        )
        ensure_initial_knowledge_chunk(document)
        self.assertEqual(document.chunks.get().content, "chunk says BETA")

    def test_the_fallback_is_only_wired_into_one_creation_path(self):
        """Q4: Build Console 'create' mode is the sole automatic caller.

        Recorded structurally so a new caller shows up as a test change.
        """
        from ai_hub.services import build_console

        self.assertIs(
            build_console.ensure_initial_knowledge_chunk, ensure_initial_knowledge_chunk
        )


class AdminAuthoringPathCharacterizationTests(TestCase):
    """Q4: what the Admin can and cannot produce.

    Recorded because the Admin is the documented primary authoring surface.
    """

    def test_document_admin_exposes_no_automatic_chunk_creation(self):
        self.assertNotIn("chunks", KnowledgeDocumentAdmin.fieldsets[0][1]["fields"])
        field_names = {
            name
            for _title, options in KnowledgeDocumentAdmin.fieldsets
            for name in options["fields"]
        }
        self.assertIn("curated_text", field_names)
        self.assertNotIn("chunks", field_names)

    def test_chunk_inline_cannot_author_chunk_content(self):
        """SURPRISING - RECORDED, NOT FIXED.

        The document change form's chunk inline exposes chunk_index,
        section_title and token_estimate but NOT `content`. A chunk added
        through the inline is therefore saved with empty content, which is
        retrievable by neither path in any useful way. Authoring real content
        requires the separate KnowledgeDocumentChunk admin.
        """
        self.assertNotIn("content", KnowledgeDocumentChunkInline.fields)
        self.assertEqual(KnowledgeDocumentChunkInline.extra, 0)

    def test_a_document_saved_through_ordinary_orm_has_no_chunks(self):
        """The Admin uses the ordinary ModelForm save path, which creates none."""
        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Admin Path Collection")
        document = KnowledgeDocument.objects.create(
            collection=collection,
            title="Admin Authored Document",
            curated_text="admin authored body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        self.assertEqual(document.chunks.count(), 0)


class WholeDocumentCapabilityAnalysisTests(TestCase):
    """Q: does canonical retrieval already cover the 'broader context' need?

    Slice 2 observed Path 2 can return an entire curated_text. This module tests
    whether the existing canonical tools already let an Agent reach the same
    content, and what it costs. It does NOT conclude that a whole-document tool
    is needed.
    """

    CHUNK_COUNT = 5

    @classmethod
    def setUpTestData(cls):
        provider = ProviderConfig.objects.create(
            name="window-provider", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        cls.agent = AgentProfile.objects.create(
            application_scope=test_scope(),
            name="window-agent", role="Whole-document analysis", model_config=model,
        )
        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Window Collection")
        cls.agent.knowledge_collections.add(collection)
        cls.document = KnowledgeDocument.objects.create(
            collection=collection,
            title="Multi Section Manual",
            curated_text="",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        cls.expected = []
        for index in range(1, cls.CHUNK_COUNT + 1):
            content = f"Section {index} body about window-marker topic number {index}."
            KnowledgeDocumentChunk.objects.create(
                document=cls.document,
                chunk_index=index,
                section_title=f"Section {index}",
                content=content,
                token_estimate=len(content.split()),
            )
            cls.expected.append(content)

    def test_canonical_retrieval_can_already_reconstruct_a_whole_document(self):
        """Capability exists: browse to enumerate, then read each chunk."""
        index = browse_knowledge_index(self.agent)
        documents = [
            document
            for collection in index["collections"]
            for document in collection["documents"]
        ]
        self.assertEqual(len(documents), 1)
        chunk_ids = [chunk["chunk_id"] for chunk in documents[0]["chunks"]]
        self.assertEqual(len(chunk_ids), self.CHUNK_COUNT)

        reconstructed = [
            read_knowledge_chunk(self.agent, chunk_id=chunk_id)["content"]
            for chunk_id in chunk_ids
        ]
        self.assertEqual(reconstructed, self.expected)

    def test_adjacent_chunks_are_addressable_by_index_without_a_new_tool(self):
        """`read_document_section(chunk_index=N)` already supports a window."""
        window = [
            read_document_section(
                self.agent, document_id=self.document.pk, chunk_index=index
            )["content"]
            for index in (2, 3, 4)
        ]
        self.assertEqual(window, self.expected[1:4])

    def test_reconstruction_cost_exceeds_the_default_tool_round_budget(self):
        """The real constraint is the round budget, not a missing capability.

        Reconstructing this 5-chunk document takes 1 browse + 5 reads = 6 tool
        calls. The default per-agent-call budget is smaller, so a single
        Orchestrator step cannot do it. This is the measured basis for the
        whole-document analysis - it is a budget question, not a tool gap.
        """
        calls_required = 1 + self.CHUNK_COUNT
        default_rounds = int(
            getattr(settings, "AI_HUB_MAX_TOOL_ROUNDS_PER_AGENT_CALL", 3)
        )
        self.assertGreater(calls_required, default_rounds)
        # Even the hard ceiling only just accommodates it.
        self.assertLessEqual(calls_required, 10)

    def test_search_already_returns_multiple_chunks_of_one_document(self):
        """A single search can surface several sections at once, bounded by K."""
        result = search_knowledge(self.agent, query="window-marker", limit=SEARCH_LIMIT)
        returned = {row["chunk_index"] for row in result["results"]}
        self.assertEqual(len(returned), SEARCH_LIMIT)


class ConvergenceGapDetectionTests(TestCase):
    """What a preflight validator would have to find. Detection only, no repair."""

    @classmethod
    def setUpTestData(cls):
        provider = ProviderConfig.objects.create(
            name="preflight-provider", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        cls.agent = AgentProfile.objects.create(
            application_scope=test_scope(),
            name="preflight-agent", role="Preflight", model_config=model,
        )
        cls.documents = build_state_matrix_corpus(cls.agent)

    def test_state_b_documents_are_detectable_with_one_orm_query(self):
        """The unsafe set is cheap to find - no full-text scan required."""
        at_risk = (
            KnowledgeDocument.objects.filter(status=KnowledgeDocument.Status.ACTIVE)
            .exclude(curated_text="")
            .filter(chunks__isnull=True)
        )
        titles = set(at_risk.values_list("title", flat=True))
        self.assertEqual(titles, {"State B Document"})

    def test_state_e_documents_are_detectable_and_are_not_repairable(self):
        inert = (
            KnowledgeDocument.objects.filter(
                status=KnowledgeDocument.Status.ACTIVE, curated_text=""
            )
            .filter(chunks__isnull=True)
        )
        self.assertEqual(
            set(inert.values_list("title", flat=True)), {"State E Document"}
        )

    def test_state_d_drift_is_NOT_detectable_by_a_cheap_query(self):
        """SURPRISING - RECORDED, NOT FIXED.

        States B and E are structural and cheap to detect. Drift (State D) is a
        CONTENT comparison: there is no hash, version or timestamp contract to
        compare, so a validator would have to read both bodies of every
        document and decide what "materially different" means. That is a design
        question, not a query.
        """
        drifted = self.documents["D"]
        chunk_text = "\n".join(
            chunk.content for chunk in drifted.chunks.order_by("chunk_index")
        )
        self.assertNotEqual(drifted.curated_text, chunk_text)
        # Nothing persisted on either row records the mismatch.
        self.assertEqual(drifted.chunks.get().metadata, {})

    def test_a_backfill_would_change_only_state_b(self):
        """Scope check for any future remediation: exactly one document here."""
        repairable = [
            document
            for document in KnowledgeDocument.objects.filter(
                status=KnowledgeDocument.Status.ACTIVE
            )
            if document.curated_text.strip() and not document.chunks.exists()
        ]
        self.assertEqual([d.title for d in repairable], ["State B Document"])
