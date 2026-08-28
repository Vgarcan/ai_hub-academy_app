"""Tests for narrow governed Knowledge repair decisions (Slice 12, RC-002 3e).

Two human decisions resolving `DERIVED_CHUNKS_MODIFIED` in opposite directions,
plus the authority transfer for `DERIVED_PROVENANCE_INCOMPLETE`.

Weighted toward what these verbs REFUSE, what they leave untouched, and what
they deliberately do NOT normalize — a lifecycle inconsistency can be the most
truthful description available.
"""
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from django.db import close_old_connections, connection
from django.test import TestCase, TransactionTestCase

from ai_hub.models import (
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    KnowledgeLifecycleEvent,
    ToolDefinition,
)
from ai_hub.services.knowledge_ingestion import ensure_initial_knowledge_chunk
from ai_hub.services.knowledge_lifecycle import (
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK,
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION,
    document_chunk_set_fingerprint,
)
from ai_hub.services.knowledge_mutation import (
    KnowledgeMutationConflict,
    KnowledgeMutationPrincipal,
    build_snapshot,
)
from ai_hub.services.knowledge_preflight import run_knowledge_preflight
from ai_hub.services.knowledge_regeneration import (
    ChunkSetModifiedError,
    GeneratorVersionAheadError,
    IncompleteProvenanceError,
    InvalidCandidateError,
    UnsupportedGeneratorForRegenerationError,
    regenerate_derived_chunk_set,
)
from ai_hub.services.knowledge_repair import (
    OPERATION_ACCEPT_CURRENT_CHUNKS_AS_EXPLICIT,
    OPERATION_DISCARD_MODIFIED_CHUNKS_AND_REGENERATE,
    ChunkSetNotModifiedError,
    IneligibleLifecycleStateError,
    KnowledgeRepairError,
    MissingChunkSetForExplicitError,
    MissingReasonCodeError,
    accept_current_chunks_as_explicit,
    discard_modified_chunks_and_regenerate,
)
from ai_hub.services.starter_toolboxes import seed_starter_toolboxes
from ai_hub.test_knowledge_regeneration import (
    _EnteredGovernedMutation,
    a_principal,
    chunk_rows,
    expected_for,
    knowledge_snapshot,
    make_derived,
    preflight_row,
    record_generation,
)

MODES = KnowledgeDocument.ChunkAuthorityMode
WHY = "operator_reviewed"


class _EnteredRepairMutation(_EnteredGovernedMutation):
    """Slice 11's spy, retargeted at THIS module's imported reference.

    `knowledge_repair` imports `_governed_knowledge_mutation` into its own
    namespace, so patching `knowledge_regeneration` would not intercept it and
    the entry assertion would silently never fire. Subclassed rather than
    parameterised so no Slice 11 test file changes.
    """

    def patch(self):
        import contextlib

        from ai_hub.services import knowledge_repair

        real = knowledge_repair._governed_knowledge_mutation

        @contextlib.contextmanager
        def wrapper(*args, **kwargs):
            with real(*args, **kwargs) as mutation:
                self.entered = True
                yield mutation

        return mock.patch.object(
            knowledge_repair, "_governed_knowledge_mutation", wrapper
        )


def make_modified(collection, title="Modified", *, curated_text="derived body",
                  edit="a human edited this outside governance"):
    """A DERIVED document whose chunks were edited outside the boundary."""
    document = make_derived(collection, title, curated_text=curated_text)
    chunk = document.chunks.get()
    chunk.content = edit
    chunk.save(update_fields=["content"])
    document.refresh_from_db()
    return document


def make_provenance_incomplete(collection, title="Incomplete", *, field="generation_chunk_set_fingerprint"):
    document = make_derived(collection, title)
    KnowledgeDocument.objects.filter(pk=document.pk).update(
        **{field: None if field == "generator_version" else ""}
    )
    document.refresh_from_db()
    return document


# ---------------------------------------------------------------------------
# Accept current chunks as EXPLICIT
# ---------------------------------------------------------------------------

class AcceptCurrentChunksAsExplicitTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Accept")

    def test_modified_chunks_can_be_blessed(self):
        document = make_modified(self.collection)
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "DERIVED_CHUNKS_MODIFIED")

        result = accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )

        self.assertEqual(result.chunk_authority_mode, MODES.EXPLICIT)
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "EXPLICIT_AUTHORITY")

    def test_provenance_incomplete_can_be_blessed(self):
        for field in (
            "generation_input_fingerprint",
            "generation_chunk_set_fingerprint",
            "generator_identity",
            "generator_version",
        ):
            with self.subTest(missing=field):
                document = make_provenance_incomplete(
                    self.collection, f"Incomplete {field}", field=field
                )
                self.assertEqual(
                    preflight_row(document.pk)["lifecycle_state"],
                    "DERIVED_PROVENANCE_INCOMPLETE",
                )
                result = accept_current_chunks_as_explicit(
                    document.pk, expected=expected_for(document),
                    principal=a_principal(), reason_code=WHY,
                )
                self.assertEqual(result.chunk_authority_mode, MODES.EXPLICIT)

    def test_not_one_chunk_row_is_touched(self):
        document = make_modified(self.collection)
        before = chunk_rows(document)

        accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )

        after = chunk_rows(document)
        self.assertEqual(after, before)
        self.assertEqual([r["pk"] for r in after], [r["pk"] for r in before])

    def test_all_generation_provenance_is_cleared(self):
        document = make_modified(self.collection)
        result = accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )
        self.assertEqual(result.generation_input_fingerprint, "")
        self.assertEqual(result.generation_chunk_set_fingerprint, "")
        self.assertEqual(result.generator_identity, "")
        self.assertIsNone(result.generator_version)

    def test_it_does_not_create_the_kp010_it_declines_to_repair(self):
        """EXPLICIT carrying provenance would itself be an inconsistency."""
        document = make_modified(self.collection)
        accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )
        row = preflight_row(document.pk)
        self.assertNotIn("KP010_LIFECYCLE_FACT_INCONSISTENCY", row["issues"])

    def test_the_source_and_document_fields_are_untouched(self):
        document = make_modified(self.collection)
        before = KnowledgeDocument.objects.filter(pk=document.pk).values(
            "title", "curated_text", "source_file", "status", "collection_id", "notes", "tags"
        )[0]
        accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )
        after = KnowledgeDocument.objects.filter(pk=document.pk).values(
            "title", "curated_text", "source_file", "status", "collection_id", "notes", "tags"
        )[0]
        self.assertEqual(before, after)

    def test_curated_text_divergence_is_irrelevant_afterwards(self):
        """D-L5: EXPLICIT chunks differing from `curated_text` is not staleness."""
        document = make_modified(self.collection, edit="nothing like the source")
        accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "EXPLICIT_AUTHORITY")

    def test_unusable_chunks_do_not_block_the_authority_decision(self):
        """Authority and structural readiness remain orthogonal (Slice 8)."""
        document = make_modified(self.collection, edit="   ")
        result = accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )
        self.assertEqual(result.chunk_authority_mode, MODES.EXPLICIT)
        row = preflight_row(document.pk)
        self.assertEqual(row["structural_state"], "UNUSABLE_CHUNKS")
        self.assertFalse(row["canonically_retrievable"])


class AcceptExplicitRefusalTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Accept Refusals")

    def _assert_refused(self, document, exception):
        before = knowledge_snapshot()
        before_chunks = chunk_rows(document)
        with self.assertRaises(exception) as caught:
            accept_current_chunks_as_explicit(
                document.pk, expected=expected_for(document),
                principal=a_principal(), reason_code=WHY,
            )
        self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(chunk_rows(document), before_chunks)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)
        return caught.exception

    def test_a_document_with_no_chunks_is_refused(self):
        """EXPLICIT means the chunk set IS the artifact; with none it is vacuous."""
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Chunkless", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        record_generation(document)
        # Recorded empty set, then made "modified" by asserting a different one.
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            generation_chunk_set_fingerprint="c1:" + "0" * 64
        )
        document.refresh_from_db()
        self._assert_refused(document, MissingChunkSetForExplicitError)

    def test_unknown_authority_is_refused(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Unknown", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        ensure_initial_knowledge_chunk(document)
        error = self._assert_refused(document, IneligibleLifecycleStateError)
        self.assertIn("adjudication", str(error))
        self.assertEqual(error.reason, IneligibleLifecycleStateError.REASON_NOT_DERIVED)

    def test_explicit_authority_is_refused(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Explicit", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
            chunk_authority_mode=MODES.EXPLICIT,
        )
        KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1, section_title="H", content="authored",
        )
        self._assert_refused(document, IneligibleLifecycleStateError)

    def test_a_verifiable_derived_claim_is_refused(self):
        """`DERIVED_CURRENT`, `DERIVED_INPUT_CHANGED` and `GENERATOR_OUTDATED`
        all still verify — there is no authority ambiguity to resolve."""
        current = make_derived(self.collection, "Current")
        self._assert_refused(current, IneligibleLifecycleStateError)

        changed = make_derived(self.collection, "Input Changed")
        KnowledgeDocument.objects.filter(pk=changed.pk).update(curated_text="moved on")
        changed.refresh_from_db()
        self.assertEqual(preflight_row(changed.pk)["lifecycle_state"], "DERIVED_INPUT_CHANGED")
        self._assert_refused(changed, IneligibleLifecycleStateError)

        outdated = make_derived(
            self.collection, "Outdated",
            version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION - 1,
        )
        self.assertEqual(preflight_row(outdated.pk)["lifecycle_state"], "GENERATOR_OUTDATED")
        self._assert_refused(outdated, IneligibleLifecycleStateError)

    def test_capability_defects_get_no_exit(self):
        """Unsupported / version-ahead must be fixed by upgrading Core, never by
        erasing truthful provenance."""
        unsupported = make_derived(self.collection, "Unsupported")
        KnowledgeDocument.objects.filter(pk=unsupported.pk).update(
            generator_identity="some_future_generator"
        )
        unsupported.refresh_from_db()
        self.assertEqual(
            preflight_row(unsupported.pk)["lifecycle_state"], "DERIVED_GENERATOR_UNSUPPORTED"
        )
        error = self._assert_refused(unsupported, IneligibleLifecycleStateError)
        self.assertIn("upgrade Core", str(error))
        self.assertEqual(
            error.reason, IneligibleLifecycleStateError.REASON_CAPABILITY_DEFECT
        )

        ahead = make_derived(
            self.collection, "Ahead",
            version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION + 1,
        )
        self.assertEqual(
            preflight_row(ahead.pk)["lifecycle_state"], "DERIVED_GENERATOR_VERSION_AHEAD"
        )
        error = self._assert_refused(ahead, IneligibleLifecycleStateError)
        self.assertEqual(
            error.reason, IneligibleLifecycleStateError.REASON_CAPABILITY_DEFECT
        )

    def test_an_empty_reason_code_is_refused(self):
        document = make_modified(self.collection)
        for bad in ("", "   ", None):
            with self.subTest(reason_code=repr(bad)):
                before = knowledge_snapshot()
                with self.assertRaises(MissingReasonCodeError):
                    accept_current_chunks_as_explicit(
                        document.pk, expected=expected_for(document),
                        principal=a_principal(), reason_code=bad,
                    )
                self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)


# ---------------------------------------------------------------------------
# Discard modified chunks and regenerate
# ---------------------------------------------------------------------------

class DiscardModifiedChunksTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Discard")

    def test_modified_chunks_are_discarded_and_regenerated(self):
        document = make_modified(self.collection, curated_text="the true source body")
        before_pks = {r["pk"] for r in chunk_rows(document)}

        result = discard_modified_chunks_and_regenerate(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code="accidental_edit",
        )

        self.assertEqual(result.chunk_authority_mode, MODES.DERIVED)
        chunk = document.chunks.get()
        self.assertEqual(chunk.content, "the true source body")
        self.assertNotIn(chunk.pk, before_pks)
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "DERIVED_CURRENT")

    def test_provenance_is_rewritten_to_the_new_truth(self):
        document = make_modified(self.collection)
        discard_modified_chunks_and_regenerate(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code="accidental_edit",
        )
        document.refresh_from_db()
        snapshot = build_snapshot(document)
        self.assertEqual(
            document.generation_chunk_set_fingerprint, snapshot.observed_chunk_set_fingerprint
        )
        self.assertEqual(
            document.generation_input_fingerprint, snapshot.observed_input_fingerprint
        )
        self.assertEqual(document.generator_identity, GENERATOR_CURATED_TEXT_SINGLE_CHUNK)
        self.assertEqual(
            document.generator_version, GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION
        )

    def test_the_candidate_equals_current_edge_case_churns_no_primary_key(self):
        """`DERIVED_CHUNKS_MODIFIED` means observed != RECORDED. It does NOT
        imply the modified chunks differ from what the generator produces NOW.

        Here an operator edited the chunk to exactly what the current source
        would generate. The lifecycle claim was stale, but there is nothing to
        destroy — so no row may be touched.
        """
        document = make_derived(self.collection, "Coincidence", curated_text="original")
        # Source moves, then someone hand-edits the chunk to match it exactly.
        KnowledgeDocument.objects.filter(pk=document.pk).update(curated_text="the new body")
        document.refresh_from_db()
        chunk = document.chunks.get()
        chunk.content = "the new body"
        chunk.save(update_fields=["content"])
        document.refresh_from_db()
        self.assertEqual(
            preflight_row(document.pk)["lifecycle_state"], "DERIVED_CHUNKS_MODIFIED"
        )

        before = chunk_rows(document)
        discard_modified_chunks_and_regenerate(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code="already_equals_generator",
        )

        self.assertEqual(chunk_rows(document), before, "no row may be touched")
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "DERIVED_CURRENT")
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 1)


class DiscardRefusalTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Discard Refusals")

    def _assert_refused(self, document, exception):
        before = knowledge_snapshot()
        before_chunks = chunk_rows(document)
        with self.assertRaises(exception) as caught:
            discard_modified_chunks_and_regenerate(
                document.pk, expected=expected_for(document),
                principal=a_principal(), reason_code=WHY,
            )
        self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(chunk_rows(document), before_chunks)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)
        return caught.exception

    def test_an_unmodified_derived_document_is_refused(self):
        document = make_derived(self.collection, "Current")
        error = self._assert_refused(document, ChunkSetNotModifiedError)
        self.assertIn("ordinary regeneration", str(error))

    def test_unknown_and_explicit_are_refused(self):
        unknown = KnowledgeDocument.objects.create(
            collection=self.collection, title="Unknown", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        ensure_initial_knowledge_chunk(unknown)
        self._assert_refused(unknown, IneligibleLifecycleStateError)

        explicit = KnowledgeDocument.objects.create(
            collection=self.collection, title="Explicit", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
            chunk_authority_mode=MODES.EXPLICIT,
        )
        KnowledgeDocumentChunk.objects.create(
            document=explicit, chunk_index=1, section_title="H", content="authored",
        )
        self._assert_refused(explicit, IneligibleLifecycleStateError)

    def test_incomplete_provenance_is_refused(self):
        """Nothing to regenerate INTO without the generator facts."""
        document = make_modified(self.collection, "Incomplete")
        KnowledgeDocument.objects.filter(pk=document.pk).update(generator_identity="")
        document.refresh_from_db()
        self._assert_refused(document, IncompleteProvenanceError)

    def test_an_unsupported_generator_is_refused(self):
        document = make_modified(self.collection, "Unsupported")
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            generator_identity="some_future_generator"
        )
        document.refresh_from_db()
        self._assert_refused(document, UnsupportedGeneratorForRegenerationError)

    def test_a_version_ahead_of_this_core_is_refused(self):
        document = make_modified(self.collection, "Ahead")
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            generator_version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION + 1
        )
        document.refresh_from_db()
        error = self._assert_refused(document, GeneratorVersionAheadError)
        self.assertIn("upgrade Core", str(error))

    def test_an_empty_reason_code_is_refused(self):
        document = make_modified(self.collection, "No Reason")
        for bad in ("", "   ", None):
            with self.subTest(reason_code=repr(bad)):
                before = knowledge_snapshot()
                with self.assertRaises(MissingReasonCodeError):
                    discard_modified_chunks_and_regenerate(
                        document.pk, expected=expected_for(document),
                        principal=a_principal(), reason_code=bad,
                    )
                self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_every_repair_refusal_shares_one_base_class(self):
        for error in (
            IneligibleLifecycleStateError, MissingReasonCodeError,
            MissingChunkSetForExplicitError, ChunkSetNotModifiedError,
        ):
            with self.subTest(error=error.__name__):
                self.assertTrue(issubclass(error, KnowledgeRepairError))


class DominantModifiedChunksOverCapabilityDefectTests(TestCase):
    """`DERIVED_CHUNKS_MODIFIED` is DOMINANT, and that dominance reaches repair.

    Once the current chunks no longer match the recorded generated set, the
    DERIVED claim has stopped describing the current artifact — whatever this
    Core makes of the historic generator. So an operator MAY accept the reviewed
    chunks as EXPLICIT even when the generator is unsupported or version-ahead.

    `discard_modified_chunks_and_regenerate` still refuses those documents,
    because it must CONSTRUCT a new artifact and cannot without a usable
    generator. The asymmetry is intentional and is the point of these tests.
    """

    CAPABILITY_DEFECTS = (
        ("unsupported generator", "generator_identity", "some_future_generator",
         UnsupportedGeneratorForRegenerationError),
        ("version ahead", "generator_version",
         GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION + 1, GeneratorVersionAheadError),
    )

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Dominance")

    def _modified_with_defect(self, label, field, value):
        document = make_modified(self.collection, label, curated_text="the true body")
        KnowledgeDocument.objects.filter(pk=document.pk).update(**{field: value})
        document.refresh_from_db()
        # Preflight agrees: chunks-modified dominates the capability defect.
        self.assertEqual(
            preflight_row(document.pk)["lifecycle_state"], "DERIVED_CHUNKS_MODIFIED"
        )
        return document

    def test_accept_succeeds_for_every_capability_defect_with_modified_chunks(self):
        for label, field, value, _ in self.CAPABILITY_DEFECTS:
            with self.subTest(defect=label):
                document = self._modified_with_defect(f"Accept {label}", field, value)
                before_chunks = chunk_rows(document)
                old_identity = document.generator_identity
                old_version = document.generator_version

                result = accept_current_chunks_as_explicit(
                    document.pk, expected=expected_for(document),
                    principal=a_principal(), reason_code="reviewed_and_owned",
                )

                # authority transferred, live provenance cleared
                self.assertEqual(result.chunk_authority_mode, MODES.EXPLICIT)
                self.assertEqual(result.generation_input_fingerprint, "")
                self.assertEqual(result.generation_chunk_set_fingerprint, "")
                self.assertEqual(result.generator_identity, "")
                self.assertIsNone(result.generator_version)

                # not one chunk row moved, PKs included
                after_chunks = chunk_rows(document)
                self.assertEqual(after_chunks, before_chunks)
                self.assertEqual(
                    [r["pk"] for r in after_chunks], [r["pk"] for r in before_chunks]
                )

                # exactly one event, and the historic generator facts survive in it
                event = KnowledgeLifecycleEvent.objects.get(document=document)
                self.assertEqual(
                    event.operation, OPERATION_ACCEPT_CURRENT_CHUNKS_AS_EXPLICIT
                )
                self.assertEqual(event.previous_generator_identity, old_identity)
                self.assertEqual(event.previous_generator_version, old_version)
                self.assertEqual(event.new_generator_identity, "")
                self.assertIsNone(event.new_generator_version)
                # observed pair equal == proof no chunk mutation occurred
                self.assertEqual(
                    event.previous_observed_chunk_set_fingerprint,
                    event.new_observed_chunk_set_fingerprint,
                )

                # and no new inconsistency was created
                self.assertNotIn(
                    "KP010_LIFECYCLE_FACT_INCONSISTENCY",
                    preflight_row(document.pk)["issues"],
                )

    def test_discard_still_refuses_every_capability_defect_with_modified_chunks(self):
        for label, field, value, expected_error in self.CAPABILITY_DEFECTS:
            with self.subTest(defect=label):
                document = self._modified_with_defect(f"Discard {label}", field, value)
                before_db = knowledge_snapshot()
                before_chunks = chunk_rows(document)

                with self.assertRaises(expected_error):
                    discard_modified_chunks_and_regenerate(
                        document.pk, expected=expected_for(document),
                        principal=a_principal(), reason_code="accidental_edit",
                    )

                self.assertEqual(knowledge_snapshot(), before_db)
                self.assertEqual(chunk_rows(document), before_chunks)
                self.assertEqual(
                    KnowledgeLifecycleEvent.objects.filter(document=document).count(), 0
                )

    def test_the_same_document_is_accepted_by_one_verb_and_refused_by_the_other(self):
        """The asymmetry stated as a single property."""
        document = self._modified_with_defect(
            "Asymmetry", "generator_identity", "some_future_generator"
        )
        with self.assertRaises(UnsupportedGeneratorForRegenerationError):
            discard_modified_chunks_and_regenerate(
                document.pk, expected=expected_for(document),
                principal=a_principal(), reason_code="accidental_edit",
            )
        result = accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code="reviewed_and_owned",
        )
        self.assertEqual(result.chunk_authority_mode, MODES.EXPLICIT)


class IneligibilityReasonContractTests(TestCase):
    """`reason` is the machine-readable half of the refusal.

    Slice 13 will consume these verbs and must decide what to offer an operator
    next without parsing English.
    """

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Reasons")

    def _reason_from_accept(self, document):
        with self.assertRaises(IneligibleLifecycleStateError) as caught:
            accept_current_chunks_as_explicit(
                document.pk, expected=expected_for(document),
                principal=a_principal(), reason_code=WHY,
            )
        return caught.exception.reason

    def test_the_vocabulary_is_small_and_closed(self):
        self.assertEqual(
            IneligibleLifecycleStateError.REASONS,
            frozenset({"not_derived", "claim_verifies", "capability_defect"}),
        )

    def test_an_unknown_reason_cannot_be_constructed(self):
        with self.assertRaises(ValueError):
            IneligibleLifecycleStateError("x", reason="something_invented")

    def test_not_derived_for_unknown_and_explicit(self):
        unknown = KnowledgeDocument.objects.create(
            collection=self.collection, title="U", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        ensure_initial_knowledge_chunk(unknown)
        self.assertEqual(self._reason_from_accept(unknown), "not_derived")

        explicit = KnowledgeDocument.objects.create(
            collection=self.collection, title="E", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
            chunk_authority_mode=MODES.EXPLICIT,
        )
        KnowledgeDocumentChunk.objects.create(
            document=explicit, chunk_index=1, section_title="H", content="authored",
        )
        self.assertEqual(self._reason_from_accept(explicit), "not_derived")

    def test_claim_verifies_when_the_derived_claim_still_checks_out(self):
        for label, mutate in (
            ("current", lambda d: None),
            ("input changed", lambda d: KnowledgeDocument.objects.filter(pk=d.pk).update(
                curated_text="moved on")),
            ("generator outdated", None),
        ):
            with self.subTest(state=label):
                if label == "generator outdated":
                    document = make_derived(
                        self.collection, f"CV {label}",
                        version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION - 1,
                    )
                else:
                    document = make_derived(self.collection, f"CV {label}")
                    mutate(document)
                    document.refresh_from_db()
                self.assertEqual(self._reason_from_accept(document), "claim_verifies")

    def test_capability_defect_only_when_the_chunks_are_clean(self):
        for label, field, value in (
            ("unsupported", "generator_identity", "some_future_generator"),
            ("ahead", "generator_version", GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION + 1),
        ):
            with self.subTest(defect=label):
                document = make_derived(self.collection, f"CD {label}")
                KnowledgeDocument.objects.filter(pk=document.pk).update(**{field: value})
                document.refresh_from_db()
                self.assertEqual(self._reason_from_accept(document), "capability_defect")

    def test_capability_defect_is_never_reported_when_chunks_are_modified(self):
        """The dominance rule, stated as a property of the reason vocabulary."""
        document = make_modified(self.collection, "Dominant")
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            generator_identity="some_future_generator"
        )
        document.refresh_from_db()
        # No exception at all — the document is eligible.
        result = accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )
        self.assertEqual(result.chunk_authority_mode, MODES.EXPLICIT)

    def test_every_reason_in_the_vocabulary_is_actually_reachable(self):
        """A vocabulary value nothing can raise would be dead contract."""
        reached = set()

        unknown = KnowledgeDocument.objects.create(
            collection=self.collection, title="R1", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        ensure_initial_knowledge_chunk(unknown)
        reached.add(self._reason_from_accept(unknown))

        reached.add(self._reason_from_accept(make_derived(self.collection, "R2")))

        defect = make_derived(self.collection, "R3")
        KnowledgeDocument.objects.filter(pk=defect.pk).update(
            generator_identity="some_future_generator"
        )
        defect.refresh_from_db()
        reached.add(self._reason_from_accept(defect))

        self.assertEqual(reached, set(IneligibleLifecycleStateError.REASONS))


# ---------------------------------------------------------------------------
# Slice 11 must NOT have widened
# ---------------------------------------------------------------------------

class Slice11RefusalIsUnchangedTests(TestCase):
    """The shared-helper extraction must not have relaxed ordinary regeneration."""

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Slice 11 Intact")

    def test_ordinary_regeneration_still_refuses_modified_chunks(self):
        document = make_modified(self.collection)
        before = knowledge_snapshot()
        with self.assertRaises(ChunkSetModifiedError) as caught:
            regenerate_derived_chunk_set(
                document.pk, expected=expected_for(document), principal=a_principal(),
            )
        self.assertIn("were NOT modified", str(caught.exception))
        self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_ordinary_regeneration_still_requires_no_reason_code(self):
        """Slice 12's stricter rule must not have leaked into Slice 11."""
        document = make_derived(self.collection, "No Reason Needed")
        KnowledgeDocument.objects.filter(pk=document.pk).update(curated_text="moved on")
        document.refresh_from_db()
        result = regenerate_derived_chunk_set(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )
        self.assertEqual(result.chunk_authority_mode, MODES.DERIVED)

    def test_the_shared_helper_is_private_on_both_sides(self):
        import ast
        import pathlib

        base = pathlib.Path(__file__).resolve().parent / "services"
        tree = ast.parse((base / "knowledge_repair.py").read_text(encoding="utf-8"))
        imported = {
            alias.name for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) for alias in node.names
        }
        borrowed = {n for n in imported if n.startswith("_")}
        self.assertEqual(
            borrowed,
            {"_governed_knowledge_mutation", "_replace_chunk_set",
             "_require_derived_with_usable_generator", "_validate_candidate"},
            "the private surface borrowed from other Core services must stay "
            "exactly this small",
        )

    def test_no_new_public_mutation_api_was_created(self):
        import inspect

        from ai_hub.services import knowledge_regeneration, knowledge_repair

        self.assertEqual(
            {n for n, o in vars(knowledge_regeneration).items()
             if not n.startswith("_") and inspect.isfunction(o)
             and o.__module__ == knowledge_regeneration.__name__},
            {"regenerate_derived_chunk_set"},
        )
        self.assertEqual(
            {n for n, o in vars(knowledge_repair).items()
             if not n.startswith("_") and inspect.isfunction(o)
             and o.__module__ == knowledge_repair.__name__},
            {"accept_current_chunks_as_explicit",
             "discard_modified_chunks_and_regenerate"},
        )


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class RepairAuditTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Repair Audit")

    def test_accept_records_the_authority_transfer_truthfully(self):
        document = make_modified(self.collection)
        before = build_snapshot(document)

        accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=KnowledgeMutationPrincipal.system("operator-console"),
            reason_code="deliberate_authoring",
        )

        event = KnowledgeLifecycleEvent.objects.get()
        self.assertEqual(event.operation, OPERATION_ACCEPT_CURRENT_CHUNKS_AS_EXPLICIT)
        self.assertEqual(event.reason_code, "deliberate_authoring")
        self.assertEqual(event.previous_authority_mode, MODES.DERIVED)
        self.assertEqual(event.new_authority_mode, MODES.EXPLICIT)
        # The old provenance survives in the event even though the row cleared it.
        self.assertEqual(
            event.previous_generation_chunk_set_fingerprint,
            before.recorded_chunk_set_fingerprint,
        )
        self.assertEqual(event.new_generation_chunk_set_fingerprint, "")
        self.assertEqual(event.previous_generator_identity, GENERATOR_CURATED_TEXT_SINGLE_CHUNK)
        self.assertEqual(event.new_generator_identity, "")
        # Observed pair unchanged: proof the chunk set did not move.
        self.assertEqual(
            event.previous_observed_chunk_set_fingerprint,
            event.new_observed_chunk_set_fingerprint,
        )
        self.assertEqual(event.previous_chunk_count, event.new_chunk_count)
        self.assertEqual(event.principal_kind, "system")

    def test_discard_records_the_chunk_replacement_truthfully(self):
        document = make_modified(self.collection, curated_text="the true body")
        accept_before = document_chunk_set_fingerprint(document)

        discard_modified_chunks_and_regenerate(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code="accidental_edit",
        )

        event = KnowledgeLifecycleEvent.objects.get()
        self.assertEqual(event.operation, OPERATION_DISCARD_MODIFIED_CHUNKS_AND_REGENERATE)
        self.assertEqual(event.previous_authority_mode, MODES.DERIVED)
        self.assertEqual(event.new_authority_mode, MODES.DERIVED)
        self.assertEqual(event.previous_observed_chunk_set_fingerprint, accept_before)
        self.assertNotEqual(
            event.previous_observed_chunk_set_fingerprint,
            event.new_observed_chunk_set_fingerprint,
        )
        # The recorded claim now matches what is actually stored.
        self.assertEqual(
            event.new_generation_chunk_set_fingerprint,
            event.new_observed_chunk_set_fingerprint,
        )
        # NOTE: here the recorded fingerprint is UNCHANGED, and that is correct.
        # The source never moved, so discarding the edit restores exactly the
        # chunk set Core had recorded generating. The recorded claim was never
        # wrong - only the chunks had drifted away from it.
        self.assertEqual(
            event.previous_generation_chunk_set_fingerprint,
            event.new_generation_chunk_set_fingerprint,
        )

    def test_discard_records_a_moved_recorded_claim_when_the_source_also_changed(self):
        """The other shape: modified chunks AND a source that moved on."""
        document = make_modified(self.collection, "Also Changed", curated_text="original")
        KnowledgeDocument.objects.filter(pk=document.pk).update(curated_text="a newer body")
        document.refresh_from_db()

        discard_modified_chunks_and_regenerate(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code="accidental_edit",
        )

        event = KnowledgeLifecycleEvent.objects.get()
        self.assertNotEqual(
            event.previous_generation_chunk_set_fingerprint,
            event.new_generation_chunk_set_fingerprint,
        )
        self.assertNotEqual(
            event.previous_generation_input_fingerprint,
            event.new_generation_input_fingerprint,
        )
        self.assertEqual(document.chunks.get().content, "a newer body")

    def test_neither_event_stores_a_knowledge_body(self):
        document = make_modified(self.collection, edit="SECRET CHUNK CONTENT")
        accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )
        event = KnowledgeLifecycleEvent.objects.get()
        for value in (
            event.operation, event.reason_code, event.principal_identifier,
            event.previous_observed_chunk_set_fingerprint,
        ):
            self.assertNotIn("SECRET", value)

    def test_exactly_one_event_per_decision(self):
        document = make_modified(self.collection)
        discard_modified_chunks_and_regenerate(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 1)


# ---------------------------------------------------------------------------
# CAS / rollback
# ---------------------------------------------------------------------------

class RepairRollbackTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Repair Rollback")
        self.document = make_modified(self.collection, curated_text="the true body")

    def test_a_stale_review_conflicts_for_both_verbs(self):
        for verb in (accept_current_chunks_as_explicit, discard_modified_chunks_and_regenerate):
            with self.subTest(verb=verb.__name__):
                document = make_modified(self.collection, f"Stale {verb.__name__}")
                expected = expected_for(document)
                chunk = document.chunks.get()
                chunk.content = "changed again after the operator reviewed it"
                chunk.save(update_fields=["content"])
                before = knowledge_snapshot()

                with self.assertRaises(KnowledgeMutationConflict) as caught:
                    verb(document.pk, expected=expected, principal=a_principal(), reason_code=WHY)

                self.assertEqual(caught.exception.field, "observed_chunk_set_fingerprint")
                self.assertEqual(knowledge_snapshot(), before)

    def test_a_source_change_between_review_and_discard_conflicts(self):
        """Discard must bind the inputs that will REPLACE the modifications."""
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            curated_text="a different replacement body"
        )
        before = knowledge_snapshot()

        with self.assertRaises(KnowledgeMutationConflict) as caught:
            discard_modified_chunks_and_regenerate(
                self.document.pk, expected=expected, principal=a_principal(), reason_code=WHY,
            )
        self.assertEqual(caught.exception.field, "observed_input_fingerprint")
        self.assertEqual(knowledge_snapshot(), before)

    def test_a_collection_move_conflicts_for_both_verbs(self):
        other = KnowledgeCollection.objects.create(name="Elsewhere")
        for verb in (accept_current_chunks_as_explicit, discard_modified_chunks_and_regenerate):
            with self.subTest(verb=verb.__name__):
                document = make_modified(self.collection, f"Moved {verb.__name__}")
                expected = expected_for(document)
                KnowledgeDocument.objects.filter(pk=document.pk).update(collection=other)
                with self.assertRaises(KnowledgeMutationConflict) as caught:
                    verb(document.pk, expected=expected, principal=a_principal(), reason_code=WHY)
                self.assertEqual(caught.exception.field, "collection_id")

    def test_event_failure_rolls_back_the_discard(self):
        before = knowledge_snapshot()
        before_chunks = chunk_rows(self.document)
        with mock.patch(
            "ai_hub.services.knowledge_mutation.KnowledgeLifecycleEvent.objects.create",
            side_effect=RuntimeError("audit write failed"),
        ):
            with self.assertRaises(RuntimeError):
                discard_modified_chunks_and_regenerate(
                    self.document.pk, expected=expected_for(self.document),
                    principal=a_principal(), reason_code=WHY,
                )
        self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(chunk_rows(self.document), before_chunks)

    def test_event_failure_rolls_back_the_accept(self):
        before = knowledge_snapshot()
        with mock.patch(
            "ai_hub.services.knowledge_mutation.KnowledgeLifecycleEvent.objects.create",
            side_effect=RuntimeError("audit write failed"),
        ):
            with self.assertRaises(RuntimeError):
                accept_current_chunks_as_explicit(
                    self.document.pk, expected=expected_for(self.document),
                    principal=a_principal(), reason_code=WHY,
                )
        self.assertEqual(knowledge_snapshot(), before)

    def test_a_generator_failure_rolls_the_discard_back_inside_the_transaction(self):
        before = knowledge_snapshot()
        before_chunks = chunk_rows(self.document)
        spy = _EnteredRepairMutation()
        with spy.patch(), mock.patch(
            "ai_hub.services.knowledge_repair.generator_output_projection",
            side_effect=RuntimeError("generator exploded"),
        ):
            with self.assertRaises(RuntimeError):
                discard_modified_chunks_and_regenerate(
                    self.document.pk, expected=expected_for(self.document),
                    principal=a_principal(), reason_code=WHY,
                )
        self.assertTrue(spy.entered, "must fail INSIDE the governed transaction")
        self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(chunk_rows(self.document), before_chunks)

    def test_an_invalid_candidate_rolls_back_before_destroying_anything(self):
        before = knowledge_snapshot()
        before_chunks = chunk_rows(self.document)
        spy = _EnteredRepairMutation()
        with spy.patch(), mock.patch(
            "ai_hub.services.knowledge_repair.generator_output_projection",
            return_value=[{"chunk_index": 1, "section_title": "x", "content": "  "}],
        ):
            with self.assertRaises(InvalidCandidateError):
                discard_modified_chunks_and_regenerate(
                    self.document.pk, expected=expected_for(self.document),
                    principal=a_principal(), reason_code=WHY,
                )
        self.assertTrue(spy.entered)
        self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(chunk_rows(self.document), before_chunks)


# ---------------------------------------------------------------------------
# What Slice 12 deliberately does NOT do
# ---------------------------------------------------------------------------

class RepairDeliberateOmissionsTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Omissions")

    def test_there_is_no_reset_to_unknown_verb(self):
        """D-3e-4: it would be a laundering route back to a convenient claim."""
        from ai_hub.services import knowledge_repair

        for forbidden in ("reset", "demote", "unknown", "clear_authority", "revert"):
            with self.subTest(name=forbidden):
                self.assertFalse(
                    any(n.startswith(forbidden) for n in dir(knowledge_repair)),
                    f"{forbidden!r} would create a laundering path",
                )

    def test_there_is_no_generic_repair_verb(self):
        from ai_hub.services import knowledge_repair

        self.assertFalse(hasattr(knowledge_repair, "repair_knowledge"))
        for forbidden in ("repair_", "fix_", "normalize", "cleanup", "bulk", "backfill", "batch"):
            with self.subTest(name=forbidden):
                self.assertFalse(any(n.startswith(forbidden) for n in dir(knowledge_repair)))

    def test_kp010_is_left_visible_and_never_normalized(self):
        """An anomaly may be forensic evidence. Preflight keeps reporting it."""
        document = make_derived(self.collection, "KP010")
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            chunk_authority_mode=MODES.EXPLICIT
        )
        document.refresh_from_db()
        row = preflight_row(document.pk)
        self.assertIn("KP010_LIFECYCLE_FACT_INCONSISTENCY", row["issues"])

        with self.assertRaises(IneligibleLifecycleStateError):
            accept_current_chunks_as_explicit(
                document.pk, expected=expected_for(document),
                principal=a_principal(), reason_code=WHY,
            )
        self.assertIn(
            "KP010_LIFECYCLE_FACT_INCONSISTENCY", preflight_row(document.pk)["issues"]
        )

    def test_no_structural_issue_is_touched(self):
        """KP001/2/3/5/6 are a different axis; re-indexing would change `c1`."""
        document = make_modified(self.collection, "Structural", edit="   ")
        row_before = preflight_row(document.pk)
        self.assertIn("KP003_UNUSABLE_CHUNK_CONTENT", row_before["issues"])

        accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )
        self.assertIn(
            "KP003_UNUSABLE_CHUNK_CONTENT", preflight_row(document.pk)["issues"]
        )

    def test_preflight_remains_read_only(self):
        document = make_modified(self.collection, "ReadOnly")
        accept_current_chunks_as_explicit(
            document.pk, expected=expected_for(document),
            principal=a_principal(), reason_code=WHY,
        )
        first = run_knowledge_preflight()
        events = KnowledgeLifecycleEvent.objects.count()
        self.assertEqual(first, run_knowledge_preflight())
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), events)

    def test_no_new_migration_is_required(self):
        from django.apps import apps
        from django.db.migrations.autodetector import MigrationAutodetector
        from django.db.migrations.loader import MigrationLoader
        from django.db.migrations.questioner import NonInteractiveMigrationQuestioner
        from django.db.migrations.state import ProjectState

        loader = MigrationLoader(None, ignore_no_migrations=True)
        autodetector = MigrationAutodetector(
            loader.project_state(), ProjectState.from_apps(apps),
            NonInteractiveMigrationQuestioner(specified_apps=set(), dry_run=True),
        )
        self.assertEqual(autodetector.changes(graph=loader.graph).get("ai_hub", []), [])


# ---------------------------------------------------------------------------
# Security / Agent isolation
# ---------------------------------------------------------------------------

class RepairSecurityBoundaryTests(TestCase):
    AGENT_FACING_MODULES = (
        "services/knowledge_retrieval.py",
        "services/game_action_dispatcher.py",
        "services/agent_runtime.py",
        "services/tools_runtime.py",
        "services/tool_resolution.py",
        "services/knowledge_ingestion.py",
        "services/knowledge_preflight.py",
    )
    DEFERRED_SURFACE_MODULES = ("admin.py", "services/build_console.py")
    REPAIR_SYMBOLS = (
        "knowledge_repair",
        "accept_current_chunks_as_explicit",
        "discard_modified_chunks_and_regenerate",
    )

    def _module_path(self, relative):
        import pathlib

        path = pathlib.Path(__file__).resolve().parent / relative
        self.assertTrue(
            path.exists(),
            f"{relative} does not exist, so this security test would silently "
            "stop checking it.",
        )
        return path

    def test_every_scanned_module_exists(self):
        for relative in self.AGENT_FACING_MODULES + self.DEFERRED_SURFACE_MODULES:
            with self.subTest(module=relative):
                self._module_path(relative)

    def test_no_agent_facing_module_imports_repair(self):
        for relative in self.AGENT_FACING_MODULES:
            with self.subTest(module=relative):
                source = self._module_path(relative).read_text(encoding="utf-8")
                for symbol in self.REPAIR_SYMBOLS:
                    self.assertNotIn(symbol, source)

    def test_no_admin_or_build_console_integration_exists(self):
        for relative in self.DEFERRED_SURFACE_MODULES:
            with self.subTest(module=relative):
                source = self._module_path(relative).read_text(encoding="utf-8")
                for symbol in self.REPAIR_SYMBOLS:
                    self.assertNotIn(symbol, source)

    def test_no_management_command_invokes_repair(self):
        """No operator surface in this slice — that is Slice 13."""
        import pathlib

        commands = pathlib.Path(__file__).resolve().parent / "management" / "commands"
        for path in sorted(commands.glob("*.py")):
            with self.subTest(command=path.name):
                source = path.read_text(encoding="utf-8")
                for symbol in self.REPAIR_SYMBOLS:
                    self.assertNotIn(symbol, source)

    def test_no_seeded_tool_definition_exposes_repair(self):
        seed_starter_toolboxes()
        self.assertGreater(ToolDefinition.objects.count(), 0)
        for tool in ToolDefinition.objects.all():
            with self.subTest(tool=tool.name):
                blob = " ".join(
                    str(p) for p in (tool.name, tool.label, tool.description, tool.config)
                ).lower()
                for forbidden in ("repair", "discard", "accept_current", "authority"):
                    self.assertNotIn(forbidden, blob)

    def test_repair_does_not_reimplement_the_mutation_foundation(self):
        import ast
        import pathlib

        tree = ast.parse(
            (
                pathlib.Path(__file__).resolve().parent / "services" / "knowledge_repair.py"
            ).read_text(encoding="utf-8")
        )
        referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        referenced |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        imported = {
            alias.name for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) for alias in node.names
        }
        for symbol in ("atomic", "select_for_update", "KnowledgeLifecycleEvent",
                       "verify_expected_state", "build_snapshot"):
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, referenced)
                self.assertNotIn(symbol, imported)
        self.assertIn("_governed_knowledge_mutation", imported)


# ---------------------------------------------------------------------------
# PostgreSQL — the destructive repair race
# ---------------------------------------------------------------------------

class DiscardConcurrencyTests(TransactionTestCase):
    """Two operators racing to discard the same reviewed modifications.

    SQLite has no row locks and cannot evidence this; validated by the
    PostgreSQL 16 CI job.
    """

    reset_sequences = True

    def test_racing_discards_produce_one_winner_and_no_partial_chunk_set(self):
        if not connection.features.has_select_for_update:
            self.skipTest(
                "SQLite cannot validate select_for_update locking semantics; run "
                "this test on PostgreSQL CI."
            )

        collection = KnowledgeCollection.objects.create(name="Discard Race")
        document = make_derived(collection, "Contended", curated_text="the true body")
        chunk = document.chunks.get()
        chunk.content = "an ungoverned edit"
        chunk.save(update_fields=["content"])
        document.refresh_from_db()
        expected = expected_for(document)

        def attempt(index):
            close_old_connections()
            try:
                discard_modified_chunks_and_regenerate(
                    document.pk, expected=expected,
                    principal=KnowledgeMutationPrincipal.system(f"racer-{index}"),
                    reason_code="accidental_edit",
                )
                return "committed"
            except KnowledgeMutationConflict:
                # ONLY a stale-review conflict counts. The broader repair and
                # regeneration exception families are deliberately NOT caught:
                # both racers start from the same valid modified state, so the
                # loser must fail because its reviewed snapshot went stale.
                return "lost_cas"
            finally:
                close_old_connections()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(attempt, range(2)))

        self.assertEqual(results.count("committed"), 1)
        self.assertEqual(results.count("lost_cas"), 1)

        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 1)
        chunks = list(document.chunks.order_by("chunk_index"))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "the true body")
        indexes = [c.chunk_index for c in chunks]
        self.assertEqual(len(indexes), len(set(indexes)), "duplicate (document, chunk_index)")

        document.refresh_from_db()
        self.assertEqual(
            document.generation_chunk_set_fingerprint,
            document_chunk_set_fingerprint(document),
        )
