"""Tests for safe DERIVED chunk-set regeneration (Slice 11, RC-002 3e).

The first governed operation that can DELETE retrievable content, so the tests
are weighted toward proving what it refuses and what it leaves untouched, not
toward proving it works.

Two branches are pinned separately, per ruling D-3e-2: when `c1` changes the
chunk set is replaced and identities change with it; when `c1` does not change,
not a single chunk row may be touched.
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
    SUPPORTED_GENERATORS,
    chunk_set_fingerprint,
    curated_text_single_chunk_projection,
    document_chunk_set_fingerprint,
    document_generation_input_fingerprint,
)
from ai_hub.services.knowledge_mutation import (
    ExpectedKnowledgeState,
    KnowledgeMutationConflict,
    KnowledgeMutationPrincipal,
    build_snapshot,
)
from ai_hub.services.knowledge_preflight import run_knowledge_preflight
from ai_hub.services.knowledge_regeneration import (
    OPERATION_REGENERATE_DERIVED_CHUNK_SET,
    AuthorityNotDerivedError,
    ChunkSetModifiedError,
    GeneratorVersionAheadError,
    IncompleteProvenanceError,
    InvalidCandidateError,
    KnowledgeRegenerationError,
    UnsupportedGeneratorForRegenerationError,
    regenerate_derived_chunk_set,
)
from ai_hub.services.starter_toolboxes import seed_starter_toolboxes

MODES = KnowledgeDocument.ChunkAuthorityMode


def a_principal():
    return KnowledgeMutationPrincipal.human("operator-1")


def expected_for(document):
    return ExpectedKnowledgeState.from_snapshot(build_snapshot(document))


def chunk_rows(document):
    """Everything database-visible about the chunk set, not merely its count."""
    return list(
        document.chunks.order_by("chunk_index").values(
            "pk", "chunk_index", "section_title", "content", "token_estimate",
            "metadata", "created_at", "updated_at",
        )
    )


def knowledge_snapshot():
    return {
        "documents": list(
            KnowledgeDocument.objects.order_by("pk").values(
                "pk", "collection_id", "title", "curated_text", "status", "notes",
                "chunk_authority_mode", "generation_input_fingerprint",
                "generation_chunk_set_fingerprint", "generator_identity",
                "generator_version",
            )
        ),
        "chunks": list(
            KnowledgeDocumentChunk.objects.order_by("pk").values(
                "pk", "document_id", "chunk_index", "section_title", "content",
                "token_estimate", "metadata", "created_at", "updated_at",
            )
        ),
        "events": list(KnowledgeLifecycleEvent.objects.order_by("pk").values("pk", "operation")),
    }


def make_derived(collection, title="Derived", *, curated_text="derived body", version=None):
    """A coherent DERIVED document whose chunks the real generator produced."""
    document = KnowledgeDocument.objects.create(
        collection=collection, title=title, curated_text=curated_text,
        status=KnowledgeDocument.Status.ACTIVE,
    )
    ensure_initial_knowledge_chunk(document)
    return record_generation(document, version=version)


def record_generation(document, *, version=None):
    document.chunk_authority_mode = MODES.DERIVED
    document.generation_input_fingerprint = document_generation_input_fingerprint(
        document, GENERATOR_CURATED_TEXT_SINGLE_CHUNK
    )
    document.generation_chunk_set_fingerprint = document_chunk_set_fingerprint(document)
    document.generator_identity = GENERATOR_CURATED_TEXT_SINGLE_CHUNK
    document.generator_version = (
        GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION if version is None else version
    )
    document.save()
    return document


class _EnteredGovernedMutation:
    """Records that the governed transaction was genuinely entered.

    The flag is set only AFTER the real context manager's `__enter__` returns —
    i.e. after the document and its chunks are locked and the compare-and-swap
    has passed. A test asserting it therefore proves the failure it is exercising
    happened INSIDE the transaction, on the destructive path, rather than before
    the transaction ever opened.

    This exists because an earlier revision validated the candidate before
    entering the mutation, which let three rollback tests pass without ever
    reaching the code they claimed to cover.
    """

    def __init__(self):
        self.entered = False

    def patch(self):
        import contextlib

        from ai_hub.services import knowledge_regeneration

        real = knowledge_regeneration._governed_knowledge_mutation

        @contextlib.contextmanager
        def wrapper(*args, **kwargs):
            with real(*args, **kwargs) as mutation:
                self.entered = True
                yield mutation

        return mock.patch.object(
            knowledge_regeneration, "_governed_knowledge_mutation", wrapper
        )


def preflight_row(document_id):
    for row in run_knowledge_preflight()["documents"]:
        if row["document_id"] == document_id:
            return row
    raise AssertionError(f"document {document_id} missing from preflight")


# ---------------------------------------------------------------------------
# Generator-version drift guard  (HIGH risk from the 3e architecture review)
# ---------------------------------------------------------------------------

class GeneratorVersionDriftGuardTests(TestCase):
    """Binds `curated_text_single_chunk` v1's OUTPUT to its declared version.

    The hazard: `c1` fingerprints output, not the generator's code. If the
    writer's behaviour changes without `GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION`
    being incremented, every previously-adopted document silently becomes
    non-reproducible and Preflight starts reporting `DERIVED_CHUNKS_MODIFIED` —
    accusing operators of tampering that never happened, and blocking exactly
    the regeneration that would fix it.

    These goldens make that failure loud and immediate instead. If one breaks,
    the correct response is almost never to update the constant: it is to bump
    the generator version and add goldens for v2.
    """

    DECLARED_VERSION = 1

    # title, curated_text  ->  c1 fingerprint of the projected chunk set.
    #
    # Several input SHAPES, not one, because a behaviour change can affect only
    # a particular shape - whitespace handling, Unicode normalization, line
    # endings - while the writer and the projection drift TOGETHER and therefore
    # still agree with each other. Agreement is not correctness; these frozen
    # values are.
    GOLDEN_OUTPUTS = {
        ("Golden Title", "golden body line one\nline two"):
            "c1:a3317100aec1028f056f2b6c0d92f7c1dec5db8a1c433f654209f7c7dc1ce00b",
        # surrounding whitespace: stripped from content, NOT from the title
        ("  Padded Title  ", "\n\n   padded body with surrounding space   \t\n"):
            "c1:f36e7db1ebcb39acc24bf62a5f6cdb30619ffc95a23126ea63b590c67764b2d6",
        # Unicode: accents, non-ASCII punctuation, NFC normalization
        ("T\u00edtulo \u00fcnicode \u2014 em dash",
         "cuerpo en espa\u00f1ol \u00b7 acentuaci\u00f3n \u00b7 \u00fcn\u00efc\u00f6d\u00e9"):
            "c1:49107681a859152fa50388743ceeb17a96d03e9d9cd8a4b83f78cbf1d9f669a9",
        # multi-line, including an interior blank line
        ("Multi Line", "first line\nsecond line\n\nfourth line after blank"):
            "c1:c75d2929f9f8d1176a7d7cc4bf4698b671b65da03e005edf4f06244a0953bcc5",
        # CRLF source: `c1` normalizes line endings, so a CRLF body and its LF
        # form must fingerprint identically
        ("CRLF Source", "windows line one\r\nwindows line two"):
            "c1:d8f01028df2941575b517098fea21d72048507192b33c431adefff478d4a5fff",
    }

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Drift Guard")

    def test_the_declared_version_is_the_one_these_goldens_describe(self):
        self.assertEqual(
            GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION, self.DECLARED_VERSION,
            "The generator version changed. These goldens describe v"
            f"{self.DECLARED_VERSION} output only — add goldens for the new "
            "version rather than editing these.",
        )
        self.assertEqual(
            SUPPORTED_GENERATORS[GENERATOR_CURATED_TEXT_SINGLE_CHUNK],
            self.DECLARED_VERSION,
        )

    def test_projected_output_still_matches_the_recorded_golden(self):
        for (title, curated_text), golden in self.GOLDEN_OUTPUTS.items():
            with self.subTest(title=title):
                document = KnowledgeDocument(
                    collection=self.collection, title=title, curated_text=curated_text
                )
                actual = chunk_set_fingerprint(
                    curated_text_single_chunk_projection(document)
                )
                self.assertEqual(
                    actual, golden,
                    "curated_text_single_chunk output changed while its declared "
                    f"version stayed at {self.DECLARED_VERSION}. Bump "
                    "GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION and record new "
                    "goldens; do not edit this constant to make the test pass.",
                )

    def test_the_real_writer_still_matches_the_recorded_golden(self):
        """Pins the WRITER too, not only the projection that models it."""
        for (title, curated_text), golden in self.GOLDEN_OUTPUTS.items():
            with self.subTest(title=title):
                document = KnowledgeDocument.objects.create(
                    collection=self.collection, title=title, curated_text=curated_text,
                    status=KnowledgeDocument.Status.ACTIVE,
                )
                ensure_initial_knowledge_chunk(document)
                self.assertEqual(document_chunk_set_fingerprint(document), golden)


# ---------------------------------------------------------------------------
# Branch A — c1 changes: replace, identities change with the artifact
# ---------------------------------------------------------------------------

class RegenerationReplacesChangedChunkSetTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Replace")
        self.document = make_derived(self.collection, "Replace Me", curated_text="original body")
        # A governed source change: the document is now DERIVED_INPUT_CHANGED.
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            curated_text="a materially different body"
        )
        self.document.refresh_from_db()

    def test_the_precondition_is_derived_input_changed(self):
        self.assertEqual(preflight_row(self.document.pk)["lifecycle_state"], "DERIVED_INPUT_CHANGED")

    def test_regeneration_installs_the_generator_output(self):
        before_pks = {row["pk"] for row in chunk_rows(self.document)}
        result = regenerate_derived_chunk_set(
            self.document.pk, expected=expected_for(self.document),
            principal=a_principal(), reason_code="source_changed",
        )
        self.assertEqual(result.chunk_authority_mode, MODES.DERIVED)
        chunk = self.document.chunks.get()
        self.assertEqual(chunk.content, "a materially different body")
        self.assertEqual(chunk.section_title, "Replace Me")
        self.assertNotIn(chunk.pk, before_pks)

    def test_chunk_identities_change_with_the_artifact(self):
        """D-3e-2: a dangling handle beats a handle resolving to other content."""
        before_pks = {row["pk"] for row in chunk_rows(self.document)}
        regenerate_derived_chunk_set(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        after_pks = {row["pk"] for row in chunk_rows(self.document)}
        self.assertTrue(before_pks.isdisjoint(after_pks))

    def test_the_document_becomes_derived_current(self):
        regenerate_derived_chunk_set(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        row = preflight_row(self.document.pk)
        self.assertEqual(row["lifecycle_state"], "DERIVED_CURRENT")
        self.assertEqual(row["issues"], [])

    def test_provenance_is_rewritten_to_the_new_truth(self):
        regenerate_derived_chunk_set(
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
        self.assertEqual(self.document.generator_identity, GENERATOR_CURATED_TEXT_SINGLE_CHUNK)
        self.assertEqual(
            self.document.generator_version, GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION
        )

    def test_regenerated_chunks_carry_no_false_ingestion_marker(self):
        """The only marker Core writes means "initial ingestion fallback",
        which a governed regeneration is not."""
        regenerate_derived_chunk_set(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        self.assertEqual(self.document.chunks.get().metadata, {})

    def test_a_multi_chunk_set_is_collapsed_to_the_generator_output(self):
        document = make_derived(self.collection, "Multi", curated_text="one body")
        KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=2, section_title="Extra", content="stray",
        )
        record_generation(document)  # re-record so the set is "as generated"
        KnowledgeDocument.objects.filter(pk=document.pk).update(curated_text="new single body")
        document.refresh_from_db()

        regenerate_derived_chunk_set(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )
        self.assertEqual(document.chunks.count(), 1)
        self.assertEqual(document.chunks.get().chunk_index, 1)


# ---------------------------------------------------------------------------
# Branch B — c1 unchanged: touch nothing (D-3e-2)
# ---------------------------------------------------------------------------

class RegenerationPreservesUnchangedChunkSetTests(TestCase):
    """No chunk identity churn when the artifact did not actually change."""

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Preserve")

    def _assert_chunks_untouched(self, document, before):
        after = chunk_rows(document)
        self.assertEqual(after, before, "no chunk row may be touched on this branch")
        self.assertEqual(
            [r["pk"] for r in after], [r["pk"] for r in before], "primary keys must survive"
        )

    def test_derived_current_regeneration_touches_no_chunk_row(self):
        document = make_derived(self.collection, "Already Current")
        before = chunk_rows(document)
        regenerate_derived_chunk_set(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )
        self._assert_chunks_untouched(document, before)

    def test_generator_version_advance_touches_no_chunk_row(self):
        """The case the ruling was written for: version moves, output does not."""
        document = make_derived(
            self.collection, "Outdated",
            version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION - 1,
        )
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "GENERATOR_OUTDATED")
        before = chunk_rows(document)

        result = regenerate_derived_chunk_set(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )

        self._assert_chunks_untouched(document, before)
        self.assertEqual(result.generator_version, GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION)
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "DERIVED_CURRENT")

    def test_timestamps_metadata_and_token_estimate_all_survive(self):
        document = make_derived(self.collection, "Untouched")
        chunk = document.chunks.get()
        chunk.metadata = {"ingestion": "initial_curated_text"}
        chunk.token_estimate = 4242
        chunk.save(update_fields=["metadata", "token_estimate"])
        before = chunk_rows(document)

        regenerate_derived_chunk_set(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )

        after = chunk_rows(document)
        self.assertEqual(after, before)
        self.assertEqual(after[0]["metadata"], {"ingestion": "initial_curated_text"})
        self.assertEqual(after[0]["token_estimate"], 4242)

    def test_an_unchanged_regeneration_still_records_one_event(self):
        """A committed governed mutation is an event even when nothing moved."""
        document = make_derived(self.collection, "No-op")
        regenerate_derived_chunk_set(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )
        event = KnowledgeLifecycleEvent.objects.get()
        self.assertEqual(event.operation, OPERATION_REGENERATE_DERIVED_CHUNK_SET)
        self.assertEqual(
            event.previous_observed_chunk_set_fingerprint,
            event.new_observed_chunk_set_fingerprint,
        )


# ---------------------------------------------------------------------------
# D-3e-1 — the conditionally permitted empty-set case
# ---------------------------------------------------------------------------

class RegenerationFromEmptyChunkSetTests(TestCase):
    """An empty chunk set grants no permission of its own.

    Permission comes from the same four-part proof as every other case; the
    empty set simply happens to satisfy condition 4 when Core recorded
    generating nothing.
    """

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Empty Set")

    def _derived_with_empty_recorded_set(self, curated_text="body for regeneration"):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Empty", curated_text=curated_text,
            status=KnowledgeDocument.Status.ACTIVE,
        )
        # No chunks; record the empty set as the generated one.
        return record_generation(document)

    def test_a_derived_document_with_an_empty_generated_set_regenerates(self):
        document = self._derived_with_empty_recorded_set()
        self.assertEqual(document.chunks.count(), 0)

        regenerate_derived_chunk_set(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )

        self.assertEqual(document.chunks.count(), 1)
        self.assertEqual(document.chunks.get().content, "body for regeneration")
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "DERIVED_CURRENT")

    def test_emptiness_alone_does_not_grant_permission(self):
        """Same empty chunk set, but UNKNOWN authority — still refused."""
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Empty Unknown", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        with self.assertRaises(AuthorityNotDerivedError):
            regenerate_derived_chunk_set(
                document.pk, expected=expected_for(document), principal=a_principal(),
            )

    def test_an_empty_set_with_a_tampered_record_is_still_refused(self):
        document = self._derived_with_empty_recorded_set()
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            generation_chunk_set_fingerprint="c1:" + "0" * 64
        )
        document.refresh_from_db()
        with self.assertRaises(ChunkSetModifiedError):
            regenerate_derived_chunk_set(
                document.pk, expected=expected_for(document), principal=a_principal(),
            )

    def test_an_empty_source_produces_no_candidate_and_is_refused(self):
        document = self._derived_with_empty_recorded_set(curated_text="   \n\t ")
        with self.assertRaises(InvalidCandidateError):
            regenerate_derived_chunk_set(
                document.pk, expected=expected_for(document), principal=a_principal(),
            )
        self.assertEqual(document.chunks.count(), 0)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)


# ---------------------------------------------------------------------------
# Refusals — and proof that each leaves everything untouched
# ---------------------------------------------------------------------------

class RegenerationRefusalTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Refusals")

    def _assert_refused(self, document, exception):
        before_db = knowledge_snapshot()
        with self.assertRaises(exception) as caught:
            regenerate_derived_chunk_set(
                document.pk, expected=expected_for(document), principal=a_principal(),
            )
        self.assertEqual(knowledge_snapshot(), before_db)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)
        return caught.exception

    def test_unknown_authority_is_refused(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Unknown", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        ensure_initial_knowledge_chunk(document)
        error = self._assert_refused(document, AuthorityNotDerivedError)
        self.assertIn("adjudication", str(error))

    def test_explicit_authority_is_refused(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Explicit", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
            chunk_authority_mode=MODES.EXPLICIT,
        )
        KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1, section_title="Hand", content="authored",
        )
        error = self._assert_refused(document, AuthorityNotDerivedError)
        self.assertIn("authored artifact", str(error))

    def test_modified_chunks_are_refused_not_repaired(self):
        document = make_derived(self.collection, "Tampered")
        chunk = document.chunks.get()
        chunk.content = "a human improved this"
        chunk.save(update_fields=["content"])

        error = self._assert_refused(document, ChunkSetModifiedError)
        self.assertIn("were NOT modified", str(error))
        self.assertNotEqual(error.recorded_fingerprint, error.observed_fingerprint)

    def test_incomplete_provenance_is_refused(self):
        for field, blank in (
            ("generation_input_fingerprint", ""),
            ("generation_chunk_set_fingerprint", ""),
            ("generator_identity", ""),
            ("generator_version", None),
        ):
            with self.subTest(field=field):
                document = make_derived(self.collection, f"Incomplete {field}")
                KnowledgeDocument.objects.filter(pk=document.pk).update(**{field: blank})
                document.refresh_from_db()
                error = self._assert_refused(document, IncompleteProvenanceError)
                self.assertIn(field, str(error))
                KnowledgeDocument.objects.filter(pk=document.pk).delete()

    def test_an_unsupported_generator_is_refused(self):
        document = make_derived(self.collection, "Unsupported")
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            generator_identity="some_future_generator"
        )
        document.refresh_from_db()
        self._assert_refused(document, UnsupportedGeneratorForRegenerationError)

    def test_a_version_ahead_of_this_core_is_refused(self):
        document = make_derived(
            self.collection, "Ahead",
            version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION + 1,
        )
        error = self._assert_refused(document, GeneratorVersionAheadError)
        self.assertIn("upgrade Core", str(error))

    def test_every_refusal_shares_one_base_class(self):
        for error in (
            AuthorityNotDerivedError, IncompleteProvenanceError,
            UnsupportedGeneratorForRegenerationError, GeneratorVersionAheadError,
            ChunkSetModifiedError, InvalidCandidateError,
        ):
            with self.subTest(error=error.__name__):
                self.assertTrue(issubclass(error, KnowledgeRegenerationError))

    def test_a_refusal_never_reaches_the_destructive_path(self):
        """Chunk rows are compared in full, not merely counted."""
        document = make_derived(self.collection, "Untouched By Refusal")
        chunk = document.chunks.get()
        chunk.content = "edited"
        chunk.save(update_fields=["content"])
        before = chunk_rows(document)

        with self.assertRaises(ChunkSetModifiedError):
            regenerate_derived_chunk_set(
                document.pk, expected=expected_for(document), principal=a_principal(),
            )
        self.assertEqual(chunk_rows(document), before)


# ---------------------------------------------------------------------------
# Rollback / atomicity
# ---------------------------------------------------------------------------

class RegenerationRollbackTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Rollback")
        self.document = make_derived(self.collection, "Rollback", curated_text="original")
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(curated_text="changed body")
        self.document.refresh_from_db()

    def test_event_creation_failure_rolls_back_the_replacement(self):
        before = knowledge_snapshot()
        with mock.patch(
            "ai_hub.services.knowledge_mutation.KnowledgeLifecycleEvent.objects.create",
            side_effect=RuntimeError("audit write failed"),
        ):
            with self.assertRaises(RuntimeError):
                regenerate_derived_chunk_set(
                    self.document.pk, expected=expected_for(self.document),
                    principal=a_principal(),
                )
        self.assertEqual(knowledge_snapshot(), before)

    def test_a_generator_failure_rolls_everything_back(self):
        before = knowledge_snapshot()
        before_chunks = chunk_rows(self.document)
        spy = _EnteredGovernedMutation()
        with spy.patch(), mock.patch(
            "ai_hub.services.knowledge_regeneration.generator_output_projection",
            side_effect=RuntimeError("generator exploded"),
        ):
            with self.assertRaises(RuntimeError):
                regenerate_derived_chunk_set(
                    self.document.pk, expected=expected_for(self.document),
                    principal=a_principal(),
                )
        self.assertTrue(
            spy.entered,
            "the generator must fail INSIDE the governed transaction, or this "
            "test proves nothing about rollback",
        )
        self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(chunk_rows(self.document), before_chunks)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_an_invalid_candidate_rolls_back_before_destroying_anything(self):
        before = knowledge_snapshot()
        before_chunks = chunk_rows(self.document)
        spy = _EnteredGovernedMutation()
        with spy.patch(), mock.patch(
            "ai_hub.services.knowledge_regeneration.generator_output_projection",
            return_value=[{"chunk_index": 1, "section_title": "x", "content": "   "}],
        ):
            with self.assertRaises(InvalidCandidateError):
                regenerate_derived_chunk_set(
                    self.document.pk, expected=expected_for(self.document),
                    principal=a_principal(),
                )
        self.assertTrue(spy.entered, "validation must happen inside the transaction")
        self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(chunk_rows(self.document), before_chunks)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_duplicate_candidate_indexes_are_rejected_before_deletion(self):
        before = knowledge_snapshot()
        before_chunks = chunk_rows(self.document)
        spy = _EnteredGovernedMutation()
        with spy.patch(), mock.patch(
            "ai_hub.services.knowledge_regeneration.generator_output_projection",
            return_value=[
                {"chunk_index": 1, "section_title": "a", "content": "one"},
                {"chunk_index": 1, "section_title": "b", "content": "two"},
            ],
        ):
            with self.assertRaises(InvalidCandidateError):
                regenerate_derived_chunk_set(
                    self.document.pk, expected=expected_for(self.document),
                    principal=a_principal(),
                )
        self.assertTrue(spy.entered, "validation must happen inside the transaction")
        self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(chunk_rows(self.document), before_chunks)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_a_stale_expected_state_conflicts_and_changes_nothing(self):
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(title="Renamed After Review")
        before = knowledge_snapshot()

        with self.assertRaises(KnowledgeMutationConflict):
            regenerate_derived_chunk_set(
                self.document.pk, expected=expected, principal=a_principal(),
            )
        self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_a_stale_review_reports_a_conflict_not_a_candidate_error(self):
        """Regression: the CAS must be reached before anything else can fail.

        This is the exact case that exposed the pre-lock projection. The review
        goes stale AND the new source projects to nothing, so two failures are
        available at once. The operator must be told the truth that matters —
        "the document changed since you reviewed it" — not "the generator
        produced nothing usable", which would send them to investigate the
        wrong thing entirely.

        With a pre-lock projection this raised `InvalidCandidateError`.
        """
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            curated_text="   \n\t "
        )
        before = knowledge_snapshot()

        with self.assertRaises(KnowledgeMutationConflict) as caught:
            regenerate_derived_chunk_set(
                self.document.pk, expected=expected, principal=a_principal(),
            )

        self.assertEqual(caught.exception.field, "observed_input_fingerprint")
        self.assertEqual(knowledge_snapshot(), before)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_nothing_is_projected_or_validated_before_the_transaction_opens(self):
        """There is exactly ONE correctness-authoritative projection.

        Pins the ordering directly: the generator projection may not be called
        at all until the governed mutation has been entered — i.e. after the
        lock and the compare-and-swap.
        """
        calls_before_entry = []
        spy = _EnteredGovernedMutation()

        def counting_projection(identity, document):
            calls_before_entry.append(spy.entered)
            return curated_text_single_chunk_projection(document)

        with spy.patch(), mock.patch(
            "ai_hub.services.knowledge_regeneration.generator_output_projection",
            side_effect=counting_projection,
        ):
            regenerate_derived_chunk_set(
                self.document.pk, expected=expected_for(self.document),
                principal=a_principal(),
            )

        self.assertEqual(
            calls_before_entry, [True],
            "the generator must be projected exactly once, and only after the "
            "governed transaction has been entered",
        )

    def test_a_collection_move_between_review_and_commit_conflicts(self):
        expected = expected_for(self.document)
        other = KnowledgeCollection.objects.create(name="Elsewhere")
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(collection=other)

        with self.assertRaises(KnowledgeMutationConflict) as caught:
            regenerate_derived_chunk_set(
                self.document.pk, expected=expected, principal=a_principal(),
            )
        self.assertEqual(caught.exception.field, "collection_id")


# ---------------------------------------------------------------------------
# Lifecycle event truthfulness
# ---------------------------------------------------------------------------

class RegenerationAuditTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Audit")
        self.document = make_derived(self.collection, "Audited", curated_text="before body")
        self.before_snapshot = build_snapshot(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(curated_text="after body")
        self.document.refresh_from_db()

    def test_exactly_one_event_describing_the_whole_transition(self):
        expected_before = build_snapshot(self.document)
        regenerate_derived_chunk_set(
            self.document.pk, expected=expected_for(self.document),
            principal=KnowledgeMutationPrincipal.system("regen-job"),
            reason_code="source_changed",
        )
        event = KnowledgeLifecycleEvent.objects.get()

        self.assertEqual(event.operation, OPERATION_REGENERATE_DERIVED_CHUNK_SET)
        self.assertEqual(event.reason_code, "source_changed")
        self.assertEqual(event.principal_kind, "system")
        self.assertEqual(event.principal_identifier, "regen-job")
        # Authority and status do not move; the event says so truthfully.
        self.assertEqual(event.previous_authority_mode, MODES.DERIVED)
        self.assertEqual(event.new_authority_mode, MODES.DERIVED)
        self.assertEqual(event.previous_status, event.new_status)
        # The recorded chunk-set fingerprint moves to the new artifact.
        self.assertEqual(
            event.previous_generation_chunk_set_fingerprint,
            expected_before.recorded_chunk_set_fingerprint,
        )
        self.assertNotEqual(
            event.previous_generation_chunk_set_fingerprint,
            event.new_generation_chunk_set_fingerprint,
        )
        # Observed input already differed before regeneration; observed chunks
        # only differ afterwards.
        self.assertNotEqual(
            event.previous_observed_chunk_set_fingerprint,
            event.new_observed_chunk_set_fingerprint,
        )
        self.assertEqual(event.previous_chunk_count, 1)
        self.assertEqual(event.new_chunk_count, 1)
        self.assertIsNotNone(event.created_at)

    def test_the_event_stores_no_knowledge_body(self):
        regenerate_derived_chunk_set(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        event = KnowledgeLifecycleEvent.objects.get()
        for value in (
            event.principal_identifier, event.reason_code, event.operation,
            event.new_generation_chunk_set_fingerprint,
        ):
            self.assertNotIn("after body", value)
            self.assertNotIn("before body", value)

    def test_a_generator_version_change_is_recorded(self):
        document = make_derived(
            self.collection, "Version Bump",
            version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION - 1,
        )
        regenerate_derived_chunk_set(
            document.pk, expected=expected_for(document), principal=a_principal(),
        )
        event = KnowledgeLifecycleEvent.objects.filter(document=document).get()
        self.assertEqual(
            event.previous_generator_version, GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION - 1
        )
        self.assertEqual(
            event.new_generator_version, GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION
        )


# ---------------------------------------------------------------------------
# Security / Agent isolation
# ---------------------------------------------------------------------------

class RegenerationSecurityBoundaryTests(TestCase):
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
    REGENERATION_SYMBOLS = (
        "knowledge_regeneration",
        "regenerate_derived_chunk_set",
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

    def test_no_agent_facing_module_imports_regeneration(self):
        for relative in self.AGENT_FACING_MODULES:
            with self.subTest(module=relative):
                source = self._module_path(relative).read_text(encoding="utf-8")
                for symbol in self.REGENERATION_SYMBOLS:
                    self.assertNotIn(symbol, source)

    def test_no_admin_or_build_console_integration_exists(self):
        for relative in self.DEFERRED_SURFACE_MODULES:
            with self.subTest(module=relative):
                source = self._module_path(relative).read_text(encoding="utf-8")
                for symbol in self.REGENERATION_SYMBOLS:
                    self.assertNotIn(symbol, source)

    def test_no_seeded_tool_definition_exposes_regeneration(self):
        seed_starter_toolboxes()
        self.assertGreater(ToolDefinition.objects.count(), 0)
        for tool in ToolDefinition.objects.all():
            with self.subTest(tool=tool.name):
                blob = " ".join(
                    str(p) for p in (tool.name, tool.label, tool.description, tool.config)
                ).lower()
                for forbidden in ("regenerat", "rechunk", "knowledge_regeneration"):
                    self.assertNotIn(forbidden, blob)

    def test_no_management_command_invokes_regeneration(self):
        """No operator surface in this slice, deliberately."""
        import pathlib

        commands = pathlib.Path(__file__).resolve().parent / "management" / "commands"
        for path in sorted(commands.glob("*.py")):
            with self.subTest(command=path.name):
                source = path.read_text(encoding="utf-8")
                for symbol in self.REGENERATION_SYMBOLS:
                    self.assertNotIn(symbol, source)

    def test_the_public_surface_is_exactly_one_operation(self):
        import inspect

        from ai_hub.services import knowledge_regeneration

        functions = {
            name for name, obj in vars(knowledge_regeneration).items()
            if not name.startswith("_")
            and inspect.isfunction(obj)
            and obj.__module__ == knowledge_regeneration.__name__
        }
        self.assertEqual(functions, {"regenerate_derived_chunk_set"})

    def test_regeneration_does_not_reimplement_the_mutation_foundation(self):
        """AST-checked, so docstring prose cannot pass or fail it."""
        import ast
        import pathlib

        tree = ast.parse(
            (
                pathlib.Path(__file__).resolve().parent
                / "services" / "knowledge_regeneration.py"
            ).read_text(encoding="utf-8")
        )
        referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        referenced |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        imported = {
            alias.name for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) for alias in node.names
        }
        for symbol in ("atomic", "select_for_update", "KnowledgeLifecycleEvent",
                       "verify_expected_state"):
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, referenced)
                self.assertNotIn(symbol, imported)
        self.assertIn("_governed_knowledge_mutation", imported)

    def test_no_bulk_or_repair_verb_exists(self):
        from ai_hub.services import knowledge_regeneration

        for forbidden in ("backfill", "bulk", "repair", "batch", "all_", "adjudicate"):
            with self.subTest(name=forbidden):
                self.assertFalse(
                    any(n.startswith(forbidden) for n in dir(knowledge_regeneration)),
                    f"{forbidden!r} belongs to a later slice",
                )


# ---------------------------------------------------------------------------
# Nothing else moved
# ---------------------------------------------------------------------------

class RegenerationLeavesEverythingElseAloneTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Isolation")
        self.document = make_derived(self.collection, "Isolated", curated_text="a")
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(curated_text="b")
        self.document.refresh_from_db()

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

    def test_other_documents_are_untouched(self):
        bystander = make_derived(self.collection, "Bystander")
        before = chunk_rows(bystander)
        regenerate_derived_chunk_set(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        self.assertEqual(chunk_rows(bystander), before)

    def test_the_compatibility_writer_still_produces_unknown(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Fresh", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        ensure_initial_knowledge_chunk(document)
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, MODES.UNKNOWN)

    def test_preflight_remains_read_only_and_idempotent(self):
        regenerate_derived_chunk_set(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        first = run_knowledge_preflight()
        events = KnowledgeLifecycleEvent.objects.count()
        self.assertEqual(first, run_knowledge_preflight())
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), events)

    def test_document_status_and_collection_are_never_changed(self):
        before = KnowledgeDocument.objects.filter(pk=self.document.pk).values(
            "status", "collection_id", "title", "curated_text", "notes"
        )[0]
        regenerate_derived_chunk_set(
            self.document.pk, expected=expected_for(self.document), principal=a_principal(),
        )
        after = KnowledgeDocument.objects.filter(pk=self.document.pk).values(
            "status", "collection_id", "title", "curated_text", "notes"
        )[0]
        self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# PostgreSQL — the first governed operation that deletes chunk rows
# ---------------------------------------------------------------------------

class RegenerationConcurrencyTests(TransactionTestCase):
    """Racing regenerations on the real delete/insert path.

    SQLite has no row locks, so it cannot evidence this; the test skips there
    following the repository's established convention and is validated by the
    PostgreSQL 16 CI job.
    """

    reset_sequences = True

    def test_racing_regenerations_produce_one_winner_and_no_partial_chunk_set(self):
        if not connection.features.has_select_for_update:
            self.skipTest(
                "SQLite cannot validate select_for_update locking semantics; run "
                "this test on PostgreSQL CI."
            )

        collection = KnowledgeCollection.objects.create(name="Regen Race")
        document = make_derived(collection, "Contended", curated_text="original body")
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            curated_text="a materially different body"
        )
        document.refresh_from_db()
        expected = expected_for(document)

        def attempt(index):
            close_old_connections()
            try:
                regenerate_derived_chunk_set(
                    document.pk, expected=expected,
                    principal=KnowledgeMutationPrincipal.system(f"racer-{index}"),
                )
                return "committed"
            except KnowledgeMutationConflict:
                # ONLY a stale-review conflict counts as losing this race. The
                # broader KnowledgeRegenerationError family is deliberately NOT
                # caught: both racers start from a valid, regenerable state, so
                # the loser must fail because its reviewed snapshot went stale -
                # not "somehow". Anything else propagates and fails the test.
                return "lost_cas"
            finally:
                close_old_connections()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(attempt, range(2)))

        self.assertEqual(results.count("committed"), 1)
        self.assertEqual(results.count("lost_cas"), 1)

        # Exactly one transition, and a coherent chunk set - not a partial one.
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 1)
        chunks = list(document.chunks.order_by("chunk_index"))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "a materially different body")
        indexes = [c.chunk_index for c in chunks]
        self.assertEqual(len(indexes), len(set(indexes)), "duplicate (document, chunk_index)")

        # The loser did not overwrite the winner.
        document.refresh_from_db()
        self.assertEqual(
            document.generation_chunk_set_fingerprint,
            document_chunk_set_fingerprint(document),
        )
