"""Tests for the governed Knowledge mutation + lifecycle audit foundation.

Slice 9, RC-002 step 3c. This module proves the MECHANISM, not any domain
operation: there is no adjudication, repair, regeneration or backfill to test,
deliberately. What is tested is that when a lifecycle change does eventually
happen it will be atomic, stale-review-safe, principal-bound and durably
auditable — and that nothing here leaked into retrieval, the Agent surface or
the existing write paths.
"""
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from django.db import close_old_connections, connection, models, transaction
from django.test import TestCase, TransactionTestCase

from ai_hub.models import (
    AgentProfile,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    KnowledgeLifecycleEvent,
    ToolDefinition,
)
from ai_hub.services.knowledge_ingestion import ensure_initial_knowledge_chunk
from ai_hub.services.knowledge_mutation import (
    CAS_FIELDS,
    ExpectedKnowledgeState,
    KnowledgeMutationConflict,
    KnowledgeMutationOperationError,
    KnowledgeMutationPrincipal,
    KnowledgeMutationPrincipalError,
    KnowledgeMutationSnapshot,
    UnauditableMutationError,
    _governed_knowledge_mutation,
    build_snapshot,
    verify_expected_state,
)
from ai_hub.services.knowledge_preflight import run_knowledge_preflight
from ai_hub.services.starter_toolboxes import seed_starter_toolboxes

MODES = KnowledgeDocument.ChunkAuthorityMode

# A slug used only to exercise the foundation. It is deliberately defined in the
# test module rather than in production code: no real lifecycle operation exists
# yet, and inventing one to make the tests read nicely would ship step 3d early.
TEST_OPERATION = "foundation_probe"


def make_document(collection, title="Doc", *, curated_text="body", chunks=((1, "chunk body"),), **kwargs):
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


def a_principal():
    return KnowledgeMutationPrincipal.human("operator-1")


def expected_for(document):
    """The FULL reviewed state, which is the only supported expectation."""
    return ExpectedKnowledgeState.from_snapshot(build_snapshot(document))


# ---------------------------------------------------------------------------
# A / B — the event schema is reference-first and cannot hold a Knowledge body
# ---------------------------------------------------------------------------

class LifecycleEventSchemaTests(TestCase):
    """The audit row records facts and references. Never content.

    Enforced structurally rather than by review habit: a future field that
    could hold a body fails these tests instead of quietly shipping.
    """

    FIELDS = {f.name: f for f in KnowledgeLifecycleEvent._meta.get_fields() if hasattr(f, "attname")}

    def test_no_unbounded_text_or_json_field_exists(self):
        for name, field in self.FIELDS.items():
            with self.subTest(field=name):
                self.assertNotIsInstance(field, models.TextField)
                self.assertNotIsInstance(field, models.JSONField)
                self.assertNotIsInstance(field, models.FileField)
                self.assertNotIsInstance(field, models.BinaryField)

    def test_every_character_field_is_tightly_bounded(self):
        """No CharField is wide enough to be abused as a content column."""
        for name, field in self.FIELDS.items():
            if isinstance(field, models.CharField):
                with self.subTest(field=name):
                    self.assertLessEqual(field.max_length, 150)

    def test_no_field_name_suggests_a_knowledge_body(self):
        """Substring match, so `chunk_content_copy` fails too, not just `content`."""
        forbidden = (
            "curated", "content", "body", "text", "snippet", "excerpt",
            "payload", "source_file", "blob", "data",
        )
        for name in self.FIELDS:
            for token in forbidden:
                with self.subTest(field=name, token=token):
                    self.assertNotIn(token, name)

    def test_the_whole_row_is_too_narrow_to_hold_a_document(self):
        """Every character column summed is far smaller than any real body."""
        total = sum(
            field.max_length for field in self.FIELDS.values()
            if isinstance(field, models.CharField)
        )
        self.assertLess(total, 1200)

    def test_no_field_stores_personal_contact_detail(self):
        for name in self.FIELDS:
            with self.subTest(field=name):
                for pii in ("email", "phone", "address", "ip_address"):
                    self.assertNotIn(pii, name)

    def test_the_answerable_questions_are_all_present(self):
        """WHAT changed, WHAT KIND, FROM what, TO what, WHO, WHEN."""
        for required in (
            "document", "document_id_snapshot", "collection_id_snapshot",
            "operation", "principal_kind", "principal_identifier",
            "previous_authority_mode", "new_authority_mode",
            "previous_generation_input_fingerprint", "new_generation_input_fingerprint",
            "previous_generation_chunk_set_fingerprint", "new_generation_chunk_set_fingerprint",
            "previous_generator_identity", "new_generator_identity",
            "previous_generator_version", "new_generator_version",
            "previous_chunk_count", "new_chunk_count",
            "created_at",
        ):
            with self.subTest(field=required):
                self.assertIn(required, self.FIELDS)

    def test_durable_id_snapshots_match_the_bigautofield_domain(self):
        """DEFAULT_AUTO_FIELD is BigAutoField, so these ids are 64-bit.

        A durable snapshot must never be narrower than the identifier it copies:
        the snapshot's whole purpose is to stay valid after the FK is gone, and
        a 32-bit column would begin failing on a long-lived corpus exactly when
        the history matters most.
        """
        for name in ("document_id_snapshot", "collection_id_snapshot"):
            with self.subTest(field=name):
                field = KnowledgeLifecycleEvent._meta.get_field(name)
                self.assertIsInstance(field, models.PositiveBigIntegerField)
                self.assertNotIsInstance(field, models.PositiveSmallIntegerField)

    def test_snapshot_columns_are_at_least_as_wide_as_the_pks_they_copy(self):
        pairs = (
            ("document_id_snapshot", KnowledgeDocument),
            ("collection_id_snapshot", KnowledgeCollection),
        )
        for name, model in pairs:
            with self.subTest(field=name):
                snapshot_range = connection.ops.integer_field_range(
                    KnowledgeLifecycleEvent._meta.get_field(name).get_internal_type()
                )
                pk_range = connection.ops.integer_field_range(
                    model._meta.pk.get_internal_type()
                )
                if snapshot_range[1] is not None and pk_range[1] is not None:
                    self.assertGreaterEqual(snapshot_range[1], pk_range[1])

    def test_migration_uses_the_wide_field_too(self):
        """The model and the migration must not disagree."""
        from django.db.migrations.loader import MigrationLoader

        migration = MigrationLoader(None, load=True).disk_migrations[
            ("ai_hub", "0022_knowledge_lifecycle_event")
        ]
        create = next(
            op for op in migration.operations if op.__class__.__name__ == "CreateModel"
        )
        declared = dict(create.fields)
        for name in ("document_id_snapshot", "collection_id_snapshot"):
            with self.subTest(field=name):
                self.assertIsInstance(declared[name], models.PositiveBigIntegerField)

    def test_no_principal_label_column_exists(self):
        """Data minimization: a display name would freeze PII into permanent
        history for no correctness benefit. Hosts resolve names at display
        time from the opaque identifier instead."""
        self.assertNotIn("principal_label", self.FIELDS)
        for name in self.FIELDS:
            with self.subTest(field=name):
                for token in ("label", "name", "display", "username", "full_name"):
                    self.assertNotIn(token, name)

    def test_durable_history_lookup_is_indexed(self):
        """After deletion the FK is NULL, so the snapshot id is the only way to
        ask "what happened to document 42?". It must not table-scan."""
        index_fields = [tuple(i.fields) for i in KnowledgeLifecycleEvent._meta.indexes]
        self.assertIn(("document_id_snapshot", "created_at"), index_fields)
        self.assertIn(("operation", "created_at"), index_fields)

    def test_the_index_set_is_small_and_not_speculative(self):
        self.assertEqual(len(KnowledgeLifecycleEvent._meta.indexes), 2)

    def test_status_transitions_are_auditable(self):
        """Status gates retrievability (D-L3), so a governed transition must not
        commit unexplained."""
        self.assertIn("previous_status", self.FIELDS)
        self.assertIn("new_status", self.FIELDS)

    def test_observed_input_fingerprints_are_recorded_both_sides(self):
        """Title / curated_text are not copied, but a change to either moves
        `i1`, so the event can still state truthfully that inputs moved."""
        self.assertIn("previous_observed_input_fingerprint", self.FIELDS)
        self.assertIn("new_observed_input_fingerprint", self.FIELDS)

    def test_operation_has_no_model_choices(self):
        """The vocabulary grows in 3d/3e; `choices` would churn migrations."""
        self.assertIsNone(KnowledgeLifecycleEvent._meta.get_field("operation").choices)

    def test_principal_kind_is_a_small_closed_vocabulary(self):
        self.assertEqual(
            set(KnowledgeLifecycleEvent.PrincipalKind.values), {"human", "system"}
        )

    def test_no_lifecycle_check_constraint_was_added(self):
        """Preflight V2 must stay able to OBSERVE an inconsistent state."""
        self.assertEqual(KnowledgeDocument._meta.constraints, [])


# ---------------------------------------------------------------------------
# C — audit durability across document deletion
# ---------------------------------------------------------------------------

class LifecycleEventDurabilityTests(TestCase):
    """History must not vanish because the document later did."""

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Durability")
        self.document = make_document(self.collection)

    def test_event_survives_document_deletion_and_stays_intelligible(self):
        with _governed_knowledge_mutation(
            self.document.pk, expected=expected_for(self.document),
            operation=TEST_OPERATION, principal=a_principal(),
        ) as mutation:
            mutation.document.notes = "touched"
            mutation.document.save(update_fields=["notes"])

        event = KnowledgeLifecycleEvent.objects.get()
        document_id, collection_id = self.document.pk, self.collection.pk

        self.document.delete()
        event.refresh_from_db()

        self.assertIsNone(event.document_id)
        self.assertEqual(event.document_id_snapshot, document_id)
        self.assertEqual(event.collection_id_snapshot, collection_id)
        self.assertEqual(event.operation, TEST_OPERATION)
        self.assertEqual(event.previous_authority_mode, MODES.UNKNOWN)

    def test_fk_is_set_null_not_cascade(self):
        field = KnowledgeLifecycleEvent._meta.get_field("document")
        self.assertIs(field.remote_field.on_delete, models.SET_NULL)


# ---------------------------------------------------------------------------
# D–I — snapshot determinism and token sensitivity
# ---------------------------------------------------------------------------

class MutationSnapshotTests(TestCase):
    """The snapshot is a stale-review token. It must move exactly when the
    evidence moves, and never when it does not."""

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Snapshot")
        self.document = make_document(self.collection, "Title A", curated_text="source body")

    def _snapshot(self):
        return build_snapshot(KnowledgeDocument.objects.get(pk=self.document.pk))

    def test_snapshot_is_deterministic_on_unchanged_state(self):
        self.assertEqual(self._snapshot(), self._snapshot())

    def test_snapshot_is_immutable(self):
        with self.assertRaises(Exception):
            self._snapshot().authority_mode = MODES.DERIVED

    def test_snapshot_carries_the_contract_that_produced_the_input_digest(self):
        """A fingerprint is meaningless without the contract behind it."""
        self.assertEqual(self._snapshot().observed_input_generator, "curated_text_single_chunk")

    def test_snapshot_records_claims_and_measurements_separately(self):
        """A recorded fingerprint is a claim; an observed one is evidence."""
        snapshot = self._snapshot()
        self.assertEqual(snapshot.recorded_chunk_set_fingerprint, "")
        self.assertTrue(snapshot.observed_chunk_set_fingerprint.startswith("c1:"))
        self.assertTrue(snapshot.observed_input_fingerprint.startswith("i1:"))

    def test_unknown_document_gets_a_neutral_fingerprint_not_a_reclassification(self):
        snapshot = self._snapshot()
        self.assertEqual(snapshot.authority_mode, MODES.UNKNOWN)
        self.assertTrue(snapshot.observed_input_fingerprint)
        self.assertEqual(snapshot.observed_input_generator, "curated_text_single_chunk")
        self.document.refresh_from_db()
        self.assertEqual(self.document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(self.document.generation_input_fingerprint, "")

    def test_chunk_content_change_moves_the_chunk_set_token(self):
        before = self._snapshot()
        chunk = self.document.chunks.get(chunk_index=1)
        chunk.content = "edited outside the governed boundary"
        chunk.save(update_fields=["content"])
        after = self._snapshot()
        self.assertNotEqual(before.observed_chunk_set_fingerprint, after.observed_chunk_set_fingerprint)
        self.assertEqual(before.observed_input_fingerprint, after.observed_input_fingerprint)

    def test_metadata_only_chunk_change_does_not_move_the_chunk_set_token(self):
        """`c1` covers retrieval-visible facts. Metadata is not one of them."""
        before = self._snapshot()
        chunk = self.document.chunks.get(chunk_index=1)
        chunk.metadata = {"ingestion": "curated_text"}
        chunk.save(update_fields=["metadata"])
        self.assertEqual(before.observed_chunk_set_fingerprint, self._snapshot().observed_chunk_set_fingerprint)

    def test_title_change_moves_the_input_token(self):
        """The title becomes `section_title`, so it IS a generation input."""
        before = self._snapshot()
        self.document.title = "Title B"
        self.document.save(update_fields=["title"])
        after = self._snapshot()
        self.assertNotEqual(before.observed_input_fingerprint, after.observed_input_fingerprint)
        self.assertEqual(before.observed_chunk_set_fingerprint, after.observed_chunk_set_fingerprint)

    def test_material_curated_text_change_moves_the_input_token(self):
        before = self._snapshot()
        self.document.curated_text = "a materially different source body"
        self.document.save(update_fields=["curated_text"])
        self.assertNotEqual(before.observed_input_fingerprint, self._snapshot().observed_input_fingerprint)

    def test_outer_whitespace_only_curated_text_change_does_not_move_the_input_token(self):
        """`i1` strips `curated_text` outer whitespace, because the generator does."""
        before = self._snapshot()
        self.document.curated_text = "\n\n  source body  \t\n"
        self.document.save(update_fields=["curated_text"])
        self.assertEqual(before.observed_input_fingerprint, self._snapshot().observed_input_fingerprint)

    def test_inner_whitespace_change_does_move_the_input_token(self):
        before = self._snapshot()
        self.document.curated_text = "source    body"
        self.document.save(update_fields=["curated_text"])
        self.assertNotEqual(before.observed_input_fingerprint, self._snapshot().observed_input_fingerprint)

    def test_building_a_snapshot_writes_nothing(self):
        before = _snapshot_database()
        build_snapshot(KnowledgeDocument.objects.get(pk=self.document.pk))
        self.assertEqual(before, _snapshot_database())


# ---------------------------------------------------------------------------
# J–M — compare-and-swap / stale review
# ---------------------------------------------------------------------------

class StaleReviewProtectionTests(TestCase):
    """The decision must apply to the EXACT state that was reviewed.

    Backend-independent: pure application logic, identical on SQLite and
    PostgreSQL.

    Fingerprints alone are not enough. Each test below changes exactly ONE fact
    after the review and asserts a conflict — including the facts that leave
    `i1` and `c1` completely untouched, which an earlier fingerprint-only design
    would have missed.
    """

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="CAS")
        self.other_collection = KnowledgeCollection.objects.create(name="CAS Other")
        self.document = make_document(self.collection, "CAS Doc", curated_text="source")

    def _assert_conflict_on(self, expected, field):
        """The body must never run, and nothing may be written."""
        before = _snapshot_database()
        with self.assertRaises(KnowledgeMutationConflict) as caught:
            with _governed_knowledge_mutation(
                self.document.pk, expected=expected,
                operation=TEST_OPERATION, principal=a_principal(),
            ) as mutation:
                raise AssertionError("body must never run on a stale expectation")
        self.assertEqual(caught.exception.field, field)
        self.assertEqual(caught.exception.document_id, self.document.pk)
        self.assertEqual(before, _snapshot_database())
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_matching_expected_state_passes(self):
        verify_expected_state(build_snapshot(self.document), expected_for(self.document))

    def test_expectation_binds_every_correctness_relevant_fact(self):
        """The expectation and the snapshot must never drift apart."""
        snapshot_fields = {f.name for f in dataclasses.fields(KnowledgeMutationSnapshot)}
        expected_fields = {f.name for f in dataclasses.fields(ExpectedKnowledgeState)}
        self.assertEqual(expected_fields, set(CAS_FIELDS))
        self.assertEqual(snapshot_fields, set(CAS_FIELDS))

    def test_no_expectation_field_is_optional(self):
        """A caller must not be able to silently downgrade to a partial CAS."""
        for field in dataclasses.fields(ExpectedKnowledgeState):
            with self.subTest(field=field.name):
                self.assertIs(field.default, dataclasses.MISSING)
                self.assertIs(field.default_factory, dataclasses.MISSING)

    def test_from_snapshot_is_the_canonical_constructor(self):
        snapshot = build_snapshot(self.document)
        self.assertEqual(
            ExpectedKnowledgeState.from_snapshot(snapshot),
            ExpectedKnowledgeState(**{n: getattr(snapshot, n) for n in CAS_FIELDS}),
        )
        with self.assertRaises(TypeError):
            ExpectedKnowledgeState.from_snapshot({"document_id": 1})

    # -- facts that move NEITHER fingerprint --------------------------------

    def test_collection_move_conflicts_even_though_i1_and_c1_are_unchanged(self):
        """The hazard that motivated the full-state CAS.

        Same title, same curated_text, same chunks — but the retrieval
        authorization boundary moved, so the reviewed decision no longer applies.
        """
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            collection=self.other_collection
        )
        after = build_snapshot(KnowledgeDocument.objects.get(pk=self.document.pk))
        self.assertEqual(expected.observed_input_fingerprint, after.observed_input_fingerprint)
        self.assertEqual(
            expected.observed_chunk_set_fingerprint, after.observed_chunk_set_fingerprint
        )
        self._assert_conflict_on(expected, "collection_id")

    def test_status_change_conflicts(self):
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            status=KnowledgeDocument.Status.ARCHIVED
        )
        self._assert_conflict_on(expected, "status")

    def test_authority_mode_change_conflicts(self):
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            chunk_authority_mode=MODES.EXPLICIT
        )
        self._assert_conflict_on(expected, "authority_mode")

    def test_recorded_input_fingerprint_change_conflicts(self):
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            generation_input_fingerprint="i1:" + "0" * 64
        )
        self._assert_conflict_on(expected, "recorded_input_fingerprint")

    def test_recorded_chunk_set_fingerprint_change_conflicts(self):
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            generation_chunk_set_fingerprint="c1:" + "0" * 64
        )
        self._assert_conflict_on(expected, "recorded_chunk_set_fingerprint")

    def test_generator_identity_change_conflicts(self):
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            generator_identity="some_other_generator"
        )
        self._assert_conflict_on(expected, "generator_identity")

    def test_generator_version_change_conflicts(self):
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(generator_version=7)
        self._assert_conflict_on(expected, "generator_version")

    def test_chunk_count_change_conflicts(self):
        expected = expected_for(self.document)
        KnowledgeDocumentChunk.objects.create(
            document=self.document, chunk_index=2, section_title="S2", content="added",
        )
        self._assert_conflict_on(expected, "chunk_count")

    # -- facts that move a fingerprint --------------------------------------

    def test_title_change_conflicts_through_i1(self):
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(title="Renamed")
        self._assert_conflict_on(expected, "observed_input_fingerprint")

    def test_material_curated_text_change_conflicts_through_i1(self):
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            curated_text="a materially different source"
        )
        self._assert_conflict_on(expected, "observed_input_fingerprint")

    def test_chunk_content_change_conflicts_through_c1(self):
        expected = expected_for(self.document)
        chunk = self.document.chunks.get(chunk_index=1)
        chunk.content = "changed after review"
        chunk.save(update_fields=["content"])
        self._assert_conflict_on(expected, "observed_chunk_set_fingerprint")

    # -- and the things that must NOT conflict ------------------------------

    def test_chunk_metadata_only_change_does_not_conflict(self):
        """`c1` covers retrieval-visible facts; metadata is not one of them."""
        expected = expected_for(self.document)
        chunk = self.document.chunks.get(chunk_index=1)
        chunk.metadata = {"ingestion": "curated_text"}
        chunk.save(update_fields=["metadata"])

        with _governed_knowledge_mutation(
            self.document.pk, expected=expected,
            operation=TEST_OPERATION, principal=a_principal(),
        ) as mutation:
            mutation.document.notes = "ok"
            mutation.document.save(update_fields=["notes"])
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 1)

    def test_outer_whitespace_only_curated_text_change_does_not_conflict(self):
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            curated_text="\n\n  source  \t\n"
        )
        with _governed_knowledge_mutation(
            self.document.pk, expected=expected,
            operation=TEST_OPERATION, principal=a_principal(),
        ) as mutation:
            mutation.document.notes = "ok"
            mutation.document.save(update_fields=["notes"])
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 1)

    def test_updated_at_is_not_the_correctness_token(self):
        """Slice 4 measured that timestamps fail in both directions.

        `notes` is not a correctness-relevant fact, so an edit that only moves
        `updated_at` must still permit the reviewed decision.
        """
        expected = expected_for(self.document)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(notes="unrelated edit")
        with _governed_knowledge_mutation(
            self.document.pk, expected=expected,
            operation=TEST_OPERATION, principal=a_principal(),
        ) as mutation:
            mutation.document.notes = "governed"
            mutation.document.save(update_fields=["notes"])
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 1)

    def test_conflict_is_never_silently_recovered(self):
        expected = expected_for(self.document)
        self.document.chunks.get(chunk_index=1).delete()
        with self.assertRaises(KnowledgeMutationConflict) as caught:
            with _governed_knowledge_mutation(
                self.document.pk, expected=expected,
                operation=TEST_OPERATION, principal=a_principal(),
            ):
                pass
        self.assertIn("decide again", str(caught.exception))

    def test_a_non_expected_state_is_rejected_outright(self):
        with self.assertRaises(TypeError):
            verify_expected_state(build_snapshot(self.document), {"chunk_count": 1})


class UnauditableMutationTests(TestCase):
    """The foundation refuses a change its event row cannot describe."""

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Auditable")
        self.other = KnowledgeCollection.objects.create(name="Auditable Other")
        self.document = make_document(self.collection, "Auditable Doc")

    def test_a_collection_move_inside_a_governed_mutation_is_refused(self):
        """Not mis-recorded, refused.

        The event carries no before/after collection, and a move changes the
        retrieval authorization boundary. A governed collection move is a
        separate, security-sensitive future operation that must extend the
        schema first.
        """
        before = _snapshot_database()
        with self.assertRaises(UnauditableMutationError) as caught:
            with _governed_knowledge_mutation(
                self.document.pk, expected=expected_for(self.document),
                operation=TEST_OPERATION, principal=a_principal(),
            ) as mutation:
                mutation.document.collection = self.other
                mutation.document.save(update_fields=["collection"])

        self.assertIn("collection move", str(caught.exception))
        self.assertEqual(before, _snapshot_database())
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)


# ---------------------------------------------------------------------------
# N / O — atomicity of mutation + event
# ---------------------------------------------------------------------------

class MutationEventAtomicityTests(TestCase):
    """Neither half may survive the other."""

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Atomic")
        self.document = make_document(self.collection, "Atomic Doc", curated_text="source")

    def test_successful_mutation_records_exactly_one_event_with_before_and_after(self):
        before_snapshot = build_snapshot(self.document)
        with _governed_knowledge_mutation(
            self.document.pk, expected=expected_for(self.document),
            operation=TEST_OPERATION, principal=a_principal(), reason_code="probe_ok",
        ) as mutation:
            mutation.document.chunk_authority_mode = MODES.EXPLICIT
            mutation.document.save(update_fields=["chunk_authority_mode"])

        self.document.refresh_from_db()
        self.assertEqual(self.document.chunk_authority_mode, MODES.EXPLICIT)

        event = KnowledgeLifecycleEvent.objects.get()
        self.assertEqual(event.document_id, self.document.pk)
        self.assertEqual(event.operation, TEST_OPERATION)
        self.assertEqual(event.reason_code, "probe_ok")
        self.assertEqual(event.previous_authority_mode, MODES.UNKNOWN)
        self.assertEqual(event.new_authority_mode, MODES.EXPLICIT)
        self.assertEqual(event.previous_chunk_count, 1)
        self.assertEqual(event.new_chunk_count, 1)
        self.assertEqual(
            event.previous_observed_chunk_set_fingerprint,
            before_snapshot.observed_chunk_set_fingerprint,
        )
        self.assertEqual(
            event.previous_observed_input_fingerprint,
            before_snapshot.observed_input_fingerprint,
        )
        self.assertEqual(event.previous_status, KnowledgeDocument.Status.ACTIVE)
        self.assertEqual(event.new_status, KnowledgeDocument.Status.ACTIVE)
        self.assertIsNotNone(event.created_at)

    def test_event_creation_failure_rolls_back_the_knowledge_mutation(self):
        before = _snapshot_database()
        with mock.patch(
            "ai_hub.services.knowledge_mutation.KnowledgeLifecycleEvent.objects.create",
            side_effect=RuntimeError("audit write failed"),
        ):
            with self.assertRaises(RuntimeError):
                with _governed_knowledge_mutation(
                    self.document.pk, expected=expected_for(self.document),
                    operation=TEST_OPERATION, principal=a_principal(),
                ) as mutation:
                    mutation.document.chunk_authority_mode = MODES.EXPLICIT
                    mutation.document.save(update_fields=["chunk_authority_mode"])

        self.document.refresh_from_db()
        self.assertEqual(self.document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)
        self.assertEqual(before, _snapshot_database())

    def test_mutation_failure_leaves_no_committed_event(self):
        with self.assertRaises(ValueError):
            with _governed_knowledge_mutation(
                self.document.pk, expected=expected_for(self.document),
                operation=TEST_OPERATION, principal=a_principal(),
            ) as mutation:
                mutation.document.chunk_authority_mode = MODES.DERIVED
                mutation.document.save(update_fields=["chunk_authority_mode"])
                raise ValueError("domain rule rejected the change")

        self.document.refresh_from_db()
        self.assertEqual(self.document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_after_facts_are_re_read_not_taken_from_memory(self):
        """A queryset update bypasses the in-memory instance; AFTER must still
        describe what actually committed."""
        with _governed_knowledge_mutation(
            self.document.pk, expected=expected_for(self.document),
            operation=TEST_OPERATION, principal=a_principal(),
        ) as mutation:
            KnowledgeDocument.objects.filter(pk=mutation.document.pk).update(
                chunk_authority_mode=MODES.EXPLICIT
            )
        self.assertEqual(KnowledgeLifecycleEvent.objects.get().new_authority_mode, MODES.EXPLICIT)

    def test_chunk_count_change_is_recorded(self):
        with _governed_knowledge_mutation(
            self.document.pk, expected=expected_for(self.document),
            operation=TEST_OPERATION, principal=a_principal(),
        ) as mutation:
            KnowledgeDocumentChunk.objects.create(
                document=mutation.document, chunk_index=2,
                section_title="Section 2", content="second chunk",
            )
        event = KnowledgeLifecycleEvent.objects.get()
        self.assertEqual(event.previous_chunk_count, 1)
        self.assertEqual(event.new_chunk_count, 2)
        self.assertNotEqual(
            event.previous_observed_chunk_set_fingerprint,
            event.new_observed_chunk_set_fingerprint,
        )

    def test_the_body_runs_inside_a_transaction(self):
        with _governed_knowledge_mutation(
            self.document.pk, expected=expected_for(self.document),
            operation=TEST_OPERATION, principal=a_principal(),
        ) as mutation:
            self.assertTrue(transaction.get_connection().in_atomic_block)
            mutation.document.notes = "x"
            mutation.document.save(update_fields=["notes"])


# ---------------------------------------------------------------------------
# Q — principal contract
# ---------------------------------------------------------------------------

class MutationPrincipalTests(TestCase):
    """Lifecycle mutation is operator/system activity, never Agent identity."""

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Principal")
        self.document = make_document(self.collection)

    def test_principal_is_immutable(self):
        with self.assertRaises(Exception):
            a_principal().identifier = "someone-else"

    def test_principal_carries_no_display_label(self):
        """Audit rows live forever; a name field is a permanent PII path."""
        principal = a_principal()
        self.assertEqual(
            {f.name for f in dataclasses.fields(principal)}, {"kind", "identifier"}
        )
        with self.assertRaises(TypeError):
            KnowledgeMutationPrincipal.human("u-1", label="Someone Real")

    def test_principal_has_no_agent_coupling(self):
        principal = a_principal()
        self.assertFalse(hasattr(principal, "agent"))
        self.assertFalse(hasattr(principal, "agent_id"))
        for name in vars(principal):
            with self.subTest(field=name):
                self.assertNotIn("agent", name)

    def test_event_model_has_no_agent_or_session_foreign_key(self):
        """Corpus mutation may have no Agent, session, step or tool at all."""
        related = {
            f.name for f in KnowledgeLifecycleEvent._meta.get_fields()
            if getattr(f, "related_model", None) is not None
        }
        self.assertEqual(related, {"document"})
        self.assertNotIn(
            AgentProfile,
            {
                getattr(f, "related_model", None)
                for f in KnowledgeLifecycleEvent._meta.get_fields()
            },
        )

    def test_both_principal_kinds_are_supported(self):
        for principal in (
            KnowledgeMutationPrincipal.human("u-1"),
            KnowledgeMutationPrincipal.system("nightly-job"),
        ):
            with self.subTest(kind=principal.kind):
                self.assertEqual(principal.validate().kind, principal.kind)

    def test_anonymous_governed_writes_are_rejected(self):
        for bad in (
            KnowledgeMutationPrincipal.human(""),
            KnowledgeMutationPrincipal.human("   "),
            KnowledgeMutationPrincipal(kind="agent", identifier="agent-7"),
        ):
            with self.subTest(principal=bad):
                with self.assertRaises(KnowledgeMutationPrincipalError):
                    with _governed_knowledge_mutation(
                        self.document.pk, expected=expected_for(self.document),
                        operation=TEST_OPERATION, principal=bad,
                    ):
                        pass
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_missing_principal_is_rejected(self):
        with self.assertRaises(KnowledgeMutationPrincipalError):
            with _governed_knowledge_mutation(
                self.document.pk, expected=expected_for(self.document),
                operation=TEST_OPERATION, principal=None,
            ):
                pass

    def test_principal_identifier_is_bounded(self):
        with self.assertRaises(KnowledgeMutationPrincipalError):
            KnowledgeMutationPrincipal.human("x" * 151).validate()

    def test_event_records_only_kind_and_identifier_for_the_principal(self):
        with _governed_knowledge_mutation(
            self.document.pk, expected=expected_for(self.document),
            operation=TEST_OPERATION, principal=KnowledgeMutationPrincipal.system("job-9"),
        ) as mutation:
            mutation.document.notes = "x"
            mutation.document.save(update_fields=["notes"])
        event = KnowledgeLifecycleEvent.objects.get()
        self.assertEqual(event.principal_kind, "system")
        self.assertEqual(event.principal_identifier, "job-9")

    def test_operation_slug_is_validated(self):
        for bad in ("", "X", "Bad Operation", "has spaces", "UPPER", "a" * 65, "1leading"):
            with self.subTest(operation=bad):
                with self.assertRaises(KnowledgeMutationOperationError):
                    with _governed_knowledge_mutation(
                        self.document.pk, expected=expected_for(self.document),
                        operation=bad, principal=a_principal(),
                    ):
                        pass
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_reason_code_is_validated_but_optional(self):
        with self.assertRaises(KnowledgeMutationOperationError):
            with _governed_knowledge_mutation(
                self.document.pk, expected=expected_for(self.document),
                operation=TEST_OPERATION, principal=a_principal(),
                reason_code="Not A Slug",
            ):
                pass


# ---------------------------------------------------------------------------
# R — security boundary
# ---------------------------------------------------------------------------

class MutationSecurityBoundaryTests(TestCase):
    """Operator/system infrastructure, never an Agent capability."""

    def test_no_seeded_tool_definition_exposes_the_mutation_foundation(self):
        """Seed the real starter tool set, then prove none of it reaches here."""
        seed_starter_toolboxes()
        self.assertGreater(ToolDefinition.objects.count(), 0)
        for tool in ToolDefinition.objects.all():
            with self.subTest(tool=tool.name):
                blob = " ".join(
                    str(part) for part in (tool.name, tool.label, tool.description, tool.config)
                ).lower()
                for forbidden in (
                    "knowledge_mutation", "lifecycle_event", "governed_knowledge_mutation",
                    "adjudicat", "authority",
                ):
                    self.assertNotIn(forbidden, blob)

    def test_no_tool_definition_declares_a_knowledge_lifecycle_write(self):
        seed_starter_toolboxes()
        knowledge_tools = ToolDefinition.objects.filter(name__icontains="knowledge")
        self.assertGreater(knowledge_tools.count(), 0)
        for tool in knowledge_tools:
            with self.subTest(tool=tool.name):
                self.assertEqual(tool.operation_mode, ToolDefinition.OperationMode.READ)

    def test_no_agent_facing_module_imports_the_mutation_foundation(self):
        import pathlib

        base = pathlib.Path(__file__).resolve().parent
        for module in (
            "services/knowledge_retrieval.py",
            "services/game_action_dispatcher.py",
            "services/agent_runtime.py",
            "services/tools_runtime.py",
            "services/tool_resolution.py",
            "services/knowledge_ingestion.py",
        ):
            path = base / module
            if not path.exists():
                continue
            with self.subTest(module=module):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("knowledge_mutation", source)
                self.assertNotIn("KnowledgeLifecycleEvent", source)
                self.assertNotIn("governed_knowledge_mutation", source)

    def test_lifecycle_event_is_not_registered_in_admin(self):
        """An Admin surface would be an ungoverned authority-write path."""
        from django.contrib import admin

        self.assertNotIn(KnowledgeLifecycleEvent, admin.site._registry)

    def test_the_generic_mutation_primitive_is_not_public(self):
        """"Mutate anything in this document transaction" is infrastructure,
        not a supported Core API. Public write verbs must be operation-specific
        and arrive with 3d/3e."""
        from ai_hub.services import knowledge_mutation

        self.assertFalse(hasattr(knowledge_mutation, "governed_knowledge_mutation"))
        self.assertFalse(hasattr(knowledge_mutation, "GovernedMutation"))
        self.assertTrue(hasattr(knowledge_mutation, "_governed_knowledge_mutation"))
        self.assertTrue(hasattr(knowledge_mutation, "_GovernedMutation"))

    def test_the_public_surface_is_read_only_concepts_only(self):
        """Everything public is a value object, an exception, or a pure read.

        No public callable mutates anything. Only names DEFINED in the module
        are considered; imported models and constants are not its API.
        """
        import inspect

        from ai_hub.services import knowledge_mutation

        public = {
            name
            for name, obj in vars(knowledge_mutation).items()
            if not name.startswith("_")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
            and getattr(obj, "__module__", "") == knowledge_mutation.__name__
        }
        self.assertEqual(
            public,
            {
                "KnowledgeMutationPrincipal", "KnowledgeMutationSnapshot",
                "ExpectedKnowledgeState", "KnowledgeMutationConflict",
                "KnowledgeMutationPrincipalError", "KnowledgeMutationOperationError",
                "UnauditableMutationError",
                "build_snapshot", "verify_expected_state",
            },
        )

    def test_no_public_callable_writes_to_the_database(self):
        """`build_snapshot` and `verify_expected_state` are the only public
        callables, and neither may change anything."""
        collection = KnowledgeCollection.objects.create(name="Public Surface")
        document = make_document(collection, "Read Only")
        before = _snapshot_database()

        snapshot = build_snapshot(document)
        verify_expected_state(snapshot, ExpectedKnowledgeState.from_snapshot(snapshot))

        self.assertEqual(before, _snapshot_database())
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_no_production_module_imports_the_internal_mutator(self):
        """Nothing calls it yet; the first consumer arrives in an authorized
        slice."""
        import pathlib

        base = pathlib.Path(__file__).resolve().parent
        offenders = []
        for path in sorted(base.rglob("*.py")):
            if path.name.startswith("test_") or path.name == "tests.py":
                continue
            if path.name == "knowledge_mutation.py":
                continue
            source = path.read_text(encoding="utf-8")
            if "_governed_knowledge_mutation" in source or "_GovernedMutation" in source:
                offenders.append(str(path.relative_to(base)))
        self.assertEqual(offenders, [])

    def test_foundation_exposes_no_domain_operation(self):
        """3d/3e must not be implemented under 3c's name."""
        from ai_hub.services import knowledge_mutation

        for forbidden in (
            "adjudicate", "repair", "regenerate", "backfill", "set_authority",
            "promote", "reclassify", "mark_derived", "mark_explicit",
            "adopt_unknown", "adjudicate_unknown",
        ):
            with self.subTest(name=forbidden):
                self.assertFalse(
                    any(name.startswith(forbidden) for name in dir(knowledge_mutation)),
                    f"{forbidden!r} would implement a downstream slice early",
                )

    def test_foundation_has_no_public_event_writer(self):
        from ai_hub.services import knowledge_mutation

        public_writers = [
            name for name in dir(knowledge_mutation)
            if not name.startswith("_")
            and any(verb in name for verb in ("record_", "create_", "update_", "delete_"))
        ]
        self.assertEqual(public_writers, [])


# ---------------------------------------------------------------------------
# S — existing writers are untouched and still produce UNKNOWN
# ---------------------------------------------------------------------------

class ExistingWriterInvarianceTests(TestCase):
    """The first consumer of the boundary arrives in a later authorized slice.

    Half-migrating writers now would create documents claiming authority nobody
    can verify.
    """

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Writers")

    def test_ingestion_fallback_still_produces_unknown_and_no_event(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Ingested", curated_text="body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        ensure_initial_knowledge_chunk(document)
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(document.generation_input_fingerprint, "")
        self.assertEqual(document.generator_identity, "")
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_raw_orm_document_creation_still_produces_unknown_and_no_event(self):
        document = make_document(self.collection, "Raw")
        self.assertEqual(document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_ungoverned_chunk_edit_creates_no_event_but_stays_detectable(self):
        """Raw ORM is an intentional escape hatch; evidence, not exclusion."""
        document = make_document(self.collection, "Escape Hatch")
        before = build_snapshot(document).observed_chunk_set_fingerprint

        chunk = document.chunks.get(chunk_index=1)
        chunk.content = "edited with no governance at all"
        chunk.save(update_fields=["content"])

        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)
        self.assertNotEqual(before, build_snapshot(document).observed_chunk_set_fingerprint)

    def test_no_events_were_fabricated_for_historical_documents(self):
        make_document(self.collection, "Legacy A")
        make_document(self.collection, "Legacy B")
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)


# ---------------------------------------------------------------------------
# T / U — preflight and retrieval remain unchanged
# ---------------------------------------------------------------------------

class PreflightAndRetrievalRegressionTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Regression")
        self.document = make_document(self.collection, "Reg Doc", curated_text="source")

    def test_preflight_is_still_read_only_and_creates_no_events(self):
        before = _snapshot_database()
        first = run_knowledge_preflight()
        second = run_knowledge_preflight()
        self.assertEqual(first, second)
        self.assertEqual(before, _snapshot_database())
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_preflight_contract_and_verdicts_are_unchanged_by_this_slice(self):
        report = run_knowledge_preflight()
        row = next(r for r in report["documents"] if r["document_id"] == self.document.pk)
        self.assertEqual(report["contract_version"], 2)
        self.assertEqual(row["structural_state"], "READY_CANONICAL")
        self.assertEqual(row["authority_mode"], MODES.UNKNOWN)
        self.assertEqual(row["lifecycle_state"], "UNKNOWN_AUTHORITY")
        self.assertTrue(row["canonically_retrievable"])

    def test_preflight_does_not_report_lifecycle_events(self):
        """Events are a separate history, not part of the corpus-health report."""
        with _governed_knowledge_mutation(
            self.document.pk, expected=expected_for(self.document),
            operation=TEST_OPERATION, principal=a_principal(),
        ) as mutation:
            mutation.document.notes = "x"
            mutation.document.save(update_fields=["notes"])

        report = run_knowledge_preflight()
        self.assertNotIn("events", report)
        self.assertNotIn("lifecycle_events", report)
        self.assertNotIn("lifecycle_events", report["lifecycle"])

    def test_a_governed_mutation_that_changes_no_lifecycle_fact_is_still_recorded(self):
        """Event existence means "a governed mutation committed", not "authority
        changed"."""
        with _governed_knowledge_mutation(
            self.document.pk, expected=expected_for(self.document),
            operation=TEST_OPERATION, principal=a_principal(),
        ) as mutation:
            mutation.document.notes = "no lifecycle fact touched"
            mutation.document.save(update_fields=["notes"])

        event = KnowledgeLifecycleEvent.objects.get()
        self.assertEqual(event.previous_authority_mode, event.new_authority_mode)


# ---------------------------------------------------------------------------
# P — locking and serialization
# ---------------------------------------------------------------------------

class GovernedMutationLockingTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(name="Locking")
        self.document = make_document(self.collection, "Locked")

    def test_document_and_chunks_are_selected_for_update(self):
        """Chunks are locked too: the chunk-set fingerprint is measured from
        them, so locking only the document would leave the evidence movable."""
        if not connection.features.has_select_for_update:
            self.skipTest(
                "SQLite cannot emit or validate select_for_update; run this test "
                "on PostgreSQL CI."
            )
        from django.test.utils import CaptureQueriesContext

        with CaptureQueriesContext(connection) as captured:
            with _governed_knowledge_mutation(
                self.document.pk, expected=expected_for(self.document),
                operation=TEST_OPERATION, principal=a_principal(),
            ) as mutation:
                mutation.document.notes = "x"
                mutation.document.save(update_fields=["notes"])

        locking = [q["sql"] for q in captured.captured_queries if "FOR UPDATE" in q["sql"].upper()]
        self.assertGreaterEqual(len(locking), 2)


class GovernedMutationConcurrencyTests(TransactionTestCase):
    """Real row-lock semantics only exist on PostgreSQL; SQLite skips.

    Follows the repository's established concurrency-test convention rather
    than inventing a new one.
    """

    reset_sequences = True

    def test_racing_governed_mutations_produce_one_commit_and_one_conflict(self):
        if not connection.features.has_select_for_update:
            self.skipTest(
                "SQLite cannot validate select_for_update locking semantics; run "
                "this test on PostgreSQL CI."
            )

        collection = KnowledgeCollection.objects.create(name="Race")
        document = make_document(collection, "Contended")
        expected = expected_for(document)

        def attempt(index):
            close_old_connections()
            try:
                with _governed_knowledge_mutation(
                    document.pk, expected=expected,
                    operation=TEST_OPERATION,
                    principal=KnowledgeMutationPrincipal.system(f"racer-{index}"),
                ) as mutation:
                    KnowledgeDocumentChunk.objects.create(
                        document=mutation.document, chunk_index=100 + index,
                        section_title="Race", content=f"written by racer {index}",
                    )
                return "committed"
            except KnowledgeMutationConflict:
                return "conflict"
            finally:
                close_old_connections()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(attempt, range(2)))

        self.assertEqual(results.count("committed"), 1)
        self.assertEqual(results.count("conflict"), 1)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 1)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _snapshot_database():
    """Full comparable snapshot of every Knowledge row, for read-only proofs."""
    return {
        "documents": list(
            KnowledgeDocument.objects.order_by("pk").values(
                "pk", "collection_id", "title", "curated_text", "status", "notes",
                "chunk_authority_mode", "generation_input_fingerprint",
                "generation_chunk_set_fingerprint", "generator_identity",
                "generator_version", "updated_at",
            )
        ),
        "chunks": list(
            KnowledgeDocumentChunk.objects.order_by("pk").values(
                "pk", "document_id", "chunk_index", "section_title", "content",
                "token_estimate", "metadata", "updated_at",
            )
        ),
        "events": list(
            KnowledgeLifecycleEvent.objects.order_by("pk").values(
                "pk", "document_id", "document_id_snapshot",
                "collection_id_snapshot", "operation", "principal_kind",
                "principal_identifier", "previous_authority_mode",
                "new_authority_mode", "previous_status", "new_status",
            )
        ),
    }
