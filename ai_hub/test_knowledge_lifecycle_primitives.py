"""Tests for the Knowledge lifecycle persistence primitives (Slice 7, corrected).

Covers the two deterministic fingerprint contracts (`i1` generation input and
`c1` chunk set), strict chunk-index handling, the generator identity, the new
document-level lifecycle fields, the migration's backward-compatible semantics,
and proof that none of it changes retrieval behavior.

This slice adds FACTS, not authority inference. Nothing here regenerates,
repairs, adjudicates or mutates Knowledge.
"""
import importlib
import unicodedata

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TestCase, TransactionTestCase

from ai_hub.models import (
    AgentProfile,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    KnowledgeLifecycleEvent,
    ModelConfig,
    ProviderConfig,
)


def current_migration_targets():
    """The CURRENT leaves of the migration graph, for every app.

    Read from the graph rather than hard-coded. A migration regression test
    necessarily names a HISTORICAL migration as its subject, but the database it
    leaves behind is shared with every test that runs afterwards, so the
    restoration target must be "wherever the project is now" - not the migration
    under test, which stops being the leaf the moment the next one ships.

    All apps, not just `ai_hub`: migrating `ai_hub` backwards also unapplies any
    migration in another app that depends on it, and those must come back too.
    """
    executor = MigrationExecutor(connection)
    executor.loader.build_graph()
    return executor.loader.graph.leaf_nodes()


def restore_migration_state():
    """Return the test database to the current migration graph leaves."""
    executor = MigrationExecutor(connection)
    executor.loader.build_graph()
    targets = executor.loader.graph.leaf_nodes()
    executor.migrate(targets)
    executor.loader.build_graph()
    return targets


def applied_ai_hub_migrations():
    executor = MigrationExecutor(connection)
    executor.loader.build_graph()
    return {node for node in executor.loader.applied_migrations if node[0] == "ai_hub"}
from ai_hub.services.knowledge_lifecycle import (
    CHUNK_SET_FINGERPRINT_CONTRACT,
    GENERATION_INPUT_FINGERPRINT_CONTRACT,
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK,
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION,
    SUPPORTED_GENERATORS,
    MissingChunkIndexError,
    UnsupportedGeneratorError,
    chunk_set_fingerprint,
    curated_text_single_chunk_input_fingerprint,
    current_generator_version,
    document_chunk_set_fingerprint,
    document_generation_input_fingerprint,
    fingerprint_contract,
    generation_input_fingerprint,
    is_supported_generator,
    normalize_chunk_text,
    normalize_curated_text,
    normalize_title_text,
)
from ai_hub.services.knowledge_retrieval import search_knowledge
from ai_hub.test_application_scope_helpers import test_scope


def chunk(index, section_title, content):
    """A chunk-shaped mapping; the fingerprint accepts these or model rows."""
    return {"chunk_index": index, "section_title": section_title, "content": content}


def input_fp(title, curated_text):
    return curated_text_single_chunk_input_fingerprint(
        title=title, curated_text=curated_text
    )


class GenerationInputFingerprintContractTests(TestCase):
    """`i1` — the COMPLETE mutable input set for curated_text_single_chunk.

    The generator reads both `title` (-> section_title) and `curated_text`
    (-> content), so both must be covered. A curated_text-only fingerprint
    could not detect a title change, and regeneration would silently produce a
    different section_title while the fingerprint still claimed a match.
    """

    def test_value_is_versioned_with_the_i1_prefix_not_s1(self):
        value = input_fp("Title", "body")
        self.assertTrue(value.startswith("i1:"))
        self.assertFalse(value.startswith("s1:"))
        self.assertEqual(
            fingerprint_contract(value), GENERATION_INPUT_FINGERPRINT_CONTRACT
        )
        digest = value.split(":", 1)[1]
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in digest))

    def test_same_title_and_same_normalized_curated_text_match(self):
        self.assertEqual(input_fp("Title", "body"), input_fp("Title", "body"))

    # -- title is a generation input -------------------------------------

    def test_title_change_DIFFERS(self):
        """The correction this contract exists for."""
        self.assertNotEqual(input_fp("Title A", "body"), input_fp("Title B", "body"))

    def test_title_outer_whitespace_change_DIFFERS(self):
        """The generator uses `document.title` exactly as persisted."""
        self.assertNotEqual(input_fp("Title", "body"), input_fp(" Title ", "body"))
        self.assertNotEqual(input_fp("Title", "body"), input_fp("Title\n", "body"))

    def test_title_unicode_canonical_equivalents_match(self):
        self.assertEqual(
            input_fp(unicodedata.normalize("NFC", "Café"), "body"),
            input_fp(unicodedata.normalize("NFD", "Café"), "body"),
        )

    def test_title_crlf_matches_lf(self):
        self.assertEqual(input_fp("A\r\nB", "body"), input_fp("A\nB", "body"))

    def test_title_case_change_DIFFERS(self):
        self.assertNotEqual(input_fp("Title", "body"), input_fp("title", "body"))

    # -- curated_text is a generation input ------------------------------

    def test_curated_text_outer_whitespace_only_matches(self):
        """The generator uses `curated_text.strip()`."""
        self.assertEqual(input_fp("Title", "body"), input_fp("Title", "  body  "))
        self.assertEqual(input_fp("Title", "body"), input_fp("Title", "\n\tbody\n"))

    def test_curated_text_internal_whitespace_change_DIFFERS(self):
        self.assertNotEqual(input_fp("Title", "a b"), input_fp("Title", "a  b"))
        self.assertNotEqual(input_fp("Title", "a\nb"), input_fp("Title", "a\n\nb"))

    def test_curated_text_case_change_DIFFERS(self):
        self.assertNotEqual(input_fp("Title", "Body"), input_fp("Title", "body"))

    def test_curated_text_content_change_DIFFERS(self):
        self.assertNotEqual(input_fp("Title", "alpha"), input_fp("Title", "beta"))

    def test_curated_text_crlf_matches_lf(self):
        self.assertEqual(input_fp("Title", "a\r\nb"), input_fp("Title", "a\nb"))

    def test_curated_text_unicode_canonical_equivalents_match(self):
        self.assertEqual(
            input_fp("Title", unicodedata.normalize("NFC", "café")),
            input_fp("Title", unicodedata.normalize("NFD", "café")),
        )

    # -- the two inputs are not interchangeable ---------------------------

    def test_swapping_title_and_curated_text_DIFFERS(self):
        """Field-keyed JSON, so the two inputs cannot be confused."""
        self.assertNotEqual(input_fp("alpha", "beta"), input_fp("beta", "alpha"))

    def test_empty_and_none_are_stable(self):
        self.assertEqual(input_fp("", ""), input_fp(None, None))

    # -- normalization helpers -------------------------------------------

    def test_title_normalization_does_not_strip(self):
        self.assertEqual(normalize_title_text("  A\r\nB  "), "  A\nB  ")

    def test_curated_text_normalization_strips_outer_only(self):
        self.assertEqual(normalize_curated_text("  A\r\nB  "), "A\nB")

    # -- generator binding ------------------------------------------------

    def test_the_contract_is_reached_through_a_generator_identity(self):
        self.assertEqual(
            generation_input_fingerprint(
                GENERATOR_CURATED_TEXT_SINGLE_CHUNK, title="T", curated_text="b"
            ),
            input_fp("T", "b"),
        )

    def test_an_unknown_generator_identity_raises_rather_than_guessing(self):
        """A fabricated match or mismatch is worse than an honest failure."""
        with self.assertRaises(UnsupportedGeneratorError):
            generation_input_fingerprint(
                "some_future_parser", title="T", curated_text="b"
            )

    def test_document_helper_uses_title_and_curated_text(self):
        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Input FP")
        document = KnowledgeDocument.objects.create(
            collection=collection, title="Doc Title", curated_text="  the body  "
        )
        self.assertEqual(
            document_generation_input_fingerprint(
                document, GENERATOR_CURATED_TEXT_SINGLE_CHUNK
            ),
            input_fp("Doc Title", "the body"),
        )

    def test_document_helper_detects_a_title_only_change(self):
        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Title Change")
        document = KnowledgeDocument.objects.create(
            collection=collection, title="Original", curated_text="unchanged body"
        )
        before = document_generation_input_fingerprint(
            document, GENERATOR_CURATED_TEXT_SINGLE_CHUNK
        )
        document.title = "Renamed"
        document.save(update_fields=["title"])
        self.assertNotEqual(
            document_generation_input_fingerprint(
                document, GENERATOR_CURATED_TEXT_SINGLE_CHUNK
            ),
            before,
        )


class ChunkSetFingerprintContractTests(TestCase):
    """`c1` — tamper evidence over the retrieval-relevant chunk set."""

    BASE = (
        chunk(1, "First", "first body"),
        chunk(2, "Second", "second body"),
    )

    def test_fingerprint_is_versioned(self):
        value = chunk_set_fingerprint(self.BASE)
        self.assertTrue(value.startswith(f"{CHUNK_SET_FINGERPRINT_CONTRACT}:"))
        self.assertEqual(fingerprint_contract(value), CHUNK_SET_FINGERPRINT_CONTRACT)
        self.assertEqual(len(value.split(":", 1)[1]), 64)

    def test_the_two_contracts_are_independent(self):
        self.assertNotEqual(
            GENERATION_INPUT_FINGERPRINT_CONTRACT, CHUNK_SET_FINGERPRINT_CONTRACT
        )

    def test_same_chunk_set_gives_the_same_fingerprint(self):
        self.assertEqual(chunk_set_fingerprint(self.BASE), chunk_set_fingerprint(self.BASE))

    def test_python_ordering_does_not_matter(self):
        self.assertEqual(
            chunk_set_fingerprint(self.BASE),
            chunk_set_fingerprint(list(reversed(self.BASE))),
        )

    def test_content_change_differs(self):
        changed = (self.BASE[0], chunk(2, "Second", "second body edited"))
        self.assertNotEqual(chunk_set_fingerprint(self.BASE), chunk_set_fingerprint(changed))

    def test_section_title_change_differs(self):
        changed = (self.BASE[0], chunk(2, "Renamed", "second body"))
        self.assertNotEqual(chunk_set_fingerprint(self.BASE), chunk_set_fingerprint(changed))

    def test_chunk_index_change_differs(self):
        changed = (self.BASE[0], chunk(3, "Second", "second body"))
        self.assertNotEqual(chunk_set_fingerprint(self.BASE), chunk_set_fingerprint(changed))

    def test_reindexing_the_whole_set_differs(self):
        reindexed = (chunk(5, "First", "first body"), chunk(6, "Second", "second body"))
        self.assertNotEqual(chunk_set_fingerprint(self.BASE), chunk_set_fingerprint(reindexed))

    def test_swapping_content_between_indexes_differs(self):
        swapped = (chunk(1, "First", "second body"), chunk(2, "Second", "first body"))
        self.assertNotEqual(chunk_set_fingerprint(self.BASE), chunk_set_fingerprint(swapped))

    def test_chunk_addition_differs(self):
        added = self.BASE + (chunk(3, "Third", "third body"),)
        self.assertNotEqual(chunk_set_fingerprint(self.BASE), chunk_set_fingerprint(added))

    def test_chunk_removal_differs(self):
        self.assertNotEqual(
            chunk_set_fingerprint(self.BASE), chunk_set_fingerprint(self.BASE[:1])
        )

    def test_empty_set_is_stable(self):
        self.assertEqual(chunk_set_fingerprint([]), chunk_set_fingerprint([]))
        self.assertNotEqual(chunk_set_fingerprint([]), chunk_set_fingerprint(self.BASE))

    def test_crlf_matches_lf(self):
        self.assertEqual(
            chunk_set_fingerprint([chunk(1, "S", "a\r\nb")]),
            chunk_set_fingerprint([chunk(1, "S", "a\nb")]),
        )

    def test_unicode_canonical_equivalents_match(self):
        self.assertEqual(
            chunk_set_fingerprint([chunk(1, "S", unicodedata.normalize("NFC", "café"))]),
            chunk_set_fingerprint([chunk(1, "S", unicodedata.normalize("NFD", "café"))]),
        )

    def test_chunk_content_whitespace_change_DIFFERS(self):
        """Deliberate asymmetry with the curated_text input contract.

        An ungoverned whitespace edit to persisted chunk content is still an
        edit, and this fingerprint exists to detect edits.
        """
        self.assertNotEqual(
            chunk_set_fingerprint([chunk(1, "S", "body")]),
            chunk_set_fingerprint([chunk(1, "S", " body ")]),
        )
        self.assertNotEqual(
            chunk_set_fingerprint([chunk(1, "S", "body")]),
            chunk_set_fingerprint([chunk(1, "S", "body\n")]),
        )

    def test_normalization_helper_does_not_strip(self):
        self.assertEqual(normalize_chunk_text("  a\r\nb  "), "  a\nb  ")

    def test_serialization_is_unambiguous_against_delimiter_forgery(self):
        self.assertNotEqual(
            chunk_set_fingerprint([chunk(1, "A", 'x"},{"chunk_index":2,"content":"y')]),
            chunk_set_fingerprint([chunk(1, "A", "x"), chunk(2, "", "y")]),
        )


class StrictChunkIndexTests(TestCase):
    """`chunk_index` is required. None and 0 are different inputs."""

    def test_none_index_is_rejected(self):
        with self.assertRaises(MissingChunkIndexError):
            chunk_set_fingerprint([chunk(None, "S", "body")])

    def test_missing_mapping_key_is_rejected(self):
        with self.assertRaises(MissingChunkIndexError):
            chunk_set_fingerprint([{"section_title": "S", "content": "body"}])

    def test_missing_attribute_on_an_object_is_rejected(self):
        class Partial:
            section_title = "S"
            content = "body"

        with self.assertRaises(MissingChunkIndexError):
            chunk_set_fingerprint([Partial()])

    def test_index_zero_succeeds_and_is_distinct_from_index_one(self):
        """Fingerprinting represents what it is given.

        Whether index 0 is a valid lifecycle state is the preflight's question,
        not this contract's — it is reported as a KP006 anomaly there.
        """
        zero = chunk_set_fingerprint([chunk(0, "S", "body")])
        one = chunk_set_fingerprint([chunk(1, "S", "body")])
        self.assertTrue(zero.startswith("c1:"))
        self.assertNotEqual(zero, one)

    def test_index_zero_is_not_treated_as_missing(self):
        self.assertEqual(
            chunk_set_fingerprint([chunk(0, "S", "body")]),
            chunk_set_fingerprint([chunk(0, "S", "body")]),
        )


class ChunkSetFingerprintIgnoresNonEvidenceTests(TestCase):
    """Fields that do not define retrieval evidence must not raise false alarms."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Ignore Fields")

    def _document(self, title):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title=title, curated_text="source"
        )
        KnowledgeDocumentChunk.objects.create(
            document=document,
            chunk_index=1,
            section_title="Section",
            content="body",
            token_estimate=2,
            metadata={"ingestion": "initial_curated_text"},
        )
        return document

    def test_different_primary_keys_same_representation_match(self):
        first = self._document("First Doc")
        second = self._document("Second Doc")
        self.assertNotEqual(first.pk, second.pk)
        self.assertNotEqual(first.chunks.get().pk, second.chunks.get().pk)
        self.assertEqual(
            document_chunk_set_fingerprint(first),
            document_chunk_set_fingerprint(second),
        )

    def test_metadata_only_change_matches(self):
        document = self._document("Metadata Doc")
        before = document_chunk_set_fingerprint(document)
        chunk_row = document.chunks.get()
        chunk_row.metadata = {"ingestion": "something_else", "extra": [1, 2, 3]}
        chunk_row.save(update_fields=["metadata"])
        self.assertEqual(document_chunk_set_fingerprint(document), before)

    def test_token_estimate_only_change_matches(self):
        document = self._document("Token Doc")
        before = document_chunk_set_fingerprint(document)
        chunk_row = document.chunks.get()
        chunk_row.token_estimate = 999
        chunk_row.save(update_fields=["token_estimate"])
        self.assertEqual(document_chunk_set_fingerprint(document), before)

    def test_timestamp_change_matches(self):
        document = self._document("Timestamp Doc")
        before_fingerprint = document_chunk_set_fingerprint(document)
        chunk_row = document.chunks.get()
        before_timestamp = chunk_row.updated_at
        chunk_row.save()
        chunk_row.refresh_from_db()
        self.assertGreater(chunk_row.updated_at, before_timestamp)
        self.assertEqual(document_chunk_set_fingerprint(document), before_fingerprint)

    def test_a_real_content_edit_is_detected(self):
        document = self._document("Edited Doc")
        before = document_chunk_set_fingerprint(document)
        chunk_row = document.chunks.get()
        chunk_row.content = "body edited by hand"
        chunk_row.save(update_fields=["content"])
        self.assertNotEqual(document_chunk_set_fingerprint(document), before)


class GeneratorIdentityTests(TestCase):
    def test_the_only_supported_generator_is_curated_text_single_chunk(self):
        self.assertEqual(
            set(SUPPORTED_GENERATORS), {GENERATOR_CURATED_TEXT_SINGLE_CHUNK}
        )
        self.assertEqual(
            current_generator_version(GENERATOR_CURATED_TEXT_SINGLE_CHUNK),
            GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION,
        )
        self.assertTrue(is_supported_generator(GENERATOR_CURATED_TEXT_SINGLE_CHUNK))

    def test_source_file_is_not_a_supported_generation_strategy(self):
        for identity in SUPPORTED_GENERATORS:
            self.assertNotIn("file", identity)
        self.assertIsNone(current_generator_version("source_file_parsed"))
        self.assertFalse(is_supported_generator("source_file_parsed"))

    def test_initial_version_is_one(self):
        self.assertEqual(GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION, 1)


class LifecycleFieldSemanticsTests(TestCase):
    """The new document-level facts, their defaults and their meaning."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Lifecycle Fields")

    def test_a_new_document_defaults_to_unknown_with_blank_provenance(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Fresh", curated_text="body"
        )
        self.assertEqual(
            document.chunk_authority_mode, KnowledgeDocument.ChunkAuthorityMode.UNKNOWN
        )
        self.assertEqual(document.generation_input_fingerprint, "")
        self.assertEqual(document.generation_chunk_set_fingerprint, "")
        self.assertEqual(document.generator_identity, "")
        self.assertIsNone(document.generator_version)

    def test_the_three_authority_modes_exist_with_stable_values(self):
        self.assertEqual(
            [value for value, _label in KnowledgeDocument.ChunkAuthorityMode.choices],
            ["unknown", "derived", "explicit"],
        )

    def test_every_mode_validates(self):
        for mode in KnowledgeDocument.ChunkAuthorityMode.values:
            with self.subTest(mode=mode):
                document = KnowledgeDocument(
                    collection=self.collection, title=f"Doc {mode}", chunk_authority_mode=mode
                )
                document.full_clean()

    def test_generator_version_is_null_not_zero_when_absent(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Null Version"
        )
        document.refresh_from_db()
        self.assertIsNone(document.generator_version)
        self.assertNotEqual(document.generator_version, 0)

    def test_fields_accept_versioned_fingerprints_at_full_length(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection,
            title="Full Length",
            chunk_authority_mode=KnowledgeDocument.ChunkAuthorityMode.DERIVED,
            generation_input_fingerprint=input_fp("Full Length", "body"),
            generation_chunk_set_fingerprint=chunk_set_fingerprint([chunk(1, "S", "body")]),
            generator_identity=GENERATOR_CURATED_TEXT_SINGLE_CHUNK,
            generator_version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION,
        )
        document.full_clean()
        document.refresh_from_db()
        self.assertEqual(fingerprint_contract(document.generation_input_fingerprint), "i1")
        self.assertEqual(
            fingerprint_contract(document.generation_chunk_set_fingerprint), "c1"
        )

    def test_inconsistent_combinations_are_still_storable(self):
        """Deliberate: the governed mutation boundary does not exist yet.

        Raw ORM remains an intentional escape hatch, and Preflight V2 must be
        able to REPORT an inconsistent state rather than be prevented from
        observing it. No DB CheckConstraint encodes the lifecycle.
        """
        document = KnowledgeDocument(
            collection=self.collection,
            title="Inconsistent",
            chunk_authority_mode=KnowledgeDocument.ChunkAuthorityMode.EXPLICIT,
            generation_input_fingerprint=input_fp("Inconsistent", "body"),
            generator_version=7,
        )
        document.full_clean()
        document.save()
        self.assertEqual(KnowledgeDocument.objects.filter(pk=document.pk).count(), 1)

    def test_no_lifecycle_check_constraint_was_added(self):
        names = {constraint.name for constraint in KnowledgeDocument._meta.constraints}
        self.assertEqual(names, set())


class LifecycleFieldsDoNotAffectRetrievalTests(TestCase):
    """The new facts must change ZERO retrieval behavior."""

    @classmethod
    def setUpTestData(cls):
        provider = ProviderConfig.objects.create(
            name="lifecycle-provider", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        cls.agent = AgentProfile.objects.create(
            application_scope=test_scope(),
            name="lifecycle-agent", role="Lifecycle", model_config=model,
        )
        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Lifecycle Retrieval")
        cls.agent.knowledge_collections.add(collection)
        cls.document = KnowledgeDocument.objects.create(
            collection=collection,
            title="Retrievable",
            curated_text="source body",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        KnowledgeDocumentChunk.objects.create(
            document=cls.document,
            chunk_index=1,
            section_title="Section",
            content="lifecycle-retrieval-marker body",
        )

    def test_search_results_are_identical_across_every_authority_mode(self):
        baseline = search_knowledge(
            self.agent, query="lifecycle-retrieval-marker", limit=5
        )
        for mode in KnowledgeDocument.ChunkAuthorityMode.values:
            with self.subTest(mode=mode):
                self.document.chunk_authority_mode = mode
                self.document.generation_input_fingerprint = input_fp("x", "y")
                self.document.generator_identity = GENERATOR_CURATED_TEXT_SINGLE_CHUNK
                self.document.generator_version = 1
                self.document.save()
                self.assertEqual(
                    search_knowledge(self.agent, query="lifecycle-retrieval-marker", limit=5),
                    baseline,
                )

    def test_no_retrieval_module_reads_the_new_fields(self):
        """Retrieval never consults lifecycle facts. Preflight V2 may.

        `knowledge_preflight` is deliberately excluded from this list: Slice 8
        makes it lifecycle-aware, and it is read-only.
        """
        for module_path in (
            "ai_hub.services.knowledge_retrieval",
            "ai_hub.services.knowledge_ingestion",
            "ai_hub.services.game_action_dispatcher",
            "ai_hub.services.agent_runtime",
        ):
            module = importlib.import_module(module_path)
            source = open(module.__file__, encoding="utf-8").read()
            for field in (
                "chunk_authority_mode",
                "generation_input_fingerprint",
                "generation_chunk_set_fingerprint",
                "generator_identity",
                "generator_version",
            ):
                with self.subTest(module=module_path, field=field):
                    self.assertNotIn(field, source)


class MigrationStateIsolationTests(TransactionTestCase):
    """A migration test must not corrupt the schema the next test expects.

    Order-independent by construction: each test performs the backwards move
    itself and then invokes the same restoration used in teardown, so the proof
    does not depend on running after `LifecycleMigrationTests`. The final test
    additionally asserts the state a following test would actually observe.
    """

    def tearDown(self):
        restore_migration_state()

    def test_restore_returns_the_database_to_the_graph_leaves(self):
        MigrationExecutor(connection).migrate([("ai_hub", "0020_approval_execution_intent")])
        self.assertNotIn(
            ("ai_hub", "0022_knowledge_lifecycle_event"), applied_ai_hub_migrations()
        )

        targets = restore_migration_state()

        self.assertEqual(set(targets), set(current_migration_targets()))
        applied = applied_ai_hub_migrations()
        for node in current_migration_targets():
            if node[0] == "ai_hub":
                with self.subTest(node=node):
                    self.assertIn(node, applied)

    def test_every_current_model_has_its_table_after_restoration(self):
        """Graph-based, so a future migration is covered without editing this."""
        MigrationExecutor(connection).migrate([("ai_hub", "0020_approval_execution_intent")])
        restore_migration_state()

        tables = set(connection.introspection.table_names())
        from django.apps import apps

        missing = [
            model._meta.db_table
            for model in apps.get_app_config("ai_hub").get_models()
            if model._meta.managed and model._meta.db_table not in tables
        ]
        self.assertEqual(missing, [])

    def test_the_lifecycle_event_table_survives_a_migration_test(self):
        """The exact table whose absence broke PostgreSQL CI on 42a071f."""
        MigrationExecutor(connection).migrate([("ai_hub", "0020_approval_execution_intent")])
        self.assertNotIn(
            KnowledgeLifecycleEvent._meta.db_table,
            set(connection.introspection.table_names()),
        )

        restore_migration_state()

        self.assertIn(
            KnowledgeLifecycleEvent._meta.db_table,
            set(connection.introspection.table_names()),
        )

    def test_a_following_test_can_actually_write_a_lifecycle_event(self):
        """Introspection is not enough - prove the ORM can use the restored
        schema, which is what the next test in the suite will do."""
        MigrationExecutor(connection).migrate([("ai_hub", "0020_approval_execution_intent")])
        restore_migration_state()

        collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Post-restore Collection")
        document = KnowledgeDocument.objects.create(
            collection=collection, title="Post-restore Document",
        )
        event = KnowledgeLifecycleEvent.objects.create(
            document=document,
            document_id_snapshot=document.pk,
            collection_id_snapshot=collection.pk,
            operation="isolation_probe",
            principal_kind=KnowledgeLifecycleEvent.PrincipalKind.SYSTEM,
            principal_identifier="isolation-test",
            previous_authority_mode="unknown",
            new_authority_mode="unknown",
            previous_status=document.status,
            new_status=document.status,
            previous_chunk_count=0,
            new_chunk_count=0,
        )
        self.assertEqual(KnowledgeLifecycleEvent.objects.get(pk=event.pk).operation, "isolation_probe")

    def test_the_restore_target_is_not_hard_coded_to_the_migration_under_test(self):
        """The bug in one assertion.

        `LifecycleMigrationTests.MIGRATE_TO` names a historical migration, and
        the graph has moved past it. Restoring to `MIGRATE_TO` would therefore
        leave the database behind - which is precisely what broke CI.
        """
        self.assertNotEqual(
            set(LifecycleMigrationTests.MIGRATE_TO), set(current_migration_targets())
        )
        self.assertNotIn(
            LifecycleMigrationTests.MIGRATE_TO[0], set(current_migration_targets())
        )


class LifecycleMigrationTests(TransactionTestCase):
    """Migration 0021 regression: additive, and every legacy row becomes UNKNOWN.

    RESTORATION CONTRACT
    --------------------
    This test deliberately moves the shared test database backwards to a
    historical migration. Whatever it does in between, it MUST hand the database
    back at the CURRENT migration graph leaves, because every test that runs
    afterwards has model state at HEAD and will find the ORM and the schema
    disagreeing otherwise.

    The restore target is therefore read from the graph, never hard-coded.
    `MIGRATE_TO` names the migration UNDER TEST (`0021`) and is only used to
    drive the assertions; it is deliberately NOT the teardown target. An earlier
    version of this class restored to `MIGRATE_TO`, which was correct exactly
    until `0022` shipped - after that it silently left the database one
    migration behind and dropped `ai_hub_knowledgelifecycleevent` for the rest
    of the suite.
    """

    MIGRATE_FROM = [("ai_hub", "0020_approval_execution_intent")]
    MIGRATE_TO = [("ai_hub", "0021_knowledge_lifecycle_facts")]

    def setUp(self):
        # Runs AFTER tearDown (unittest cleanups precede Django's _post_teardown)
        # and inside THIS test, so a broken restoration fails here rather than
        # silently corrupting whatever runs next. Deliberately not a separate
        # test class: another class's own teardown would heal the leak first and
        # hide it, which is how this survived two of three CI jobs.
        self.addCleanup(self._assert_migration_state_restored)

    def tearDown(self):
        # Graph leaves, NOT self.MIGRATE_TO. See the restoration contract above.
        restore_migration_state()

    def _assert_migration_state_restored(self):
        applied = applied_ai_hub_migrations()
        for node in current_migration_targets():
            if node[0] == "ai_hub":
                self.assertIn(
                    node, applied,
                    f"{type(self).__name__} left the database behind the migration "
                    f"graph at {node}; the next test would see the ORM and the "
                    "schema disagree.",
                )
        self.assertIn(
            KnowledgeLifecycleEvent._meta.db_table,
            set(connection.introspection.table_names()),
        )

    def test_existing_documents_migrate_to_unknown_and_are_otherwise_untouched(self):
        executor = MigrationExecutor(connection)
        executor.migrate(self.MIGRATE_FROM)
        executor.loader.build_graph()
        old_apps = executor.loader.project_state(self.MIGRATE_FROM).apps

        OldCollection = old_apps.get_model("ai_hub", "KnowledgeCollection")
        OldDocument = old_apps.get_model("ai_hub", "KnowledgeDocument")
        OldChunk = old_apps.get_model("ai_hub", "KnowledgeDocumentChunk")

        collection = OldCollection.objects.create(name="Pre-migration Collection")
        document = OldDocument.objects.create(
            collection=collection,
            title="Pre-migration Document",
            curated_text="  pre-migration body  ",
            source_file="ai_hub/knowledge/manual.txt",
            tags=["a", "b"],
            language="en",
            status="active",
            notes="editorial note",
        )
        OldChunk.objects.create(
            document=document,
            chunk_index=1,
            section_title="Pre-migration Section",
            content="pre-migration chunk body",
            token_estimate=4,
            metadata={"ingestion": "initial_curated_text"},
        )
        self.assertFalse(hasattr(document, "chunk_authority_mode"))

        document_count = OldDocument.objects.count()
        chunk_count = OldChunk.objects.count()

        executor = MigrationExecutor(connection)
        executor.migrate(self.MIGRATE_TO)
        executor.loader.build_graph()
        new_apps = executor.loader.project_state(self.MIGRATE_TO).apps

        NewDocument = new_apps.get_model("ai_hub", "KnowledgeDocument")
        NewChunk = new_apps.get_model("ai_hub", "KnowledgeDocumentChunk")
        migrated = NewDocument.objects.get(title="Pre-migration Document")

        self.assertEqual(migrated.chunk_authority_mode, "unknown")
        self.assertEqual(migrated.generation_input_fingerprint, "")
        self.assertEqual(migrated.generation_chunk_set_fingerprint, "")
        self.assertEqual(migrated.generator_identity, "")
        self.assertIsNone(migrated.generator_version)

        self.assertEqual(migrated.curated_text, "  pre-migration body  ")
        self.assertEqual(migrated.source_file, "ai_hub/knowledge/manual.txt")
        self.assertEqual(migrated.tags, ["a", "b"])
        self.assertEqual(migrated.language, "en")
        self.assertEqual(migrated.status, "active")
        self.assertEqual(migrated.notes, "editorial note")

        self.assertEqual(NewDocument.objects.count(), document_count)
        self.assertEqual(NewChunk.objects.count(), chunk_count)
        migrated_chunk = NewChunk.objects.get(document=migrated)
        self.assertEqual(migrated_chunk.chunk_index, 1)
        self.assertEqual(migrated_chunk.section_title, "Pre-migration Section")
        self.assertEqual(migrated_chunk.content, "pre-migration chunk body")
        self.assertEqual(migrated_chunk.token_estimate, 4)
        self.assertEqual(migrated_chunk.metadata, {"ingestion": "initial_curated_text"})

    def test_the_migration_is_additive_only(self):
        migration = importlib.import_module(
            "ai_hub.migrations.0021_knowledge_lifecycle_facts"
        )
        operations = migration.Migration.operations
        self.assertEqual(len(operations), 5)
        added = set()
        for operation in operations:
            self.assertEqual(type(operation).__name__, "AddField")
            self.assertEqual(operation.model_name, "knowledgedocument")
            added.add(operation.name)
        self.assertEqual(
            added,
            {
                "chunk_authority_mode",
                "generation_chunk_set_fingerprint",
                "generation_input_fingerprint",
                "generator_identity",
                "generator_version",
            },
        )

    def test_the_migration_contains_no_data_function(self):
        migration = importlib.import_module(
            "ai_hub.migrations.0021_knowledge_lifecycle_facts"
        )
        for operation in migration.Migration.operations:
            self.assertNotEqual(type(operation).__name__, "RunPython")
