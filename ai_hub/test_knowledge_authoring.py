"""T2 - `author_explicit_chunk_set`, the operation-specific Knowledge authoring service.

A deliberately ordinary host: it has its own ApplicationScope, collection and
agent, and imports nothing from Academy. Every refusal is asserted to leave zero
changes and zero lifecycle events; every change is asserted against the stored
rows, the audit event, the retrieval surface and the vector store.

Windows/SQLite note: `select_for_update()` is a no-op on SQLite, so this module
proves the application logic (authorization, eligibility, identity, compare-and-
swap, rollback) but NOT concurrent serialisation, database-level length limits or
the ANN shadow backend. Those are PostgreSQL/Linux checks and are pending.
"""
import ast
import pathlib
import unicodedata
from unittest import mock

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction
from django.test import TestCase

from ai_hub.models import (
    AgentProfile,
    EmbeddingModelConfig,
    KnowledgeChunkEmbedding,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    KnowledgeLifecycleEvent,
    ModelConfig,
    ProviderConfig,
    ApplicationScope,
)
from ai_hub.services import knowledge_authoring as authoring
from ai_hub.services.chunk_embedding_identity import chunk_embedding_fingerprint
from ai_hub.services.embedding_contract import resolve_embedding_contract
from ai_hub.services.knowledge_adjudication import (
    adjudicate_unknown_as_explicit,
    adopt_unknown_as_derived,
)
from ai_hub.services.knowledge_authoring import (
    AuthorityRefused,
    AuthoredChunk,
    AuthoringIntegrityError,
    AuthoringStateRefused,
    ChunkMetadataRefused,
    KnowledgeAuthoringInvalidInput,
    KnowledgeAuthoringNotAuthorized,
    OPERATION_AUTHOR_EXPLICIT_CHUNK_SET,
    ProvenanceRefused,
    SourceRepresentationRefused,
    StatusRefused,
    author_explicit_chunk_set,
)
from ai_hub.services.knowledge_ingestion import ensure_initial_knowledge_chunk
from ai_hub.services.knowledge_lifecycle import chunk_set_fingerprint
from ai_hub.services.knowledge_mutation import (
    ExpectedKnowledgeState,
    KnowledgeMutationConflict,
    KnowledgeMutationOperationError,
    KnowledgeMutationPrincipal,
    KnowledgeMutationPrincipalError,
    build_snapshot,
)
from ai_hub.services.knowledge_preflight import run_knowledge_preflight
from ai_hub.services.knowledge_regeneration import (
    KnowledgeRegenerationError,
    regenerate_derived_chunk_set,
)
from ai_hub.services.knowledge_retrieval import (
    cite_knowledge_source,
    read_knowledge_chunk,
    search_knowledge,
)
from ai_hub.services.vector_store import load_current_vectors, store_chunk_vector
from ai_hub.test_application_scope_helpers import test_scope

Status = KnowledgeDocument.Status
Mode = KnowledgeDocument.ChunkAuthorityMode


def C(content, title=""):
    return AuthoredChunk(content=content, section_title=title)


def db_state():
    """Everything an authoring call could touch, comparable for equality."""
    return (
        list(KnowledgeDocument.objects.order_by("pk").values()),
        list(KnowledgeDocumentChunk.objects.order_by("pk").values()),
        list(KnowledgeChunkEmbedding.objects.order_by("pk").values_list("pk", "chunk_id", "k1")),
        list(KnowledgeLifecycleEvent.objects.order_by("pk").values_list("pk", flat=True)),
    )


class AuthoringFixture:
    def setUp(self):
        super().setUp()
        self.scope = test_scope("Authoring Host", slug="authoring-host")
        self.other_scope = test_scope("Authoring Other", slug="authoring-other")
        self.collection = KnowledgeCollection.objects.create(
            application_scope=self.scope, name="Authoring Docs"
        )
        self.principal = KnowledgeMutationPrincipal.system("host-sync")

    def shell(self, title="Doc", **kwargs):
        kwargs.setdefault("status", Status.DRAFT)
        return KnowledgeDocument.objects.create(
            collection=self.collection, title=title, **kwargs
        )

    def expected(self, document):
        document.refresh_from_db()
        return ExpectedKnowledgeState.from_snapshot(build_snapshot(document))

    def author(self, document, chunks, **overrides):
        kwargs = {
            "application_scope": self.scope,
            "expected": self.expected(document),
            "principal": self.principal,
        }
        kwargs.update(overrides)
        return author_explicit_chunk_set(document.pk, chunks, **kwargs)

    def authored(self, texts, title="Doc"):
        document = self.shell(title)
        self.author(document, [C(text) for text in texts])
        document.refresh_from_db()
        return document

    def rows(self, document):
        return [
            (row.pk, row.chunk_index, row.section_title, row.content)
            for row in document.chunks.order_by("chunk_index")
        ]

    def contents(self, document):
        return [row[3] for row in self.rows(document)]

    def ids_by_content(self, document):
        return {row[3]: row[0] for row in self.rows(document)}

    def events(self):
        return KnowledgeLifecycleEvent.objects.count()

    def foreign_document(self):
        other = KnowledgeCollection.objects.create(
            application_scope=self.other_scope, name="Foreign Docs"
        )
        return KnowledgeDocument.objects.create(
            collection=other, title="Foreign", status=Status.DRAFT
        )

    def agent(self):
        provider = ProviderConfig.objects.create(name="authoring-p", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="m")
        agent = AgentProfile.objects.create(
            name="authoring-agent", role="r", model_config=model,
            application_scope=self.scope,
        )
        agent.knowledge_collections.add(self.collection)
        return agent

    def activate(self, document):
        KnowledgeDocument.objects.filter(pk=document.pk).update(status=Status.ACTIVE)
        document.refresh_from_db()

    def assertNothingChanged(self, before):
        self.assertEqual(db_state(), before)


# ---------------------------------------------------------------------------
# Scope authorization
# ---------------------------------------------------------------------------

class ScopeAuthorizationTests(AuthoringFixture, TestCase):
    def test_the_owning_active_scope_can_author_a_first_chunk_set(self):
        document = self.shell()
        result = self.author(document, [C("alpha"), C("bravo")])
        self.assertEqual(result.outcome, authoring.OUTCOME_CHANGED)
        self.assertEqual(self.contents(document), ["alpha", "bravo"])

    def test_a_foreign_scope_is_refused_and_nothing_changes(self):
        document = self.shell()
        before = db_state()
        with self.assertRaises(KnowledgeAuthoringNotAuthorized):
            self.author(document, [C("x")], application_scope=self.other_scope)
        self.assertNothingChanged(before)

    def test_a_document_of_another_scope_is_refused(self):
        foreign = self.foreign_document()
        before = db_state()
        with self.assertRaises(KnowledgeAuthoringNotAuthorized):
            self.author(foreign, [C("x")])
        self.assertNothingChanged(before)

    def test_missing_foreign_and_inactive_cases_are_indistinguishable(self):
        document = self.shell()
        foreign = self.foreign_document()
        expected = self.expected(document)
        messages = set()
        cases = {
            "missing": dict(document_id=999_999, application_scope=self.scope),
            "foreign": dict(document_id=foreign.pk, application_scope=self.scope),
            "wrong scope": dict(document_id=document.pk, application_scope=self.other_scope),
        }
        for name, call in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(KnowledgeAuthoringNotAuthorized) as raised:
                    author_explicit_chunk_set(
                        call["document_id"], [C("x")], application_scope=call["application_scope"],
                        expected=expected, principal=self.principal,
                    )
                messages.add(str(raised.exception))
        ApplicationScope.objects.filter(pk=self.scope.pk).update(is_active=False)
        with self.assertRaises(KnowledgeAuthoringNotAuthorized) as raised:
            author_explicit_chunk_set(
                document.pk, [C("x")], application_scope=self.scope,
                expected=expected, principal=self.principal,
            )
        messages.add(str(raised.exception))
        self.assertEqual(messages, {authoring.NOT_AUTHORIZED_MESSAGE})

    def test_an_inactive_scope_is_refused(self):
        document = self.shell()
        expected = self.expected(document)
        ApplicationScope.objects.filter(pk=self.scope.pk).update(is_active=False)
        before = db_state()
        with self.assertRaises(KnowledgeAuthoringNotAuthorized):
            author_explicit_chunk_set(
                document.pk, [C("x")], application_scope=self.scope,
                expected=expected, principal=self.principal,
            )
        self.assertNothingChanged(before)

    def test_a_scope_deactivated_after_the_object_was_fetched_is_refused(self):
        document = self.shell()
        stale = ApplicationScope.objects.get(pk=self.scope.pk)
        expected = self.expected(document)
        ApplicationScope.objects.filter(pk=self.scope.pk).update(is_active=False)
        self.assertTrue(stale.is_active)  # the in-memory object is stale
        with self.assertRaises(KnowledgeAuthoringNotAuthorized):
            author_explicit_chunk_set(
                document.pk, [C("x")], application_scope=stale,
                expected=expected, principal=self.principal,
            )

    def test_untrusted_document_ids_are_refused_uniformly(self):
        document = self.shell()
        expected = self.expected(document)
        for bad in ("1", str(document.pk), None, True, 1.0, [document.pk]):
            with self.subTest(document_id=bad):
                with self.assertRaises(KnowledgeAuthoringNotAuthorized):
                    author_explicit_chunk_set(
                        bad, [C("x")], application_scope=self.scope,
                        expected=expected, principal=self.principal,
                    )

    def test_the_scope_must_be_an_application_scope_object(self):
        document = self.shell()
        expected = self.expected(document)
        for bad in (self.scope.slug, self.scope.pk, None, object()):
            with self.subTest(scope=bad):
                with self.assertRaises(KnowledgeAuthoringNotAuthorized):
                    author_explicit_chunk_set(
                        document.pk, [C("x")], application_scope=bad,
                        expected=expected, principal=self.principal,
                    )

    def test_authorization_precedes_validation_and_conflict(self):
        foreign = self.foreign_document()
        unrelated = self.expected(self.shell())
        with self.assertRaises(KnowledgeAuthoringNotAuthorized):
            author_explicit_chunk_set(
                foreign.pk, [], application_scope=self.scope,
                expected=unrelated, principal=self.principal,
            )

    def test_authorization_precedes_every_control_argument_error(self):
        """T2 audit finding 3: an inaccessible document never reveals itself through a
        different error, whatever else is malformed."""
        foreign = self.foreign_document()
        good = self.expected(self.shell("Mine"))
        cases = {
            "expected None": dict(expected=None),
            "expected wrong type": dict(expected="state"),
            "principal None": dict(principal=None),
            "principal wrong type": dict(principal="operator"),
            "principal blank identifier": dict(principal=KnowledgeMutationPrincipal("human", " ")),
            "bad reason code": dict(reason_code="Bad Code"),
            "bad chunk payload": dict(chunks=[]),
            "chunk payload of wrong type": dict(chunks="text"),
            "everything malformed": dict(
                expected=None, principal=None, reason_code="Bad Code", chunks=[]
            ),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name):
                kwargs = dict(
                    chunks=[C("x")], application_scope=self.scope,
                    expected=good, principal=self.principal, reason_code="",
                )
                kwargs.update(overrides)
                chunks = kwargs.pop("chunks")
                for target in (foreign.pk, 999_999):  # foreign and missing alike
                    with self.assertRaises(KnowledgeAuthoringNotAuthorized) as raised:
                        author_explicit_chunk_set(target, chunks, **kwargs)
                    self.assertEqual(str(raised.exception), authoring.NOT_AUTHORIZED_MESSAGE)

    def test_the_principal_is_attribution_not_authorization(self):
        """A well-formed principal grants nothing: scope ownership alone decides."""
        foreign = self.foreign_document()
        expected = self.expected(foreign)
        for principal in (
            KnowledgeMutationPrincipal.human("an-admin"),
            KnowledgeMutationPrincipal.system("a-trusted-service"),
        ):
            with self.assertRaises(KnowledgeAuthoringNotAuthorized):
                author_explicit_chunk_set(
                    foreign.pk, [C("x")], application_scope=self.scope,
                    expected=expected, principal=principal,
                )

    def test_ownership_is_rechecked_inside_the_governed_transaction(self):
        """The collection leaves the scope after the pre-check but before the lock."""
        document = self.shell()
        before = db_state()
        real = authoring._governed_knowledge_mutation

        def flip_scope_after_the_precheck(*args, **kwargs):
            # The collection keeps its id (so the compare-and-swap still passes) but
            # changes owner between the first authorization and the lock.
            KnowledgeCollection.objects.filter(pk=self.collection.pk).update(
                application_scope=self.other_scope
            )
            return real(*args, **kwargs)

        with mock.patch.object(authoring, "_governed_knowledge_mutation", flip_scope_after_the_precheck):
            with self.assertRaises(KnowledgeAuthoringNotAuthorized):
                self.author(document, [C("x")])
        document.refresh_from_db()
        self.assertEqual(document.chunks.count(), 0)
        self.assertEqual(document.chunk_authority_mode, Mode.UNKNOWN)
        self.assertEqual(self.events(), 0)
        self.assertEqual(db_state()[3], before[3])


# ---------------------------------------------------------------------------
# Authority and document states
# ---------------------------------------------------------------------------

class AuthorityStateTests(AuthoringFixture, TestCase):
    def test_first_authoring_declares_explicit_and_audits_once(self):
        document = self.shell()
        result = self.author(document, [C("alpha"), C("bravo", "Title")], reason_code="initial_load")
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, Mode.EXPLICIT)
        self.assertEqual(document.status, Status.DRAFT)  # never activated here
        event = KnowledgeLifecycleEvent.objects.get()
        self.assertEqual(event.operation, OPERATION_AUTHOR_EXPLICIT_CHUNK_SET)
        self.assertEqual(event.reason_code, "initial_load")
        self.assertEqual(event.principal_identifier, "host-sync")
        self.assertEqual(event.document_id_snapshot, document.pk)
        self.assertEqual(event.collection_id_snapshot, self.collection.pk)
        self.assertEqual((event.previous_authority_mode, event.new_authority_mode), (Mode.UNKNOWN, Mode.EXPLICIT))
        self.assertEqual((event.previous_chunk_count, event.new_chunk_count), (0, 2))
        self.assertEqual((event.previous_status, event.new_status), (Status.DRAFT, Status.DRAFT))
        self.assertEqual(event.new_observed_chunk_set_fingerprint, result.chunk_set_fingerprint)

    def test_an_explicit_document_can_be_updated_in_any_status(self):
        for status in (Status.DRAFT, Status.ACTIVE, Status.ARCHIVED):
            with self.subTest(status=status):
                document = self.authored(["one"], title=f"Doc {status}")
                KnowledgeDocument.objects.filter(pk=document.pk).update(status=status)
                result = self.author(document, [C("one"), C("two")])
                self.assertEqual(result.outcome, authoring.OUTCOME_CHANGED)
                self.assertEqual(self.contents(document), ["one", "two"])

    def test_a_derived_document_is_refused_and_untouched(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Derivable", curated_text="derived body",
            status=Status.ACTIVE,
        )
        ensure_initial_knowledge_chunk(document)
        adopt_unknown_as_derived(
            document.pk, expected=self.expected(document),
            principal=KnowledgeMutationPrincipal.human("operator-1"),
        )
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, Mode.DERIVED)
        before = db_state()
        with self.assertRaises(AuthorityRefused):
            self.author(document, [C("replacement")])
        self.assertNothingChanged(before)

    def test_a_populated_unknown_document_is_refused_until_adjudicated(self):
        document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Legacy", status=Status.ACTIVE
        )
        KnowledgeDocumentChunk.objects.create(document=document, chunk_index=1, content="legacy body")
        before = db_state()
        with self.assertRaises(AuthorityRefused):
            self.author(document, [C("new")])
        self.assertNothingChanged(before)

        adjudicate_unknown_as_explicit(
            document.pk, expected=self.expected(document),
            principal=KnowledgeMutationPrincipal.human("operator-1"),
        )
        result = self.author(document, [C("legacy body"), C("new")])
        self.assertEqual(result.outcome, authoring.OUTCOME_CHANGED)
        self.assertEqual(self.contents(document), ["legacy body", "new"])

    def test_empty_unknown_documents_must_be_draft(self):
        for status in (Status.ACTIVE, Status.ARCHIVED):
            with self.subTest(status=status):
                document = self.shell(title=f"Empty {status}", status=status)
                before = db_state()
                with self.assertRaises(StatusRefused):
                    self.author(document, [C("x")])
                self.assertNothingChanged(before)

    def test_every_generation_fact_is_refused_independently(self):
        """T2 audit finding 2: `generator_version=0` is a stored fact, not an absence."""
        facts = {
            "generation_input_fingerprint": "i1:abc",
            "generation_chunk_set_fingerprint": "c1:abc",
            "generator_identity": "curated_text_single_chunk",
            "generator_version=0": 0,
            "generator_version=1": 1,
        }
        for mode in (Mode.UNKNOWN, Mode.EXPLICIT):
            for label, value in facts.items():
                with self.subTest(mode=mode, fact=label):
                    field = label.split("=")[0]
                    document = self.shell(title=f"{mode} {label}", chunk_authority_mode=mode)
                    KnowledgeDocument.objects.filter(pk=document.pk).update(**{field: value})
                    before = db_state()
                    with self.assertRaises(ProvenanceRefused):
                        self.author(document, [C("x")])
                    self.assertNothingChanged(before)  # nothing cleared, nothing written
                    document.refresh_from_db()
                    self.assertEqual(getattr(document, field), value)
                    self.assertEqual(document.chunk_authority_mode, mode)

    def test_a_null_generator_version_with_blank_facts_is_accepted(self):
        document = self.shell("No facts")
        self.assertIsNone(document.generator_version)
        self.assertEqual(self.author(document, [C("x")]).outcome, authoring.OUTCOME_CHANGED)

    def test_source_text_or_file_is_refused_for_first_authoring_and_updates(self):
        first = self.shell("With text", curated_text="a separate source representation")
        before = db_state()
        with self.assertRaises(SourceRepresentationRefused):
            self.author(first, [C("x")])
        self.assertNothingChanged(before)

        updated = self.authored(["one"], title="Later gains text")
        KnowledgeDocument.objects.filter(pk=updated.pk).update(curated_text="now there is a source")
        before = db_state()
        with self.assertRaises(SourceRepresentationRefused):
            self.author(updated, [C("two")])
        self.assertNothingChanged(before)

        filed = self.shell("With file")
        KnowledgeDocument.objects.filter(pk=filed.pk).update(source_file="ai_hub/knowledge/spec.txt")
        before = db_state()
        with self.assertRaises(SourceRepresentationRefused):
            self.author(filed, [C("x")])
        self.assertNothingChanged(before)

    def test_whitespace_only_source_text_is_not_a_source_representation(self):
        document = self.shell("Blank text", curated_text="  \n\t ")
        self.assertEqual(self.author(document, [C("x")]).outcome, authoring.OUTCOME_CHANGED)

    def test_existing_chunk_metadata_is_refused_not_rewritten(self):
        document = self.authored(["one", "two"])
        chunk = document.chunks.get(chunk_index=1)
        chunk.metadata = {"window": "w001"}
        chunk.save(update_fields=["metadata"])
        before = db_state()
        with self.assertRaises(ChunkMetadataRefused):
            self.author(document, [C("one"), C("two"), C("three")])
        self.assertNothingChanged(before)
        document.chunks.filter(pk=chunk.pk).update(metadata=["not", "an", "object"])
        with self.assertRaises(ChunkMetadataRefused):
            self.author(document, [C("one")])

    def test_empty_metadata_objects_are_accepted_and_left_alone(self):
        document = self.authored(["one"])
        self.assertEqual(document.chunks.get().metadata, {})
        self.author(document, [C("one"), C("two")])
        self.assertEqual({row.metadata == {} for row in document.chunks.all()}, {True})

    def test_lifecycle_verbs_refuse_an_authored_document(self):
        document = self.authored(["one", "two"])
        before = db_state()
        with self.assertRaises(KnowledgeRegenerationError):
            regenerate_derived_chunk_set(
                document.pk, expected=self.expected(document),
                principal=KnowledgeMutationPrincipal.human("operator-1"),
            )
        self.assertNothingChanged(before)

    def test_all_refusals_are_authoring_state_errors(self):
        for error in (AuthorityRefused, StatusRefused, ProvenanceRefused,
                      SourceRepresentationRefused, ChunkMetadataRefused):
            self.assertTrue(issubclass(error, AuthoringStateRefused))
            self.assertTrue(issubclass(error, authoring.KnowledgeAuthoringError))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class InputValidationTests(AuthoringFixture, TestCase):
    SECRET = "secret-body-text"

    def assertRefused(self, document, chunks, error=KnowledgeAuthoringInvalidInput):
        before = db_state()
        with self.assertRaises(error) as raised:
            self.author(document, chunks)
        self.assertNothingChanged(before)
        self.assertNotIn(self.SECRET, str(raised.exception))
        return raised.exception

    def test_malformed_sequences_are_refused(self):
        document = self.shell()
        for bad in ([], (), "abc", b"abc", {"content": "x"}, 5, None, {C("x")}, (c for c in [C("x")])):
            with self.subTest(chunks=repr(bad)[:30]):
                self.assertRefused(document, bad)

    def test_entries_must_be_authored_chunks(self):
        document = self.shell()
        for bad in (("content", "title"), {"content": "x"}, "text", None, 7):
            with self.subTest(entry=repr(bad)):
                self.assertRefused(document, [C("fine"), bad])

    def test_content_must_be_non_blank_text(self):
        document = self.shell()
        for bad in ("", "   ", "\n\t \n", None, 5, b"bytes"):
            with self.subTest(content=repr(bad)):
                self.assertRefused(document, [AuthoredChunk(content=bad)])

    def test_nul_and_unencodable_text_are_refused(self):
        document = self.shell()
        for content in ("a\x00b", "bad \ud800 surrogate"):
            with self.subTest(field="content", value=repr(content)):
                self.assertRefused(document, [C(content)])
        self.assertRefused(document, [C("ok", "ti\x00tle")])
        self.assertRefused(document, [C("ok", "ti\ud800tle")])

    def test_titles_must_be_text(self):
        document = self.shell()
        for bad in (None, 3, b"t"):
            with self.subTest(title=repr(bad)):
                self.assertRefused(document, [AuthoredChunk(content="ok", section_title=bad)])

    def test_limits_are_enforced(self):
        document = self.shell()
        self.assertRefused(document, [C("x")] * (authoring.MAX_AUTHORED_CHUNKS + 1))
        self.assertRefused(document, [C("x" * (authoring.MAX_CHUNK_CONTENT_CHARS + 1))])
        self.assertRefused(document, [C("ok", "t" * (authoring.MAX_SECTION_TITLE_CHARS + 1))])
        with mock.patch.object(authoring, "MAX_AUTHORED_TOTAL_CHARS", 10):
            self.assertRefused(document, [C("123456"), C("123456")])

    def test_limits_at_the_boundary_are_accepted(self):
        document = self.shell()
        result = self.author(document, [
            C("x" * authoring.MAX_CHUNK_CONTENT_CHARS, "t" * authoring.MAX_SECTION_TITLE_CHARS),
        ])
        self.assertEqual(result.chunk_count, 1)

    def test_the_section_title_limit_is_the_model_limit(self):
        field = KnowledgeDocumentChunk._meta.get_field("section_title")
        self.assertEqual(authoring.MAX_SECTION_TITLE_CHARS, field.max_length)

    def test_messages_never_echo_content(self):
        document = self.shell()
        self.assertRefused(document, [AuthoredChunk(content=f"{self.SECRET}\x00")])
        self.assertRefused(document, [C(self.SECRET * 5000)])

    def test_the_maximum_set_is_accepted(self):
        document = self.shell()
        result = self.author(document, [C(f"chunk {i}") for i in range(authoring.MAX_AUTHORED_CHUNKS)])
        self.assertEqual(result.chunk_count, authoring.MAX_AUTHORED_CHUNKS)
        self.assertEqual(document.chunks.count(), authoring.MAX_AUTHORED_CHUNKS)

    def test_arguments_are_validated_on_the_changed_and_unchanged_paths(self):
        document = self.authored(["one"])
        for chunks in ([C("one")], [C("one"), C("two")]):  # unchanged, changed
            with self.subTest(chunks=len(chunks)):
                before = db_state()
                with self.assertRaises(KnowledgeMutationOperationError):
                    self.author(document, chunks, reason_code="Bad Code")
                for principal in (None, "operator", object()):
                    with self.assertRaises(KnowledgeMutationPrincipalError):
                        self.author(document, chunks, principal=principal)
                with self.assertRaises(KnowledgeMutationPrincipalError):
                    self.author(document, chunks, principal=KnowledgeMutationPrincipal("human", " "))
                with self.assertRaises(TypeError):
                    self.author(document, chunks, expected=None)
                self.assertNothingChanged(before)


# ---------------------------------------------------------------------------
# Chunk identity
# ---------------------------------------------------------------------------

class ChunkIdentityTests(AuthoringFixture, TestCase):
    def test_first_authoring_assigns_contiguous_indexes_in_order(self):
        document = self.shell()
        self.author(document, [C("alpha", "A"), C("bravo"), C("charlie", "C")])
        self.assertEqual(
            [(r[1], r[2], r[3]) for r in self.rows(document)],
            [(1, "A", "alpha"), (2, "", "bravo"), (3, "C", "charlie")],
        )
        for chunk in document.chunks.all():
            self.assertEqual(chunk.metadata, {})
            self.assertEqual(chunk.token_estimate, max(len(chunk.content.split()), 1))

    def test_inserting_at_the_top_keeps_every_existing_id_and_k1(self):
        document = self.authored(["alpha", "bravo", "charlie"])
        ids = self.ids_by_content(document)
        k1 = {c.pk: chunk_embedding_fingerprint(c) for c in document.chunks.all()}
        result = self.author(document, [C("intro"), C("alpha"), C("bravo"), C("charlie")])
        after = self.ids_by_content(document)
        self.assertEqual((result.kept, result.inserted, result.removed, result.moved), (3, 1, 0, 3))
        for content in ("alpha", "bravo", "charlie"):
            self.assertEqual(after[content], ids[content])
        self.assertNotIn(after["intro"], ids.values())
        self.assertEqual(self.contents(document), ["intro", "alpha", "bravo", "charlie"])
        for chunk in document.chunks.filter(pk__in=ids.values()):
            self.assertEqual(chunk_embedding_fingerprint(chunk), k1[chunk.pk])

    def test_a_cited_id_never_silently_refers_to_different_text(self):
        document = self.authored(["alpha", "bravo", "charlie"])
        cited = {content: pk for content, pk in self.ids_by_content(document).items()}
        self.author(document, [C("new intro"), C("alpha"), C("bravo"), C("charlie")])
        for content, pk in cited.items():
            self.assertEqual(document.chunks.get(pk=pk).content, content)

    def test_appending_keeps_existing_ids(self):
        document = self.authored(["alpha", "bravo"])
        ids = self.ids_by_content(document)
        result = self.author(document, [C("alpha"), C("bravo"), C("charlie")])
        self.assertEqual((result.kept, result.inserted, result.moved), (2, 1, 0))
        self.assertEqual(self.ids_by_content(document)["alpha"], ids["alpha"])

    def test_deleting_a_middle_chunk_keeps_the_rest_and_closes_the_gap(self):
        document = self.authored(["alpha", "bravo", "charlie"])
        ids = self.ids_by_content(document)
        result = self.author(document, [C("alpha"), C("charlie")])
        self.assertEqual((result.kept, result.removed, result.moved), (2, 1, 1))
        self.assertEqual([r[1] for r in self.rows(document)], [1, 2])
        self.assertEqual(self.ids_by_content(document)["charlie"], ids["charlie"])
        self.assertFalse(KnowledgeDocumentChunk.objects.filter(pk=ids["bravo"]).exists())

    def test_reordering_swaps_in_two_phases_and_preserves_ids_and_k1(self):
        document = self.authored(["alpha", "bravo", "charlie"])
        ids = self.ids_by_content(document)
        k1 = {c.pk: chunk_embedding_fingerprint(c) for c in document.chunks.all()}
        result = self.author(document, [C("charlie"), C("bravo"), C("alpha")])
        self.assertEqual((result.kept, result.inserted, result.removed, result.moved), (3, 0, 0, 2))
        self.assertEqual(self.contents(document), ["charlie", "bravo", "alpha"])
        self.assertEqual(self.ids_by_content(document), ids)
        for chunk in document.chunks.all():
            self.assertEqual(chunk_embedding_fingerprint(chunk), k1[chunk.pk])

    def test_changed_content_gets_a_new_id_and_the_old_one_is_gone(self):
        document = self.authored(["alpha", "bravo"])
        ids = self.ids_by_content(document)
        self.author(document, [C("alpha"), C("bravo, revised")])
        self.assertFalse(KnowledgeDocumentChunk.objects.filter(pk=ids["bravo"]).exists())
        self.assertNotIn(self.ids_by_content(document)["bravo, revised"], ids.values())
        self.assertEqual(self.ids_by_content(document)["alpha"], ids["alpha"])

    def test_a_title_only_change_is_a_new_chunk(self):
        document = self.shell()
        self.author(document, [C("body", "Old title")])
        old = document.chunks.get().pk
        self.author(document, [C("body", "New title")])
        self.assertNotEqual(document.chunks.get().pk, old)

    def test_whitespace_changes_are_changes(self):
        document = self.authored(["alpha"])
        old = document.chunks.get().pk
        result = self.author(document, [C("alpha ")])
        self.assertEqual(result.outcome, authoring.OUTCOME_CHANGED)
        self.assertNotEqual(document.chunks.get().pk, old)

    def test_duplicates_are_matched_first_in_first_out(self):
        document = self.authored(["A", "B", "A"])
        first, middle, last = [r[0] for r in self.rows(document)]
        self.author(document, [C("A"), C("A")])
        self.assertEqual([r[0] for r in self.rows(document)], [first, last])
        self.assertFalse(KnowledgeDocumentChunk.objects.filter(pk=middle).exists())

    def test_duplicates_with_reordering_are_deterministic(self):
        document = self.authored(["A", "B", "A"])
        first, middle, last = [r[0] for r in self.rows(document)]
        self.author(document, [C("B"), C("A"), C("A")])
        self.assertEqual([r[0] for r in self.rows(document)], [middle, first, last])

    def test_surplus_duplicates_keep_the_lowest_index_row(self):
        document = self.authored(["A", "A", "A"])
        first = self.rows(document)[0][0]
        result = self.author(document, [C("A")])
        self.assertEqual((result.kept, result.removed), (1, 2))
        self.assertEqual([r[0] for r in self.rows(document)], [first])

    def test_the_plan_is_a_pure_function_of_its_inputs(self):
        existing = [
            {"pk": 7, "chunk_index": 1, "section_title": "", "content": "A"},
            {"pk": 9, "chunk_index": 2, "section_title": "", "content": "B"},
            {"pk": 4, "chunk_index": 3, "section_title": "", "content": "A"},
        ]
        desired = [C("A"), C("A"), C("Z")]
        self.assertEqual(authoring._plan(existing, desired), authoring._plan(existing, desired))
        plan = authoring._plan(existing, desired)
        self.assertEqual([item[0] for item in plan.kept], [7, 4])
        self.assertEqual(plan.removed, (9,))

    def test_line_ending_and_unicode_form_differences_are_not_edits(self):
        document = self.shell()
        self.author(document, [C("line one\nline two", "Café")])
        before = db_state()
        composed = unicodedata.normalize("NFC", "Café")
        decomposed = unicodedata.normalize("NFD", "Café")
        self.assertNotEqual(composed, decomposed)
        for content, title in (("line one\r\nline two", composed), ("line one\rline two", decomposed)):
            with self.subTest(content=repr(content)):
                result = self.author(document, [C(content, title)])
                self.assertEqual(result.outcome, authoring.OUTCOME_UNCHANGED)
        self.assertNothingChanged(before)  # the stored text was not rewritten

    def test_database_ids_are_not_reused_after_a_delete(self):
        document = self.authored(["alpha", "bravo"])
        highest = max(r[0] for r in self.rows(document))
        self.author(document, [C("alpha")])
        self.author(document, [C("alpha"), C("fresh")])
        self.assertGreater(self.ids_by_content(document)["fresh"], highest)

    def test_the_stored_set_equals_the_requested_one_under_c1(self):
        document = self.authored(["alpha", "bravo"])
        requested = [C("zulu", "Z"), C("alpha"), C("yankee")]
        result = self.author(document, requested)
        document.refresh_from_db()
        self.assertEqual(
            build_snapshot(document).observed_chunk_set_fingerprint, result.chunk_set_fingerprint
        )
        self.assertEqual(
            result.chunk_set_fingerprint,
            chunk_set_fingerprint(authoring._desired_rows(requested)),
        )

    def test_the_result_counts_describe_the_change(self):
        document = self.authored(["a", "b", "c", "d"])
        result = self.author(document, [C("d"), C("a"), C("new")])
        self.assertEqual(result.document_id, document.pk)
        self.assertEqual(result.chunk_count, 3)
        self.assertEqual((result.kept, result.inserted, result.removed), (2, 1, 2))
        self.assertEqual(result.moved, 2)
        self.assertFalse(hasattr(result, "event_id"))

    def test_a_document_with_irregular_indexes_is_normalised_to_contiguous(self):
        document = self.authored(["alpha", "bravo", "charlie"])
        document.chunks.filter(content="bravo").update(chunk_index=7)
        document.chunks.filter(content="charlie").update(chunk_index=40)
        self.author(document, [C("alpha"), C("bravo"), C("charlie")])
        self.assertEqual([r[1] for r in self.rows(document)], [1, 2, 3])


# ---------------------------------------------------------------------------
# No-op results, compare-and-swap, audit
# ---------------------------------------------------------------------------

class NoOpAndConflictTests(AuthoringFixture, TestCase):
    def test_an_identical_request_is_unchanged_with_no_write_and_no_event(self):
        document = self.authored(["alpha", "bravo"])
        before = db_state()
        events = self.events()
        result = self.author(document, [C("alpha"), C("bravo")])
        self.assertEqual(result.outcome, authoring.OUTCOME_UNCHANGED)
        self.assertEqual((result.kept, result.inserted, result.removed, result.moved), (2, 0, 0, 0))
        self.assertEqual(self.events(), events)
        self.assertNothingChanged(before)

    def governed_spy(self):
        """Record whether the governed boundary was entered and whether it was
        active at each `_load_existing` call."""
        spy = {"entered": 0, "active": False, "loads_outside": 0}
        real_governed = authoring._governed_knowledge_mutation
        real_load = authoring._load_existing

        class Spy:
            def __init__(self, *args, **kwargs):
                self.inner = real_governed(*args, **kwargs)

            def __enter__(self):
                spy["entered"] += 1
                value = self.inner.__enter__()
                spy["active"] = True
                return value

            def __exit__(self, *exc):
                spy["active"] = False
                return self.inner.__exit__(*exc)

        def load(document):
            if not spy["active"]:
                spy["loads_outside"] += 1
            return real_load(document)

        return spy, Spy, load

    def test_an_identical_request_is_decided_inside_the_governed_boundary(self):
        """T2 audit finding 1: no unlocked read may decide `unchanged`."""
        document = self.authored(["alpha"])
        spy, Spy, load = self.governed_spy()
        before = db_state()
        with mock.patch.object(authoring, "_governed_knowledge_mutation", Spy), \
                mock.patch.object(authoring, "_load_existing", load):
            result = self.author(document, [C("alpha")])
        self.assertEqual(result.outcome, authoring.OUTCOME_UNCHANGED)
        self.assertEqual(spy["entered"], 1)
        self.assertEqual(spy["loads_outside"], 0)
        self.assertNothingChanged(before)  # rolled back: no write, no event

    def test_a_concurrent_change_before_the_read_cannot_become_unchanged(self):
        """T2 audit finding 1, the reproduced race: stored A, expected A, a writer
        commits A -> B, the host asks for B. It must never return `unchanged` on the
        strength of an unlocked read."""
        document = self.authored(["alpha"])
        planned = self.expected(document)  # state A
        spy, Spy, real_load = self.governed_spy()
        raced = []

        def load_after_a_concurrent_writer(doc):
            if not spy["active"] and not raced:  # the old unlocked read
                raced.append(1)
                doc.chunks.update(content="beta")  # A -> B, committed elsewhere
            return real_load(doc)

        with mock.patch.object(authoring, "_governed_knowledge_mutation", Spy), \
                mock.patch.object(authoring, "_load_existing", load_after_a_concurrent_writer):
            try:
                result = self.author(document, [C("beta")], expected=planned)
            except KnowledgeMutationConflict:
                return  # acceptable: the stale expectation was caught
        self.assertNotEqual(result.outcome, authoring.OUTCOME_UNCHANGED)

    def test_a_writer_committing_just_before_the_lock_is_a_conflict(self):
        document = self.authored(["alpha"])
        planned = self.expected(document)
        real_validate = authoring._validate_chunks

        def writer_then_validate(chunks):
            desired = real_validate(chunks)
            document.chunks.update(content="beta")  # committed before the transaction starts
            return desired

        before_events = self.events()
        with mock.patch.object(authoring, "_validate_chunks", writer_then_validate):
            with self.assertRaises(KnowledgeMutationConflict):
                self.author(document, [C("beta")], expected=planned)
        self.assertEqual(self.events(), before_events)
        self.assertEqual(self.contents(document), ["beta"])  # the other writer's state stands

    def test_repeating_a_changed_request_is_unchanged_the_second_time(self):
        document = self.shell()
        requested = [C("alpha"), C("bravo")]
        self.assertEqual(self.author(document, requested).outcome, authoring.OUTCOME_CHANGED)
        self.assertEqual(self.author(document, requested).outcome, authoring.OUTCOME_UNCHANGED)
        self.assertEqual(self.events(), 1)

    def test_a_stale_expectation_is_a_conflict_even_when_the_state_matches(self):
        document = self.authored(["alpha"])
        planned = self.expected(document)
        KnowledgeDocument.objects.filter(pk=document.pk).update(status=Status.ACTIVE)
        before = db_state()
        with self.assertRaises(KnowledgeMutationConflict):
            self.author(document, [C("alpha")], expected=planned)  # identical request
        self.assertNothingChanged(before)

    def test_a_changed_request_with_an_edited_chunk_conflicts(self):
        document = self.authored(["alpha", "bravo"])
        planned = self.expected(document)
        document.chunks.filter(content="bravo").update(content="edited in the admin")
        before = db_state()
        with self.assertRaises(KnowledgeMutationConflict):
            self.author(document, [C("alpha"), C("bravo"), C("charlie")], expected=planned)
        self.assertNothingChanged(before)

    def test_status_and_collection_drift_conflict(self):
        document = self.authored(["alpha"])
        planned = self.expected(document)
        other = KnowledgeCollection.objects.create(application_scope=self.scope, name="Second")
        KnowledgeDocument.objects.filter(pk=document.pk).update(collection=other)
        before = db_state()
        with self.assertRaises(KnowledgeMutationConflict):
            self.author(document, [C("alpha"), C("more")], expected=planned)
        self.assertNothingChanged(before)

    def test_exactly_one_content_free_event_per_change(self):
        document = self.authored(["alpha secret", "bravo secret"])
        self.author(document, [C("bravo secret"), C("charlie secret")])
        self.assertEqual(self.events(), 2)
        event = KnowledgeLifecycleEvent.objects.order_by("-pk").first()
        self.assertEqual((event.previous_chunk_count, event.new_chunk_count), (2, 2))
        self.assertEqual((event.previous_authority_mode, event.new_authority_mode), (Mode.EXPLICIT, Mode.EXPLICIT))
        for field in event._meta.concrete_fields:
            value = getattr(event, field.attname)
            if isinstance(value, str):
                self.assertNotIn("secret", value)
                self.assertLessEqual(len(value), 200)
        self.assertNotEqual(
            event.previous_observed_chunk_set_fingerprint, event.new_observed_chunk_set_fingerprint
        )

    def test_the_result_never_claims_an_event_it_cannot_prove(self):
        fields = {f.name for f in authoring.AuthoringResult.__dataclass_fields__.values()}
        self.assertNotIn("event_id", fields)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

class RollbackTests(AuthoringFixture, TestCase):
    def mixed_document(self):
        return self.authored(["alpha", "bravo", "charlie"])

    # existing [alpha, bravo, charlie] -> [charlie, NEW, alpha]: one removal, two
    # moves and one insertion, so every phase has work to do.
    REQUEST = [C("charlie"), C("new"), C("alpha")]

    def test_a_failure_in_any_phase_rolls_everything_back(self):
        for phase in ("_delete_rows", "_move_rows", "_insert_rows", "_declare_explicit", "_verify_result"):
            with self.subTest(phase=phase):
                document = self.mixed_document()
                planned = self.expected(document)
                before = db_state()
                with mock.patch.object(authoring, phase, side_effect=RuntimeError(f"{phase} failed")):
                    with self.assertRaises(RuntimeError):
                        self.author(document, self.REQUEST, expected=planned)
                self.assertNothingChanged(before)
                self.assertEqual(self.contents(document), ["alpha", "bravo", "charlie"])
                document.refresh_from_db()
                self.assertEqual(document.chunk_authority_mode, Mode.EXPLICIT)
                KnowledgeDocument.objects.filter(pk=document.pk).delete()

    def test_an_audit_event_failure_rolls_everything_back(self):
        document = self.mixed_document()
        planned = self.expected(document)
        before = db_state()
        with mock.patch(
            "ai_hub.services.knowledge_mutation._record_event", side_effect=RuntimeError("audit down")
        ):
            with self.assertRaises(RuntimeError):
                self.author(document, self.REQUEST, expected=planned)
        self.assertNothingChanged(before)

    def test_a_first_authoring_failure_leaves_the_shell_untouched(self):
        document = self.shell()
        before = db_state()
        with mock.patch.object(authoring, "_declare_explicit", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.author(document, [C("alpha")])
        self.assertNothingChanged(before)
        document.refresh_from_db()
        self.assertEqual((document.chunk_authority_mode, document.chunks.count()), (Mode.UNKNOWN, 0))

    def test_a_wrong_stored_set_is_detected_and_rolled_back(self):
        document = self.mixed_document()
        planned = self.expected(document)
        before = db_state()

        def insert_the_wrong_text(doc, plan):
            for position, _chunk in plan.inserted:
                KnowledgeDocumentChunk.objects.create(
                    document=doc, chunk_index=position, content="tampered"
                )

        with mock.patch.object(authoring, "_insert_rows", side_effect=insert_the_wrong_text):
            with self.assertRaises(AuthoringIntegrityError):
                self.author(document, self.REQUEST, expected=planned)
        self.assertNothingChanged(before)

    def test_a_host_transaction_can_publish_authoring_and_activation_atomically(self):
        document = self.shell()
        planned = self.expected(document)
        before = db_state()
        with self.assertRaises(RuntimeError):
            with transaction.atomic():
                author_explicit_chunk_set(
                    document.pk, [C("alpha")], application_scope=self.scope,
                    expected=planned, principal=self.principal,
                )
                KnowledgeDocument.objects.filter(pk=document.pk).update(status=Status.ACTIVE)
                raise RuntimeError("the host aborts its own publication")
        self.assertNothingChanged(before)

    def test_a_conflict_leaves_nothing_behind(self):
        document = self.mixed_document()
        planned = self.expected(document)
        document.chunks.filter(content="alpha").update(content="edited")
        before = db_state()
        with self.assertRaises(KnowledgeMutationConflict):
            self.author(document, self.REQUEST, expected=planned)
        self.assertNothingChanged(before)


# ---------------------------------------------------------------------------
# Publication, retrieval, citations and vectors
# ---------------------------------------------------------------------------

class PublicationAndRetrievalTests(AuthoringFixture, TestCase):
    def found(self, agent, term):
        return search_knowledge(agent, query=term, limit=5)["total"]

    def test_authoring_never_activates_and_activation_is_a_separate_step(self):
        agent = self.agent()
        document = self.shell()
        self.author(document, [C("zebrafish migration notes")])
        document.refresh_from_db()
        self.assertEqual(document.status, Status.DRAFT)
        self.assertEqual(self.found(agent, "zebrafish"), 0)
        self.activate(document)
        self.assertEqual(self.found(agent, "zebrafish"), 1)

    def test_an_update_to_an_active_document_is_visible_after_the_call(self):
        agent = self.agent()
        document = self.authored(["narwhal old text"])
        self.activate(document)
        self.assertEqual((self.found(agent, "narwhal"), self.found(agent, "pelican")), (1, 0))
        self.author(document, [C("pelican new text")])
        self.assertEqual((self.found(agent, "narwhal"), self.found(agent, "pelican")), (0, 1))

    def test_citation_integrity_across_an_update(self):
        agent = self.agent()
        document = self.authored(["kept paragraph", "changed paragraph"])
        self.activate(document)
        ids = self.ids_by_content(document)
        self.author(document, [C("a new first paragraph"), C("kept paragraph"), C("changed paragraph v2")])
        kept = read_knowledge_chunk(agent, chunk_id=ids["kept paragraph"])
        self.assertEqual(kept["content"], "kept paragraph")
        self.assertEqual(cite_knowledge_source(agent, chunk_id=ids["kept paragraph"])["citation"]["chunk_index"], 2)
        with self.assertRaises(ValidationError):
            read_knowledge_chunk(agent, chunk_id=ids["changed paragraph"])
        with self.assertRaises(ValidationError):
            cite_knowledge_source(agent, chunk_id=ids["changed paragraph"])

    def embedding_config(self):
        provider = ProviderConfig.objects.create(
            name="embed-p", provider_type="openai",
            declared_locality=ProviderConfig.DeclaredLocality.LOCAL,
        )
        return EmbeddingModelConfig.objects.create(
            name="embed", provider=provider, model_name="m", model_revision="v1",
            vector_dimension=4,
            distance_metric=EmbeddingModelConfig.DistanceMetric.COSINE,
            normalization=EmbeddingModelConfig.Normalization.L2,
        )

    def current_vector_chunks(self, e1):
        return {
            row.chunk_id
            for row in load_current_vectors(
                application_scope=self.scope, e1=e1, collection_ids=[self.collection.pk]
            )
        }

    def test_vector_staleness_follows_chunk_identity(self):
        config = self.embedding_config()
        e1 = resolve_embedding_contract(config).e1
        document = self.authored(["alpha", "bravo", "charlie"])
        ids = self.ids_by_content(document)
        for chunk in document.chunks.all():
            store_chunk_vector(
                application_scope=self.scope, chunk=chunk,
                embedding_model_config=config, vector=[0.1, 0.2, 0.3, 0.4],
            )
        self.assertEqual(self.current_vector_chunks(e1), set(ids.values()))

        # alpha and charlie are kept (and moved); bravo is edited; one chunk is new.
        self.author(document, [C("intro"), C("charlie"), C("alpha"), C("bravo edited")])

        kept = {ids["alpha"], ids["charlie"]}
        self.assertEqual(self.current_vector_chunks(e1), kept)  # still current, untouched
        self.assertFalse(KnowledgeChunkEmbedding.objects.filter(chunk_id=ids["bravo"]).exists())
        for new_row in document.chunks.exclude(pk__in=kept):
            self.assertFalse(new_row.embeddings.exists())  # missing until indexed

    def test_preflight_sees_an_authored_and_activated_document_as_clean(self):
        document = self.authored(["alpha", "bravo"])
        self.activate(document)
        report = run_knowledge_preflight(collection_ids=[self.collection.pk])
        row = next(r for r in report["documents"] if r["document_id"] == document.pk)
        self.assertEqual(
            (row["structural_state"], row["authority_mode"], row["issues"]),
            ("READY_CANONICAL", "explicit", []),
        )


# ---------------------------------------------------------------------------
# Isolation: not Academy, not model-executable, not an operator verb
# ---------------------------------------------------------------------------

class IsolationTests(TestCase):
    MODULE = pathlib.Path(authoring.__file__).resolve()
    BASE = MODULE.parent.parent  # ai_hub/

    def test_the_module_imports_nothing_from_a_host(self):
        tree = ast.parse(self.MODULE.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertTrue(imported)
        for name in imported:
            self.assertFalse(
                name.split(".")[0] in {"academy", "support_demo", "dashboard"},
                f"imports host module {name}",
            )

    def test_no_production_module_references_the_service(self):
        """It is not a tool, not an Admin action, not part of the lifecycle command."""
        offenders = []
        for path in sorted(self.BASE.rglob("*.py")):
            if path == self.MODULE or path.name.startswith("test_") or path.name == "tests.py":
                continue
            if "knowledge_authoring" in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(self.BASE).as_posix())
        self.assertEqual(offenders, [])

    def test_it_is_not_a_registered_or_allowed_tool_callable(self):
        from ai_hub.services.knowledge_tooling import KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES

        callables = set(KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES.values())
        allowed = set(getattr(settings, "AI_HUB_ALLOWED_TOOL_CALLABLES", ()))
        for path in callables | allowed:
            self.assertNotIn("authoring", path)
            self.assertNotIn("author_explicit", path)

    def test_the_lifecycle_command_does_not_expose_it(self):
        command = self.BASE / "management" / "commands" / "knowledge_lifecycle_action.py"
        text = command.read_text(encoding="utf-8")
        self.assertNotIn("author_explicit_chunk_set", text)
        self.assertNotIn("knowledge_authoring", text)

    def test_the_public_surface_is_the_one_operation(self):
        public = {
            name for name, value in vars(authoring).items()
            if callable(value) and not name.startswith("_") and getattr(value, "__module__", "") == authoring.__name__
            and not isinstance(value, type)
        }
        self.assertEqual(public, {"author_explicit_chunk_set"})
