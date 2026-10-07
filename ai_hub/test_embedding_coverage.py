"""S-27 — operator corpus embedding: coverage report and governed execution.

Every provider call is mocked. **No real Ollama server, no network.**

What this file proves, in the order a reviewer should care:

    report mode sends nothing and writes nothing
    every write goes through S-20's index_chunk_embedding_local()
    all target collections authorize, or nothing is sent
    every S-20 failure category is classified run-fatal or chunk-local
    a partial index is never reported as complete
    no chunk content ever reaches the output
"""

import ast
import inspect
import json
from io import StringIO
from pathlib import Path
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from ai_hub.models import (
    ApplicationScope,
    EmbeddingModelConfig,
    KnowledgeChunkEmbedding,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ProviderConfig,
    ProviderGrant,
    ToolDefinition,
)
from ai_hub.services import embedding_coverage
from ai_hub.services.embedding_client import (
    EmbeddingProviderExecutionError,
    EmbeddingProviderResult,
    ErrorCategory,
)
from ai_hub.services.embedding_coverage import (
    CHUNK_LOCAL_CATEGORIES,
    RUN_FATAL_CATEGORIES,
    CoverageState,
    ExtraFailureCategory,
    IndexTargetError,
    PreconditionCode,
    TargetErrorCategory,
    build_coverage_report,
    execute_embedding_index,
    is_run_fatal,
    resolve_index_target,
)
from ai_hub.services.embedding_execution import (
    EmbeddingExecutionError,
    ExecutionStatus,
    FailureCategory,
    index_chunk_embedding_local,
)

LOCALITY = ProviderConfig.DeclaredLocality
ACTIVE = KnowledgeDocument.Status.ACTIVE
ARCHIVED = KnowledgeDocument.Status.ARCHIVED

#: Where S-20 reaches the transport registry. Patching here intercepts every
#: provider call the run could make, without any socket.
TRANSPORT_PATH = "ai_hub.services.embedding_execution.resolve_embedding_transport"
OLLAMA_POST = "ai_hub.services.embedding_client.requests.post"

#: Distinctive text that must never appear in any output.
MARKER = "ZEBRA-MARKER-9137"

VALUES = (0.1, 0.2, 0.3, 0.4)

ROOT = Path(__file__).resolve().parent
SERVICE_FILE = ROOT / "services" / "embedding_coverage.py"
COMMAND_FILE = ROOT / "management" / "commands" / "knowledge_embedding_index.py"


def fake_transport(values=VALUES, *, hook=None, error=None, values_for=None):
    """Records calls; never touches a socket.

    `hook(call_index)` runs before returning; `error(call_index)` may return an
    exception to raise; `values_for(call_index)` may override the vector.
    """
    calls = []

    def transport(*, provider, contract, text):
        index = len(calls)
        calls.append(text)
        if hook is not None:
            hook(index)
        if error is not None:
            exc = error(index)
            if exc is not None:
                raise exc
        returned = values_for(index) if values_for is not None else values
        return EmbeddingProviderResult(
            values=tuple(returned), provider_type=provider.provider_type,
            provider_model="fake",
        )

    transport.calls = calls
    return transport


class WorldMixin:
    """One scope, two active collections, a LOCAL granted Ollama provider."""

    def build_world(
        self, *, locality=LOCALITY.LOCAL, provider_type="ollama", grant=True,
        max_input_chars=8000, chunks_per_collection=2,
    ):
        self.scope = ApplicationScope.objects.create(name="Index App", slug="index-app")
        self.collection_a = KnowledgeCollection.objects.create(
            name="Index A", application_scope=self.scope
        )
        self.collection_b = KnowledgeCollection.objects.create(
            name="Index B", application_scope=self.scope
        )
        self.chunks = []
        for collection in (self.collection_a, self.collection_b):
            for number in range(chunks_per_collection):
                self.chunks.append(self.make_chunk(collection, f"{collection.name} {number}"))
        self.provider = ProviderConfig.objects.create(
            name="Index Ollama", provider_type=provider_type,
            base_url="http://ollama.internal:11434", declared_locality=locality,
        )
        self.config = EmbeddingModelConfig.objects.create(
            name="index-embed", provider=self.provider, model_name="nomic-embed-text",
            model_revision="v1", vector_dimension=4,
            distance_metric=EmbeddingModelConfig.DistanceMetric.COSINE,
            normalization=EmbeddingModelConfig.Normalization.NONE,
            max_input_chars=max_input_chars,
        )
        if grant is not None:
            ProviderGrant.objects.create(
                application_scope=self.scope, provider=self.provider,
                allow_embeddings=grant,
            )

    def make_chunk(self, collection, title, *, content=None, section_title=None, status=ACTIVE):
        document = KnowledgeDocument.objects.create(
            collection=collection, title=title, curated_text="x", status=status,
        )
        return KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1,
            section_title=f"Section {MARKER}" if section_title is None else section_title,
            content=f"Body {title} {MARKER}" if content is None else content,
        )

    def target(self, **overrides):
        kwargs = dict(scope_ref=self.scope.slug, embedding_config_ref=str(self.config.pk))
        kwargs.update(overrides)
        return resolve_index_target(**kwargs)

    def execute(self, transport, *, limit=None, **target_overrides):
        with mock.patch(TRANSPORT_PATH, return_value=transport):
            return execute_embedding_index(self.target(**target_overrides), limit=limit)

    def index_all(self):
        transport = fake_transport()
        run = self.execute(transport)
        self.assertTrue(run.complete)
        return run

    def states(self, report):
        return {chunk.chunk_id: chunk.state for chunk in report.chunks}


def db_snapshot():
    return {
        "chunks": list(
            KnowledgeDocumentChunk.objects.order_by("pk").values(
                "pk", "document_id", "section_title", "content", "metadata", "updated_at",
            )
        ),
        "documents": list(
            KnowledgeDocument.objects.order_by("pk").values("pk", "status", "updated_at")
        ),
        "embeddings": list(
            KnowledgeChunkEmbedding.objects.order_by("pk").values(
                "pk", "chunk_id", "e1", "k1", "vector_bytes", "updated_at",
            )
        ),
    }


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

class TargetResolutionTests(WorldMixin, TestCase):
    def setUp(self):
        self.build_world()

    def test_scope_by_slug_or_id_and_config_by_name_or_id(self):
        for scope_ref, config_ref in (
            (self.scope.slug, self.config.name),
            (str(self.scope.pk), str(self.config.pk)),
        ):
            with self.subTest(scope_ref=scope_ref, config_ref=config_ref):
                target = resolve_index_target(
                    scope_ref=scope_ref, embedding_config_ref=config_ref
                )
                self.assertEqual(target.application_scope, self.scope)
                self.assertEqual(target.embedding_model_config, self.config)

    def test_there_is_no_default_scope_or_configuration(self):
        for kwargs, category in (
            ({"scope_ref": "", "embedding_config_ref": self.config.name},
             TargetErrorCategory.SCOPE_NOT_FOUND),
            ({"scope_ref": None, "embedding_config_ref": self.config.name},
             TargetErrorCategory.SCOPE_NOT_FOUND),
            ({"scope_ref": self.scope.slug, "embedding_config_ref": ""},
             TargetErrorCategory.EMBEDDING_CONFIG_NOT_FOUND),
            ({"scope_ref": "nope", "embedding_config_ref": self.config.name},
             TargetErrorCategory.SCOPE_NOT_FOUND),
            ({"scope_ref": self.scope.slug, "embedding_config_ref": "nope"},
             TargetErrorCategory.EMBEDDING_CONFIG_NOT_FOUND),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(IndexTargetError) as caught:
                    resolve_index_target(**kwargs)
                self.assertEqual(caught.exception.category, category)

    def test_implicit_target_is_every_active_collection_of_the_scope(self):
        inactive = KnowledgeCollection.objects.create(
            name="Index Inactive", application_scope=self.scope, is_active=False
        )
        other_scope = ApplicationScope.objects.create(name="Other", slug="other")
        KnowledgeCollection.objects.create(name="Foreign", application_scope=other_scope)

        target = self.target()
        self.assertEqual(
            target.collection_ids, (self.collection_a.pk, self.collection_b.pk)
        )
        self.assertFalse(target.collections_explicit)
        self.assertEqual(target.excluded_inactive_collection_ids, (inactive.pk,))

    def test_a_collection_outside_the_scope_is_refused_not_dropped(self):
        other_scope = ApplicationScope.objects.create(name="Other", slug="other")
        foreign = KnowledgeCollection.objects.create(
            name="Foreign", application_scope=other_scope
        )
        with self.assertRaises(IndexTargetError) as caught:
            self.target(collection_ids=[self.collection_a.pk, foreign.pk])
        self.assertEqual(
            caught.exception.category, TargetErrorCategory.COLLECTION_OUTSIDE_SCOPE
        )

    def test_an_unknown_collection_is_refused(self):
        with self.assertRaises(IndexTargetError) as caught:
            self.target(collection_ids=[999999])
        self.assertEqual(caught.exception.category, TargetErrorCategory.COLLECTION_NOT_FOUND)

    def test_explicit_collections_narrow_the_target(self):
        target = self.target(collection_ids=[self.collection_b.pk, self.collection_b.pk])
        self.assertEqual(target.collection_ids, (self.collection_b.pk,))
        self.assertTrue(target.collections_explicit)


# ---------------------------------------------------------------------------
# Run-level preconditions: all or nothing, before any provider call
# ---------------------------------------------------------------------------

class PreconditionTests(WorldMixin, TestCase):
    def assert_refused_without_calls(self, expected_code, **target_overrides):
        transport = fake_transport()
        with mock.patch(OLLAMA_POST) as post:
            run = self.execute(transport, **target_overrides)
        self.assertTrue(run.refused)
        self.assertFalse(run.complete)
        self.assertEqual(run.outcomes, ())
        self.assertEqual(transport.calls, [])
        post.assert_not_called()
        self.assertIn(expected_code, run.before.preconditions.refusal_codes)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)
        return run

    def test_no_provider_grant_refuses(self):
        self.build_world(grant=None)
        self.assert_refused_without_calls("no_provider_grant")

    def test_grant_without_embeddings_refuses(self):
        self.build_world(grant=False)
        self.assert_refused_without_calls("grant_does_not_allow_embeddings")

    def test_undeclared_locality_refuses(self):
        self.build_world(locality=LOCALITY.UNKNOWN)
        self.assert_refused_without_calls("provider_locality_undeclared")

    def test_external_provider_refuses_even_when_egress_is_allowed(self):
        self.build_world(locality=LOCALITY.EXTERNAL)
        self.scope.allow_external_embedding_corpus_egress = True
        self.scope.save()
        self.assert_refused_without_calls(PreconditionCode.LOCAL_ONLY_EXECUTION_REQUIRED)

    def test_unsupported_transport_refuses(self):
        self.build_world(provider_type="openai")
        self.assert_refused_without_calls(ErrorCategory.UNSUPPORTED_EMBEDDING_TRANSPORT)

    def test_inactive_configuration_refuses_and_reports_no_coverage(self):
        self.build_world()
        self.config.is_active = False
        self.config.save()
        run = self.assert_refused_without_calls(PreconditionCode.EMBEDDING_CONTRACT_INVALID)
        self.assertFalse(run.before.coverage_available)

    def test_one_denying_collection_refuses_the_whole_run(self):
        self.build_world()
        inactive = KnowledgeCollection.objects.create(
            name="Index Inactive", application_scope=self.scope, is_active=False
        )
        self.make_chunk(inactive, "Hidden")
        run = self.assert_refused_without_calls(
            "collection_inactive",
            collection_ids=[self.collection_a.pk, inactive.pk],
        )
        allowed = {
            decision.collection_id: decision.allowed
            for decision in run.before.preconditions.collection_decisions
        }
        self.assertEqual(allowed, {self.collection_a.pk: True, inactive.pk: False})

    def test_inactive_scope_refuses(self):
        self.build_world()
        self.scope.is_active = False
        self.scope.save()
        self.assert_refused_without_calls("scope_inactive")


# ---------------------------------------------------------------------------
# Coverage states
# ---------------------------------------------------------------------------

class CoverageStateTests(WorldMixin, TestCase):
    def setUp(self):
        self.build_world()

    def test_everything_is_missing_before_indexing(self):
        report = build_coverage_report(self.target())
        self.assertEqual(report.counts()[CoverageState.MISSING], 4)
        self.assertFalse(report.complete)

    def test_everything_is_current_after_indexing(self):
        self.index_all()
        report = build_coverage_report(self.target())
        self.assertEqual(report.counts()[CoverageState.CURRENT], 4)
        self.assertTrue(report.complete)

    def test_an_edited_chunk_is_stale_with_its_reason(self):
        self.index_all()
        chunk = self.chunks[0]
        chunk.content = "edited body"
        chunk.save()
        report = build_coverage_report(self.target())
        self.assertEqual(self.states(report)[chunk.pk], CoverageState.STALE)
        self.assertEqual(report.stale_reason_counts(), {"k1_mismatch": 1})

    def test_non_active_documents_are_not_eligible(self):
        draft = self.make_chunk(self.collection_a, "Draft", status=KnowledgeDocument.Status.DRAFT)
        report = build_coverage_report(self.target())
        self.assertNotIn(draft.pk, self.states(report))
        self.assertEqual(len(report.chunks), 4)

    def test_archived_after_indexing_is_counted_outside_eligibility_and_kept(self):
        self.index_all()
        document = self.chunks[0].document
        document.status = ARCHIVED
        document.save()
        report = build_coverage_report(self.target())
        self.assertEqual(report.outside_eligibility_vector_count, 1)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 4)

    def test_empty_and_oversized_chunks_are_not_embeddable(self):
        empty = self.make_chunk(self.collection_a, "Empty", content="", section_title="")
        large = self.make_chunk(self.collection_a, "Large", content="y" * 9000)
        report = build_coverage_report(self.target())
        reasons = {
            chunk.chunk_id: chunk.reason_codes
            for chunk in report.chunks_in(CoverageState.NOT_EMBEDDABLE)
        }
        self.assertEqual(
            reasons,
            {
                empty.pk: (FailureCategory.EMBEDDING_INPUT_EMPTY,),
                large.pk: (FailureCategory.EMBEDDING_INPUT_TOO_LARGE,),
            },
        )

    def test_report_rule_agrees_with_the_authoritative_S20_check(self):
        """The report's copy of the input rules must not drift from S-20's."""
        large = self.make_chunk(self.collection_a, "Large", content="y" * 9000)
        report = build_coverage_report(self.target())
        self.assertEqual(self.states(report)[large.pk], CoverageState.NOT_EMBEDDABLE)
        with mock.patch(TRANSPORT_PATH, return_value=fake_transport()):
            with self.assertRaises(EmbeddingExecutionError) as caught:
                index_chunk_embedding_local(
                    application_scope=self.scope, chunk=large,
                    embedding_model_config=self.config,
                )
        self.assertEqual(caught.exception.category, FailureCategory.EMBEDDING_INPUT_TOO_LARGE)

    def test_a_vector_under_another_contract_does_not_count(self):
        self.index_all()
        self.config.model_revision = "v2"
        self.config.save()
        report = build_coverage_report(self.target())
        self.assertEqual(report.counts()[CoverageState.MISSING], 4)


# ---------------------------------------------------------------------------
# Report mode is read-only
# ---------------------------------------------------------------------------

class ReportReadOnlyTests(WorldMixin, TestCase):
    def setUp(self):
        self.build_world()

    def test_report_changes_nothing_and_calls_no_provider(self):
        self.index_all()
        stale = self.chunks[1]
        stale.content = "changed"
        stale.save()
        before = db_snapshot()
        transport_registry = mock.Mock()
        with mock.patch(TRANSPORT_PATH, transport_registry), mock.patch(OLLAMA_POST) as post:
            build_coverage_report(self.target())
            call_command(
                "knowledge_embedding_index", "--scope", self.scope.slug,
                "--embedding-config", str(self.config.pk), stdout=StringIO(),
            )
            call_command(
                "knowledge_embedding_index", "--scope", self.scope.slug,
                "--embedding-config", str(self.config.pk), "--json", stdout=StringIO(),
            )
        transport_registry.assert_not_called()
        post.assert_not_called()
        self.assertEqual(before, db_snapshot())


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class ExecutionTests(WorldMixin, TestCase):
    def setUp(self):
        self.build_world()

    def test_missing_chunks_are_stored_in_chunk_id_order(self):
        transport = fake_transport()
        run = self.execute(transport)
        self.assertTrue(run.complete)
        self.assertEqual(run.outcome_counts(), {ExecutionStatus.STORED: 4})
        self.assertEqual(
            [outcome.chunk_id for outcome in run.outcomes],
            sorted(chunk.pk for chunk in self.chunks),
        )
        self.assertEqual(len(transport.calls), 4)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 4)

    def test_a_second_run_makes_zero_provider_calls(self):
        self.index_all()
        transport = fake_transport()
        run = self.execute(transport)
        self.assertTrue(run.complete)
        self.assertEqual(run.attempted, 0)
        self.assertEqual(transport.calls, [])

    def test_a_stale_chunk_is_re_embedded_and_becomes_current(self):
        self.index_all()
        chunk = self.chunks[2]
        chunk.content = "edited"
        chunk.save()
        transport = fake_transport()
        run = self.execute(transport)
        self.assertEqual([outcome.chunk_id for outcome in run.outcomes], [chunk.pk])
        self.assertTrue(run.complete)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 4)

    def test_limit_is_never_silent(self):
        transport = fake_transport()
        run = self.execute(transport, limit=3)
        self.assertEqual(run.attempted, 3)
        self.assertEqual(run.remaining_due_to_limit, 1)
        self.assertFalse(run.complete)

    def test_limit_must_be_positive(self):
        for bad in (0, -1, True, "2"):
            with self.subTest(limit=bad):
                with self.assertRaises(ValueError):
                    execute_embedding_index(self.target(), limit=bad)

    def test_not_embeddable_chunks_are_never_sent_and_keep_the_run_incomplete(self):
        large = self.make_chunk(self.collection_a, "Large", content="y" * 9000)
        transport = fake_transport()
        run = self.execute(transport)
        self.assertNotIn(large.pk, [outcome.chunk_id for outcome in run.outcomes])
        self.assertEqual(len(transport.calls), 4)
        self.assertFalse(run.complete)

    def test_only_scope_and_target_collections_are_touched(self):
        other_scope = ApplicationScope.objects.create(name="Other", slug="other")
        foreign = KnowledgeCollection.objects.create(name="Foreign", application_scope=other_scope)
        foreign_chunk = self.make_chunk(foreign, "Foreign")
        transport = fake_transport()
        run = self.execute(transport, collection_ids=[self.collection_b.pk])
        touched = {outcome.chunk_id for outcome in run.outcomes}
        self.assertEqual(
            touched,
            {chunk.pk for chunk in self.chunks if chunk.document.collection_id == self.collection_b.pk},
        )
        self.assertFalse(KnowledgeChunkEmbedding.objects.filter(chunk=foreign_chunk).exists())


# ---------------------------------------------------------------------------
# The closed failure table
# ---------------------------------------------------------------------------

def _category_values(cls):
    return {
        value for name, value in vars(cls).items()
        if not name.startswith("_") and isinstance(value, str)
    }


class FailureTableTests(WorldMixin, TestCase):
    def test_every_known_category_is_classified_exactly_once(self):
        known = (
            _category_values(FailureCategory)
            | _category_values(ErrorCategory)
            | _category_values(ExtraFailureCategory)
        )
        self.assertEqual(RUN_FATAL_CATEGORIES & CHUNK_LOCAL_CATEGORIES, set())
        self.assertEqual(RUN_FATAL_CATEGORIES | CHUNK_LOCAL_CATEGORIES, known)

    def test_an_unclassified_category_fails_closed(self):
        self.assertTrue(is_run_fatal("a_category_added_later"))
        for category in CHUNK_LOCAL_CATEGORIES:
            self.assertFalse(is_run_fatal(category))

    def test_a_run_fatal_failure_stops_after_exactly_one_call(self):
        self.build_world()
        transport = fake_transport(
            error=lambda index: EmbeddingProviderExecutionError(
                ErrorCategory.PROVIDER_UNREACHABLE
            )
        )
        run = self.execute(transport)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(run.stopped_by, ErrorCategory.PROVIDER_UNREACHABLE)
        self.assertEqual(run.not_attempted_after_stop, 3)
        self.assertFalse(run.complete)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_a_dimension_mismatch_is_run_fatal(self):
        self.build_world()
        transport = fake_transport(values=(0.1, 0.2, 0.3))
        run = self.execute(transport)
        self.assertEqual(run.stopped_by, FailureCategory.VECTOR_DIMENSION_MISMATCH)
        self.assertEqual(len(transport.calls), 1)

    def test_work_stored_before_a_stop_is_kept_and_reported(self):
        self.build_world()
        transport = fake_transport(
            error=lambda index: (
                EmbeddingProviderExecutionError(ErrorCategory.MODEL_NOT_FOUND)
                if index == 2 else None
            )
        )
        run = self.execute(transport)
        self.assertEqual(run.outcome_counts(), {
            ExecutionStatus.STORED: 2, ErrorCategory.MODEL_NOT_FOUND: 1,
        })
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 2)
        self.assertEqual(run.after.counts()[CoverageState.CURRENT], 2)

    def test_a_chunk_local_failure_continues_with_the_next_chunk(self):
        self.build_world()
        transport = fake_transport(
            values_for=lambda index: (float("nan"), 0.2, 0.3, 0.4) if index == 0 else VALUES
        )
        run = self.execute(transport)
        self.assertEqual(len(transport.calls), 4)
        self.assertEqual(run.outcome_counts(), {
            ExecutionStatus.STORED: 3, FailureCategory.VECTOR_NON_FINITE: 1,
        })
        self.assertEqual(run.stopped_by, "")
        self.assertFalse(run.complete)

    def test_an_in_flight_edit_is_discarded_never_retried(self):
        self.build_world()
        first = min(chunk.pk for chunk in self.chunks)

        def edit_first(index):
            if index == 0:
                KnowledgeDocumentChunk.objects.filter(pk=first).update(content="raced")

        transport = fake_transport(hook=edit_first)
        run = self.execute(transport)
        self.assertEqual(len(transport.calls), 4)
        outcomes = {outcome.chunk_id: outcome.outcome for outcome in run.outcomes}
        self.assertEqual(outcomes[first], FailureCategory.STALE_CHUNK_AFTER_PROVIDER_CALL)
        self.assertFalse(KnowledgeChunkEmbedding.objects.filter(chunk_id=first).exists())

    def test_a_chunk_deleted_after_the_report_is_chunk_local(self):
        self.build_world()
        doomed = max(chunk.pk for chunk in self.chunks)

        def delete_last(index):
            if index == 0:
                KnowledgeDocumentChunk.objects.filter(pk=doomed).delete()

        transport = fake_transport(hook=delete_last)
        run = self.execute(transport)
        outcomes = {outcome.chunk_id: outcome.outcome for outcome in run.outcomes}
        self.assertEqual(outcomes[doomed], FailureCategory.CHUNK_MISSING)
        self.assertEqual(run.stopped_by, "")

    def test_unexpected_exceptions_are_not_swallowed(self):
        self.build_world()
        transport = fake_transport(error=lambda index: RuntimeError("a defect"))
        with self.assertRaises(RuntimeError):
            self.execute(transport)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

class CommandTests(WorldMixin, TestCase):
    def setUp(self):
        self.build_world()

    def run_command(self, *args, transport=None):
        out = StringIO()
        err = StringIO()
        argv = ["--scope", self.scope.slug, "--embedding-config", str(self.config.pk), *args]
        with mock.patch(TRANSPORT_PATH, return_value=transport or fake_transport()):
            call_command("knowledge_embedding_index", *argv, stdout=out, stderr=err)
        return out.getvalue()

    def test_scope_and_configuration_are_required(self):
        for argv in (
            ["--embedding-config", "1"],
            ["--scope", "index-app"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(CommandError):
                    call_command("knowledge_embedding_index", *argv, stdout=StringIO(), stderr=StringIO())

    def test_target_errors_become_command_errors(self):
        with self.assertRaisesMessage(CommandError, TargetErrorCategory.SCOPE_NOT_FOUND):
            call_command(
                "knowledge_embedding_index", "--scope", "nope",
                "--embedding-config", str(self.config.pk), stdout=StringIO(),
            )

    def test_report_mode_says_nothing_was_written(self):
        output = self.run_command()
        self.assertIn("Report only", output)
        self.assertIn("missing", output)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_limit_requires_execute_and_must_be_positive(self):
        with self.assertRaisesMessage(CommandError, "--limit only applies"):
            self.run_command("--limit", "2")
        with self.assertRaises(CommandError):
            self.run_command("--execute", "--limit", "0")

    def test_a_complete_execute_exits_cleanly(self):
        output = self.run_command("--execute")
        self.assertIn("Coverage after", output)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 4)

    def test_a_partial_execute_exits_non_zero(self):
        with self.assertRaisesMessage(CommandError, "--limit left 2"):
            self.run_command("--execute", "--limit", "2")

    def test_a_refused_execute_exits_non_zero(self):
        ProviderGrant.objects.all().delete()
        with self.assertRaisesMessage(CommandError, "no_provider_grant"):
            self.run_command("--execute")

    def test_json_is_valid_in_both_modes(self):
        report = json.loads(self.run_command("--json"))
        self.assertEqual(report["counts"][CoverageState.MISSING], 4)
        run = json.loads(self.run_command("--execute", "--json"))
        self.assertTrue(run["complete"])
        self.assertEqual(run["outcomes"], {ExecutionStatus.STORED: 4})

    def test_no_content_or_k1_ever_reaches_the_output(self):
        self.make_chunk(self.collection_a, "Large", content="y" * 9000)
        outputs = [self.run_command(), self.run_command("--json")]
        failing = fake_transport(
            values_for=lambda index: (float("nan"), 0.2, 0.3, 0.4) if index == 0 else VALUES
        )
        for args in (("--execute",), ("--execute", "--json")):
            out = StringIO()
            argv = ["--scope", self.scope.slug, "--embedding-config", str(self.config.pk), *args]
            with mock.patch(TRANSPORT_PATH, return_value=failing):
                with self.assertRaises(CommandError) as caught:
                    call_command("knowledge_embedding_index", *argv, stdout=out, stderr=StringIO())
            outputs.extend([out.getvalue(), str(caught.exception)])
        stored_k1 = list(KnowledgeChunkEmbedding.objects.values_list("k1", flat=True))
        self.assertTrue(stored_k1)
        for output in outputs:
            self.assertNotIn(MARKER, output)
            self.assertNotIn("k1:", output)
            for value in stored_k1:
                self.assertNotIn(value, output)

    def test_json_keys_carry_no_content_fields(self):
        payload = json.loads(self.run_command("--execute", "--json"))
        forbidden = {"content", "section_title", "text", "k1", "values", "vector", "title"}

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    self.assertNotIn(key, forbidden)
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------

def _parsed(path):
    return ast.parse(path.read_text(encoding="utf-8"))


class BoundaryTests(TestCase):
    def test_the_scanned_files_exist(self):
        for path in (SERVICE_FILE, COMMAND_FILE):
            self.assertTrue(path.is_file(), path)

    def test_the_only_write_path_is_index_chunk_embedding_local(self):
        for path in (SERVICE_FILE, COMMAND_FILE):
            tree = _parsed(path)
            called = {
                getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
            }
            with self.subTest(path=path.name):
                for forbidden in (
                    "store_chunk_vector", "embed_text_via_ollama", "create",
                    "update_or_create", "get_or_create", "bulk_create", "bulk_update",
                    "update", "delete", "save", "post",
                ):
                    self.assertNotIn(forbidden, called)
        service_calls = {
            getattr(node.func, "id", None)
            for node in ast.walk(_parsed(SERVICE_FILE))
            if isinstance(node, ast.Call)
        }
        self.assertIn("index_chunk_embedding_local", service_calls)

    def test_no_network_hashing_signals_or_queues(self):
        for path in (SERVICE_FILE, COMMAND_FILE):
            tree = _parsed(path)
            imported = {
                alias.name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            } | {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            }
            with self.subTest(path=path.name):
                for forbidden in (
                    "requests", "hashlib", "struct", "threading", "concurrent",
                    "django.db.models.signals", "celery",
                ):
                    self.assertNotIn(forbidden, imported)

    def test_not_an_agent_surface(self):
        self.assertFalse(
            ToolDefinition.objects.filter(config__callable__icontains="embedding_coverage").exists()
        )
        self.assertFalse(
            ToolDefinition.objects.filter(name__icontains="embedding_index").exists()
        )
        from ai_hub.services.knowledge_tooling import KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES

        for callable_path in KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES.values():
            self.assertNotIn("embedding", callable_path)

        import ai_hub.services.agent_runtime as agent_runtime
        import ai_hub.services.game_action_dispatcher as game_action_dispatcher
        import ai_hub.tools.knowledge as knowledge_tools

        for module in (agent_runtime, game_action_dispatcher, knowledge_tools):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("embedding_coverage", source)
                self.assertNotIn("knowledge_embedding_index", source)

    def test_the_service_takes_no_agent(self):
        for function in (resolve_index_target, build_coverage_report, execute_embedding_index):
            with self.subTest(function=function.__name__):
                self.assertNotIn("agent", inspect.signature(function).parameters)

    def test_the_command_has_no_destructive_or_bypass_flags(self):
        for flag in ("--delete", "--purge", "--force", "--yes", "--all-scopes", "--external"):
            with self.subTest(flag=flag):
                with self.assertRaises(CommandError):
                    call_command(
                        "knowledge_embedding_index", "--scope", "x",
                        "--embedding-config", "1", flag,
                        stdout=StringIO(), stderr=StringIO(),
                    )

    def test_search_knowledge_is_still_lexical(self):
        source = inspect.getsource(__import__("ai_hub.tools.knowledge", fromlist=["x"]))
        for forbidden in ("semantic_retrieval", "hybrid_retrieval", "embedding_coverage"):
            self.assertNotIn(forbidden, source)
        self.assertTrue(hasattr(embedding_coverage, "execute_embedding_index"))
