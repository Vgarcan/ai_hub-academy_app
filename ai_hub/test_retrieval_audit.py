"""S-23: reference-first retrieval audit.

Three families of test carry this module.

**Leakage absence.** The corpus, the query and the provider all carry distinctive
markers, and after a real audited retrieval every value in every audit column is
scanned for them. A structural pass separately proves the models have no
`TextField`, `JSONField`, `BinaryField` or `FileField` at all - the shapes a
query, a snippet or a provider body would eventually arrive in.

**Truthful incompleteness.** A run with no outcome is the record of an
interrupted operation, so the tests assert that absence deliberately rather than
treating it as a gap: audit-start failure, unexpected exception, completion
rollback and mid-flight inspection all pin what must and must not exist.

**Injected evidence.** The audit must refuse to record a result that disagrees
with itself. Those properties are unreachable through the real path - the
computation is correct - so the hostile cases hand the audit layer a fabricated
`HybridComputation` directly.
"""

import ast
import inspect
from decimal import Decimal
from unittest import mock

from django.db import connection, models, transaction
from django.db.models import ProtectedError
from django.test import TestCase

from ai_hub.models import (
    AgentProfile,
    ApplicationScope,
    EmbeddingModelConfig,
    GameWorkspace,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    ProviderConfig,
    ProviderGrant,
    RetrievalHit,
    RetrievalOutcome,
    RetrievalRun,
    RetrievalRunCollection,
)
from ai_hub.services import hybrid_retrieval, knowledge_retrieval, retrieval_audit
from ai_hub.services.chunk_embedding_identity import chunk_embedding_fingerprint
from ai_hub.services.embedding_client import (
    EmbeddingProviderExecutionError,
    EmbeddingProviderResult,
    ErrorCategory,
)
from ai_hub.services.hybrid_retrieval import (
    FUSION_VERSION,
    HYBRID_BRANCH_DEPTH,
    RRF_K,
    BranchStatus,
    HybridComputation,
    HybridHitIdentity,
    HybridMode,
    HybridRetrievalError,
    HybridFailureCategory,
    rrf_contribution,
)
from ai_hub.services.retrieval_audit import (
    EVIDENCE_VERSION,
    AuditFailureCategory,
    AuditedRetrievalRefused,
    RetrievalAuditError,
    audited_hybrid_search_knowledge_local,
    read_retrieval_evidence,
)
from ai_hub.services.vector_store import store_chunk_vector

METRIC = EmbeddingModelConfig.DistanceMetric
NORMALIZATION = EmbeddingModelConfig.Normalization
LOCALITY = ProviderConfig.DeclaredLocality

TRANSPORT_PATH = "ai_hub.services.semantic_retrieval.resolve_embedding_transport"
RESOLVER = "resolve_effective_knowledge_scope"

#: Distinctive markers. None of these may ever reach an audit column.
QUERY_SECRET = "QUERY-SECRET-ALPHAWIDGET-7731"
KNOWLEDGE_SECRET = "KNOWLEDGE-SECRET-CORPUS-4402"
PROVIDER_SECRET = "PROVIDER-SECRET-BODY-8815"
FOREIGN_SECRET = "FOREIGN-SECRET-SCOPEB-9926"

QUERY = f"alpha widget {QUERY_SECRET}"

AUDIT_MODELS = (RetrievalRun, RetrievalRunCollection, RetrievalOutcome, RetrievalHit)


class AuditFixtureMixin:
    """The S-22 corpus, with markers in the query, the Knowledge and the provider."""

    _world_counter = 0

    def build_corpus(self, *, locality=LOCALITY.LOCAL, grant_a=True):
        self._world_counter += 1
        tag = self._world_counter

        chat_provider = ProviderConfig.objects.create(
            name=f"Chat P{tag}", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        self.model_config = ModelConfig.objects.create(
            provider=chat_provider, model_name="training",
            temperature_default=Decimal("0.10"),
        )

        self.scope_a = ApplicationScope.objects.create(
            name=f"App A{tag}", slug=f"app-a{tag}"
        )
        self.scope_b = ApplicationScope.objects.create(
            name=f"App B{tag}", slug=f"app-b{tag}"
        )

        self.embed_provider = ProviderConfig.objects.create(
            name=f"Embed P{tag}",
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://ollama.internal:11434",
            declared_locality=locality,
        )
        self.config = EmbeddingModelConfig.objects.create(
            name=f"embed{tag}", provider=self.embed_provider,
            model_name="ollama/nomic-embed-text", model_revision="v1",
            vector_dimension=4, distance_metric=METRIC.COSINE,
            normalization=NORMALIZATION.NONE, max_input_chars=8000,
        )
        if grant_a is not None:
            ProviderGrant.objects.create(
                application_scope=self.scope_a, provider=self.embed_provider,
                allow_embeddings=grant_a,
            )
        ProviderGrant.objects.create(
            application_scope=self.scope_b, provider=self.embed_provider,
            allow_embeddings=True,
        )

        self.coll_a1 = self._collection(self.scope_a, f"A One {tag}")
        self.coll_a2 = self._collection(self.scope_a, f"A Two {tag}")
        self.coll_a3 = self._collection(self.scope_a, f"A Three {tag}")
        self.coll_b1 = self._collection(self.scope_b, f"B One {tag}")

        self.a1 = self._chunk(
            self.coll_a1, f"Alpha Widget Guide {tag}", "Alpha",
            f"{KNOWLEDGE_SECRET} alpha widget alpha widget",
        )
        self.a2 = self._chunk(
            self.coll_a1, f"Release Notes {tag}", "General",
            f"{KNOWLEDGE_SECRET} alpha only here",
        )
        self.a3 = self._chunk(
            self.coll_a2, f"Widget Manual {tag}", "Widget",
            f"{KNOWLEDGE_SECRET} widget only here",
        )
        self.unassigned = self._chunk(
            self.coll_a3, f"Alpha Widget Unassigned {tag}", "Alpha Widget",
            f"{KNOWLEDGE_SECRET} alpha widget alpha widget alpha widget",
        )
        self.foreign = self._chunk(
            self.coll_b1, f"Alpha Widget Beta {tag}", "Alpha Widget",
            f"{FOREIGN_SECRET} alpha widget alpha widget alpha widget",
        )

        self.agent_a = self._agent(f"A Agent {tag}", self.scope_a)
        self.agent_a.knowledge_collections.add(self.coll_a1, self.coll_a2)
        self.agent_b = self._agent(f"B Agent {tag}", self.scope_b)
        self.agent_b.knowledge_collections.add(self.coll_b1)

        for name, values in {
            "a1": (0.6, 0.8, 0.0, 0.0),
            "a2": (1.0, 0.0, 0.0, 0.0),
            "a3": (0.8, 0.6, 0.0, 0.0),
            "unassigned": (0.5, 0.0, 0.0, 0.0),
            "foreign": (0.25, 0.0, 0.0, 0.0),
        }.items():
            self.index(getattr(self, name), values)
        return self

    def _collection(self, scope, name):
        return KnowledgeCollection.objects.create(
            name=name, description="", application_scope=scope
        )

    def _chunk(self, collection, title, section, content):
        document = KnowledgeDocument.objects.create(
            collection=collection, title=title, curated_text=content,
            status=KnowledgeDocument.Status.ACTIVE,
        )
        return KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1,
            section_title=section, content=content,
        )

    def _agent(self, name, scope):
        return AgentProfile.objects.create(
            name=name, role="r", model_config=self.model_config,
            application_scope=scope, knowledge_max_chars=6000,
        )

    def index(self, chunk, values):
        return store_chunk_vector(
            application_scope=chunk.document.collection.application_scope,
            chunk=chunk,
            embedding_model_config=self.config,
            vector=tuple(values),
        )

    def transport_returning(self, values=(1.0, 0.0, 0.0, 0.0), *, hook=None):
        calls = []

        def fake_transport(*, provider, contract, text):
            calls.append(text)
            if hook is not None:
                hook()
            return EmbeddingProviderResult(
                values=tuple(values), provider_type=provider.provider_type,
                # A marker in the provider's own diagnostic field.
                provider_model=PROVIDER_SECRET,
            )

        fake_transport.calls = calls
        return fake_transport

    def audited(self, transport=None, *, agent=None, query=QUERY, **kwargs):
        self.transport = transport or self.transport_returning()
        with mock.patch(TRANSPORT_PATH, return_value=self.transport):
            return audited_hybrid_search_knowledge_local(
                agent if agent is not None else self.agent_a,
                query=query,
                embedding_model_config=self.config,
                **kwargs,
            )

    def counts(self):
        return {model.__name__: model.objects.count() for model in AUDIT_MODELS}


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

class CompletedRunTests(AuditFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_a_completed_hybrid_retrieval_records_run_targets_outcome_and_hits(self):
        audited = self.audited(limit=3)
        result = audited.retrieval

        self.assertEqual(result.mode, HybridMode.HYBRID)
        self.assertEqual(RetrievalRun.objects.count(), 1)

        run = RetrievalRun.objects.get(pk=audited.retrieval_run_id)
        self.assertEqual(run.application_scope_id, self.scope_a.pk)
        self.assertEqual(run.agent_id_snapshot, self.agent_a.pk)
        self.assertIsNone(run.workspace_id_snapshot)
        self.assertEqual(run.embedding_model_config_id_snapshot, self.config.pk)
        self.assertEqual(run.provider_id_snapshot, self.embed_provider.pk)
        self.assertEqual(run.requested_limit, 3)
        self.assertEqual(run.evidence_version, EVIDENCE_VERSION)
        self.assertEqual(run.fusion_version, FUSION_VERSION)
        self.assertEqual(run.rrf_k, RRF_K)
        self.assertEqual(run.branch_depth, HYBRID_BRANCH_DEPTH)

        self.assertEqual(
            sorted(
                row.collection_id_snapshot
                for row in run.target_collections.all()
            ),
            sorted([self.coll_a1.pk, self.coll_a2.pk]),
        )

        outcome = RetrievalOutcome.objects.get(retrieval_run=run)
        self.assertEqual(outcome.outcome, RetrievalOutcome.Outcome.COMPLETED)
        self.assertEqual(outcome.failure_category, "")
        self.assertEqual(outcome.mode, HybridMode.HYBRID)
        self.assertFalse(outcome.degraded)
        self.assertEqual(outcome.lexical_status, BranchStatus.USED)
        self.assertEqual(outcome.semantic_status, BranchStatus.USED)
        self.assertEqual(outcome.returned_count, len(result.matches))
        self.assertTrue(outcome.e1.startswith("e1:sha256:"))
        self.assertEqual(outcome.semantic_metric, METRIC.COSINE)

        hits = list(
            RetrievalHit.objects.filter(retrieval_run=run).order_by("final_rank")
        )
        self.assertEqual(len(hits), len(result.matches))
        for hit, match in zip(hits, result.matches):
            self.assertEqual(hit.final_rank, match.rank)
            self.assertEqual(hit.chunk_id_snapshot, match.chunk_id)
            self.assertEqual(hit.document_id_snapshot, match.document_id)
            self.assertEqual(hit.collection_id_snapshot, match.collection_id)
            self.assertEqual(hit.lexical_rank, match.lexical_rank)
            self.assertEqual(hit.semantic_rank, match.semantic_rank)
            self.assertEqual(
                hit.k1,
                chunk_embedding_fingerprint(
                    KnowledgeDocumentChunk.objects.get(pk=match.chunk_id)
                ),
            )

    def test_only_final_returned_matches_become_hits(self):
        """Not the branch top-20, not the filtered candidates, not the corpus."""
        audited = self.audited(limit=1)
        run_id = audited.retrieval_run_id
        self.assertEqual(len(audited.retrieval.matches), 1)
        self.assertEqual(
            RetrievalHit.objects.filter(retrieval_run_id=run_id).count(), 1
        )

        outcome = RetrievalOutcome.objects.get(retrieval_run_id=run_id)
        # The branch counts prove more was considered than was persisted.
        self.assertGreater(outcome.lexical_candidates_scanned, 1)
        self.assertGreater(outcome.semantic_candidate_count, 1)

    def test_branch_counts_are_recorded_without_the_candidates(self):
        audited = self.audited(limit=3)
        outcome = RetrievalOutcome.objects.get(
            retrieval_run_id=audited.retrieval_run_id
        )
        self.assertEqual(outcome.lexical_candidates_scanned, 3)
        self.assertFalse(outcome.lexical_candidates_truncated)
        self.assertEqual(outcome.semantic_candidate_count, 3)
        self.assertEqual(outcome.semantic_scored_count, 3)

    def test_the_workspace_snapshot_is_recorded_when_one_is_supplied(self):
        workspace = GameWorkspace.objects.create(
            name="Own WS", application_scope=self.scope_a
        )
        audited = self.audited(workspace=workspace)
        run = RetrievalRun.objects.get(pk=audited.retrieval_run_id)
        self.assertEqual(run.workspace_id_snapshot, workspace.pk)

    def test_the_rrf_contribution_is_reconstructable_from_durable_facts(self):
        """`rrf_k` + branch ranks are enough. No raw score is persisted."""
        audited = self.audited(limit=3)
        run = RetrievalRun.objects.get(pk=audited.retrieval_run_id)
        hits = list(RetrievalHit.objects.filter(retrieval_run=run))

        recomputed = []
        for hit in hits:
            total = 0.0
            if hit.lexical_rank is not None:
                total += 1.0 / (run.rrf_k + hit.lexical_rank)
            if hit.semantic_rank is not None:
                total += 1.0 / (run.rrf_k + hit.semantic_rank)
            recomputed.append((total, hit.chunk_id_snapshot, hit.final_rank))

        recomputed.sort(key=lambda entry: (-entry[0], entry[1]))
        self.assertEqual(
            [entry[2] for entry in recomputed],
            list(range(1, len(hits) + 1)),
            "recomputed order must reproduce the stored final ranks",
        )
        for total, chunk_id, _rank in recomputed:
            match = next(
                m for m in audited.retrieval.matches if m.chunk_id == chunk_id
            )
            self.assertAlmostEqual(total, match.fusion_score, places=15)
        self.assertAlmostEqual(rrf_contribution(1), 1.0 / (RRF_K + 1), places=15)


class DegradedAndEmptyCompletionTests(AuditFixtureMixin, TestCase):
    def test_a_degraded_result_is_completed_not_refused(self):
        self.build_corpus(grant_a=None)
        audited = self.audited()

        outcome = RetrievalOutcome.objects.get(
            retrieval_run_id=audited.retrieval_run_id
        )
        self.assertEqual(outcome.outcome, RetrievalOutcome.Outcome.COMPLETED)
        self.assertEqual(outcome.failure_category, "")
        self.assertEqual(outcome.mode, HybridMode.LEXICAL_ONLY)
        self.assertTrue(outcome.degraded)
        self.assertEqual(outcome.lexical_status, BranchStatus.USED)
        self.assertEqual(outcome.semantic_status, BranchStatus.UNAVAILABLE)
        self.assertEqual(outcome.semantic_failure_kind, "policy")
        self.assertEqual(outcome.e1, "")
        self.assertEqual(outcome.semantic_candidate_count, 0)
        self.assertGreater(outcome.returned_count, 0)
        self.assertEqual(
            RetrievalHit.objects.filter(
                retrieval_run_id=audited.retrieval_run_id
            ).count(),
            outcome.returned_count,
        )

    def test_a_provider_failure_degrades_and_still_completes(self):
        self.build_corpus()

        def unreachable(*, provider, contract, text):
            raise EmbeddingProviderExecutionError(
                ErrorCategory.PROVIDER_UNREACHABLE, PROVIDER_SECRET
            )

        with mock.patch(TRANSPORT_PATH, return_value=unreachable):
            audited = audited_hybrid_search_knowledge_local(
                self.agent_a, query=QUERY, embedding_model_config=self.config
            )
        outcome = RetrievalOutcome.objects.get(
            retrieval_run_id=audited.retrieval_run_id
        )
        self.assertEqual(outcome.outcome, RetrievalOutcome.Outcome.COMPLETED)
        self.assertEqual(outcome.semantic_failure_kind, "provider")
        self.assertNotIn(PROVIDER_SECRET, outcome.semantic_failure_kind)

    def test_an_empty_result_is_completed_with_no_hits(self):
        self.build_corpus()
        audited = self.audited(query="zzzznonmatchingtoken")
        outcome = RetrievalOutcome.objects.get(
            retrieval_run_id=audited.retrieval_run_id
        )
        self.assertEqual(outcome.outcome, RetrievalOutcome.Outcome.COMPLETED)
        self.assertEqual(outcome.mode, HybridMode.SEMANTIC_ONLY)
        self.assertFalse(outcome.degraded)

    def test_both_branches_empty_is_completed_and_not_degraded(self):
        self.build_corpus()
        from ai_hub.models import KnowledgeChunkEmbedding

        KnowledgeChunkEmbedding.objects.all().delete()
        audited = self.audited(query="zzzznonmatchingtoken")
        outcome = RetrievalOutcome.objects.get(
            retrieval_run_id=audited.retrieval_run_id
        )
        self.assertEqual(outcome.outcome, RetrievalOutcome.Outcome.COMPLETED)
        self.assertEqual(outcome.mode, HybridMode.EMPTY)
        self.assertFalse(outcome.degraded)
        self.assertEqual(outcome.returned_count, 0)
        self.assertEqual(
            RetrievalHit.objects.filter(
                retrieval_run_id=audited.retrieval_run_id
            ).count(),
            0,
        )


# ---------------------------------------------------------------------------
# When a run must and must not exist
# ---------------------------------------------------------------------------

class RunCreationBoundaryTests(AuditFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_invalid_caller_input_creates_no_run(self):
        """A rejected call is not a retrieval operation.

        Load-bearing for privacy as much as for correctness: persisting a
        rejected free-text query would store exactly the thing this slice
        refuses to store.
        """
        for kwargs in (
            {"limit": -1},
            {"limit": 999},
            {"limit": 1.5},
            {"query": "   "},
            {"query": "\r\n"},
            {"query": None},
        ):
            with self.subTest(**kwargs):
                before = self.counts()
                with self.assertRaises(HybridRetrievalError):
                    self.audited(**kwargs)
                self.assertEqual(self.counts(), before)

    def test_validation_precedes_authorization_in_source(self):
        source = inspect.getsource(audited_hybrid_search_knowledge_local)
        tree = ast.parse(source.lstrip())
        lines = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                lines.setdefault(node.func.id, node.lineno)
        self.assertLess(lines["validate_hybrid_request"], lines[RESOLVER])
        self.assertLess(lines[RESOLVER], lines["_start_run"])
        self.assertLess(lines["_start_run"], lines["search_hybrid_with_scope"])

    def test_deny_all_creates_no_run_at_all(self):
        """No trusted namespace means no namespace to scope evidence to."""
        AgentProfile.objects.filter(pk=self.agent_a.pk).update(is_active=False)
        self.agent_a.refresh_from_db()

        transport = self.transport_returning()
        audited = self.audited(transport)

        self.assertIsNone(audited.retrieval_run_id)
        self.assertEqual(audited.retrieval.matches, ())
        self.assertEqual(audited.retrieval.mode, HybridMode.EMPTY)
        self.assertEqual(RetrievalRun.objects.count(), 0)
        self.assertEqual(transport.calls, [])

    def test_a_foreign_workspace_creates_no_run(self):
        workspace = GameWorkspace.objects.create(
            name="Foreign WS", application_scope=self.scope_b
        )
        audited = self.audited(workspace=workspace)
        self.assertIsNone(audited.retrieval_run_id)
        self.assertEqual(RetrievalRun.objects.count(), 0)

    def test_a_trusted_scope_with_no_collections_still_creates_a_run(self):
        """The decisive distinction from DENY_ALL.

        A real principal in a real namespace asked a real question and got
        nothing. That is a governed retrieval operation and it must leave a
        record; DENY_ALL is not, and must not.
        """
        self.agent_a.knowledge_collections.clear()
        audited = self.audited()

        self.assertIsNotNone(audited.retrieval_run_id)
        run = RetrievalRun.objects.get(pk=audited.retrieval_run_id)
        self.assertEqual(run.application_scope_id, self.scope_a.pk)
        self.assertEqual(run.agent_id_snapshot, self.agent_a.pk)
        self.assertEqual(run.target_collections.count(), 0)

        outcome = RetrievalOutcome.objects.get(retrieval_run=run)
        self.assertEqual(outcome.outcome, RetrievalOutcome.Outcome.COMPLETED)
        self.assertEqual(outcome.mode, HybridMode.EMPTY)
        self.assertEqual(RetrievalHit.objects.filter(retrieval_run=run).count(), 0)

    def test_an_inaccessible_collection_records_a_run_with_zero_targets(self):
        for label, requested in (
            ("unassigned same scope", self.coll_a3.pk),
            ("foreign scope", self.coll_b1.pk),
            ("nonexistent", 9_999_999),
        ):
            with self.subTest(label=label):
                audited = self.audited(collection_id=requested)
                run = RetrievalRun.objects.get(pk=audited.retrieval_run_id)
                self.assertEqual(run.target_collections.count(), 0)
                outcome = RetrievalOutcome.objects.get(retrieval_run=run)
                self.assertEqual(
                    outcome.outcome, RetrievalOutcome.Outcome.COMPLETED
                )
                self.assertEqual(outcome.returned_count, 0)

    def test_the_requested_unauthorized_collection_id_is_never_persisted(self):
        """ADR-N5 must survive into durable storage.

        Otherwise the audit becomes exactly the oracle S-15 and S-22 refuse to
        be: a queryable history of which collection ids somebody probed.
        """
        probed = self.coll_b1.pk
        self.audited(collection_id=probed)

        stored = set()
        for model in AUDIT_MODELS:
            for row in model.objects.all():
                for field in model._meta.get_fields():
                    if not getattr(field, "concrete", False):
                        continue
                    stored.add(getattr(row, field.attname, None))
        self.assertNotIn(probed, stored)
        self.assertEqual(RetrievalRunCollection.objects.count(), 0)

    def test_limit_zero_records_a_governed_operation_with_no_work(self):
        transport = self.transport_returning()
        audited = self.audited(transport, limit=0)

        run = RetrievalRun.objects.get(pk=audited.retrieval_run_id)
        self.assertEqual(run.requested_limit, 0)
        self.assertEqual(
            sorted(r.collection_id_snapshot for r in run.target_collections.all()),
            sorted([self.coll_a1.pk, self.coll_a2.pk]),
        )
        outcome = RetrievalOutcome.objects.get(retrieval_run=run)
        self.assertEqual(outcome.outcome, RetrievalOutcome.Outcome.COMPLETED)
        self.assertEqual(outcome.returned_count, 0)
        self.assertEqual(RetrievalHit.objects.filter(retrieval_run=run).count(), 0)
        self.assertEqual(transport.calls, [], "no branch work for limit=0")


# ---------------------------------------------------------------------------
# Transactions and interruption
# ---------------------------------------------------------------------------

class TransactionBoundaryTests(AuditFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_the_run_exists_before_the_provider_is_dispatched(self):
        """Load-bearing: if the process dies here, the attempt is still recorded."""
        observed = {}

        def inspect_at_dispatch():
            observed["runs"] = RetrievalRun.objects.count()
            observed["outcomes"] = RetrievalOutcome.objects.count()
            observed["targets"] = RetrievalRunCollection.objects.count()

        self.audited(self.transport_returning(hook=inspect_at_dispatch))

        self.assertEqual(observed["runs"], 1, "the run exists")
        self.assertEqual(observed["outcomes"], 0, "and has no outcome yet")
        self.assertEqual(observed["targets"], 2)

    def test_no_transaction_is_open_across_the_provider_call(self):
        """Compared against a baseline, because Django `TestCase` wraps tests.

        `connection.in_atomic_block` is unconditionally True inside a TestCase,
        so asserting it is False would fail on correct code. Savepoint depth is
        the honest runtime signal: an open audit transaction would add one.
        """
        baseline = len(connection.savepoint_ids)
        observed = {}

        def inspect_at_dispatch():
            observed["depth"] = len(connection.savepoint_ids)

        self.audited(self.transport_returning(hook=inspect_at_dispatch))
        self.assertEqual(observed["depth"], baseline)

    def test_the_retrieval_call_is_not_inside_a_transaction_in_source(self):
        """Structural, since the runtime check above can only be a proxy."""
        tree = ast.parse(
            inspect.getsource(audited_hybrid_search_knowledge_local).lstrip()
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.With, ast.AsyncWith)):
                inner = {
                    call.func.id
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                }
                self.assertNotIn("search_hybrid_with_scope", inner)

    def test_audit_start_is_atomic_and_blocks_retrieval_on_failure(self):
        transport = self.transport_returning()
        real_create = RetrievalRunCollection.objects.create

        def explode(*args, **kwargs):
            real_create(*args, **kwargs)
            raise RuntimeError("target row persistence failed")

        with mock.patch.object(
            RetrievalRunCollection.objects, "create", explode
        ):
            with mock.patch(TRANSPORT_PATH, return_value=transport):
                with self.assertRaises(RuntimeError):
                    audited_hybrid_search_knowledge_local(
                        self.agent_a, query=QUERY,
                        embedding_model_config=self.config,
                    )

        self.assertEqual(RetrievalRun.objects.count(), 0, "run rolled back")
        self.assertEqual(RetrievalRunCollection.objects.count(), 0)
        self.assertEqual(transport.calls, [], "retrieval never executed")

    def test_completion_is_atomic_and_leaves_no_partial_evidence(self):
        real_create = RetrievalHit.objects.create
        state = {"calls": 0}

        def explode(*args, **kwargs):
            state["calls"] += 1
            row = real_create(*args, **kwargs)
            if state["calls"] >= 2:
                raise RuntimeError("hit persistence failed")
            return row

        with mock.patch.object(RetrievalHit.objects, "create", explode):
            with self.assertRaises(RuntimeError):
                self.audited(limit=3)

        self.assertEqual(RetrievalRun.objects.count(), 1, "the start survives")
        run = RetrievalRun.objects.get()
        self.assertEqual(run.target_collections.count(), 2)
        self.assertFalse(
            RetrievalOutcome.objects.filter(retrieval_run=run).exists(),
            "no outcome: the run is truthfully incomplete",
        )
        self.assertEqual(
            RetrievalHit.objects.filter(retrieval_run=run).count(),
            0,
            "not an outcome plus half the hits",
        )

    def test_the_result_is_not_returned_when_completion_persistence_fails(self):
        """The governance invariant: never answer while failing to record it."""
        with mock.patch.object(
            RetrievalHit.objects, "create",
            side_effect=RuntimeError("hit persistence failed"),
        ):
            with self.assertRaises(RuntimeError):
                self.audited(limit=3)

    def test_an_unexpected_exception_leaves_an_incomplete_run(self):
        with mock.patch.object(
            retrieval_audit, "search_hybrid_with_scope",
            side_effect=RuntimeError("a real bug"),
        ):
            with self.assertRaises(RuntimeError):
                self.audited()

        self.assertEqual(RetrievalRun.objects.count(), 1)
        run = RetrievalRun.objects.get()
        self.assertEqual(run.target_collections.count(), 2)
        self.assertEqual(RetrievalOutcome.objects.filter(retrieval_run=run).count(), 0)
        self.assertEqual(RetrievalHit.objects.filter(retrieval_run=run).count(), 0)

    def test_the_audited_operation_catches_no_broad_exception(self):
        tree = ast.parse(
            inspect.getsource(audited_hybrid_search_knowledge_local).lstrip()
        )
        handlers = [
            node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)
        ]
        self.assertTrue(handlers)
        for handler in handlers:
            with self.subTest(lineno=handler.lineno):
                self.assertIsNotNone(handler.type, "bare `except:` is not allowed")
                caught = set()
                if isinstance(handler.type, ast.Tuple):
                    caught = {
                        element.id for element in handler.type.elts
                        if isinstance(element, ast.Name)
                    }
                elif isinstance(handler.type, ast.Name):
                    caught = {handler.type.id}
                self.assertNotIn("Exception", caught)
                self.assertNotIn("BaseException", caught)


class RefusalTests(AuditFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_a_bounded_refusal_leaves_durable_evidence(self):
        """Through the real S-22 path: lexical incomplete + semantic unavailable."""
        with mock.patch.object(knowledge_retrieval, "MAX_SEARCH_CANDIDATES", 1):
            with mock.patch.object(
                hybrid_retrieval, "search_semantic_with_scope",
                side_effect=EmbeddingProviderExecutionError(
                    ErrorCategory.PROVIDER_UNREACHABLE, PROVIDER_SECRET
                ),
            ):
                with self.assertRaises(AuditedRetrievalRefused) as raised:
                    self.audited()

        self.assertEqual(
            raised.exception.category,
            HybridFailureCategory.NO_COMPLETE_RETRIEVAL_BRANCH,
        )
        run = RetrievalRun.objects.get(pk=raised.exception.retrieval_run_id)
        outcome = RetrievalOutcome.objects.get(retrieval_run=run)
        self.assertEqual(outcome.outcome, RetrievalOutcome.Outcome.REFUSED)
        self.assertEqual(
            outcome.failure_category,
            HybridFailureCategory.NO_COMPLETE_RETRIEVAL_BRANCH,
        )
        self.assertEqual(outcome.returned_count, 0)
        self.assertEqual(RetrievalHit.objects.filter(retrieval_run=run).count(), 0)
        self.assertEqual(run.target_collections.count(), 2)

    def test_the_refusal_exposes_the_run_id_and_nothing_else(self):
        with mock.patch.object(knowledge_retrieval, "MAX_SEARCH_CANDIDATES", 1):
            with mock.patch.object(
                hybrid_retrieval, "search_semantic_with_scope",
                side_effect=EmbeddingProviderExecutionError(
                    ErrorCategory.PROVIDER_UNREACHABLE, PROVIDER_SECRET
                ),
            ):
                with self.assertRaises(AuditedRetrievalRefused) as raised:
                    self.audited()

        message = str(raised.exception)
        for secret in (QUERY_SECRET, KNOWLEDGE_SECRET, PROVIDER_SECRET):
            self.assertNotIn(secret, message)
        self.assertIsNotNone(raised.exception.retrieval_run_id)

    def test_no_raw_exception_text_is_ever_persisted(self):
        with mock.patch.object(knowledge_retrieval, "MAX_SEARCH_CANDIDATES", 1):
            with mock.patch.object(
                hybrid_retrieval, "search_semantic_with_scope",
                side_effect=HybridRetrievalError(
                    HybridFailureCategory.NO_COMPLETE_RETRIEVAL_BRANCH,
                    f"secret detail {KNOWLEDGE_SECRET}",
                ),
            ):
                with self.assertRaises(AuditedRetrievalRefused):
                    self.audited()

        outcome = RetrievalOutcome.objects.get()
        self.assertEqual(
            outcome.failure_category,
            HybridFailureCategory.NO_COMPLETE_RETRIEVAL_BRANCH,
        )
        self.assertNotIn(KNOWLEDGE_SECRET, outcome.failure_category)


# ---------------------------------------------------------------------------
# Integrity: the audit refuses evidence that disagrees with itself
# ---------------------------------------------------------------------------

class EvidenceIntegrityTests(AuditFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def _run_with(self, computation_factory):
        real = retrieval_audit.search_hybrid_with_scope

        def patched(scope, **kwargs):
            return computation_factory(real(scope, **kwargs))

        with mock.patch.object(
            retrieval_audit, "search_hybrid_with_scope", patched
        ):
            return self.audited(limit=3)

    def assertRefusedByIntegrity(self, factory):
        with self.assertRaises(RetrievalAuditError) as raised:
            self._run_with(factory)
        self.assertEqual(
            raised.exception.category,
            AuditFailureCategory.EVIDENCE_INTEGRITY_VIOLATION,
        )
        self.assertEqual(RetrievalRun.objects.count(), 1)
        self.assertEqual(RetrievalOutcome.objects.count(), 0, "no outcome")
        self.assertEqual(RetrievalHit.objects.count(), 0, "no hits")
        return raised.exception

    def test_a_hit_from_another_application_scope_is_refused(self):
        def corrupt(computation):
            first = computation.evidence.final_hit_identities[0]
            hostile = HybridHitIdentity(
                final_rank=first.final_rank,
                chunk_id=first.chunk_id,
                document_id=first.document_id,
                collection_id=first.collection_id,
                application_scope_id=self.scope_b.pk,
                k1=first.k1,
                lexical_rank=first.lexical_rank,
                semantic_rank=first.semantic_rank,
            )
            rest = computation.evidence.final_hit_identities[1:]
            return self._replace_identities(computation, (hostile,) + rest)

        self.assertRefusedByIntegrity(corrupt)

    def test_a_hit_outside_the_target_collections_is_refused(self):
        def corrupt(computation):
            first = computation.evidence.final_hit_identities[0]
            hostile = HybridHitIdentity(
                final_rank=first.final_rank,
                chunk_id=first.chunk_id,
                document_id=first.document_id,
                collection_id=self.coll_a3.pk,
                application_scope_id=first.application_scope_id,
                k1=first.k1,
                lexical_rank=first.lexical_rank,
                semantic_rank=first.semantic_rank,
            )
            rest = computation.evidence.final_hit_identities[1:]
            return self._replace_identities(computation, (hostile,) + rest)

        self.assertRefusedByIntegrity(corrupt)

    def test_evidence_that_disagrees_with_the_public_result_is_refused(self):
        def corrupt(computation):
            return self._replace_identities(
                computation, computation.evidence.final_hit_identities[:-1]
            )

        self.assertRefusedByIntegrity(corrupt)

    def test_non_dense_final_ranks_are_refused(self):
        def corrupt(computation):
            identities = list(computation.evidence.final_hit_identities)
            first = identities[0]
            identities[0] = HybridHitIdentity(
                final_rank=7, chunk_id=first.chunk_id,
                document_id=first.document_id, collection_id=first.collection_id,
                application_scope_id=first.application_scope_id, k1=first.k1,
                lexical_rank=first.lexical_rank, semantic_rank=first.semantic_rank,
            )
            return self._replace_identities(computation, tuple(identities))

        self.assertRefusedByIntegrity(corrupt)

    def test_a_hit_with_no_branch_rank_is_refused(self):
        def corrupt(computation):
            identities = list(computation.evidence.final_hit_identities)
            first = identities[0]
            identities[0] = HybridHitIdentity(
                final_rank=first.final_rank, chunk_id=first.chunk_id,
                document_id=first.document_id, collection_id=first.collection_id,
                application_scope_id=first.application_scope_id, k1=first.k1,
                lexical_rank=None, semantic_rank=None,
            )
            return self._replace_identities(computation, tuple(identities))

        self.assertRefusedByIntegrity(corrupt)

    def test_a_result_from_a_different_namespace_is_refused(self):
        def corrupt(computation):
            import dataclasses

            return HybridComputation(
                result=dataclasses.replace(
                    computation.result, application_scope_id=self.scope_b.pk
                ),
                evidence=computation.evidence,
            )

        self.assertRefusedByIntegrity(corrupt)

    def test_a_result_with_foreign_target_collections_is_refused(self):
        def corrupt(computation):
            import dataclasses

            return HybridComputation(
                result=dataclasses.replace(
                    computation.result, collection_ids=(self.coll_b1.pk,)
                ),
                evidence=computation.evidence,
            )

        self.assertRefusedByIntegrity(corrupt)

    def test_a_result_under_a_different_fusion_contract_is_refused(self):
        for field, value in (
            ("fusion_version", "rrf99"),
            ("rrf_k", 7),
            ("branch_depth", 3),
        ):
            with self.subTest(field=field):
                def corrupt(computation, field=field, value=value):
                    import dataclasses

                    return HybridComputation(
                        result=dataclasses.replace(
                            computation.result, **{field: value}
                        ),
                        evidence=computation.evidence,
                    )

                with self.assertRaises(RetrievalAuditError):
                    self._run_with(corrupt)
                RetrievalRun.objects.all().delete()

    def _replace_identities(self, computation, identities):
        import dataclasses

        return HybridComputation(
            result=computation.result,
            evidence=dataclasses.replace(
                computation.evidence, final_hit_identities=identities
            ),
        )


class IdentityRaceTests(AuditFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_the_audit_persists_the_captured_k1_and_never_requeries_it(self):
        """The race the capture exists to close.

        The chunk is edited AFTER the composition boundary validated it and
        BEFORE the audit is written. If persistence recomputed the fingerprint,
        the audit would describe a version of the corpus the caller never saw.
        """
        real = retrieval_audit.search_hybrid_with_scope
        captured = {}

        def edit_after_composition(scope, **kwargs):
            computation = real(scope, **kwargs)
            captured["k1"] = {
                identity.chunk_id: identity.k1
                for identity in computation.evidence.final_hit_identities
            }
            KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).update(
                content=f"{KNOWLEDGE_SECRET} rewritten after composition"
            )
            return computation

        with mock.patch.object(
            retrieval_audit, "search_hybrid_with_scope", edit_after_composition
        ):
            audited = self.audited(limit=3)

        hit = RetrievalHit.objects.get(
            retrieval_run_id=audited.retrieval_run_id,
            chunk_id_snapshot=self.a1.pk,
        )
        self.assertEqual(hit.k1, captured["k1"][self.a1.pk])
        current = chunk_embedding_fingerprint(
            KnowledgeDocumentChunk.objects.get(pk=self.a1.pk)
        )
        self.assertNotEqual(
            hit.k1, current, "the corpus moved on; the evidence did not"
        )

    def test_the_persistence_layer_never_computes_a_fingerprint(self):
        """Structural: the audit module must not be able to recompute `k1`."""
        tree = ast.parse(inspect.getsource(retrieval_audit))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported |= {alias.name for alias in node.names}
        for forbidden in (
            "chunk_embedding_fingerprint", "canonical_chunk_embedding_text",
            "sha256", "hashlib",
        ):
            self.assertNotIn(forbidden, names)
            self.assertNotIn(forbidden, imported)

    def test_deleting_a_chunk_cannot_erase_hit_evidence(self):
        audited = self.audited(limit=3)
        before = RetrievalHit.objects.filter(
            retrieval_run_id=audited.retrieval_run_id
        ).count()
        self.assertGreater(before, 0)

        KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).delete()
        KnowledgeDocument.objects.filter(pk=self.a2.document_id).delete()

        self.assertEqual(
            RetrievalHit.objects.filter(
                retrieval_run_id=audited.retrieval_run_id
            ).count(),
            before,
        )

    def test_deleting_operational_configuration_cannot_erase_a_run(self):
        audited = self.audited()
        run_id = audited.retrieval_run_id
        GameWorkspace.objects.all().delete()
        self.agent_a.knowledge_collections.clear()
        AgentProfile.objects.filter(pk=self.agent_b.pk).delete()

        self.assertTrue(RetrievalRun.objects.filter(pk=run_id).exists())


# ---------------------------------------------------------------------------
# The single-snapshot invariant, preserved through the audited boundary
# ---------------------------------------------------------------------------

class SingleScopeSnapshotTests(AuditFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_one_audited_retrieval_resolves_the_scope_exactly_once(self):
        calls = []
        real = retrieval_audit.resolve_effective_knowledge_scope

        def spy(agent, *, workspace=None):
            calls.append(getattr(agent, "pk", None))
            return real(agent, workspace=workspace)

        with mock.patch.object(retrieval_audit, RESOLVER, spy):
            self.audited()
        self.assertEqual(len(calls), 1)

    def test_no_other_layer_resolves_a_scope(self):
        from ai_hub.services import semantic_retrieval

        boom = mock.Mock(side_effect=AssertionError("must not resolve"))
        with mock.patch.object(hybrid_retrieval, RESOLVER, boom):
            with mock.patch.object(knowledge_retrieval, RESOLVER, boom):
                with mock.patch.object(semantic_retrieval, RESOLVER, boom):
                    audited = self.audited()
        self.assertEqual(boom.call_count, 0)
        self.assertEqual(audited.retrieval.mode, HybridMode.HYBRID)

    def test_the_audit_module_resolves_exactly_once_in_source(self):
        tree = ast.parse(
            inspect.getsource(audited_hybrid_search_knowledge_local).lstrip()
        )
        resolutions = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == RESOLVER
        ]
        self.assertEqual(len(resolutions), 1)

    def test_the_audited_api_takes_the_same_caller_facts_as_s22(self):
        parameters = set(
            inspect.signature(audited_hybrid_search_knowledge_local).parameters
        )
        self.assertEqual(
            parameters,
            {
                "agent", "query", "embedding_model_config",
                "workspace", "collection_id", "limit",
            },
        )
        for forbidden in (
            "application_scope", "scope", "collection_ids",
            "retrieval_run", "evidence_version", "k1",
        ):
            self.assertNotIn(forbidden, parameters)


# ---------------------------------------------------------------------------
# The read side
# ---------------------------------------------------------------------------

class EvidenceReadTests(AuditFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_the_reader_returns_exact_bounded_evidence(self):
        audited = self.audited(limit=3)
        evidence = read_retrieval_evidence(
            application_scope=self.scope_a,
            retrieval_run_id=audited.retrieval_run_id,
        )

        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.retrieval_run_id, audited.retrieval_run_id)
        self.assertEqual(evidence.application_scope_id, self.scope_a.pk)
        self.assertEqual(evidence.agent_id_snapshot, self.agent_a.pk)
        self.assertEqual(evidence.evidence_version, EVIDENCE_VERSION)
        self.assertEqual(evidence.fusion_version, FUSION_VERSION)
        self.assertEqual(evidence.rrf_k, RRF_K)
        self.assertEqual(
            evidence.collection_ids,
            tuple(sorted([self.coll_a1.pk, self.coll_a2.pk])),
        )
        self.assertEqual(
            evidence.outcome.outcome, RetrievalOutcome.Outcome.COMPLETED
        )
        self.assertEqual(
            [hit.chunk_id for hit in evidence.hits],
            [match.chunk_id for match in audited.retrieval.matches],
        )
        self.assertEqual(
            [hit.final_rank for hit in evidence.hits],
            list(range(1, len(audited.retrieval.matches) + 1)),
        )

    def test_a_cross_scope_read_is_indistinguishable_from_a_missing_run(self):
        audited = self.audited()
        cross = read_retrieval_evidence(
            application_scope=self.scope_b,
            retrieval_run_id=audited.retrieval_run_id,
        )
        missing = read_retrieval_evidence(
            application_scope=self.scope_b, retrieval_run_id=9_999_999,
        )
        self.assertIsNone(cross)
        self.assertIsNone(missing)
        self.assertEqual(cross, missing)

    def test_the_reader_scopes_in_the_lookup_not_afterwards(self):
        """Fetching globally then comparing would put the row in memory first."""
        source = inspect.getsource(read_retrieval_evidence)
        tree = ast.parse(source.lstrip())
        filters = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "filter"
        ]
        self.assertTrue(filters)
        keywords = {
            keyword.arg for call in filters for keyword in call.keywords
        }
        self.assertIn("application_scope_id", keywords)
        self.assertIn("pk", keywords)

    def test_an_incomplete_run_reads_back_with_no_outcome(self):
        with mock.patch.object(
            retrieval_audit, "search_hybrid_with_scope",
            side_effect=RuntimeError("a real bug"),
        ):
            with self.assertRaises(RuntimeError):
                self.audited()

        run = RetrievalRun.objects.get()
        evidence = read_retrieval_evidence(
            application_scope=self.scope_a, retrieval_run_id=run.pk
        )
        self.assertIsNotNone(evidence)
        self.assertIsNone(evidence.outcome, "truthfully incomplete")
        self.assertEqual(evidence.hits, ())

    def test_the_reader_takes_no_agent_and_no_effective_scope(self):
        """Knowledge retrieval and audit administration are separate concerns."""
        parameters = set(inspect.signature(read_retrieval_evidence).parameters)
        self.assertEqual(parameters, {"application_scope", "retrieval_run_id"})
        self.assertNotIn("agent", parameters)

        tree = ast.parse(inspect.getsource(read_retrieval_evidence).lstrip())
        called = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn(RESOLVER, called)
        self.assertNotIn("authorized_chunks", called)

    def test_the_reader_returns_none_for_a_missing_scope_or_id(self):
        self.assertIsNone(
            read_retrieval_evidence(application_scope=None, retrieval_run_id=1)
        )
        self.assertIsNone(
            read_retrieval_evidence(
                application_scope=self.scope_a, retrieval_run_id=None
            )
        )


# ---------------------------------------------------------------------------
# Structural absence: what the audit schema may never hold
# ---------------------------------------------------------------------------

class ModelShapeTests(TestCase):
    FORBIDDEN_FIELD_TYPES = (
        models.TextField,
        models.JSONField,
        models.BinaryField,
        models.FileField,
    )
    FORBIDDEN_NAME_FRAGMENTS = (
        "query", "content", "snippet", "metadata", "vector", "payload",
        "prompt", "response_body", "error_message", "hash", "fingerprint",
        "details", "context", "body", "text",
    )

    def test_no_audit_model_has_an_unbounded_or_opaque_field(self):
        """A hard guard against future audit bloat and leakage.

        Every one of these types is a place a query, a snippet or a provider
        body could arrive without anyone noticing the migration.
        """
        for model in AUDIT_MODELS:
            for field in model._meta.get_fields():
                if not getattr(field, "concrete", False):
                    continue
                with self.subTest(model=model.__name__, field=field.name):
                    self.assertNotIsInstance(field, self.FORBIDDEN_FIELD_TYPES)

    def test_no_audit_field_name_suggests_content(self):
        for model in AUDIT_MODELS:
            for field in model._meta.get_fields():
                if not getattr(field, "concrete", False):
                    continue
                for fragment in self.FORBIDDEN_NAME_FRAGMENTS:
                    with self.subTest(model=model.__name__, field=field.name):
                        self.assertNotIn(fragment, field.name.lower())

    def test_no_query_hash_or_fingerprint_column_exists(self):
        """Deliberately absent, not merely unused.

        A raw deterministic hash of a query is still a durable identifier for
        what somebody searched, and it is dictionary-attackable for predictable
        queries. No keyed privacy contract is approved, so the safe amount is
        none.
        """
        for model in AUDIT_MODELS:
            names = {
                field.name.lower() for field in model._meta.get_fields()
                if getattr(field, "concrete", False)
            }
            for forbidden in ("q1", "query_hash", "query_fingerprint", "qhash"):
                self.assertNotIn(forbidden, names)

    def test_the_hit_table_stores_no_score(self):
        names = {
            field.name for field in RetrievalHit._meta.get_fields()
            if getattr(field, "concrete", False)
        }
        for forbidden in (
            "fusion_score", "score", "lexical_score", "semantic_score",
            "metric_value", "similarity", "distance",
        ):
            self.assertNotIn(forbidden, names)
        self.assertEqual(
            names,
            {
                "id", "retrieval_run", "chunk_id_snapshot", "document_id_snapshot",
                "collection_id_snapshot", "k1", "final_rank",
                "lexical_rank", "semantic_rank",
            },
        )

    def test_mutable_operational_references_are_snapshots_not_foreign_keys(self):
        """Only `ApplicationScope` is a durable FK, deliberately."""
        foreign_keys = {
            field.name: field.related_model
            for field in RetrievalRun._meta.get_fields()
            if isinstance(field, models.ForeignKey)
        }
        self.assertEqual(list(foreign_keys), ["application_scope"])
        self.assertEqual(
            foreign_keys["application_scope"].__name__, "ApplicationScope"
        )
        for model in (RetrievalRunCollection, RetrievalHit):
            related = {
                field.related_model.__name__
                for field in model._meta.get_fields()
                if isinstance(field, models.ForeignKey)
            }
            self.assertEqual(related, {"RetrievalRun"})

    def test_the_application_scope_is_protected(self):
        field = RetrievalRun._meta.get_field("application_scope")
        self.assertIs(field.remote_field.on_delete, models.PROTECT)

    def test_a_run_can_have_at_most_one_outcome(self):
        field = RetrievalOutcome._meta.get_field("retrieval_run")
        self.assertIsInstance(field, models.OneToOneField)

    def test_the_hit_constraints_are_present(self):
        names = {
            constraint.name for constraint in RetrievalHit._meta.constraints
        }
        self.assertEqual(
            names,
            {
                "ai_hub_unique_retrieval_hit_rank",
                "ai_hub_unique_retrieval_hit_chunk",
                "ai_hub_retrieval_hit_rank_positive",
                "ai_hub_retrieval_hit_lexical_rank_positive",
                "ai_hub_retrieval_hit_semantic_rank_positive",
                "ai_hub_retrieval_hit_has_a_branch",
            },
        )
        self.assertEqual(
            {c.name for c in RetrievalRunCollection._meta.constraints},
            {"ai_hub_unique_retrieval_run_collection"},
        )

    def test_the_expected_indexes_exist_and_no_more(self):
        self.assertEqual(
            [index.name for index in RetrievalRun._meta.indexes],
            ["aihub_retrun_scope_time_idx"],
        )
        self.assertEqual(
            [index.name for index in RetrievalHit._meta.indexes],
            ["aihub_rethit_chunk_idx"],
        )

    def test_no_audit_model_is_registered_in_admin(self):
        """Append-only evidence must not gain an accidental edit/delete UI."""
        from django.contrib import admin

        for model in AUDIT_MODELS:
            with self.subTest(model=model.__name__):
                self.assertNotIn(model, admin.site._registry)

    def test_the_service_exposes_no_update_or_delete_verb(self):
        names = {
            name for name in dir(retrieval_audit) if not name.startswith("__")
        }
        for forbidden in (
            "update_retrieval_run", "update_outcome", "edit_hit",
            "delete_hit", "delete_run", "purge_retrieval_evidence",
            "cleanup_retrieval_runs", "retry_completion",
        ):
            self.assertNotIn(forbidden, names)

        tree = ast.parse(inspect.getsource(retrieval_audit))
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        for forbidden in ("delete", "update", "bulk_update", "save"):
            self.assertNotIn(forbidden, attributes)


class DatabaseLeakTests(AuditFixtureMixin, TestCase):
    def test_no_marker_reaches_any_audit_column(self):
        self.build_corpus()
        audited = self.audited(limit=3)
        self.assertTrue(audited.retrieval.matches)

        values = []
        for model in AUDIT_MODELS:
            for row in model.objects.all():
                for field in model._meta.get_fields():
                    if not getattr(field, "concrete", False):
                        continue
                    values.append(str(getattr(row, field.attname, "")))
        blob = "\n".join(values)

        for secret in (
            QUERY_SECRET, KNOWLEDGE_SECRET, PROVIDER_SECRET, FOREIGN_SECRET,
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, blob)
        for fragment in ("alpha", "widget", "Alpha Widget", "Release Notes"):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, blob)

    def test_the_provider_diagnostic_string_never_reaches_the_audit(self):
        self.build_corpus()
        transport = self.transport_returning()
        audited = self.audited(transport)
        outcome = RetrievalOutcome.objects.get(
            retrieval_run_id=audited.retrieval_run_id
        )
        for value in (
            outcome.semantic_failure_kind, outcome.semantic_metric, outcome.e1
        ):
            self.assertNotIn(PROVIDER_SECRET, value)


class AbsenceTests(TestCase):
    """Everything S-23 deliberately does not add.

    AST/identifier based: this module's docstrings name nearly every forbidden
    concept while explaining why it is absent, so a source-text scan would match
    its own prose.
    """

    def _identifiers(self, module):
        tree = ast.parse(inspect.getsource(module))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        names |= {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
                names |= {alias.name for alias in node.names}
        return names

    def test_the_audit_stores_no_query_derived_value(self):
        names = self._identifiers(retrieval_audit)
        for forbidden in (
            "q1", "query_hash", "query_fingerprint", "hashlib", "sha256",
            "hmac", "canonical_query_embedding_text",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_audit_reads_no_knowledge_content(self):
        names = self._identifiers(retrieval_audit)
        for forbidden in (
            "KnowledgeDocumentChunk", "KnowledgeDocument", "KnowledgeCollection",
            "content", "snippet", "curated_text", "section_title", "tags",
            "decode_vector", "vector_bytes",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_audit_adds_no_retention_or_cleanup_machinery(self):
        names = self._identifiers(retrieval_audit)
        for forbidden in (
            "purge", "cleanup", "ttl", "retention", "post_delete",
            "pre_delete", "receiver", "shared_task", "celery",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_audit_adds_no_threshold_reranker_or_ann(self):
        names = self._identifiers(retrieval_audit)
        for forbidden in (
            "threshold", "no_answer", "rerank", "reranker", "cross_encoder",
            "pgvector", "hnsw", "ivfflat", "chroma", "faiss", "numpy", "scipy",
            "litellm", "completion",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_audit_decides_no_authorization_of_its_own(self):
        names = self._identifiers(retrieval_audit)
        for forbidden in (
            "authorized_chunks", "authorized_collections", "resolve_embedding_access",
            "ProviderGrant", "ApplicationScope", "AgentProfile",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_tool_layer_is_untouched(self):
        from ai_hub.services import knowledge_tooling
        from ai_hub.tools import knowledge as knowledge_tools

        for module in (knowledge_tooling, knowledge_tools):
            with self.subTest(module=module.__name__):
                names = self._identifiers(module)
                for forbidden in (
                    "audited_hybrid_search_knowledge_local",
                    "retrieval_audit", "read_retrieval_evidence",
                    "hybrid_search_knowledge_local", "RetrievalRun",
                ):
                    self.assertNotIn(forbidden, names)


class MigrationTests(TestCase):
    def test_the_migration_leaf_is_the_retrieval_audit_foundation(self):
        from django.db.migrations.loader import MigrationLoader

        loader = MigrationLoader(None, ignore_no_migrations=True)
        ai_hub_migrations = sorted(
            name for app, name in loader.disk_migrations if app == "ai_hub"
        )
        self.assertEqual(ai_hub_migrations[-1], "0029_retrieval_audit_foundation")
        self.assertEqual(
            len([name for name in ai_hub_migrations if name.startswith("0029")]),
            1,
            "exactly one leaf; no alternate 0029",
        )

    def test_the_migration_is_schema_only_and_portable(self):
        """No data migration, and nothing PostgreSQL-only.

        A single schema migration also avoids the DDL -> RunPython -> DDL shape
        that CI #43 proved PostgreSQL rejects inside one atomic migration.
        """
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parent
            / "migrations"
            / "0029_retrieval_audit_foundation.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        for forbidden in (
            "RunPython", "RunSQL", "ArrayField", "JSONField", "HStoreField",
            "SearchVectorField",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_scope_cannot_be_deleted_while_evidence_references_it(self):
        scope = ApplicationScope.objects.create(name="Doomed", slug="doomed")
        RetrievalRun.objects.create(
            application_scope=scope,
            agent_id_snapshot=1,
            requested_limit=5,
            evidence_version=EVIDENCE_VERSION,
            fusion_version=FUSION_VERSION,
            rrf_k=RRF_K,
            branch_depth=HYBRID_BRANCH_DEPTH,
        )
        with self.assertRaises(ProtectedError):
            with transaction.atomic():
                scope.delete()
        self.assertTrue(ApplicationScope.objects.filter(pk=scope.pk).exists())
