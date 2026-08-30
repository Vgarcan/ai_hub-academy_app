"""S-22: governed hybrid fusion.

Two families of test carry this module.

**Adversarial branch spies.** As in S-21, "the forbidden chunk is absent from the
results" proves nothing - a design that ranked it and then dropped it passes that
assertion. So the unauthorized chunks here carry BOTH the best lexical content
and the best semantic vectors, and the assertions are on whether the lexical
scorer and the metric scorer were ever *called* with them.

**Injected branch output.** Several properties are unreachable end to end because
the branches are themselves correct - a stale semantic match, for instance, is
already dropped inside S-21. Those are tested by handing hybrid a deliberately
malformed or hostile branch result directly, which is also the only way to prove
hybrid does not treat its own children as an authorization boundary.
"""

import ast
import inspect
import json
from decimal import Decimal
from unittest import mock

from django.core.exceptions import ValidationError
from django.test import TestCase

from ai_hub.models import (
    AgentProfile,
    ApplicationScope,
    EmbeddingModelConfig,
    GameWorkspace,
    KnowledgeChunkEmbedding,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    ProviderConfig,
    ProviderGrant,
)
from ai_hub.services import (
    hybrid_retrieval,
    knowledge_retrieval,
    semantic_retrieval,
)
from ai_hub.services.chunk_embedding_identity import chunk_embedding_fingerprint
from ai_hub.services.embedding_client import (
    EmbeddingProviderExecutionError,
    EmbeddingProviderResult,
    ErrorCategory,
)
from ai_hub.services.embedding_contract import EmbeddingContractError
from ai_hub.services.hybrid_retrieval import (
    FUSION_VERSION,
    HYBRID_BRANCH_DEPTH,
    MAX_HYBRID_RESULTS,
    RRF_K,
    BranchStatus,
    DegradationReason,
    HybridFailureCategory,
    HybridMatch,
    HybridMode,
    HybridRetrievalError,
    SemanticFailureKind,
    classify_semantic_failure,
    compress_branch_ranks,
    fuse_ranked_branches,
    hybrid_search_knowledge_local,
    rrf_contribution,
)
from ai_hub.services.knowledge_authorization import resolve_effective_knowledge_scope
from ai_hub.services.knowledge_retrieval import (
    LexicalCandidate,
    rank_knowledge_chunks_with_scope,
    search_knowledge,
)
from ai_hub.services.semantic_retrieval import (
    METRIC_SCORERS,
    MetricSpec,
    RetrievalFailureCategory,
    SemanticMatch,
    SemanticRetrievalError,
    SemanticRetrievalResult,
    semantic_search_knowledge_local,
)
from ai_hub.services.vector_store import store_chunk_vector

METRIC = EmbeddingModelConfig.DistanceMetric
NORMALIZATION = EmbeddingModelConfig.Normalization
LOCALITY = ProviderConfig.DeclaredLocality

TRANSPORT_PATH = "ai_hub.services.semantic_retrieval.resolve_embedding_transport"
RESOLVER = "resolve_effective_knowledge_scope"

A_SECRET = "ALPHA-CONFIDENTIAL-9911"
B_SECRET = "BETA-CONFIDENTIAL-2277"
QUERY = "alpha widget"


class HybridFixtureMixin:
    """A two-application corpus that is adversarial in BOTH modalities.

    `unassigned` (same scope, no assignment) and `foreign` (another application)
    carry the strongest lexical text AND vectors that tie for the best cosine
    score. If either branch could reach them they would rank first, so their
    absence from the scorer call logs is a real property rather than an accident
    of them being uninteresting.
    """

    _world_counter = 0

    def build_corpus(
        self,
        *,
        metric=METRIC.COSINE,
        locality=LOCALITY.LOCAL,
        grant_a=True,
        max_input_chars=8000,
    ):
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
            vector_dimension=4, distance_metric=metric,
            normalization=NORMALIZATION.NONE, max_input_chars=max_input_chars,
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

        # Lexical strength ranks roughly: forbidden >> a1 > a4 > a3 > a2.
        self.a1 = self._chunk(
            self.coll_a1, f"Alpha Widget Guide {tag}", "Alpha",
            f"{A_SECRET} alpha widget alpha widget",
        )
        self.a2 = self._chunk(
            self.coll_a1, f"Release Notes {tag}", "General",
            f"{A_SECRET} alpha only here",
        )
        self.a3 = self._chunk(
            self.coll_a1, f"Widget Manual {tag}", "Widget",
            f"{A_SECRET} widget only here",
        )
        self.a4 = self._chunk(
            self.coll_a2, f"Shared Reference {tag}", "Misc",
            f"{A_SECRET} alpha widget",
        )
        self.unassigned = self._chunk(
            self.coll_a3, f"Alpha Widget Alpha Widget {tag}", "Alpha Widget",
            f"{A_SECRET} alpha widget alpha widget alpha widget",
        )
        self.foreign = self._chunk(
            self.coll_b1, f"Alpha Widget Alpha Widget Beta {tag}", "Alpha Widget",
            f"{B_SECRET} alpha widget alpha widget alpha widget",
        )

        self.agent_a = self._agent(f"A Agent {tag}", self.scope_a)
        self.agent_a.knowledge_collections.add(self.coll_a1, self.coll_a2)
        self.agent_b = self._agent(f"B Agent {tag}", self.scope_b)
        self.agent_b.knowledge_collections.add(self.coll_b1)

        # Semantic order against the query vector (1,0,0,0), by cosine:
        #   unassigned 1.0 / foreign 1.0   forbidden, would rank first
        #   a2 1.0, a3 0.8, a1 0.6, a4 0.0
        # Deliberately close to the REVERSE of the lexical order, so fusion has
        # something real to reconcile.
        raw = {
            "a1": (0.6, 0.8, 0.0, 0.0),
            "a2": (1.0, 0.0, 0.0, 0.0),
            "a3": (0.8, 0.6, 0.0, 0.0),
            "a4": (0.0, 1.0, 0.0, 0.0),
            "unassigned": (0.5, 0.0, 0.0, 0.0),
            "foreign": (0.25, 0.0, 0.0, 0.0),
        }
        for name, values in raw.items():
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

    # -- driving a hybrid search -------------------------------------------

    def transport_returning(self, values=(1.0, 0.0, 0.0, 0.0), *, hook=None):
        calls = []

        def fake_transport(*, provider, contract, text):
            calls.append({"provider": provider, "contract": contract, "text": text})
            if hook is not None:
                hook()
            return EmbeddingProviderResult(
                values=tuple(values), provider_type=provider.provider_type,
                provider_model="nomic-embed-text",
            )

        fake_transport.calls = calls
        return fake_transport

    def search(self, transport=None, *, agent=None, query=QUERY, **kwargs):
        self.transport = transport or self.transport_returning()
        with mock.patch(TRANSPORT_PATH, return_value=self.transport):
            return hybrid_search_knowledge_local(
                agent if agent is not None else self.agent_a,
                query=query,
                embedding_model_config=self.config,
                **kwargs,
            )

    # -- spies --------------------------------------------------------------

    def lexical_spy(self):
        """Record every chunk the LEXICAL scorer is invoked with."""
        seen = []
        real = knowledge_retrieval._score_chunk

        def spy(chunk, words, *, content=None):
            seen.append(chunk.pk)
            return real(chunk, words, content=content)

        return mock.patch.object(knowledge_retrieval, "_score_chunk", spy), seen

    def semantic_spy(self):
        """Record every chunk the SEMANTIC metric is invoked for."""
        seen = []
        wrapped = {}
        for metric, spec in METRIC_SCORERS.items():
            def make(inner):
                def spy(query, candidate):
                    seen.append(tuple(candidate))
                    return inner(query, candidate)
                return spy
            wrapped[metric] = MetricSpec(
                score=make(spec.score), higher_is_better=spec.higher_is_better
            )
        return mock.patch.dict(
            semantic_retrieval.METRIC_SCORERS, wrapped, clear=True
        ), seen

    def stored_vector(self, chunk):
        record = KnowledgeChunkEmbedding.objects.get(chunk_id=chunk.pk)
        from ai_hub.services.vector_store import decode_vector
        return decode_vector(
            record.vector_bytes, expected_dimension=record.vector_dimension
        )

    def chunk_ids(self, result):
        return [match.chunk_id for match in result.matches]


# ---------------------------------------------------------------------------
# THE primary invariant: one authorization answer per hybrid operation
# ---------------------------------------------------------------------------

class SingleScopeSnapshotTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def _resolver_spy(self, module):
        calls = []
        real = getattr(module, RESOLVER)

        def spy(agent, *, workspace=None):
            calls.append((getattr(agent, "pk", None), workspace))
            return real(agent, workspace=workspace)

        return mock.patch.object(module, RESOLVER, spy), calls

    def test_one_hybrid_search_resolves_the_scope_exactly_once(self):
        """The load-bearing test for S-22.

        Two branches resolving separately could run one user-visible search
        under two different authorization answers, and the result would be a
        blend with nothing to say which half was which.
        """
        hybrid_patch, hybrid_calls = self._resolver_spy(hybrid_retrieval)
        with hybrid_patch:
            self.search()
        self.assertEqual(len(hybrid_calls), 1)

    def test_neither_branch_resolves_a_scope_of_its_own(self):
        """Proved by making a second resolution fatal in EITHER branch module."""
        boom = mock.Mock(side_effect=AssertionError("branch must not resolve"))
        with mock.patch.object(knowledge_retrieval, RESOLVER, boom):
            with mock.patch.object(semantic_retrieval, RESOLVER, boom):
                result = self.search()
        self.assertEqual(boom.call_count, 0)
        self.assertEqual(result.mode, HybridMode.HYBRID)

    def test_the_branch_entry_points_never_reference_the_resolver(self):
        """Structural, so a future edit cannot quietly reintroduce a split.

        AST-based: these modules' docstrings discuss the resolver at length
        while explaining why the branches must not call it, so a source-text
        scan would match their own prose.
        """
        from ai_hub.services.semantic_retrieval import search_semantic_with_scope

        for target in (rank_knowledge_chunks_with_scope, search_semantic_with_scope):
            with self.subTest(target=target.__name__):
                tree = ast.parse(inspect.getsource(target).lstrip())
                called = {
                    node.func.id
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                }
                self.assertNotIn(RESOLVER, called)

    def test_the_hybrid_entry_point_resolves_exactly_once_in_source(self):
        tree = ast.parse(inspect.getsource(hybrid_search_knowledge_local).lstrip())
        resolutions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == RESOLVER
        ]
        self.assertEqual(len(resolutions), 1)

    def test_the_caller_cannot_supply_any_authorization_or_fusion_fact(self):
        parameters = set(
            inspect.signature(hybrid_search_knowledge_local).parameters
        )
        self.assertEqual(
            parameters,
            {
                "agent", "query", "embedding_model_config",
                "workspace", "collection_id", "limit",
            },
        )
        for forbidden in (
            "application_scope", "collection_ids", "scope", "query_vector",
            "e1", "distance_metric", "lexical_weight", "semantic_weight",
            "rrf_k", "branch_depth", "fusion_version", "weights", "threshold",
        ):
            self.assertNotIn(forbidden, parameters)

    def test_an_assignment_revoked_mid_operation_uses_the_frozen_scope(self):
        """S-21's snapshot semantics, preserved at the hybrid layer.

        The revocation takes effect on the NEXT operation, which the second half
        proves - so the window is bounded by one search, not open-ended.
        """
        def revoke():
            self.agent_a.knowledge_collections.remove(self.coll_a1)

        in_flight = self.search(self.transport_returning(hook=revoke))
        self.assertIn(self.a1.pk, self.chunk_ids(in_flight))

        after = self.search()
        self.assertEqual(self.chunk_ids(after), [self.a4.pk])
        self.assertEqual(after.collection_ids, (self.coll_a2.pk,))


# ---------------------------------------------------------------------------
# Authorization: neither branch may reach forbidden Knowledge
# ---------------------------------------------------------------------------

class AuthorizationAdversarialTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_forbidden_chunks_are_never_scored_by_either_branch(self):
        lexical_patch, lexical_seen = self.lexical_spy()
        semantic_patch, semantic_seen = self.semantic_spy()
        with lexical_patch, semantic_patch:
            result = self.search()

        for forbidden in (self.unassigned, self.foreign):
            with self.subTest(chunk=forbidden.pk):
                self.assertNotIn(forbidden.pk, lexical_seen)
                self.assertNotIn(
                    self.stored_vector(forbidden), semantic_seen
                )
                self.assertNotIn(forbidden.pk, self.chunk_ids(result))

        self.assertEqual(
            sorted(set(lexical_seen)),
            sorted(chunk.pk for chunk in (self.a1, self.a2, self.a3, self.a4)),
        )

    def test_the_forbidden_chunks_really_would_have_won(self):
        """Proves the previous test is not vacuous, in BOTH modalities."""
        words = knowledge_retrieval._query_words(QUERY)
        forbidden_score = knowledge_retrieval._score_chunk(self.unassigned, words)
        best_allowed = max(
            knowledge_retrieval._score_chunk(chunk, words)
            for chunk in (self.a1, self.a2, self.a3, self.a4)
        )
        self.assertGreater(forbidden_score, best_allowed)

        from ai_hub.services.semantic_retrieval import cosine_similarity

        query_vector = (1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(
            cosine_similarity(query_vector, self.stored_vector(self.unassigned)),
            1.0, places=6,
        )
        self.assertAlmostEqual(
            cosine_similarity(query_vector, self.stored_vector(self.foreign)),
            1.0, places=6,
        )

    def test_a_foreign_scope_agent_sees_only_its_own_corpus(self):
        result = self.search(agent=self.agent_b)
        self.assertEqual(self.chunk_ids(result), [self.foreign.pk])
        self.assertEqual(result.application_scope_id, self.scope_b.pk)

    def test_an_empty_scope_runs_no_branch_at_all(self):
        AgentProfile.objects.filter(pk=self.agent_a.pk).update(is_active=False)
        self.agent_a.refresh_from_db()

        lexical_patch, lexical_seen = self.lexical_spy()
        transport = self.transport_returning()
        with lexical_patch:
            result = self.search(transport)

        self.assertEqual(result.matches, ())
        self.assertEqual(result.mode, HybridMode.EMPTY)
        self.assertEqual(result.lexical_status, BranchStatus.NOT_RUN)
        self.assertEqual(result.semantic_status, BranchStatus.NOT_RUN)
        self.assertFalse(result.degraded)
        self.assertEqual(lexical_seen, [])
        self.assertEqual(transport.calls, [], "no provider call for an empty scope")

    def test_an_archived_document_never_enters_fusion(self):
        KnowledgeDocument.objects.filter(pk=self.a1.document_id).update(
            status=KnowledgeDocument.Status.ARCHIVED
        )
        lexical_patch, lexical_seen = self.lexical_spy()
        with lexical_patch:
            result = self.search()
        self.assertNotIn(self.a1.pk, lexical_seen)
        self.assertNotIn(self.a1.pk, self.chunk_ids(result))


# ---------------------------------------------------------------------------
# Collection narrowing
# ---------------------------------------------------------------------------

class CollectionNarrowingTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_without_narrowing_both_branches_target_every_assignment(self):
        result = self.search()
        self.assertEqual(
            result.collection_ids, tuple(sorted([self.coll_a1.pk, self.coll_a2.pk]))
        )
        self.assertIn(self.a4.pk, self.chunk_ids(result))

    def test_narrowing_excludes_the_other_collection_from_both_branches(self):
        lexical_patch, lexical_seen = self.lexical_spy()
        semantic_patch, semantic_seen = self.semantic_spy()
        with lexical_patch, semantic_patch:
            result = self.search(collection_id=self.coll_a1.pk)

        self.assertEqual(result.collection_ids, (self.coll_a1.pk,))
        self.assertNotIn(self.a4.pk, lexical_seen)
        self.assertNotIn(self.stored_vector(self.a4), semantic_seen)
        self.assertNotIn(self.a4.pk, self.chunk_ids(result))
        for match in result.matches:
            self.assertIsNot(match.chunk_id, self.a4.pk)

    def test_narrowing_can_never_widen(self):
        """ADR-N5: every inaccessible collection yields the same empty shape."""
        for label, requested in (
            ("unassigned same scope", self.coll_a3.pk),
            ("foreign scope", self.coll_b1.pk),
            ("nonexistent", 9_999_999),
        ):
            with self.subTest(label=label):
                lexical_patch, lexical_seen = self.lexical_spy()
                transport = self.transport_returning()
                with lexical_patch:
                    result = self.search(transport, collection_id=requested)
                self.assertEqual(result.matches, ())
                self.assertEqual(result.mode, HybridMode.EMPTY)
                self.assertEqual(result.lexical_status, BranchStatus.NOT_RUN)
                self.assertEqual(result.semantic_status, BranchStatus.NOT_RUN)
                self.assertEqual(lexical_seen, [])
                self.assertEqual(transport.calls, [])


# ---------------------------------------------------------------------------
# Pure fusion mathematics
# ---------------------------------------------------------------------------

class FusionMathTests(TestCase):
    def test_the_fusion_contract_is_versioned_and_fixed(self):
        self.assertEqual(FUSION_VERSION, "rrf1")
        self.assertEqual(RRF_K, 60)
        self.assertEqual(HYBRID_BRANCH_DEPTH, 20)
        self.assertEqual(MAX_HYBRID_RESULTS, 20)

    def test_the_contribution_formula_is_exactly_one_over_k_plus_rank(self):
        for rank in (1, 2, 7, 20):
            with self.subTest(rank=rank):
                self.assertAlmostEqual(
                    rrf_contribution(rank), 1.0 / (60 + rank), places=15
                )

    def test_the_worked_example(self):
        """Lexical A,B and semantic B,C fuse to B, A, C."""
        fused = fuse_ranked_branches({1: 1, 2: 2}, {2: 1, 3: 2})
        order = [chunk_id for _score, chunk_id, _l, _s in fused]
        self.assertEqual(order, [2, 1, 3])

        scores = {chunk_id: score for score, chunk_id, _l, _s in fused}
        self.assertAlmostEqual(scores[1], 1 / 61, places=15)
        self.assertAlmostEqual(scores[2], 1 / 62 + 1 / 61, places=15)
        self.assertAlmostEqual(scores[3], 1 / 62, places=15)

    def test_a_chunk_in_both_branches_receives_both_contributions(self):
        fused = fuse_ranked_branches({5: 3}, {5: 4})
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0][0], 1 / 63 + 1 / 64, places=15)
        self.assertEqual(fused[0][2], 3)
        self.assertEqual(fused[0][3], 4)

    def test_ties_break_on_chunk_id_ascending(self):
        fused = fuse_ranked_branches({9: 1, 3: 1}, {})
        self.assertEqual([chunk_id for _s, chunk_id, _l, _sr in fused], [3, 9])

    def test_fusion_cannot_see_a_score_because_it_is_not_given_one(self):
        """The signature IS the guarantee.

        `fuse_ranked_branches` accepts rank maps and nothing else, so a lexical
        score magnitude or a cosine value is not merely ignored - it is
        unreachable from the fusion code.
        """
        parameters = list(inspect.signature(fuse_ranked_branches).parameters)
        self.assertEqual(parameters, ["lexical_ranks", "semantic_ranks"])

        tree = ast.parse(inspect.getsource(fuse_ranked_branches).lstrip())
        names = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for forbidden in (
            "score", "metric_value", "higher_is_better", "metric",
            "cosine", "weight", "threshold",
        ):
            self.assertNotIn(forbidden, names)

    def test_compression_renumbers_survivors_from_one(self):
        self.assertEqual(compress_branch_ranks([7, 4, 9]), {7: 1, 4: 2, 9: 3})

    def test_compression_keeps_only_a_first_occurrence_per_modality(self):
        """One chunk contributes at most once per branch, even if repeated."""
        self.assertEqual(compress_branch_ranks([7, 4, 7, 9]), {7: 1, 4: 2, 9: 3})

    def test_a_duplicate_within_one_branch_is_not_double_counted(self):
        ranks = compress_branch_ranks([7, 7])
        fused = fuse_ranked_branches(ranks, {})
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0][0], 1 / 61, places=15)


class FusionIgnoresRawScoresTests(HybridFixtureMixin, TestCase):
    """End-to-end proof that only ORDER crosses the branch boundary."""

    def setUp(self):
        self.build_corpus()

    def _run_with_branches(self, lexical_scores, semantic_values):
        """Same ranks, radically different raw magnitudes."""
        scope = resolve_effective_knowledge_scope(self.agent_a)
        ids = [self.a1.pk, self.a2.pk, self.a3.pk]
        lexical = knowledge_retrieval.LexicalRanking(
            query=QUERY, words=("alpha",), rows=(),
            candidates=tuple(
                LexicalCandidate(
                    rank=position, chunk_id=chunk_id,
                    document_id=0, collection_id=0,
                    score=lexical_scores[position - 1],
                    k1=chunk_embedding_fingerprint(
                        KnowledgeDocumentChunk.objects.get(pk=chunk_id)
                    ),
                )
                for position, chunk_id in enumerate(ids, start=1)
            ),
            candidates_scanned=3, candidates_truncated=False,
        )
        semantic = SemanticRetrievalResult(
            matches=tuple(
                SemanticMatch(
                    rank=position, chunk_id=chunk_id, document_id=0,
                    collection_id=0, application_scope_id=0,
                    k1=chunk_embedding_fingerprint(
                        KnowledgeDocumentChunk.objects.get(pk=chunk_id)
                    ),
                    e1="e1", metric=METRIC.COSINE,
                    metric_value=semantic_values[position - 1],
                    higher_is_better=True,
                )
                for position, chunk_id in enumerate(reversed(ids), start=1)
            ),
            application_scope_id=scope.application_scope_id, agent_id=None,
            workspace_id=None, collection_ids=(), embedding_model_config_id=None,
            e1="e1", metric=METRIC.COSINE, higher_is_better=True,
            candidate_count=3, scored_count=3, provider_invoked=True,
        )
        with mock.patch.object(
            hybrid_retrieval, "rank_knowledge_chunks_with_scope",
            return_value=lexical,
        ):
            with mock.patch.object(
                hybrid_retrieval, "search_semantic_with_scope",
                return_value=semantic,
            ):
                return hybrid_search_knowledge_local(
                    self.agent_a, query=QUERY,
                    embedding_model_config=self.config, limit=5,
                )

    def test_identical_ranks_with_wild_scores_fuse_identically(self):
        modest = self._run_with_branches([3, 2, 1], [0.9, 0.5, 0.1])
        absurd = self._run_with_branches(
            [10_000_000, 2, 1], [0.0001, 0.00005, 0.00001]
        )
        self.assertEqual(self.chunk_ids(modest), self.chunk_ids(absurd))
        for left, right in zip(modest.matches, absurd.matches):
            self.assertEqual(left.fusion_score, right.fusion_score)
            self.assertEqual(left.lexical_rank, right.lexical_rank)
            self.assertEqual(left.semantic_rank, right.semantic_rank)

    def test_a_negative_or_distance_style_metric_value_changes_nothing(self):
        """Euclidean reports LOWER-is-better; fusion must not notice."""
        higher = self._run_with_branches([3, 2, 1], [0.9, 0.5, 0.1])
        lower = self._run_with_branches([3, 2, 1], [0.1, 0.5, 0.9])
        self.assertEqual(self.chunk_ids(higher), self.chunk_ids(lower))


# ---------------------------------------------------------------------------
# Branch depth
# ---------------------------------------------------------------------------

class BranchDepthTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_both_branches_are_asked_for_the_fusion_depth_not_the_limit(self):
        """Otherwise a chunk ranked #6 in BOTH branches could never surface."""
        lexical_calls = []
        semantic_calls = []
        real_lexical = hybrid_retrieval.rank_knowledge_chunks_with_scope
        real_semantic = hybrid_retrieval.search_semantic_with_scope

        def lexical_spy(scope, **kwargs):
            lexical_calls.append(kwargs)
            return real_lexical(scope, **kwargs)

        def semantic_spy(scope, **kwargs):
            semantic_calls.append(kwargs)
            return real_semantic(scope, **kwargs)

        with mock.patch.object(
            hybrid_retrieval, "rank_knowledge_chunks_with_scope", lexical_spy
        ):
            with mock.patch.object(
                hybrid_retrieval, "search_semantic_with_scope", semantic_spy
            ):
                self.search(limit=2)

        self.assertEqual(lexical_calls[0]["limit"], HYBRID_BRANCH_DEPTH)
        self.assertEqual(semantic_calls[0]["limit"], HYBRID_BRANCH_DEPTH)
        self.assertTrue(lexical_calls[0]["capture_identity"])

    def test_the_final_limit_is_applied_after_fusion(self):
        full = self.search(limit=20)
        capped = self.search(limit=2)
        self.assertEqual(len(capped.matches), 2)
        self.assertEqual(self.chunk_ids(capped), self.chunk_ids(full)[:2])


# ---------------------------------------------------------------------------
# Final revalidation, rank compression and defence in depth
# ---------------------------------------------------------------------------

class FinalCompositionTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()
        self.scope = resolve_effective_knowledge_scope(self.agent_a)

    def _lexical(self, entries, *, truncated=False):
        return knowledge_retrieval.LexicalRanking(
            query=QUERY, words=("alpha",), rows=(),
            candidates=tuple(
                LexicalCandidate(
                    rank=position, chunk_id=chunk_id, document_id=0,
                    collection_id=0, score=100 - position, k1=k1,
                )
                for position, (chunk_id, k1) in enumerate(entries, start=1)
            ),
            candidates_scanned=len(entries), candidates_truncated=truncated,
        )

    def _semantic(self, entries):
        return SemanticRetrievalResult(
            matches=tuple(
                SemanticMatch(
                    rank=position, chunk_id=chunk_id, document_id=0,
                    collection_id=0, application_scope_id=0, k1=k1,
                    e1="e1", metric=METRIC.COSINE, metric_value=1.0,
                    higher_is_better=True,
                )
                for position, (chunk_id, k1) in enumerate(entries, start=1)
            ),
            application_scope_id=self.scope.application_scope_id, agent_id=None,
            workspace_id=None, collection_ids=(), embedding_model_config_id=None,
            e1="e1:injected", metric=METRIC.COSINE, higher_is_better=True,
            candidate_count=len(entries), scored_count=len(entries),
            provider_invoked=True,
        )

    def _compose(self, lexical, semantic, **kwargs):
        with mock.patch.object(
            hybrid_retrieval, "rank_knowledge_chunks_with_scope",
            return_value=lexical,
        ):
            with mock.patch.object(
                hybrid_retrieval, "search_semantic_with_scope",
                return_value=semantic,
            ):
                return hybrid_search_knowledge_local(
                    self.agent_a, query=QUERY,
                    embedding_model_config=self.config, **kwargs,
                )

    def k1(self, chunk):
        return chunk_embedding_fingerprint(
            KnowledgeDocumentChunk.objects.get(pk=chunk.pk)
        )

    def test_a_hostile_branch_result_is_dropped_by_final_revalidation(self):
        """Hybrid does not trust its children as an authorization boundary."""
        hostile = self._lexical([
            (self.foreign.pk, self.k1(self.foreign)),
            (self.unassigned.pk, self.k1(self.unassigned)),
            (self.a1.pk, self.k1(self.a1)),
        ])
        result = self._compose(hostile, self._semantic([]))

        self.assertEqual(self.chunk_ids(result), [self.a1.pk])
        self.assertNotIn(self.foreign.pk, self.chunk_ids(result))
        self.assertNotIn(self.unassigned.pk, self.chunk_ids(result))

    def test_a_removed_candidate_consumes_no_rank(self):
        """Load-bearing: a rank slot is itself influence.

        If the dropped candidates kept slots 1 and 2, `a1` would fuse at rank 3
        and score 1/63 - lower, purely because of chunks the caller may not see.
        """
        hostile = self._lexical([
            (self.foreign.pk, self.k1(self.foreign)),
            (self.unassigned.pk, self.k1(self.unassigned)),
            (self.a1.pk, self.k1(self.a1)),
        ])
        result = self._compose(hostile, self._semantic([]))
        match = result.matches[0]

        self.assertEqual(match.lexical_rank, 1, "compressed to rank 1")
        self.assertAlmostEqual(match.fusion_score, 1 / 61, places=15)
        self.assertNotAlmostEqual(match.fusion_score, 1 / 63, places=15)

    def test_a_hostile_candidate_does_not_change_any_authorized_score(self):
        clean = self._lexical([(self.a1.pk, self.k1(self.a1))])
        polluted = self._lexical([
            (self.foreign.pk, self.k1(self.foreign)),
            (self.a1.pk, self.k1(self.a1)),
        ])
        clean_result = self._compose(clean, self._semantic([]))
        polluted_result = self._compose(polluted, self._semantic([]))

        self.assertEqual(
            clean_result.matches[0].fusion_score,
            polluted_result.matches[0].fusion_score,
        )

    def test_a_stale_lexical_k1_drops_only_the_lexical_contribution(self):
        """Modality-specific staleness.

        Injected rather than driven end to end, because S-21 already discards a
        stale semantic match internally - the only way to observe hybrid's own
        per-modality behaviour is to hand it one stale branch and one current
        one for the SAME chunk.
        """
        current = self.k1(self.a1)
        stale = "k1:sha256:" + ("0" * 64)
        result = self._compose(
            self._lexical([(self.a1.pk, stale)]),
            self._semantic([(self.a1.pk, current)]),
        )
        self.assertEqual(self.chunk_ids(result), [self.a1.pk])
        match = result.matches[0]
        self.assertIsNone(match.lexical_rank)
        self.assertEqual(match.semantic_rank, 1)
        self.assertAlmostEqual(match.fusion_score, 1 / 61, places=15)
        self.assertEqual(result.mode, HybridMode.SEMANTIC_ONLY)

    def test_a_stale_semantic_k1_drops_only_the_semantic_contribution(self):
        current = self.k1(self.a1)
        stale = "k1:sha256:" + ("0" * 64)
        result = self._compose(
            self._lexical([(self.a1.pk, current)]),
            self._semantic([(self.a1.pk, stale)]),
        )
        match = result.matches[0]
        self.assertEqual(match.lexical_rank, 1)
        self.assertIsNone(match.semantic_rank)
        self.assertEqual(result.mode, HybridMode.LEXICAL_ONLY)
        self.assertFalse(result.degraded, "staleness is not lost capability")

    def test_a_deleted_chunk_contributes_nothing(self):
        entries = [(self.a1.pk, self.k1(self.a1)), (self.a2.pk, self.k1(self.a2))]
        lexical = self._lexical(entries)
        KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).delete()
        result = self._compose(lexical, self._semantic([]))
        self.assertEqual(self.chunk_ids(result), [self.a2.pk])
        self.assertEqual(result.matches[0].lexical_rank, 1)

    def test_a_branch_result_outside_the_REQUESTED_collection_is_dropped(self):
        """Isolates the request-narrowing half of the composition filter.

        `a4` lives in a collection this Agent IS authorized for, so
        `authorized_chunks(scope)` admits it. Only the target-collection
        predicate can drop it - which makes this the test that keeps that
        predicate honest, since the two guards otherwise overlap.
        """
        hostile = self._lexical([
            (self.a4.pk, self.k1(self.a4)),
            (self.a1.pk, self.k1(self.a1)),
        ])
        result = self._compose(
            hostile, self._semantic([]), collection_id=self.coll_a1.pk
        )
        self.assertEqual(self.chunk_ids(result), [self.a1.pk])
        self.assertEqual(result.matches[0].lexical_rank, 1, "and it takes rank 1")

    def test_an_inactive_collection_is_dropped_at_composition(self):
        entries = [(self.a1.pk, self.k1(self.a1)), (self.a4.pk, self.k1(self.a4))]
        lexical = self._lexical(entries)
        KnowledgeCollection.objects.filter(pk=self.coll_a1.pk).update(is_active=False)
        result = self._compose(lexical, self._semantic([]))
        self.assertEqual(self.chunk_ids(result), [self.a4.pk])


class LexicalStalenessDuringSemanticTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_a_chunk_edited_during_the_provider_call_loses_its_lexical_rank(self):
        """The real timing window, driven end to end.

        The lexical branch scores and fingerprints `a1`, then the chunk is
        edited while the semantic provider is answering. The lexical rank was
        earned by text nobody is reading any more, so it must not survive.
        """
        def edit_mid_flight():
            KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).update(
                content=f"{A_SECRET} entirely rewritten in flight"
            )

        before = self.search()
        self.assertIn(self.a1.pk, self.chunk_ids(before))

        result = self.search(self.transport_returning(hook=edit_mid_flight))
        self.assertNotIn(self.a1.pk, self.chunk_ids(result))
        for match in result.matches:
            self.assertNotEqual(match.chunk_id, self.a1.pk)

    def test_a_collection_deactivated_mid_flight_is_dropped_at_composition(self):
        """Isolates the `authorized_chunks` half of the composition filter.

        The collection is deactivated AFTER the scope snapshot, so it is still
        in the frozen target set - the request-narrowing predicate cannot drop
        it, and only re-querying the authorized chunk set can.
        """
        def deactivate_mid_flight():
            KnowledgeCollection.objects.filter(pk=self.coll_a1.pk).update(
                is_active=False
            )

        result = self.search(self.transport_returning(hook=deactivate_mid_flight))
        self.assertEqual(
            result.collection_ids,
            tuple(sorted([self.coll_a1.pk, self.coll_a2.pk])),
            "the frozen target set still names it",
        )
        self.assertEqual(self.chunk_ids(result), [self.a4.pk])

    def test_a_document_archived_mid_flight_is_dropped_at_composition(self):
        def archive_mid_flight():
            KnowledgeDocument.objects.filter(pk=self.a1.document_id).update(
                status=KnowledgeDocument.Status.ARCHIVED
            )

        result = self.search(self.transport_returning(hook=archive_mid_flight))
        self.assertNotIn(self.a1.pk, self.chunk_ids(result))
        self.assertTrue(result.matches, "the rest of the search still stands")

    def test_the_lexical_k1_is_captured_from_the_row_that_was_scored(self):
        """Not re-read afterwards, which would fingerprint the CURRENT text."""
        scope = resolve_effective_knowledge_scope(self.agent_a)
        ranking = rank_knowledge_chunks_with_scope(
            scope, query=QUERY,
            collection_ids=scope.collection_ids,
            limit=HYBRID_BRANCH_DEPTH, capture_identity=True,
        )
        expected = {
            chunk.pk: chunk_embedding_fingerprint(chunk)
            for chunk in KnowledgeDocumentChunk.objects.filter(
                pk__in=[c.chunk_id for c in ranking.candidates]
            )
        }
        self.assertTrue(ranking.candidates)
        for candidate in ranking.candidates:
            with self.subTest(chunk=candidate.chunk_id):
                self.assertEqual(candidate.k1, expected[candidate.chunk_id])
                self.assertTrue(candidate.k1.startswith("k1:sha256:"))

    def test_the_public_lexical_path_captures_no_identity(self):
        """It has no composition boundary to defend, so it pays nothing."""
        scope = resolve_effective_knowledge_scope(self.agent_a)
        ranking = rank_knowledge_chunks_with_scope(
            scope, query=QUERY, collection_ids=scope.collection_ids, limit=5
        )
        self.assertTrue(ranking.candidates)
        for candidate in ranking.candidates:
            self.assertEqual(candidate.k1, "")


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

class SemanticDegradationTests(HybridFixtureMixin, TestCase):
    def assertLexicalOnly(self, result, kind):
        self.assertTrue(result.matches, "complete lexical results must survive")
        self.assertEqual(result.mode, HybridMode.LEXICAL_ONLY)
        self.assertTrue(result.degraded)
        self.assertIn(
            DegradationReason.SEMANTIC_UNAVAILABLE, result.degradation_reasons
        )
        self.assertEqual(result.semantic_status, BranchStatus.UNAVAILABLE)
        self.assertEqual(result.lexical_status, BranchStatus.USED)
        self.assertEqual(result.semantic_failure_kind, kind)
        self.assertEqual(result.e1, "")
        self.assertEqual(result.semantic_metric, "")

    def test_a_missing_provider_grant_degrades_to_lexical(self):
        self.build_corpus(grant_a=None)
        self.assertLexicalOnly(self.search(), SemanticFailureKind.POLICY)

    def test_query_egress_denied_degrades_to_lexical(self):
        self.build_corpus(locality=LOCALITY.EXTERNAL)
        ApplicationScope.objects.filter(pk=self.scope_a.pk).update(
            allow_external_embedding_corpus_egress=True,
            allow_external_embedding_query_egress=False,
        )
        self.assertLexicalOnly(self.search(), SemanticFailureKind.POLICY)

    def test_an_external_provider_degrades_to_lexical(self):
        self.build_corpus(locality=LOCALITY.EXTERNAL)
        ApplicationScope.objects.filter(pk=self.scope_a.pk).update(
            allow_external_embedding_corpus_egress=True,
            allow_external_embedding_query_egress=True,
        )
        self.assertLexicalOnly(self.search(), SemanticFailureKind.CAPABILITY)

    def test_an_unsupported_transport_degrades_to_lexical(self):
        """Deliberately does NOT patch the transport resolver.

        Patching it would bypass the very capability lookup under test - the
        search would succeed and the assertion would pass for the wrong reason.
        """
        self.build_corpus()
        ProviderConfig.objects.filter(pk=self.embed_provider.pk).update(
            provider_type=ProviderConfig.ProviderType.TRAINING
        )
        self.config.refresh_from_db()
        result = hybrid_search_knowledge_local(
            self.agent_a, query=QUERY, embedding_model_config=self.config
        )
        self.assertLexicalOnly(result, SemanticFailureKind.CAPABILITY)

    def test_an_unreachable_provider_degrades_to_lexical(self):
        self.build_corpus()

        def unreachable(*, provider, contract, text):
            raise EmbeddingProviderExecutionError(
                ErrorCategory.PROVIDER_UNREACHABLE, "The provider is unreachable."
            )

        with mock.patch(TRANSPORT_PATH, return_value=unreachable):
            result = hybrid_search_knowledge_local(
                self.agent_a, query=QUERY, embedding_model_config=self.config
            )
        self.assertLexicalOnly(result, SemanticFailureKind.PROVIDER)

    def test_a_model_not_found_degrades_to_lexical(self):
        self.build_corpus()

        def missing(*, provider, contract, text):
            raise EmbeddingProviderExecutionError(
                ErrorCategory.MODEL_NOT_FOUND, "No such model."
            )

        with mock.patch(TRANSPORT_PATH, return_value=missing):
            result = hybrid_search_knowledge_local(
                self.agent_a, query=QUERY, embedding_model_config=self.config
            )
        self.assertLexicalOnly(result, SemanticFailureKind.PROVIDER)

    def test_an_invalid_provider_response_degrades_to_lexical(self):
        self.build_corpus()
        self.assertLexicalOnly(
            self.search(self.transport_returning((1.0, 0.0))),
            SemanticFailureKind.PROVIDER,
        )

    def test_an_inactive_embedding_config_degrades_to_lexical(self):
        """Deactivated AFTER indexing: an inactive config cannot store vectors,
        so the corpus has to exist before the configuration is switched off."""
        self.build_corpus()
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(
            is_active=False
        )
        self.config.refresh_from_db()
        self.assertLexicalOnly(self.search(), SemanticFailureKind.CONFIGURATION)

    def test_a_semantic_candidate_overflow_degrades_to_lexical(self):
        self.build_corpus()
        with mock.patch.object(
            semantic_retrieval, "MAX_REFERENCE_SEMANTIC_CANDIDATES", 1
        ):
            result = self.search()
        self.assertLexicalOnly(result, SemanticFailureKind.REFERENCE_LIMIT)

    def test_a_query_too_large_for_the_contract_degrades_to_lexical(self):
        self.build_corpus(max_input_chars=4)
        self.assertLexicalOnly(
            self.search(query="alpha widget"),
            SemanticFailureKind.QUERY_INCOMPATIBLE,
        )

    def test_a_semantic_failure_never_selects_another_provider(self):
        self.build_corpus(grant_a=None)
        other = ProviderConfig.objects.create(
            name="Fallback P", provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://other.internal:11434",
            declared_locality=LOCALITY.LOCAL,
        )
        ProviderGrant.objects.create(
            application_scope=self.scope_a, provider=other, allow_embeddings=True
        )
        result = self.search()
        self.assertEqual(result.semantic_status, BranchStatus.UNAVAILABLE)
        self.assertEqual(result.e1, "")

    def test_no_raw_exception_text_reaches_the_result(self):
        self.build_corpus(grant_a=None)
        result = self.search()
        blob = json.dumps(result.__dict__, default=str)
        for secret in (A_SECRET, B_SECRET, QUERY, "not authorized", "reason_code"):
            self.assertNotIn(secret, blob)

    def test_only_bounded_semantic_failures_are_absorbed(self):
        """A programming error is not a degraded search."""
        self.build_corpus()
        with mock.patch.object(
            hybrid_retrieval, "search_semantic_with_scope",
            side_effect=RuntimeError("a real bug"),
        ):
            with self.assertRaises(RuntimeError):
                self.search()

    def test_the_semantic_handler_is_narrow(self):
        tree = ast.parse(inspect.getsource(hybrid_search_knowledge_local).lstrip())
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

    def test_the_absorbed_error_tuple_is_explicit(self):
        self.assertEqual(
            hybrid_retrieval.SEMANTIC_DEGRADATION_ERRORS,
            (
                SemanticRetrievalError,
                EmbeddingContractError,
                EmbeddingProviderExecutionError,
            ),
        )

    def test_every_failure_kind_is_a_bounded_lowercase_token(self):
        kinds = [
            value for name, value in vars(SemanticFailureKind).items()
            if not name.startswith("_")
        ]
        self.assertTrue(kinds)
        for kind in kinds:
            with self.subTest(kind=kind):
                self.assertIsInstance(kind, str)
                self.assertRegex(kind, r"^[a-z_]*$")

    def test_an_unmapped_category_is_reported_as_unknown_not_guessed(self):
        exc = SemanticRetrievalError("some_future_category", "x")
        self.assertEqual(classify_semantic_failure(exc), SemanticFailureKind.UNKNOWN)


class LexicalIncompleteTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def _truncated(self):
        return mock.patch.object(knowledge_retrieval, "MAX_SEARCH_CANDIDATES", 1)

    def test_an_incomplete_lexical_ranking_is_never_fused(self):
        with self._truncated():
            result = self.search()

        self.assertEqual(result.lexical_status, BranchStatus.INCOMPLETE)
        self.assertEqual(result.mode, HybridMode.SEMANTIC_ONLY)
        self.assertTrue(result.degraded)
        self.assertIn(
            DegradationReason.LEXICAL_INCOMPLETE, result.degradation_reasons
        )
        self.assertTrue(result.matches)
        for match in result.matches:
            self.assertIsNone(match.lexical_rank, "no partial lexical contribution")
            self.assertIsNotNone(match.semantic_rank)

    def test_neither_branch_trustworthy_refuses(self):
        with self._truncated():
            with mock.patch.object(
                hybrid_retrieval, "search_semantic_with_scope",
                side_effect=SemanticRetrievalError(
                    RetrievalFailureCategory.EMBEDDING_NOT_AUTHORIZED, "denied"
                ),
            ):
                with self.assertRaises(HybridRetrievalError) as raised:
                    self.search()
        self.assertEqual(
            raised.exception.category,
            HybridFailureCategory.NO_COMPLETE_RETRIEVAL_BRANCH,
        )

    def test_the_refusal_never_degrades_into_a_partial_lexical_answer(self):
        with self._truncated():
            with mock.patch.object(
                hybrid_retrieval, "search_semantic_with_scope",
                side_effect=EmbeddingContractError("inactive"),
            ):
                with self.assertRaises(HybridRetrievalError):
                    self.search()


class EmptyBranchMatrixTests(HybridFixtureMixin, TestCase):
    """An EMPTY branch is a real answer; only lost capability is degradation."""

    def setUp(self):
        self.build_corpus()

    def test_lexical_empty_with_semantic_matches(self):
        result = self.search(query="zzzznonmatchingtoken")
        self.assertEqual(result.lexical_status, BranchStatus.EMPTY)
        self.assertEqual(result.semantic_status, BranchStatus.USED)
        self.assertEqual(result.mode, HybridMode.SEMANTIC_ONLY)
        self.assertFalse(result.degraded)
        self.assertEqual(result.degradation_reasons, ())
        self.assertTrue(result.matches)

    def test_semantic_empty_with_lexical_matches(self):
        KnowledgeChunkEmbedding.objects.all().delete()
        result = self.search()
        self.assertEqual(result.lexical_status, BranchStatus.USED)
        self.assertEqual(result.semantic_status, BranchStatus.EMPTY)
        self.assertEqual(result.mode, HybridMode.LEXICAL_ONLY)
        self.assertFalse(result.degraded)
        self.assertTrue(result.matches)

    def test_both_complete_and_empty(self):
        KnowledgeChunkEmbedding.objects.all().delete()
        result = self.search(query="zzzznonmatchingtoken")
        self.assertEqual(result.mode, HybridMode.EMPTY)
        self.assertEqual(result.matches, ())
        self.assertFalse(result.degraded)
        self.assertEqual(result.lexical_status, BranchStatus.EMPTY)
        self.assertEqual(result.semantic_status, BranchStatus.EMPTY)

    def test_both_branches_contributing_is_hybrid_and_not_degraded(self):
        result = self.search()
        self.assertEqual(result.mode, HybridMode.HYBRID)
        self.assertFalse(result.degraded)
        self.assertEqual(result.lexical_status, BranchStatus.USED)
        self.assertEqual(result.semantic_status, BranchStatus.USED)


# ---------------------------------------------------------------------------
# limit, workspace, and the result contract
# ---------------------------------------------------------------------------

class LimitTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_limit_zero_spends_no_retrieval_work(self):
        lexical_patch, lexical_seen = self.lexical_spy()
        transport = self.transport_returning()
        with lexical_patch:
            result = self.search(transport, limit=0)

        self.assertEqual(result.matches, ())
        self.assertEqual(result.mode, HybridMode.EMPTY)
        self.assertEqual(lexical_seen, [], "no lexical ranking")
        self.assertEqual(transport.calls, [], "no provider call")
        self.assertEqual(result.lexical_status, BranchStatus.NOT_RUN)
        self.assertEqual(result.semantic_status, BranchStatus.NOT_RUN)

    def test_an_out_of_range_limit_is_refused_not_clamped(self):
        for bad in (-1, MAX_HYBRID_RESULTS + 1, 1000):
            with self.subTest(limit=bad):
                transport = self.transport_returning()
                with self.assertRaises(HybridRetrievalError) as raised:
                    self.search(transport, limit=bad)
                self.assertEqual(
                    raised.exception.category, HybridFailureCategory.INVALID_LIMIT
                )
                self.assertEqual(transport.calls, [])

    def test_a_non_integer_limit_is_refused(self):
        for bad in (1.5, "3", True, None):
            with self.subTest(limit=bad):
                with self.assertRaises(HybridRetrievalError) as raised:
                    self.search(limit=bad)
                self.assertEqual(
                    raised.exception.category, HybridFailureCategory.INVALID_LIMIT
                )

    def test_the_maximum_limit_is_accepted(self):
        result = self.search(limit=MAX_HYBRID_RESULTS)
        self.assertLessEqual(len(result.matches), MAX_HYBRID_RESULTS)

    def test_a_blank_query_is_refused_before_any_branch(self):
        for bad in ("", "   ", "\r\n", None, 123):
            with self.subTest(query=bad):
                lexical_patch, lexical_seen = self.lexical_spy()
                transport = self.transport_returning()
                with lexical_patch:
                    with self.assertRaises(HybridRetrievalError) as raised:
                        self.search(transport, query=bad)
                self.assertEqual(
                    raised.exception.category, HybridFailureCategory.QUERY_EMPTY
                )
                self.assertEqual(lexical_seen, [])
                self.assertEqual(transport.calls, [])

    def test_the_blank_predicate_reuses_the_s21_contract(self):
        source = inspect.getsource(hybrid_search_knowledge_local)
        called = {
            node.func.id
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("canonical_query_embedding_text", called)


class WorkspaceTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_a_coherent_workspace_neither_widens_nor_narrows(self):
        workspace = GameWorkspace.objects.create(
            name="Own WS", application_scope=self.scope_a
        )
        result = self.search(workspace=workspace)
        self.assertEqual(result.workspace_id, workspace.pk)
        self.assertEqual(result.mode, HybridMode.HYBRID)
        self.assertEqual(
            sorted(self.chunk_ids(result)),
            sorted(c.pk for c in (self.a1, self.a2, self.a3, self.a4))[
                : len(result.matches)
            ],
        )

    def test_a_foreign_workspace_runs_no_branch(self):
        workspace = GameWorkspace.objects.create(
            name="Foreign WS", application_scope=self.scope_b
        )
        lexical_patch, lexical_seen = self.lexical_spy()
        transport = self.transport_returning()
        with lexical_patch:
            result = self.search(transport, workspace=workspace)
        self.assertEqual(result.matches, ())
        self.assertEqual(lexical_seen, [])
        self.assertEqual(transport.calls, [])

    def test_an_inactive_workspace_fails_closed_the_same_way(self):
        workspace = GameWorkspace.objects.create(
            name="Inactive WS", application_scope=self.scope_a, is_active=False
        )
        lexical_patch, lexical_seen = self.lexical_spy()
        transport = self.transport_returning()
        with lexical_patch:
            result = self.search(transport, workspace=workspace)
        self.assertEqual(result.matches, ())
        self.assertEqual(lexical_seen, [])
        self.assertEqual(transport.calls, [])


class ResultContractTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_the_match_shape_is_identifiers_and_an_ordering_value(self):
        result = self.search()
        self.assertEqual(
            set(result.matches[0].__dict__),
            {
                "rank", "chunk_id", "document_id", "collection_id",
                "application_scope_id", "fusion_score",
                "lexical_rank", "semantic_rank",
            },
        )

    def test_the_result_shape_reports_the_fusion_contract(self):
        result = self.search()
        self.assertEqual(result.fusion_version, FUSION_VERSION)
        self.assertEqual(result.rrf_k, RRF_K)
        self.assertEqual(result.branch_depth, HYBRID_BRANCH_DEPTH)
        self.assertEqual(result.embedding_model_config_id, self.config.pk)
        self.assertTrue(result.e1.startswith("e1:sha256:"))
        self.assertEqual(result.semantic_metric, METRIC.COSINE)

    def test_the_matches_are_immutable(self):
        result = self.search()
        with self.assertRaises(Exception):
            result.matches[0].rank = 99
        with self.assertRaises(Exception):
            result.mode = "hybrid"

    def test_ranks_are_dense_and_ordered_by_fusion_score_then_chunk_id(self):
        result = self.search(limit=20)
        self.assertEqual(
            [match.rank for match in result.matches],
            list(range(1, len(result.matches) + 1)),
        )
        keys = [
            (-match.fusion_score, match.chunk_id) for match in result.matches
        ]
        self.assertEqual(keys, sorted(keys))

    def test_every_fusion_score_is_reproducible_from_the_reported_ranks(self):
        """Self-consistency: nothing else fed the number."""
        result = self.search(limit=20)
        for match in result.matches:
            with self.subTest(chunk=match.chunk_id):
                expected = 0.0
                if match.lexical_rank is not None:
                    expected += 1 / (RRF_K + match.lexical_rank)
                if match.semantic_rank is not None:
                    expected += 1 / (RRF_K + match.semantic_rank)
                self.assertAlmostEqual(match.fusion_score, expected, places=15)

    def test_no_query_content_or_vector_reaches_the_result(self):
        result = self.search()
        blob = json.dumps(
            {
                "result": {
                    key: value for key, value in result.__dict__.items()
                    if key != "matches"
                },
                "matches": [match.__dict__ for match in result.matches],
            },
            default=str,
        )
        for forbidden in (
            A_SECRET, B_SECRET, QUERY, "alpha", "widget",
            "snippet", "citation", "content", "vector", "api_key",
        ):
            self.assertNotIn(forbidden, blob)

    def test_the_result_exposes_no_raw_branch_score(self):
        result = self.search()
        for field in ("score", "metric_value", "lexical_score", "semantic_score"):
            self.assertNotIn(field, result.matches[0].__dict__)

    def test_fusion_score_is_documented_as_an_ordering_value_only(self):
        """It must never be rendered as a confidence or a percentage."""
        doc = " ".join((HybridMatch.__doc__ or "").lower().split())
        for disclaimer in ("not a probability", "not a confidence"):
            self.assertIn(disclaimer, doc)
        result = self.search()
        # An RRF score is bounded by 2/(k+1); reading it as a percentage would
        # make every result look near-zero relevance, which is the tell.
        for match in result.matches:
            self.assertLessEqual(match.fusion_score, 2.0 / (RRF_K + 1))


# ---------------------------------------------------------------------------
# Public API compatibility, and what S-22 deliberately does not add
# ---------------------------------------------------------------------------

class PublicApiCompatibilityTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_the_public_lexical_signature_is_unchanged(self):
        parameters = inspect.signature(search_knowledge).parameters
        self.assertEqual(
            list(parameters), ["agent", "query", "collection_id", "limit"]
        )
        self.assertEqual(parameters["limit"].default, 5)
        self.assertIsNone(parameters["collection_id"].default)

    def test_the_public_lexical_response_shape_is_unchanged(self):
        payload = search_knowledge(self.agent_a, query=QUERY)
        self.assertEqual(
            set(payload),
            {
                "query", "results", "total", "candidates_scanned",
                "candidate_limit", "candidates_truncated",
            },
        )
        self.assertEqual(
            set(payload["results"][0]),
            {
                "chunk_id", "document_id", "title", "collection",
                "section_title", "chunk_index", "snippet",
                "content_window_truncated", "score", "citation",
            },
        )

    def test_the_public_lexical_ordering_is_unchanged(self):
        payload = search_knowledge(self.agent_a, query=QUERY, limit=20)
        scope = resolve_effective_knowledge_scope(self.agent_a)
        ranking = rank_knowledge_chunks_with_scope(
            scope, query=QUERY, collection_ids=scope.collection_ids, limit=20
        )
        self.assertEqual(
            [row["chunk_id"] for row in payload["results"]],
            [candidate.chunk_id for candidate in ranking.candidates],
        )

    def test_the_public_lexical_refusals_are_unchanged(self):
        with self.assertRaises(ValidationError):
            search_knowledge(self.agent_a, query="   ")
        with self.assertRaises(ValidationError):
            search_knowledge(self.agent_a, query=QUERY, collection_id="abc")
        with self.assertRaises(ValidationError):
            search_knowledge(
                self.agent_a, query=QUERY, collection_id=self.coll_b1.pk
            )

    def test_the_public_semantic_signature_is_unchanged(self):
        parameters = inspect.signature(semantic_search_knowledge_local).parameters
        self.assertEqual(
            list(parameters),
            [
                "agent", "query", "embedding_model_config",
                "workspace", "collection_id", "limit",
            ],
        )
        self.assertEqual(parameters["limit"].default, 5)

    def test_there_is_exactly_one_lexical_scoring_implementation(self):
        """No duplicated algorithm to drift between the two projections."""
        source = inspect.getsource(knowledge_retrieval)
        tree = ast.parse(source)
        definitions = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and "score" in node.name
        ]
        self.assertEqual(definitions, ["_score_chunk"])

        hybrid_tree = ast.parse(inspect.getsource(hybrid_retrieval))
        hybrid_functions = {
            node.name for node in ast.walk(hybrid_tree)
            if isinstance(node, ast.FunctionDef)
        }
        for forbidden in ("_score_chunk", "_query_words", "_snippet"):
            self.assertNotIn(forbidden, hybrid_functions)


class AbsenceTests(TestCase):
    """Everything S-22 deliberately does NOT add.

    AST/identifier based rather than text based: this module's docstrings name
    almost every forbidden concept while explaining why it is absent, so a
    source-text scan would match its own prose.
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

    def test_no_score_blending_normalization_or_weighting(self):
        names = self._identifiers(hybrid_retrieval)
        for forbidden in (
            "normalize_scores", "min_max", "blend", "weighted_sum",
            "lexical_weight", "semantic_weight", "weights", "alpha_weight",
        ):
            self.assertNotIn(forbidden, names)

    def test_no_threshold_or_no_answer_policy(self):
        names = self._identifiers(hybrid_retrieval)
        for forbidden in (
            "threshold", "min_score", "minimum_score", "min_similarity",
            "max_distance", "confidence", "no_answer", "answerable",
        ):
            self.assertNotIn(forbidden, names)

    def test_no_reranker_and_no_model_call(self):
        names = self._identifiers(hybrid_retrieval)
        for forbidden in (
            "rerank", "reranker", "cross_encoder", "litellm",
            "run_agent", "completion", "chat", "llm",
        ):
            self.assertNotIn(forbidden, names)

    def test_no_persistence_and_no_audit_model(self):
        names = self._identifiers(hybrid_retrieval)
        for forbidden in (
            "save", "create", "bulk_create", "get_or_create", "update_or_create",
            "delete", "atomic", "select_for_update",
            "RetrievalRun", "RetrievalHit", "HybridRetrievalRun",
        ):
            self.assertNotIn(forbidden, names)

    def test_no_ann_or_alternative_vector_backend(self):
        names = self._identifiers(hybrid_retrieval)
        for forbidden in (
            "pgvector", "VectorField", "hnsw", "ivfflat",
            "chromadb", "chroma", "faiss", "numpy", "scipy", "np",
        ):
            self.assertNotIn(forbidden, names)

    def test_no_provider_fallback_or_scope_logic_of_its_own(self):
        names = self._identifiers(hybrid_retrieval)
        for forbidden in (
            "ProviderConfig", "ProviderGrant", "ApplicationScope",
            "EmbeddingModelConfig", "fallback", "declared_locality", "base_url",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_tool_layer_is_untouched(self):
        from ai_hub.services import knowledge_tooling
        from ai_hub.tools import knowledge as knowledge_tools

        for module in (knowledge_tooling, knowledge_tools):
            with self.subTest(module=module.__name__):
                names = self._identifiers(module)
                self.assertNotIn("hybrid_search_knowledge_local", names)
                self.assertNotIn("hybrid_retrieval", names)

    def test_the_semantic_and_lexical_metrics_are_unchanged(self):
        """S-22 consumes orderings; it defines no metric of its own."""
        names = self._identifiers(hybrid_retrieval)
        for forbidden in (
            "cosine_similarity", "dot_product_similarity", "euclidean_distance",
            "METRIC_SCORERS", "resolve_metric_scorer",
        ):
            self.assertNotIn(forbidden, names)


class NoSchemaChangeTests(TestCase):
    def test_this_slice_adds_no_migration(self):
        from django.db.migrations.loader import MigrationLoader

        loader = MigrationLoader(None, ignore_no_migrations=True)
        ai_hub_migrations = sorted(
            name for app, name in loader.disk_migrations if app == "ai_hub"
        )
        self.assertEqual(ai_hub_migrations[-1], "0028_knowledge_chunk_embedding")
