"""S-21: pre-filtered semantic retrieval.

The load-bearing tests here are NOT "the wrong document is absent from the top
five". Absence proves nothing about ordering: a global ranker that scored the
whole corpus and then filtered would pass that assertion while having already
read, decoded and ranked Knowledge the caller may not see.

So the corpus below is built adversarially. The unauthorized chunks carry the
vector that would rank FIRST, and the tests assert on what the metric scorer was
ever *called with* - the observable that separates pre-filtering from
post-filtering.
"""

import ast
import inspect
import json
import unicodedata
from decimal import Decimal
from unittest import mock

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
from ai_hub.services import embedding_vector, semantic_retrieval
from ai_hub.services.embedding_client import (
    EmbeddingProviderExecutionError,
    EmbeddingProviderResult,
    ErrorCategory,
)
from ai_hub.services.embedding_contract import (
    EmbeddingContractError,
    resolve_embedding_contract,
)
from ai_hub.services.embedding_egress import PAYLOAD_CORPUS, PAYLOAD_QUERY
from ai_hub.services.semantic_retrieval import (
    MAX_REFERENCE_SEMANTIC_CANDIDATES,
    canonical_query_embedding_text,
    search_semantic_with_scope,
    METRIC_SCORERS,
    MetricSpec,
    RetrievalFailureCategory,
    SemanticRetrievalError,
    cosine_similarity,
    dot_product_similarity,
    euclidean_distance,
    resolve_metric_scorer,
    semantic_search_knowledge_local,
)
from ai_hub.services.knowledge_authorization import (
    resolve_effective_knowledge_scope,
)
from ai_hub.services.vector_store import decode_vector, store_chunk_vector

METRIC = EmbeddingModelConfig.DistanceMetric
NORMALIZATION = EmbeddingModelConfig.Normalization
LOCALITY = ProviderConfig.DeclaredLocality
POLICY = KnowledgeCollection.ExternalEmbeddingEgressPolicy

TRANSPORT_PATH = "ai_hub.services.semantic_retrieval.resolve_embedding_transport"

#: Markers that must never appear in a result, a match or an error message.
A_SECRET = "ALPHA-CONFIDENTIAL-9911"
B_SECRET = "BETA-CONFIDENTIAL-2277"
QUERY_SECRET = "QUERY-CONFIDENTIAL-4455"


class RetrievalFixtureMixin:
    """An adversarial two-application corpus.

    The two chunks the caller may NOT reach - `unassigned` (same scope, a
    collection the Agent has no assignment to) and `foreign` (a different
    application entirely) - both score 1.0 against the query, strictly better
    than every authorized chunk. A leak therefore surfaces as a visible, rank-1
    result rather than as something buried below the cut.

    All six vectors are distinct, so the scorer call log identifies precisely
    which chunk was scored.
    """

    _world_counter = 0

    def build_corpus(
        self,
        *,
        metric=METRIC.COSINE,
        normalization=NORMALIZATION.NONE,
        locality=LOCALITY.LOCAL,
        grant_a=True,
        grant_b=True,
        max_input_chars=8000,
    ):
        self._world_counter += 1
        tag = self._world_counter

        provider = ProviderConfig.objects.create(
            name=f"Chat P{tag}", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        self.model_config = ModelConfig.objects.create(
            provider=provider, model_name="training",
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
            normalization=normalization, max_input_chars=max_input_chars,
        )
        if grant_a is not None:
            ProviderGrant.objects.create(
                application_scope=self.scope_a, provider=self.embed_provider,
                allow_embeddings=grant_a,
            )
        if grant_b is not None:
            ProviderGrant.objects.create(
                application_scope=self.scope_b, provider=self.embed_provider,
                allow_embeddings=grant_b,
            )

        # -- Scope A ------------------------------------------------------
        self.coll_a1 = self._collection(self.scope_a, f"A One {tag}", A_SECRET)
        self.coll_a2 = self._collection(self.scope_a, f"A Two {tag}", A_SECRET)
        # Same scope, deliberately NOT assigned to the agent.
        self.coll_a3 = self._collection(self.scope_a, f"A Three {tag}", A_SECRET)
        # -- Scope B, an entirely different application --------------------
        self.coll_b1 = self._collection(self.scope_b, f"B One {tag}", B_SECRET)

        self.a1 = self._chunk(self.coll_a1, 1, "Alpha", f"{A_SECRET} one")
        self.a2 = self._chunk(self.coll_a1, 2, "Alpha", f"{A_SECRET} two")
        self.a3 = self._chunk(self.coll_a1, 3, "Alpha", f"{A_SECRET} three")
        self.a4 = self._chunk(self.coll_a2, 1, "Alpha", f"{A_SECRET} four")
        self.unassigned = self._chunk(self.coll_a3, 1, "Alpha", f"{A_SECRET} five")
        self.foreign = self._chunk(self.coll_b1, 1, "Beta", f"{B_SECRET} one")

        self.agent_a = self._agent(f"A Agent {tag}", self.scope_a)
        self.agent_a.knowledge_collections.add(self.coll_a1, self.coll_a2)
        self.agent_b = self._agent(f"B Agent {tag}", self.scope_b)
        self.agent_b.knowledge_collections.add(self.coll_b1)

        # Every vector is DISTINCT, so a scorer call log identifies exactly
        # which chunk was scored. Against the query (1, 0, 0, 0), cosine gives:
        #   unassigned -> 1.0   SAME SCOPE, unassigned collection. Would rank 1.
        #   foreign    -> 1.0   FOREIGN SCOPE. Would rank 1. (Cosine ignores
        #                       scale, so a shorter vector still ties at 1.0
        #                       while remaining a different tuple.)
        #   a1         -> 0.8   authorized, the best the caller may actually see
        #   a3         -> 0.6   authorized
        #   a2         -> 0.0   authorized, ties with a4
        #   a4         -> 0.0   authorized, ties with a2 (higher chunk id)
        raw = {
            "a1": (0.8, 0.6, 0.0, 0.0),
            "a2": (0.0, 1.0, 0.0, 0.0),
            "a3": (0.6, 0.8, 0.0, 0.0),
            "a4": (0.0, 0.0, 1.0, 0.0),
            "unassigned": (1.0, 0.0, 0.0, 0.0),
            "foreign": (0.5, 0.0, 0.0, 0.0),
        }
        # `f32le1` is float32, so 0.6 does not round-trip to 0.6. Assertions
        # compare against what was actually STORED, decoded through the same
        # S-19 codec the retriever uses - never against the Python literals,
        # which would make a test fail on a rounding artefact.
        self.vectors = {}
        for name, values in raw.items():
            record = self.index(getattr(self, name), values)
            self.vectors[name] = decode_vector(
                record.vector_bytes, expected_dimension=record.vector_dimension
            )
        return self

    def _collection(self, scope, name, secret):
        return KnowledgeCollection.objects.create(
            name=name, description=f"{secret} description", application_scope=scope
        )

    def _chunk(self, collection, index, section, content):
        document, _ = KnowledgeDocument.objects.get_or_create(
            collection=collection, title=f"Doc {collection.name}",
            defaults={
                "curated_text": content,
                "status": KnowledgeDocument.Status.ACTIVE,
            },
        )
        return KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=index,
            section_title=section, content=content,
        )

    def _agent(self, name, scope):
        return AgentProfile.objects.create(
            name=name, role="r", model_config=self.model_config,
            application_scope=scope, knowledge_max_chars=6000,
        )

    def index(self, chunk, values):
        """Persist a vector through the ONLY canonical S-19 storage boundary."""
        return store_chunk_vector(
            application_scope=chunk.document.collection.application_scope,
            chunk=chunk,
            embedding_model_config=self.config,
            vector=tuple(values),
        )

    # -- driving the search ------------------------------------------------

    def transport_returning(self, values, *, hook=None):
        """A fake transport that records calls and never touches a socket."""
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

    def search(self, transport=None, *, agent=None, query="find alpha", **kwargs):
        transport = transport or self.transport_returning((1.0, 0.0, 0.0, 0.0))
        self.transport = transport
        with mock.patch(TRANSPORT_PATH, return_value=transport):
            return semantic_search_knowledge_local(
                agent if agent is not None else self.agent_a,
                query=query,
                embedding_model_config=self.config,
                **kwargs,
            )

    def scoring_spy(self):
        """Replace every scorer with a recording wrapper.

        Records the exact vector each scorer is invoked with, which is what makes
        "was this candidate ever scored" observable rather than inferred from the
        final result list.
        """
        seen = []
        wrapped = {}
        for metric, spec in METRIC_SCORERS.items():
            inner = spec.score

            def make(inner_fn):
                def spy(query, candidate):
                    seen.append(tuple(candidate))
                    return inner_fn(query, candidate)
                return spy

            wrapped[metric] = MetricSpec(
                score=make(inner), higher_is_better=spec.higher_is_better
            )
        wrapped_seen = seen
        patcher = mock.patch.dict(
            semantic_retrieval.METRIC_SCORERS, wrapped, clear=True
        )
        return patcher, wrapped_seen

    def chunk_ids(self, result):
        return [match.chunk_id for match in result.matches]


# ---------------------------------------------------------------------------
# THE primary invariant: authorization -> candidate generation -> ranking
# ---------------------------------------------------------------------------

class OrderingInvariantTests(RetrievalFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_unauthorized_chunks_are_never_scored(self):
        """The load-bearing test. Absence from the results is NOT the assertion.

        `unassigned` and `foreign` both hold (1,0,0,0) - the top-scoring vector.
        A global-rank-then-filter implementation would score them, rank them
        first, then drop them, and every assertion about the final list would
        still pass. Only the scorer call log distinguishes the two designs.
        """
        patcher, seen = self.scoring_spy()
        with patcher:
            result = self.search()

        self.assertNotIn(self.vectors["unassigned"], seen)
        self.assertNotIn(self.vectors["foreign"], seen)
        self.assertEqual(
            sorted(seen),
            sorted(
                self.vectors[name] for name in ("a1", "a2", "a3", "a4")
            ),
        )
        self.assertEqual(result.candidate_count, 4)
        self.assertEqual(result.scored_count, 4)

    def test_the_unauthorized_vectors_really_would_have_won(self):
        """Proves the previous test is not vacuous.

        If the forbidden vectors were reachable they would tie for rank 1, so
        their absence from the scorer log is a real property of the pipeline and
        not an artefact of them being uninteresting.
        """
        query = (1.0, 0.0, 0.0, 0.0)
        best_authorized = max(
            cosine_similarity(query, self.vectors[name])
            for name in ("a1", "a2", "a3", "a4")
        )
        for name in ("foreign", "unassigned"):
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    cosine_similarity(query, self.vectors[name]), 1.0, places=6
                )
                self.assertGreater(
                    cosine_similarity(query, self.vectors[name]), best_authorized,
                    "the forbidden vectors must genuinely outrank every "
                    "authorized one, or their absence proves nothing",
                )

    def test_candidates_are_generated_before_the_provider_is_called(self):
        """No candidates means no inference at all."""
        KnowledgeChunkEmbedding.objects.all().delete()
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        result = self.search(transport)

        self.assertEqual(transport.calls, [])
        self.assertFalse(result.provider_invoked)
        self.assertEqual(result.matches, ())
        self.assertEqual(result.candidate_count, 0)

    def test_a_foreign_scope_agent_sees_only_its_own_corpus(self):
        patcher, seen = self.scoring_spy()
        with patcher:
            result = self.search(agent=self.agent_b)

        self.assertEqual(seen, [self.vectors["foreign"]])
        self.assertEqual(self.chunk_ids(result), [self.foreign.pk])
        self.assertEqual(result.application_scope_id, self.scope_b.pk)

    def test_source_never_ranks_then_filters(self):
        """Structural, so a future refactor cannot quietly invert the order.

        Scans the INTERNAL scope-aware implementation, which is where the work
        now lives - scanning the public wrapper would inspect three lines of
        delegation and prove nothing.

        Asserted on the AST rather than on the text, because this module's own
        docstrings name every forbidden shape while explaining why it is
        forbidden - a source-text scan would match its own prose.
        """
        source = inspect.getsource(search_semantic_with_scope)
        tree = ast.parse(source.lstrip())
        calls = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.setdefault(node.func.id, node.lineno)

        self.assertIn("load_current_vectors", calls)
        self.assertIn("resolve_embedding_access", calls)
        self.assertIn("authorized_chunks", calls)

        self.assertLess(
            calls["authorized_chunks"], calls["load_current_vectors"],
            "the authorized chunk set must precede candidate generation",
        )
        self.assertLess(
            calls["resolve_embedding_access"], calls["load_current_vectors"],
            "egress authorization must precede candidate generation",
        )
        self.assertLess(
            calls["load_current_vectors"], calls["transport"],
            "candidates must be generated before the provider is called",
        )

    def test_the_scoped_load_is_the_only_candidate_source(self):
        """Candidates come from the scoped loader, never from a bare queryset."""
        source = inspect.getsource(search_semantic_with_scope)
        called = {
            node.func.attr
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for forbidden in ("all", "raw", "extra"):
            self.assertNotIn(forbidden, called)


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------

class AuthorizationTests(RetrievalFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def assertDeniedEmpty(self, result, transport):
        self.assertEqual(result.matches, ())
        self.assertFalse(result.provider_invoked)
        self.assertEqual(transport.calls, [], "a denied search must not embed")

    def test_an_inactive_agent_retrieves_nothing(self):
        AgentProfile.objects.filter(pk=self.agent_a.pk).update(is_active=False)
        self.agent_a.refresh_from_db()
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        self.assertDeniedEmpty(self.search(transport), transport)

    def test_an_inactive_scope_retrieves_nothing(self):
        ApplicationScope.objects.filter(pk=self.scope_a.pk).update(is_active=False)
        self.agent_a.refresh_from_db()
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        self.assertDeniedEmpty(self.search(transport), transport)

    def test_an_agent_with_no_assignments_retrieves_nothing(self):
        self.agent_a.knowledge_collections.clear()
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        self.assertDeniedEmpty(self.search(transport), transport)

    def test_a_foreign_workspace_retrieves_nothing(self):
        workspace = GameWorkspace.objects.create(
            name="Foreign WS", application_scope=self.scope_b
        )
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        self.assertDeniedEmpty(self.search(transport, workspace=workspace), transport)

    def test_a_coherent_workspace_does_not_widen_or_narrow(self):
        workspace = GameWorkspace.objects.create(
            name="Own WS", application_scope=self.scope_a
        )
        result = self.search(workspace=workspace)
        self.assertEqual(result.workspace_id, workspace.pk)
        self.assertEqual(len(result.matches), 4)

    def test_collection_narrowing_restricts_the_search(self):
        patcher, seen = self.scoring_spy()
        with patcher:
            result = self.search(collection_id=self.coll_a2.pk)

        self.assertEqual(seen, [self.vectors["a4"]])
        self.assertEqual(self.chunk_ids(result), [self.a4.pk])
        self.assertEqual(result.collection_ids, (self.coll_a2.pk,))

    def test_collection_narrowing_can_never_widen(self):
        """Naming an unassigned collection returns the same empty answer.

        Same shape as a caller whose corpus is simply empty (ADR-N5), so the
        refusal cannot be used to enumerate which collections exist.
        """
        for label, collection in (
            ("unassigned same scope", self.coll_a3),
            ("foreign scope", self.coll_b1),
        ):
            with self.subTest(label=label):
                transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
                result = self.search(transport, collection_id=collection.pk)
                self.assertDeniedEmpty(result, transport)

    def test_an_unknown_collection_id_is_refused_the_same_way(self):
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        self.assertDeniedEmpty(
            self.search(transport, collection_id=9_999_999), transport
        )

    def test_a_deactivated_collection_drops_out_of_the_search(self):
        KnowledgeCollection.objects.filter(pk=self.coll_a1.pk).update(is_active=False)
        patcher, seen = self.scoring_spy()
        with patcher:
            result = self.search()

        self.assertEqual(seen, [self.vectors["a4"]])
        self.assertEqual(result.collection_ids, (self.coll_a2.pk,))

    def test_an_archived_document_is_never_a_candidate(self):
        KnowledgeDocument.objects.filter(
            pk=self.a1.document_id
        ).update(status=KnowledgeDocument.Status.ARCHIVED)
        patcher, seen = self.scoring_spy()
        with patcher:
            result = self.search()

        self.assertNotIn(self.vectors["a1"], seen)
        self.assertNotIn(self.a1.pk, self.chunk_ids(result))
        self.assertEqual(result.candidate_count, 1)


# ---------------------------------------------------------------------------
# S-17 egress: QUERY, never CORPUS; LOCAL only
# ---------------------------------------------------------------------------

class QueryEgressTests(RetrievalFixtureMixin, TestCase):
    def test_the_query_payload_kind_is_used_not_the_corpus_one(self):
        """The decisive test, and it does not mock anything.

        The scope permits external CORPUS egress but NOT external QUERY egress -
        a legal S-17 state, since query egress is a strict subset. A retriever
        that asked under `PAYLOAD_CORPUS` would be allowed here. This one must
        be refused.
        """
        self.build_corpus(locality=LOCALITY.EXTERNAL)
        ApplicationScope.objects.filter(pk=self.scope_a.pk).update(
            allow_external_embedding_corpus_egress=True,
            allow_external_embedding_query_egress=False,
        )
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(transport)

        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.EMBEDDING_NOT_AUTHORIZED,
        )
        self.assertEqual(transport.calls, [])

    def test_every_target_collection_must_authorize(self):
        """One denying collection refuses the WHOLE search, never a subset.

        A silently narrower search is the dangerous outcome: it returns a
        confident ranking drawn from less Knowledge than the caller believes it
        searched, and nothing in the result says so.
        """
        self.build_corpus(locality=LOCALITY.EXTERNAL)
        ApplicationScope.objects.filter(pk=self.scope_a.pk).update(
            allow_external_embedding_corpus_egress=True,
            allow_external_embedding_query_egress=True,
        )
        # The SECOND collection alphabetically denies. The first would allow.
        KnowledgeCollection.objects.filter(pk=self.coll_a2.pk).update(
            external_embedding_egress_policy=POLICY.DENY
        )
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(transport)

        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.EMBEDDING_NOT_AUTHORIZED,
        )
        self.assertEqual(transport.calls, [])

    def test_authorization_is_complete_before_capability_is_considered(self):
        """Every collection is checked even when the first is already fatal.

        Ordering matters: if the local-only capability refusal fired on the
        first collection, a denying second collection would never be examined,
        and the refusal reported would be the wrong one.
        """
        self.build_corpus(locality=LOCALITY.EXTERNAL)
        ApplicationScope.objects.filter(pk=self.scope_a.pk).update(
            allow_external_embedding_corpus_egress=True,
            allow_external_embedding_query_egress=True,
        )
        KnowledgeCollection.objects.filter(pk=self.coll_a2.pk).update(
            external_embedding_egress_policy=POLICY.DENY
        )
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(self.transport_returning((1.0, 0.0, 0.0, 0.0)))
        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.EMBEDDING_NOT_AUTHORIZED,
        )

    def test_one_query_decision_is_taken_per_target_collection(self):
        self.build_corpus()
        seen = []
        real = semantic_retrieval.resolve_embedding_access

        def spy(scope, provider, *, collection, payload_kind):
            seen.append((collection.pk, payload_kind))
            return real(scope, provider, collection=collection,
                        payload_kind=payload_kind)

        with mock.patch.object(
            semantic_retrieval, "resolve_embedding_access", spy
        ):
            self.search()

        self.assertEqual(
            sorted(seen),
            sorted([
                (self.coll_a1.pk, PAYLOAD_QUERY),
                (self.coll_a2.pk, PAYLOAD_QUERY),
            ]),
        )
        self.assertNotIn(
            PAYLOAD_CORPUS, [payload_kind for _pk, payload_kind in seen]
        )

    def test_a_missing_provider_grant_refuses_before_any_inference(self):
        self.build_corpus(grant_a=None)
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(transport)
        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.EMBEDDING_NOT_AUTHORIZED,
        )
        self.assertEqual(transport.calls, [])

    def test_a_grant_that_forbids_embeddings_refuses(self):
        self.build_corpus(grant_a=False)
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(transport)
        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.EMBEDDING_NOT_AUTHORIZED,
        )
        self.assertEqual(transport.calls, [])

    def test_undeclared_locality_is_never_treated_as_local(self):
        self.build_corpus(locality=LOCALITY.UNKNOWN)
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(transport)
        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.EMBEDDING_NOT_AUTHORIZED,
        )
        self.assertEqual(transport.calls, [])

    def test_an_authorized_external_provider_is_still_refused_here(self):
        """S-17 may allow it; this slice does not implement it.

        A capability limit, stated as one. It is not an inference about where
        the provider actually is.
        """
        self.build_corpus(locality=LOCALITY.EXTERNAL)
        ApplicationScope.objects.filter(pk=self.scope_a.pk).update(
            allow_external_embedding_corpus_egress=True,
            allow_external_embedding_query_egress=True,
        )
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(transport)
        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.LOCAL_ONLY_EXECUTION_REQUIRED,
        )
        self.assertEqual(transport.calls, [])

    def test_locality_is_never_inferred_from_the_url(self):
        """A loopback base URL grants nothing on its own."""
        self.build_corpus(locality=LOCALITY.UNKNOWN)
        ProviderConfig.objects.filter(pk=self.embed_provider.pk).update(
            base_url="http://127.0.0.1:11434"
        )
        self.embed_provider.refresh_from_db()
        self.config.refresh_from_db()
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError):
            self.search(transport)
        self.assertEqual(transport.calls, [])


# ---------------------------------------------------------------------------
# Exact ranking
# ---------------------------------------------------------------------------

class RankingTests(RetrievalFixtureMixin, TestCase):
    def test_cosine_ranks_by_direction_and_reports_similarity(self):
        self.build_corpus(metric=METRIC.COSINE)
        result = self.search(limit=4)

        self.assertEqual(result.metric, METRIC.COSINE)
        self.assertTrue(result.higher_is_better)
        self.assertEqual(
            self.chunk_ids(result),
            # a1 (0.8), a3 (0.6), then the 0.0 tie broken by chunk id ASC.
            [self.a1.pk, self.a3.pk, self.a2.pk, self.a4.pk],
        )
        self.assertAlmostEqual(result.matches[0].metric_value, 0.8, places=6)
        self.assertAlmostEqual(result.matches[1].metric_value, 0.6, places=6)
        self.assertAlmostEqual(result.matches[2].metric_value, 0.0, places=6)
        self.assertEqual([match.rank for match in result.matches], [1, 2, 3, 4])

    def test_ties_break_on_chunk_id_ascending(self):
        self.build_corpus(metric=METRIC.COSINE)
        result = self.search(limit=4)
        tied = [
            match for match in result.matches
            if abs(match.metric_value) < 1e-12
        ]
        self.assertEqual(len(tied), 2)
        self.assertEqual(
            [match.chunk_id for match in tied],
            sorted(match.chunk_id for match in tied),
        )

    def test_dot_product_ranks_by_inner_product(self):
        self.build_corpus(metric=METRIC.DOT_PRODUCT)
        result = self.search(limit=4)

        self.assertEqual(result.metric, METRIC.DOT_PRODUCT)
        self.assertTrue(result.higher_is_better)
        self.assertEqual(
            self.chunk_ids(result),
            [self.a1.pk, self.a3.pk, self.a2.pk, self.a4.pk],
        )
        self.assertAlmostEqual(result.matches[0].metric_value, 0.8, places=6)
        self.assertAlmostEqual(result.matches[1].metric_value, 0.6, places=6)

    def test_euclidean_reports_raw_distance_and_ranks_ascending(self):
        self.build_corpus(metric=METRIC.EUCLIDEAN)
        result = self.search(limit=4)

        self.assertEqual(result.metric, METRIC.EUCLIDEAN)
        self.assertFalse(result.higher_is_better)
        self.assertEqual(self.chunk_ids(result)[0], self.a1.pk)
        # The RAW distance from (1,0,0,0) to (0.8,0.6,0,0), reported as-is and
        # never inverted into a similarity.
        self.assertAlmostEqual(result.matches[0].metric_value, 0.632455, places=5)
        self.assertGreater(result.matches[0].metric_value, 0.0)
        values = [match.metric_value for match in result.matches]
        self.assertEqual(values, sorted(values))

    def test_cosine_does_not_assume_unit_vectors(self):
        """cosine + normalization=none is legal, and must still be cosine.

        With the query (1,1,0,0), the inner product prefers the long vector and
        cosine prefers the aligned one. A scorer that shortcut to a dot product
        "because the vectors are probably normalized" would rank these the wrong
        way round, and nothing else in the pipeline would notice.
        """
        long_but_skewed = (3.0, 0.0, 0.0, 0.0)
        short_but_aligned = (1.0, 1.0, 0.0, 0.0)
        query = (1.0, 1.0, 0.0, 0.0)

        self.assertGreater(
            dot_product_similarity(query, long_but_skewed),
            dot_product_similarity(query, short_but_aligned),
        )
        self.assertLess(
            cosine_similarity(query, long_but_skewed),
            cosine_similarity(query, short_but_aligned),
        )
        self.assertAlmostEqual(
            cosine_similarity(query, short_but_aligned), 1.0, places=12
        )

    def test_the_metrics_are_exact(self):
        self.assertAlmostEqual(
            cosine_similarity((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
            32.0 / ((14.0 ** 0.5) * (77.0 ** 0.5)),
            places=12,
        )
        self.assertAlmostEqual(
            dot_product_similarity((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), 32.0, places=12
        )
        self.assertAlmostEqual(
            euclidean_distance((1.0, 2.0, 3.0), (4.0, 6.0, 3.0)), 5.0, places=12
        )

    def test_cosine_refuses_a_zero_vector_rather_than_inventing_a_score(self):
        with self.assertRaises(SemanticRetrievalError) as raised:
            cosine_similarity((1.0, 0.0), (0.0, 0.0))
        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.UNSCORABLE_ZERO_VECTOR,
        )

    def test_an_unsupported_metric_refuses_before_any_inference(self):
        """S-18 refuses first, and that is the correct layer.

        Named for what it proves - that no inference happens - not for which
        module refuses. `resolve_metric_scorer` is reached only if a metric ever
        becomes contract-valid without a scorer, so its own refusal is tested
        directly below rather than pretended to fire here.
        """
        self.build_corpus()
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(
            distance_metric="manhattan"
        )
        self.config.refresh_from_db()
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(EmbeddingContractError):
            self.search(transport)
        self.assertEqual(transport.calls, [])

    def test_an_unsupported_metric_never_falls_back_to_cosine(self):
        """The defensive branch, exercised directly rather than by proxy."""
        for bad in ("manhattan", None, ""):
            with self.subTest(metric=bad):
                with self.assertRaises(SemanticRetrievalError) as raised:
                    resolve_metric_scorer(bad)
                self.assertEqual(
                    raised.exception.category,
                    RetrievalFailureCategory.UNSUPPORTED_DISTANCE_METRIC,
                )

    def test_every_declared_metric_has_a_scorer_and_a_direction(self):
        for metric, _label in METRIC.choices:
            with self.subTest(metric=metric):
                spec = resolve_metric_scorer(metric)
                self.assertTrue(callable(spec.score))
                self.assertIsInstance(spec.higher_is_better, bool)


# ---------------------------------------------------------------------------
# limit
# ---------------------------------------------------------------------------

class LimitTests(RetrievalFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_limit_truncates_the_ranking_not_the_candidate_set(self):
        result = self.search(limit=2)
        self.assertEqual(len(result.matches), 2)
        self.assertEqual(self.chunk_ids(result), [self.a1.pk, self.a3.pk])
        # All four were still generated and scored.
        self.assertEqual(result.candidate_count, 4)
        self.assertEqual(result.scored_count, 4)

    def test_a_limit_above_the_candidate_count_returns_everything(self):
        result = self.search(limit=99)
        self.assertEqual(len(result.matches), 4)

    def test_the_default_limit_is_five(self):
        self.assertEqual(
            inspect.signature(
                semantic_search_knowledge_local
            ).parameters["limit"].default,
            5,
        )

    def test_limit_zero_returns_empty_without_calling_the_provider(self):
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        result = self.search(transport, limit=0)
        self.assertEqual(result.matches, ())
        self.assertFalse(result.provider_invoked)
        self.assertEqual(transport.calls, [], "limit=0 must not embed the query")

    def test_a_negative_or_non_integer_limit_is_refused(self):
        for bad in (-1, 1.5, "3", True, None):
            with self.subTest(limit=bad):
                transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
                with self.assertRaises(SemanticRetrievalError) as raised:
                    self.search(transport, limit=bad)
                self.assertEqual(
                    raised.exception.category,
                    RetrievalFailureCategory.INVALID_LIMIT,
                )
                self.assertEqual(transport.calls, [])


# ---------------------------------------------------------------------------
# The reference candidate ceiling
# ---------------------------------------------------------------------------

class ReferenceCeilingTests(RetrievalFixtureMixin, TestCase):
    def test_the_ceiling_is_one_thousand(self):
        self.assertEqual(MAX_REFERENCE_SEMANTIC_CANDIDATES, 1000)

    def test_exceeding_the_ceiling_refuses_and_never_truncates(self):
        """Refusal, not a silent top-N of the first 1000.

        Patched to a small ceiling so the boundary is exercised honestly rather
        than approximated: the code path, the comparison and the category are
        the real ones, only the number is smaller.
        """
        self.build_corpus()
        with mock.patch.object(
            semantic_retrieval, "MAX_REFERENCE_SEMANTIC_CANDIDATES", 3
        ):
            transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
            with self.assertRaises(SemanticRetrievalError) as raised:
                self.search(transport)

        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.REFERENCE_CANDIDATE_LIMIT_EXCEEDED,
        )
        self.assertEqual(transport.calls, [], "a refused search must not embed")

    def test_exactly_the_ceiling_is_allowed(self):
        self.build_corpus()
        with mock.patch.object(
            semantic_retrieval, "MAX_REFERENCE_SEMANTIC_CANDIDATES", 4
        ):
            result = self.search(limit=4)
        self.assertEqual(result.candidate_count, 4)
        self.assertEqual(len(result.matches), 4)

    def test_the_ceiling_counts_only_authorized_candidates(self):
        """Six vectors exist; only the four authorized ones count toward it."""
        self.build_corpus()
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 6)
        with mock.patch.object(
            semantic_retrieval, "MAX_REFERENCE_SEMANTIC_CANDIDATES", 5
        ):
            result = self.search(limit=4)
        self.assertEqual(result.candidate_count, 4)


# ---------------------------------------------------------------------------
# Vector currency, before and after the provider call
# ---------------------------------------------------------------------------

class CandidateCurrencyTests(RetrievalFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_a_stale_k1_is_never_a_candidate(self):
        """Editing the chunk invalidates its stored vector immediately."""
        KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).update(
            content=f"{A_SECRET} rewritten"
        )
        patcher, seen = self.scoring_spy()
        with patcher:
            result = self.search()

        self.assertNotIn(self.vectors["a1"], seen)
        self.assertNotIn(self.a1.pk, self.chunk_ids(result))
        self.assertEqual(result.candidate_count, 3)

    def test_a_vector_in_another_space_is_never_a_candidate(self):
        """A different `e1` is a different vector space, not a stale row."""
        contract = resolve_embedding_contract(self.config)
        KnowledgeChunkEmbedding.objects.filter(chunk_id=self.a1.pk).update(
            e1="e1:sha256:" + ("0" * 64)
        )
        result = self.search()
        self.assertEqual(result.e1, contract.e1)
        self.assertNotIn(self.a1.pk, self.chunk_ids(result))

    def test_a_chunk_edited_during_the_provider_call_is_dropped(self):
        """Post-inference revalidation, exercised on the real timing window."""
        def edit_mid_flight():
            KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).update(
                content=f"{A_SECRET} changed in flight"
            )

        transport = self.transport_returning(
            (1.0, 0.0, 0.0, 0.0), hook=edit_mid_flight
        )
        result = self.search(transport)

        self.assertEqual(result.candidate_count, 4, "it WAS a candidate")
        self.assertEqual(result.scored_count, 3, "but it was not scored")
        self.assertNotIn(self.a1.pk, self.chunk_ids(result))

    def test_a_collection_deactivated_during_the_provider_call_is_dropped(self):
        def deactivate_mid_flight():
            KnowledgeCollection.objects.filter(pk=self.coll_a1.pk).update(
                is_active=False
            )

        transport = self.transport_returning(
            (1.0, 0.0, 0.0, 0.0), hook=deactivate_mid_flight
        )
        result = self.search(transport)

        self.assertEqual(result.candidate_count, 4, "they WERE candidates")
        self.assertEqual(result.scored_count, 1, "only collection A Two survives")
        self.assertEqual(self.chunk_ids(result), [self.a4.pk])

    def test_a_chunk_deleted_during_the_provider_call_is_dropped(self):
        def delete_mid_flight():
            KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).delete()

        transport = self.transport_returning(
            (1.0, 0.0, 0.0, 0.0), hook=delete_mid_flight
        )
        result = self.search(transport)
        self.assertNotIn(self.a1.pk, self.chunk_ids(result))
        self.assertEqual(result.scored_count, 3)

    def test_the_resolved_scope_is_deliberately_not_re_resolved(self):
        """A boundary, recorded as one rather than left to be discovered.

        Removing the Agent's assignment mid-flight does NOT retroactively narrow
        the search already in progress: `EffectiveKnowledgeScope` is the
        authorization answer FOR this operation, taken once. Re-resolving it
        halfway would mean one search ran under two different authorization
        answers. The revocation takes full effect on the next search, which the
        second half of this test proves - so the window is bounded by one
        operation, not open-ended.
        """
        def revoke_mid_flight():
            self.agent_a.knowledge_collections.remove(self.coll_a1)

        transport = self.transport_returning(
            (1.0, 0.0, 0.0, 0.0), hook=revoke_mid_flight
        )
        in_flight = self.search(transport)
        self.assertIn(self.a1.pk, self.chunk_ids(in_flight))

        after = self.search(self.transport_returning((1.0, 0.0, 0.0, 0.0)))
        self.assertEqual(self.chunk_ids(after), [self.a4.pk])

    def test_a_document_archived_during_the_provider_call_is_dropped(self):
        def archive_mid_flight():
            KnowledgeDocument.objects.filter(pk=self.a1.document_id).update(
                status=KnowledgeDocument.Status.ARCHIVED
            )

        transport = self.transport_returning(
            (1.0, 0.0, 0.0, 0.0), hook=archive_mid_flight
        )
        result = self.search(transport)
        self.assertEqual(self.chunk_ids(result), [self.a4.pk])

    def test_everything_withdrawn_mid_flight_yields_an_empty_result(self):
        def deactivate_all():
            KnowledgeCollection.objects.filter(
                pk__in=[self.coll_a1.pk, self.coll_a2.pk]
            ).update(is_active=False)

        transport = self.transport_returning(
            (1.0, 0.0, 0.0, 0.0), hook=deactivate_all
        )
        result = self.search(transport)
        self.assertEqual(result.matches, ())
        self.assertEqual(result.scored_count, 0)

    def test_revalidation_never_re_embeds_on_the_callers_behalf(self):
        def edit_mid_flight():
            KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).update(
                content=f"{A_SECRET} changed in flight"
            )

        before = KnowledgeChunkEmbedding.objects.count()
        transport = self.transport_returning(
            (1.0, 0.0, 0.0, 0.0), hook=edit_mid_flight
        )
        self.search(transport)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), before)
        self.assertEqual(len(transport.calls), 1, "exactly one inference")


# ---------------------------------------------------------------------------
# The query itself
# ---------------------------------------------------------------------------

class QueryHandlingTests(RetrievalFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def dispatched(self, query):
        """The exact string the transport was handed for `query`."""
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        self.search(transport, query=query)
        self.assertEqual(len(transport.calls), 1)
        return transport.calls[0]["text"]

    def test_crlf_becomes_lf(self):
        """A Windows client and a Unix client typed the same query."""
        self.assertEqual(self.dispatched("A\r\nB"), "A\nB")

    def test_a_lone_cr_becomes_lf(self):
        self.assertEqual(self.dispatched("A\rB"), "A\nB")

    def test_a_crlf_pair_never_becomes_two_newlines(self):
        """Order matters: CRLF must be replaced before the lone-CR rule runs.

        A lone-CR rule applied first would turn every CRLF into `\n\n` and
        silently double every line break in the query.
        """
        self.assertEqual(self.dispatched("A\r\n\rB"), "A\n\nB")
        self.assertEqual(canonical_query_embedding_text("\r\n"), "\n")

    def test_unicode_is_composed_to_nfc(self):
        """Decomposed and precomposed spellings must reach the same vector."""
        decomposed = "cafe\u0301"
        precomposed = "caf\u00e9"
        self.assertNotEqual(decomposed, precomposed)

        sent = self.dispatched(decomposed)
        self.assertEqual(sent, precomposed)
        self.assertEqual(unicodedata.normalize("NFC", sent), sent)
        self.assertEqual(self.dispatched(precomposed), precomposed)

    def test_whitespace_case_and_punctuation_are_preserved_exactly(self):
        """Only REPRESENTATION is canonical. Meaning is untouched.

        The surrounding spaces are content: a query is free text, and trimming
        it would embed something the caller did not send.
        """
        self.assertEqual(
            self.dispatched("  Alpha  Beta  "), "  Alpha  Beta  "
        )
        self.assertEqual(self.dispatched("\tAlpha!?  "), "\tAlpha!?  ")
        self.assertEqual(self.dispatched("AlPhA"), "AlPhA")

    def test_blank_detection_never_mutates_the_dispatched_text(self):
        """`.strip()` is the emptiness predicate, never the payload.

        Decisive because every query here would survive a stripped dispatch
        looking perfectly plausible - only the exact bytes reveal it.
        """
        for query in ("  Alpha  ", "\r\n Alpha \r\n", " \u00e9 "):
            with self.subTest(query=query):
                sent = self.dispatched(query)
                self.assertEqual(sent, canonical_query_embedding_text(query))
                self.assertNotEqual(sent, sent.strip())
                self.assertNotEqual(sent, query.strip())

    def test_the_helper_is_pure_and_writes_nothing(self):
        """No `q1`, no persistence, no lifecycle normalization reused."""
        before = KnowledgeChunkEmbedding.objects.count()
        self.assertEqual(canonical_query_embedding_text(None), "")
        self.assertEqual(canonical_query_embedding_text(123), "")
        self.assertEqual(canonical_query_embedding_text(""), "")
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), before)

        tree = ast.parse(
            inspect.getsource(canonical_query_embedding_text).lstrip()
        )
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("chunk_embedding_input_fingerprint", called)
        self.assertNotIn("canonical_chunk_embedding_text", called)
        self.assertNotIn("sha256", called)

    def test_the_canonical_query_is_what_gets_length_checked(self):
        """NFC composition SHORTENS a string, so the two measurements differ.

        `"e" + U+0301` is 2 characters; the composed `U+00E9` is 1. A ceiling of
        4 admits the canonical form of a 6-character raw input and refuses the
        7-character one - a boundary that only holds if the check reads the
        canonical representation.
        """
        self.build_corpus(max_input_chars=4)

        fits_once_composed = "cafe\u0301"  # raw 5, canonical 4
        self.assertEqual(len(fits_once_composed), 5)
        self.assertEqual(len(canonical_query_embedding_text(fits_once_composed)), 4)
        sent = self.dispatched(fits_once_composed)
        self.assertEqual(sent, "caf\u00e9")
        self.assertEqual(len(sent), 4)

        too_long_even_composed = "cafes\u0301x"  # raw 7, canonical 6
        self.assertEqual(len(canonical_query_embedding_text(too_long_even_composed)), 6)
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(transport, query=too_long_even_composed)
        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.QUERY_INPUT_TOO_LARGE,
        )
        self.assertEqual(transport.calls, [])

    def test_the_refusal_length_reported_is_the_canonical_one(self):
        """The operator must be told the number the check actually used."""
        self.build_corpus(max_input_chars=4)
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(
                self.transport_returning((1.0, 0.0, 0.0, 0.0)),
                query="cafes\u0301x",
            )
        self.assertIn("6", str(raised.exception))
        self.assertNotIn("7", str(raised.exception))

    def test_a_query_that_is_blank_only_after_canonicalization_is_refused(self):
        for raw in ("\r\n", "\r", " \r\n\t "):
            with self.subTest(raw=raw):
                transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
                with self.assertRaises(SemanticRetrievalError) as raised:
                    self.search(transport, query=raw)
                self.assertEqual(
                    raised.exception.category, RetrievalFailureCategory.QUERY_EMPTY
                )
                self.assertEqual(transport.calls, [])

    def test_a_blank_query_refuses_before_any_inference(self):
        for bad in ("", "   ", "\n\t", None, 123):
            with self.subTest(query=bad):
                transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
                with self.assertRaises(SemanticRetrievalError) as raised:
                    self.search(transport, query=bad)
                self.assertEqual(
                    raised.exception.category, RetrievalFailureCategory.QUERY_EMPTY
                )
                self.assertEqual(transport.calls, [])

    def test_an_oversized_query_is_refused_never_truncated(self):
        self.build_corpus(max_input_chars=20)
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(transport, query="x" * 21)
        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.QUERY_INPUT_TOO_LARGE,
        )
        self.assertEqual(transport.calls, [], "nothing is sent to be truncated")

    def test_the_query_vector_is_never_persisted(self):
        before = KnowledgeChunkEmbedding.objects.count()
        self.search()
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), before)

    def test_no_query_fingerprint_contract_is_invented(self):
        """There is deliberately no `q1`.

        Checked against code strings and identifiers, never the raw source: the
        module's docstring explains at length why `q1` does not exist, and a
        text scan would match that explanation.
        """
        tree = ast.parse(inspect.getsource(semantic_retrieval))
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertNotIn("q1", {name.lower() for name in names})

        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        } - docstrings
        for literal in literals:
            self.assertNotIn("q1:", literal)

    def test_a_query_vector_of_the_wrong_dimension_is_refused(self):
        transport = self.transport_returning((1.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(transport)
        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.VECTOR_DIMENSION_MISMATCH,
        )

    def test_a_non_finite_query_vector_is_refused(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                transport = self.transport_returning((bad, 0.0, 0.0, 0.0))
                with self.assertRaises(SemanticRetrievalError) as raised:
                    self.search(transport)
                self.assertEqual(
                    raised.exception.category,
                    RetrievalFailureCategory.VECTOR_NON_FINITE,
                )

    def test_a_zero_query_vector_under_l2_is_refused(self):
        self.build_corpus(normalization=NORMALIZATION.L2)
        transport = self.transport_returning((0.0, 0.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(transport)
        self.assertEqual(
            raised.exception.category,
            RetrievalFailureCategory.ZERO_VECTOR_CANNOT_L2_NORMALIZE,
        )

    def test_a_transport_failure_propagates_without_the_query(self):
        def fail(*, provider, contract, text):
            raise EmbeddingProviderExecutionError(
                ErrorCategory.PROVIDER_UNREACHABLE, "The provider is unreachable."
            )

        with mock.patch(TRANSPORT_PATH, return_value=fail):
            with self.assertRaises(EmbeddingProviderExecutionError) as raised:
                semantic_search_knowledge_local(
                    self.agent_a, query=QUERY_SECRET,
                    embedding_model_config=self.config,
                )
        self.assertNotIn(QUERY_SECRET, str(raised.exception))


# ---------------------------------------------------------------------------
# Shared normalization: corpus and query must go through the SAME code
# ---------------------------------------------------------------------------

class SharedVectorPathTests(RetrievalFixtureMixin, TestCase):
    def test_the_query_is_normalized_by_the_shared_module(self):
        source = inspect.getsource(search_semantic_with_scope)
        called = {
            node.func.id
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("normalize_embedding_vector", called)
        self.assertIn("validate_embedding_vector", called)

    def test_the_corpus_path_uses_the_same_functions(self):
        from ai_hub.services import embedding_execution

        for wrapper, shared in (
            ("_normalize_vector", "normalize_embedding_vector"),
            ("_validate_raw_vector", "validate_embedding_vector"),
        ):
            with self.subTest(wrapper=wrapper):
                source = inspect.getsource(getattr(embedding_execution, wrapper))
                called = {
                    node.func.id
                    for node in ast.walk(ast.parse(source.lstrip()))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                }
                self.assertIn(shared, called)
        self.assertIs(
            embedding_execution.normalize_embedding_vector,
            embedding_vector.normalize_embedding_vector,
        )

    def test_an_l2_contract_really_normalizes_the_query(self):
        """Decisive because the metric is scale-SENSITIVE.

        Under cosine this would prove nothing - cosine ignores magnitude, so an
        un-normalized query would score identically. With `dot_product`, the
        provider's (2,0,0,0) scores 1.6 against a1 if it is passed through raw
        and 0.8 if the L2 contract was actually applied.
        """
        self.build_corpus(
            normalization=NORMALIZATION.L2, metric=METRIC.DOT_PRODUCT
        )
        result = self.search(self.transport_returning((2.0, 0.0, 0.0, 0.0)))
        self.assertEqual(self.chunk_ids(result)[0], self.a1.pk)
        self.assertAlmostEqual(result.matches[0].metric_value, 0.8, places=6)

    def test_the_shared_module_does_no_io_and_holds_no_policy(self):
        tree = ast.parse(inspect.getsource(embedding_vector))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
                imported.update(alias.name for alias in node.names)
        for forbidden in (
            "requests", "transaction", "urllib", "socket", "os",
            "resolve_embedding_access", "resolve_embedding_contract",
        ):
            self.assertNotIn(forbidden, imported)


# ---------------------------------------------------------------------------
# Read-only, and content-free
# ---------------------------------------------------------------------------

class ReadOnlyDisciplineTests(RetrievalFixtureMixin, TestCase):
    def test_retrieval_opens_no_transaction_and_takes_no_lock(self):
        """Ranking must never hold a database lock across a network call."""
        tree = ast.parse(inspect.getsource(semantic_retrieval))
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        for forbidden in (
            "atomic", "select_for_update", "save", "delete",
            "bulk_create", "get_or_create", "update_or_create",
        ):
            self.assertNotIn(forbidden, attributes)

    def test_retrieval_never_imports_the_corpus_payload_kind(self):
        tree = ast.parse(inspect.getsource(semantic_retrieval))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
        self.assertIn("PAYLOAD_QUERY", imported)
        self.assertNotIn("PAYLOAD_CORPUS", imported)

    def test_retrieval_never_inspects_a_url_or_a_hostname(self):
        tree = ast.parse(inspect.getsource(semantic_retrieval))
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for forbidden in ("base_url", "hostname", "urlparse", "gethostbyname"):
            self.assertNotIn(forbidden, attributes | names)

    def test_a_search_writes_nothing(self):
        self.build_corpus()
        before = {
            model.__name__: model.objects.count()
            for model in (
                KnowledgeChunkEmbedding, KnowledgeDocumentChunk,
                KnowledgeDocument, KnowledgeCollection,
            )
        }
        self.search()
        after = {
            model.__name__: model.objects.count()
            for model in (
                KnowledgeChunkEmbedding, KnowledgeDocumentChunk,
                KnowledgeDocument, KnowledgeCollection,
            )
        }
        self.assertEqual(before, after)

    def test_a_match_carries_identifiers_and_a_score_never_content(self):
        self.build_corpus()
        result = self.search()
        blob = json.dumps(
            [match.__dict__ for match in result.matches], default=str
        )
        for secret in (A_SECRET, B_SECRET):
            self.assertNotIn(secret, blob)
        self.assertNotIn("Alpha", blob)
        self.assertEqual(
            set(result.matches[0].__dict__),
            {
                "rank", "chunk_id", "document_id", "collection_id",
                "application_scope_id", "k1", "e1", "metric",
                "metric_value", "higher_is_better",
            },
        )

    def test_the_result_carries_no_query_and_no_query_vector(self):
        self.build_corpus()
        result = self.search(query=QUERY_SECRET)
        blob = json.dumps(result.__dict__, default=str)
        self.assertNotIn(QUERY_SECRET, blob)
        self.assertNotIn("query_vector", result.__dict__)

    def test_no_refusal_message_echoes_the_query_or_the_corpus(self):
        self.build_corpus(grant_a=None)
        transport = self.transport_returning((1.0, 0.0, 0.0, 0.0))
        with self.assertRaises(SemanticRetrievalError) as raised:
            self.search(transport, query=QUERY_SECRET)
        message = str(raised.exception)
        self.assertNotIn(QUERY_SECRET, message)
        self.assertNotIn(A_SECRET, message)

    def test_every_failure_category_is_a_bounded_string(self):
        categories = [
            value for name, value in vars(RetrievalFailureCategory).items()
            if not name.startswith("_")
        ]
        self.assertTrue(categories)
        for category in categories:
            with self.subTest(category=category):
                self.assertIsInstance(category, str)
                self.assertRegex(category, r"^[a-z0-9_]+$")

    def test_the_public_entry_point_resolves_the_scope_exactly_once(self):
        """One public call, one authorization answer.

        Load-bearing for S-22: a composing caller must be able to resolve once
        and hand the frozen answer to several branches. If the public wrapper
        resolved more than once, "one operation, one authorization" would
        already be false before hybrid fusion existed.
        """
        self.build_corpus()
        real = semantic_retrieval.resolve_effective_knowledge_scope
        calls = []

        def spy(agent, *, workspace=None):
            calls.append((getattr(agent, "pk", None), workspace))
            return real(agent, workspace=workspace)

        with mock.patch.object(
            semantic_retrieval, "resolve_effective_knowledge_scope", spy
        ):
            with mock.patch(
                TRANSPORT_PATH,
                return_value=self.transport_returning((1.0, 0.0, 0.0, 0.0)),
            ):
                semantic_search_knowledge_local(
                    self.agent_a, query="find alpha",
                    embedding_model_config=self.config,
                )
        self.assertEqual(len(calls), 1)

    def test_the_internal_entry_point_never_resolves_a_scope(self):
        """It consumes a frozen scope; it must never mint a second one."""
        self.build_corpus()
        scope = resolve_effective_knowledge_scope(self.agent_a)
        with mock.patch.object(
            semantic_retrieval,
            "resolve_effective_knowledge_scope",
            mock.Mock(side_effect=AssertionError("must not resolve again")),
        ):
            with mock.patch(
                TRANSPORT_PATH,
                return_value=self.transport_returning((1.0, 0.0, 0.0, 0.0)),
            ):
                result = search_semantic_with_scope(
                    scope, query="find alpha",
                    embedding_model_config=self.config,
                )
        self.assertEqual(self.chunk_ids(result)[0], self.a1.pk)

        source = inspect.getsource(search_semantic_with_scope)
        called = {
            node.func.id
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("resolve_effective_knowledge_scope", called)

    def test_the_internal_signature_takes_no_agent_and_no_workspace(self):
        """Authorization enters as a resolved scope, never as raw principals."""
        parameters = set(
            inspect.signature(search_semantic_with_scope).parameters
        )
        self.assertEqual(
            parameters,
            {"scope", "query", "embedding_model_config", "collection_id", "limit"},
        )

    def test_the_public_api_takes_no_authorization_facts_from_the_caller(self):
        parameters = set(
            inspect.signature(semantic_search_knowledge_local).parameters
        )
        self.assertEqual(
            parameters,
            {
                "agent", "query", "embedding_model_config",
                "workspace", "collection_id", "limit",
            },
        )
        for forbidden in ("scope", "collection_ids", "e1", "metric", "vector"):
            self.assertNotIn(forbidden, parameters)


# ---------------------------------------------------------------------------
# No schema change
# ---------------------------------------------------------------------------

class SemanticRetrievalOwnsNoSchemaTests(TestCase):
    """Semantic retrieval is read-only and defines no persistence of its own.

    Replaces an earlier migration-leaf assertion. That check named `0028`, so it
    was really asserting "no LATER slice has added a migration" - a claim about
    other people's work that failed the moment S-23 legitimately added retrieval
    audit tables. The leaf assertion now lives with the slice that owns it; what
    belongs here is that THIS module still persists nothing.
    """

    def test_the_module_defines_no_model_and_writes_nothing(self):
        tree = ast.parse(inspect.getsource(semantic_retrieval))
        classes = [
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        for node in classes:
            bases = {
                base.attr if isinstance(base, ast.Attribute) else
                getattr(base, "id", "")
                for base in node.bases
            }
            self.assertNotIn("Model", bases)

        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        for forbidden in (
            "save", "create", "bulk_create", "get_or_create",
            "update_or_create", "delete", "atomic", "select_for_update",
        ):
            self.assertNotIn(forbidden, attributes)
