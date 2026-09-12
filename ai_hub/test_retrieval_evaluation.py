"""S-25: golden-query evaluation and the pgvector promotion gate.

An evaluation slice has to be tested adversarially, because a broken evaluator
fails in the one direction nobody notices: it says everything is fine.

So the structure is deliberate. The metric mathematics is hand-verified against
values computed on paper, not against the implementation. The evaluator's
judgement is proved by feeding it deliberately corrupted backend results -
which needs no database, so every adversarial case runs on SQLite. Only the real
exact-vs-ANN parity needs PostgreSQL, and those tests skip locally and say so.

The single most important test here is
`test_controlled_evidence_alone_cannot_satisfy_a_representative_policy`: perfect
synthetic metrics must NOT produce a production-style verdict.
"""

import ast
import dataclasses
import inspect
import json
from decimal import Decimal
from unittest import mock, skipUnless

from django.db import connection
from django.test import TestCase

from ai_hub.models import (
    AgentProfile,
    ApplicationScope,
    EmbeddingModelConfig,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    ProviderConfig,
    RetrievalHit,
    RetrievalOutcome,
    RetrievalRun,
    RetrievalRunCollection,
)
from ai_hub.services import pgvector_ann, retrieval_evaluation, semantic_retrieval
from ai_hub.services.embedding_contract import resolve_embedding_contract
from ai_hub.services.knowledge_authorization import resolve_effective_knowledge_scope
from ai_hub.services.pgvector_ann import (
    PGVECTOR_BACKEND_VERSION,
    PgvectorAnnError,
    PgvectorAnnMatch,
    PgvectorAnnResult,
    PgvectorFailureCategory,
    provision_pgvector_ann_leaf,
    rebuild_pgvector_ann_leaf,
)
from ai_hub.services.retrieval_evaluation import (
    BINARY_RELEVANCE_THRESHOLD,
    CONTROLLED_POLICY,
    EVALUATION_CUTOFFS,
    EVALUATION_VERSION,
    PROMOTION_POLICY_VERSION,
    REPRESENTATIVE_POLICY_TEMPLATE,
    SCORE_PARITY_TOLERANCE,
    CaseOutcome,
    EvaluationError,
    EvidenceClass,
    GoldenQueryCase,
    GoldenRelevanceJudgment,
    HardInvariant,
    PromotionDecision,
    PromotionPolicy,
    RefusalKind,
    classify_refusal,
    evaluate_pgvector_candidate,
    evaluate_pgvector_suite,
    evaluation_fingerprint,
    first_divergence_rank,
    is_catastrophic_miss,
    mean_reciprocal_rank,
    ndcg_at_k,
    oracle_recall_at_k,
    precision_at_k,
    prefix_equal_at_k,
    rank_displacements,
    recall_at_k,
    validate_relevance_grade,
)
from ai_hub.services.semantic_retrieval import (
    SemanticMatch,
    SemanticRetrievalError,
    SemanticRetrievalResult,
    rank_semantic_vector_with_scope,
)
from ai_hub.services.vector_store import store_chunk_vector

METRIC = EmbeddingModelConfig.DistanceMetric
NORMALIZATION = EmbeddingModelConfig.Normalization
LOCALITY = ProviderConfig.DeclaredLocality

POSTGRES = connection.vendor == "postgresql"
requires_postgres = skipUnless(
    POSTGRES, "real exact-vs-ANN parity requires PostgreSQL + pgvector"
)

E1_STUB = "e1:sha256:" + ("ab" * 32)
KNOWLEDGE_SECRET = "KNOWLEDGE-SECRET-EVAL-6621"

#: Payload keys that must never appear anywhere in evaluation evidence.
FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "query", "query_text", "canonical_query", "query_values", "query_vector",
    "vector", "vector_bytes", "embedding", "embeddings",
    "content", "section_title", "snippet", "title",
    "metadata", "tags", "curated_text",
    "provider_response", "response_body",
    "api_key", "credential", "token",
})


def schema_keys(value):
    """Field names of OUR dataclasses, recursively. Nothing else.

    Dataclass INSTANCES, tuples, lists and dict values only; everything else is
    an atomic leaf. Deliberately not the `hasattr(value, "__dict__")` walker
    that recursed into `TextChoices` internals and hit `RecursionError` in
    CI #54 - that lesson is reused here rather than relearned.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield field.name
            yield from schema_keys(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from schema_keys(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from schema_keys(child)


# ---------------------------------------------------------------------------
# Metric mathematics, verified against hand calculations
# ---------------------------------------------------------------------------

class PrecisionRecallTests(TestCase):
    def test_precision_at_k_counts_relevant_hits_over_results_returned(self):
        # [a, b, c] with a=3 and c=2 relevant -> 2 of 3.
        self.assertAlmostEqual(
            precision_at_k(["a", "b", "c"], {"a": 3, "c": 2}, 3), 2 / 3, places=12
        )
        self.assertAlmostEqual(
            precision_at_k(["a", "b", "c"], {"a": 3, "c": 2}, 1), 1.0, places=12
        )

    def test_precision_divides_by_results_returned_not_by_k(self):
        """Dividing by K would fault a backend for a small corpus."""
        self.assertAlmostEqual(
            precision_at_k(["a", "b"], {"a": 3, "b": 3}, 10), 1.0, places=12
        )

    def test_precision_of_an_empty_ranking_is_zero_not_nan(self):
        value = precision_at_k([], {"a": 3}, 5)
        self.assertEqual(value, 0.0)
        self.assertEqual(value, value, "never NaN")

    def test_recall_at_k_divides_by_all_relevant_documents(self):
        # a and c relevant; only a is inside the top 2.
        self.assertAlmostEqual(
            recall_at_k(["a", "b"], {"a": 3, "c": 3}, 2), 0.5, places=12
        )
        self.assertAlmostEqual(
            recall_at_k(["a", "b", "c"], {"a": 3, "c": 3}, 3), 1.0, places=12
        )

    def test_recall_with_fewer_relevant_documents_than_k(self):
        self.assertAlmostEqual(
            recall_at_k(["a", "b", "c"], {"a": 3}, 10), 1.0, places=12
        )

    def test_recall_with_no_relevant_documents_is_defined_as_one(self):
        """An explicit convention, asserted rather than implied.

        There is nothing to recall, so the quantity is undefined. Defined as 1.0
        because a backend cannot be faulted for failing to retrieve documents
        that do not exist - and never as NaN, which cannot be aggregated.
        """
        self.assertEqual(recall_at_k(["a"], {"a": 0, "b": 1}, 5), 1.0)
        self.assertEqual(recall_at_k([], {}, 5), 1.0)

    def test_recall_of_an_empty_ranking_with_relevant_documents_is_zero(self):
        self.assertEqual(recall_at_k([], {"a": 3}, 5), 0.0)

    def test_the_binary_threshold_is_explicit(self):
        self.assertEqual(BINARY_RELEVANCE_THRESHOLD, 2)
        # Grade 1 is "marginal" and is NOT relevant for the binary metrics.
        self.assertEqual(precision_at_k(["a"], {"a": 1}, 1), 0.0)
        self.assertEqual(precision_at_k(["a"], {"a": 2}, 1), 1.0)


class MeanReciprocalRankTests(TestCase):
    def test_first_relevant_at_rank_one(self):
        self.assertEqual(mean_reciprocal_rank(["a", "b"], {"a": 3}), 1.0)

    def test_first_relevant_below_rank_one(self):
        self.assertAlmostEqual(
            mean_reciprocal_rank(["x", "y", "a"], {"a": 3}), 1 / 3, places=12
        )

    def test_no_relevant_result_is_zero(self):
        self.assertEqual(mean_reciprocal_rank(["x", "y"], {"a": 3}), 0.0)
        self.assertEqual(mean_reciprocal_rank([], {"a": 3}), 0.0)


class NdcgTests(TestCase):
    def test_perfect_ordering_is_one(self):
        self.assertAlmostEqual(
            ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, 2), 1.0, places=12
        )

    def test_reversed_graded_ordering_is_below_one(self):
        """Hand-calculated, so the test does not merely echo the code.

        grades a=3, b=1.  gain = 2**g - 1, discount = log2(rank + 1)
        DCG([b, a]) = 1/log2(2) + 7/log2(3) = 1 + 4.4165083 = 5.4165083
        IDCG        = 7/log2(2) + 1/log2(3) = 7 + 0.6309298 = 7.6309298
        ratio       = 0.7098097
        """
        self.assertAlmostEqual(
            ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, 2), 0.7098097, places=7
        )
        self.assertLess(ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, 2), 1.0)

    def test_ndcg_uses_graded_relevance_not_the_binary_threshold(self):
        """A grade-1 document contributes gain even though it is not "relevant"."""
        self.assertGreater(ndcg_at_k(["b"], {"a": 3, "b": 1}, 1), 0.0)

    def test_no_positive_gain_is_defined_as_one(self):
        self.assertEqual(ndcg_at_k(["a"], {"a": 0}, 1), 1.0)
        self.assertEqual(ndcg_at_k([], {}, 5), 1.0)

    def test_ndcg_never_emits_nan_or_infinity(self):
        for ranking, judgments, k in (
            ([], {}, 1), (["a"], {"a": 0}, 5), (["a", "b"], {"a": 3, "b": 3}, 2),
        ):
            value = ndcg_at_k(ranking, judgments, k)
            with self.subTest(ranking=ranking):
                self.assertEqual(value, value)
                self.assertLess(abs(value), float("inf"))


class RelevanceGradeTests(TestCase):
    def test_valid_grades_are_accepted(self):
        for grade in (0, 1, 2, 3):
            self.assertEqual(validate_relevance_grade(grade), grade)

    def test_invalid_grades_are_refused_never_coerced(self):
        """Coercing to 0 would relabel a judgment as "not relevant" silently."""
        for bad in (-1, 4, 2.5, "3", True, None):
            with self.subTest(grade=bad):
                with self.assertRaises(EvaluationError):
                    validate_relevance_grade(bad)

    def test_a_judgment_validates_its_grade_on_construction(self):
        GoldenRelevanceJudgment(chunk_id=1, grade=3)
        with self.assertRaises(EvaluationError):
            GoldenRelevanceJudgment(chunk_id=1, grade=9)


# ---------------------------------------------------------------------------
# Oracle parity mathematics
# ---------------------------------------------------------------------------

class OracleParityMathTests(TestCase):
    def test_oracle_recall_denominator_is_the_exact_result_count(self):
        """Dividing by K would report 0.3 for a perfect reproduction."""
        self.assertEqual(oracle_recall_at_k(["a", "b", "c"], ["a", "b", "c"], 10), 1.0)
        self.assertAlmostEqual(
            oracle_recall_at_k(["a"], ["a", "b", "c"], 10), 1 / 3, places=12
        )

    def test_oracle_recall_both_empty_is_perfect_agreement(self):
        self.assertEqual(oracle_recall_at_k([], [], 5), 1.0)

    def test_oracle_recall_ann_empty_against_a_non_empty_oracle_is_zero(self):
        self.assertEqual(oracle_recall_at_k([], ["a"], 5), 0.0)

    def test_prefix_equality_is_ordered_and_stricter_than_overlap(self):
        self.assertTrue(prefix_equal_at_k(["a", "b"], ["a", "b"], 2))
        self.assertFalse(prefix_equal_at_k(["b", "a"], ["a", "b"], 2))
        # Same set, different order: full overlap yet not prefix-equal.
        self.assertEqual(oracle_recall_at_k(["b", "a"], ["a", "b"], 2), 1.0)

    def test_first_divergence_rank(self):
        self.assertIsNone(first_divergence_rank(["a", "b"], ["a", "b"]))
        self.assertEqual(first_divergence_rank(["a", "c", "b"], ["a", "b", "c"]), 2)
        self.assertEqual(first_divergence_rank(["a"], ["a", "b"]), 2)
        self.assertEqual(first_divergence_rank([], ["a"]), 1)

    def test_rank_displacement_excludes_missing_results(self):
        """A fabricated rank for an absent result would average into the metric."""
        displacements = rank_displacements(["a", "c"], ["a", "b", "c"])
        self.assertEqual(displacements, {"a": 0, "c": 1})
        self.assertNotIn("b", displacements)

    def test_catastrophic_miss_positive_and_negative(self):
        self.assertTrue(is_catastrophic_miss(["z"], ["a", "b"], 2))
        self.assertFalse(is_catastrophic_miss(["b"], ["a", "b"], 2))
        self.assertFalse(is_catastrophic_miss([], [], 5), "nothing to miss")


# ---------------------------------------------------------------------------
# Injected-result evaluation: the evaluator's own judgement, provable on SQLite
# ---------------------------------------------------------------------------

class InjectedBackendMixin:
    """Feed the evaluator constructed backend results.

    No database and no PostgreSQL needed, which is the point: the adversarial
    cases that decide whether the evaluator can be trusted all run locally.
    """

    SCOPE_ID = 3
    COLLECTIONS = (9, 11)

    class _Scope:
        is_empty = False
        application_scope_id = 3
        agent_id = 5
        workspace_id = None
        collection_ids = frozenset({9, 11})

    class _Config:
        pk = 21

    def exact_result(self, ids, *, scores=None, collection_ids=None, scope_id=None):
        scores = scores or {cid: 1.0 - index * 0.1 for index, cid in enumerate(ids)}
        return SemanticRetrievalResult(
            matches=tuple(
                SemanticMatch(
                    rank=position, chunk_id=cid, document_id=cid * 10,
                    collection_id=(collection_ids or self.COLLECTIONS)[0],
                    application_scope_id=scope_id or self.SCOPE_ID,
                    k1="k1:sha256:" + "0" * 64, e1=E1_STUB,
                    metric=METRIC.COSINE, metric_value=scores[cid],
                    higher_is_better=True,
                )
                for position, cid in enumerate(ids, start=1)
            ),
            application_scope_id=scope_id or self.SCOPE_ID, agent_id=5,
            workspace_id=None,
            collection_ids=tuple(collection_ids or self.COLLECTIONS),
            embedding_model_config_id=21, e1=E1_STUB, metric=METRIC.COSINE,
            higher_is_better=True, candidate_count=len(ids),
            scored_count=len(ids), provider_invoked=False,
        )

    def ann_result(
        self, ids, *, scores=None, collection_ids=None, scope_id=None,
        e1=E1_STUB, metric=METRIC.COSINE, backend_version=None,
        match_scope_id=None, match_collection_id=None,
    ):
        scores = scores or {cid: 1.0 - index * 0.1 for index, cid in enumerate(ids)}
        collections = tuple(collection_ids or self.COLLECTIONS)
        return PgvectorAnnResult(
            matches=tuple(
                PgvectorAnnMatch(
                    rank=position, chunk_id=cid, document_id=cid * 10,
                    collection_id=(
                        match_collection_id if match_collection_id is not None
                        else collections[0]
                    ),
                    application_scope_id=(
                        match_scope_id if match_scope_id is not None
                        else (scope_id or self.SCOPE_ID)
                    ),
                    k1="k1:sha256:" + "0" * 64, e1=e1, metric=metric,
                    metric_value=scores[cid],
                )
                for position, cid in enumerate(ids, start=1)
            ),
            application_scope_id=scope_id or self.SCOPE_ID,
            collection_ids=collections, embedding_model_config_id=21, e1=e1,
            backend_version=backend_version or PGVECTOR_BACKEND_VERSION,
            metric=metric, ann_candidate_pool=100,
            ann_candidates_returned=len(ids),
        )

    def evaluate(
        self, *, exact, ann, case=None, probe=None, exact_error=None,
        ann_error=None,
    ):
        case = case or self.case()

        def fake_exact(*args, **kwargs):
            if exact_error:
                raise exact_error
            return exact

        def fake_ann(*args, **kwargs):
            if ann_error:
                raise ann_error
            return ann

        with mock.patch.object(
            retrieval_evaluation, "rank_semantic_vector_with_scope", fake_exact
        ):
            with mock.patch.object(
                retrieval_evaluation, "search_pgvector_ann_with_scope", fake_ann
            ):
                return evaluate_pgvector_candidate(
                    self._Scope(), case,
                    embedding_model_config=self._Config(),
                    source_generation_probe=probe,
                )

    def suite(self, cases_and_results, *, policy=None, probe=None):
        """Evaluate several injected cases under one policy."""
        results = list(cases_and_results)
        state = {"index": 0}

        def fake_exact(*args, **kwargs):
            return results[state["index"]][1]

        def fake_ann(*args, **kwargs):
            pair = results[state["index"]]
            state["index"] += 1
            return pair[2]

        with mock.patch.object(
            retrieval_evaluation, "rank_semantic_vector_with_scope", fake_exact
        ):
            with mock.patch.object(
                retrieval_evaluation, "search_pgvector_ann_with_scope", fake_ann
            ):
                return evaluate_pgvector_suite(
                    self._Scope(), [pair[0] for pair in results],
                    embedding_model_config=self._Config(),
                    policy=policy, source_generation_probe=probe,
                )

    def case(self, *, case_id="case-001", judgments=(), limit=5,
             evidence_class=EvidenceClass.CONTROLLED):
        return GoldenQueryCase(
            case_id=case_id, query_values=(1.0, 0.0, 0.0, 0.0),
            evidence_class=evidence_class, limit=limit, judgments=judgments,
        )


class EvaluatorJudgementTests(InjectedBackendMixin, TestCase):
    def test_identical_rankings_compare_cleanly(self):
        result = self.evaluate(
            exact=self.exact_result([1, 2, 3]), ann=self.ann_result([1, 2, 3])
        )
        self.assertEqual(result.outcome, CaseOutcome.COMPARED)
        self.assertEqual(result.hard_invariants, ())
        self.assertIsNone(result.parity.first_divergence_rank)
        for entry in result.parity.cutoffs:
            with self.subTest(cutoff=entry.cutoff):
                self.assertEqual(entry.oracle_recall, 1.0)
                self.assertTrue(entry.prefix_equal)
                self.assertFalse(entry.catastrophic_miss)

    # -- non-vacuity: each corruption must be detected -------------------

    def test_a_removed_top_result_lowers_oracle_recall(self):
        result = self.evaluate(
            exact=self.exact_result([1, 2, 3]), ann=self.ann_result([2, 3])
        )
        at_three = next(e for e in result.parity.cutoffs if e.cutoff == 3)
        self.assertAlmostEqual(at_three.oracle_recall, 2 / 3, places=12)
        self.assertFalse(at_three.prefix_equal)
        self.assertEqual(result.parity.first_divergence_rank, 1)

    def test_two_swapped_ranks_break_prefix_equality_but_not_recall(self):
        result = self.evaluate(
            exact=self.exact_result([1, 2, 3]),
            ann=self.ann_result(
                [2, 1, 3], scores={2: 1.0, 1: 0.9, 3: 0.8}
            ),
        )
        at_three = next(e for e in result.parity.cutoffs if e.cutoff == 3)
        self.assertEqual(at_three.oracle_recall, 1.0, "same set")
        self.assertFalse(at_three.prefix_equal, "different order")
        self.assertEqual(result.parity.first_divergence_rank, 1)
        self.assertEqual(result.parity.max_rank_displacement, 1)

    def test_an_injected_foreign_chunk_is_a_hard_invariant_failure(self):
        """Not a precision problem. A result from outside the namespace."""
        result = self.evaluate(
            exact=self.exact_result([1, 2]),
            ann=self.ann_result([1, 2], match_scope_id=999),
        )
        self.assertIn(HardInvariant.FOREIGN_RESULT, result.hard_invariants)
        self.assertEqual(result.outcome, CaseOutcome.INVARIANT_VIOLATION)

    def test_a_chunk_from_an_unauthorized_collection_is_a_hard_failure(self):
        result = self.evaluate(
            exact=self.exact_result([1, 2]),
            ann=self.ann_result([1, 2], match_collection_id=777),
        )
        self.assertIn(HardInvariant.FOREIGN_RESULT, result.hard_invariants)

    def test_an_altered_semantic_score_is_a_hard_invariant_failure(self):
        """Both backends rerank with the SAME scorers; disagreement is a defect."""
        result = self.evaluate(
            exact=self.exact_result([1, 2], scores={1: 1.0, 2: 0.9}),
            ann=self.ann_result([1, 2], scores={1: 1.0, 2: 0.5}),
        )
        self.assertIn(HardInvariant.SCORE_DISAGREEMENT, result.hard_invariants)
        self.assertGreater(result.parity.score_parity_max_delta, SCORE_PARITY_TOLERANCE)

    def test_a_float_difference_inside_the_tolerance_is_not_a_failure(self):
        result = self.evaluate(
            exact=self.exact_result([1], scores={1: 1.0}),
            ann=self.ann_result([1], scores={1: 1.0 + 1e-12}),
        )
        self.assertEqual(result.hard_invariants, ())

    def test_a_mismatched_e1_is_a_hard_invariant_failure(self):
        result = self.evaluate(
            exact=self.exact_result([1]),
            ann=self.ann_result([1], e1="e1:sha256:" + "cd" * 32),
        )
        self.assertIn(HardInvariant.E1_MISMATCH, result.hard_invariants)

    def test_a_mismatched_metric_is_a_hard_invariant_failure(self):
        result = self.evaluate(
            exact=self.exact_result([1]),
            ann=self.ann_result([1], metric=METRIC.EUCLIDEAN),
        )
        self.assertIn(HardInvariant.METRIC_MISMATCH, result.hard_invariants)

    def test_a_mismatched_collection_set_is_a_hard_invariant_failure(self):
        result = self.evaluate(
            exact=self.exact_result([1]),
            ann=self.ann_result([1], collection_ids=(9,)),
        )
        self.assertIn(
            HardInvariant.COLLECTION_SET_MISMATCH, result.hard_invariants
        )

    def test_a_different_backend_version_is_a_hard_invariant_failure(self):
        """`pgv-hnsw1` evidence must describe `pgv-hnsw1`."""
        result = self.evaluate(
            exact=self.exact_result([1]),
            ann=self.ann_result([1], backend_version="pgv-hnsw2"),
        )
        self.assertIn(
            HardInvariant.BACKEND_VERSION_MISMATCH, result.hard_invariants
        )

    def test_a_mismatched_application_scope_is_a_hard_invariant_failure(self):
        result = self.evaluate(
            exact=self.exact_result([1]),
            ann=self.ann_result([1], scope_id=42),
        )
        self.assertIn(
            HardInvariant.APPLICATION_SCOPE_MISMATCH, result.hard_invariants
        )

    # -- refusal semantics ------------------------------------------------

    def test_one_backend_answering_where_the_other_refuses_is_a_hard_failure(self):
        """Deliberately NOT scored as Recall@K = 0."""
        result = self.evaluate(
            exact=self.exact_result([1]), ann=None,
            ann_error=PgvectorAnnError(
                PgvectorFailureCategory.LEAF_NOT_READY, "not ready"
            ),
        )
        self.assertEqual(result.outcome, CaseOutcome.REFUSAL_MISMATCH)
        self.assertIn(
            HardInvariant.REFUSAL_SEMANTICS_MISMATCH, result.hard_invariants
        )
        self.assertIsNone(result.parity)

    def test_equivalent_refusals_on_both_sides_are_parity_not_failure(self):
        result = self.evaluate(
            exact=None, ann=None,
            exact_error=SemanticRetrievalError("unscorable_zero_vector", "x"),
            ann_error=PgvectorAnnError("unscorable_zero_vector", "x"),
        )
        self.assertEqual(result.outcome, CaseOutcome.BOTH_REFUSED)
        self.assertEqual(result.hard_invariants, ())
        self.assertEqual(result.exact_refusal, RefusalKind.UNSCORABLE_QUERY)
        self.assertEqual(result.ann_refusal, RefusalKind.UNSCORABLE_QUERY)

    def test_differing_refusal_kinds_are_a_hard_failure(self):
        result = self.evaluate(
            exact=None, ann=None,
            exact_error=SemanticRetrievalError("unscorable_zero_vector", "x"),
            ann_error=PgvectorAnnError(
                PgvectorFailureCategory.LEAF_NOT_READY, "x"
            ),
        )
        self.assertEqual(result.outcome, CaseOutcome.REFUSAL_MISMATCH)

    def test_both_empty_is_a_legitimate_answer(self):
        result = self.evaluate(
            exact=self.exact_result([]), ann=self.ann_result([])
        )
        self.assertEqual(result.outcome, CaseOutcome.BOTH_EMPTY)
        self.assertEqual(result.hard_invariants, ())
        for entry in result.parity.cutoffs:
            self.assertEqual(entry.oracle_recall, 1.0)
            self.assertFalse(entry.catastrophic_miss)

    def test_an_unexpected_exception_propagates(self):
        """A programming error is not an evaluation datum."""
        with self.assertRaises(RuntimeError):
            self.evaluate(
                exact=None, ann=None, exact_error=RuntimeError("a real bug")
            )

    def test_refusal_classification_never_uses_exception_text(self):
        exc = SemanticRetrievalError("unscorable_zero_vector", KNOWLEDGE_SECRET)
        self.assertEqual(classify_refusal(exc), RefusalKind.UNSCORABLE_QUERY)
        self.assertNotIn(KNOWLEDGE_SECRET, classify_refusal(exc))
        self.assertEqual(
            classify_refusal(SemanticRetrievalError("future_category", "x")),
            RefusalKind.OTHER,
        )

    # -- frozen corpus ----------------------------------------------------

    def test_a_source_generation_change_invalidates_the_case(self):
        """One evaluation attempt is one coherent corpus state."""
        readings = iter([{9: 1}, {9: 2}])
        result = self.evaluate(
            exact=self.exact_result([1]), ann=self.ann_result([1]),
            probe=lambda: next(readings),
        )
        self.assertEqual(result.outcome, CaseOutcome.SOURCE_CHANGED)
        self.assertIn(
            HardInvariant.SOURCE_CHANGED_DURING_EVALUATION,
            result.hard_invariants,
        )
        self.assertIsNone(result.parity, "never averaged into quality metrics")

    def test_a_stable_generation_does_not_invalidate(self):
        result = self.evaluate(
            exact=self.exact_result([1]), ann=self.ann_result([1]),
            probe=lambda: {9: 7},
        )
        self.assertEqual(result.outcome, CaseOutcome.COMPARED)

    def test_the_evaluator_never_retries(self):
        calls = []

        def probe():
            calls.append(1)
            return {9: len(calls)}

        self.evaluate(
            exact=self.exact_result([1]), ann=self.ann_result([1]), probe=probe
        )
        self.assertEqual(len(calls), 2, "read once before and once after; no retry")

    # -- relevance --------------------------------------------------------

    def test_a_changed_relevance_grade_changes_the_metrics(self):
        judged = (
            GoldenRelevanceJudgment(chunk_id=1, grade=3),
            GoldenRelevanceJudgment(chunk_id=2, grade=3),
        )
        demoted = (
            GoldenRelevanceJudgment(chunk_id=1, grade=3),
            GoldenRelevanceJudgment(chunk_id=2, grade=0),
        )
        high = self.evaluate(
            exact=self.exact_result([1, 2]), ann=self.ann_result([1, 2]),
            case=self.case(judgments=judged),
        )
        low = self.evaluate(
            exact=self.exact_result([1, 2]), ann=self.ann_result([1, 2]),
            case=self.case(judgments=demoted),
        )
        high_p = next(e for e in high.relevance.cutoffs if e.cutoff == 3).precision
        low_p = next(e for e in low.relevance.cutoffs if e.cutoff == 3).precision
        self.assertGreater(high_p, low_p)

    def test_a_disappeared_expected_relevant_chunk_lowers_recall(self):
        judged = (
            GoldenRelevanceJudgment(chunk_id=1, grade=3),
            GoldenRelevanceJudgment(chunk_id=2, grade=3),
        )
        result = self.evaluate(
            exact=self.exact_result([1]), ann=self.ann_result([1]),
            case=self.case(judgments=judged),
        )
        at_three = next(e for e in result.relevance.cutoffs if e.cutoff == 3)
        self.assertAlmostEqual(at_three.recall, 0.5, places=12)

    def test_cutoffs_larger_than_the_requested_limit_are_skipped(self):
        """Recall@10 over a top-3 request would measure the request."""
        result = self.evaluate(
            exact=self.exact_result([1, 2, 3]), ann=self.ann_result([1, 2, 3]),
            case=self.case(limit=3),
        )
        self.assertEqual(
            [entry.cutoff for entry in result.parity.cutoffs], [1, 3]
        )


# ---------------------------------------------------------------------------
# Promotion policy
# ---------------------------------------------------------------------------

class PromotionPolicyTests(InjectedBackendMixin, TestCase):
    def perfect_cases(self, count, *, evidence_class, judged=True):
        judgments = (
            (GoldenRelevanceJudgment(chunk_id=1, grade=3),) if judged else ()
        )
        return [
            (
                self.case(
                    case_id=f"case-{index:03d}", judgments=judgments,
                    evidence_class=evidence_class,
                ),
                self.exact_result([1, 2, 3]),
                self.ann_result([1, 2, 3]),
            )
            for index in range(count)
        ]

    def test_a_perfect_controlled_suite_is_eligible_under_the_controlled_policy(self):
        summary = self.suite(
            self.perfect_cases(3, evidence_class=EvidenceClass.CONTROLLED),
            policy=CONTROLLED_POLICY,
        )
        self.assertEqual(summary.decision, PromotionDecision.ELIGIBLE)
        self.assertEqual(summary.decision_reasons, ())
        self.assertEqual(summary.hard_invariant_failures, ())

    def test_controlled_evidence_alone_cannot_satisfy_a_representative_policy(self):
        """THE test this slice exists for.

        Every metric is perfect. The verdict must still be
        `insufficient_evidence`, because a deterministic fixture built to make
        the answer unambiguous says nothing whatsoever about a real corpus.
        """
        summary = self.suite(
            self.perfect_cases(100, evidence_class=EvidenceClass.CONTROLLED),
            policy=REPRESENTATIVE_POLICY_TEMPLATE,
        )
        self.assertEqual(summary.mean_oracle_recall[5], 1.0)
        self.assertEqual(summary.mean_ndcg[5], 1.0)
        self.assertEqual(summary.mean_mrr, 1.0)
        self.assertEqual(summary.catastrophic_miss_count[5], 0)

        self.assertEqual(
            summary.decision, PromotionDecision.INSUFFICIENT_EVIDENCE,
            "perfect synthetic metrics must never read as production quality",
        )
        self.assertNotEqual(summary.decision, PromotionDecision.ELIGIBLE)
        self.assertIn(
            "mixed_or_insufficient_evidence_class", summary.decision_reasons
        )
        self.assertEqual(summary.controlled_case_count, 100)
        self.assertEqual(summary.representative_case_count, 0)

    def test_too_few_representative_cases_is_insufficient_evidence(self):
        summary = self.suite(
            self.perfect_cases(2, evidence_class=EvidenceClass.REPRESENTATIVE),
            policy=REPRESENTATIVE_POLICY_TEMPLATE,
        )
        self.assertEqual(
            summary.decision, PromotionDecision.INSUFFICIENT_EVIDENCE
        )
        self.assertIn("insufficient_case_count", summary.decision_reasons)

    def test_a_single_hard_invariant_failure_overrides_perfect_averages(self):
        """No average may hide a structural failure."""
        cases = self.perfect_cases(20, evidence_class=EvidenceClass.CONTROLLED)
        cases[7] = (
            cases[7][0], self.exact_result([1, 2, 3]),
            self.ann_result([1, 2, 3], match_scope_id=999),
        )
        summary = self.suite(cases, policy=CONTROLLED_POLICY)
        self.assertEqual(summary.decision, PromotionDecision.NOT_ELIGIBLE)
        self.assertIn(HardInvariant.FOREIGN_RESULT, summary.hard_invariant_failures)
        self.assertGreaterEqual(summary.mean_oracle_recall[3], 0.9)

    def test_one_catastrophic_miss_overrides_a_high_average(self):
        cases = self.perfect_cases(20, evidence_class=EvidenceClass.CONTROLLED)
        cases[3] = (
            cases[3][0], self.exact_result([1, 2, 3]),
            self.ann_result([7, 8, 9]),
        )
        summary = self.suite(cases, policy=CONTROLLED_POLICY)
        self.assertGreater(summary.mean_oracle_recall[3], 0.9)
        self.assertEqual(summary.catastrophic_miss_count[3], 1)
        self.assertEqual(summary.decision, PromotionDecision.NOT_ELIGIBLE)
        self.assertIn("catastrophic_miss", summary.decision_reasons)

    def test_a_source_change_in_one_case_blocks_the_whole_suite(self):
        readings = iter([{9: 1}, {9: 1}, {9: 1}, {9: 5}])
        summary = self.suite(
            self.perfect_cases(2, evidence_class=EvidenceClass.CONTROLLED),
            policy=CONTROLLED_POLICY, probe=lambda: next(readings),
        )
        self.assertEqual(summary.invalid_case_count, 1)
        self.assertEqual(summary.decision, PromotionDecision.NOT_ELIGIBLE)
        self.assertIn(
            HardInvariant.SOURCE_CHANGED_DURING_EVALUATION,
            summary.decision_reasons,
        )

    def test_evidence_completeness_is_reported_not_hidden(self):
        cases = self.perfect_cases(4, evidence_class=EvidenceClass.CONTROLLED)
        cases[1] = (
            cases[1][0], self.exact_result([1]),
            self.ann_result([1], e1="e1:sha256:" + "cd" * 32),
        )
        summary = self.suite(cases, policy=CONTROLLED_POLICY)
        self.assertEqual(summary.requested_case_count, 4)
        self.assertEqual(summary.failed_case_count, 1)
        self.assertEqual(summary.completed_case_count, 3)
        self.assertEqual(
            summary.requested_case_count,
            summary.completed_case_count
            + summary.failed_case_count
            + summary.invalid_case_count,
            "no denominator may omit failures",
        )

    def test_a_case_that_vanishes_from_the_arithmetic_blocks_promotion(self):
        """The defensive completeness guard, exercised directly.

        Every known outcome is accounted for, so this is only reachable if a new
        outcome is added without teaching the summary about it. Injected rather
        than left untested: an unreachable guard nobody has ever run is a guard
        nobody knows works.
        """
        real = retrieval_evaluation.evaluate_pgvector_candidate

        def vanishing(scope, case, **kwargs):
            result = real(scope, case, **kwargs)
            if case.case_id.endswith("001"):
                return dataclasses.replace(result, outcome="a_new_outcome")
            return result

        cases = self.perfect_cases(3, evidence_class=EvidenceClass.CONTROLLED)
        with mock.patch.object(
            retrieval_evaluation, "evaluate_pgvector_candidate", vanishing
        ):
            summary = self.suite(cases, policy=CONTROLLED_POLICY)

        self.assertEqual(summary.decision, PromotionDecision.NOT_ELIGIBLE)
        self.assertIn(
            HardInvariant.INCOMPLETE_EVALUATION, summary.decision_reasons
        )

    def test_a_legitimate_refusal_parity_case_still_counts_as_completed(self):
        """A refused-on-both-sides case is evidence, not a hole in the suite.

        Dropping it from the reported count would understate the denominator
        while every other number looked healthy.
        """
        results = list(
            self.perfect_cases(2, evidence_class=EvidenceClass.CONTROLLED)
        )
        state = {"index": 0}

        def fake_exact(*args, **kwargs):
            if state["index"] == 0:
                raise SemanticRetrievalError("unscorable_zero_vector", "x")
            return results[state["index"]][1]

        def fake_ann(*args, **kwargs):
            index = state["index"]
            state["index"] += 1
            if index == 0:
                raise PgvectorAnnError("unscorable_zero_vector", "x")
            return results[index][2]

        with mock.patch.object(
            retrieval_evaluation, "rank_semantic_vector_with_scope", fake_exact
        ):
            with mock.patch.object(
                retrieval_evaluation, "search_pgvector_ann_with_scope", fake_ann
            ):
                summary = evaluate_pgvector_suite(
                    self._Scope(), [pair[0] for pair in results],
                    embedding_model_config=self._Config(),
                    policy=CONTROLLED_POLICY,
                )

        self.assertEqual(summary.requested_case_count, 2)
        self.assertEqual(
            summary.completed_case_count, 2,
            "the refused-parity case is accounted for, not silently dropped",
        )
        self.assertEqual(summary.failed_case_count, 0)
        self.assertEqual(summary.invalid_case_count, 0)

    def test_the_reported_counts_always_account_for_every_case(self):
        cases = self.perfect_cases(5, evidence_class=EvidenceClass.CONTROLLED)
        cases[2] = (
            cases[2][0], self.exact_result([1]),
            self.ann_result([1], metric=METRIC.EUCLIDEAN),
        )
        summary = self.suite(cases, policy=CONTROLLED_POLICY)
        self.assertEqual(
            summary.completed_case_count
            + summary.failed_case_count
            + summary.invalid_case_count,
            summary.requested_case_count,
        )

    def test_thresholds_live_in_one_versioned_policy_object(self):
        self.assertEqual(PROMOTION_POLICY_VERSION, "pgv-promotion1")
        self.assertEqual(CONTROLLED_POLICY.version, PROMOTION_POLICY_VERSION)
        self.assertEqual(
            CONTROLLED_POLICY.required_evidence_class, EvidenceClass.CONTROLLED
        )
        self.assertEqual(
            REPRESENTATIVE_POLICY_TEMPLATE.required_evidence_class,
            EvidenceClass.REPRESENTATIVE,
        )
        self.assertEqual(CONTROLLED_POLICY.maximum_catastrophic_misses, 0)

    def test_the_controlled_policy_demands_exact_reproduction(self):
        """A small deterministic corpus must reproduce exactly, or investigate."""
        for cutoff in (1, 3, 5):
            self.assertEqual(
                CONTROLLED_POLICY.minimum_mean_oracle_recall[cutoff], 1.0
            )
            self.assertEqual(
                CONTROLLED_POLICY.minimum_prefix_match_rate[cutoff], 1.0
            )

    def test_the_three_decision_states_exist(self):
        self.assertEqual(
            {
                PromotionDecision.ELIGIBLE,
                PromotionDecision.NOT_ELIGIBLE,
                PromotionDecision.INSUFFICIENT_EVIDENCE,
            },
            {"eligible", "not_eligible", "insufficient_evidence"},
        )


# ---------------------------------------------------------------------------
# Determinism, fingerprint and schema privacy
# ---------------------------------------------------------------------------

class DeterminismAndPrivacyTests(InjectedBackendMixin, TestCase):
    def test_the_summary_is_deterministic(self):
        cases = [
            (
                self.case(case_id=f"case-{i:03d}"),
                self.exact_result([1, 2, 3]),
                self.ann_result([1, 2, 3]),
            )
            for i in range(5)
        ]
        first = self.suite(cases, policy=CONTROLLED_POLICY)
        second = self.suite(cases, policy=CONTROLLED_POLICY)
        self.assertEqual(first.decision, second.decision)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.mean_oracle_recall, second.mean_oracle_recall)
        self.assertEqual(
            [c.case_id for c in first.cases], [c.case_id for c in second.cases]
        )

    def test_no_timestamp_appears_in_equality_sensitive_evidence(self):
        summary = self.suite(
            [(self.case(), self.exact_result([1]), self.ann_result([1]))],
            policy=CONTROLLED_POLICY,
        )
        keys = set(schema_keys(summary))
        for forbidden in ("timestamp", "created_at", "started_at", "finished_at",
                          "evaluated_at", "now"):
            self.assertNotIn(forbidden, keys)

    def test_the_fingerprint_is_content_free_and_structural(self):
        one = evaluation_fingerprint(
            policy=CONTROLLED_POLICY, backend_version=PGVECTOR_BACKEND_VERSION,
            e1=E1_STUB, metric=METRIC.COSINE,
            evidence_class=EvidenceClass.CONTROLLED,
        )
        self.assertTrue(one.startswith("eval1:sha256:"))
        self.assertEqual(len(one), len("eval1:sha256:") + 64)
        # Every structural input changes it - including the policy THRESHOLDS,
        # not merely its version string. Both policies share `pgv-promotion1`
        # while demanding very different things; a fingerprint that collapsed
        # them would claim two incomparable evaluations were the same
        # configuration. (The first draft did exactly that.)
        self.assertEqual(
            CONTROLLED_POLICY.version,
            REPRESENTATIVE_POLICY_TEMPLATE.version,
            "same version string, different thresholds",
        )
        self.assertNotEqual(
            one,
            evaluation_fingerprint(
                policy=REPRESENTATIVE_POLICY_TEMPLATE,
                backend_version=PGVECTOR_BACKEND_VERSION, e1=E1_STUB,
                metric=METRIC.COSINE, evidence_class=EvidenceClass.CONTROLLED,
            ),
        )
        self.assertNotEqual(
            one,
            evaluation_fingerprint(
                policy=dataclasses.replace(
                    CONTROLLED_POLICY, maximum_catastrophic_misses=1
                ),
                backend_version=PGVECTOR_BACKEND_VERSION, e1=E1_STUB,
                metric=METRIC.COSINE, evidence_class=EvidenceClass.CONTROLLED,
            ),
            "a relaxed threshold must produce a different fingerprint",
        )
        self.assertNotEqual(
            one,
            evaluation_fingerprint(
                policy=CONTROLLED_POLICY,
                backend_version=PGVECTOR_BACKEND_VERSION, e1=E1_STUB,
                metric=METRIC.EUCLIDEAN,
                evidence_class=EvidenceClass.CONTROLLED,
            ),
        )

    def test_the_fingerprint_source_takes_no_query_or_content(self):
        parameters = set(inspect.signature(evaluation_fingerprint).parameters)
        self.assertEqual(
            parameters,
            {"policy", "backend_version", "e1", "metric", "evidence_class"},
        )

    def test_no_payload_key_appears_anywhere_in_the_evidence(self):
        summary = self.suite(
            [(
                self.case(
                    judgments=(GoldenRelevanceJudgment(chunk_id=1, grade=3),)
                ),
                self.exact_result([1, 2]),
                self.ann_result([1, 2]),
            )],
            policy=CONTROLLED_POLICY,
        )
        observed = set(schema_keys(summary))
        self.assertTrue(observed)
        self.assertEqual(observed & FORBIDDEN_PAYLOAD_KEYS, set())

    def test_the_case_result_schema_is_exactly_the_allowed_fields(self):
        result = self.evaluate(
            exact=self.exact_result([1]), ann=self.ann_result([1])
        )
        self.assertEqual(
            set(result.__dict__),
            {
                "case_id", "evidence_class", "outcome", "application_scope_id",
                "collection_ids", "embedding_model_config_id", "e1", "metric",
                "backend_version", "limit", "exact_refusal", "ann_refusal",
                "parity", "relevance", "hard_invariants",
            },
        )

    def test_the_summary_schema_is_exactly_the_allowed_fields(self):
        summary = self.suite(
            [(self.case(), self.exact_result([1]), self.ann_result([1]))],
            policy=CONTROLLED_POLICY,
        )
        self.assertEqual(
            set(summary.__dict__),
            {
                "evaluation_version", "policy_version", "backend_version",
                "fingerprint", "requested_case_count", "completed_case_count",
                "invalid_case_count", "failed_case_count",
                "controlled_case_count", "representative_case_count",
                "mean_oracle_recall", "worst_oracle_recall",
                "prefix_match_rate", "catastrophic_miss_count", "mean_ndcg",
                "mean_mrr", "hard_invariant_failures", "decision",
                "decision_reasons", "cases",
            },
        )

    def test_a_leaked_payload_field_would_be_caught(self):
        """Proves the forbidden-key assertion is not vacuous."""

        @dataclasses.dataclass(frozen=True)
        class Leaky:
            case_id: str
            query_text: str

        observed = set(schema_keys(Leaky("case-001", KNOWLEDGE_SECRET)))
        self.assertIn("query_text", observed)
        self.assertNotEqual(observed & FORBIDDEN_PAYLOAD_KEYS, set())

    def test_the_case_input_carries_no_query_text(self):
        parameters = {
            field.name for field in dataclasses.fields(GoldenQueryCase)
        }
        self.assertEqual(
            parameters,
            {
                "case_id", "query_values", "evidence_class", "collection_id",
                "limit", "judgments",
            },
        )
        for forbidden in ("query", "query_text", "text", "prompt"):
            self.assertNotIn(forbidden, parameters)


# ---------------------------------------------------------------------------
# Structural: what S-25 must not be
# ---------------------------------------------------------------------------

class EvaluatorIsolationTests(TestCase):
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

    def test_there_is_no_second_exact_oracle(self):
        """The baseline must come from S-21, never be reimplemented here."""
        names = self._identifiers(retrieval_evaluation)
        self.assertIn("rank_semantic_vector_with_scope", names)
        for forbidden in (
            "cosine_similarity", "dot_product_similarity", "euclidean_distance",
            "METRIC_SCORERS", "resolve_metric_scorer", "decode_vector",
            "chunk_embedding_fingerprint", "load_current_vectors",
            "authorized_chunks",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_evaluator_never_resolves_authorization(self):
        names = self._identifiers(retrieval_evaluation)
        self.assertNotIn("resolve_effective_knowledge_scope", names)

    def test_the_evaluator_calls_no_llm_or_provider(self):
        names = self._identifiers(retrieval_evaluation)
        for forbidden in (
            "requests", "litellm", "openai", "anthropic", "azure",
            "resolve_embedding_transport", "embed_text_via_ollama",
            "ragas", "langchain", "llama_index", "urllib", "socket",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_evaluator_persists_nothing(self):
        names = self._identifiers(retrieval_evaluation)
        for forbidden in (
            "save", "create", "bulk_create", "objects", "atomic",
            "RetrievalRun", "RetrievalHit", "RetrievalOutcome",
            "RetrievalRunCollection", "models",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_evaluator_adds_no_runtime_backend_selector(self):
        names = self._identifiers(retrieval_evaluation)
        for forbidden in (
            "settings", "getenv", "environ", "USE_PGVECTOR",
            "RETRIEVAL_BACKEND", "select_backend", "fallback", "route",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_evaluator_uses_no_randomness_and_no_clock(self):
        names = self._identifiers(retrieval_evaluation)
        for forbidden in (
            "random", "shuffle", "sample", "uuid4", "now", "timezone",
            "datetime", "time",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_evaluator_absorbs_only_bounded_backend_errors(self):
        self.assertEqual(
            retrieval_evaluation.EXPECTED_BACKEND_ERRORS,
            (SemanticRetrievalError, PgvectorAnnError),
        )
        tree = ast.parse(
            inspect.getsource(evaluate_pgvector_candidate).lstrip()
        )
        handlers = [
            node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)
        ]
        self.assertTrue(handlers)
        for handler in handlers:
            with self.subTest(lineno=handler.lineno):
                self.assertIsNotNone(handler.type)
                caught = set()
                if isinstance(handler.type, ast.Tuple):
                    caught = {
                        e.id for e in handler.type.elts if isinstance(e, ast.Name)
                    }
                elif isinstance(handler.type, ast.Name):
                    caught = {handler.type.id}
                self.assertNotIn("Exception", caught)
                self.assertNotIn("BaseException", caught)

    def test_the_retrieval_chain_does_not_reference_the_evaluator(self):
        """Evaluation must never be reachable from a runtime path."""
        from ai_hub.services import (
            hybrid_retrieval, knowledge_retrieval, knowledge_tooling,
            retrieval_audit, semantic_retrieval as sr, vector_store,
        )
        from ai_hub.tools import knowledge as knowledge_tools

        for module in (
            sr, hybrid_retrieval, retrieval_audit, knowledge_retrieval,
            vector_store, knowledge_tooling, knowledge_tools, pgvector_ann,
        ):
            with self.subTest(module=module.__name__):
                names = self._identifiers(module)
                for forbidden in (
                    "retrieval_evaluation", "evaluate_pgvector_suite",
                    "evaluate_pgvector_candidate", "PromotionDecision",
                ):
                    self.assertNotIn(forbidden, names)

    def test_the_backend_parameters_are_not_retuned(self):
        """`pgv-hnsw1` evidence must describe a stable backend."""
        self.assertEqual(pgvector_ann.PGVECTOR_BACKEND_VERSION, "pgv-hnsw1")
        self.assertEqual(pgvector_ann.HNSW_M, 16)
        self.assertEqual(pgvector_ann.HNSW_EF_CONSTRUCTION, 64)
        self.assertEqual(pgvector_ann.HNSW_EF_SEARCH, 100)
        self.assertEqual(pgvector_ann.ANN_CANDIDATE_POOL, 100)

    def test_the_evaluation_contract_is_versioned(self):
        self.assertEqual(EVALUATION_VERSION, "goldeneval1")
        self.assertEqual(EVALUATION_CUTOFFS, (1, 3, 5, 10, 20))


class SharedOracleTests(TestCase):
    """The S-21 extraction: one implementation, reached by both callers."""

    def _called(self, target):
        tree = ast.parse(inspect.getsource(target).lstrip())
        return {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    def test_the_retriever_and_the_oracle_share_the_ranking_core(self):
        for target in (
            semantic_retrieval.search_semantic_with_scope,
            rank_semantic_vector_with_scope,
        ):
            with self.subTest(target=target.__name__):
                self.assertIn("rank_semantic_candidates", self._called(target))

    def test_the_retriever_and_the_oracle_share_candidate_generation(self):
        for target in (
            semantic_retrieval.search_semantic_with_scope,
            rank_semantic_vector_with_scope,
        ):
            with self.subTest(target=target.__name__):
                self.assertIn(
                    "load_current_semantic_candidates", self._called(target)
                )

    def test_the_retriever_and_the_oracle_share_target_narrowing(self):
        for target in (
            semantic_retrieval.search_semantic_with_scope,
            rank_semantic_vector_with_scope,
        ):
            with self.subTest(target=target.__name__):
                self.assertIn("resolve_semantic_targets", self._called(target))

    def test_the_oracle_calls_no_provider_and_resolves_no_scope(self):
        called = self._called(rank_semantic_vector_with_scope)
        for forbidden in (
            "resolve_embedding_transport", "resolve_effective_knowledge_scope",
            "resolve_embedding_access", "normalize_embedding_vector",
            "canonical_query_embedding_text",
        ):
            self.assertNotIn(forbidden, called)

    def test_the_oracle_reports_that_no_provider_ran(self):
        source = inspect.getsource(rank_semantic_vector_with_scope)
        self.assertIn("provider_invoked=False", source)

    def test_the_public_semantic_signature_is_unchanged(self):
        parameters = inspect.signature(
            semantic_retrieval.semantic_search_knowledge_local
        ).parameters
        self.assertEqual(
            list(parameters),
            [
                "agent", "query", "embedding_model_config",
                "workspace", "collection_id", "limit",
            ],
        )
        self.assertEqual(parameters["limit"].default, 5)

    def test_the_internal_semantic_signature_is_unchanged(self):
        parameters = inspect.signature(
            semantic_retrieval.search_semantic_with_scope
        ).parameters
        self.assertEqual(
            list(parameters),
            ["scope", "query", "embedding_model_config", "collection_id", "limit"],
        )


# ---------------------------------------------------------------------------
# PostgreSQL: real exact-vs-ANN parity
# ---------------------------------------------------------------------------

class EvaluationCorpusMixin:
    """A deterministic corpus whose geometry makes every answer explicit.

    No English prose standing in for semantics: the vectors ARE the ground
    truth, so an expected ranking can be verified by hand rather than asserted
    on faith. Distractors exist so HNSW genuinely has work to do.
    """

    def build_corpus(self, *, metric=METRIC.COSINE, distractors=40):
        chat = ProviderConfig.objects.create(
            name="Chat P", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        self.model_config = ModelConfig.objects.create(
            provider=chat, model_name="training",
            temperature_default=Decimal("0.10"),
        )
        self.scope_a = ApplicationScope.objects.create(name="App A", slug="app-a")
        self.scope_b = ApplicationScope.objects.create(name="App B", slug="app-b")

        self.provider = ProviderConfig.objects.create(
            name="Embed P", provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://ollama.internal:11434",
            declared_locality=LOCALITY.LOCAL,
        )
        self.config = EmbeddingModelConfig.objects.create(
            name="embed-a", provider=self.provider,
            model_name="ollama/nomic-embed-text", model_revision="v1",
            vector_dimension=4, distance_metric=metric,
            normalization=NORMALIZATION.NONE,
        )
        self.config_other = EmbeddingModelConfig.objects.create(
            name="embed-b", provider=self.provider,
            model_name="ollama/nomic-embed-text", model_revision="v2",
            vector_dimension=4, distance_metric=metric,
            normalization=NORMALIZATION.NONE,
        )

        self.coll_a1 = self._collection(self.scope_a, "A One")
        self.coll_a2 = self._collection(self.scope_a, "A Two")
        self.coll_a3 = self._collection(self.scope_a, "A Three")   # unassigned
        self.coll_b1 = self._collection(self.scope_b, "B One")

        # Query is (1, 0, 0, 0).
        #   near   (1.00, 0.00) - the clear nearest neighbour
        #   close  (0.98, 0.20) - a close competitor
        #   tie_a  (0.50, 0.50) - deterministic tie with tie_b
        #   tie_b  (0.50, 0.50)
        #   far    (0.00, 1.00) - orthogonal
        self.near = self._chunk(self.coll_a1, "near", (1.00, 0.00, 0.0, 0.0))
        self.close = self._chunk(self.coll_a1, "close", (0.98, 0.20, 0.0, 0.0))
        self.tie_a = self._chunk(self.coll_a1, "tie-a", (0.50, 0.50, 0.0, 0.0))
        self.tie_b = self._chunk(self.coll_a1, "tie-b", (0.50, 0.50, 0.0, 0.0))
        self.far = self._chunk(self.coll_a2, "far", (0.00, 1.00, 0.0, 0.0))

        # Forbidden, and deliberately PERFECT matches.
        self.unassigned = self._chunk(
            self.coll_a3, "unassigned", (1.0, 0.0, 0.0, 0.0)
        )
        self.foreign = self._chunk(self.coll_b1, "foreign", (1.0, 0.0, 0.0, 0.0))

        # Distractors, so the graph is not trivially small.
        for index in range(distractors):
            angle = 0.1 + (index % 20) * 0.04
            self._chunk(
                self.coll_a2, f"distractor-{index}",
                (1.0 - angle, angle, 0.0, 0.0),
            )

        self.agent = AgentProfile.objects.create(
            name="A Agent", role="r", model_config=self.model_config,
            application_scope=self.scope_a, knowledge_max_chars=6000,
        )
        self.agent.knowledge_collections.add(self.coll_a1, self.coll_a2)

        self.scope = resolve_effective_knowledge_scope(self.agent)
        self.contract = resolve_embedding_contract(self.config)
        self.query = (1.0, 0.0, 0.0, 0.0)
        return self

    def _collection(self, scope, name):
        return KnowledgeCollection.objects.create(
            name=name, description="", application_scope=scope
        )

    def _chunk(self, collection, label, values, *, config=None):
        document = KnowledgeDocument.objects.create(
            collection=collection, title=f"Doc {label}",
            curated_text=f"{KNOWLEDGE_SECRET} {label}",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        chunk = KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1, section_title="S",
            content=f"{KNOWLEDGE_SECRET} {label}",
        )
        store_chunk_vector(
            application_scope=collection.application_scope, chunk=chunk,
            embedding_model_config=config or self.config, vector=tuple(values),
        )
        return chunk

    def prepare_leaves(self, *collections, config=None):
        for collection in collections or (self.coll_a1, self.coll_a2):
            provision_pgvector_ann_leaf(
                application_scope=collection.application_scope,
                collection=collection,
                embedding_model_config=config or self.config,
            )
            rebuild_pgvector_ann_leaf(
                application_scope=collection.application_scope,
                collection=collection,
                embedding_model_config=config or self.config,
            )

    def generation_probe(self):
        def probe():
            return {
                cid: pgvector_ann.current_generation(self.scope_a.pk, cid)
                for cid in sorted(self.scope.collection_ids)
            }
        return probe


@requires_postgres
class RealOracleParityTests(EvaluationCorpusMixin, TestCase):
    def evaluate(self, case=None, **kwargs):
        return evaluate_pgvector_candidate(
            self.scope, case or GoldenQueryCase(
                case_id="case-001", query_values=self.query, limit=5,
            ),
            embedding_model_config=self.config,
            source_generation_probe=self.generation_probe(),
            **kwargs,
        )

    def test_cosine_parity_against_the_exact_oracle(self):
        self.build_corpus(metric=METRIC.COSINE)
        self.prepare_leaves()
        result = self.evaluate()

        self.assertEqual(result.outcome, CaseOutcome.COMPARED)
        self.assertEqual(result.hard_invariants, ())
        for entry in result.parity.cutoffs:
            with self.subTest(cutoff=entry.cutoff):
                self.assertEqual(entry.oracle_recall, 1.0)
                self.assertTrue(entry.prefix_equal)
                self.assertFalse(entry.catastrophic_miss)
        self.assertLessEqual(
            result.parity.score_parity_max_delta, SCORE_PARITY_TOLERANCE
        )

    def test_dot_product_parity_against_the_exact_oracle(self):
        self.build_corpus(metric=METRIC.DOT_PRODUCT)
        self.prepare_leaves()
        result = self.evaluate()
        self.assertEqual(result.hard_invariants, ())
        for entry in result.parity.cutoffs:
            self.assertEqual(entry.oracle_recall, 1.0)

    def test_euclidean_parity_against_the_exact_oracle(self):
        self.build_corpus(metric=METRIC.EUCLIDEAN)
        self.prepare_leaves()
        result = self.evaluate()
        self.assertEqual(result.hard_invariants, ())
        for entry in result.parity.cutoffs:
            self.assertEqual(entry.oracle_recall, 1.0)

    def test_deterministic_ties_resolve_identically_in_both_backends(self):
        self.build_corpus(metric=METRIC.COSINE, distractors=0)
        self.prepare_leaves()
        exact = rank_semantic_vector_with_scope(
            self.scope, query_values=self.query,
            embedding_model_config=self.config, limit=5,
        )
        result = self.evaluate()
        tied = [
            match.chunk_id for match in exact.matches
            if match.chunk_id in {self.tie_a.pk, self.tie_b.pk}
        ]
        self.assertEqual(tied, sorted(tied), "chunk_id ASC")
        self.assertTrue(
            all(entry.prefix_equal for entry in result.parity.cutoffs)
        )

    def test_multi_collection_search_is_evaluated_as_one_target_set(self):
        self.build_corpus()
        self.prepare_leaves()
        result = self.evaluate()
        self.assertEqual(
            tuple(sorted(result.collection_ids)),
            tuple(sorted([self.coll_a1.pk, self.coll_a2.pk])),
        )

    def test_collection_narrowing_is_honoured_by_both_backends(self):
        self.build_corpus()
        self.prepare_leaves()
        result = self.evaluate(
            case=GoldenQueryCase(
                case_id="narrow-001", query_values=self.query,
                collection_id=self.coll_a1.pk, limit=5,
            )
        )
        self.assertEqual(result.collection_ids, (self.coll_a1.pk,))
        self.assertEqual(result.hard_invariants, ())

    def test_an_unauthorized_same_scope_collection_never_appears(self):
        self.build_corpus()
        self.prepare_leaves()
        self.prepare_leaves(self.coll_a3)
        exact = rank_semantic_vector_with_scope(
            self.scope, query_values=self.query,
            embedding_model_config=self.config, limit=20,
        )
        result = self.evaluate(
            case=GoldenQueryCase(
                case_id="adversarial-001", query_values=self.query, limit=20
            )
        )
        self.assertNotIn(
            self.unassigned.pk, [m.chunk_id for m in exact.matches]
        )
        self.assertNotIn(self.coll_a3.pk, result.collection_ids)
        self.assertEqual(result.hard_invariants, ())

    def test_cross_scope_knowledge_never_appears(self):
        self.build_corpus()
        self.prepare_leaves()
        provision_pgvector_ann_leaf(
            application_scope=self.scope_b, collection=self.coll_b1,
            embedding_model_config=self.config,
        )
        rebuild_pgvector_ann_leaf(
            application_scope=self.scope_b, collection=self.coll_b1,
            embedding_model_config=self.config,
        )
        result = self.evaluate(
            case=GoldenQueryCase(
                case_id="adversarial-002", query_values=self.query, limit=20
            )
        )
        self.assertNotIn(self.coll_b1.pk, result.collection_ids)
        self.assertEqual(result.hard_invariants, ())

    def test_an_inaccessible_collection_yields_the_same_empty_shape(self):
        self.build_corpus()
        self.prepare_leaves()
        for requested in (self.coll_a3.pk, self.coll_b1.pk, 9_999_999):
            with self.subTest(requested=requested):
                result = self.evaluate(
                    case=GoldenQueryCase(
                        case_id="adr-n5", query_values=self.query,
                        collection_id=requested, limit=5,
                    )
                )
                self.assertEqual(result.outcome, CaseOutcome.BOTH_EMPTY)
                self.assertEqual(result.collection_ids, ())
                self.assertEqual(result.hard_invariants, ())

    def test_a_stale_vector_is_excluded_from_both_backends(self):
        self.build_corpus()
        self.prepare_leaves()
        KnowledgeDocumentChunk.objects.filter(pk=self.close.pk).update(
            content="edited after indexing"
        )
        # The leaf is now stale, so ANN refuses while the exact oracle answers -
        # a REFUSAL MISMATCH, which is exactly the hard failure it should be.
        result = self.evaluate()
        self.assertEqual(result.outcome, CaseOutcome.REFUSAL_MISMATCH)
        self.assertEqual(result.ann_refusal, RefusalKind.BACKEND_NOT_READY)

    def test_an_archived_document_is_excluded_from_both_backends(self):
        self.build_corpus()
        KnowledgeDocument.objects.filter(pk=self.close.document_id).update(
            status=KnowledgeDocument.Status.ARCHIVED
        )
        self.prepare_leaves()
        result = self.evaluate()
        self.assertEqual(result.outcome, CaseOutcome.COMPARED)
        self.assertEqual(result.hard_invariants, ())

    def test_a_different_e1_is_not_mixed_into_the_evaluation(self):
        self.build_corpus()
        self.prepare_leaves()
        self._chunk(
            self.coll_a1, "other-space", (1.0, 0.0, 0.0, 0.0),
            config=self.config_other,
        )
        self.prepare_leaves(self.coll_a1, self.coll_a2)
        result = self.evaluate()
        self.assertEqual(result.e1, self.contract.e1)
        self.assertEqual(result.hard_invariants, ())

    def test_limit_zero_compares_two_empty_answers(self):
        self.build_corpus()
        self.prepare_leaves()
        result = self.evaluate(
            case=GoldenQueryCase(
                case_id="limit-zero", query_values=self.query, limit=0
            )
        )
        self.assertEqual(result.outcome, CaseOutcome.BOTH_EMPTY)
        self.assertEqual(result.hard_invariants, ())

    def test_a_zero_cosine_query_refuses_on_both_sides(self):
        self.build_corpus(metric=METRIC.COSINE)
        self.prepare_leaves()
        result = self.evaluate(
            case=GoldenQueryCase(
                case_id="zero-query", query_values=(0.0, 0.0, 0.0, 0.0), limit=5
            )
        )
        self.assertEqual(result.outcome, CaseOutcome.BOTH_REFUSED)
        self.assertEqual(result.exact_refusal, RefusalKind.UNSCORABLE_QUERY)
        self.assertEqual(result.ann_refusal, RefusalKind.UNSCORABLE_QUERY)

    def test_an_invalid_query_vector_refuses_on_both_sides(self):
        self.build_corpus()
        self.prepare_leaves()
        result = self.evaluate(
            case=GoldenQueryCase(
                case_id="bad-vector", query_values=(1.0, 0.0), limit=5
            )
        )
        self.assertEqual(result.outcome, CaseOutcome.BOTH_REFUSED)
        self.assertEqual(result.exact_refusal, RefusalKind.INVALID_VECTOR)
        self.assertEqual(result.ann_refusal, RefusalKind.INVALID_VECTOR)

    def test_a_source_change_during_the_comparison_invalidates_the_case(self):
        self.build_corpus()
        self.prepare_leaves()
        real = retrieval_evaluation.search_pgvector_ann_with_scope

        def mutate_mid_comparison(scope, **kwargs):
            result = real(scope, **kwargs)
            KnowledgeDocumentChunk.objects.filter(pk=self.far.pk).update(
                content="changed during evaluation"
            )
            return result

        with mock.patch.object(
            retrieval_evaluation, "search_pgvector_ann_with_scope",
            mutate_mid_comparison,
        ):
            result = self.evaluate()
        self.assertEqual(result.outcome, CaseOutcome.SOURCE_CHANGED)
        self.assertIsNone(result.parity)

    def test_the_controlled_suite_is_eligible_under_the_controlled_policy(self):
        self.build_corpus()
        self.prepare_leaves()
        exact = rank_semantic_vector_with_scope(
            self.scope, query_values=self.query,
            embedding_model_config=self.config, limit=5,
        )
        judgments = tuple(
            GoldenRelevanceJudgment(chunk_id=match.chunk_id, grade=3)
            for match in exact.matches[:1]
        )
        cases = [
            GoldenQueryCase(
                case_id=f"controlled-{index:03d}", query_values=self.query,
                evidence_class=EvidenceClass.CONTROLLED, limit=5,
                judgments=judgments,
            )
            for index in range(3)
        ]
        summary = evaluate_pgvector_suite(
            self.scope, cases, embedding_model_config=self.config,
            policy=CONTROLLED_POLICY,
            source_generation_probe=self.generation_probe(),
        )
        self.assertEqual(summary.hard_invariant_failures, ())
        self.assertEqual(summary.decision, PromotionDecision.ELIGIBLE)
        self.assertEqual(summary.mean_oracle_recall[5], 1.0)

    def test_the_same_controlled_suite_cannot_production_promote(self):
        """Real ANN evidence, still synthetic. Still not representative."""
        self.build_corpus()
        self.prepare_leaves()
        cases = [
            GoldenQueryCase(
                case_id=f"controlled-{index:03d}", query_values=self.query,
                evidence_class=EvidenceClass.CONTROLLED, limit=5,
            )
            for index in range(3)
        ]
        summary = evaluate_pgvector_suite(
            self.scope, cases, embedding_model_config=self.config,
            policy=REPRESENTATIVE_POLICY_TEMPLATE,
            source_generation_probe=self.generation_probe(),
        )
        self.assertEqual(
            summary.decision, PromotionDecision.INSUFFICIENT_EVIDENCE
        )

    def test_evaluation_writes_no_audit_row_and_no_data(self):
        self.build_corpus()
        self.prepare_leaves()
        before = {
            model.__name__: model.objects.count()
            for model in (
                RetrievalRun, RetrievalRunCollection, RetrievalOutcome,
                RetrievalHit,
            )
        }
        self.evaluate()
        after = {
            model.__name__: model.objects.count()
            for model in (
                RetrievalRun, RetrievalRunCollection, RetrievalOutcome,
                RetrievalHit,
            )
        }
        self.assertEqual(before, after)
        self.assertEqual(RetrievalRun.objects.count(), 0)

    def test_no_provider_is_contacted_during_parity_evaluation(self):
        self.build_corpus()
        self.prepare_leaves()

        def explode(*args, **kwargs):
            raise AssertionError("no provider may be contacted")

        with mock.patch("requests.post", explode):
            with mock.patch("requests.get", explode):
                with mock.patch(
                    "ai_hub.services.semantic_retrieval."
                    "resolve_embedding_transport",
                    explode,
                ):
                    result = self.evaluate()
        self.assertEqual(result.outcome, CaseOutcome.COMPARED)

    def test_the_evidence_contains_no_corpus_marker(self):
        self.build_corpus()
        self.prepare_leaves()
        summary = evaluate_pgvector_suite(
            self.scope,
            [GoldenQueryCase(case_id="c1", query_values=self.query, limit=5)],
            embedding_model_config=self.config, policy=CONTROLLED_POLICY,
            source_generation_probe=self.generation_probe(),
        )
        blob = json.dumps(
            dataclasses.asdict(summary), default=str, sort_keys=True
        )
        for forbidden in (KNOWLEDGE_SECRET, "near", "distractor", "[1.0"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, blob)


@requires_postgres
class SkipAccountingTests(TestCase):
    def test_the_postgresql_campaign_cannot_silently_skip(self):
        self.assertEqual(POSTGRES, connection.vendor == "postgresql")
        for name in ("RealOracleParityTests",):
            case = globals()[name]
            self.assertFalse(getattr(case, "__unittest_skip__", False))


class NoMigrationTests(TestCase):
    def test_this_slice_adds_no_migration(self):
        from django.db.migrations.loader import MigrationLoader

        loader = MigrationLoader(None, ignore_no_migrations=True)
        names = sorted(
            name for app, name in loader.disk_migrations if app == "ai_hub"
        )
        self.assertEqual(names[-1], "0030_pgvector_ann_foundation")
        self.assertEqual(
            [name for name in names if name.startswith("0031")], []
        )
