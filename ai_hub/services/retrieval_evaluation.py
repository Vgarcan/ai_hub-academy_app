"""Golden-query evaluation and the pgvector promotion GATE. Evidence, not routing.

    resolve S-15 scope ONCE
              |
      +-------+-------+
      |               |
  S-21 exact      S-24 pgvector ANN
   oracle          (shadow only)
      |               |
      +-------+-------+
              |
      oracle parity + golden relevance
              |
        PromotionDecision   <- evidence. Nothing consumes it.

**This module answers one question and changes nothing.** Does `pgv-hnsw1`
reproduce the exact S-21 ranking closely enough to become ELIGIBLE for a later,
separately authorized promotion decision? Producing that verdict must never
route a single production request through pgvector: an evaluator that modified
the system it measures is not an evaluator.

**There is exactly one exact oracle, and this module does not contain it.**
Every baseline ranking comes from `rank_semantic_vector_with_scope`, the same
code S-21 uses. A second "equivalent" implementation of cosine, of `k1`
currentness or of tie-breaking would eventually disagree with the retriever, and
every parity number measured against it would be describing the wrong thing.

**Both backends see ONE frozen authorization snapshot and ONE query vector.** If
each branch resolved its own scope, the comparison would straddle two
authorization states; if the query were embedded twice, provider variation would
be attributed to ANN approximation. Either makes the evidence invalid rather
than merely noisy.

**Synthetic fixtures cannot establish production quality, and this module
refuses to pretend otherwise.** Evidence carries a CLASS. A suite of
`controlled_test_evidence` can prove the evaluator and the backend are
mathematically correct; it can never satisfy a policy that requires
`representative_evidence`, no matter how perfect its metrics.

Deliberately absent: no LLM judge, no embedding-provider judge, no external
evaluation framework, no network access, no persistence, no migration, no
runtime feature flag, no backend selector, no fallback.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field

from ai_hub.services.pgvector_ann import (
    PGVECTOR_BACKEND_VERSION,
    PgvectorAnnError,
    search_pgvector_ann_with_scope,
)
from ai_hub.services.semantic_retrieval import (
    SemanticRetrievalError,
    rank_semantic_vector_with_scope,
)

#: The meaning of these metrics and result shapes. A later change to what is
#: computed becomes a NEW version rather than a silent reinterpretation of every
#: historical number.
EVALUATION_VERSION = "goldeneval1"

#: The promotion policy contract. Separate from the evaluation version because a
#: policy can be tightened without the metrics changing meaning.
PROMOTION_POLICY_VERSION = "pgv-promotion1"

#: Cutoffs evaluated for every case. A cutoff larger than the requested result
#: size is skipped rather than padded - a Recall@10 computed over a top-5
#: request would be measuring the request, not the backend.
EVALUATION_CUTOFFS = (1, 3, 5, 10, 20)

#: Grades below this are NOT relevant for the binary metrics (Precision, Recall,
#: MRR). NDCG uses the graded values directly and ignores this threshold.
BINARY_RELEVANCE_THRESHOLD = 2

#: `2**grade - 1`, the conventional DCG gain, with a `log2(rank + 1)` discount.
#: Named and versioned so a later change cannot silently re-scale history.
NDCG_GAIN = "pow2_minus_1"

#: Exact scores must agree to this tolerance. Both backends rank with the SAME
#: S-21 scorers over the SAME canonical `f32le1` vectors, so any disagreement is
#: a real defect and not float drift - the tolerance covers summation order
#: only.
SCORE_PARITY_TOLERANCE = 1e-9

VALID_RELEVANCE_GRADES = frozenset({0, 1, 2, 3})


class EvidenceClass:
    """How much a case is allowed to prove. Load-bearing, not a label.

    `controlled_test_evidence` is a deterministic fixture built to make expected
    relationships mathematically explicit. It proves implementation correctness.
    It says NOTHING about how the backend behaves on a real corpus, and the
    policy below enforces that distinction rather than trusting anyone to
    remember it.
    """

    CONTROLLED = "controlled_test_evidence"
    REPRESENTATIVE = "representative_evidence"


VALID_EVIDENCE_CLASSES = frozenset({
    EvidenceClass.CONTROLLED, EvidenceClass.REPRESENTATIVE
})


class CaseOutcome:
    """What happened to one evaluated case. Bounded; never exception text."""

    COMPARED = "compared"
    #: Both backends legitimately produced nothing. A real answer, not a failure.
    BOTH_EMPTY = "both_empty"
    #: Both refused in an equivalent way. Parity, not a quality problem.
    BOTH_REFUSED = "both_refused"
    #: The backends disagreed about whether to answer at all. A HARD failure:
    #: one returning results where the other refuses is a semantics mismatch,
    #: never a Recall@K of zero.
    REFUSAL_MISMATCH = "refusal_mismatch"
    #: The corpus moved during the comparison, so the two observations describe
    #: different states. Invalid evidence - never averaged in, never retried.
    SOURCE_CHANGED = "source_changed"
    #: A structural disagreement that no metric may average away.
    INVARIANT_VIOLATION = "invariant_violation"


class RefusalKind:
    """Evaluator-level classification of a refusal.

    Deliberately its OWN small vocabulary rather than forcing S-21 and S-24 to
    share literal category strings. They are different backends with different
    failure modes on purpose; flattening their vocabularies would misreport what
    each one actually said.
    """

    NONE = ""
    UNSCORABLE_QUERY = "unscorable_query"
    INVALID_VECTOR = "invalid_vector"
    BACKEND_NOT_READY = "backend_not_ready"
    BACKEND_INTEGRITY_FAILURE = "backend_integrity_failure"
    CANDIDATE_SOURCE_CHANGED = "candidate_source_changed"
    NOT_AUTHORIZED = "not_authorized"
    REFERENCE_LIMIT = "reference_limit"
    OTHER = "other"


class HardInvariant:
    """Zero-tolerance failures. No average may hide one of these."""

    APPLICATION_SCOPE_MISMATCH = "application_scope_mismatch"
    COLLECTION_SET_MISMATCH = "collection_set_mismatch"
    E1_MISMATCH = "e1_mismatch"
    METRIC_MISMATCH = "metric_mismatch"
    BACKEND_VERSION_MISMATCH = "backend_version_mismatch"
    FOREIGN_RESULT = "foreign_result"
    SCORE_DISAGREEMENT = "score_disagreement"
    REFUSAL_SEMANTICS_MISMATCH = "refusal_semantics_mismatch"
    SOURCE_CHANGED_DURING_EVALUATION = "source_changed_during_evaluation"
    INCOMPLETE_EVALUATION = "incomplete_evaluation"


class PromotionDecision:
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvaluationError(RuntimeError):
    """A refusal from the evaluator itself. Bounded category, no content."""

    def __init__(self, category: str, message: str = ""):
        self.category = category
        super().__init__(message or category)


# ---------------------------------------------------------------------------
# Metric mathematics. Pure, deterministic, no I/O, no database.
# ---------------------------------------------------------------------------

def validate_relevance_grade(grade) -> int:
    """A grade in 0..3, or refuse. Never silently coerced.

    Coercing an unexpected value to 0 would quietly relabel a judgment as "not
    relevant" and every metric downstream would be computed from a fiction.
    """
    if isinstance(grade, bool) or not isinstance(grade, int):
        raise EvaluationError(
            "invalid_relevance_grade", "A relevance grade must be an integer."
        )
    if grade not in VALID_RELEVANCE_GRADES:
        raise EvaluationError(
            "invalid_relevance_grade",
            f"Relevance grades are {sorted(VALID_RELEVANCE_GRADES)}.",
        )
    return grade


def is_relevant(grade: int) -> bool:
    """The ONE binary threshold, applied everywhere Precision/Recall/MRR need it."""
    return validate_relevance_grade(grade) >= BINARY_RELEVANCE_THRESHOLD


def precision_at_k(ranked_ids, judgments, k: int) -> float:
    """Relevant hits in the top-K, over the number actually returned.

    The denominator is `min(k, len(ranked))`, not `k`. Dividing by `k` when only
    two results exist would report a backend as imprecise for returning
    everything the corpus had.

    An empty ranking has no precision to speak of and returns 0.0 rather than
    NaN - a metric that emits NaN cannot be aggregated or compared.
    """
    top = list(ranked_ids)[:k]
    if not top:
        return 0.0
    hits = sum(1 for chunk_id in top if is_relevant(judgments.get(chunk_id, 0)))
    return hits / len(top)


def recall_at_k(ranked_ids, judgments, k: int) -> float:
    """Relevant hits in the top-K, over ALL relevant documents.

    With no relevant documents at all the quantity is undefined: there is
    nothing to recall. Defined here as 1.0 - a backend cannot be faulted for
    failing to retrieve documents that do not exist - and tested explicitly so
    the convention is visible rather than implied.
    """
    relevant = {
        chunk_id for chunk_id, grade in judgments.items() if is_relevant(grade)
    }
    if not relevant:
        return 1.0
    top = set(list(ranked_ids)[:k])
    return len(top & relevant) / len(relevant)


def mean_reciprocal_rank(ranked_ids, judgments) -> float:
    """`1 / rank` of the FIRST relevant result; 0.0 if there is none."""
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if is_relevant(judgments.get(chunk_id, 0)):
            return 1.0 / position
    return 0.0


def _dcg(grades) -> float:
    return sum(
        (2 ** grade - 1) / math.log2(position + 1)
        for position, grade in enumerate(grades, start=1)
    )


def ndcg_at_k(ranked_ids, judgments, k: int) -> float:
    """Graded NDCG with `2**grade - 1` gain and a `log2(rank + 1)` discount.

    Uses the GRADED judgments directly and ignores the binary threshold, which
    is the point of having grades at all. With no positive gain available the
    ideal DCG is zero and the ratio is undefined; defined as 1.0 for the same
    reason as `recall_at_k`, and tested.
    """
    top = list(ranked_ids)[:k]
    actual = _dcg(validate_relevance_grade(judgments.get(cid, 0)) for cid in top)
    ideal_grades = sorted(
        (validate_relevance_grade(grade) for grade in judgments.values()),
        reverse=True,
    )[:k]
    ideal = _dcg(ideal_grades)
    if ideal <= 0.0:
        return 1.0
    return actual / ideal


def oracle_recall_at_k(ann_ids, exact_ids, k: int) -> float:
    """Overlap with the EXACT top-K, over what the exact ranking actually had.

    The denominator is `min(k, len(exact))`. Dividing by `k` when the authorized
    corpus holds three vectors would report Recall@10 = 0.3 for a backend that
    reproduced the oracle perfectly.

    Both empty is perfect agreement, not a division by zero.
    """
    exact_top = list(exact_ids)[:k]
    if not exact_top:
        return 1.0
    ann_top = set(list(ann_ids)[:k])
    return len(ann_top & set(exact_top)) / len(exact_top)


def prefix_equal_at_k(ann_ids, exact_ids, k: int) -> bool:
    """ORDERED equality of the two top-K sequences. Stricter than overlap."""
    return list(ann_ids)[:k] == list(exact_ids)[:k]


def first_divergence_rank(ann_ids, exact_ids):
    """1-based position of the first ordering difference, or `None`.

    A length difference counts as divergence at the first missing position: a
    backend that returned four of five results diverged, and reporting `None`
    would call that agreement.
    """
    ann = list(ann_ids)
    exact = list(exact_ids)
    for position in range(max(len(ann), len(exact))):
        left = ann[position] if position < len(ann) else None
        right = exact[position] if position < len(exact) else None
        if left != right:
            return position + 1
    return None


def rank_displacements(ann_ids, exact_ids) -> dict:
    """`|ann_rank - exact_rank|` for chunks present in BOTH rankings.

    Missing results are deliberately excluded rather than assigned a punitive
    synthetic rank. A fabricated rank would turn "absent" into a number that
    averages, and absence is exactly what `oracle_recall_at_k` and
    `catastrophic_miss` exist to measure honestly.
    """
    ann_rank = {cid: index for index, cid in enumerate(ann_ids, start=1)}
    exact_rank = {cid: index for index, cid in enumerate(exact_ids, start=1)}
    return {
        cid: abs(ann_rank[cid] - exact_rank[cid])
        for cid in ann_rank.keys() & exact_rank.keys()
    }


def is_catastrophic_miss(ann_ids, exact_ids, k: int) -> bool:
    """The exact ranking had hits at K and the ANN ranking shares NONE of them.

    Tracked separately because an average cannot express it. A suite can hold a
    mean oracle Recall@5 of 0.95 while one query returns nothing a user wanted,
    and that query is the one that matters.
    """
    exact_top = set(list(exact_ids)[:k])
    if not exact_top:
        return False
    return not (set(list(ann_ids)[:k]) & exact_top)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoldenRelevanceJudgment:
    """One human/fixture judgment. An id and a grade - never text."""

    chunk_id: int
    grade: int

    def __post_init__(self):
        validate_relevance_grade(self.grade)


@dataclass(frozen=True)
class GoldenQueryCase:
    """One evaluated query.

    Carries `query_values`, NOT a query string: the evaluator never embeds
    anything, so it never needs the text, and never holding it is the simplest
    way to guarantee it is never stored. `case_id` is a bounded opaque label.
    """

    case_id: str
    query_values: tuple
    evidence_class: str = EvidenceClass.CONTROLLED
    collection_id: int | None = None
    limit: int = 5
    judgments: tuple = ()

    def judgment_map(self) -> dict:
        return {
            judgment.chunk_id: validate_relevance_grade(judgment.grade)
            for judgment in self.judgments
        }


# ---------------------------------------------------------------------------
# Outputs. Content-free by construction.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CutoffParity:
    cutoff: int
    oracle_recall: float
    prefix_equal: bool
    catastrophic_miss: bool


@dataclass(frozen=True)
class CutoffRelevance:
    cutoff: int
    precision: float
    recall: float
    ndcg: float


@dataclass(frozen=True)
class OracleParityMetrics:
    exact_result_count: int
    ann_result_count: int
    first_divergence_rank: int | None
    max_rank_displacement: int
    mean_rank_displacement: float
    score_parity_max_delta: float
    cutoffs: tuple


@dataclass(frozen=True)
class GoldenRelevanceMetrics:
    exact_mrr: float
    ann_mrr: float
    cutoffs: tuple


@dataclass(frozen=True)
class RetrievalEvaluationCaseResult:
    """One case. Ids appear only as counts and bounded diagnostics."""

    case_id: str
    evidence_class: str
    outcome: str
    application_scope_id: int | None
    collection_ids: tuple
    embedding_model_config_id: int | None
    e1: str
    metric: str
    backend_version: str
    limit: int
    exact_refusal: str
    ann_refusal: str
    parity: OracleParityMetrics | None
    relevance: GoldenRelevanceMetrics | None
    hard_invariants: tuple


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    """The suite verdict. Deterministic; carries no query, content or vector."""

    evaluation_version: str
    policy_version: str
    backend_version: str
    fingerprint: str

    requested_case_count: int
    completed_case_count: int
    invalid_case_count: int
    failed_case_count: int
    controlled_case_count: int
    representative_case_count: int

    mean_oracle_recall: dict
    worst_oracle_recall: dict
    prefix_match_rate: dict
    catastrophic_miss_count: dict
    mean_ndcg: dict
    mean_mrr: float

    hard_invariant_failures: tuple
    decision: str
    decision_reasons: tuple
    cases: tuple = field(default=(), repr=False)


# ---------------------------------------------------------------------------
# Promotion policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromotionPolicy:
    """What the evidence must show. Thresholds live HERE, once.

    Deliberately not scattered as literals across tests: a threshold repeated in
    twelve assertions is twelve places to quietly relax it, and the number that
    decided a promotion has to be inspectable as one object.
    """

    version: str = PROMOTION_POLICY_VERSION
    required_evidence_class: str = EvidenceClass.REPRESENTATIVE
    minimum_case_count: int = 1
    minimum_mean_oracle_recall: dict = field(default_factory=dict)
    minimum_worst_oracle_recall: dict = field(default_factory=dict)
    minimum_prefix_match_rate: dict = field(default_factory=dict)
    maximum_catastrophic_misses: int = 0
    minimum_mean_ndcg: dict = field(default_factory=dict)
    minimum_mean_mrr: float = 0.0


#: The strict expectation for the deterministic S-25 regression corpus. These
#: fixtures are built so the answer is mathematically unambiguous, so anything
#: short of exact reproduction is a signal to investigate `pgv-hnsw1` - never a
#: reason to lower the bar.
CONTROLLED_POLICY = PromotionPolicy(
    required_evidence_class=EvidenceClass.CONTROLLED,
    minimum_case_count=1,
    minimum_mean_oracle_recall={1: 1.0, 3: 1.0, 5: 1.0},
    minimum_worst_oracle_recall={1: 1.0, 3: 1.0, 5: 1.0},
    minimum_prefix_match_rate={1: 1.0, 3: 1.0, 5: 1.0},
    maximum_catastrophic_misses=0,
    minimum_mean_ndcg={5: 1.0},
    minimum_mean_mrr=1.0,
)

#: The shape a real promotion would need. The numbers here are deliberately NOT
#: presented as statistically justified: this repository contains no judged
#: representative corpus, so nobody has measured what a defensible threshold is.
#: The policy exists so that a future slice fills it in with evidence rather
#: than inventing it at promotion time.
REPRESENTATIVE_POLICY_TEMPLATE = PromotionPolicy(
    required_evidence_class=EvidenceClass.REPRESENTATIVE,
    minimum_case_count=50,
    minimum_mean_oracle_recall={5: 0.95, 10: 0.95},
    minimum_worst_oracle_recall={5: 0.60},
    minimum_prefix_match_rate={1: 0.95},
    maximum_catastrophic_misses=0,
    minimum_mean_ndcg={5: 0.90},
    minimum_mean_mrr=0.80,
)


def evaluation_fingerprint(*, policy, backend_version, e1, metric, evidence_class):
    """A deterministic id for ONE evaluation configuration. Content-free.

    Canonical JSON over bounded structural facts only - version strings, the
    contract fingerprint, the metric, the cutoffs and the evidence class. No
    query, no vector, no Knowledge, no credential ever reaches the digest.
    """
    # The policy's THRESHOLDS, not merely its version string. Two policies can
    # share `pgv-promotion1` while demanding very different things, and a
    # fingerprint that collapsed them would claim two incomparable evaluations
    # were the same configuration.
    payload = {
        "contract": EVALUATION_VERSION,
        "policy": policy.version,
        "policy_thresholds": {
            "required_evidence_class": policy.required_evidence_class,
            "minimum_case_count": policy.minimum_case_count,
            "minimum_mean_oracle_recall": {
                str(k): v
                for k, v in sorted(policy.minimum_mean_oracle_recall.items())
            },
            "minimum_worst_oracle_recall": {
                str(k): v
                for k, v in sorted(policy.minimum_worst_oracle_recall.items())
            },
            "minimum_prefix_match_rate": {
                str(k): v
                for k, v in sorted(policy.minimum_prefix_match_rate.items())
            },
            "maximum_catastrophic_misses": policy.maximum_catastrophic_misses,
            "minimum_mean_ndcg": {
                str(k): v for k, v in sorted(policy.minimum_mean_ndcg.items())
            },
            "minimum_mean_mrr": policy.minimum_mean_mrr,
        },
        "backend": backend_version,
        "e1": e1,
        "metric": metric,
        "cutoffs": list(EVALUATION_CUTOFFS),
        "evidence_class": evidence_class,
        "binary_threshold": BINARY_RELEVANCE_THRESHOLD,
        "ndcg_gain": NDCG_GAIN,
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    return "eval1:sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Refusal classification
# ---------------------------------------------------------------------------

_SEMANTIC_REFUSALS = {
    "unscorable_zero_vector": RefusalKind.UNSCORABLE_QUERY,
    "vector_dimension_mismatch": RefusalKind.INVALID_VECTOR,
    "vector_non_finite": RefusalKind.INVALID_VECTOR,
    "zero_vector_cannot_l2_normalize": RefusalKind.INVALID_VECTOR,
    "candidate_dimension_mismatch": RefusalKind.BACKEND_INTEGRITY_FAILURE,
    "reference_candidate_limit_exceeded": RefusalKind.REFERENCE_LIMIT,
    "embedding_not_authorized": RefusalKind.NOT_AUTHORIZED,
    "scope_unavailable": RefusalKind.NOT_AUTHORIZED,
}

_PGVECTOR_REFUSALS = {
    "unscorable_zero_vector": RefusalKind.UNSCORABLE_QUERY,
    "pgvector_query_vector_invalid": RefusalKind.INVALID_VECTOR,
    "pgvector_leaf_not_ready": RefusalKind.BACKEND_NOT_READY,
    "pgvector_ann_integrity_mismatch": RefusalKind.BACKEND_INTEGRITY_FAILURE,
    "pgvector_ann_source_changed_during_search":
        RefusalKind.CANDIDATE_SOURCE_CHANGED,
    "pgvector_collection_foreign_scope": RefusalKind.NOT_AUTHORIZED,
}


def classify_refusal(exc) -> str:
    """Map ONE bounded backend refusal to the evaluator's own vocabulary.

    Never `str(exc)`. A raw message can echo configuration or - worse - content,
    and evaluation evidence must not become the place that leaks it.
    """
    category = getattr(exc, "category", None)
    if isinstance(exc, PgvectorAnnError):
        return _PGVECTOR_REFUSALS.get(category, RefusalKind.OTHER)
    if isinstance(exc, SemanticRetrievalError):
        return _SEMANTIC_REFUSALS.get(category, RefusalKind.OTHER)
    return RefusalKind.OTHER


#: Only these are absorbed as evaluation outcomes. Anything else is a
#: programming error and must propagate: an evaluator that swallowed every
#: exception would report a broken backend as merely unremarkable.
EXPECTED_BACKEND_ERRORS = (SemanticRetrievalError, PgvectorAnnError)


# ---------------------------------------------------------------------------
# The evaluation itself
# ---------------------------------------------------------------------------

def _ids(result):
    return tuple(match.chunk_id for match in result.matches)


def _scores(result):
    return {match.chunk_id: match.metric_value for match in result.matches}


def _check_hard_invariants(case, exact, ann, targets_ok):
    """Structural agreement. Any failure here is zero-tolerance.

    These are not quality signals and must never be averaged. A backend that
    answered from the wrong namespace did not score poorly; it did something it
    must never do.
    """
    failures = []
    if exact.application_scope_id != ann.application_scope_id:
        failures.append(HardInvariant.APPLICATION_SCOPE_MISMATCH)
    if tuple(sorted(exact.collection_ids)) != tuple(sorted(ann.collection_ids)):
        failures.append(HardInvariant.COLLECTION_SET_MISMATCH)
    if exact.e1 != ann.e1:
        failures.append(HardInvariant.E1_MISMATCH)
    if exact.metric != ann.metric:
        failures.append(HardInvariant.METRIC_MISMATCH)
    if ann.backend_version != PGVECTOR_BACKEND_VERSION:
        failures.append(HardInvariant.BACKEND_VERSION_MISMATCH)

    authorized = set(exact.collection_ids)
    for match in ann.matches:
        if (
            match.application_scope_id != exact.application_scope_id
            or match.collection_id not in authorized
        ):
            failures.append(HardInvariant.FOREIGN_RESULT)
            break

    # Both backends rerank with the SAME S-21 scorers over the SAME canonical
    # vectors, so a disagreement is a defect rather than approximation.
    exact_scores = _scores(exact)
    ann_scores = _scores(ann)
    max_delta = 0.0
    for chunk_id in exact_scores.keys() & ann_scores.keys():
        delta = abs(exact_scores[chunk_id] - ann_scores[chunk_id])
        max_delta = max(max_delta, delta)
    if max_delta > SCORE_PARITY_TOLERANCE:
        failures.append(HardInvariant.SCORE_DISAGREEMENT)

    if not targets_ok:
        failures.append(HardInvariant.COLLECTION_SET_MISMATCH)
    return tuple(dict.fromkeys(failures)), max_delta


def evaluate_pgvector_candidate(
    scope,
    case,
    *,
    embedding_model_config,
    source_generation_probe=None,
) -> RetrievalEvaluationCaseResult:
    """Compare ONE query: exact S-21 oracle against the S-24 pgvector backend.

    Both branches receive the SAME frozen `scope` and the SAME `query_values`
    tuple - no re-resolution, no second embedding, no renormalization, no
    rounding. Anything else would measure authorization drift or provider
    variance and call it ANN approximation.

    `source_generation_probe` is an optional callable returning a comparable
    snapshot of corpus freshness. It is read before and after; a change makes
    the case INVALID rather than a data point, because the two observations then
    describe different corpora. Never retried automatically - one attempt is one
    coherent corpus state, and retrying until it passes is how a flaky result
    becomes a promotion.
    """
    if case.evidence_class not in VALID_EVIDENCE_CLASSES:
        raise EvaluationError(
            "invalid_evidence_class",
            "A case must declare a known evidence class.",
        )

    generation_before = (
        source_generation_probe() if source_generation_probe else None
    )

    exact_refusal = RefusalKind.NONE
    ann_refusal = RefusalKind.NONE
    exact = None
    ann = None

    try:
        exact = rank_semantic_vector_with_scope(
            scope,
            query_values=case.query_values,
            embedding_model_config=embedding_model_config,
            collection_id=case.collection_id,
            limit=case.limit,
        )
    except EXPECTED_BACKEND_ERRORS as exc:
        exact_refusal = classify_refusal(exc)

    try:
        ann = search_pgvector_ann_with_scope(
            scope,
            query_values=case.query_values,
            embedding_model_config=embedding_model_config,
            collection_id=case.collection_id,
            limit=case.limit,
        )
    except EXPECTED_BACKEND_ERRORS as exc:
        ann_refusal = classify_refusal(exc)

    generation_after = (
        source_generation_probe() if source_generation_probe else None
    )

    def shell(outcome, *, invariants=(), parity=None, relevance=None):
        reference = exact if exact is not None else ann
        return RetrievalEvaluationCaseResult(
            case_id=case.case_id,
            evidence_class=case.evidence_class,
            outcome=outcome,
            application_scope_id=(
                reference.application_scope_id if reference else None
            ),
            collection_ids=(
                tuple(reference.collection_ids) if reference else ()
            ),
            embedding_model_config_id=getattr(
                embedding_model_config, "pk", None
            ),
            e1=reference.e1 if reference else "",
            metric=reference.metric if reference else "",
            backend_version=PGVECTOR_BACKEND_VERSION,
            limit=case.limit,
            exact_refusal=exact_refusal,
            ann_refusal=ann_refusal,
            parity=parity,
            relevance=relevance,
            hard_invariants=tuple(invariants),
        )

    if generation_before != generation_after:
        # Never averaged in, never retried.
        return shell(
            CaseOutcome.SOURCE_CHANGED,
            invariants=(HardInvariant.SOURCE_CHANGED_DURING_EVALUATION,),
        )

    if (exact is None) != (ann is None):
        # One answered where the other refused. A semantics mismatch, and
        # deliberately NOT recorded as a recall of zero.
        return shell(
            CaseOutcome.REFUSAL_MISMATCH,
            invariants=(HardInvariant.REFUSAL_SEMANTICS_MISMATCH,),
        )

    if exact is None and ann is None:
        if exact_refusal != ann_refusal:
            return shell(
                CaseOutcome.REFUSAL_MISMATCH,
                invariants=(HardInvariant.REFUSAL_SEMANTICS_MISMATCH,),
            )
        return shell(CaseOutcome.BOTH_REFUSED)

    targets_ok = tuple(sorted(exact.collection_ids)) == tuple(
        sorted(ann.collection_ids)
    )
    invariants, max_delta = _check_hard_invariants(case, exact, ann, targets_ok)

    exact_ids = _ids(exact)
    ann_ids = _ids(ann)
    displacements = rank_displacements(ann_ids, exact_ids)

    usable_cutoffs = tuple(
        cutoff for cutoff in EVALUATION_CUTOFFS if cutoff <= case.limit
    )
    parity = OracleParityMetrics(
        exact_result_count=len(exact_ids),
        ann_result_count=len(ann_ids),
        first_divergence_rank=first_divergence_rank(ann_ids, exact_ids),
        max_rank_displacement=max(displacements.values(), default=0),
        mean_rank_displacement=(
            sum(displacements.values()) / len(displacements)
            if displacements else 0.0
        ),
        score_parity_max_delta=max_delta,
        cutoffs=tuple(
            CutoffParity(
                cutoff=cutoff,
                oracle_recall=oracle_recall_at_k(ann_ids, exact_ids, cutoff),
                prefix_equal=prefix_equal_at_k(ann_ids, exact_ids, cutoff),
                catastrophic_miss=is_catastrophic_miss(
                    ann_ids, exact_ids, cutoff
                ),
            )
            for cutoff in usable_cutoffs
        ),
    )

    relevance = None
    judgments = case.judgment_map()
    if judgments:
        relevance = GoldenRelevanceMetrics(
            exact_mrr=mean_reciprocal_rank(exact_ids, judgments),
            ann_mrr=mean_reciprocal_rank(ann_ids, judgments),
            cutoffs=tuple(
                CutoffRelevance(
                    cutoff=cutoff,
                    precision=precision_at_k(ann_ids, judgments, cutoff),
                    recall=recall_at_k(ann_ids, judgments, cutoff),
                    ndcg=ndcg_at_k(ann_ids, judgments, cutoff),
                )
                for cutoff in usable_cutoffs
            ),
        )

    outcome = CaseOutcome.COMPARED
    if invariants:
        outcome = CaseOutcome.INVARIANT_VIOLATION
    elif not exact_ids and not ann_ids:
        outcome = CaseOutcome.BOTH_EMPTY

    return shell(
        outcome, invariants=invariants, parity=parity, relevance=relevance
    )


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_pgvector_suite(
    scope,
    cases,
    *,
    embedding_model_config,
    policy=None,
    source_generation_probe=None,
) -> RetrievalEvaluationSummary:
    """Evaluate every case under ONE frozen scope and return a verdict.

    The verdict is EVIDENCE. Nothing in this codebase consumes it to route a
    request, and that separation is the point: promotion has to be a decision
    somebody makes, not a side effect of a test passing.
    """
    policy = policy or PromotionPolicy()
    cases = tuple(cases)
    results = tuple(
        evaluate_pgvector_candidate(
            scope,
            case,
            embedding_model_config=embedding_model_config,
            source_generation_probe=source_generation_probe,
        )
        for case in cases
    )

    completed = tuple(
        r for r in results
        if r.outcome in (CaseOutcome.COMPARED, CaseOutcome.BOTH_EMPTY)
    )
    invalid = tuple(r for r in results if r.outcome == CaseOutcome.SOURCE_CHANGED)
    failed = tuple(
        r for r in results
        if r.outcome in (
            CaseOutcome.INVARIANT_VIOLATION, CaseOutcome.REFUSAL_MISMATCH
        )
    )
    refused = tuple(r for r in results if r.outcome == CaseOutcome.BOTH_REFUSED)
    # ONE notion of "accounted for", shared by the reported count and the
    # completeness guard below. The first draft used `completed` for the guard
    # and `completed + refused` for the count, so a suite of legitimate refusal
    # parity could be reported complete while failing the guard.
    accounted = completed + refused + invalid + failed

    cutoffs = sorted({
        entry.cutoff
        for r in completed if r.parity
        for entry in r.parity.cutoffs
    })

    def parity_values(cutoff, attribute):
        return [
            getattr(entry, attribute)
            for r in completed if r.parity
            for entry in r.parity.cutoffs if entry.cutoff == cutoff
        ]

    mean_oracle_recall = {
        k: _mean(parity_values(k, "oracle_recall")) for k in cutoffs
    }
    worst_oracle_recall = {
        k: min(parity_values(k, "oracle_recall"), default=0.0) for k in cutoffs
    }
    prefix_match_rate = {
        k: _mean(1.0 if v else 0.0 for v in parity_values(k, "prefix_equal"))
        for k in cutoffs
    }
    catastrophic = {
        k: sum(1 for v in parity_values(k, "catastrophic_miss") if v)
        for k in cutoffs
    }
    mean_ndcg = {
        k: _mean([
            entry.ndcg
            for r in completed if r.relevance
            for entry in r.relevance.cutoffs if entry.cutoff == k
        ])
        for k in cutoffs
    }
    mean_mrr = _mean(
        r.relevance.ann_mrr for r in completed if r.relevance
    )

    hard_failures = tuple(dict.fromkeys(
        invariant for r in results for invariant in r.hard_invariants
    ))

    reference = next(
        (r for r in results if r.e1), None
    )
    fingerprint = evaluation_fingerprint(
        policy=policy,
        backend_version=PGVECTOR_BACKEND_VERSION,
        e1=reference.e1 if reference else "",
        metric=reference.metric if reference else "",
        evidence_class=policy.required_evidence_class,
    )

    decision, reasons = _decide(
        policy=policy,
        cases=cases,
        results=results,
        accounted=accounted,
        completed=completed,
        invalid=invalid,
        failed=failed,
        hard_failures=hard_failures,
        mean_oracle_recall=mean_oracle_recall,
        worst_oracle_recall=worst_oracle_recall,
        prefix_match_rate=prefix_match_rate,
        catastrophic=catastrophic,
        mean_ndcg=mean_ndcg,
        mean_mrr=mean_mrr,
    )

    return RetrievalEvaluationSummary(
        evaluation_version=EVALUATION_VERSION,
        policy_version=policy.version,
        backend_version=PGVECTOR_BACKEND_VERSION,
        fingerprint=fingerprint,
        requested_case_count=len(cases),
        completed_case_count=len(completed) + len(refused),
        # `requested == completed + invalid + failed` holds by construction.
        invalid_case_count=len(invalid),
        failed_case_count=len(failed),
        controlled_case_count=sum(
            1 for c in cases if c.evidence_class == EvidenceClass.CONTROLLED
        ),
        representative_case_count=sum(
            1 for c in cases if c.evidence_class == EvidenceClass.REPRESENTATIVE
        ),
        mean_oracle_recall=mean_oracle_recall,
        worst_oracle_recall=worst_oracle_recall,
        prefix_match_rate=prefix_match_rate,
        catastrophic_miss_count=catastrophic,
        mean_ndcg=mean_ndcg,
        mean_mrr=mean_mrr,
        hard_invariant_failures=hard_failures,
        decision=decision,
        decision_reasons=reasons,
        cases=results,
    )


def _decide(
    *, policy, cases, results, accounted, completed, invalid, failed,
    hard_failures,
    mean_oracle_recall, worst_oracle_recall, prefix_match_rate, catastrophic,
    mean_ndcg, mean_mrr,
):
    """Hard invariants first, then evidence sufficiency, then quality.

    Order matters. A structural failure must never be reachable by
    `insufficient_evidence`, and no quality average may be consulted while one
    stands - that is what "zero tolerance" has to mean operationally.
    """
    reasons = []

    if hard_failures:
        return PromotionDecision.NOT_ELIGIBLE, tuple(hard_failures)
    if failed:
        return (
            PromotionDecision.NOT_ELIGIBLE,
            (HardInvariant.REFUSAL_SEMANTICS_MISMATCH,),
        )
    if invalid:
        return (
            PromotionDecision.NOT_ELIGIBLE,
            (HardInvariant.SOURCE_CHANGED_DURING_EVALUATION,),
        )

    # Evidence sufficiency. A controlled fixture proves the implementation is
    # correct; it can never stand in for a judged representative corpus, and no
    # perfect metric may buy its way past this.
    qualifying = [
        case for case in cases
        if case.evidence_class == policy.required_evidence_class
    ]
    if len(qualifying) < policy.minimum_case_count:
        reasons.append("insufficient_case_count")
    if len(qualifying) != len(cases):
        reasons.append("mixed_or_insufficient_evidence_class")
    if reasons:
        return PromotionDecision.INSUFFICIENT_EVIDENCE, tuple(reasons)
    if len(accounted) != len(cases):
        # A case that silently disappeared must not shrink the denominator.
        # Defensive: every known outcome is accounted for above, so reaching
        # this means a new outcome was added without teaching the summary about
        # it - which would otherwise let cases vanish from the arithmetic.
        return (
            PromotionDecision.NOT_ELIGIBLE,
            (HardInvariant.INCOMPLETE_EVALUATION,),
        )

    for cutoff, minimum in sorted(policy.minimum_mean_oracle_recall.items()):
        if mean_oracle_recall.get(cutoff, 0.0) + 1e-12 < minimum:
            reasons.append(f"mean_oracle_recall_at_{cutoff}")
    for cutoff, minimum in sorted(policy.minimum_worst_oracle_recall.items()):
        if worst_oracle_recall.get(cutoff, 0.0) + 1e-12 < minimum:
            reasons.append(f"worst_oracle_recall_at_{cutoff}")
    for cutoff, minimum in sorted(policy.minimum_prefix_match_rate.items()):
        if prefix_match_rate.get(cutoff, 0.0) + 1e-12 < minimum:
            reasons.append(f"prefix_match_rate_at_{cutoff}")
    if sum(catastrophic.values()) > policy.maximum_catastrophic_misses:
        reasons.append("catastrophic_miss")
    for cutoff, minimum in sorted(policy.minimum_mean_ndcg.items()):
        if mean_ndcg.get(cutoff, 0.0) + 1e-12 < minimum:
            reasons.append(f"mean_ndcg_at_{cutoff}")
    if mean_mrr + 1e-12 < policy.minimum_mean_mrr:
        reasons.append("mean_mrr")

    if reasons:
        return PromotionDecision.NOT_ELIGIBLE, tuple(reasons)
    return PromotionDecision.ELIGIBLE, ()
