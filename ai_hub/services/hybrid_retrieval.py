"""Governed hybrid retrieval: ONE authorization answer, two branches, rank fusion.

    resolve_effective_knowledge_scope(...)   <- exactly once, here
                    |
             frozen scope + target collections
              /                        \\
        lexical branch            semantic branch
              \\                        /
        final eligibility + identity revalidation
                    |
              rank-only fusion (rrf1)
                    |
           governed hybrid candidates

**One operation, one authorization answer.** Both branches consume the same
frozen `EffectiveKnowledgeScope`. If each branch resolved its own, a single
user-visible search could straddle two different authorization states - and the
result would be a blend of what the caller may see and what they may no longer
see, with nothing in the output to say which half was which. That is why neither
internal branch entry point is allowed to call the S-15 resolver at all.

**Fusion is rank-only, and that is a correctness requirement rather than a
preference.** The lexical score is an unnormalized keyword heuristic; the
semantic number is whatever the operator's distance metric means - higher-better
for cosine and dot product, LOWER-better for Euclidean. There is no calibrated
common scale between them, and inventing one (min-max normalization, weighted
sums, inverting a distance) would produce an ordering that looks principled and
is arbitrary. S-21 already did the only honest conversion: it turned metric
meaning into an ORDER. This module consumes that order and nothing else.

**A partial ranking is worse than no ranking.** A lexical branch that truncated
its candidate set does not know its own top-20 is globally correct, so its ranks
are omitted entirely rather than fused as if they were. If neither branch is
trustworthy, this refuses. A confident answer built on a ranking known to be
incomplete is the failure mode worth avoiding.

**Hybrid does not trust its own branches as an authorization boundary.** Both
are already governed, but the union is revalidated against `authorized_chunks`
and each branch's captured `k1` before anything is fused - and a removed
candidate does not keep its rank slot, because a rank slot is itself influence.

Deliberately NOT in this module: no relevance threshold, no no-answer policy, no
reranker, no LLM call, no learned or operator-configurable weights, no provider
fallback, no persistence, no Tool exposure, no schema.
"""

from dataclasses import dataclass

from django.core.exceptions import ValidationError

from ai_hub.services.chunk_embedding_identity import chunk_embedding_fingerprint
from ai_hub.services.embedding_client import (
    EmbeddingProviderExecutionError,
    ErrorCategory as ProviderErrorCategory,
)
from ai_hub.services.embedding_contract import EmbeddingContractError
from ai_hub.services.knowledge_authorization import (
    authorized_chunks,
    resolve_effective_knowledge_scope,
)
from ai_hub.services.knowledge_retrieval import (
    validated_lexical_query,
    rank_knowledge_chunks_with_scope,
)
from ai_hub.services.semantic_retrieval import (
    RetrievalFailureCategory,
    SemanticRetrievalError,
    canonical_query_embedding_text,
    search_semantic_with_scope,
)

#: The fusion contract, versioned so it can be replaced rather than tuned.
#:
#: `rrf1` is a REFERENCE contract, not a claim of optimality. `RRF_K = 60` is the
#: value the Reciprocal Rank Fusion literature uses; nothing here has measured it
#: against this corpus. If golden-query evaluation later justifies a different
#: constant, that must become a NEW fusion version - silently retuning `rrf1`
#: would change every historical ordering while the name still claimed
#: otherwise.
FUSION_VERSION = "rrf1"
RRF_K = 60

#: Each branch is asked for more than the caller wants, because fusion needs
#: depth to work. If each branch returned only the final `limit`, a chunk ranked
#: #6 by BOTH branches - a strong hybrid signal - could never beat a chunk that
#: appeared in only one branch's top 5.
HYBRID_BRANCH_DEPTH = 20

#: The reference ranking is exact and in-process, so the answer stays small.
MAX_HYBRID_RESULTS = 20


class BranchStatus:
    """What actually happened to one retrieval branch. Small and explicit."""

    USED = "used"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"
    INCOMPLETE = "incomplete"
    NOT_RUN = "not_run"


class HybridMode:
    """Which trustworthy branches actually contributed to the ranking."""

    HYBRID = "hybrid"
    LEXICAL_ONLY = "lexical_only"
    SEMANTIC_ONLY = "semantic_only"
    EMPTY = "empty"


class SemanticFailureKind:
    """Bounded, high-level reasons the semantic branch was unavailable.

    Deliberately coarse and content-free. A raw exception message can carry a
    provider body, a configuration detail or - worst - an echo of the query, and
    a degraded search must not become a side channel for any of them.
    """

    NONE = ""
    POLICY = "policy"
    CONFIGURATION = "configuration"
    CAPABILITY = "capability"
    PROVIDER = "provider"
    QUERY_INCOMPATIBLE = "query_incompatible"
    REFERENCE_LIMIT = "reference_limit"
    #: An unmapped category. Deliberately its own value rather than folded into
    #: one of the above: mislabelling an unknown failure is worse than admitting
    #: the mapping has a gap.
    UNKNOWN = "unknown"


class DegradationReason:
    LEXICAL_INCOMPLETE = "lexical_incomplete"
    SEMANTIC_UNAVAILABLE = "semantic_unavailable"


class HybridFailureCategory:
    INVALID_LIMIT = "invalid_limit"
    QUERY_EMPTY = "query_empty"
    NO_COMPLETE_RETRIEVAL_BRANCH = "no_complete_retrieval_branch"


class HybridRetrievalError(RuntimeError):
    """A refusal, carrying a bounded category and never the query."""

    def __init__(self, category: str, message: str = ""):
        self.category = category
        super().__init__(message or category)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

_SEMANTIC_FAILURE_KINDS = {
    # Authorization and policy outcomes.
    RetrievalFailureCategory.EMBEDDING_NOT_AUTHORIZED: SemanticFailureKind.POLICY,
    RetrievalFailureCategory.SCOPE_UNAVAILABLE: SemanticFailureKind.POLICY,
    # This build cannot execute it; that is capability, never a statement about
    # where the provider sits.
    RetrievalFailureCategory.LOCAL_ONLY_EXECUTION_REQUIRED:
        SemanticFailureKind.CAPABILITY,
    RetrievalFailureCategory.UNSUPPORTED_DISTANCE_METRIC:
        SemanticFailureKind.CONFIGURATION,
    RetrievalFailureCategory.CANDIDATE_DIMENSION_MISMATCH:
        SemanticFailureKind.CONFIGURATION,
    RetrievalFailureCategory.UNSCORABLE_ZERO_VECTOR:
        SemanticFailureKind.CONFIGURATION,
    # The query cannot be embedded under this contract.
    RetrievalFailureCategory.QUERY_INPUT_TOO_LARGE:
        SemanticFailureKind.QUERY_INCOMPATIBLE,
    RetrievalFailureCategory.QUERY_EMPTY: SemanticFailureKind.QUERY_INCOMPATIBLE,
    RetrievalFailureCategory.INVALID_LIMIT: SemanticFailureKind.CONFIGURATION,
    # The reference backend cannot honestly rank this corpus.
    RetrievalFailureCategory.REFERENCE_CANDIDATE_LIMIT_EXCEEDED:
        SemanticFailureKind.REFERENCE_LIMIT,
    # What the provider actually returned.
    RetrievalFailureCategory.VECTOR_DIMENSION_MISMATCH: SemanticFailureKind.PROVIDER,
    RetrievalFailureCategory.VECTOR_NON_FINITE: SemanticFailureKind.PROVIDER,
    RetrievalFailureCategory.ZERO_VECTOR_CANNOT_L2_NORMALIZE:
        SemanticFailureKind.PROVIDER,
    # Transport-level categories.
    ProviderErrorCategory.INVALID_PROVIDER_CONFIGURATION:
        SemanticFailureKind.CONFIGURATION,
    ProviderErrorCategory.UNSUPPORTED_EMBEDDING_TRANSPORT:
        SemanticFailureKind.CAPABILITY,
    ProviderErrorCategory.PROVIDER_UNREACHABLE: SemanticFailureKind.PROVIDER,
    ProviderErrorCategory.MODEL_NOT_FOUND: SemanticFailureKind.PROVIDER,
    ProviderErrorCategory.PROVIDER_RETURNED_ERROR: SemanticFailureKind.PROVIDER,
    ProviderErrorCategory.INVALID_PROVIDER_RESPONSE: SemanticFailureKind.PROVIDER,
}


def classify_semantic_failure(exc) -> str:
    """Map ONE bounded semantic failure to a bounded kind. Never a message.

    `EmbeddingContractError` carries no category - it means the operator's
    configuration cannot form a contract at all, which is exactly
    `configuration`.
    """
    if isinstance(exc, EmbeddingContractError):
        return SemanticFailureKind.CONFIGURATION
    return _SEMANTIC_FAILURE_KINDS.get(
        getattr(exc, "category", None), SemanticFailureKind.UNKNOWN
    )


#: The ONLY semantic failures hybrid absorbs. Deliberately enumerated: a
#: programming error is not a degraded search, and catching `Exception` here
#: would turn every future bug in the semantic path into a silent
#: "semantic unavailable" that nobody investigates.
SEMANTIC_DEGRADATION_ERRORS = (
    SemanticRetrievalError,
    EmbeddingContractError,
    EmbeddingProviderExecutionError,
)


# ---------------------------------------------------------------------------
# Pure fusion
# ---------------------------------------------------------------------------

def rrf_contribution(rank: int) -> float:
    """`1 / (RRF_K + rank)`, ranks starting at 1."""
    return 1.0 / (RRF_K + rank)


def fuse_ranked_branches(lexical_ranks, semantic_ranks):
    """Pure Reciprocal Rank Fusion over two rank maps. Ranks only.

    Takes `{chunk_id: rank}` and NOTHING else - not a score, not a metric, not a
    direction. The signature is the guarantee: a raw lexical score or a cosine
    value cannot influence this ordering because neither is reachable from here.

    Ties break on `chunk_id` ASC so equal fusion scores never reorder between
    runs, backends or database row orders.
    """
    fused = []
    for chunk_id in set(lexical_ranks) | set(semantic_ranks):
        # Named `total`, not `score`: nothing resembling a branch score exists
        # anywhere in this function, and the name should not suggest otherwise.
        total = 0.0
        lexical_rank = lexical_ranks.get(chunk_id)
        semantic_rank = semantic_ranks.get(chunk_id)
        if lexical_rank is not None:
            total += rrf_contribution(lexical_rank)
        if semantic_rank is not None:
            total += rrf_contribution(semantic_rank)
        fused.append((total, chunk_id, lexical_rank, semantic_rank))
    fused.sort(key=lambda entry: (-entry[0], entry[1]))
    return fused


def compress_branch_ranks(chunk_ids):
    """Renumber survivors 1..N, preserving order, first occurrence only.

    Load-bearing. If a dropped candidate kept its rank slot, an unauthorized or
    stale chunk would still influence the final ordering by pushing everything
    below it further down - influence without appearing anywhere, which is the
    hardest kind to notice.

    Deduplicating here means one chunk contributes at most once PER MODALITY,
    even if a branch handed back the same id twice.
    """
    ranks = {}
    for chunk_id in chunk_ids:
        if chunk_id in ranks:
            continue
        ranks[chunk_id] = len(ranks) + 1
    return ranks


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridMatch:
    """One fused chunk. Identifiers and an ordering value - never content.

    `fusion_score` is the RRF ORDERING VALUE and nothing else. It is NOT a
    probability, NOT a confidence, NOT a relevance percentage and NOT a measure
    of whether the corpus can answer the question. Its magnitude depends only on
    `RRF_K` and on which branches matched, so rendering it as "92% relevant"
    would be inventing a claim the number cannot support.
    """

    rank: int
    chunk_id: int
    document_id: int
    collection_id: int
    application_scope_id: int
    fusion_score: float
    lexical_rank: int | None
    semantic_rank: int | None


@dataclass(frozen=True)
class HybridRetrievalResult:
    """One hybrid operation. Ephemeral: never persisted, never a model.

    Reports the shape of the search - which namespace, which collections, which
    branches were trustworthy, whether it degraded and why - so an operator can
    audit a thin result without guessing whether it meant "denied", "nothing
    indexed", "provider down" or "nothing matched". It carries no query text, no
    Knowledge content, no snippet, no vector and no provider body.
    """

    matches: tuple

    application_scope_id: int | None
    agent_id: int | None
    workspace_id: int | None
    collection_ids: tuple

    fusion_version: str
    rrf_k: int
    branch_depth: int

    mode: str
    degraded: bool
    degradation_reasons: tuple

    lexical_status: str
    semantic_status: str
    semantic_failure_kind: str

    embedding_model_config_id: int | None
    e1: str
    semantic_metric: str


@dataclass(frozen=True)
class HybridHitIdentity:
    """One returned match as the composition boundary saw it. INTERNAL.

    Carries the `k1` captured at that boundary, which is the whole point: an
    auditor writing this down later must record what the caller was actually
    handed, not what the chunk says today. No text, no raw branch score, no
    vector.
    """

    final_rank: int
    chunk_id: int
    document_id: int
    collection_id: int
    application_scope_id: int
    k1: str
    lexical_rank: int | None
    semantic_rank: int | None


@dataclass(frozen=True)
class HybridRetrievalEvidence:
    """Bounded facts about HOW one hybrid computation ran. INTERNAL, not public.

    Deliberately not folded into `HybridRetrievalResult`: adding audit-only
    fields to a public contract because one consumer wants them is how public
    contracts grow things nobody asked for. Callers of the S-22 API see exactly
    what they saw before; only the audited boundary reads this.

    Counts and codes only - never the query, never Knowledge, never a vector,
    never a provider body.
    """

    lexical_candidates_scanned: int
    lexical_candidates_truncated: bool
    semantic_candidate_count: int
    semantic_scored_count: int
    #: The same bounded `SemanticFailureKind` the public result reports. One
    #: vocabulary, deliberately: a second failure taxonomy visible only to the
    #: audit would be a second thing to keep true.
    semantic_failure_code: str
    final_hit_identities: tuple


@dataclass(frozen=True)
class HybridComputation:
    """What the shared core returns: the public result plus internal evidence."""

    result: HybridRetrievalResult
    evidence: HybridRetrievalEvidence


EMPTY_EVIDENCE = HybridRetrievalEvidence(
    lexical_candidates_scanned=0,
    lexical_candidates_truncated=False,
    semantic_candidate_count=0,
    semantic_scored_count=0,
    semantic_failure_code=SemanticFailureKind.NONE,
    final_hit_identities=(),
)


def _empty_result(
    scope,
    *,
    collection_ids=(),
    lexical_status=BranchStatus.NOT_RUN,
    semantic_status=BranchStatus.NOT_RUN,
    semantic_failure_kind=SemanticFailureKind.NONE,
    embedding_model_config_id=None,
) -> HybridRetrievalResult:
    return HybridRetrievalResult(
        matches=(),
        application_scope_id=scope.application_scope_id,
        agent_id=scope.agent_id,
        workspace_id=scope.workspace_id,
        collection_ids=tuple(collection_ids),
        fusion_version=FUSION_VERSION,
        rrf_k=RRF_K,
        branch_depth=HYBRID_BRANCH_DEPTH,
        mode=HybridMode.EMPTY,
        degraded=False,
        degradation_reasons=(),
        lexical_status=lexical_status,
        semantic_status=semantic_status,
        semantic_failure_kind=semantic_failure_kind,
        embedding_model_config_id=embedding_model_config_id,
        e1="",
        semantic_metric="",
    )


# ---------------------------------------------------------------------------
# The public operation
# ---------------------------------------------------------------------------

def validate_hybrid_request(*, query, limit) -> None:
    """Caller-only validation. Runs BEFORE authorization, and writes nothing.

    Inspects the arguments and nothing else, so it leaks nothing about the
    corpus - and so a rejected request never becomes a durable record of what
    somebody typed. An invalid call is not a retrieval operation.
    """
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 0
        or limit > MAX_HYBRID_RESULTS
    ):
        # Refuse rather than clamp. Silently capping a caller's 500 to 20 answers
        # a question they did not ask and looks like a complete answer.
        raise HybridRetrievalError(
            HybridFailureCategory.INVALID_LIMIT,
            f"limit must be an integer between 0 and {MAX_HYBRID_RESULTS}.",
        )

    # S-21's contract, reused rather than reinvented: there must not be two
    # answers to "is this query blank".
    if canonical_query_embedding_text(query).strip() == "":
        raise HybridRetrievalError(
            HybridFailureCategory.QUERY_EMPTY, "A non-empty query is required."
        )


def derive_hybrid_targets(scope, collection_id) -> tuple:
    """The EFFECTIVE authorized target collections, narrowed but never widened.

    Returns an ordered tuple, possibly empty. An empty result covers three cases
    that must stay indistinguishable (ADR-N5): the scope authorizes nothing, the
    requested collection does not exist, or it exists somewhere this caller
    cannot reach. Collapsing them here is what stops a durable audit row from
    later becoming an oracle for which collections exist.
    """
    if scope.is_empty:
        return ()
    if collection_id is not None:
        if not scope.allows(collection_id):
            return ()
        return (int(collection_id),)
    return tuple(sorted(scope.collection_ids))


def search_hybrid_with_scope(
    scope,
    *,
    query,
    embedding_model_config,
    collection_id=None,
    limit=5,
    target_ids=(),
) -> HybridComputation:
    """The ONE hybrid implementation, inside an ALREADY-RESOLVED scope.

    Deliberately does NOT call `resolve_effective_knowledge_scope`, and does not
    re-derive the target set. Both the public S-22 API and the audited S-23
    boundary funnel through here, so there is exactly one hybrid algorithm - two
    would drift, and the audited one would end up recording something the
    unaudited one never did.
    """
    ordered_targets = tuple(sorted(target_ids))

    # -- 4. nothing to search, or nothing asked for ------------------------
    if not ordered_targets or limit == 0:
        return HybridComputation(
            result=_empty_result(scope, collection_ids=ordered_targets),
            evidence=EMPTY_EVIDENCE,
        )

    # -- 5. lexical branch, at FUSION depth --------------------------------
    # Depth, not the caller's limit: fusion needs more than the final answer.
    try:
        lexical_query = validated_lexical_query(query)
    except ValidationError as exc:
        # Unreachable through the canonical blank check above, which is stricter
        # for every input that reaches here. Mapped rather than left to escape as
        # a different exception type from a different layer.
        raise HybridRetrievalError(
            HybridFailureCategory.QUERY_EMPTY, "A non-empty query is required."
        ) from exc

    lexical = rank_knowledge_chunks_with_scope(
        scope,
        query=lexical_query,
        collection_ids=ordered_targets,
        limit=HYBRID_BRANCH_DEPTH,
        capture_identity=True,
    )
    lexical_complete = not lexical.candidates_truncated

    # -- 6. semantic branch, same frozen scope, same depth -----------------
    semantic = None
    semantic_failure_kind = SemanticFailureKind.NONE
    try:
        semantic = search_semantic_with_scope(
            scope,
            query=query,
            embedding_model_config=embedding_model_config,
            collection_id=collection_id,
            limit=HYBRID_BRANCH_DEPTH,
        )
    except SEMANTIC_DEGRADATION_ERRORS as exc:
        # A bounded, expected semantic failure. It never authorizes another
        # provider, another model or another `e1` - the branch is simply
        # unavailable for this operation.
        semantic_failure_kind = classify_semantic_failure(exc)

    semantic_available = semantic is not None

    # -- 7. no trustworthy branch -> refuse --------------------------------
    if not lexical_complete and not semantic_available:
        # A correct refusal beats a confident ranking built on a lexical list
        # that is known not to be globally correct.
        raise HybridRetrievalError(
            HybridFailureCategory.NO_COMPLETE_RETRIEVAL_BRANCH,
            "Neither retrieval branch produced a complete, usable ranking.",
        )

    # -- 8. FINAL revalidation, before anything is fused -------------------
    # Both branches are governed, but hybrid still fails closed: a branch is a
    # collaborator, not an authorization boundary. Time also passed while the
    # semantic provider was answering.
    lexical_entries = lexical.candidates if lexical_complete else ()
    semantic_entries = semantic.matches if semantic_available else ()

    union_ids = {entry.chunk_id for entry in lexical_entries}
    union_ids |= {match.chunk_id for match in semantic_entries}

    surviving = {}
    if union_ids:
        surviving = {
            chunk.pk: chunk
            for chunk in authorized_chunks(scope).filter(
                pk__in=union_ids, document__collection_id__in=ordered_targets
            )
        }
    # One fingerprint per surviving chunk, shared by both CAS checks.
    current_k1 = {
        chunk_id: chunk_embedding_fingerprint(chunk)
        for chunk_id, chunk in surviving.items()
    }

    def _survivors(entries):
        """Drop anything unauthorized or whose captured identity went stale.

        Per MODALITY. The same chunk can legitimately survive one branch and
        lose the other - a lexical rank captured before an edit is stale even
        when a semantic match taken after that edit is current - so a stale
        lexical contribution must not take a valid semantic one down with it.
        """
        kept = []
        for entry in entries:
            chunk = surviving.get(entry.chunk_id)
            if chunk is None:
                continue
            if current_k1.get(entry.chunk_id) != entry.k1:
                continue
            kept.append(entry.chunk_id)
        return kept

    # -- 9. compress ranks, THEN fuse --------------------------------------
    lexical_ranks = compress_branch_ranks(_survivors(lexical_entries))
    semantic_ranks = compress_branch_ranks(_survivors(semantic_entries))
    fused = fuse_ranked_branches(lexical_ranks, semantic_ranks)

    identity = {
        chunk_id: surviving[chunk_id]
        for chunk_id in set(lexical_ranks) | set(semantic_ranks)
    }
    matches = []
    hit_identities = []
    for position, (score, chunk_id, lexical_rank, semantic_rank) in enumerate(
        fused[:limit], start=1
    ):
        chunk = identity[chunk_id]
        document_id = chunk.document_id
        collection_pk = chunk.document.collection_id
        scope_pk = chunk.document.collection.application_scope_id
        matches.append(
            HybridMatch(
                rank=position,
                chunk_id=chunk_id,
                document_id=document_id,
                collection_id=collection_pk,
                application_scope_id=scope_pk,
                fusion_score=score,
                lexical_rank=lexical_rank,
                semantic_rank=semantic_rank,
            )
        )
        hit_identities.append(
            HybridHitIdentity(
                final_rank=position,
                chunk_id=chunk_id,
                document_id=document_id,
                collection_id=collection_pk,
                application_scope_id=scope_pk,
                # The `k1` this boundary VALIDATED, carried forward. An auditor
                # must never re-read the chunk to obtain it: the row could have
                # changed since, and the evidence would then describe a corpus
                # the caller never saw.
                k1=current_k1[chunk_id],
                lexical_rank=lexical_rank,
                semantic_rank=semantic_rank,
            )
        )
    matches = tuple(matches)

    # -- 10. report what actually happened ---------------------------------
    # Status is decided AFTER revalidation, so `used` never means "ran but
    # contributed nothing". `incomplete` and `unavailable` describe capability
    # and are decided before it.
    if not lexical_complete:
        lexical_status = BranchStatus.INCOMPLETE
    else:
        lexical_status = (
            BranchStatus.USED if lexical_ranks else BranchStatus.EMPTY
        )
    if not semantic_available:
        semantic_status = BranchStatus.UNAVAILABLE
    else:
        semantic_status = (
            BranchStatus.USED if semantic_ranks else BranchStatus.EMPTY
        )

    if lexical_ranks and semantic_ranks:
        mode = HybridMode.HYBRID
    elif lexical_ranks:
        mode = HybridMode.LEXICAL_ONLY
    elif semantic_ranks:
        mode = HybridMode.SEMANTIC_ONLY
    else:
        mode = HybridMode.EMPTY

    reasons = []
    if not lexical_complete:
        reasons.append(DegradationReason.LEXICAL_INCOMPLETE)
    if not semantic_available:
        reasons.append(DegradationReason.SEMANTIC_UNAVAILABLE)

    result = HybridRetrievalResult(
        matches=matches,
        application_scope_id=scope.application_scope_id,
        agent_id=scope.agent_id,
        workspace_id=scope.workspace_id,
        collection_ids=ordered_targets,
        fusion_version=FUSION_VERSION,
        rrf_k=RRF_K,
        branch_depth=HYBRID_BRANCH_DEPTH,
        mode=mode,
        # Reserved for missing CAPABILITY. A complete branch that simply matched
        # nothing is a real answer, not a degraded one.
        degraded=bool(reasons),
        degradation_reasons=tuple(reasons),
        lexical_status=lexical_status,
        semantic_status=semantic_status,
        semantic_failure_kind=semantic_failure_kind,
        embedding_model_config_id=getattr(embedding_model_config, "pk", None),
        e1=semantic.e1 if semantic_available else "",
        semantic_metric=semantic.metric if semantic_available else "",
    )
    evidence = HybridRetrievalEvidence(
        lexical_candidates_scanned=lexical.candidates_scanned,
        lexical_candidates_truncated=lexical.candidates_truncated,
        semantic_candidate_count=(
            semantic.candidate_count if semantic_available else 0
        ),
        semantic_scored_count=semantic.scored_count if semantic_available else 0,
        semantic_failure_code=semantic_failure_kind,
        final_hit_identities=tuple(hit_identities),
    )
    return HybridComputation(result=result, evidence=evidence)


def hybrid_search_knowledge_local(
    agent,
    *,
    query,
    embedding_model_config,
    workspace=None,
    collection_id=None,
    limit=5,
) -> HybridRetrievalResult:
    """Fuse the lexical and semantic rankings of ONE authorized search.

    The caller supplies who is asking, what they are asking and which vector
    space to ask in. It supplies no scope, no collection set, no `e1`, no metric,
    no weights, no `RRF_K` and no branch depth - those are Core-owned, because a
    caller-supplied authorization fact is not an authorization fact and a
    caller-supplied fusion weight is a relevance model nobody approved.

    **This is the computational API and it writes NOTHING.** It records no
    retrieval evidence, and it must never acquire that as a side effect - a
    caller reaching for a pure computation should not silently start populating
    an audit table. Durable evidence is the separate, explicit audited boundary
    in `retrieval_audit.py`.
    """
    validate_hybrid_request(query=query, limit=limit)
    scope = resolve_effective_knowledge_scope(agent, workspace=workspace)
    return search_hybrid_with_scope(
        scope,
        query=query,
        embedding_model_config=embedding_model_config,
        collection_id=collection_id,
        limit=limit,
        target_ids=derive_hybrid_targets(scope, collection_id),
    ).result
