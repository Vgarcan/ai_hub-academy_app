"""Reference-first retrieval audit: durable evidence for governed hybrid search.

    caller-only validation            (no run: not a retrieval operation)
              |
    ONE EffectiveKnowledgeScope
              |
    trusted audit namespace?          (no namespace -> no run at all)
              |
    RetrievalRun START  + target refs (one short transaction, no retrieval)
              |
    S-22 shared hybrid core           (branches, provider call, NO transaction)
              |
    integrity validation
              |
    RetrievalOutcome + hits           (one short transaction)
              |
    return

**Reference first.** Ids, ranks, fingerprints, contract identifiers, bounded
statuses, bounded failure codes, timestamps and counts. Never the query, never
Knowledge, never a vector, never a provider body. An audit row that carried what
was searched for and what was read would be a durable search-history database
and a second Knowledge store, arrived at by accident.

**There is deliberately no query hash.** Not `q1`, not `query_hash`, not
`sha256(query)`. A raw deterministic hash is still a durable identifier for what
somebody searched, and it is dictionary-attackable for any predictable query. No
keyed/HMAC privacy contract has been approved, so the correct amount of
query-derived durable data is zero. The query stays transient.

**A run is immutable; the outcome is appended.** Nothing here updates a
`RetrievalRun`. A run with no outcome therefore means exactly one thing - the
operation began and no terminal outcome was ever recorded - and that absence is
evidence worth keeping legible. A mutable `status` column would have erased it.

**Audit success is a precondition for returning a result.** If the terminal
evidence cannot be written, the caller does not get the answer. An audited API
that returned results it had failed to record would make every downstream claim
about "what retrieval did" unfalsifiable.

**Unexpected exceptions are not caught.** A programming error must propagate and
leave an incomplete run behind, because that is what actually happened. Wrapping
the operation in `except Exception` to manufacture a tidy outcome would turn
every future bug into a clean-looking audit row.

This module is INTERNAL. It creates no Tool, registers nothing in Admin, and is
not an Agent-facing surface.
"""

from dataclasses import dataclass

from django.db import transaction

from ai_hub.models import (
    RetrievalHit,
    RetrievalOutcome,
    RetrievalRun,
    RetrievalRunCollection,
)
from ai_hub.services.hybrid_retrieval import (
    FUSION_VERSION,
    HYBRID_BRANCH_DEPTH,
    RRF_K,
    HybridRetrievalError,
    HybridRetrievalResult,
    derive_hybrid_targets,
    search_hybrid_with_scope,
    validate_hybrid_request,
)
from ai_hub.services.knowledge_authorization import resolve_effective_knowledge_scope

#: What the persisted evidence MEANS - the shape and semantics of these rows.
#: Deliberately NOT a relevance-model version and not the fusion version: if a
#: later slice records different facts, that becomes a new evidence version
#: rather than a silent reinterpretation of every historical row.
EVIDENCE_VERSION = "retrieval1"


class AuditFailureCategory:
    """Bounded machine codes for audit-layer refusals. Never a message."""

    EVIDENCE_INTEGRITY_VIOLATION = "evidence_integrity_violation"


class RetrievalAuditError(RuntimeError):
    """The audit layer refused. Carries a bounded category and no content."""

    def __init__(self, category: str, message: str = "", *, retrieval_run_id=None):
        self.category = category
        self.retrieval_run_id = retrieval_run_id
        super().__init__(message or category)


class AuditedRetrievalRefused(RuntimeError):
    """A bounded retrieval refusal that HAS durable evidence.

    Raised instead of the underlying `HybridRetrievalError` so the caller gets
    the one thing the plain error cannot give them: the id of the run whose
    REFUSED outcome was just written. It carries the bounded failure category
    and nothing else - no query, no Knowledge, no provider response.
    """

    def __init__(self, category: str, *, retrieval_run_id):
        self.category = category
        self.retrieval_run_id = retrieval_run_id
        super().__init__(category)


@dataclass(frozen=True)
class AuditedHybridRetrievalResult:
    """The retrieval result, plus the id of the evidence that records it.

    `retrieval_run_id` is `None` in exactly one case: no trusted audit namespace
    existed, so there was nothing to scope a run to. See the DENY_ALL note in
    `audited_hybrid_search_knowledge_local`.
    """

    retrieval_run_id: int | None
    retrieval: HybridRetrievalResult


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalHitEvidence:
    final_rank: int
    chunk_id: int
    document_id: int
    collection_id: int
    k1: str
    lexical_rank: int | None
    semantic_rank: int | None


@dataclass(frozen=True)
class RetrievalOutcomeEvidence:
    outcome: str
    finished_at: object
    mode: str
    degraded: bool
    lexical_status: str
    semantic_status: str
    semantic_failure_kind: str
    failure_category: str
    e1: str
    semantic_metric: str
    lexical_candidates_scanned: int
    lexical_candidates_truncated: bool
    semantic_candidate_count: int
    semantic_scored_count: int
    returned_count: int


@dataclass(frozen=True)
class RetrievalEvidence:
    """Everything the audit knows about one run. Content-free by construction."""

    retrieval_run_id: int
    application_scope_id: int
    agent_id_snapshot: int
    workspace_id_snapshot: int | None
    embedding_model_config_id_snapshot: int | None
    provider_id_snapshot: int | None
    requested_limit: int
    evidence_version: str
    fusion_version: str
    rrf_k: int
    branch_depth: int
    started_at: object
    collection_ids: tuple
    outcome: RetrievalOutcomeEvidence | None
    hits: tuple


def read_retrieval_evidence(*, application_scope, retrieval_run_id):
    """One run's evidence, or `None`. Fail closed, and never an oracle.

    The namespace is part of the LOOKUP, not a check applied afterwards: fetching
    globally and then comparing scopes would mean the row was already in memory,
    and every difference in timing or error shape becomes a way to confirm that
    somebody else's run exists. A nonexistent id and a cross-scope id return the
    identical `None`.

    Deliberately does NOT take an Agent and does NOT consult
    `EffectiveKnowledgeScope`. Reading Knowledge and administering retrieval
    evidence are different concerns, and reusing the Knowledge authorization
    boundary here would quietly make every Agent an auditor. This service
    requires explicit `ApplicationScope` context and stays internal; an operator
    permission surface is separate future work.
    """
    scope_id = getattr(application_scope, "pk", None)
    if scope_id is None or retrieval_run_id is None:
        return None
    run = (
        RetrievalRun.objects.filter(
            pk=retrieval_run_id, application_scope_id=scope_id
        )
        .prefetch_related("target_collections", "hits")
        .first()
    )
    if run is None:
        return None

    outcome = RetrievalOutcome.objects.filter(retrieval_run=run).first()
    return RetrievalEvidence(
        retrieval_run_id=run.pk,
        application_scope_id=run.application_scope_id,
        agent_id_snapshot=run.agent_id_snapshot,
        workspace_id_snapshot=run.workspace_id_snapshot,
        embedding_model_config_id_snapshot=run.embedding_model_config_id_snapshot,
        provider_id_snapshot=run.provider_id_snapshot,
        requested_limit=run.requested_limit,
        evidence_version=run.evidence_version,
        fusion_version=run.fusion_version,
        rrf_k=run.rrf_k,
        branch_depth=run.branch_depth,
        started_at=run.started_at,
        collection_ids=tuple(
            sorted(
                row.collection_id_snapshot for row in run.target_collections.all()
            )
        ),
        outcome=(
            None
            if outcome is None
            else RetrievalOutcomeEvidence(
                outcome=outcome.outcome,
                finished_at=outcome.finished_at,
                mode=outcome.mode,
                degraded=outcome.degraded,
                lexical_status=outcome.lexical_status,
                semantic_status=outcome.semantic_status,
                semantic_failure_kind=outcome.semantic_failure_kind,
                failure_category=outcome.failure_category,
                e1=outcome.e1,
                semantic_metric=outcome.semantic_metric,
                lexical_candidates_scanned=outcome.lexical_candidates_scanned,
                lexical_candidates_truncated=outcome.lexical_candidates_truncated,
                semantic_candidate_count=outcome.semantic_candidate_count,
                semantic_scored_count=outcome.semantic_scored_count,
                returned_count=outcome.returned_count,
            )
        ),
        hits=tuple(
            RetrievalHitEvidence(
                final_rank=hit.final_rank,
                chunk_id=hit.chunk_id_snapshot,
                document_id=hit.document_id_snapshot,
                collection_id=hit.collection_id_snapshot,
                k1=hit.k1,
                lexical_rank=hit.lexical_rank,
                semantic_rank=hit.semantic_rank,
            )
            for hit in sorted(run.hits.all(), key=lambda row: row.final_rank)
        ),
    )


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------

def _start_run(
    scope, *, embedding_model_config, limit, target_ids
) -> RetrievalRun:
    """Create the run and its target references in ONE short transaction.

    No retrieval work, no provider call, no lexical scoring and no vector
    loading happen inside this block - a database transaction must never be open
    across a network inference.
    """
    with transaction.atomic():
        run = RetrievalRun.objects.create(
            application_scope_id=scope.application_scope_id,
            agent_id_snapshot=scope.agent_id,
            workspace_id_snapshot=scope.workspace_id,
            embedding_model_config_id_snapshot=getattr(
                embedding_model_config, "pk", None
            ),
            provider_id_snapshot=getattr(
                embedding_model_config, "provider_id", None
            ),
            requested_limit=limit,
            evidence_version=EVIDENCE_VERSION,
            fusion_version=FUSION_VERSION,
            rrf_k=RRF_K,
            branch_depth=HYBRID_BRANCH_DEPTH,
        )
        for collection_id in target_ids:
            RetrievalRunCollection.objects.create(
                retrieval_run=run, collection_id_snapshot=collection_id
            )
    return run


def _validate_evidence(run, result, evidence, target_ids):
    """Refuse to record evidence that disagrees with itself.

    The audit is only worth keeping if it cannot be made to say something the
    computation did not do. So before anything is written: the result must
    belong to the namespace, agent, workspace, model and fusion contract this
    run recorded; every returned match must sit inside the authorized target
    set; and the internal hit evidence must correspond one-for-one with the
    public matches.

    A mismatch is REFUSED, never repaired. Silently relocating a foreign hit
    into the run's namespace would be the audit inventing history.
    """
    def fail(detail):
        raise RetrievalAuditError(
            AuditFailureCategory.EVIDENCE_INTEGRITY_VIOLATION,
            f"Retrieval evidence is inconsistent: {detail}.",
            retrieval_run_id=run.pk,
        )

    if result.application_scope_id != run.application_scope_id:
        fail("application scope")
    if result.agent_id != run.agent_id_snapshot:
        fail("agent")
    if result.workspace_id != run.workspace_id_snapshot:
        fail("workspace")
    if result.fusion_version != run.fusion_version:
        fail("fusion version")
    if result.rrf_k != run.rrf_k:
        fail("rrf k")
    if result.branch_depth != run.branch_depth:
        fail("branch depth")
    if tuple(sorted(result.collection_ids)) != tuple(sorted(target_ids)):
        fail("target collections")

    # The public result reports the configuration it was ASKED to use, which the
    # run snapshotted from the same argument. A blank/None result value is the
    # legitimate pre-semantic empty case.
    if (
        result.embedding_model_config_id is not None
        and result.embedding_model_config_id
        != run.embedding_model_config_id_snapshot
    ):
        fail("embedding model configuration")

    identities = evidence.final_hit_identities
    if len(identities) != len(result.matches):
        fail("hit count")
    for identity, match in zip(identities, result.matches):
        if (
            identity.final_rank != match.rank
            or identity.chunk_id != match.chunk_id
            or identity.document_id != match.document_id
            or identity.collection_id != match.collection_id
            or identity.lexical_rank != match.lexical_rank
            or identity.semantic_rank != match.semantic_rank
        ):
            fail("hit identity")
        if identity.application_scope_id != run.application_scope_id:
            fail("hit application scope")
        if identity.collection_id not in target_ids:
            fail("hit collection")
        if not identity.k1:
            fail("hit identity fingerprint")
        if identity.lexical_rank is None and identity.semantic_rank is None:
            fail("hit with no branch rank")

    expected_ranks = list(range(1, len(identities) + 1))
    if [identity.final_rank for identity in identities] != expected_ranks:
        fail("final ranks")


def _complete_run(run, result, evidence, target_ids):
    """Append the terminal outcome AND every hit, in ONE short transaction.

    Atomic on purpose. A half-written completion - an outcome saved and some
    hits missing - is worse than no completion at all, because it looks finished
    while describing a result that never existed.
    """
    _validate_evidence(run, result, evidence, target_ids)
    with transaction.atomic():
        RetrievalOutcome.objects.create(
            retrieval_run=run,
            outcome=RetrievalOutcome.Outcome.COMPLETED,
            mode=result.mode,
            degraded=result.degraded,
            lexical_status=result.lexical_status,
            semantic_status=result.semantic_status,
            semantic_failure_kind=result.semantic_failure_kind,
            failure_category="",
            e1=result.e1,
            semantic_metric=result.semantic_metric,
            lexical_candidates_scanned=evidence.lexical_candidates_scanned,
            lexical_candidates_truncated=evidence.lexical_candidates_truncated,
            semantic_candidate_count=evidence.semantic_candidate_count,
            semantic_scored_count=evidence.semantic_scored_count,
            returned_count=len(result.matches),
        )
        for identity in evidence.final_hit_identities:
            RetrievalHit.objects.create(
                retrieval_run=run,
                chunk_id_snapshot=identity.chunk_id,
                document_id_snapshot=identity.document_id,
                collection_id_snapshot=identity.collection_id,
                # The fingerprint CAPTURED at the composition boundary. Never
                # recomputed here: the chunk may already have changed, and the
                # audit must describe what was returned, not what the corpus
                # says now.
                k1=identity.k1,
                final_rank=identity.final_rank,
                lexical_rank=identity.lexical_rank,
                semantic_rank=identity.semantic_rank,
            )


def _refuse_run(run, category):
    """Append a REFUSED outcome. Bounded category only; never a message."""
    with transaction.atomic():
        RetrievalOutcome.objects.create(
            retrieval_run=run,
            outcome=RetrievalOutcome.Outcome.REFUSED,
            failure_category=category,
            returned_count=0,
        )


def audited_hybrid_search_knowledge_local(
    agent,
    *,
    query,
    embedding_model_config,
    workspace=None,
    collection_id=None,
    limit=5,
) -> AuditedHybridRetrievalResult:
    """Run ONE governed hybrid retrieval and record durable evidence of it.

    The canonical integration boundary. A later Tool or execution slice should
    call this rather than the computational S-22 API, so that anything an Agent
    actually retrieved leaves a record.

    Adds no authorization of its own: the same single `EffectiveKnowledgeScope`
    governs the search AND scopes the evidence. The audit observes the decisions
    S-15, S-17, S-21 and S-22 already made; it never makes one.
    """
    # -- 1. caller-only validation. NOT a retrieval operation ---------------
    # Deliberately before the run exists: an invalid call must not create a
    # durable record of a rejected free-text query.
    validate_hybrid_request(query=query, limit=limit)

    # -- 2. AUTHORIZATION. Resolved EXACTLY ONCE, and never again -----------
    scope = resolve_effective_knowledge_scope(agent, workspace=workspace)

    # -- 3. is there a trusted namespace to scope evidence to? --------------
    if scope.application_scope_id is None or scope.agent_id is None:
        # S-15's canonical DENY_ALL. There is no namespace this evidence could
        # honestly belong to, and inventing a global audit row for it would
        # create exactly the cross-tenant table `ApplicationScope` exists to
        # prevent. Security-denial telemetry with no trusted namespace is a
        # separate concern; it is not retrieval evidence.
        return AuditedHybridRetrievalResult(
            retrieval_run_id=None,
            retrieval=search_hybrid_with_scope(
                scope,
                query=query,
                embedding_model_config=embedding_model_config,
                collection_id=collection_id,
                limit=limit,
                target_ids=(),
            ).result,
        )

    # -- 4. the effective authorized target set -----------------------------
    # Only this is ever persisted. A requested-but-inaccessible collection id is
    # NOT recorded anywhere: doing so would rebuild, durably and queryably, the
    # oracle ADR-N5 exists to deny.
    target_ids = derive_hybrid_targets(scope, collection_id)

    # -- 5. the run exists BEFORE any retrieval work ------------------------
    # If start persistence fails, nothing is searched and no provider is called:
    # an audited operation that cannot establish its record must not proceed.
    run = _start_run(
        scope,
        embedding_model_config=embedding_model_config,
        limit=limit,
        target_ids=target_ids,
    )

    # -- 6. the retrieval itself. NO transaction, NO locks ------------------
    # Only bounded `HybridRetrievalError` is intercepted, and only to record the
    # refusal before re-raising it. Anything else - a programming error, an
    # assertion failure - propagates untouched and leaves this run without an
    # outcome, which is the truthful record of what happened.
    try:
        computation = search_hybrid_with_scope(
            scope,
            query=query,
            embedding_model_config=embedding_model_config,
            collection_id=collection_id,
            limit=limit,
            target_ids=target_ids,
        )
    except HybridRetrievalError as exc:
        _refuse_run(run, exc.category)
        raise AuditedRetrievalRefused(
            exc.category, retrieval_run_id=run.pk
        ) from exc

    # -- 7. terminal evidence, then and only then the result ----------------
    # If this raises, the caller does not get the result. An audited API that
    # returned an answer it had failed to record would make every later claim
    # about what retrieval did unfalsifiable.
    _complete_run(run, computation.result, computation.evidence, target_ids)

    return AuditedHybridRetrievalResult(
        retrieval_run_id=run.pk, retrieval=computation.result
    )
