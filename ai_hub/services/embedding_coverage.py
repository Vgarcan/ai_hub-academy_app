"""Operator corpus embedding: coverage report and governed execution (S-27).

S-20 built the only correct way to turn one Knowledge chunk into a stored
vector, `index_chunk_embedding_local()`. This module makes it reachable for a
whole corpus by an OPERATOR, and adds exactly three things on top of it:

    which chunks need work          a read-only coverage report
    may this run happen at all      run-level preconditions, before any call
    what happened                   ordering, tallying, a closed failure table

Every rule about WHETHER a chunk may be embedded, and every write, belongs to
S-17..S-20. This module never stores a vector, never calls a transport, never
computes `k1` or `e1` itself and never retries.

**Why coverage must be visible before anything routes to semantic retrieval.**
A semantic search over a corpus with no current vectors returns an
authorized-but-empty answer, and S-22 treats a complete branch with zero
matches as an answer rather than a degradation. An un-indexed corpus therefore
yields a lexical-only ranking that reports itself as a healthy hybrid one, and a
partially indexed corpus is ranked silently from whatever happens to be there.
Neither is a bug in those services. It is why the operator has to be able to
see, and close, the gap first.

**All target collections authorize, or nothing runs.** Same principle as S-21:
dropping a denied collection and indexing the rest produces a run that looks
complete and is not. The operator narrows explicitly instead.

**Never an Agent surface.** No `ToolDefinition`, no Orchestrator or GAME route,
no signal, no background task. Indexing sends Knowledge text to a provider, and
that is an operator decision.
"""

from dataclasses import dataclass

from ai_hub.models import (
    ApplicationScope,
    EmbeddingModelConfig,
    KnowledgeChunkEmbedding,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
)
from ai_hub.services.chunk_embedding_identity import canonical_chunk_embedding_text
from ai_hub.services.embedding_client import (
    EmbeddingProviderExecutionError,
    ErrorCategory,
    resolve_embedding_transport,
)
from ai_hub.services.embedding_contract import (
    EmbeddingContractError,
    resolve_embedding_contract,
)
from ai_hub.services.embedding_egress import (
    PAYLOAD_CORPUS,
    ReasonCode,
    resolve_embedding_access,
)
from ai_hub.services.embedding_execution import (
    EmbeddingExecutionError,
    ExecutionStatus,
    FailureCategory,
    index_chunk_embedding_local,
)
from ai_hub.services.vector_store import VectorStoreError, inspect_vector_record


COVERAGE_CONTRACT_VERSION = 1


class CoverageState:
    """Exactly one per eligible chunk."""

    CURRENT = "current"
    STALE = "stale"
    MISSING = "missing"
    NOT_EMBEDDABLE = "not_embeddable"


COVERAGE_STATES = (
    CoverageState.CURRENT,
    CoverageState.STALE,
    CoverageState.MISSING,
    CoverageState.NOT_EMBEDDABLE,
)

#: Only these states are ever sent to `index_chunk_embedding_local()`.
WORK_STATES = frozenset({CoverageState.MISSING, CoverageState.STALE})


class TargetErrorCategory:
    """Refusals of the run's TARGET. Raised before any coverage is computed."""

    SCOPE_NOT_FOUND = "scope_not_found"
    EMBEDDING_CONFIG_NOT_FOUND = "embedding_config_not_found"
    COLLECTION_NOT_FOUND = "collection_not_found"
    COLLECTION_OUTSIDE_SCOPE = "collection_outside_scope"


class IndexTargetError(ValueError):
    """The requested scope, configuration or collections cannot be a target."""

    def __init__(self, category: str, message: str = ""):
        self.category = category
        super().__init__(message or category)


class PreconditionCode:
    """Run-level refusals reported by this module, beyond S-17's reason codes."""

    EMBEDDING_CONTRACT_INVALID = "embedding_contract_invalid"
    LOCAL_ONLY_EXECUTION_REQUIRED = FailureCategory.LOCAL_ONLY_EXECUTION_REQUIRED


class ExtraFailureCategory:
    """Bounded outcomes this module adds to S-20's two vocabularies.

    `EMBEDDING_CONTRACT_INVALID` - the configuration stopped resolving mid-run.
    `VECTOR_STORE_REFUSED` - S-19 refused to encode the vector (for example a
    component that overflows float32). Neither carries content.
    """

    EMBEDDING_CONTRACT_INVALID = PreconditionCode.EMBEDDING_CONTRACT_INVALID
    VECTOR_STORE_REFUSED = "vector_store_refused"


# ---------------------------------------------------------------------------
# The closed failure table
# ---------------------------------------------------------------------------

#: Stop after the first occurrence. Each says something about the whole run -
#: permission, configuration, the provider, or the model's vector space - so
#: every later chunk would fail the same way, or hammer a provider that is down.
RUN_FATAL_CATEGORIES = frozenset({
    FailureCategory.SCOPE_MISMATCH,
    FailureCategory.EMBEDDING_NOT_AUTHORIZED,
    FailureCategory.LOCAL_ONLY_EXECUTION_REQUIRED,
    FailureCategory.VECTOR_DIMENSION_MISMATCH,
    FailureCategory.EMBEDDING_CONTRACT_CHANGED_AFTER_PROVIDER_CALL,
    ErrorCategory.INVALID_PROVIDER_CONFIGURATION,
    ErrorCategory.UNSUPPORTED_EMBEDDING_TRANSPORT,
    ErrorCategory.PROVIDER_UNREACHABLE,
    ErrorCategory.MODEL_NOT_FOUND,
    ErrorCategory.PROVIDER_RETURNED_ERROR,
    ErrorCategory.INVALID_PROVIDER_RESPONSE,
    ExtraFailureCategory.EMBEDDING_CONTRACT_INVALID,
    ExtraFailureCategory.VECTOR_STORE_REFUSED,
})

#: Record and continue. Each is a fact about one chunk, or one lost race.
CHUNK_LOCAL_CATEGORIES = frozenset({
    FailureCategory.CHUNK_MISSING,
    FailureCategory.DOCUMENT_NOT_ACTIVE,
    FailureCategory.EMBEDDING_INPUT_EMPTY,
    FailureCategory.EMBEDDING_INPUT_TOO_LARGE,
    FailureCategory.VECTOR_NON_FINITE,
    FailureCategory.ZERO_VECTOR_CANNOT_L2_NORMALIZE,
    FailureCategory.STALE_CHUNK_AFTER_PROVIDER_CALL,
})


def is_run_fatal(category: str) -> bool:
    """Classify a failure category. An unclassified category is FATAL.

    Failing closed is deliberate: a category added to S-20 later must not
    quietly become "skip this chunk and carry on" because nobody updated the
    table. The tests enumerate both vocabularies so that never happens silently.
    """
    return category not in CHUNK_LOCAL_CATEGORIES


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexTarget:
    """What one run is about. Resolved once; never re-resolved mid-run."""

    application_scope: ApplicationScope
    embedding_model_config: EmbeddingModelConfig
    collection_ids: tuple
    collections_explicit: bool
    #: Inactive collections of the scope left out of an implicit target. They
    #: hold no retrievable Knowledge; naming one explicitly is still refused by
    #: S-17 (`collection_inactive`), never skipped.
    excluded_inactive_collection_ids: tuple


def _lookup(model, ref, *, text_field, category, label):
    text = str(ref if ref is not None else "").strip()
    if not text:
        raise IndexTargetError(category, f"A {label} is required.")
    lookup = {"pk": int(text)} if text.isdigit() else {text_field: text}
    try:
        return model.objects.get(**lookup)
    except model.DoesNotExist as exc:
        raise IndexTargetError(category, f"No {label} matches {text!r}.") from exc


def resolve_index_target(*, scope_ref, embedding_config_ref, collection_ids=None):
    """Resolve the explicit target of a run. No default of any kind.

    `scope_ref` is a slug or a numeric id; `embedding_config_ref` a name or a
    numeric id. There is deliberately no "the only active scope" and no "first
    active configuration": an indexing run must name what it writes to.
    """
    scope = _lookup(
        ApplicationScope, scope_ref, text_field="slug",
        category=TargetErrorCategory.SCOPE_NOT_FOUND, label="application scope",
    )
    config = _lookup(
        EmbeddingModelConfig, embedding_config_ref, text_field="name",
        category=TargetErrorCategory.EMBEDDING_CONFIG_NOT_FOUND,
        label="embedding model configuration",
    )

    if collection_ids:
        requested = sorted(set(int(value) for value in collection_ids))
        found = {
            row["pk"]: row["application_scope_id"]
            for row in KnowledgeCollection.objects.filter(pk__in=requested).values(
                "pk", "application_scope_id"
            )
        }
        for collection_id in requested:
            if collection_id not in found:
                raise IndexTargetError(
                    TargetErrorCategory.COLLECTION_NOT_FOUND,
                    f"No Knowledge collection has id {collection_id}.",
                )
            if found[collection_id] != scope.pk:
                # Refused, never silently dropped: the operator believed this
                # collection belonged to the scope being indexed.
                raise IndexTargetError(
                    TargetErrorCategory.COLLECTION_OUTSIDE_SCOPE,
                    f"Collection {collection_id} does not belong to this scope.",
                )
        return IndexTarget(
            application_scope=scope,
            embedding_model_config=config,
            collection_ids=tuple(requested),
            collections_explicit=True,
            excluded_inactive_collection_ids=(),
        )

    rows = list(
        KnowledgeCollection.objects.filter(application_scope_id=scope.pk)
        .order_by("pk")
        .values_list("pk", "is_active")
    )
    return IndexTarget(
        application_scope=scope,
        embedding_model_config=config,
        collection_ids=tuple(pk for pk, active in rows if active),
        collections_explicit=False,
        excluded_inactive_collection_ids=tuple(pk for pk, active in rows if not active),
    )


# ---------------------------------------------------------------------------
# Run-level preconditions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CollectionDecision:
    collection_id: int
    allowed: bool
    code: str


@dataclass(frozen=True)
class RunPreconditions:
    """Whether a run may send anything, decided before any provider call."""

    contract: object  # ResolvedEmbeddingContract | None
    contract_error: str
    collection_decisions: tuple
    transport_error: str

    @property
    def ok(self) -> bool:
        return (
            self.contract is not None
            and not self.contract_error
            and not self.transport_error
            and all(decision.allowed for decision in self.collection_decisions)
        )

    @property
    def refusal_codes(self) -> tuple:
        codes = []
        if self.contract_error:
            codes.append(self.contract_error)
        codes.extend(
            decision.code
            for decision in self.collection_decisions
            if not decision.allowed
        )
        if self.transport_error:
            codes.append(self.transport_error)
        return tuple(sorted(set(codes)))


def evaluate_run_preconditions(target: IndexTarget) -> RunPreconditions:
    """Contract, S-17 decision per target collection, transport capability.

    Evaluated identically in report mode and in execute mode, so an operator can
    fix configuration without ever triggering a provider call. Each check reuses
    the S-17..S-20 function that owns it; none is reimplemented.
    """
    config = target.embedding_model_config
    try:
        contract = resolve_embedding_contract(config)
    except EmbeddingContractError:
        return RunPreconditions(
            contract=None,
            contract_error=PreconditionCode.EMBEDDING_CONTRACT_INVALID,
            collection_decisions=(),
            transport_error="",
        )

    provider = config.provider
    collections = {
        collection.pk: collection
        for collection in KnowledgeCollection.objects.filter(
            pk__in=target.collection_ids
        )
    }
    decisions = []
    for collection_id in target.collection_ids:
        decision = resolve_embedding_access(
            target.application_scope, provider,
            collection=collections.get(collection_id),
            payload_kind=PAYLOAD_CORPUS,
        )
        if not decision.allowed:
            decisions.append(
                CollectionDecision(collection_id, False, decision.reason_code)
            )
        elif (
            decision.reason_code != ReasonCode.ALLOWED_LOCAL
            or decision.requires_external_egress
        ):
            # S-17 may allow EXTERNAL egress; S-20 still refuses it, and this
            # run must say so before sending anything rather than after.
            decisions.append(
                CollectionDecision(
                    collection_id, False, PreconditionCode.LOCAL_ONLY_EXECUTION_REQUIRED
                )
            )
        else:
            decisions.append(CollectionDecision(collection_id, True, decision.reason_code))

    try:
        resolve_embedding_transport(provider)
        transport_error = ""
    except EmbeddingProviderExecutionError as exc:
        transport_error = exc.category

    return RunPreconditions(
        contract=contract,
        contract_error="",
        collection_decisions=tuple(decisions),
        transport_error=transport_error,
    )


# ---------------------------------------------------------------------------
# Coverage (read-only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkCoverage:
    """One eligible chunk. Ids and bounded codes only - never text, never k1."""

    chunk_id: int
    document_id: int
    collection_id: int
    state: str
    reason_codes: tuple = ()


@dataclass(frozen=True)
class CoverageReport:
    target: IndexTarget
    preconditions: RunPreconditions
    #: `None` when the contract does not resolve: without `e1` and
    #: `max_input_chars` there is nothing truthful to say about coverage.
    chunks: tuple | None
    outside_eligibility_vector_count: int

    @property
    def coverage_available(self) -> bool:
        return self.chunks is not None

    def chunks_in(self, *states) -> tuple:
        if self.chunks is None:
            return ()
        wanted = set(states)
        return tuple(chunk for chunk in self.chunks if chunk.state in wanted)

    def counts(self) -> dict:
        counts = {state: 0 for state in COVERAGE_STATES}
        for chunk in self.chunks or ():
            counts[chunk.state] += 1
        return counts

    def stale_reason_counts(self) -> dict:
        counts = {}
        for chunk in self.chunks_in(CoverageState.STALE):
            for code in chunk.reason_codes:
                counts[code] = counts.get(code, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def complete(self) -> bool:
        """Every eligible chunk has a current vector."""
        return self.coverage_available and all(
            chunk.state == CoverageState.CURRENT for chunk in self.chunks
        )


def _not_embeddable_reason(chunk, *, max_input_chars) -> str:
    """The S-20 pre-dispatch input rules, applied for REPORTING only.

    `index_chunk_embedding_local()` stays authoritative: it re-checks both rules
    itself before any dispatch. This copy exists so the report can name the
    incompatibility without a provider call; a test binds the two together.
    """
    text = canonical_chunk_embedding_text(chunk)
    if text == "":
        return FailureCategory.EMBEDDING_INPUT_EMPTY
    if len(text) > max_input_chars:
        return FailureCategory.EMBEDDING_INPUT_TOO_LARGE
    return ""


def build_coverage_report(target: IndexTarget) -> CoverageReport:
    """Read-only coverage of the target under the configuration's `e1`.

    Eligible chunks are those of ACTIVE documents in the target collections.
    Currentness is S-19's `inspect_vector_record()`, never a local rule. Stale
    rows are reported and never deleted; rows whose document is no longer
    ACTIVE are counted as outside eligibility and left alone (spike §17 item 8:
    deletion versus retention is still undecided).
    """
    preconditions = evaluate_run_preconditions(target)
    contract = preconditions.contract
    if contract is None:
        return CoverageReport(
            target=target,
            preconditions=preconditions,
            chunks=None,
            outside_eligibility_vector_count=0,
        )

    chunks = list(
        KnowledgeDocumentChunk.objects.filter(
            document__collection_id__in=target.collection_ids,
            document__collection__application_scope_id=target.application_scope.pk,
            document__status=KnowledgeDocument.Status.ACTIVE,
        )
        .select_related("document")
        .order_by("pk")
    )
    records = {
        record.chunk_id: record
        for record in KnowledgeChunkEmbedding.objects.filter(
            chunk_id__in=[chunk.pk for chunk in chunks], e1=contract.e1
        ).select_related(
            "chunk", "chunk__document", "chunk__document__collection",
            "embedding_model_config", "embedding_model_config__provider",
        )
    }

    coverage = []
    for chunk in chunks:
        record = records.get(chunk.pk)
        inspection = inspect_vector_record(record) if record is not None else None
        if inspection is not None and inspection.current:
            state, reasons = CoverageState.CURRENT, ()
        else:
            blocked = _not_embeddable_reason(
                chunk, max_input_chars=contract.max_input_chars
            )
            if blocked:
                state, reasons = CoverageState.NOT_EMBEDDABLE, (blocked,)
            elif inspection is not None:
                state, reasons = CoverageState.STALE, tuple(inspection.reason_codes)
            else:
                state, reasons = CoverageState.MISSING, ()
        coverage.append(
            ChunkCoverage(
                chunk_id=chunk.pk,
                document_id=chunk.document_id,
                collection_id=chunk.document.collection_id,
                state=state,
                reason_codes=reasons,
            )
        )

    outside = (
        KnowledgeChunkEmbedding.objects.filter(
            e1=contract.e1,
            chunk__document__collection_id__in=target.collection_ids,
        )
        .exclude(chunk__document__status=KnowledgeDocument.Status.ACTIVE)
        .count()
    )

    return CoverageReport(
        target=target,
        preconditions=preconditions,
        chunks=tuple(coverage),
        outside_eligibility_vector_count=outside,
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkOutcome:
    chunk_id: int
    outcome: str  # ExecutionStatus value, or a bounded failure category


@dataclass(frozen=True)
class EmbeddingIndexRun:
    before: CoverageReport
    after: CoverageReport | None
    refused: bool
    outcomes: tuple
    stopped_by: str
    not_attempted_after_stop: int
    remaining_due_to_limit: int

    @property
    def attempted(self) -> int:
        return len(self.outcomes)

    def outcome_counts(self) -> dict:
        counts = {}
        for outcome in self.outcomes:
            counts[outcome.outcome] = counts.get(outcome.outcome, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def complete(self) -> bool:
        """True only when the target is fully indexed and nothing went wrong.

        A partial index must never look complete to a script: a refusal, a
        stop, a chunk-local failure, a `--limit` truncation or a chunk that
        cannot be embedded at all each make this False.
        """
        succeeded = {ExecutionStatus.STORED, ExecutionStatus.ALREADY_CURRENT}
        return (
            not self.refused
            and not self.stopped_by
            and self.remaining_due_to_limit == 0
            and all(outcome.outcome in succeeded for outcome in self.outcomes)
            and self.after is not None
            and self.after.complete
        )


def _index_one(target: IndexTarget, chunk_id: int) -> str:
    """One call to the S-20 boundary. Returns its status or a bounded category.

    Only bounded, typed errors are caught. Anything else is a defect and must
    surface as one, not become a tally line nobody investigates.
    """
    try:
        result = index_chunk_embedding_local(
            application_scope=target.application_scope,
            chunk=KnowledgeDocumentChunk(pk=chunk_id),
            embedding_model_config=target.embedding_model_config,
        )
    except KnowledgeDocumentChunk.DoesNotExist:
        # Deleted between the coverage read and this call.
        return FailureCategory.CHUNK_MISSING
    except EmbeddingExecutionError as exc:
        return exc.category
    except EmbeddingProviderExecutionError as exc:
        return exc.category
    except EmbeddingContractError:
        return ExtraFailureCategory.EMBEDDING_CONTRACT_INVALID
    except VectorStoreError:
        return ExtraFailureCategory.VECTOR_STORE_REFUSED
    return result.status


def execute_embedding_index(target: IndexTarget, *, limit=None) -> EmbeddingIndexRun:
    """Embed every MISSING or STALE eligible chunk, in `chunk_id` order.

    Refuses the whole run, with zero provider calls, unless every precondition
    holds. Then calls `index_chunk_embedding_local()` once per chunk - one
    chunk, one call, no concurrency, no retry - and stops at the first run-fatal
    category. `limit` caps the number of chunks attempted; what it leaves behind
    is counted, never hidden.
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None.")

    before = build_coverage_report(target)
    if not before.preconditions.ok:
        return EmbeddingIndexRun(
            before=before, after=None, refused=True, outcomes=(),
            stopped_by="", not_attempted_after_stop=0, remaining_due_to_limit=0,
        )

    work = [chunk.chunk_id for chunk in before.chunks_in(*WORK_STATES)]
    work.sort()
    selected = work if limit is None else work[:limit]
    remaining = len(work) - len(selected)

    outcomes = []
    stopped_by = ""
    for index, chunk_id in enumerate(selected):
        outcome = _index_one(target, chunk_id)
        outcomes.append(ChunkOutcome(chunk_id=chunk_id, outcome=outcome))
        if outcome not in (ExecutionStatus.STORED, ExecutionStatus.ALREADY_CURRENT) and is_run_fatal(outcome):
            stopped_by = outcome
            not_attempted = len(selected) - index - 1
            break
    else:
        not_attempted = 0

    return EmbeddingIndexRun(
        before=before,
        after=build_coverage_report(target),
        refused=False,
        outcomes=tuple(outcomes),
        stopped_by=stopped_by,
        not_attempted_after_stop=not_attempted,
        remaining_due_to_limit=remaining,
    )


# ---------------------------------------------------------------------------
# Serialization - ids, counts and bounded codes only
# ---------------------------------------------------------------------------

def _target_as_dict(target: IndexTarget) -> dict:
    config = target.embedding_model_config
    return {
        "application_scope_id": target.application_scope.pk,
        "application_scope_slug": target.application_scope.slug,
        "embedding_model_config_id": config.pk,
        "collection_ids": list(target.collection_ids),
        "collections_explicit": target.collections_explicit,
        "excluded_inactive_collection_ids": list(target.excluded_inactive_collection_ids),
    }


def _contract_as_dict(contract) -> dict | None:
    if contract is None:
        return None
    return {
        "e1": contract.e1,
        "provider_id": contract.provider_id,
        "provider_type": contract.provider_type,
        "declared_locality": contract.declared_locality,
        "model_name": contract.model_name,
        "model_revision": contract.model_revision,
        "vector_dimension": contract.vector_dimension,
        "distance_metric": contract.distance_metric,
        "normalization": contract.normalization,
        "max_input_chars": contract.max_input_chars,
    }


def report_as_dict(report: CoverageReport) -> dict:
    preconditions = report.preconditions
    return {
        "coverage_contract_version": COVERAGE_CONTRACT_VERSION,
        "target": _target_as_dict(report.target),
        "contract": _contract_as_dict(preconditions.contract),
        "preconditions": {
            "ok": preconditions.ok,
            "refusal_codes": list(preconditions.refusal_codes),
            "collections": [
                {
                    "collection_id": decision.collection_id,
                    "allowed": decision.allowed,
                    "code": decision.code,
                }
                for decision in preconditions.collection_decisions
            ],
        },
        "coverage_available": report.coverage_available,
        "counts": report.counts() if report.coverage_available else None,
        "eligible_chunks": len(report.chunks) if report.coverage_available else None,
        "stale_reasons": report.stale_reason_counts(),
        "not_embeddable": [
            {"chunk_id": chunk.chunk_id, "reason": chunk.reason_codes[0]}
            for chunk in report.chunks_in(CoverageState.NOT_EMBEDDABLE)
        ],
        "outside_eligibility_vectors": report.outside_eligibility_vector_count,
        "complete": report.complete,
    }


def run_as_dict(run: EmbeddingIndexRun) -> dict:
    return {
        "coverage_contract_version": COVERAGE_CONTRACT_VERSION,
        "refused": run.refused,
        "attempted": run.attempted,
        "outcomes": run.outcome_counts(),
        "failed_chunks": [
            {"chunk_id": outcome.chunk_id, "category": outcome.outcome}
            for outcome in run.outcomes
            if outcome.outcome not in (ExecutionStatus.STORED, ExecutionStatus.ALREADY_CURRENT)
        ],
        "stopped_by": run.stopped_by or None,
        "not_attempted_after_stop": run.not_attempted_after_stop,
        "remaining_due_to_limit": run.remaining_due_to_limit,
        "complete": run.complete,
        "before": report_as_dict(run.before),
        "after": report_as_dict(run.after) if run.after is not None else None,
    }
