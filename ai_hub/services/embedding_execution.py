"""Local corpus embedding execution: policy, normalization, CAS, persistence.

The one path that turns a Knowledge chunk into a stored vector:

    chunk -> canonical text / k1 (S-19)
          -> embedding contract / e1 (S-18)
          -> ProviderGrant + egress decision (S-17)
          -> LOCAL transport
          -> dimension / finite validation
          -> normalization contract (S-18)
          -> atomic persistence (S-19)

**The load-bearing problem this module exists to solve.** A provider call happens
outside the database transaction, so this is possible:

    render text A  ->  send A  ->  a human edits the chunk to B
                                ->  provider returns vector(A)
                                ->  storage computes current k1(B)

Storing `vector(A)` under `k1(B)` produces an index that is corrupt while
claiming to be current - the worst possible failure, because nothing downstream
can detect it. So the chunk identity and the contract identity are both
snapshotted before dispatch and re-verified under lock afterwards, and any
mismatch DISCARDS the provider result rather than persisting it.

**Local-only, deliberately.** Even when S-17 authorizes external egress
(`allowed_external`), S-20 refuses. External execution is a later slice, and a
capability this slice does not implement must not be reachable by configuration.

**Authorization is evaluated at DISPATCH.** Sending the text is the external side
effect, and it has already happened by the time the provider replies. S-20 does
not pretend it can retroactively revoke a sent request, and it deliberately does
NOT re-run S-17 afterwards to decide vector identity - authorization decides
whether to send, `k1`/`e1` decide what the result means.
"""

import math
from dataclasses import dataclass

from django.db import transaction

from ai_hub.models import (
    EmbeddingModelConfig,
    KnowledgeChunkEmbedding,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
)
from ai_hub.services.chunk_embedding_identity import (
    canonical_chunk_embedding_text,
    chunk_embedding_input_fingerprint,
)
from ai_hub.services.embedding_client import (
    EmbeddingProviderExecutionError,
    resolve_embedding_transport,
)
from ai_hub.services.embedding_contract import resolve_embedding_contract
from ai_hub.services.embedding_egress import (
    PAYLOAD_CORPUS,
    ReasonCode,
    resolve_embedding_access,
)
from ai_hub.services.vector_store import (
    inspect_vector_record,
    store_chunk_vector,
)


class ExecutionStatus:
    STORED = "stored"
    ALREADY_CURRENT = "already_current"


class FailureCategory:
    """Bounded machine-readable refusals. None carries content or secrets."""

    CHUNK_MISSING = "chunk_missing"
    SCOPE_MISMATCH = "scope_mismatch"
    DOCUMENT_NOT_ACTIVE = "document_not_active"
    EMBEDDING_NOT_AUTHORIZED = "embedding_not_authorized"
    LOCAL_ONLY_EXECUTION_REQUIRED = "local_only_execution_required"
    EMBEDDING_INPUT_EMPTY = "embedding_input_empty"
    EMBEDDING_INPUT_TOO_LARGE = "embedding_input_too_large"
    VECTOR_DIMENSION_MISMATCH = "vector_dimension_mismatch"
    VECTOR_NON_FINITE = "vector_non_finite"
    ZERO_VECTOR_CANNOT_L2_NORMALIZE = "zero_vector_cannot_l2_normalize"
    STALE_CHUNK_AFTER_PROVIDER_CALL = "stale_chunk_after_provider_call"
    EMBEDDING_CONTRACT_CHANGED_AFTER_PROVIDER_CALL = (
        "embedding_contract_changed_after_provider_call"
    )


class EmbeddingExecutionError(RuntimeError):
    """A refusal, carrying a bounded category and never submitted content."""

    def __init__(self, category: str, message: str = ""):
        self.category = category
        super().__init__(message or category)


@dataclass(frozen=True)
class ChunkEmbeddingExecutionResult:
    """Bounded outcome. No text, no vector values, no credentials."""

    status: str
    record_id: int | None
    chunk_id: int
    collection_id: int
    application_scope_id: int
    embedding_model_config_id: int
    provider_id: int
    k1: str
    e1: str
    vector_dimension: int


def _normalize_vector(values, *, normalization) -> tuple:
    """Apply the S-18 normalization contract. Never infer it.

    Deliberately does not look at the distance metric, the provider, the model
    or the vector's own norm: metric and normalization are two independent
    contract facts, and normalizing "because the metric is cosine" would make a
    stored vector disagree with the contract that describes it.
    """
    if normalization == EmbeddingModelConfig.Normalization.NONE:
        return tuple(values)

    magnitude = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        # A zero vector has no direction, so it cannot be L2-normalized. Refuse
        # rather than leaving it unchanged, substituting an epsilon or inventing
        # a unit vector - each of those stores a vector that lies about what the
        # contract says it is.
        raise EmbeddingExecutionError(
            FailureCategory.ZERO_VECTOR_CANNOT_L2_NORMALIZE,
            "A zero-magnitude vector cannot be L2-normalized.",
        )
    normalized = tuple(value / magnitude for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise EmbeddingExecutionError(
            FailureCategory.VECTOR_NON_FINITE,
            "Normalization produced a non-finite component.",
        )
    return normalized


def _validate_raw_vector(values, *, expected_dimension: int) -> tuple:
    """Re-check the provider result here, before normalization.

    S-19's encoder validates too, but normalization happens in between - a
    non-finite component would become a non-finite magnitude and the failure
    would surface as something less specific and further from its cause.
    """
    if len(values) != expected_dimension:
        raise EmbeddingExecutionError(
            FailureCategory.VECTOR_DIMENSION_MISMATCH,
            f"Provider returned {len(values)} components; "
            f"the contract requires {expected_dimension}.",
        )
    for index, raw in enumerate(values):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise EmbeddingExecutionError(
                FailureCategory.VECTOR_NON_FINITE,
                f"Vector component {index} is not a number.",
            )
        if not math.isfinite(float(raw)):
            raise EmbeddingExecutionError(
                FailureCategory.VECTOR_NON_FINITE,
                f"Vector component {index} is not finite.",
            )
    return tuple(float(value) for value in values)


def index_chunk_embedding_local(
    *, application_scope, chunk, embedding_model_config
) -> ChunkEmbeddingExecutionResult:
    """Embed one Knowledge chunk through a LOCAL provider and persist the vector.

    The caller supplies only the namespace it believes it is writing to, the
    chunk and the configuration. Collection, provider, canonical text, `k1`,
    `e1`, dimension and normalization are all derived - a caller-supplied
    fingerprint or text would be a way to make a wrong vector look right.

    No Agent, no `EffectiveKnowledgeScope`, no workspace: corpus index
    construction is not retrieval, and those concepts decide who may READ
    Knowledge. Security here is canonical chunk ownership plus the explicit
    scope plus S-17's grant and egress decision.
    """
    # -- 1. content-free structural checks ---------------------------------
    if chunk is None or getattr(chunk, "pk", None) is None:
        raise EmbeddingExecutionError(
            FailureCategory.CHUNK_MISSING, "A persisted Knowledge chunk is required."
        )
    if application_scope is None or getattr(application_scope, "pk", None) is None:
        raise EmbeddingExecutionError(
            FailureCategory.SCOPE_MISMATCH, "An application scope is required."
        )

    chunk = (
        KnowledgeDocumentChunk.objects.select_related(
            "document", "document__collection",
            "document__collection__application_scope",
        ).get(pk=chunk.pk)
    )
    document = chunk.document
    collection = document.collection

    if collection.application_scope_id != application_scope.pk:
        # Refuse rather than silently relocating: a mismatch means the caller
        # believed it was indexing something else.
        raise EmbeddingExecutionError(
            FailureCategory.SCOPE_MISMATCH,
            "The requested application scope does not own this chunk.",
        )

    # -- 2. only retrievable Knowledge is indexed --------------------------
    if document.status != KnowledgeDocument.Status.ACTIVE:
        # Deliberately NOT part of `k1` or `e1`: an already-stored vector may
        # outlive archival, and future retrieval excludes archived documents
        # independently.
        raise EmbeddingExecutionError(
            FailureCategory.DOCUMENT_NOT_ACTIVE,
            "Only ACTIVE Knowledge documents are indexed.",
        )

    # -- 3. the embedding contract -----------------------------------------
    contract = resolve_embedding_contract(embedding_model_config)
    provider = embedding_model_config.provider

    # -- 4. S-17 authorization, unchanged and never duplicated -------------
    # Resolved BEFORE the already-current shortcut, so the public operation is
    # consistently gated: an unauthorized caller learns nothing about whether a
    # vector happens to exist.
    decision = resolve_embedding_access(
        application_scope, provider,
        collection=collection, payload_kind=PAYLOAD_CORPUS,
    )
    if not decision.allowed:
        raise EmbeddingExecutionError(
            FailureCategory.EMBEDDING_NOT_AUTHORIZED,
            f"Embedding is not authorized ({decision.reason_code}).",
        )

    # -- 5. local-only, by declared locality alone -------------------------
    if (
        decision.reason_code != ReasonCode.ALLOWED_LOCAL
        or decision.requires_external_egress
    ):
        # S-17 may legitimately allow EXTERNAL egress. S-20 still refuses,
        # because it does not implement external execution.
        raise EmbeddingExecutionError(
            FailureCategory.LOCAL_ONLY_EXECUTION_REQUIRED,
            "This operation executes only against providers declared LOCAL.",
        )

    # -- 6. transport capability (independent of locality) -----------------
    transport = resolve_embedding_transport(provider)

    # -- 7. only now is content rendered -----------------------------------
    embedding_text = canonical_chunk_embedding_text(chunk)
    expected_k1 = chunk_embedding_input_fingerprint(embedding_text)

    # -- 8. our own input ceiling, before the provider sees anything -------
    if embedding_text == "":
        # Compared to "" exactly - deliberately NOT `.strip()`, because a
        # whitespace-only canonical string is a real, distinct `k1` input and
        # treating it as empty would make this check disagree with the contract.
        raise EmbeddingExecutionError(
            FailureCategory.EMBEDDING_INPUT_EMPTY,
            "The canonical embedding text is empty.",
        )
    if len(embedding_text) > contract.max_input_chars:
        # Never truncate, slice, summarize or re-chunk. If the canonical chunk
        # exceeds the configured ceiling, the configuration and the content are
        # incompatible and that must be visible.
        raise EmbeddingExecutionError(
            FailureCategory.EMBEDDING_INPUT_TOO_LARGE,
            f"The canonical embedding text is {len(embedding_text)} characters; "
            f"the contract allows {contract.max_input_chars}.",
        )

    # -- the pre-dispatch snapshot, derived by Core ------------------------
    expected_chunk_id = chunk.pk
    expected_collection_id = collection.pk
    expected_scope_id = collection.application_scope_id
    expected_e1 = contract.e1

    # -- optional short circuit, using canonical S-19 inspection ----------
    existing = (
        KnowledgeChunkEmbedding.objects.select_related(
            "chunk", "chunk__document", "chunk__document__collection",
            "embedding_model_config", "embedding_model_config__provider",
        )
        .filter(chunk_id=expected_chunk_id, e1=expected_e1)
        .first()
    )
    if existing is not None and inspect_vector_record(existing).current:
        return ChunkEmbeddingExecutionResult(
            status=ExecutionStatus.ALREADY_CURRENT,
            record_id=existing.pk,
            chunk_id=expected_chunk_id,
            collection_id=expected_collection_id,
            application_scope_id=expected_scope_id,
            embedding_model_config_id=embedding_model_config.pk,
            provider_id=provider.pk,
            k1=existing.k1,
            e1=existing.e1,
            vector_dimension=existing.vector_dimension,
        )

    # -- 9. the provider call. OUTSIDE any transaction, holding no locks ---
    provider_result = transport(
        provider=provider, contract=contract, text=embedding_text
    )

    # -- validation, then the S-18 normalization contract ------------------
    raw_values = _validate_raw_vector(
        provider_result.values, expected_dimension=contract.vector_dimension
    )
    values = _normalize_vector(raw_values, normalization=contract.normalization)

    # -- 10. short transaction: lock, re-verify, store ---------------------
    with transaction.atomic():
        locked_chunk = (
            KnowledgeDocumentChunk.objects.select_for_update()
            .select_related(
                "document", "document__collection",
                "document__collection__application_scope",
            )
            .get(pk=expected_chunk_id)
        )
        locked_document = locked_chunk.document
        locked_collection = locked_document.collection

        # CHUNK identity CAS. This is the check that stops vector(A) being
        # stored under k1(B).
        if (
            locked_collection.pk != expected_collection_id
            or locked_collection.application_scope_id != expected_scope_id
            or locked_document.status != KnowledgeDocument.Status.ACTIVE
            or chunk_embedding_input_fingerprint(
                canonical_chunk_embedding_text(locked_chunk)
            ) != expected_k1
        ):
            # Discard the provider result. No automatic retry and no automatic
            # re-embed: a caller may explicitly run the operation again against
            # the new state, which is a decision, not a side effect.
            raise EmbeddingExecutionError(
                FailureCategory.STALE_CHUNK_AFTER_PROVIDER_CALL,
                "The chunk changed while the embedding request was in flight.",
            )

        # CONTRACT identity CAS. Never stamp an old vector with a new `e1`.
        locked_config = (
            EmbeddingModelConfig.objects.select_for_update()
            .select_related("provider")
            .get(pk=embedding_model_config.pk)
        )
        # The OPERATIONAL resolver: a new write still requires an active config
        # and provider (S-19), which also refuses if either was deactivated
        # while the request was in flight.
        current_contract = resolve_embedding_contract(locked_config)
        if current_contract.e1 != expected_e1:
            raise EmbeddingExecutionError(
                FailureCategory.EMBEDDING_CONTRACT_CHANGED_AFTER_PROVIDER_CALL,
                "The embedding contract changed while the request was in flight.",
            )

        # The ONLY canonical storage boundary. Never a direct model write, and
        # never a reimplementation of k1, e1 or f32le1.
        record = store_chunk_vector(
            application_scope=locked_collection.application_scope,
            chunk=locked_chunk,
            embedding_model_config=locked_config,
            vector=values,
        )

    return ChunkEmbeddingExecutionResult(
        status=ExecutionStatus.STORED,
        record_id=record.pk,
        chunk_id=record.chunk_id,
        collection_id=record.collection_id,
        application_scope_id=record.application_scope_id,
        embedding_model_config_id=record.embedding_model_config_id,
        provider_id=provider.pk,
        k1=record.k1,
        e1=record.e1,
        vector_dimension=record.vector_dimension,
    )
