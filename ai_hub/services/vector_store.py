"""Persistence for Knowledge chunk vectors. No embedding, no similarity, no ANN.

S-19 owns the persistence LIFECYCLE of a vector and nothing else. It never calls
an embedding provider, never produces a vector, never compares two vectors and
never ranks anything. `store_chunk_vector()` receives numbers that already
exist.

**Why there is no permission check here.** Storage does not send content
anywhere, so it deliberately does not consult `ProviderGrant`, the external
egress flags, `declared_locality` or `EmbeddingAccessDecision`. Those govern the
future operation that PRODUCES a vector by transmitting text to a provider, and
S-20 must satisfy S-17 BEFORE making that call. Folding egress into storage
would put the check in the one place where it cannot protect anything - the text
has already left by then.

Likewise this module knows nothing about Agents. `EffectiveKnowledgeScope`
(S-15) decides which collections a principal may read; `load_current_vectors()`
requires the caller to hand it collection ids it has ALREADY authorized. The
future composition is:

    EffectiveKnowledgeScope.collection_ids
            v
    load_current_vectors(application_scope=..., collection_ids=..., e1=...)

**The reference backend is not the production backend.** `DjangoBinaryVectorStore`
persists vectors as `f32le1` bytes in an ordinary `BinaryField`. It exists for
correctness, contract testing, local development and the future local-only
pipeline. It is explicitly NOT the production ANN backend, and its
`application_scope` COLUMN must not later be mistaken for structural ANN
isolation: a `WHERE` clause on one global ANN index does not satisfy the
approved requirement that ApplicationScope isolation be a structural
namespace/partition/table boundary, with collection authorization as a predicate
INSIDE that boundary.
"""

import math
import struct
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from django.db import transaction

from ai_hub.models import (
    EmbeddingModelConfig,
    KnowledgeChunkEmbedding,
    KnowledgeDocumentChunk,
)
from ai_hub.services.chunk_embedding_identity import chunk_embedding_fingerprint
from ai_hub.services.embedding_contract import (
    embedding_contract_fingerprint,
    resolve_embedding_contract,
)


#: IEEE-754 float32, little endian, version 1. Deliberately not part of `e1`.
VECTOR_FORMAT_F32LE1 = KnowledgeChunkEmbedding.VectorFormat.F32LE1
_F32_STRUCT_CHAR = "<f"
_F32_BYTES = 4
#: Largest finite float32. A float64 beyond this would silently become inf.
_F32_MAX = 3.4028235677973366e38


class VectorStoreError(ValueError):
    """A vector or its persistence request is not acceptable."""


class VectorEncodingError(VectorStoreError):
    """The vector cannot be encoded or decoded under the declared format."""


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_vector(values, *, expected_dimension: int) -> bytes:
    """Encode a sequence of numbers as `f32le1` bytes. Refuse anything unclear.

    Never truncates, never pads, never drops an invalid component and never
    normalizes. Normalization is the S-18 `normalization` contract and belongs to
    the future execution pipeline; silently applying it here would mean stored
    vectors no longer matched what the model returned.
    """
    if expected_dimension is None or expected_dimension < 1:
        raise VectorEncodingError("Expected vector dimension must be at least 1.")
    try:
        components = list(values)
    except TypeError as exc:
        raise VectorEncodingError("Vector must be an iterable of numbers.") from exc
    if len(components) != expected_dimension:
        raise VectorEncodingError(
            f"Vector has {len(components)} components; "
            f"the contract requires exactly {expected_dimension}."
        )

    encoded = bytearray()
    for index, raw in enumerate(components):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise VectorEncodingError(
                f"Vector component {index} is not a number."
            )
        value = float(raw)
        if math.isnan(value):
            raise VectorEncodingError(f"Vector component {index} is NaN.")
        if math.isinf(value):
            raise VectorEncodingError(f"Vector component {index} is infinite.")
        if abs(value) > _F32_MAX:
            # struct would happily emit `inf` here. Overflowing to infinity is a
            # silent corruption of the vector, so it is refused instead.
            raise VectorEncodingError(
                f"Vector component {index} overflows float32."
            )
        encoded += struct.pack(_F32_STRUCT_CHAR, value)
    return bytes(encoded)


def decode_vector(data, *, expected_dimension: int, vector_format=VECTOR_FORMAT_F32LE1):
    """Decode `f32le1` bytes. Unknown formats fail closed - no fallback decoder."""
    if vector_format != VECTOR_FORMAT_F32LE1:
        raise VectorEncodingError(f"Unknown vector format {vector_format!r}.")
    if expected_dimension is None or expected_dimension < 1:
        raise VectorEncodingError("Expected vector dimension must be at least 1.")
    payload = bytes(data or b"")
    if len(payload) != expected_dimension * _F32_BYTES:
        raise VectorEncodingError(
            f"Vector payload is {len(payload)} bytes; the contract requires "
            f"{expected_dimension * _F32_BYTES}."
        )
    return tuple(
        struct.unpack_from(_F32_STRUCT_CHAR, payload, offset * _F32_BYTES)[0]
        for offset in range(expected_dimension)
    )


# ---------------------------------------------------------------------------
# Currentness
# ---------------------------------------------------------------------------

class InspectionReason:
    NAMESPACE_MISMATCH = "namespace_mismatch"
    COLLECTION_MISMATCH = "collection_mismatch"
    K1_MISMATCH = "k1_mismatch"
    E1_MISMATCH = "e1_mismatch"
    DIMENSION_MISMATCH = "dimension_mismatch"
    ENCODING_INVALID = "encoding_invalid"
    #: The recorded configuration's SEMANTIC fields cannot form a contract at
    #: all (e.g. a blank revision written through raw ORM). Deliberately NOT
    #: raised for an inactive configuration or provider - see
    #: `inspect_vector_record`.
    CONTRACT_MALFORMED = "contract_malformed"


@dataclass(frozen=True)
class VectorRecordInspection:
    """Why a stored vector is or is not current. Carries no content, no values."""

    current: bool
    namespace_matches: bool
    collection_matches: bool
    k1_matches: bool
    e1_matches: bool
    dimension_matches: bool
    encoding_valid: bool
    reason_codes: tuple


def inspect_vector_record(record) -> VectorRecordInspection:
    """Structural and identity currentness for one stored vector.

    **Mathematical identity only.** This deliberately does NOT ask whether the
    embedding configuration is operationally usable right now:

        vector currentness  !=  operational usability

    `is_active` (on either the configuration or its provider) is excluded from
    `e1` by S-18, so deactivating one must not change what a stored vector IS.
    Nothing about the bytes changed; only permission to do new work with them
    did. Same for `declared_locality`, `ProviderGrant` and the egress flags -
    none is consulted here.

    This is why `e1` is recomputed below from the recorded configuration's
    SEMANTIC fields via the pure S-18 fingerprint, rather than through
    `resolve_embedding_contract()`. That resolver is the OPERATIONAL gate: it
    refuses an inactive config or provider, which is exactly right for a new
    write and exactly wrong for inspecting a vector that already exists.

        WRITE a new vector    -> operational resolver (active required)
        INSPECT an existing   -> pure fingerprint (active irrelevant)
    """
    reasons = []

    chunk = record.chunk
    canonical_collection = chunk.document.collection
    collection_matches = record.collection_id == canonical_collection.pk
    namespace_matches = (
        record.application_scope_id == canonical_collection.application_scope_id
    )
    if not collection_matches:
        reasons.append(InspectionReason.COLLECTION_MISMATCH)
    if not namespace_matches:
        reasons.append(InspectionReason.NAMESPACE_MISMATCH)

    k1_matches = record.k1 == chunk_embedding_fingerprint(chunk)
    if not k1_matches:
        reasons.append(InspectionReason.K1_MISMATCH)

    e1_matches = False
    dimension_matches = False
    config = record.embedding_model_config
    try:
        # The canonical S-18 pure function - never a reimplementation of the
        # fingerprint algorithm, and never the operational resolver.
        expected_e1 = embedding_contract_fingerprint(
            provider_type=config.provider.provider_type,
            model_name=config.model_name,
            model_revision=config.model_revision,
            vector_dimension=config.vector_dimension,
            distance_metric=config.distance_metric,
            normalization=config.normalization,
        )
    except (TypeError, ValueError):
        # Narrow on purpose. These are the only failures malformed SEMANTIC
        # facts can produce here - `int()` on a non-coercible dimension raises
        # TypeError, a non-numeric string raises ValueError. A broad `except`
        # would let CONTRACT_MALFORMED swallow ordinary programming errors and
        # report them as a data problem.
        #
        # An INACTIVE config or provider never reaches this branch: nothing in
        # the pure fingerprint consults active state.
        reasons.append(InspectionReason.CONTRACT_MALFORMED)
    else:
        e1_matches = record.e1 == expected_e1
        dimension_matches = record.vector_dimension == config.vector_dimension
        if not e1_matches:
            reasons.append(InspectionReason.E1_MISMATCH)
        if not dimension_matches:
            reasons.append(InspectionReason.DIMENSION_MISMATCH)

    encoding_valid = True
    try:
        decode_vector(
            record.vector_bytes,
            expected_dimension=record.vector_dimension,
            vector_format=record.vector_format,
        )
    except VectorEncodingError:
        encoding_valid = False
        reasons.append(InspectionReason.ENCODING_INVALID)

    return VectorRecordInspection(
        current=not reasons,
        namespace_matches=namespace_matches,
        collection_matches=collection_matches,
        k1_matches=k1_matches,
        e1_matches=e1_matches,
        dimension_matches=dimension_matches,
        encoding_valid=encoding_valid,
        reason_codes=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

@runtime_checkable
class VectorStore(Protocol):
    """Persistence lifecycle only.

    There is deliberately no `nearest_neighbors`, `similarity_search`, `rank`,
    `score`, `cosine_search` or `ann_search` here. Designing that API before a
    backend is chosen would bake a ranking model into an interface that has to
    survive the backend decision.
    """

    def store_chunk_vector(self, *, application_scope, chunk,
                           embedding_model_config, vector): ...

    def load_current_vectors(self, *, application_scope, e1, collection_ids): ...


@dataclass(frozen=True)
class StoredChunkVector:
    """What a scoped load returns. Values are decoded; no Knowledge text."""

    record_id: int
    chunk_id: int
    document_id: int
    collection_id: int
    application_scope_id: int
    k1: str
    e1: str
    vector_dimension: int
    values: tuple


class DjangoBinaryVectorStore:
    """The portable correctness/reference backend. NOT the production ANN backend.

    Stores `f32le1` bytes in an ordinary Django `BinaryField`, so it runs
    unchanged on SQLite and PostgreSQL and adds no dependency. It is intended for
    correctness, contract testing, local development, the future local-only
    pipeline and exact persistence semantics - never for production-scale
    nearest-neighbour search.

    Selected explicitly by callers. There is no backend discovery, no "first
    available store", no pgvector detection and no environment sniffing.
    """

    def store_chunk_vector(
        self, *, application_scope, chunk, embedding_model_config, vector
    ):
        """Persist one vector for `(chunk, e1)`. Every fact is server-derived.

        The caller supplies only the namespace it BELIEVES it is writing to, the
        chunk, the configuration and the numbers. `collection`, `k1`, `e1`,
        `vector_dimension` and `vector_format` are all derived here - a
        caller-supplied fingerprint must never become authoritative, because a
        wrong one would make a stale vector look current forever.
        """
        if chunk is None or getattr(chunk, "pk", None) is None:
            raise VectorStoreError("A persisted Knowledge chunk is required.")
        if application_scope is None or getattr(application_scope, "pk", None) is None:
            raise VectorStoreError("An application scope is required.")

        chunk = (
            KnowledgeDocumentChunk.objects.select_related(
                "document", "document__collection",
                "document__collection__application_scope",
            ).get(pk=chunk.pk)
        )
        collection = chunk.document.collection

        # The caller states its intended namespace and we CHECK it, rather than
        # silently correcting: a mismatch means the caller believed it was
        # writing somewhere else, and quietly relocating the write would hide a
        # cross-application bug instead of surfacing it.
        if collection.application_scope_id != application_scope.pk:
            raise VectorStoreError(
                "The requested application scope does not own this chunk."
            )

        # Reuse the canonical S-18 resolver; never reimplement e1 validation.
        contract = resolve_embedding_contract(embedding_model_config)

        k1 = chunk_embedding_fingerprint(chunk)
        # Encode and validate BEFORE touching the existing row, so a rejected
        # vector leaves the previous one untouched.
        vector_bytes = encode_vector(
            vector, expected_dimension=contract.vector_dimension
        )

        with transaction.atomic():
            record, _created = KnowledgeChunkEmbedding.objects.update_or_create(
                chunk=chunk,
                e1=contract.e1,
                defaults={
                    "application_scope_id": collection.application_scope_id,
                    "collection": collection,
                    # Record the configuration that actually performed THIS
                    # write. Two configs may share one `e1`; the slot is keyed on
                    # the contract, and this field says who last wrote it.
                    "embedding_model_config_id": contract.embedding_model_config_id,
                    "k1": k1,
                    "vector_dimension": contract.vector_dimension,
                    "vector_format": VECTOR_FORMAT_F32LE1,
                    "vector_bytes": vector_bytes,
                },
            )
        return record

    def load_current_vectors(self, *, application_scope, e1, collection_ids):
        """CURRENT vectors inside one namespace, narrowed to authorized collections.

        `collection_ids` must already be authorized by the caller - typically
        `EffectiveKnowledgeScope.collection_ids`. This method performs no Agent
        authorization of its own.

        **An empty `collection_ids` means ZERO rows, never "no filter".** Same
        fail-closed shape as S-15: an empty authorization set is a meaningful
        answer, not an absent one.

        Denormalized routing columns narrow in SQL; canonical chunk lineage is
        then re-derived and compared, so a corrupt row whose columns claim one
        namespace while its chunk belongs to another is never returned.
        """
        if application_scope is None or getattr(application_scope, "pk", None) is None:
            return []
        ids = frozenset(collection_ids or ())
        if not ids:
            return []

        rows = (
            KnowledgeChunkEmbedding.objects.filter(
                application_scope_id=application_scope.pk,
                e1=e1,
                collection_id__in=ids,
                # Belt and braces: the canonical lineage must agree in SQL too.
                chunk__document__collection_id__in=ids,
                chunk__document__collection__application_scope_id=application_scope.pk,
            )
            .select_related(
                "chunk", "chunk__document", "chunk__document__collection",
                "embedding_model_config", "embedding_model_config__provider",
            )
        )

        loaded = []
        for record in rows:
            inspection = inspect_vector_record(record)
            if not inspection.current:
                # A stale k1 or e1 must never become a future semantic candidate.
                continue
            loaded.append(
                StoredChunkVector(
                    record_id=record.pk,
                    chunk_id=record.chunk_id,
                    document_id=record.chunk.document_id,
                    collection_id=record.collection_id,
                    application_scope_id=record.application_scope_id,
                    k1=record.k1,
                    e1=record.e1,
                    vector_dimension=record.vector_dimension,
                    values=decode_vector(
                        record.vector_bytes,
                        expected_dimension=record.vector_dimension,
                        vector_format=record.vector_format,
                    ),
                )
            )
        return loaded


#: Module-level convenience bound to the reference backend. Explicit, not
#: discovered: when a second backend exists, callers choose between them.
_reference_store = DjangoBinaryVectorStore()


def store_chunk_vector(*, application_scope, chunk, embedding_model_config, vector):
    return _reference_store.store_chunk_vector(
        application_scope=application_scope,
        chunk=chunk,
        embedding_model_config=embedding_model_config,
        vector=vector,
    )


def load_current_vectors(*, application_scope, e1, collection_ids):
    return _reference_store.load_current_vectors(
        application_scope=application_scope, e1=e1, collection_ids=collection_ids
    )
