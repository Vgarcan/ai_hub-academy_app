"""PostgreSQL/pgvector HNSW backend. SHADOW ONLY - nothing routes here yet.

    ApplicationScope partition
        -> Collection partition
            -> e1 partition (a leaf)
                -> ONE HNSW index

**Why the partitioning is a security mechanism and not a performance choice.**
A single global HNSW index answered with

    ... WHERE application_scope_id = ? AND collection_id IN (...)
        ORDER BY embedding <=> ? LIMIT 20

is NOT authorization-first. An approximate index selects a finite candidate pool
FIRST and applies relational filters to what it found. Vectors the caller may
never see therefore consume ANN candidate slots, and the answer silently gets
worse the more foreign Knowledge exists next to it. That inverts the S-15
invariant that unauthorized candidates consume zero retrieval capacity, and no
amount of `hnsw.iterative_scan` turns it back into a boundary - iterative
scanning is a recall feature, not a security one.

Partitioning makes the filter physical instead. Every HNSW graph belongs to
exactly one `(ApplicationScope, KnowledgeCollection, e1)`, so an unauthorized
collection is not filtered out of a scan - **its graph is never opened**.

`e1` is part of the physical key for the same reason it exists at all: one `e1`
fixes provider type, model, revision, dimension, metric and normalization.
Mixing two of them in one graph would put incomparable vectors in one metric
space, and would leave the leaf with no single dimension or operator class.

**This is a derived mirror, never authority.** `KnowledgeChunkEmbedding` remains
the canonical, portable, source-of-truth vector store. Everything here is
rebuildable from it, is never written automatically, and never re-embeds
anything - no provider is contacted by any function in this module.

**Nothing routes here yet.** S-21's exact Python retriever remains the
correctness oracle and S-22/S-23 remain reference-backed. There is deliberately
no default, no auto-detection, no "if postgres then ANN" and no fallback:
promotion needs the golden-query evaluation that has not happened.
"""

import hashlib
import re
from dataclasses import dataclass

from django.db import connection, transaction

from ai_hub.models import KnowledgeChunkEmbedding, KnowledgeDocument
from ai_hub.services.embedding_contract import resolve_embedding_contract
from ai_hub.services.embedding_vector import (
    EmbeddingVectorError,
    validate_embedding_vector,
)
from ai_hub.services.semantic_retrieval import (
    RetrievalFailureCategory,
    SemanticRetrievalError,
    cosine_similarity,
    resolve_metric_scorer,
)
from ai_hub.services.vector_store import decode_vector, inspect_vector_record

try:  # pragma: no cover - psycopg is present via the PostgreSQL backend
    from psycopg import sql
except ImportError:  # pragma: no cover - SQLite-only environments
    sql = None


#: The backend contract, versioned so it can be replaced rather than tuned.
#: `pgv-hnsw1` names one combination of index type, build parameters, search
#: parameters and candidate pool. None of these numbers is claimed to be
#: quality-optimal - they are the pgvector reference defaults, and S-25 is where
#: quality gets measured. If evaluation later justifies different values that
#: must become a NEW backend version: silently retuning `pgv-hnsw1` would change
#: every previously indexed leaf's meaning while the name still claimed
#: otherwise.
PGVECTOR_BACKEND_VERSION = "pgv-hnsw1"

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 100

#: How many rows HNSW may return before canonical revalidation and exact
#: reranking. Deliberately much larger than any final `limit`: ANN is the
#: candidate generator, not the ranker.
ANN_CANDIDATE_POOL = 100
MAX_ANN_RESULTS = 20

#: pgvector's `vector` type supports HNSW up to 2000 dimensions. S-24 uses
#: `vector` and nothing else - no `halfvec`, no binary quantization, no
#: subvector indexing - so a contract above this ceiling is REFUSED. Never
#: truncated, never dimension-reduced, never quietly downgraded to a lossy type:
#: each of those would index something other than the vectors the contract
#: describes.
MAX_HNSW_VECTOR_DIMENSION = 2000

#: The minimum pgvector version this backend is validated against.
MIN_PGVECTOR_VERSION = (0, 8, 6)

ANN_PARENT_TABLE = "ai_hub_pgvector_ann_embedding"
ANN_GENERATION_TABLE = "ai_hub_pgvector_ann_generation"
ANN_LEAF_STATE_TABLE = "ai_hub_pgvector_ann_leaf_state"

#: A fixed namespace for `pg_advisory_xact_lock`, so AI Hub's leaf locks cannot
#: collide with another application's advisory locks in the same database.
ADVISORY_LOCK_NAMESPACE = 0x41_48_50_47  # "AHPG"

#: PostgreSQL truncates identifiers at 63 bytes. Everything generated here is
#: well inside that, and `_checked_identifier` refuses anything that is not.
MAX_IDENTIFIER_LENGTH = 63

E1_PATTERN = re.compile(r"^e1:sha256:[0-9a-f]{64}$")
E1_DIGEST_CHARS = 12


class PgvectorFailureCategory:
    """Bounded machine-readable refusals. None carries content or a query."""

    UNSUPPORTED_DATABASE_VENDOR = "pgvector_unsupported_database_vendor"
    EXTENSION_MISSING = "pgvector_extension_missing"
    EXTENSION_TOO_OLD = "pgvector_extension_too_old"
    INVALID_SCOPE = "pgvector_invalid_scope"
    INVALID_COLLECTION = "pgvector_invalid_collection"
    COLLECTION_FOREIGN_SCOPE = "pgvector_collection_foreign_scope"
    INVALID_E1 = "pgvector_invalid_e1"
    DIMENSION_UNSUPPORTED = "pgvector_ann_dimension_unsupported"
    UNSUPPORTED_METRIC = "pgvector_unsupported_metric"
    IDENTIFIER_TOO_LONG = "pgvector_identifier_too_long"
    INVALID_LIMIT = "pgvector_invalid_limit"
    QUERY_VECTOR_INVALID = "pgvector_query_vector_invalid"
    #: Deliberately the SAME string S-21 uses. A zero-magnitude vector is
    #: unscorable under cosine in the reference oracle, and this backend must
    #: report the identical condition by the identical name rather than
    #: inventing a second vocabulary for one semantic fact.
    UNSCORABLE_ZERO_VECTOR = "unscorable_zero_vector"
    LEAF_NOT_READY = "pgvector_leaf_not_ready"
    ANN_INTEGRITY_MISMATCH = "pgvector_ann_integrity_mismatch"
    SOURCE_CHANGED_DURING_SEARCH = "pgvector_ann_source_changed_during_search"


class PgvectorAnnError(RuntimeError):
    """A refusal, carrying a bounded category and never Knowledge or a query."""

    def __init__(self, category: str, message: str = ""):
        self.category = category
        super().__init__(message or category)


# ---------------------------------------------------------------------------
# Metric mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricBackend:
    """How ONE S-18 metric is expressed in pgvector.

    `operator` is the pgvector ORDERING operator, which is a backend convention
    and not a Core value. `<#>` in particular returns the NEGATIVE inner product
    so that "smaller is nearer" holds for every operator. That number must never
    escape this module as a semantic score - the final ranking is recomputed by
    S-21's exact scorers, which is why this dataclass carries no direction flag.
    """

    opclass: str
    operator: str


METRIC_BACKENDS = {
    "cosine": MetricBackend(opclass="vector_cosine_ops", operator="<=>"),
    "dot_product": MetricBackend(opclass="vector_ip_ops", operator="<#>"),
    "euclidean": MetricBackend(opclass="vector_l2_ops", operator="<->"),
}


def is_cosine_unscorable(values) -> bool:
    """Would S-21's exact cosine scorer refuse this vector?

    Asks the reference oracle instead of re-deriving "zero magnitude". A second
    definition here - `all(v == 0)`, or a norm compared against some epsilon -
    could disagree with S-21 at the edges, and the disagreement would surface as
    the ANN backend quietly answering a query the exact oracle refuses.

    Self-similarity is the probe: `cosine(v, v)` is 1.0 for any vector with a
    direction and raises `UNSCORABLE_ZERO_VECTOR` for one without.
    """
    try:
        cosine_similarity(values, values)
    except SemanticRetrievalError as exc:
        if exc.category == RetrievalFailureCategory.UNSCORABLE_ZERO_VECTOR:
            return True
        raise
    return False


def resolve_metric_backend(metric: str) -> MetricBackend:
    """The pgvector operator class for a declared metric, or refuse.

    Never a default. An unknown metric must not fall back to cosine: building a
    cosine graph for a configuration that says Euclidean produces an index that
    is confidently wrong.
    """
    backend = METRIC_BACKENDS.get(metric)
    if backend is None:
        raise PgvectorAnnError(
            PgvectorFailureCategory.UNSUPPORTED_METRIC,
            f"No pgvector operator class is mapped for metric {metric!r}.",
        )
    return backend


# ---------------------------------------------------------------------------
# Identifier construction
# ---------------------------------------------------------------------------

def validate_e1(e1) -> str:
    """Accept ONLY the canonical `e1:sha256:<64 lowercase hex>` form.

    The full value becomes a partition BOUND (a literal, safely quoted); only a
    bounded hex slice of it ever becomes part of an identifier. Refusing
    anything else here is what makes the identifier construction below provably
    injection-free rather than merely careful.
    """
    text = e1 if isinstance(e1, str) else ""
    if not E1_PATTERN.match(text):
        raise PgvectorAnnError(
            PgvectorFailureCategory.INVALID_E1,
            "A canonical e1:sha256:<64 hex> contract fingerprint is required.",
        )
    return text


def e1_digest(e1: str) -> str:
    """A bounded hex slice of a validated `e1`, for use inside identifiers."""
    return validate_e1(e1)[len("e1:sha256:"):][:E1_DIGEST_CHARS]


def _checked_id(value, *, label: str) -> int:
    """A positive integer, or refuse. Identifiers are built from these ONLY."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PgvectorAnnError(
            PgvectorFailureCategory.INVALID_SCOPE
            if label == "application scope"
            else PgvectorFailureCategory.INVALID_COLLECTION,
            f"A persisted {label} is required.",
        )
    return value


def _checked_identifier(name: str) -> str:
    if len(name) > MAX_IDENTIFIER_LENGTH:
        raise PgvectorAnnError(
            PgvectorFailureCategory.IDENTIFIER_TOO_LONG,
            f"Generated identifier exceeds {MAX_IDENTIFIER_LENGTH} bytes.",
        )
    return name


@dataclass(frozen=True)
class LeafIdentity:
    """The physical names of one `(scope, collection, e1)` leaf.

    Built from NUMERIC ids and a hex digest and from nothing else. A scope name,
    a collection name, a model name, a provider name or a document title must
    never reach DDL - they are operator-supplied free text, and an identifier is
    the one place SQL cannot be parameterized.
    """

    application_scope_id: int
    collection_id: int
    e1: str
    scope_partition: str
    collection_partition: str
    leaf_table: str
    leaf_index: str


def leaf_identity(application_scope_id, collection_id, e1) -> LeafIdentity:
    scope_id = _checked_id(application_scope_id, label="application scope")
    coll_id = _checked_id(collection_id, label="collection")
    digest = e1_digest(e1)
    scope_partition = _checked_identifier(f"ah_pgv_s{scope_id}")
    collection_partition = _checked_identifier(f"ah_pgv_s{scope_id}_c{coll_id}")
    leaf_table = _checked_identifier(
        f"ah_pgv_s{scope_id}_c{coll_id}_e{digest}"
    )
    return LeafIdentity(
        application_scope_id=scope_id,
        collection_id=coll_id,
        e1=validate_e1(e1),
        scope_partition=scope_partition,
        collection_partition=collection_partition,
        leaf_table=leaf_table,
        leaf_index=_checked_identifier(f"{leaf_table}_hnsw"),
    )


def _advisory_key(identity: LeafIdentity) -> int:
    """A deterministic signed 32-bit key for ONE leaf.

    Deterministic so two processes provisioning the same leaf contend on the
    same lock. `IF NOT EXISTS` alone is not enough: two concurrent CREATEs of
    the same partition can still deadlock or interleave with the index build.
    """
    raw = f"{identity.application_scope_id}:{identity.collection_id}:{identity.e1}"
    digest = hashlib.sha256(raw.encode("utf-8")).digest()[:4]
    return int.from_bytes(digest, "big", signed=True)


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------

def _require_postgresql():
    if connection.vendor != "postgresql":
        raise PgvectorAnnError(
            PgvectorFailureCategory.UNSUPPORTED_DATABASE_VENDOR,
            "The pgvector ANN backend requires PostgreSQL.",
        )


def pgvector_extension_version():
    """The installed pgvector version as a tuple, or `None` if absent."""
    _require_postgresql()
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        row = cursor.fetchone()
    if not row or not row[0]:
        return None
    parts = []
    for chunk in str(row[0]).split("."):
        digits = "".join(character for character in chunk if character.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def require_pgvector_backend():
    """PostgreSQL plus a pgvector new enough to have been validated, or refuse."""
    _require_postgresql()
    version = pgvector_extension_version()
    if version is None:
        raise PgvectorAnnError(
            PgvectorFailureCategory.EXTENSION_MISSING,
            "The PostgreSQL vector extension is not installed.",
        )
    if version < MIN_PGVECTOR_VERSION:
        raise PgvectorAnnError(
            PgvectorFailureCategory.EXTENSION_TOO_OLD,
            "The installed pgvector extension predates the validated version.",
        )
    return version


# ---------------------------------------------------------------------------
# DDL construction (pure: builds SQL, executes nothing)
# ---------------------------------------------------------------------------

def _validated_dimension(dimension) -> int:
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 1
        or dimension > MAX_HNSW_VECTOR_DIMENSION
    ):
        raise PgvectorAnnError(
            PgvectorFailureCategory.DIMENSION_UNSUPPORTED,
            f"HNSW over pgvector `vector` supports 1..{MAX_HNSW_VECTOR_DIMENSION} "
            f"dimensions; this contract declares {dimension!r}.",
        )
    return dimension


def scope_partition_sql(identity: LeafIdentity):
    return sql.SQL(
        "CREATE TABLE IF NOT EXISTS {partition} "
        "PARTITION OF {parent} FOR VALUES IN ({scope_id}) "
        "PARTITION BY LIST (collection_id)"
    ).format(
        partition=sql.Identifier(identity.scope_partition),
        parent=sql.Identifier(ANN_PARENT_TABLE),
        scope_id=sql.Literal(identity.application_scope_id),
    )


def collection_partition_sql(identity: LeafIdentity):
    return sql.SQL(
        "CREATE TABLE IF NOT EXISTS {partition} "
        "PARTITION OF {scope_partition} FOR VALUES IN ({collection_id}) "
        "PARTITION BY LIST (e1)"
    ).format(
        partition=sql.Identifier(identity.collection_partition),
        scope_partition=sql.Identifier(identity.scope_partition),
        collection_id=sql.Literal(identity.collection_id),
    )


def leaf_partition_sql(identity: LeafIdentity, *, dimension: int):
    """The e1 leaf, with a DATABASE-enforced dimension check.

    The check exists because insertion code is not the only way rows arrive: raw
    SQL, a future bulk loader or a mistaken migration could all put a
    wrong-length vector in a leaf whose HNSW index assumes one length.
    """
    dimension = _validated_dimension(dimension)
    return sql.SQL(
        "CREATE TABLE IF NOT EXISTS {leaf} "
        "PARTITION OF {collection_partition} FOR VALUES IN ({e1})"
    ).format(
        leaf=sql.Identifier(identity.leaf_table),
        collection_partition=sql.Identifier(identity.collection_partition),
        e1=sql.Literal(identity.e1),
    )


def leaf_constraint_sql(identity: LeafIdentity, *, dimension: int):
    """Dimension check plus the uniqueness S-19 already guarantees upstream.

    S-19 permits exactly one canonical vector per `(chunk, e1)`, and a leaf IS
    one `e1`, so both `source_embedding_id` and `chunk_id` are unique inside it.
    Enforcing that here turns a mirror bug into a constraint violation instead of
    a duplicated ANN candidate.
    """
    dimension = _validated_dimension(dimension)
    return [
        (
            f"{identity.leaf_table}_dims",
            sql.SQL(
                "ALTER TABLE {leaf} ADD CONSTRAINT {name} "
                "CHECK (vector_dims(embedding) = {dimension})"
            ).format(
                leaf=sql.Identifier(identity.leaf_table),
                name=sql.Identifier(f"{identity.leaf_table}_dims"),
                dimension=sql.Literal(dimension),
            ),
        ),
        (
            f"{identity.leaf_table}_src",
            sql.SQL(
                "ALTER TABLE {leaf} ADD CONSTRAINT {name} "
                "UNIQUE (source_embedding_id)"
            ).format(
                leaf=sql.Identifier(identity.leaf_table),
                name=sql.Identifier(f"{identity.leaf_table}_src"),
            ),
        ),
        (
            f"{identity.leaf_table}_chunk",
            sql.SQL(
                "ALTER TABLE {leaf} ADD CONSTRAINT {name} UNIQUE (chunk_id)"
            ).format(
                leaf=sql.Identifier(identity.leaf_table),
                name=sql.Identifier(f"{identity.leaf_table}_chunk"),
            ),
        ),
    ]


def leaf_index_sql(identity: LeafIdentity, *, dimension: int, metric: str):
    """ONE HNSW index, on ONE leaf, for ONE metric.

    The expression cast pins the dimension the graph was built for, and the
    operator class pins the metric. There is deliberately no index on the parent
    or on any intermediate partition: an index that spanned a scope or a
    collection would be exactly the global graph this design exists to prevent.
    """
    dimension = _validated_dimension(dimension)
    backend = resolve_metric_backend(metric)
    return sql.SQL(
        "CREATE INDEX IF NOT EXISTS {index} ON {leaf} "
        "USING hnsw ((embedding::vector({dimension})) {opclass}) "
        "WITH (m = {m}, ef_construction = {ef})"
    ).format(
        index=sql.Identifier(identity.leaf_index),
        leaf=sql.Identifier(identity.leaf_table),
        dimension=sql.SQL(str(dimension)),
        opclass=sql.SQL(backend.opclass),
        m=sql.SQL(str(HNSW_M)),
        ef=sql.SQL(str(HNSW_EF_CONSTRUCTION)),
    )


def render_sql(composed) -> str:
    """Render composed SQL to text. Used by tests; never to build a query."""
    return composed.as_string(None)


# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeafProvisioning:
    identity: LeafIdentity
    vector_dimension: int
    distance_metric: str
    backend_version: str


def _resolved_leaf(application_scope, collection, embedding_model_config):
    """Validate the triple and derive everything else. No caller-supplied facts."""
    scope_id = getattr(application_scope, "pk", None)
    if scope_id is None:
        raise PgvectorAnnError(
            PgvectorFailureCategory.INVALID_SCOPE,
            "A persisted application scope is required.",
        )
    collection_id = getattr(collection, "pk", None)
    if collection_id is None:
        raise PgvectorAnnError(
            PgvectorFailureCategory.INVALID_COLLECTION,
            "A persisted knowledge collection is required.",
        )
    if getattr(collection, "application_scope_id", None) != scope_id:
        # Refuse rather than relocate. A collection provisioned under the wrong
        # scope would place its HNSW graph inside another application's
        # partition subtree - the one failure this whole design exists to make
        # impossible.
        raise PgvectorAnnError(
            PgvectorFailureCategory.COLLECTION_FOREIGN_SCOPE,
            "The collection does not belong to the requested application scope.",
        )

    contract = resolve_embedding_contract(embedding_model_config)
    dimension = _validated_dimension(contract.vector_dimension)
    resolve_metric_backend(contract.distance_metric)
    identity = leaf_identity(scope_id, collection_id, contract.e1)
    return LeafProvisioning(
        identity=identity,
        vector_dimension=dimension,
        distance_metric=contract.distance_metric,
        backend_version=PGVECTOR_BACKEND_VERSION,
    ), contract


def provision_pgvector_ann_leaf(
    *, application_scope, collection, embedding_model_config
) -> LeafProvisioning:
    """Create the partition path and HNSW index for ONE leaf. Idempotent.

    Administrative index maintenance, NOT Agent retrieval authorization - which
    is why it takes an `ApplicationScope` and a `KnowledgeCollection` directly
    rather than an `EffectiveKnowledgeScope`. It creates physical structure; it
    decides nothing about who may read anything.

    Serialized on a deterministic advisory lock. `IF NOT EXISTS` alone is not
    enough for concurrent callers: two simultaneous CREATE PARTITION statements
    for the same bounds can still collide, and an index build racing a partition
    create can leave a leaf without its graph.
    """
    require_pgvector_backend()
    provisioning, _contract = _resolved_leaf(
        application_scope, collection, embedding_model_config
    )
    identity = provisioning.identity

    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s, %s)",
                [ADVISORY_LOCK_NAMESPACE, _advisory_key(identity)],
            )
            cursor.execute(scope_partition_sql(identity))
            cursor.execute(collection_partition_sql(identity))
            cursor.execute(
                leaf_partition_sql(
                    identity, dimension=provisioning.vector_dimension
                )
            )
            for name, statement in leaf_constraint_sql(
                identity, dimension=provisioning.vector_dimension
            ):
                # `ADD CONSTRAINT` has no `IF NOT EXISTS`. A catalog check under
                # the advisory lock is simpler and safer than re-embedding the
                # statement inside a dollar-quoted `DO` block.
                cursor.execute(
                    "SELECT 1 FROM pg_constraint WHERE conname = %s", [name]
                )
                if cursor.fetchone() is None:
                    cursor.execute(statement)
            cursor.execute(
                leaf_index_sql(
                    identity,
                    dimension=provisioning.vector_dimension,
                    metric=provisioning.distance_metric,
                )
            )
    return provisioning


# ---------------------------------------------------------------------------
# Generation: freshness that raw SQL cannot bypass
# ---------------------------------------------------------------------------

def current_generation(application_scope_id, collection_id) -> int:
    """The source generation for one namespace. Absent means 0.

    Bumped exclusively by DATABASE TRIGGERS, never by Django signals. A signal
    is bypassed by `queryset.update()`, by `bulk_create`, by raw SQL and by any
    future management command that touches the corpus directly - and a missed
    invalidation means a stale HNSW graph keeps consuming candidate slots while
    reporting itself ready.
    """
    _require_postgresql()
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                "SELECT generation FROM {table} "
                "WHERE application_scope_id = %s AND collection_id = %s"
            ).format(table=sql.Identifier(ANN_GENERATION_TABLE)),
            [application_scope_id, collection_id],
        )
        row = cursor.fetchone()
    return int(row[0]) if row else 0


def current_generations(application_scope_id, collection_ids) -> dict:
    return {
        collection_id: current_generation(application_scope_id, collection_id)
        for collection_id in collection_ids
    }


# ---------------------------------------------------------------------------
# Leaf state and readiness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeafState:
    application_scope_id: int
    collection_id: int
    e1: str
    vector_dimension: int
    distance_metric: str
    backend_version: str
    indexed_generation: int
    source_count: int


def read_leaf_state(application_scope_id, collection_id, e1):
    _require_postgresql()
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                "SELECT vector_dimension, distance_metric, backend_version, "
                "indexed_generation, source_count FROM {table} "
                "WHERE application_scope_id = %s AND collection_id = %s "
                "AND e1 = %s"
            ).format(table=sql.Identifier(ANN_LEAF_STATE_TABLE)),
            [application_scope_id, collection_id, e1],
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return LeafState(
        application_scope_id=application_scope_id,
        collection_id=collection_id,
        e1=e1,
        vector_dimension=int(row[0]),
        distance_metric=str(row[1]),
        backend_version=str(row[2]),
        indexed_generation=int(row[3]),
        source_count=int(row[4]),
    )


@dataclass(frozen=True)
class LeafReadiness:
    """Whether one leaf may be searched, and if not, why. Bounded reasons only."""

    collection_id: int
    ready: bool
    reason: str
    indexed_generation: int
    current_generation: int


def leaf_readiness(application_scope_id, collection_id, contract) -> LeafReadiness:
    """A leaf is searchable only if it is provably in step with its source."""
    generation = current_generation(application_scope_id, collection_id)
    state = read_leaf_state(application_scope_id, collection_id, contract.e1)
    if state is None:
        return LeafReadiness(
            collection_id=collection_id, ready=False,
            reason="leaf_absent", indexed_generation=0,
            current_generation=generation,
        )
    if state.backend_version != PGVECTOR_BACKEND_VERSION:
        return LeafReadiness(
            collection_id=collection_id, ready=False,
            reason="backend_version_mismatch",
            indexed_generation=state.indexed_generation,
            current_generation=generation,
        )
    if state.vector_dimension != contract.vector_dimension:
        return LeafReadiness(
            collection_id=collection_id, ready=False,
            reason="dimension_mismatch",
            indexed_generation=state.indexed_generation,
            current_generation=generation,
        )
    if state.distance_metric != contract.distance_metric:
        return LeafReadiness(
            collection_id=collection_id, ready=False,
            reason="metric_mismatch",
            indexed_generation=state.indexed_generation,
            current_generation=generation,
        )
    if state.indexed_generation != generation:
        # The corpus moved after this graph was built. A stale graph is not
        # merely out of date: its rows still occupy finite ANN candidate slots,
        # so searching it degrades the answer invisibly.
        return LeafReadiness(
            collection_id=collection_id, ready=False,
            reason="stale_generation",
            indexed_generation=state.indexed_generation,
            current_generation=generation,
        )
    return LeafReadiness(
        collection_id=collection_id, ready=True, reason="ready",
        indexed_generation=state.indexed_generation,
        current_generation=generation,
    )


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeafRebuildResult:
    identity: LeafIdentity
    source_count: int
    indexed_generation: int
    current_generation: int
    ready: bool


def _mirrorable_rows(application_scope_id, collection_id, contract):
    """Canonical vectors that are BOTH current AND currently retrievable.

    Two independent conditions, and both matter. S-19 currentness says the
    stored vector still represents its chunk under this contract. Retrievability
    says S-15/S-21 would let it be returned at all - an archived document or a
    deactivated collection must not sit in the graph consuming candidate slots
    for results that would be filtered out anyway.

    Reuses `inspect_vector_record` and `decode_vector` rather than
    reimplementing `k1`/`e1`. A second fingerprint implementation would drift,
    and the mirror would then disagree with the store it mirrors.
    """
    records = (
        KnowledgeChunkEmbedding.objects.filter(
            application_scope_id=application_scope_id,
            collection_id=collection_id,
            e1=contract.e1,
            chunk__document__collection_id=collection_id,
            chunk__document__collection__application_scope_id=application_scope_id,
            chunk__document__collection__is_active=True,
            chunk__document__status=KnowledgeDocument.Status.ACTIVE,
        )
        .select_related(
            "chunk", "chunk__document", "chunk__document__collection",
            "embedding_model_config", "embedding_model_config__provider",
        )
        .order_by("chunk_id")
    )
    rows = []
    for record in records:
        if not inspect_vector_record(record).current:
            continue
        try:
            values = decode_vector(
                record.vector_bytes,
                expected_dimension=record.vector_dimension,
                vector_format=record.vector_format,
            )
        except (TypeError, ValueError):
            # A corrupt encoding is not mirrored. It is also not repaired here:
            # S-19 owns the canonical store, and inventing a replacement vector
            # would be the index deciding what Knowledge means.
            continue
        if len(values) != contract.vector_dimension:
            continue
        rows.append((record, values))
    return rows


def _vector_literal(values) -> str:
    return "[" + ",".join(repr(float(value)) for value in values) + "]"


def rebuild_pgvector_ann_leaf(
    *, application_scope, collection, embedding_model_config
) -> LeafRebuildResult:
    """Replace ONE leaf's contents from canonical S-19 state. Explicit only.

    There is no signal, no background task and no automatic trigger-driven
    rebuild in S-24. Rebuilding is index maintenance an operator asks for; doing
    it implicitly would make ordinary corpus edits silently expensive and would
    hide when the mirror actually changed.

    **No provider is contacted.** The vectors already exist; this is a
    representation change from `f32le1` bytes into a pgvector column and nothing
    more. Re-embedding here would spend egress to reproduce data we already hold
    and could produce vectors that disagree with the canonical store.

    All-or-nothing: the delete, the insert and the leaf-state update share one
    transaction, so a failure can never leave half the old graph, half the new
    one and a state row claiming the leaf is ready.
    """
    require_pgvector_backend()
    provisioning, contract = _resolved_leaf(
        application_scope, collection, embedding_model_config
    )
    identity = provisioning.identity
    scope_id = identity.application_scope_id
    collection_id = identity.collection_id

    # Read BEFORE loading sources. If the corpus moves while we work, the
    # generation we stamp will not match the one we end with, and the leaf
    # reports itself NOT READY rather than quietly claiming synchronization.
    generation_before = current_generation(scope_id, collection_id)
    rows = _mirrorable_rows(scope_id, collection_id, contract)

    # -- cosine parity with the exact oracle -------------------------------
    # pgvector does not index zero vectors for cosine distance, and S-21's exact
    # scorer REFUSES them. Left alone those two facts combine into the worst
    # possible outcome: the reference oracle refuses the query while the ANN
    # backend silently omits the offending chunk and returns a confident,
    # shorter ranking that nothing marks as incomplete.
    #
    # So the leaf is not allowed to exist in that state. Refusing the rebuild
    # leaves `leaf_state` untouched, so the leaf stays NOT READY and every
    # search over it refuses - which is the same answer S-21 gives.
    #
    # Deliberately not "fixed" here: normalizing the vector, substituting an
    # epsilon or scoring it as 0.0 would each make the mirror disagree with the
    # canonical store it mirrors. The canonical vector and S-21 are both left
    # exactly as they are.
    if contract.distance_metric == "cosine":
        for record, vector_values in rows:
            if is_cosine_unscorable(vector_values):
                raise PgvectorAnnError(
                    PgvectorFailureCategory.UNSCORABLE_ZERO_VECTOR,
                    "A retrievable cosine source vector has zero magnitude; "
                    "pgvector would omit it from the HNSW graph while the exact "
                    "oracle refuses it.",
                )

    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s, %s)",
                [ADVISORY_LOCK_NAMESPACE, _advisory_key(identity)],
            )
            cursor.execute(
                sql.SQL("DELETE FROM {leaf}").format(
                    leaf=sql.Identifier(identity.leaf_table)
                )
            )
            for record, values in rows:
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {leaf} (source_embedding_id, "
                        "application_scope_id, collection_id, chunk_id, k1, e1, "
                        "embedding) VALUES (%s, %s, %s, %s, %s, %s, %s::vector)"
                    ).format(leaf=sql.Identifier(identity.leaf_table)),
                    [
                        record.pk, scope_id, collection_id, record.chunk_id,
                        record.k1, record.e1, _vector_literal(values),
                    ],
                )
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {table} (application_scope_id, collection_id, "
                    "e1, vector_dimension, distance_metric, backend_version, "
                    "indexed_generation, source_count, rebuilt_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW()) "
                    "ON CONFLICT (application_scope_id, collection_id, e1) "
                    "DO UPDATE SET vector_dimension = EXCLUDED.vector_dimension, "
                    "distance_metric = EXCLUDED.distance_metric, "
                    "backend_version = EXCLUDED.backend_version, "
                    "indexed_generation = EXCLUDED.indexed_generation, "
                    "source_count = EXCLUDED.source_count, "
                    "rebuilt_at = EXCLUDED.rebuilt_at"
                ).format(table=sql.Identifier(ANN_LEAF_STATE_TABLE)),
                [
                    scope_id, collection_id, contract.e1,
                    contract.vector_dimension, contract.distance_metric,
                    PGVECTOR_BACKEND_VERSION, generation_before, len(rows),
                ],
            )

    generation_after = current_generation(scope_id, collection_id)
    return LeafRebuildResult(
        identity=identity,
        source_count=len(rows),
        indexed_generation=generation_before,
        current_generation=generation_after,
        # Never an automatic retry. A racing rebuild is reported, and the
        # operator decides whether to run it again.
        ready=generation_after == generation_before,
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PgvectorAnnMatch:
    """One reranked match. Identifiers and an EXACT canonical score."""

    rank: int
    chunk_id: int
    document_id: int
    collection_id: int
    application_scope_id: int
    k1: str
    e1: str
    metric: str
    metric_value: float


@dataclass(frozen=True)
class PgvectorAnnResult:
    """One ANN operation. Ephemeral, content-free, never persisted."""

    matches: tuple
    application_scope_id: int | None
    collection_ids: tuple
    embedding_model_config_id: int | None
    e1: str
    backend_version: str
    metric: str
    ann_candidate_pool: int
    ann_candidates_returned: int


def _empty_result(scope, *, collection_ids=(), config_id=None, e1="", metric=""):
    return PgvectorAnnResult(
        matches=(),
        application_scope_id=scope.application_scope_id,
        collection_ids=tuple(collection_ids),
        embedding_model_config_id=config_id,
        e1=e1,
        backend_version=PGVECTOR_BACKEND_VERSION,
        metric=metric,
        ann_candidate_pool=ANN_CANDIDATE_POOL,
        ann_candidates_returned=0,
    )


def build_ann_candidate_sql(
    *, application_scope_id, collection_ids, e1, dimension, metric, query_values
):
    """The finite-candidate query. Structural predicates and nothing else.

    All three WHERE predicates are PARTITION KEYS - scope, collection and `e1` -
    so PostgreSQL prunes to exactly the authorized leaves before any graph is
    opened. Unauthorized vectors do not lose a filter comparison; they are never
    reached.

    No Knowledge table is joined here, deliberately. A join would push the
    authorization decision after finite candidate selection, which is the shape
    this whole design exists to avoid. Canonical revalidation happens afterwards,
    on the small candidate set.

    Exposed separately so tests can `EXPLAIN` the exact production query rather
    than a lookalike.
    """
    backend = resolve_metric_backend(metric)
    dimension = _validated_dimension(dimension)
    statement = sql.SQL(
        "SELECT source_embedding_id, chunk_id, collection_id "
        "FROM {parent} "
        "WHERE application_scope_id = %s "
        "AND collection_id = ANY(%s) "
        "AND e1 = %s "
        "ORDER BY (embedding::vector({dimension})) {operator} "
        "(%s::vector({dimension})) "
        "LIMIT {pool}"
    ).format(
        parent=sql.Identifier(ANN_PARENT_TABLE),
        dimension=sql.SQL(str(dimension)),
        operator=sql.SQL(backend.operator),
        pool=sql.SQL(str(ANN_CANDIDATE_POOL)),
    )
    parameters = [
        application_scope_id,
        list(collection_ids),
        validate_e1(e1),
        _vector_literal(query_values),
    ]
    return statement, parameters


def search_pgvector_ann_with_scope(
    scope,
    *,
    query_values,
    embedding_model_config,
    collection_id=None,
    limit=20,
) -> PgvectorAnnResult:
    """ANN candidate generation inside an ALREADY-RESOLVED scope, then EXACT rerank.

    Deliberately does NOT call `resolve_effective_knowledge_scope`, so a future
    composing layer can hand one frozen authorization answer to this backend and
    to every other branch - the same single-snapshot property S-22 established.

    Deliberately does NOT call an embedding provider: `query_values` are
    ephemeral numbers the caller already produced under this contract. They are
    validated, used and discarded - never persisted, and never normalized by a
    second implementation, because S-20/S-21 own that.

    HNSW is the candidate GENERATOR. Final ordering is recomputed with S-21's
    exact scorers over canonical decoded vectors, so the backend's ordering
    operator - `<#>` returns a negated inner product - never escapes as a
    semantic value.
    """
    require_pgvector_backend()

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 0
        or limit > MAX_ANN_RESULTS
    ):
        raise PgvectorAnnError(
            PgvectorFailureCategory.INVALID_LIMIT,
            f"limit must be an integer between 0 and {MAX_ANN_RESULTS}.",
        )

    contract = resolve_embedding_contract(embedding_model_config)
    dimension = _validated_dimension(contract.vector_dimension)
    backend = resolve_metric_backend(contract.distance_metric)
    metric_spec = resolve_metric_scorer(contract.distance_metric)

    # -- authorization narrowing. Never widening ---------------------------
    if scope.is_empty:
        return _empty_result(scope)
    target_ids = tuple(sorted(scope.collection_ids))
    if collection_id is not None:
        if not scope.allows(collection_id):
            # ADR-N5: identical to "the corpus is empty".
            return _empty_result(scope)
        target_ids = (int(collection_id),)
    if not target_ids or limit == 0:
        return _empty_result(
            scope, collection_ids=target_ids,
            config_id=getattr(embedding_model_config, "pk", None),
            e1=contract.e1, metric=contract.distance_metric,
        )

    try:
        values = validate_embedding_vector(
            query_values, expected_dimension=dimension
        )
    except EmbeddingVectorError as exc:
        raise PgvectorAnnError(
            PgvectorFailureCategory.QUERY_VECTOR_INVALID, str(exc)
        ) from exc

    # -- cosine parity, BEFORE any ANN SQL ---------------------------------
    # A zero-magnitude query has no direction, so its cosine to anything is
    # undefined and S-21 refuses it. PostgreSQL's answer for `<=>` against a
    # zero vector is not consulted: the canonical semantics decide, and they
    # decide before a candidate query is ever issued.
    if contract.distance_metric == "cosine" and is_cosine_unscorable(values):
        raise PgvectorAnnError(
            PgvectorFailureCategory.UNSCORABLE_ZERO_VECTOR,
            "Cosine similarity is undefined for a zero-magnitude query vector.",
        )

    # -- EVERY target leaf must be ready -----------------------------------
    # One unready collection refuses the whole operation. Dropping it and
    # searching the rest would return a confident ranking drawn from less
    # Knowledge than the caller believes was searched, with nothing to say so.
    for target in target_ids:
        readiness = leaf_readiness(scope.application_scope_id, target, contract)
        if not readiness.ready:
            raise PgvectorAnnError(
                PgvectorFailureCategory.LEAF_NOT_READY,
                f"Collection {target} ANN leaf is not ready ({readiness.reason}).",
            )

    generations_before = current_generations(
        scope.application_scope_id, target_ids
    )

    # -- the ANN query. Structural predicates ONLY -------------------------
    candidate_sql, parameters = build_ann_candidate_sql(
        application_scope_id=scope.application_scope_id,
        collection_ids=target_ids,
        e1=contract.e1,
        dimension=dimension,
        metric=contract.distance_metric,
        query_values=values,
    )

    with transaction.atomic():
        with connection.cursor() as cursor:
            cursor.execute(f"SET LOCAL hnsw.ef_search = {int(HNSW_EF_SEARCH)}")
            cursor.execute(candidate_sql, parameters)
            candidate_rows = cursor.fetchall()

    source_ids = [int(row[0]) for row in candidate_rows]

    # -- canonical revalidation. ANN rows are DERIVED state ----------------
    matches = _rerank_candidates(
        scope, source_ids, contract, target_ids, metric_spec, values
    )

    generations_after = current_generations(
        scope.application_scope_id, target_ids
    )
    if generations_after != generations_before:
        # A concurrent corpus mutation means the graph we just consulted no
        # longer describes the corpus. Refuse rather than return a ranking over
        # an obsolete index; no retry, because retrying is the caller's decision.
        raise PgvectorAnnError(
            PgvectorFailureCategory.SOURCE_CHANGED_DURING_SEARCH,
            "The Knowledge source changed while the ANN search was running.",
        )

    return PgvectorAnnResult(
        matches=tuple(matches[:limit]),
        application_scope_id=scope.application_scope_id,
        collection_ids=target_ids,
        embedding_model_config_id=getattr(embedding_model_config, "pk", None),
        e1=contract.e1,
        backend_version=PGVECTOR_BACKEND_VERSION,
        metric=contract.distance_metric,
        ann_candidate_pool=ANN_CANDIDATE_POOL,
        ann_candidates_returned=len(source_ids),
    )


def _rerank_candidates(scope, source_ids, contract, target_ids, metric_spec, values):
    """Re-load canonical records, verify the mirror, then rank EXACTLY.

    Every ANN candidate must still be present, in this namespace, in an
    authorized target collection, under this `e1`, current by S-19 inspection,
    inside an ACTIVE document and an active collection. If any candidate fails,
    the mirror is corrupt or stale and the whole operation is REFUSED - quietly
    dropping it would return a shorter ranking that looks complete while hiding
    the corruption that produced it.
    """
    if not source_ids:
        return []
    records = {
        record.pk: record
        for record in KnowledgeChunkEmbedding.objects.filter(
            pk__in=source_ids,
            application_scope_id=scope.application_scope_id,
            collection_id__in=target_ids,
            e1=contract.e1,
            chunk__document__collection__is_active=True,
            chunk__document__status=KnowledgeDocument.Status.ACTIVE,
        ).select_related(
            "chunk", "chunk__document", "chunk__document__collection",
            "embedding_model_config", "embedding_model_config__provider",
        )
    }
    scored = []
    for source_id in source_ids:
        record = records.get(source_id)
        if record is None:
            raise PgvectorAnnError(
                PgvectorFailureCategory.ANN_INTEGRITY_MISMATCH,
                "An ANN candidate has no canonical, retrievable source record.",
            )
        if not inspect_vector_record(record).current:
            raise PgvectorAnnError(
                PgvectorFailureCategory.ANN_INTEGRITY_MISMATCH,
                "An ANN candidate's canonical vector is no longer current.",
            )
        if record.vector_dimension != contract.vector_dimension:
            raise PgvectorAnnError(
                PgvectorFailureCategory.ANN_INTEGRITY_MISMATCH,
                "An ANN candidate's canonical dimension does not match.",
            )
        canonical = decode_vector(
            record.vector_bytes,
            expected_dimension=record.vector_dimension,
            vector_format=record.vector_format,
        )
        scored.append((metric_spec.score(values, canonical), record))

    # Exactly S-21's ordering, including the deterministic tie-break. The ANN
    # distance plays no part: it selected the pool and nothing else.
    scored.sort(
        key=lambda entry: (
            -entry[0] if metric_spec.higher_is_better else entry[0],
            entry[1].chunk_id,
        )
    )
    return [
        PgvectorAnnMatch(
            rank=position,
            chunk_id=record.chunk_id,
            document_id=record.chunk.document_id,
            collection_id=record.collection_id,
            application_scope_id=record.application_scope_id,
            k1=record.k1,
            e1=record.e1,
            metric=contract.distance_metric,
            metric_value=float(value),
        )
        for position, (value, record) in enumerate(scored, start=1)
    ]
