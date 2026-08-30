"""Pre-filtered semantic retrieval over CURRENT chunk vectors, LOCAL providers only.

The whole point of this module is an **ordering**, not an algorithm:

    authorization  ->  candidate generation  ->  ranking

Nothing unauthorized is ever loaded, so nothing unauthorized is ever decoded,
scored, ranked, counted or tie-broken. The opposite arrangement - rank the corpus
globally, then drop what the caller may not see - is FORBIDDEN here, and it is
forbidden for a reason that survives changing the backend: a global ranking has
already read foreign Knowledge into memory, and its *scores* leak the existence
and similarity of documents the caller was never entitled to know about. Even the
count of what got filtered away is a signal. Post-filtering is not a slower
version of pre-filtering; it is a different and weaker security property.

That is why the pre-filter here is expressed as a **query predicate** (S-15
`collection_ids` narrowing S-19's scoped load) rather than as a `if allowed`
check inside a scoring loop. When this is re-expressed against a real ANN backend,
the same rule holds: the filter must reach the index, not the result set.

Scope of this slice, deliberately narrow:

* **Reference ranking only.** Exact metrics over decoded `f32le1` vectors in
  Python. This is the correctness oracle a future ANN backend must agree with,
  not the production retrieval path - hence the hard candidate ceiling below.
* **LOCAL providers only**, by `ProviderConfig.declared_locality` and by nothing
  else. No URL, hostname, scheme or provider-name inspection anywhere.
* **`PAYLOAD_QUERY`, never `PAYLOAD_CORPUS`.** A query is the operator's or the
  end user's own text, which is why S-17 gives it a strictly narrower egress
  gate than the corpus. Sending a query under the corpus decision would let
  material leave under permission granted for something else.
* **Read-only.** No transaction, no `select_for_update`, no write. Ranking must
  never hold a database lock across a network call.

The query text is canonicalized for **representation only** before it is sent -
CRLF and lone CR become LF, and Unicode is composed to NFC - so that two callers
who typed the same thing on different platforms land in the same place in the
vector space. Nothing else is touched: leading, trailing and internal whitespace,
case and punctuation all survive exactly. See `canonical_query_embedding_text`.

The query vector is **ephemeral**: computed, used to rank, discarded. It is never
persisted, never cached and has no fingerprint contract. There is deliberately no
`q1`: `k1` and `e1` exist because a *stored* artefact must be able to prove what
it represents and which vector space it lives in. A query vector is stored
nowhere, so it has nothing to prove - and inventing an identity for it would
create a durable record of what people searched for.
"""

import math
import unicodedata
from dataclasses import dataclass

from ai_hub.models import EmbeddingModelConfig
from ai_hub.services.chunk_embedding_identity import chunk_embedding_fingerprint
from ai_hub.services.embedding_client import resolve_embedding_transport
from ai_hub.services.embedding_contract import resolve_embedding_contract
from ai_hub.services.embedding_egress import (
    PAYLOAD_QUERY,
    ReasonCode,
    resolve_embedding_access,
)
from ai_hub.services.embedding_vector import (
    EmbeddingVectorError,
    normalize_embedding_vector,
    validate_embedding_vector,
)
from ai_hub.services.knowledge_authorization import (
    authorized_chunks,
    authorized_collections,
    resolve_effective_knowledge_scope,
)
from ai_hub.services.vector_store import load_current_vectors

#: The reference backend decodes and scores every candidate in Python, so its
#: honest working range is small. Exceeding this ceiling is a REFUSAL, never a
#: truncation: silently ranking the first N of M candidates would return a
#: confident top-5 that is not actually the top-5, and nothing in the result
#: would reveal that. A corpus that outgrows this needs a real vector index -
#: that is a capability decision for a human, not something this module may make
#: on its own by quietly dropping data.
MAX_REFERENCE_SEMANTIC_CANDIDATES = 1000


class RetrievalFailureCategory:
    """Bounded machine-readable refusals.

    None of these carries query text, Knowledge content, vector values or a
    credential. A retrieval error must never become a way to read back what was
    searched for or what was searched over.
    """

    QUERY_EMPTY = "query_empty"
    INVALID_LIMIT = "invalid_limit"
    SCOPE_UNAVAILABLE = "scope_unavailable"
    EMBEDDING_NOT_AUTHORIZED = "embedding_not_authorized"
    LOCAL_ONLY_EXECUTION_REQUIRED = "local_only_execution_required"
    UNSUPPORTED_DISTANCE_METRIC = "unsupported_distance_metric"
    QUERY_INPUT_TOO_LARGE = "query_input_too_large"
    REFERENCE_CANDIDATE_LIMIT_EXCEEDED = "reference_candidate_limit_exceeded"
    VECTOR_DIMENSION_MISMATCH = "vector_dimension_mismatch"
    VECTOR_NON_FINITE = "vector_non_finite"
    ZERO_VECTOR_CANNOT_L2_NORMALIZE = "zero_vector_cannot_l2_normalize"
    CANDIDATE_DIMENSION_MISMATCH = "candidate_dimension_mismatch"
    UNSCORABLE_ZERO_VECTOR = "unscorable_zero_vector"


class SemanticRetrievalError(RuntimeError):
    """A refusal, carrying a bounded category and never the query."""

    def __init__(self, category: str, message: str = ""):
        self.category = category
        super().__init__(message or category)


def canonical_query_embedding_text(query) -> str:
    """The REPRESENTATION-ONLY canonical form of a query. Pure; never persisted.

    Two callers who typed the same thing on different platforms must reach the
    same point in the vector space. A Windows client sending CRLF and a Unix
    client sending LF have typed the same query; a macOS client sending
    decomposed `e` + U+0301 and a Linux client sending precomposed U+00E9 have
    typed the same word. Left alone, those produce different embeddings and
    therefore different rankings for identical intent - a difference no operator
    would ever be able to see or explain.

    So this normalizes REPRESENTATION and nothing else:

        CRLF     -> LF
        lone CR  -> LF
        Unicode  -> NFC

    Everything a person could have meant is preserved exactly: leading and
    trailing whitespace, internal whitespace, case and punctuation. Leading and
    trailing spaces are content here, not noise - a query is free text, and
    trimming it would silently embed something the caller did not send.

    Deliberately NOT a fingerprint. There is no `q1`, nothing is stored, and
    this must never be confused with `canonical_chunk_embedding_text`/`k1`
    (which fingerprints a STORED artefact and is not touched by this) or with
    Knowledge lifecycle normalization (which governs curated content, not a
    transient search string).
    """
    text = query if isinstance(query, str) else ""
    # CRLF first, so a CRLF pair never becomes two newlines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# Exact metrics
#
# Each takes two already-validated, equal-length sequences and returns a float.
# They are pure: no policy, no authorization, no persistence, no I/O. They are
# reached ONLY through `METRIC_SCORERS`, so the ordering guarantee above can be
# proven by observing which candidates a scorer is ever called with.
# ---------------------------------------------------------------------------

def cosine_similarity(query, candidate) -> float:
    """Exact cosine similarity. Computes BOTH norms; assumes no unit vectors.

    Deliberately does not shortcut to a dot product "because the metric is
    cosine". Normalization is an independent S-18 contract fact, so a
    `cosine` + `none` configuration is entirely legal, and a scorer that assumed
    unit length there would silently rank by magnitude instead of by direction.
    """
    dot = 0.0
    query_square = 0.0
    candidate_square = 0.0
    for left, right in zip(query, candidate):
        dot += left * right
        query_square += left * left
        candidate_square += right * right
    query_norm = math.sqrt(query_square)
    candidate_norm = math.sqrt(candidate_square)
    if query_norm <= 0.0 or candidate_norm <= 0.0:
        # A zero vector has no direction, so its cosine to anything is
        # undefined. Refuse rather than substituting 0.0, which would be an
        # invented "perfectly orthogonal" score.
        raise SemanticRetrievalError(
            RetrievalFailureCategory.UNSCORABLE_ZERO_VECTOR,
            "Cosine similarity is undefined for a zero-magnitude vector.",
        )
    return dot / (query_norm * candidate_norm)


def dot_product_similarity(query, candidate) -> float:
    """Exact inner product. No normalization is applied or assumed."""
    return float(sum(left * right for left, right in zip(query, candidate)))


def euclidean_distance(query, candidate) -> float:
    """Exact L2 DISTANCE - raw, not inverted and not converted to a similarity.

    Reported as-is so the number in `metric_value` means what its metric says it
    means. Flipping it into a similarity here would silently redefine the
    operator-declared metric, and any consumer reading `metric_value` alongside
    `metric` would be reading a lie.
    """
    total = 0.0
    for left, right in zip(query, candidate):
        delta = left - right
        total += delta * delta
    return math.sqrt(total)


@dataclass(frozen=True)
class MetricSpec:
    """A scorer plus its ordering direction, kept together on purpose.

    Direction is a property OF the metric. Storing it anywhere else invites a
    ranking loop that sorts descending by habit and quietly returns the least
    similar results for a distance metric.
    """

    score: object
    higher_is_better: bool


#: Explicit registry, not discovery. Adding a metric is a deliberate act that
#: must also state its ordering direction.
METRIC_SCORERS = {
    EmbeddingModelConfig.DistanceMetric.COSINE: MetricSpec(
        score=cosine_similarity, higher_is_better=True
    ),
    EmbeddingModelConfig.DistanceMetric.DOT_PRODUCT: MetricSpec(
        score=dot_product_similarity, higher_is_better=True
    ),
    EmbeddingModelConfig.DistanceMetric.EUCLIDEAN: MetricSpec(
        score=euclidean_distance, higher_is_better=False
    ),
}


def resolve_metric_scorer(metric: str) -> MetricSpec:
    """The scorer for a declared metric, or refuse. Never a default.

    An unknown metric must not fall back to cosine. A configuration naming a
    metric this build cannot compute is a configuration error, and answering it
    with a different metric's ranking would be worse than answering nothing.
    """
    spec = METRIC_SCORERS.get(metric)
    if spec is None:
        raise SemanticRetrievalError(
            RetrievalFailureCategory.UNSUPPORTED_DISTANCE_METRIC,
            f"No exact scorer is implemented for distance metric {metric!r}.",
        )
    return spec


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticMatch:
    """One ranked chunk. Identifiers and a score - never Knowledge text.

    Text retrieval is a separate, authorized read. Returning content from the
    ranking layer would make every future caller of this function an implicit
    content-disclosure path, and the scoring result is not the place to decide
    what a caller may read.
    """

    rank: int
    chunk_id: int
    document_id: int
    collection_id: int
    application_scope_id: int
    k1: str
    e1: str
    metric: str
    metric_value: float
    higher_is_better: bool


@dataclass(frozen=True)
class SemanticRetrievalResult:
    """One retrieval operation. Ephemeral; never persisted.

    Reports the *shape* of the search - which namespace, which collections, how
    many candidates were eligible, whether the provider was called - so an
    operator can audit an empty result without having to guess whether it meant
    "denied", "nothing indexed" or "nothing matched". It deliberately carries no
    query text, no query vector and no Knowledge content.
    """

    matches: tuple
    application_scope_id: int | None
    agent_id: int | None
    workspace_id: int | None
    collection_ids: tuple
    embedding_model_config_id: int | None
    e1: str
    metric: str
    higher_is_better: bool
    candidate_count: int
    scored_count: int
    provider_invoked: bool


_EMPTY_COLLECTIONS = ()


def _empty_result(
    scope,
    *,
    collection_ids=_EMPTY_COLLECTIONS,
    embedding_model_config_id=None,
    e1="",
    metric="",
    higher_is_better=True,
    candidate_count=0,
) -> SemanticRetrievalResult:
    """An authorized-but-empty answer. Never an error, never a partial one."""
    return SemanticRetrievalResult(
        matches=(),
        application_scope_id=scope.application_scope_id,
        agent_id=scope.agent_id,
        workspace_id=scope.workspace_id,
        collection_ids=tuple(collection_ids),
        embedding_model_config_id=embedding_model_config_id,
        e1=e1,
        metric=metric,
        higher_is_better=higher_is_better,
        candidate_count=candidate_count,
        scored_count=0,
        provider_invoked=False,
    )


def _translate_vector_error(exc: EmbeddingVectorError) -> SemanticRetrievalError:
    """Re-raise a shared vector refusal in this module's vocabulary.

    An identity mapping: `RetrievalFailureCategory` reuses the same strings as
    S-20's `FailureCategory` for the same conditions, so a corpus vector and a
    query vector that fail the same way report the same way.
    """
    return SemanticRetrievalError(exc.category, str(exc))


def semantic_search_knowledge_local(
    agent,
    *,
    query,
    embedding_model_config,
    workspace=None,
    collection_id=None,
    limit=5,
) -> SemanticRetrievalResult:
    """Rank the Agent's AUTHORIZED chunks against one query. Fail closed.

    The caller supplies who is asking, what they are asking and which vector
    space to ask in. It does not supply a scope, a collection set, an `e1`, a
    metric or a vector - every one of those is derived, because a caller-supplied
    authorization fact is not an authorization fact.

    `collection_id` can only ever NARROW what S-15 already granted; it can never
    add a collection. A caller naming a collection it may not reach gets the same
    empty answer as a caller whose corpus is simply empty (ADR-N5): the shape of
    a refusal must not become a way to enumerate collections.

    This is the ONLY place in the semantic path that resolves authorization. The
    work itself lives in `search_semantic_with_scope`, so a composing caller
    (S-22 hybrid) can hand ONE frozen scope to several branches instead of each
    branch resolving its own - two branches resolving separately could run one
    user-visible search under two different authorization answers.
    """
    scope = resolve_effective_knowledge_scope(agent, workspace=workspace)
    return search_semantic_with_scope(
        scope,
        query=query,
        embedding_model_config=embedding_model_config,
        collection_id=collection_id,
        limit=limit,
    )


def search_semantic_with_scope(
    scope,
    *,
    query,
    embedding_model_config,
    collection_id=None,
    limit=5,
) -> SemanticRetrievalResult:
    """Rank AUTHORIZED chunks inside an ALREADY-RESOLVED scope. Fail closed.

    Deliberately does NOT call `resolve_effective_knowledge_scope`, and is not a
    way to hand in authorization from outside: `EffectiveKnowledgeScope` is only
    ever produced by S-15, and every narrowing below still intersects with it.

    Everything the public entry point promised holds here unchanged - the
    ordering invariant, `PAYLOAD_QUERY`, LOCAL-only execution, the candidate
    ceiling and post-inference revalidation.
    """
    # -- 1. content-free structural checks ---------------------------------
    # Before authorization, because they inspect nothing but the caller's own
    # arguments and leak nothing about the corpus.
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise SemanticRetrievalError(
            RetrievalFailureCategory.INVALID_LIMIT,
            "limit must be a non-negative integer.",
        )
    # Canonicalized ONCE, here, and this exact value is what gets length-checked
    # and dispatched. Deriving it twice, or checking one form and sending
    # another, is how a query silently becomes a different query.
    query_text = canonical_query_embedding_text(query)
    if query_text.strip() == "":
        # `.strip()` is the emptiness PREDICATE only; it never produces the text
        # that is sent. Unlike S-20's exact `== ""` check on canonical chunk
        # text - where a whitespace-only chunk is a real `k1` input that must
        # round-trip byte for byte - a whitespace-only query carries no
        # retrieval intent at all, so refusing it costs nothing. What is
        # dispatched below is `query_text`, unstripped.
        raise SemanticRetrievalError(
            RetrievalFailureCategory.QUERY_EMPTY,
            "A non-empty query is required.",
        )

    # -- 2. AUTHORIZATION. Already resolved; the source of every narrowing --
    if scope.is_empty:
        return _empty_result(scope)

    # -- 3. the authorized target collections ------------------------------
    if collection_id is not None and not scope.allows(collection_id):
        # Narrow-only. Never widened, never "helpfully" ignored.
        return _empty_result(scope)

    collections = list(
        authorized_collections(scope).select_related("application_scope")
    )
    if collection_id is not None:
        collections = [
            collection
            for collection in collections
            if collection.pk == int(collection_id)
        ]
    if not collections:
        return _empty_result(scope)

    target_ids = tuple(sorted(collection.pk for collection in collections))

    application_scope = collections[0].application_scope
    if (
        application_scope is None
        or application_scope.pk != scope.application_scope_id
        or not application_scope.is_active
    ):
        # S-15 already guarantees this; asserting it here means a future change
        # to either module cannot quietly widen the namespace being searched.
        raise SemanticRetrievalError(
            RetrievalFailureCategory.SCOPE_UNAVAILABLE,
            "The authorized application scope is not usable.",
        )

    # -- 4. the embedding contract -----------------------------------------
    # The OPERATIONAL resolver: a query embedding is a NEW inference, so the
    # configuration and provider must be active right now. (Contrast S-19's pure
    # fingerprint, which decides whether a STORED vector is still current and
    # must not depend on active state.)
    contract = resolve_embedding_contract(embedding_model_config)
    provider = embedding_model_config.provider

    # -- 5. S-17 QUERY egress, for EVERY target collection -----------------
    # Every collection is authorized BEFORE any capability question is asked,
    # so a search is never permitted on the strength of the collections that
    # happened to be checked first.
    decisions = []
    for collection in collections:
        decision = resolve_embedding_access(
            application_scope,
            provider,
            collection=collection,
            payload_kind=PAYLOAD_QUERY,
        )
        if not decision.allowed:
            # Refuse the whole search rather than dropping this collection and
            # ranking the rest. A silently narrower search returns a confident
            # top-5 drawn from less Knowledge than the caller believes it
            # searched, and nothing in the result would say so.
            raise SemanticRetrievalError(
                RetrievalFailureCategory.EMBEDDING_NOT_AUTHORIZED,
                f"Query embedding is not authorized ({decision.reason_code}).",
            )
        decisions.append(decision)

    # -- 6. local-only, by declared locality alone -------------------------
    for decision in decisions:
        if (
            decision.reason_code != ReasonCode.ALLOWED_LOCAL
            or decision.requires_external_egress
        ):
            # S-17 may legitimately permit EXTERNAL query egress. This slice
            # still refuses, because it does not implement external execution.
            # A capability limit, never an inference about where a provider is.
            raise SemanticRetrievalError(
                RetrievalFailureCategory.LOCAL_ONLY_EXECUTION_REQUIRED,
                "This operation executes only against providers declared LOCAL.",
            )

    # -- 7. capability checks, before anything expensive -------------------
    # Resolved now so an unsupported metric or transport refuses without having
    # spent an inference or sent the query anywhere.
    metric_spec = resolve_metric_scorer(contract.distance_metric)
    transport = resolve_embedding_transport(provider)

    if len(query_text) > contract.max_input_chars:
        # Measured on the CANONICAL form, because that is what will be sent.
        # NFC composition can shorten a string (decomposed `e` + U+0301 is two
        # characters; the composed U+00E9 is one), so measuring the raw input
        # would refuse queries that fit and admit queries that do not.
        #
        # Never truncate. A truncated query embeds something the caller did not
        # ask, and the results would look perfectly reasonable.
        raise SemanticRetrievalError(
            RetrievalFailureCategory.QUERY_INPUT_TOO_LARGE,
            f"The query is {len(query_text)} characters; "
            f"the contract allows {contract.max_input_chars}.",
        )

    # -- 8. CANDIDATE GENERATION. Pre-filtered, and before any inference ---
    # The eligible chunk set is derived from S-15 FIRST, so the candidate set is
    # bounded by authorization rather than corrected by it afterwards.
    authorized_ids = frozenset(
        authorized_chunks(scope).values_list("pk", flat=True)
    )

    # `load_current_vectors` narrows in SQL by scope, `e1` and the authorized
    # collection ids, and drops anything whose `k1`/`e1` is no longer current.
    # Nothing outside `target_ids` is ever loaded, so nothing belonging to
    # another application or another collection can be decoded, scored, ranked
    # or counted.
    #
    # S-19's loader takes collection ids only, so ONE authorization predicate -
    # `KnowledgeDocument.status` - cannot reach that SQL. It is applied here, in
    # the same step, before the ceiling and before anything is ranked or
    # counted. The residue it removes is narrow and same-tenant by construction:
    # vectors of ARCHIVED documents inside collections this caller is already
    # authorized to read. A real vector index must carry this predicate into the
    # index too, exactly as it must carry the collection filter.
    candidates = [
        candidate
        for candidate in load_current_vectors(
            application_scope=application_scope,
            e1=contract.e1,
            collection_ids=target_ids,
        )
        if candidate.chunk_id in authorized_ids
    ]
    candidate_count = len(candidates)

    if candidate_count > MAX_REFERENCE_SEMANTIC_CANDIDATES:
        raise SemanticRetrievalError(
            RetrievalFailureCategory.REFERENCE_CANDIDATE_LIMIT_EXCEEDED,
            f"{candidate_count} eligible candidates exceed the reference "
            f"backend ceiling of {MAX_REFERENCE_SEMANTIC_CANDIDATES}.",
        )

    empty_shape = {
        "collection_ids": target_ids,
        "embedding_model_config_id": embedding_model_config.pk,
        "e1": contract.e1,
        "metric": contract.distance_metric,
        "higher_is_better": metric_spec.higher_is_better,
        "candidate_count": candidate_count,
    }
    if candidate_count == 0 or limit == 0:
        # No candidates means nothing to rank; `limit == 0` means nothing was
        # asked for. Either way the provider is NOT called: an embedding request
        # whose answer cannot change the outcome is pure egress with no purpose.
        return _empty_result(scope, **empty_shape)

    # -- 9. the provider call. Outside any transaction, holding no locks ---
    # Ranking is read-only, so there is nothing to lock; and a network call
    # inside a transaction would hold database resources for the length of an
    # inference.
    provider_result = transport(
        provider=provider, contract=contract, text=query_text
    )
    try:
        raw_values = validate_embedding_vector(
            provider_result.values, expected_dimension=contract.vector_dimension
        )
        # The SAME normalization the corpus went through, from the SAME module.
        # Two vectors compared by a metric must have been produced identically.
        query_values = normalize_embedding_vector(
            raw_values, normalization=contract.normalization
        )
    except EmbeddingVectorError as exc:
        raise _translate_vector_error(exc) from exc

    # -- 10. post-inference revalidation -----------------------------------
    # Time passed during the network call, so the authorized chunk set and each
    # chunk's identity are re-derived from the database before anything is
    # scored: a chunk whose collection was deactivated, whose document was
    # archived, which was deleted, or which was edited mid-flight is dropped
    # rather than ranked.
    #
    # The `scope` itself is deliberately NOT re-resolved. It is the
    # authorization decision FOR this operation, taken once at step 2; resolving
    # it again halfway through would mean one search ran under two different
    # authorization answers, which is harder to audit than a search that ran
    # under the one it started with. A revoked assignment takes effect on the
    # next search, which is a decision, not a race.
    surviving = {
        chunk.pk: chunk
        for chunk in authorized_chunks(scope).filter(
            pk__in=[candidate.chunk_id for candidate in candidates]
        )
    }

    scored = []
    for candidate in candidates:
        chunk = surviving.get(candidate.chunk_id)
        if chunk is None:
            continue
        if chunk_embedding_fingerprint(chunk) != candidate.k1:
            # The chunk's canonical embedding input changed while the query was
            # in flight, so this vector no longer represents this chunk. Drop
            # it; never re-embed on the caller's behalf.
            continue
        if len(candidate.values) != contract.vector_dimension:
            # Unreachable through `e1` alone, but a vector of the wrong length
            # would otherwise be scored against a truncating `zip` and produce a
            # plausible number from a partial comparison.
            raise SemanticRetrievalError(
                RetrievalFailureCategory.CANDIDATE_DIMENSION_MISMATCH,
                f"A stored vector has {len(candidate.values)} components; "
                f"the contract requires {contract.vector_dimension}.",
            )
        scored.append(
            (metric_spec.score(query_values, candidate.values), candidate)
        )

    if not scored:
        return _empty_result(scope, **empty_shape)

    # -- 11. ranking, entirely over already-authorized candidates ----------
    # `chunk_id` ASC breaks ties deterministically, so equal scores never
    # reorder between runs, backends or database row orders.
    scored.sort(
        key=lambda entry: (
            -entry[0] if metric_spec.higher_is_better else entry[0],
            entry[1].chunk_id,
        )
    )

    matches = tuple(
        SemanticMatch(
            rank=position,
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            collection_id=candidate.collection_id,
            application_scope_id=candidate.application_scope_id,
            k1=candidate.k1,
            e1=candidate.e1,
            metric=contract.distance_metric,
            metric_value=float(value),
            higher_is_better=metric_spec.higher_is_better,
        )
        for position, (value, candidate) in enumerate(scored[:limit], start=1)
    )

    return SemanticRetrievalResult(
        matches=matches,
        application_scope_id=scope.application_scope_id,
        agent_id=scope.agent_id,
        workspace_id=scope.workspace_id,
        collection_ids=target_ids,
        embedding_model_config_id=embedding_model_config.pk,
        e1=contract.e1,
        metric=contract.distance_metric,
        higher_is_better=metric_spec.higher_is_better,
        candidate_count=candidate_count,
        scored_count=len(scored),
        provider_invoked=True,
    )
