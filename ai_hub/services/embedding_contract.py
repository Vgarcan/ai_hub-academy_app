"""The `e1` embedding contract fingerprint, and how to resolve one.

`e1` identifies the VECTOR-SPACE CONTRACT: whether two vectors belong to the
same compatible retrieval space. It is not a permission, not a routing fact and
not an operational setting.

Future vector validity will conceptually be:

    vector is current  <=>  k1 content identity  AND  e1 contract identity

S-18 implements `e1` only. `k1` is deliberately NOT defined here - it belongs
with the indexing/persistence pipeline that will need it, and choosing its
representation now would mean choosing a persistence design now. `i1` and `c1`
(Knowledge lifecycle) are untouched and orthogonal.

**What this module must never do:** call a provider, read a credential, resolve
an API key, perform egress authorization, generate an embedding, normalize a
vector, touch Knowledge text, or create vector storage. It resolves and
fingerprints configuration, and nothing else.

The four boundaries a future embedding execution must satisfy INDEPENDENTLY:

    EffectiveKnowledgeScope   which Knowledge may be used          (S-15)
    ResolvedEmbeddingContract what vector-space contract applies   (S-18, here)
    EmbeddingAccessDecision   whether that provider path is allowed (S-17)
    execution adapter         actually performs the embedding      (future)

None of them may substitute for or widen another. In particular, resolving a
contract successfully says nothing about permission: a provider whose
`declared_locality` is UNKNOWN still yields a perfectly valid contract, and S-17
will still refuse to let it be used.
"""

import hashlib
import json
import unicodedata
from dataclasses import dataclass

from ai_hub.models import EmbeddingModelConfig


#: Self-describing so persisted future evidence can be read without a lookup
#: table. Never return a bare digest.
E1_PREFIX = "e1:sha256:"


class EmbeddingContractError(ValueError):
    """The configuration cannot yield a usable embedding contract."""


def _canonical_text(value) -> str:
    """Unicode NFC only.

    Deliberately does NOT lowercase, strip or otherwise rewrite: model
    identifiers are opaque vendor strings, and `Text-Embedding-3` is not
    necessarily the same model as `text-embedding-3`. NFC alone means two byte
    sequences that Unicode says are the same character sequence fingerprint
    identically, which is a representation fix rather than a semantic one.
    """
    return unicodedata.normalize("NFC", "" if value is None else str(value))


def embedding_contract_payload(
    *,
    provider_type,
    model_name,
    model_revision,
    vector_dimension,
    distance_metric,
    normalization,
) -> dict:
    """The canonical `e1` payload: vector-space compatibility facts ONLY.

    `provider_type` is included as a conservative namespace - two unrelated
    provider families can use the same model string and revision label, and
    assuming those produce interchangeable vectors would be an unforced error.

    Everything about routing, deployment, authorization, credentials and
    operational limits is excluded, because none of it changes what a vector
    means: moving the same model between two internal endpoints must not
    invalidate a corpus.
    """
    return {
        "contract": "e1",
        "provider_type": _canonical_text(provider_type),
        "model_name": _canonical_text(model_name),
        "model_revision": _canonical_text(model_revision),
        "vector_dimension": int(vector_dimension),
        "distance_metric": _canonical_text(distance_metric),
        "normalization": _canonical_text(normalization),
    }


def embedding_contract_fingerprint(**contract_facts) -> str:
    """`e1:sha256:<64 hex>` over the canonical payload.

    Canonical JSON rather than delimiter concatenation, matching the `i1`/`c1`
    house convention: escaping makes the serialization unambiguous, so a value
    containing a separator cannot forge another field. `sort_keys` removes any
    dependence on dict insertion order, `ensure_ascii` makes the encode step
    platform-independent, and `separators` removes incidental whitespace.
    """
    payload = embedding_contract_payload(**contract_facts)
    serialized = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{E1_PREFIX}{digest}"


@dataclass(frozen=True)
class ResolvedEmbeddingContract:
    """One resolved embedding configuration. Ephemeral: never persisted.

    Carries configuration facts and the `e1` fingerprint. Never a credential,
    never an environment variable value, never Knowledge text, never a query,
    never a vector.

    `declared_locality` is REPORTED here for observability and is read from
    `ProviderConfig` - the single authoritative source established in S-17. It
    is deliberately not stored on `EmbeddingModelConfig`, and it is deliberately
    not part of `e1`: locality governs permission, not vector identity.
    """

    embedding_model_config_id: int
    provider_id: int
    provider_type: str

    model_name: str
    model_revision: str

    vector_dimension: int
    distance_metric: str
    normalization: str

    max_input_chars: int
    request_timeout_seconds: int

    declared_locality: str

    e1: str


def resolve_embedding_contract(embedding_model_config) -> ResolvedEmbeddingContract:
    """Resolve a usable contract, or raise. Fail closed; never repair.

    Raises rather than returning a sentinel, because unlike an authorization
    denial there is no adversary to withhold a reason from: a malformed
    configuration is an operator's own mistake, and naming the field is the
    helpful thing to do.

    Never silently repairs a malformed value, never switches to another
    configuration, never switches provider, and never selects a "default"
    embedding model. Nothing is inferred.

    Re-validates everything `clean()` checks, because raw ORM bypasses
    `full_clean()` entirely and this is the runtime boundary.
    """
    config = embedding_model_config
    if config is None or getattr(config, "pk", None) is None:
        raise EmbeddingContractError("An embedding model configuration is required.")
    if not config.is_active:
        raise EmbeddingContractError(
            f"Embedding model configuration '{config.name}' is not active."
        )

    provider = getattr(config, "provider", None)
    if provider is None or getattr(provider, "pk", None) is None:
        raise EmbeddingContractError(
            f"Embedding model configuration '{config.name}' has no provider."
        )
    if not provider.is_active:
        raise EmbeddingContractError(
            f"Provider '{provider.name}' is not active."
        )

    model_name = _canonical_text(config.model_name).strip()
    if not model_name:
        raise EmbeddingContractError("Embedding model name is required.")
    model_revision = _canonical_text(config.model_revision).strip()
    if not model_revision:
        # Load-bearing: the endpoint is not part of e1, so the revision is the
        # operator's only assertion that the vector space is unchanged.
        raise EmbeddingContractError(
            "Embedding model revision is required and identifies the vector space."
        )

    for field in ("vector_dimension", "max_input_chars", "request_timeout_seconds"):
        value = getattr(config, field, None)
        if not isinstance(value, int) or value <= 0:
            raise EmbeddingContractError(
                f"{field.replace('_', ' ').capitalize()} must be a positive integer."
            )

    if config.distance_metric not in EmbeddingModelConfig.DistanceMetric.values:
        raise EmbeddingContractError(
            f"Unsupported distance metric {config.distance_metric!r}."
        )
    if config.normalization not in EmbeddingModelConfig.Normalization.values:
        raise EmbeddingContractError(
            f"Unsupported normalization contract {config.normalization!r}."
        )

    return ResolvedEmbeddingContract(
        embedding_model_config_id=config.pk,
        provider_id=provider.pk,
        provider_type=provider.provider_type,
        model_name=model_name,
        model_revision=model_revision,
        vector_dimension=config.vector_dimension,
        distance_metric=config.distance_metric,
        normalization=config.normalization,
        max_input_chars=config.max_input_chars,
        request_timeout_seconds=config.request_timeout_seconds,
        # Reported, not owned. S-17's ProviderConfig remains authoritative, and
        # this value has no effect on `e1`.
        declared_locality=provider.declared_locality,
        e1=embedding_contract_fingerprint(
            provider_type=provider.provider_type,
            model_name=model_name,
            model_revision=model_revision,
            vector_dimension=config.vector_dimension,
            distance_metric=config.distance_metric,
            normalization=config.normalization,
        ),
    )
