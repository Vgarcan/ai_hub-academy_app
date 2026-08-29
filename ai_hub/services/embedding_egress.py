"""May this application's text be sent to this provider for embedding?

The policy boundary that must be satisfied BEFORE AI Hub sends Knowledge corpus
text or query text to any embedding provider. It generates no embeddings, calls
no provider, and creates no vectors. It answers only:

    May this ApplicationScope use this provider for embeddings?
    Would the operation leave the declared trust boundary?
    If so - may corpus text leave? may query text leave?
    Does this KnowledgeCollection narrow that permission?

Four concepts, kept deliberately apart:

    ProviderConfig     a global definition of a configured endpoint. Describes
                       the deployment. Grants nothing.
    ProviderGrant      an explicit ApplicationScope -> ProviderConfig permission
                       for embedding use.
    ApplicationScope   OWNS the external egress policy.
    KnowledgeCollection may NARROW it. Never widen it.

**THE RESOLVER RECEIVES NO CONTENT.** It takes a scope, a provider, a collection
and a payload kind - never document text, chunk text, query text, a prompt or an
embedding input. The decision is made before sensitive content reaches the
provider boundary, so a policy denial cannot be a place where content has
already been assembled.

Two invariants worth stating plainly, because a future reader will be tempted to
collapse them:

    Provider authorization NEVER grants Knowledge authorization.
    Knowledge authorization NEVER grants provider egress authorization.

`EffectiveKnowledgeScope` (S-15) answers "may this Agent read this Collection?".
This module answers "may this Collection's text be processed through this
Provider?". A future external semantic operation needs BOTH to succeed
independently, and neither may widen the other. They are not merged, and must
not be.
"""

from dataclasses import dataclass

from ai_hub.models import ProviderConfig, ProviderGrant


PAYLOAD_CORPUS = "corpus"
PAYLOAD_QUERY = "query"
#: Exactly two kinds. There is no generic payload kind on purpose: a caller that
#: cannot say which of these it is holding has not decided what it is sending.
PAYLOAD_KINDS = frozenset({PAYLOAD_CORPUS, PAYLOAD_QUERY})


class ReasonCode:
    """Machine-readable denial reasons.

    Deliberately descriptive for OPERATORS reading logs, and deliberately never
    surfaced to a model or an end user - a denial reason names configuration,
    and configuration is not something a caller is entitled to enumerate.
    """

    ALLOWED_LOCAL = "allowed_local"
    ALLOWED_EXTERNAL = "allowed_external"

    UNKNOWN_PAYLOAD_KIND = "unknown_payload_kind"
    SCOPE_MISSING = "scope_missing"
    SCOPE_INACTIVE = "scope_inactive"
    PROVIDER_MISSING = "provider_missing"
    PROVIDER_INACTIVE = "provider_inactive"
    PROVIDER_LOCALITY_UNDECLARED = "provider_locality_undeclared"
    COLLECTION_MISSING = "collection_missing"
    COLLECTION_INACTIVE = "collection_inactive"
    COLLECTION_FOREIGN_SCOPE = "collection_foreign_scope"
    NO_PROVIDER_GRANT = "no_provider_grant"
    GRANT_DOES_NOT_ALLOW_EMBEDDINGS = "grant_does_not_allow_embeddings"
    SCOPE_DENIES_EXTERNAL_CORPUS = "scope_denies_external_corpus"
    SCOPE_DENIES_EXTERNAL_QUERY = "scope_denies_external_query"
    COLLECTION_DENIES_EXTERNAL = "collection_denies_external"


@dataclass(frozen=True)
class EmbeddingAccessDecision:
    """One policy decision. Ephemeral: never a model, never persisted.

    Carries no content and no credential - only the identifiers and the facts
    that produced the outcome.
    """

    allowed: bool
    application_scope_id: int | None
    provider_id: int | None
    collection_id: int | None
    payload_kind: str
    declared_locality: str
    requires_external_egress: bool
    reason_code: str


def _deny(reason, *, scope=None, provider=None, collection=None,
          payload_kind="", locality="", requires_external=False):
    return EmbeddingAccessDecision(
        allowed=False,
        application_scope_id=getattr(scope, "pk", None),
        provider_id=getattr(provider, "pk", None),
        collection_id=getattr(collection, "pk", None),
        payload_kind=payload_kind,
        declared_locality=locality,
        requires_external_egress=requires_external,
        reason_code=reason,
    )


def resolve_embedding_access(
    application_scope,
    provider,
    *,
    collection,
    payload_kind,
) -> EmbeddingAccessDecision:
    """Decide whether embedding use would be permitted. Fail closed throughout.

    Never falls back to another provider, never downgrades to a local provider,
    never ignores a collection policy and never proceeds anyway. Provider
    selection and fallback are future work; this function's only job is to say
    yes or no about the exact combination it was handed.
    """
    localities = ProviderConfig.DeclaredLocality

    # -- payload kind -------------------------------------------------------
    if payload_kind not in PAYLOAD_KINDS:
        return _deny(
            ReasonCode.UNKNOWN_PAYLOAD_KIND,
            scope=application_scope, provider=provider, collection=collection,
            payload_kind=str(payload_kind or ""),
        )

    # -- scope --------------------------------------------------------------
    if application_scope is None or getattr(application_scope, "pk", None) is None:
        return _deny(ReasonCode.SCOPE_MISSING, payload_kind=payload_kind)
    if not application_scope.is_active:
        return _deny(
            ReasonCode.SCOPE_INACTIVE,
            scope=application_scope, provider=provider, collection=collection,
            payload_kind=payload_kind,
        )

    # -- provider -----------------------------------------------------------
    if provider is None or getattr(provider, "pk", None) is None:
        return _deny(
            ReasonCode.PROVIDER_MISSING,
            scope=application_scope, collection=collection, payload_kind=payload_kind,
        )
    if not provider.is_active:
        return _deny(
            ReasonCode.PROVIDER_INACTIVE,
            scope=application_scope, provider=provider, collection=collection,
            payload_kind=payload_kind, locality=provider.declared_locality,
        )

    locality = provider.declared_locality
    if locality == localities.UNKNOWN:
        # An unclassified provider is refused no matter how permissive every
        # other setting is. Core cannot verify the operator's declaration, but
        # it can insist that one was made.
        return _deny(
            ReasonCode.PROVIDER_LOCALITY_UNDECLARED,
            scope=application_scope, provider=provider, collection=collection,
            payload_kind=payload_kind, locality=locality,
        )

    # -- collection ---------------------------------------------------------
    if collection is None or getattr(collection, "pk", None) is None:
        return _deny(
            ReasonCode.COLLECTION_MISSING,
            scope=application_scope, provider=provider,
            payload_kind=payload_kind, locality=locality,
        )
    if not collection.is_active:
        return _deny(
            ReasonCode.COLLECTION_INACTIVE,
            scope=application_scope, provider=provider, collection=collection,
            payload_kind=payload_kind, locality=locality,
        )
    if collection.application_scope_id != application_scope.pk:
        # Evaluating a foreign collection against this scope's policy would let
        # one application's permissions decide another's data. Refused before
        # any egress question is even asked.
        return _deny(
            ReasonCode.COLLECTION_FOREIGN_SCOPE,
            scope=application_scope, provider=provider, collection=collection,
            payload_kind=payload_kind, locality=locality,
        )

    # -- grant --------------------------------------------------------------
    grant = ProviderGrant.objects.filter(
        application_scope_id=application_scope.pk, provider_id=provider.pk
    ).first()
    if grant is None:
        return _deny(
            ReasonCode.NO_PROVIDER_GRANT,
            scope=application_scope, provider=provider, collection=collection,
            payload_kind=payload_kind, locality=locality,
        )
    if not grant.allow_embeddings:
        return _deny(
            ReasonCode.GRANT_DOES_NOT_ALLOW_EMBEDDINGS,
            scope=application_scope, provider=provider, collection=collection,
            payload_kind=payload_kind, locality=locality,
        )

    # -- LOCAL: nothing crosses the declared boundary -----------------------
    if locality == localities.LOCAL:
        # The external egress flags are deliberately NOT consulted here. They
        # govern crossing the trust boundary, and by the operator's declaration
        # this does not cross it. A granted local provider must not be blocked
        # by a policy about externality.
        return EmbeddingAccessDecision(
            allowed=True,
            application_scope_id=application_scope.pk,
            provider_id=provider.pk,
            collection_id=collection.pk,
            payload_kind=payload_kind,
            declared_locality=locality,
            requires_external_egress=False,
            reason_code=ReasonCode.ALLOWED_LOCAL,
        )

    # -- EXTERNAL -----------------------------------------------------------
    # The collection's narrowing applies to BOTH payload kinds. Query text must
    # not be allowed to leave on behalf of a collection whose own corpus is
    # forbidden from leaving, which gives the per-collection invariant:
    #     external_query_allowed  ⊆  external_corpus_allowed
    if (
        collection.external_embedding_egress_policy
        == collection.ExternalEmbeddingEgressPolicy.DENY
    ):
        return _deny(
            ReasonCode.COLLECTION_DENIES_EXTERNAL,
            scope=application_scope, provider=provider, collection=collection,
            payload_kind=payload_kind, locality=locality, requires_external=True,
        )

    if not application_scope.allow_external_embedding_corpus_egress:
        # Checked for BOTH kinds: query egress can never exceed corpus egress,
        # and the database constraint makes the inverse state unrepresentable.
        return _deny(
            ReasonCode.SCOPE_DENIES_EXTERNAL_CORPUS,
            scope=application_scope, provider=provider, collection=collection,
            payload_kind=payload_kind, locality=locality, requires_external=True,
        )

    if payload_kind == PAYLOAD_QUERY and not (
        application_scope.allow_external_embedding_query_egress
    ):
        return _deny(
            ReasonCode.SCOPE_DENIES_EXTERNAL_QUERY,
            scope=application_scope, provider=provider, collection=collection,
            payload_kind=payload_kind, locality=locality, requires_external=True,
        )

    return EmbeddingAccessDecision(
        allowed=True,
        application_scope_id=application_scope.pk,
        provider_id=provider.pk,
        collection_id=collection.pk,
        payload_kind=payload_kind,
        declared_locality=locality,
        requires_external_egress=True,
        reason_code=ReasonCode.ALLOWED_EXTERNAL,
    )
