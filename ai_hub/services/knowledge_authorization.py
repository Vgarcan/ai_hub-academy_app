"""The single Core boundary that decides which Knowledge an Agent may reach.

S-14 established OWNERSHIP: every `KnowledgeCollection`, `AgentProfile` and
`GameWorkspace` belongs to exactly one `ApplicationScope`. S-15 turns that
ownership into an EFFECTIVE SECURITY BOUNDARY.

The contract:

    effective Knowledge authorization
        =  ApplicationScope ownership
        ∩  Agent Knowledge assignment
        ∩  execution / workspace coherence
        ∩  request narrowing

**`AgentProfile.knowledge_collections` is an ASSIGNMENT mechanism, not a
security boundary.** A cross-scope row may physically exist - through raw ORM,
a fixture, an Admin mistake, or a pre-S-14 database - and it must grant exactly
zero authorization. Nothing here trusts the Admin, model validation, or an
operator's discipline: the scope predicate is part of the authorization query
itself, so an unauthorized collection can never become a candidate.

**Fail closed, always.** A missing agent, an inactive agent, a missing scope, an
inactive scope, a foreign workspace or an inactive workspace all resolve to an
empty authorization - never to "unrestricted". This is the opposite of the
legacy GAME allow-list rule (`no rows -> allow all`), which is deliberately left
alone as GAME execution compatibility and is explicitly NOT an ApplicationScope
rule.

**Do not call `require_single_active_scope()` from here.** That S-14 helper is a
creation-time compatibility bridge for surfaces without a scope selector. Using
it for runtime authorization would silently attach a principal to whatever scope
happens to be the only active one - a fail-open path wearing a helper's name.

STRONG FUTURE INVARIANT, recorded here because this is where a future retriever
will look:

    No lexical, semantic or hybrid retriever may generate candidates outside the
    EffectiveKnowledgeScope.

For an ANN backend that later means an ApplicationScope structural
partition/namespace PLUS authorized-collection narrowing - never a global
nearest-neighbour search followed by a filter. Retrieving globally and filtering
afterwards returns the same rows but lets unauthorized vectors consume finite
candidate slots, silently displacing authorized results. S-15 implements the
authorization policy only; no semantic code exists yet.
"""

from dataclasses import dataclass

from ai_hub.models import KnowledgeCollection, KnowledgeDocument, KnowledgeDocumentChunk


@dataclass(frozen=True)
class EffectiveKnowledgeScope:
    """The resolved authorization result for ONE retrieval operation.

    Ephemeral by design: never persisted, never a model, never cached across
    requests. It is computed once per retrieval operation and then used to
    constrain a single queryset, so authorization cannot drift between the
    decision and the query.

    `collection_ids` is the complete set of collections this principal may
    reach. An EMPTY set is a valid, meaningful result: it means "nothing", and
    every consumer must treat it as such rather than as "unfiltered".
    """

    application_scope_id: int | None
    agent_id: int | None
    workspace_id: int | None
    collection_ids: frozenset[int]

    @property
    def is_empty(self) -> bool:
        return not self.collection_ids

    def allows(self, collection_id) -> bool:
        """Narrow-only membership test for an explicitly requested collection."""
        try:
            return int(collection_id) in self.collection_ids
        except (TypeError, ValueError):
            return False


#: The single fail-closed result. Every refusal returns this, so no caller can
#: accidentally distinguish "inactive scope" from "foreign workspace" from
#: "unknown agent" by inspecting the shape of what it got back (ADR-N5).
DENY_ALL = EffectiveKnowledgeScope(
    application_scope_id=None,
    agent_id=None,
    workspace_id=None,
    collection_ids=frozenset(),
)


def resolve_effective_knowledge_scope(agent, *, workspace=None) -> EffectiveKnowledgeScope:
    """Resolve what `agent` may read, optionally restricted by `workspace`.

    Returns `DENY_ALL` rather than raising, for two reasons. Refusing by value
    keeps every caller on one code path, and it keeps ADR-N5 intact: a caller
    cannot turn "why was this denied" into an oracle, because there is only one
    denial and it carries no reason.
    """
    # -- the agent must be a real, active, scoped principal -------------------
    agent_id = getattr(agent, "pk", None)
    if agent_id is None:
        return DENY_ALL
    if not getattr(agent, "is_active", False):
        return DENY_ALL

    application_scope_id = getattr(agent, "application_scope_id", None)
    if application_scope_id is None:
        return DENY_ALL

    application_scope = getattr(agent, "application_scope", None)
    if application_scope is None or not application_scope.is_active:
        # An inactive scope authorizes nothing. There is deliberately no
        # fallback to another scope: a deactivated application is off, not
        # relocated.
        return DENY_ALL

    # -- the workspace, when supplied, must be coherent with that scope -------
    workspace_id = None
    if workspace is not None:
        workspace_id = getattr(workspace, "pk", None)
        if workspace_id is None:
            return DENY_ALL
        if not getattr(workspace, "is_active", False):
            return DENY_ALL
        if getattr(workspace, "application_scope_id", None) != application_scope_id:
            # A workspace never defines the root scope; it may only restrict
            # execution inside one. A foreign workspace is a refusal, never a
            # scope switch.
            return DENY_ALL
        # S-15 deliberately adds NO workspace-level Knowledge allow-list.
        # ADR-N3 did not authorize one. Workspace restriction here means
        # coherence only; a future allow-list intersects at this same point.

    # -- the effective collection set, resolved in ONE scoped query ----------
    # The scope predicate is part of THIS query. It is not applied afterwards,
    # so a cross-scope assignment row cannot produce a candidate at any stage.
    collection_ids = frozenset(
        KnowledgeCollection.objects.filter(
            agents__id=agent_id,
            application_scope_id=application_scope_id,
            application_scope__is_active=True,
            is_active=True,
        ).values_list("id", flat=True)
    )

    return EffectiveKnowledgeScope(
        application_scope_id=application_scope_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        collection_ids=collection_ids,
    )


class PipelineScopeError(ValueError):
    """A pipeline whose participating agents do not share one ApplicationScope."""


def pipeline_scope_ids(pipeline) -> set:
    """Every ApplicationScope id reachable through this pipeline's agents.

    Entry agent, every step agent, and every configured fallback agent. Fallback
    agents are part of the security boundary: an error path that switches to an
    agent in another application is still a bridge between applications.
    """
    scope_ids = set()
    entry_scope_id = getattr(pipeline, "entry_agent", None)
    if entry_scope_id is not None:
        scope_ids.add(entry_scope_id.application_scope_id)
    for step in pipeline.steps.all():
        if step.agent_id:
            scope_ids.add(step.agent.application_scope_id)
        if step.fallback_agent_id:
            scope_ids.add(step.fallback_agent.application_scope_id)
    return scope_ids


def require_coherent_pipeline_scope(pipeline):
    """Fail closed unless every participating agent shares ONE scope.

    Called at configuration time AND again at runtime. Runtime is the
    authoritative check, because raw ORM bypasses `full_clean()`, and because a
    pipeline can be edited after a session referencing it was created.

    This exists even though every individual retrieval call is already scoped:
    a pipeline that mixes scopes transfers Scope A's retrieved content into
    Scope B's execution through the step output mapping, so per-call correctness
    is not sufficient.

    Deliberately NOT solved by adding an ApplicationScope FK to
    PipelineDefinition. Scope is DERIVED from the participating agents, which
    keeps PipelineDefinition a generic composed definition and avoids a second,
    contradictable source of truth.
    """
    scope_ids = pipeline_scope_ids(pipeline)
    if not scope_ids:
        return None
    if None in scope_ids:
        raise PipelineScopeError(
            "Every agent in a pipeline must belong to an application scope."
        )
    if len(scope_ids) > 1:
        # The count is safe to state; the scope identities are not.
        raise PipelineScopeError(
            "All agents in a pipeline must belong to the same application "
            f"scope; this pipeline spans {len(scope_ids)}."
        )
    return next(iter(scope_ids))


def authorized_collections(scope: EffectiveKnowledgeScope):
    """Collections the scope authorizes, as a queryset. Empty scope -> none()."""
    if scope.is_empty:
        return KnowledgeCollection.objects.none()
    return KnowledgeCollection.objects.filter(
        id__in=scope.collection_ids,
        application_scope_id=scope.application_scope_id,
        is_active=True,
    ).order_by("name")


def authorized_documents(scope: EffectiveKnowledgeScope):
    """ACTIVE documents inside the authorized collections."""
    if scope.is_empty:
        return KnowledgeDocument.objects.none()
    return KnowledgeDocument.objects.filter(
        collection_id__in=scope.collection_ids,
        collection__application_scope_id=scope.application_scope_id,
        collection__is_active=True,
        status=KnowledgeDocument.Status.ACTIVE,
    )


def authorized_chunks(scope: EffectiveKnowledgeScope):
    """The ONLY queryset any retriever may generate chunk candidates from.

    Every predicate below is part of the query, so unauthorized chunks are
    never selected, never scored, and never counted. A future semantic or
    hybrid retriever must start from this same restriction - expressed against
    its own backend - rather than filtering after nearest-neighbour selection.
    """
    if scope.is_empty:
        return KnowledgeDocumentChunk.objects.none()
    return (
        KnowledgeDocumentChunk.objects.filter(
            document__collection_id__in=scope.collection_ids,
            document__collection__application_scope_id=scope.application_scope_id,
            document__collection__is_active=True,
            document__status=KnowledgeDocument.Status.ACTIVE,
        )
        .select_related("document", "document__collection")
        .order_by("document__collection__name", "document__title", "chunk_index")
    )
