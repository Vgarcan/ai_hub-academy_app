"""Agent-facing hybrid Knowledge search (S-28).

The first route by which an Agent reaches semantic retrieval. It composes what
already exists and adds exactly three things:

    when is it offered      a gate: settings flag + scope configuration + not GAME
    what the Agent reads    hydration of the audited result, under the SAME scope
    what the Agent is told  mode, degraded, coverage - nothing about configuration

**One authorization answer for the whole operation.** The scope is resolved
once here and handed to `_audited_hybrid_search_with_scope`, and hydration reads
through `authorized_chunks()` of that same scope. Resolving a second time to
hydrate would let one search straddle two authorization answers (S-22).

**A chunk that changed after it was ranked is never served.** The audit
recorded each hit's `k1` at the composition boundary; hydration recomputes it
and, on mismatch, withholds the content and says so. Dropping it would make the
list look complete; serving it would hand the model text that was not ranked.

**No score reaches the model.** `fusion_score` is an ordering value, not
relevance; order is the signal.

**Configuration never reaches the model.** Provider, model, `e1`, failure kinds,
S-17 reason codes and the run id stay in the audit (ADR-N5 / S-17).

`search_knowledge` is untouched and stays LEXICAL.
"""

from django.conf import settings
from django.core.exceptions import ValidationError

from ai_hub.models import ExecutionSession, RetrievalHit, RetrievalOutcome
from ai_hub.services.chunk_embedding_identity import chunk_embedding_fingerprint
from ai_hub.services.hybrid_retrieval import (
    HybridRetrievalError,
    validate_hybrid_request,
)
from ai_hub.services.knowledge_authorization import (
    authorized_chunks,
    resolve_effective_knowledge_scope,
)
from ai_hub.services.knowledge_retrieval import _citation_for_chunk
from ai_hub.services.retrieval_audit import (
    AuditedRetrievalRefused,
    _audited_hybrid_search_with_scope,
)


#: Uniform refusals. Neither says WHY - that would describe configuration.
NOT_AVAILABLE_MESSAGE = "Hybrid knowledge search is not available for this agent."
TEMPORARILY_UNAVAILABLE_MESSAGE = "Knowledge search is temporarily unavailable."
#: Path 1's exact wording (`knowledge_retrieval.search_knowledge`), so the two
#: tools cannot be told apart by how they refuse a collection (ADR-N5).
COLLECTION_NOT_ACCESSIBLE_MESSAGE = "Knowledge collection is not accessible to this agent."

CHANGED_SINCE_SEARCH = "changed_since_search"

DEFAULT_LIMIT = 5
MAX_LIMIT = 10
DEFAULT_MAX_EXCERPT_CHARS = 700


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def hybrid_search_configuration(agent):
    """The embedding configuration this Agent's searches use, or None.

    None whenever the capability is off for this Agent: the settings flag is
    off, the Agent has no scope, the scope names no retrieval configuration, or
    that configuration or its provider is inactive. S-17 permission is NOT
    decided here - the search itself asks, per call, and a denial degrades the
    semantic branch rather than hiding the tool.
    """
    if not getattr(settings, "AI_HUB_HYBRID_KNOWLEDGE_SEARCH_ENABLED", False):
        return None
    scope = getattr(agent, "application_scope", None)
    if scope is None or not scope.is_active:
        return None
    config = scope.retrieval_embedding_model_config
    if config is None or not config.is_active or not config.provider.is_active:
        return None
    return config


def is_game_context(*, workspace=None, execution_context=None) -> bool:
    """GAME reaches Knowledge through its own path until convergence (ADR-N1)."""
    if workspace is not None:
        return True
    session = (execution_context or {}).get("session")
    return (
        session is not None
        and getattr(session, "runtime_kind", None) == ExecutionSession.RuntimeKind.GAME
    )


def hybrid_search_offered(agent, *, workspace=None, execution_context=None) -> bool:
    """Whether `search_knowledge_hybrid` may appear in this Agent's manifest."""
    if is_game_context(workspace=workspace, execution_context=execution_context):
        return False
    return hybrid_search_configuration(agent) is not None


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------

def _excerpt(content: str, max_chars: int) -> tuple:
    text = str(content or "")
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _matched_by(match) -> list:
    sources = []
    if match.lexical_rank is not None:
        sources.append("lexical")
    if match.semantic_rank is not None:
        sources.append("semantic")
    return sources


def search_knowledge_hybrid(
    agent,
    *,
    query,
    collection_id=None,
    limit=DEFAULT_LIMIT,
    max_excerpt_chars=DEFAULT_MAX_EXCERPT_CHARS,
) -> dict:
    """One audited hybrid search for `agent`, hydrated for a model to read.

    Execution re-checks the gate: a stale manifest or a direct call must not
    bypass resolution. Returns plain JSON-serializable data.
    """
    config = hybrid_search_configuration(agent)
    if config is None:
        raise ValidationError(NOT_AVAILABLE_MESSAGE)

    try:
        validate_hybrid_request(query=query, limit=limit)
    except HybridRetrievalError as exc:
        # Caller-only validation: the message names the argument, never the
        # corpus, and nothing has been recorded.
        raise ValidationError(str(exc)) from exc
    if limit < 1 or limit > MAX_LIMIT:
        raise ValidationError(f"limit must be an integer between 1 and {MAX_LIMIT}.")

    # -- ONE authorization answer for search AND hydration ------------------
    scope = resolve_effective_knowledge_scope(agent, workspace=None)

    if collection_id is not None:
        try:
            requested = int(collection_id)
        except (TypeError, ValueError) as exc:
            raise ValidationError("collection_id must be an integer.") from exc
        if not scope.allows(requested):
            raise ValidationError(COLLECTION_NOT_ACCESSIBLE_MESSAGE)
        collection_id = requested

    try:
        audited = _audited_hybrid_search_with_scope(
            scope,
            query=query,
            embedding_model_config=config,
            collection_id=collection_id,
            limit=limit,
        )
    except AuditedRetrievalRefused as exc:
        # Recorded as REFUSED with its bounded category; the model is told only
        # that search is unavailable.
        raise ValidationError(TEMPORARILY_UNAVAILABLE_MESSAGE) from exc

    result = audited.retrieval
    run_id = audited.retrieval_run_id

    # -- hydration, under the same scope ------------------------------------
    match_ids = [match.chunk_id for match in result.matches]
    chunks = {
        chunk.pk: chunk
        for chunk in authorized_chunks(scope).filter(pk__in=match_ids)
    }
    recorded_k1 = (
        dict(
            RetrievalHit.objects.filter(retrieval_run_id=run_id).values_list(
                "chunk_id_snapshot", "k1"
            )
        )
        if run_id is not None
        else {}
    )

    results = []
    for match in result.matches:
        chunk = chunks.get(match.chunk_id)
        if chunk is None or chunk_embedding_fingerprint(chunk) != recorded_k1.get(match.chunk_id):
            # No longer authorized, deleted, or edited since it was ranked. The
            # entry stays, so the list never looks complete when it is not.
            results.append({"chunk_id": match.chunk_id, "unavailable": CHANGED_SINCE_SEARCH})
            continue
        excerpt, truncated = _excerpt(chunk.content, max_excerpt_chars)
        results.append({
            "chunk_id": chunk.pk,
            "document_id": chunk.document_id,
            "title": chunk.document.title,
            "collection": chunk.document.collection.name,
            "section_title": chunk.section_title,
            "chunk_index": chunk.chunk_index,
            "excerpt": excerpt,
            "excerpt_truncated": truncated,
            "matched_by": _matched_by(match),
            "citation": _citation_for_chunk(chunk),
        })

    # -- coverage: how much of what was searchable could match semantically --
    indexed = 0
    if run_id is not None:
        indexed = (
            RetrievalOutcome.objects.filter(retrieval_run_id=run_id)
            .values_list("semantic_candidate_count", flat=True)
            .first()
            or 0
        )
    searchable = (
        authorized_chunks(scope)
        .filter(document__collection_id__in=result.collection_ids)
        .count()
        if result.collection_ids
        else 0
    )

    return {
        "results": results,
        "total": len(results),
        "mode": result.mode,
        "degraded": result.degraded,
        "coverage": {"indexed_chunks": indexed, "searchable_chunks": searchable},
    }
