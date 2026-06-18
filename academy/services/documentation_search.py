"""
Documentation search: semantic (bge-m3 embeddings) with keyword fallback.

Semantic search is used when:
  - At least one DocumentationChunk has an embedding stored
  - The Ollama provider with bge-m3 is reachable

If either condition is not met, falls back to keyword scoring.
"""
from functools import reduce

from django.db.models import Q

from academy.models import DocumentationChunk


def search_documentation(query: str, limit: int = 5) -> list:
    """Return up to `limit` DocumentationChunk objects ranked by relevance."""
    if not query or not query.strip():
        return []

    results = _semantic_search(query, limit)
    if results:
        return results
    return _keyword_search(query, limit)


# ── semantic ─────────────────────────────────────────────

def _semantic_search(query: str, limit: int) -> list:
    from academy.services.embeddings import cosine_similarity, get_embedding

    # Only attempt if we have embedded chunks
    if not DocumentationChunk.objects.filter(is_active=True, embedding__isnull=False).exists():
        return []

    query_emb = get_embedding(query)
    if not query_emb:
        return []

    chunks = (
        DocumentationChunk.objects.filter(is_active=True, embedding__isnull=False)
        .select_related("page")
    )

    scored = [
        (cosine_similarity(query_emb, chunk.embedding), chunk)
        for chunk in chunks
    ]
    scored.sort(key=lambda x: -x[0])
    return [chunk for _, chunk in scored[:limit]]


# ── keyword fallback ──────────────────────────────────────

def _keyword_search(query: str, limit: int) -> list:
    words = [w.lower() for w in query.strip().split() if len(w) > 1]
    if not words:
        return []

    # Pre-filter at DB level to avoid loading all chunks into memory
    db_filter = reduce(
        lambda acc, w: acc | Q(search_text__icontains=w) | Q(heading__icontains=w),
        words,
        Q(),
    )
    chunks = (
        DocumentationChunk.objects.filter(is_active=True)
        .filter(db_filter)
        .select_related("page")
    )

    results = []
    for chunk in chunks:
        heading_lower = chunk.heading.lower()
        search_lower = chunk.search_text.lower()
        heading_hits = sum(1 for w in words if w in heading_lower)
        body_hits = sum(1 for w in words if w in search_lower)
        score = heading_hits * 3 + body_hits
        if score > 0:
            results.append((score, chunk.page.order, chunk.order, chunk))

    results.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [item[3] for item in results[:limit]]
