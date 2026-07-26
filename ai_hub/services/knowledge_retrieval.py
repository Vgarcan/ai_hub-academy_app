from functools import reduce

from django.core.exceptions import ValidationError
from django.db.models import Q
from django.db.models.functions import Length, Substr

from ai_hub.models import AgentProfile, KnowledgeDocument, KnowledgeDocumentChunk


MAX_COLLECTION_RESULTS = 100
MAX_COLLECTION_DESCRIPTION_CHARS = 500
MAX_BROWSE_DOCUMENT_RESULTS = 100
MAX_BROWSE_CHUNKS_PER_DOCUMENT = 100
MAX_SEARCH_RESULTS = 20
MAX_SEARCH_CANDIDATES = 1000
MAX_SEARCH_CANDIDATE_CONTENT_CHARS = 20000
MAX_SEARCH_QUERY_CHARS = 1000
MAX_SEARCH_QUERY_WORDS = 20
MAX_DOCUMENT_TAGS = 20
MAX_DOCUMENT_TAG_CHARS = 100


def _agent_collection_ids(agent: AgentProfile) -> set[int]:
    return set(agent.knowledge_collections.filter(is_active=True).values_list("id", flat=True))


def _accessible_chunks(agent: AgentProfile):
    collection_ids = _agent_collection_ids(agent)
    return (
        KnowledgeDocumentChunk.objects.filter(
            document__collection_id__in=collection_ids,
            document__collection__is_active=True,
            document__status=KnowledgeDocument.Status.ACTIVE,
        )
        .select_related("document", "document__collection")
        .order_by("document__collection__name", "document__title", "chunk_index")
    )


def _bounded_tags(tags) -> list[str]:
    if not isinstance(tags, list):
        return []
    return [
        str(tag)[:MAX_DOCUMENT_TAG_CHARS]
        for tag in tags[:MAX_DOCUMENT_TAGS]
    ]


def _bounded_integer(value, *, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be an integer.") from exc
    return min(max(parsed, minimum), maximum)


def list_knowledge_libraries(agent: AgentProfile, *, limit: int = 50) -> dict:
    collections = agent.knowledge_collections.filter(is_active=True).order_by("name")
    total = collections.count()
    bounded_limit = _bounded_integer(
        limit,
        name="limit",
        minimum=1,
        maximum=MAX_COLLECTION_RESULTS,
    )
    libraries = []
    for collection in collections[:bounded_limit]:
        active_documents = collection.documents.filter(status=KnowledgeDocument.Status.ACTIVE)
        libraries.append(
            {
                "collection_id": collection.pk,
                "name": collection.name,
                "description": str(collection.description or "")[:MAX_COLLECTION_DESCRIPTION_CHARS],
                "active_documents": active_documents.count(),
                "chunk_count": KnowledgeDocumentChunk.objects.filter(document__in=active_documents).count(),
            }
        )
    return {
        "libraries": libraries,
        "total": total,
        "returned": len(libraries),
        "has_more": len(libraries) < total,
    }


def browse_knowledge_index(
    agent: AgentProfile,
    *,
    collection_id=None,
    limit: int = 50,
    chunk_limit: int = 25,
) -> dict:
    collections = agent.knowledge_collections.filter(is_active=True).order_by("name")
    if collection_id is not None:
        collections = collections.filter(pk=collection_id)
    collection_count = collections.count()
    collections = list(collections[:MAX_COLLECTION_RESULTS])
    document_limit = _bounded_integer(
        limit,
        name="limit",
        minimum=1,
        maximum=MAX_BROWSE_DOCUMENT_RESULTS,
    )
    per_document_chunk_limit = _bounded_integer(
        chunk_limit,
        name="chunk_limit",
        minimum=1,
        maximum=MAX_BROWSE_CHUNKS_PER_DOCUMENT,
    )
    index = []
    returned_documents = 0
    total_documents = sum(
        collection.documents.filter(status=KnowledgeDocument.Status.ACTIVE).count()
        for collection in collections
    )
    for collection in collections:
        documents = []
        for document in collection.documents.filter(status=KnowledgeDocument.Status.ACTIVE).order_by("title"):
            if returned_documents >= document_limit:
                break
            chunk_queryset = document.chunks.order_by("chunk_index")
            total_chunks = chunk_queryset.count()
            chunks = [
                {
                    "chunk_id": chunk.pk,
                    "chunk_index": chunk.chunk_index,
                    "section_title": chunk.section_title,
                    "token_estimate": chunk.token_estimate,
                }
                for chunk in chunk_queryset[:per_document_chunk_limit]
            ]
            documents.append(
                {
                    "document_id": document.pk,
                    "title": document.title,
                    "language": document.language,
                    "tags": _bounded_tags(document.tags),
                    "chunks": chunks,
                    "total_chunks": total_chunks,
                    "returned_chunks": len(chunks),
                    "chunks_have_more": len(chunks) < total_chunks,
                }
            )
            returned_documents += 1
        index.append(
            {
                "collection_id": collection.pk,
                "name": collection.name,
                "description": str(collection.description or "")[:MAX_COLLECTION_DESCRIPTION_CHARS],
                "documents": documents,
            }
        )
        if returned_documents >= document_limit:
            break
    return {
        "collections": index,
        "total": collection_count,
        "returned_collections": len(index),
        "collections_have_more": len(index) < collection_count,
        "total_documents": total_documents,
        "returned_documents": returned_documents,
        "has_more": returned_documents < total_documents,
    }


def _query_words(query: str) -> list[str]:
    words = []
    seen = set()
    for word in str(query or "").split():
        normalized = word.strip().lower()
        if len(normalized) <= 1 or normalized in seen:
            continue
        words.append(normalized)
        seen.add(normalized)
        if len(words) >= MAX_SEARCH_QUERY_WORDS:
            break
    return words


def _score_chunk(
    chunk: KnowledgeDocumentChunk,
    words: list[str],
    *,
    content: str | None = None,
) -> int:
    title = chunk.document.title.lower()
    tags = " ".join(str(tag).lower() for tag in (chunk.document.tags or []))
    section = chunk.section_title.lower()
    searchable_content = (
        chunk.content if content is None else content
    ).lower()
    return (
        sum(1 for word in words if word in title) * 4
        + sum(1 for word in words if word in tags) * 3
        + sum(1 for word in words if word in section) * 3
        + sum(1 for word in words if word in searchable_content)
    )


def _snippet(content: str, words: list[str], max_chars: int = 500) -> str:
    if not content:
        return ""
    lowered = content.lower()
    first_hit = min((lowered.find(word) for word in words if word in lowered), default=-1)
    if first_hit == -1:
        return content[:max_chars]
    start = max(first_hit - 80, 0)
    return content[start:start + max_chars]


def search_knowledge(agent: AgentProfile, *, query: str, collection_id=None, limit: int = 5) -> dict:
    query = str(query or "").strip()[:MAX_SEARCH_QUERY_CHARS]
    if not query:
        raise ValidationError("search_knowledge requires a non-empty query.")
    words = _query_words(query)
    if not words:
        return {
            "query": query,
            "results": [],
            "total": 0,
            "candidates_scanned": 0,
            "candidate_limit": MAX_SEARCH_CANDIDATES,
            "candidates_truncated": False,
        }

    chunks = _accessible_chunks(agent)
    if collection_id is not None:
        try:
            requested_collection_id = int(collection_id)
        except (TypeError, ValueError) as exc:
            raise ValidationError("collection_id must be an integer.") from exc
        if requested_collection_id not in _agent_collection_ids(agent):
            raise ValidationError("Knowledge collection is not accessible to this agent.")
        chunks = chunks.filter(document__collection_id=requested_collection_id)

    db_filter = reduce(
        lambda acc, word: (
            acc
            | Q(content__icontains=word)
            | Q(section_title__icontains=word)
            | Q(document__title__icontains=word)
            | Q(document__tags__icontains=word)
        ),
        words,
        Q(),
    )

    candidate_chunks = list(
        chunks.filter(db_filter)
        .annotate(
            search_content=Substr(
                "content",
                1,
                MAX_SEARCH_CANDIDATE_CONTENT_CHARS,
            ),
            search_content_chars=Length("content"),
        )
        .defer("content", "metadata")[:MAX_SEARCH_CANDIDATES + 1]
    )
    candidates_truncated = len(candidate_chunks) > MAX_SEARCH_CANDIDATES
    candidate_chunks = candidate_chunks[:MAX_SEARCH_CANDIDATES]
    scored = []
    for chunk in candidate_chunks:
        score = _score_chunk(chunk, words, content=chunk.search_content)
        # The database filter may match beyond the bounded scoring window.
        # Keep that candidate at low relevance without loading the full body.
        if score == 0:
            score = 1
        if score > 0:
            scored.append((score, chunk.document.title, chunk.chunk_index, chunk))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    bounded_limit = _bounded_integer(
        limit,
        name="limit",
        minimum=0,
        maximum=MAX_SEARCH_RESULTS,
    )
    selected = [item for item in scored[:bounded_limit]]

    results = [
        {
            "chunk_id": chunk.pk,
            "document_id": chunk.document_id,
            "title": chunk.document.title,
            "collection": chunk.document.collection.name,
            "section_title": chunk.section_title,
            "chunk_index": chunk.chunk_index,
            "snippet": _snippet(chunk.search_content, words),
            "content_window_truncated": (
                chunk.search_content_chars
                > MAX_SEARCH_CANDIDATE_CONTENT_CHARS
            ),
            "score": score,
            "citation": _citation_for_chunk(chunk),
        }
        for score, _title, _index, chunk in selected
    ]
    return {
        "query": query,
        "results": results,
        "total": len(results),
        "candidates_scanned": len(candidate_chunks),
        "candidate_limit": MAX_SEARCH_CANDIDATES,
        "candidates_truncated": candidates_truncated,
    }


def read_knowledge_chunk(agent: AgentProfile, *, chunk_id: int) -> dict:
    try:
        chunk = _accessible_chunks(agent).get(pk=chunk_id)
    except KnowledgeDocumentChunk.DoesNotExist as exc:
        raise ValidationError("Knowledge chunk not found or not accessible to this agent.") from exc
    return {
        "chunk_id": chunk.pk,
        "document_id": chunk.document_id,
        "title": chunk.document.title,
        "collection": chunk.document.collection.name,
        "section_title": chunk.section_title,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "token_estimate": chunk.token_estimate,
        "metadata": chunk.metadata,
        "citation": cite_knowledge_source(agent, chunk_id=chunk.pk)["citation"],
    }


def read_document_section(agent: AgentProfile, *, document_id: int, section_title: str = "", chunk_index=None) -> dict:
    chunks = _accessible_chunks(agent).filter(document_id=document_id)
    if chunk_index is not None:
        chunks = chunks.filter(chunk_index=chunk_index)
    elif section_title:
        chunks = chunks.filter(section_title__iexact=section_title)
    else:
        raise ValidationError("read_document_section requires section_title or chunk_index.")
    chunk = chunks.first()
    if chunk is None:
        raise ValidationError("Document section not found or not accessible to this agent.")
    return read_knowledge_chunk(agent, chunk_id=chunk.pk)


def _citation_for_chunk(chunk: KnowledgeDocumentChunk) -> dict:
    return {
        "collection": chunk.document.collection.name,
        "document_id": chunk.document_id,
        "document_title": chunk.document.title,
        "chunk_id": chunk.pk,
        "section_title": chunk.section_title,
        "chunk_index": chunk.chunk_index,
        "language": chunk.document.language,
        "tags": _bounded_tags(chunk.document.tags),
    }


def cite_knowledge_source(agent: AgentProfile, *, chunk_id: int) -> dict:
    try:
        chunk = _accessible_chunks(agent).get(pk=chunk_id)
    except KnowledgeDocumentChunk.DoesNotExist as exc:
        raise ValidationError("Knowledge chunk not found or not accessible to this agent.") from exc
    return {"citation": _citation_for_chunk(chunk)}
