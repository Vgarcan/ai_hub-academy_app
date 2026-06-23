from functools import reduce

from django.core.exceptions import ValidationError
from django.db.models import Q

from ai_hub.models import AgentProfile, KnowledgeDocument, KnowledgeDocumentChunk


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


def list_knowledge_libraries(agent: AgentProfile) -> dict:
    collections = agent.knowledge_collections.filter(is_active=True).order_by("name")
    libraries = []
    for collection in collections:
        active_documents = collection.documents.filter(status=KnowledgeDocument.Status.ACTIVE)
        libraries.append(
            {
                "collection_id": collection.pk,
                "name": collection.name,
                "description": collection.description,
                "active_documents": active_documents.count(),
                "chunk_count": KnowledgeDocumentChunk.objects.filter(document__in=active_documents).count(),
            }
        )
    return {"libraries": libraries, "total": len(libraries)}


def browse_knowledge_index(agent: AgentProfile, *, collection_id=None) -> dict:
    collections = agent.knowledge_collections.filter(is_active=True).order_by("name")
    if collection_id is not None:
        collections = collections.filter(pk=collection_id)
    index = []
    for collection in collections:
        documents = []
        for document in collection.documents.filter(status=KnowledgeDocument.Status.ACTIVE).order_by("title"):
            chunks = [
                {
                    "chunk_id": chunk.pk,
                    "chunk_index": chunk.chunk_index,
                    "section_title": chunk.section_title,
                    "token_estimate": chunk.token_estimate,
                }
                for chunk in document.chunks.order_by("chunk_index")
            ]
            documents.append(
                {
                    "document_id": document.pk,
                    "title": document.title,
                    "language": document.language,
                    "tags": document.tags,
                    "chunks": chunks,
                }
            )
        index.append(
            {
                "collection_id": collection.pk,
                "name": collection.name,
                "description": collection.description,
                "documents": documents,
            }
        )
    return {"collections": index, "total": len(index)}


def _query_words(query: str) -> list[str]:
    return [word.lower() for word in str(query or "").split() if len(word.strip()) > 1]


def _score_chunk(chunk: KnowledgeDocumentChunk, words: list[str]) -> int:
    title = chunk.document.title.lower()
    tags = " ".join(str(tag).lower() for tag in (chunk.document.tags or []))
    section = chunk.section_title.lower()
    content = chunk.content.lower()
    return (
        sum(1 for word in words if word in title) * 4
        + sum(1 for word in words if word in tags) * 3
        + sum(1 for word in words if word in section) * 3
        + sum(1 for word in words if word in content)
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
    query = str(query or "").strip()
    if not query:
        raise ValidationError("search_knowledge requires a non-empty query.")
    words = _query_words(query)
    if not words:
        return {"query": query, "results": [], "total": 0}

    chunks = _accessible_chunks(agent)
    if collection_id is not None:
        if int(collection_id) not in _agent_collection_ids(agent):
            raise ValidationError("Knowledge collection is not accessible to this agent.")
        chunks = chunks.filter(document__collection_id=collection_id)

    db_filter = reduce(
        lambda acc, word: (
            acc
            | Q(content__icontains=word)
            | Q(section_title__icontains=word)
            | Q(document__title__icontains=word)
        ),
        words,
        Q(),
    )

    scored = []
    for chunk in chunks.filter(db_filter):
        score = _score_chunk(chunk, words)
        if score > 0:
            scored.append((score, chunk.document.title, chunk.chunk_index, chunk))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = [item for item in scored[: max(int(limit), 0)]]

    results = [
        {
            "chunk_id": chunk.pk,
            "document_id": chunk.document_id,
            "title": chunk.document.title,
            "collection": chunk.document.collection.name,
            "section_title": chunk.section_title,
            "chunk_index": chunk.chunk_index,
            "snippet": _snippet(chunk.content, words),
            "score": score,
            "citation": cite_knowledge_source(agent, chunk_id=chunk.pk)["citation"],
        }
        for score, _title, _index, chunk in selected
    ]
    return {"query": query, "results": results, "total": len(results)}


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


def cite_knowledge_source(agent: AgentProfile, *, chunk_id: int) -> dict:
    try:
        chunk = _accessible_chunks(agent).get(pk=chunk_id)
    except KnowledgeDocumentChunk.DoesNotExist as exc:
        raise ValidationError("Knowledge chunk not found or not accessible to this agent.") from exc
    citation = {
        "collection": chunk.document.collection.name,
        "document_id": chunk.document_id,
        "document_title": chunk.document.title,
        "chunk_id": chunk.pk,
        "section_title": chunk.section_title,
        "chunk_index": chunk.chunk_index,
        "language": chunk.document.language,
        "tags": chunk.document.tags,
    }
    return {"citation": citation}
