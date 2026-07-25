from ai_hub.models import KnowledgeDocument, KnowledgeDocumentChunk


def ensure_initial_knowledge_chunk(
    document: KnowledgeDocument,
) -> KnowledgeDocumentChunk | None:
    """Create one retrievable chunk for curated text when no chunks exist.

    This is intentionally a minimal ingestion fallback, not an automatic
    semantic chunking pipeline. Explicitly curated chunks always win.
    """
    if document.chunks.exists():
        return document.chunks.order_by("chunk_index").first()
    content = str(document.curated_text or "").strip()
    if not content:
        return None
    return KnowledgeDocumentChunk.objects.create(
        document=document,
        chunk_index=1,
        section_title=document.title,
        content=content,
        token_estimate=max(len(content.split()), 1),
        metadata={"ingestion": "initial_curated_text"},
    )
