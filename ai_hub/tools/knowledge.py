from django.core.exceptions import ValidationError

from ai_hub.models import AgentProfile
from ai_hub.services import knowledge_retrieval


def _agent_from_payload(payload: dict) -> AgentProfile:
    agent_id = payload.get("agent_id")
    agent_name = payload.get("agent_name")
    try:
        if agent_id:
            return AgentProfile.objects.get(pk=agent_id)
        if agent_name:
            return AgentProfile.objects.get(name=agent_name)
    except AgentProfile.DoesNotExist as exc:
        raise ValidationError("Agent not found for knowledge tool call.") from exc
    raise ValidationError("Knowledge tool call requires agent_id or agent_name.")


def list_knowledge_libraries(payload: dict, config: dict) -> dict:
    return knowledge_retrieval.list_knowledge_libraries(_agent_from_payload(payload))


def browse_knowledge_index(payload: dict, config: dict) -> dict:
    return knowledge_retrieval.browse_knowledge_index(
        _agent_from_payload(payload),
        collection_id=payload.get("collection_id"),
    )


def search_knowledge(payload: dict, config: dict) -> dict:
    return knowledge_retrieval.search_knowledge(
        _agent_from_payload(payload),
        query=payload.get("query", ""),
        collection_id=payload.get("collection_id"),
        limit=int(payload.get("limit") or config.get("limit") or 5),
    )


def read_knowledge_chunk(payload: dict, config: dict) -> dict:
    return knowledge_retrieval.read_knowledge_chunk(
        _agent_from_payload(payload),
        chunk_id=int(payload.get("chunk_id")),
    )


def read_document_section(payload: dict, config: dict) -> dict:
    return knowledge_retrieval.read_document_section(
        _agent_from_payload(payload),
        document_id=int(payload.get("document_id")),
        section_title=str(payload.get("section_title") or ""),
        chunk_index=payload.get("chunk_index"),
    )


def cite_knowledge_source(payload: dict, config: dict) -> dict:
    return knowledge_retrieval.cite_knowledge_source(
        _agent_from_payload(payload),
        chunk_id=int(payload.get("chunk_id")),
    )
