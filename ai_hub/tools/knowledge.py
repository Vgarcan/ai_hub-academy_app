import json

from django.core.exceptions import ValidationError

from ai_hub.models import AgentProfile
from ai_hub.services import knowledge_retrieval


def _bounded_int(value, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _bounded_content(result: dict, config: dict) -> dict:
    max_chars = _bounded_int(
        config.get("max_content_chars"),
        default=8000,
        minimum=256,
        maximum=20000,
    )
    content = str(result.get("content") or "")
    if len(content) <= max_chars:
        return result
    return {
        **result,
        "content": content[:max_chars],
        "content_chars": len(content),
        "content_truncated": True,
    }


def _bounded_metadata(result: dict, config: dict) -> dict:
    metadata = result.get("metadata")
    if metadata in (None, {}, []):
        return result
    max_chars = _bounded_int(
        config.get("max_metadata_chars"),
        default=2000,
        minimum=256,
        maximum=10000,
    )
    try:
        serialized = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        serialized = str(metadata)
    if len(serialized) <= max_chars:
        return result
    return {
        **result,
        "metadata": {
            "truncated": True,
            "json_preview": serialized[:max_chars],
        },
        "metadata_chars": len(serialized),
        "metadata_truncated": True,
    }


def _bounded_read_result(result: dict, config: dict) -> dict:
    return _bounded_metadata(_bounded_content(result, config), config)


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
    return knowledge_retrieval.list_knowledge_libraries(
        _agent_from_payload(payload),
        limit=_bounded_int(
            payload.get("limit") or config.get("limit"),
            default=50,
            minimum=1,
            maximum=100,
        ),
    )


def browse_knowledge_index(payload: dict, config: dict) -> dict:
    return knowledge_retrieval.browse_knowledge_index(
        _agent_from_payload(payload),
        collection_id=payload.get("collection_id"),
        limit=_bounded_int(
            payload.get("limit") or config.get("limit"),
            default=50,
            minimum=1,
            maximum=100,
        ),
        chunk_limit=_bounded_int(
            payload.get("chunk_limit") or config.get("chunk_limit"),
            default=25,
            minimum=1,
            maximum=100,
        ),
    )


def search_knowledge(payload: dict, config: dict) -> dict:
    return knowledge_retrieval.search_knowledge(
        _agent_from_payload(payload),
        query=payload.get("query", ""),
        collection_id=payload.get("collection_id"),
        limit=_bounded_int(
            payload.get("limit") or config.get("limit"),
            default=5,
            minimum=1,
            maximum=20,
        ),
    )


def read_knowledge_chunk(payload: dict, config: dict) -> dict:
    return _bounded_read_result(
        knowledge_retrieval.read_knowledge_chunk(
            _agent_from_payload(payload),
            chunk_id=int(payload.get("chunk_id")),
        ),
        config,
    )


def read_document_section(payload: dict, config: dict) -> dict:
    return _bounded_read_result(
        knowledge_retrieval.read_document_section(
            _agent_from_payload(payload),
            document_id=int(payload.get("document_id")),
            section_title=str(payload.get("section_title") or ""),
            chunk_index=payload.get("chunk_index"),
        ),
        config,
    )


def cite_knowledge_source(payload: dict, config: dict) -> dict:
    return knowledge_retrieval.cite_knowledge_source(
        _agent_from_payload(payload),
        chunk_id=int(payload.get("chunk_id")),
    )
