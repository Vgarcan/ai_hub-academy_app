import json

from django.core.exceptions import ValidationError

from ai_hub.models import AgentProfile
from ai_hub.services.contracts import validate_payload
from ai_hub.services.litellm_client import completion_call
from ai_hub.services.provider_registry import resolve_model_config
from ai_hub.services.tools_runtime import execute_tools


def get_mapped_value(source: dict, source_key: str):
    current = source
    for part in str(source_key).split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def apply_mapping(source: dict, mapping: dict) -> dict:
    if not mapping:
        return dict(source)
    result = {}
    for target_key, source_key in mapping.items():
        result[target_key] = get_mapped_value(source, source_key)
    return result


def read_document_text(document) -> str:
    if document.curated_text:
        return document.curated_text
    if not document.source_file:
        return ""
    try:
        with document.source_file.open("rb") as source:
            return source.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def build_agent_knowledge_context(agent: AgentProfile) -> dict:
    max_chars = max(agent.knowledge_max_chars or 0, 0)
    documents = []
    remaining = max_chars
    collections = agent.knowledge_collections.filter(is_active=True).prefetch_related("documents")

    for collection in collections:
        active_documents = collection.documents.filter(status="active").order_by("title")
        for document in active_documents:
            text = read_document_text(document).strip()
            if not text:
                continue
            if max_chars and remaining <= 0:
                break
            selected_text = text[:remaining] if max_chars else text
            if max_chars:
                remaining -= len(selected_text)
            documents.append(
                {
                    "collection": collection.name,
                    "title": document.title,
                    "language": document.language,
                    "tags": document.tags,
                    "content": selected_text,
                }
            )

    text_blocks = [
        f"[{doc['collection']} / {doc['title']}]\n{doc['content']}"
        for doc in documents
    ]
    return {
        "collections": [collection.name for collection in collections],
        "documents": documents,
        "text": "\n\n".join(text_blocks),
        "truncated": bool(max_chars and remaining <= 0),
        "max_chars": max_chars,
    }


def prepare_agent_payload(agent: AgentProfile, context: dict, mapping: dict) -> dict:
    payload = apply_mapping(context, mapping or {})
    payload["knowledge_context"] = build_agent_knowledge_context(agent)
    return payload


def execute_agent(agent: AgentProfile, payload: dict) -> dict:
    validate_payload(payload, agent.input_contract or {}, f"Agent '{agent.name}' input")
    tools_data = execute_tools(agent.tools.filter(is_active=True), payload)
    model_cfg = resolve_model_config(agent.model_config)

    # Include tool results in the same LLM call so the agent can reason about them immediately.
    # Without this, tool output only appears in previous_response on the *next* iteration.
    user_content: dict = {"context": payload}
    if tools_data:
        user_content["tool_results"] = tools_data
    user_message = json.dumps(user_content, default=str)

    llm_result = completion_call(
        model=model_cfg["model"],
        messages=[
            {"role": "system", "content": agent.system_prompt or ""},
            {"role": "user", "content": user_message},
        ],
        api_key=model_cfg["api_key"],
        base_url=model_cfg["base_url"],
        timeout=model_cfg["timeout"],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
    )
    output_payload = {
        "agent": agent.name,
        "tools": tools_data,
        "llm": llm_result,
    }
    validate_payload(output_payload, agent.output_contract or {}, f"Agent '{agent.name}' output")
    return output_payload
