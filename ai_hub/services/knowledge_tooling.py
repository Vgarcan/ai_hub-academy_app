"""Stable identifiers for the built-in read-only Knowledge tool adapters."""

BIND_AGENT_CONTEXT_CONFIG_KEY = "bind_agent_context"

KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES = {
    "list_knowledge_libraries": "ai_hub.tools.knowledge.list_knowledge_libraries",
    "browse_knowledge_index": "ai_hub.tools.knowledge.browse_knowledge_index",
    "search_knowledge": "ai_hub.tools.knowledge.search_knowledge",
    "read_knowledge_chunk": "ai_hub.tools.knowledge.read_knowledge_chunk",
    "read_document_section": "ai_hub.tools.knowledge.read_document_section",
    "cite_knowledge_source": "ai_hub.tools.knowledge.cite_knowledge_source",
    # S-28. Bound like the six above, but GATED: `tool_resolution` offers it
    # only when `hybrid_knowledge_search.hybrid_search_offered()` says so.
    "search_knowledge_hybrid": "ai_hub.tools.knowledge.search_knowledge_hybrid",
}

#: The gated tool's name, so callers never spell it twice.
HYBRID_KNOWLEDGE_SEARCH_TOOL_NAME = "search_knowledge_hybrid"

KNOWLEDGE_RETRIEVAL_TOOL_NAMES = tuple(KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES)


def is_bound_knowledge_tool(tool) -> bool:
    config = tool.config or {}
    expected_callable = KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES.get(tool.name)
    return bool(
        tool.is_system_tool
        and expected_callable
        and config.get("callable") == expected_callable
        and config.get(BIND_AGENT_CONTEXT_CONFIG_KEY) is True
    )
