"""
Python Callable tool: search_docs

Searches DocumentationChunk records for content relevant to the agent's goal.
The query is taken (in priority order) from:
  1. previous_response.decision.search_query  — agent requested a refined search
  2. payload.search_query                      — explicit override in context
  3. payload.goal_text / payload.goal          — the original user question

Returns a dict that the GAME agent receives as `tool_results.search_documentation`.
"""
from academy.services.documentation_search import search_documentation


def search_docs(payload: dict, config: dict) -> dict:
    prev = payload.get("previous_response") or {}
    prev_decision = (prev.get("decision") or {}) if isinstance(prev, dict) else {}

    query = (
        prev_decision.get("search_query")
        or payload.get("search_query")
        or payload.get("goal_text")
        or payload.get("goal")
        or ""
    )

    limit = int(config.get("limit", 6))
    chunks = search_documentation(query, limit=limit)

    results = [
        {
            "page": chunk.page.title,
            "section": chunk.heading,
            "content": chunk.body_markdown[:900],
            "relevance_rank": i + 1,
        }
        for i, chunk in enumerate(chunks)
    ]

    return {
        "query": query,
        "results": results,
        "total": len(results),
        "message": (
            f"Found {len(results)} relevant section(s) for: '{query}'"
            if results
            else f"No documentation found for: '{query}'"
        ),
    }
