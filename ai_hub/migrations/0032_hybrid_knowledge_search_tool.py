"""S-28: register the `search_knowledge_hybrid` system tool.

Data only, split from the 0031 schema change (CI #43: one migration mixing
schema and data is not PostgreSQL-safe).

Deliberately NOT added to the `knowledge-discovery-retrieval` toolbox and
deliberately given no `game_tool_category`: the tool must reach an Agent only
through `resolve_agent_tools`' gated knowledge path, never through a toolbox
assignment and never through a GAME context.
"""
from django.db import migrations

TOOL_NAME = "search_knowledge_hybrid"
CALLABLE = "ai_hub.tools.knowledge.search_knowledge_hybrid"


def create_tool(apps, schema_editor):
    ToolDefinition = apps.get_model("ai_hub", "ToolDefinition")
    ToolDefinition.objects.update_or_create(
        name=TOOL_NAME,
        defaults={
            "label": "Search knowledge (hybrid)",
            "description": (
                "Search assigned active knowledge by meaning AND by words. Use it when "
                "a question may be phrased differently from the source text; use "
                "search_knowledge for exact terms and identifiers. Results come in "
                "relevance order with excerpts and citations; read_knowledge_chunk "
                "returns the full text."
            ),
            "tool_kind": "python_callable",
            "input_schema": {
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "collection_id": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
            },
            "output_schema": {"required": ["results", "total", "mode", "degraded", "coverage"]},
            "config": {
                "callable": CALLABLE,
                "read_only": True,
                "bind_agent_context": True,
                "limit": 5,
                "max_excerpt_chars": 700,
            },
            "risk_level": "low",
            "operation_mode": "read",
            "requires_approval": False,
            "is_system_tool": True,
            "is_active": True,
        },
    )


def remove_tool(apps, schema_editor):
    """Remove the tool - or, once it has execution history, only deactivate it.

    `ToolExecutionRun.tool` is PROTECT: audit rows must outlive a rollback, so
    a tool that was ever executed is kept, inactive, rather than deleted.
    """
    ToolDefinition = apps.get_model("ai_hub", "ToolDefinition")
    ToolExecutionRun = apps.get_model("ai_hub", "ToolExecutionRun")
    for tool in ToolDefinition.objects.filter(name=TOOL_NAME, config__callable=CALLABLE):
        if ToolExecutionRun.objects.filter(tool_id=tool.pk).exists():
            tool.is_active = False
            tool.save(update_fields=["is_active"])
        else:
            tool.delete()


class Migration(migrations.Migration):

    dependencies = [
        ("ai_hub", "0031_scope_retrieval_embedding_config"),
    ]

    operations = [
        migrations.RunPython(create_tool, remove_tool),
    ]
