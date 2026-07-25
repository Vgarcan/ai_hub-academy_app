from django.db import migrations


TOOL_SPECS = (
    {
        "name": "list_knowledge_libraries",
        "label": "List knowledge libraries",
        "description": "List knowledge libraries assigned to the current agent.",
        "callable": "ai_hub.tools.knowledge.list_knowledge_libraries",
        "input_schema": {"properties": {"limit": {"type": "integer"}}},
        "output_schema": {"required": ["libraries", "total"]},
        "extra_config": {"limit": 50},
    },
    {
        "name": "browse_knowledge_index",
        "label": "Browse knowledge index",
        "description": "Browse assigned collections, documents and chunk metadata.",
        "callable": "ai_hub.tools.knowledge.browse_knowledge_index",
        "input_schema": {
            "properties": {
                "collection_id": {"type": "integer"},
                "limit": {"type": "integer"},
                "chunk_limit": {"type": "integer"},
            }
        },
        "output_schema": {"required": ["collections", "total"]},
        "extra_config": {"limit": 50, "chunk_limit": 25},
    },
    {
        "name": "search_knowledge",
        "label": "Search knowledge",
        "description": "Search assigned active knowledge chunks by lexical relevance.",
        "callable": "ai_hub.tools.knowledge.search_knowledge",
        "input_schema": {
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "collection_id": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        },
        "output_schema": {"required": ["query", "results", "total"]},
        "extra_config": {"limit": 5},
    },
    {
        "name": "read_knowledge_chunk",
        "label": "Read knowledge chunk",
        "description": "Read one authorized knowledge chunk.",
        "callable": "ai_hub.tools.knowledge.read_knowledge_chunk",
        "input_schema": {
            "required": ["chunk_id"],
            "properties": {"chunk_id": {"type": "integer"}},
        },
        "output_schema": {"required": ["chunk_id", "content", "citation"]},
        "extra_config": {"max_content_chars": 8000, "max_metadata_chars": 2000},
    },
    {
        "name": "read_document_section",
        "label": "Read document section",
        "description": "Read one authorized document section by title or chunk index.",
        "callable": "ai_hub.tools.knowledge.read_document_section",
        "input_schema": {
            "required": ["document_id"],
            "properties": {
                "document_id": {"type": "integer"},
                "section_title": {"type": "string"},
                "chunk_index": {"type": "integer"},
            },
        },
        "output_schema": {"required": ["chunk_id", "content", "citation"]},
        "extra_config": {"max_content_chars": 8000, "max_metadata_chars": 2000},
    },
    {
        "name": "cite_knowledge_source",
        "label": "Cite knowledge source",
        "description": "Return citation metadata for one authorized knowledge chunk.",
        "callable": "ai_hub.tools.knowledge.cite_knowledge_source",
        "input_schema": {
            "required": ["chunk_id"],
            "properties": {"chunk_id": {"type": "integer"}},
        },
        "output_schema": {"required": ["citation"]},
    },
)


def establish_retrieval_foundation(apps, schema_editor):
    ToolDefinition = apps.get_model("ai_hub", "ToolDefinition")
    Toolbox = apps.get_model("ai_hub", "Toolbox")
    ToolboxTool = apps.get_model("ai_hub", "ToolboxTool")
    KnowledgeDocument = apps.get_model("ai_hub", "KnowledgeDocument")
    KnowledgeDocumentChunk = apps.get_model("ai_hub", "KnowledgeDocumentChunk")

    old_list_tool = ToolDefinition.objects.filter(
        name="list_available_knowledge_libraries",
        config__callable="ai_hub.tools.knowledge.list_knowledge_libraries",
    ).first()
    renamed_old_list_tool = False
    if old_list_tool and not ToolDefinition.objects.filter(name="list_knowledge_libraries").exists():
        old_list_tool.name = "list_knowledge_libraries"
        old_list_tool.save(update_fields=["name"])
        renamed_old_list_tool = True
    elif old_list_tool:
        # Preserve historical audit references while closing the old adapter,
        # whose schema let model output choose an arbitrary agent identity.
        old_list_tool.is_active = False
        old_list_tool.save(update_fields=["is_active"])
        ToolboxTool.objects.filter(tool_id=old_list_tool.pk).delete()

    toolbox, _ = Toolbox.objects.get_or_create(
        slug="knowledge-discovery-retrieval",
        defaults={
            "name": "Knowledge Discovery and Retrieval",
            "label": "Knowledge Discovery and Retrieval",
            "description": "Search, read and cite assigned knowledge chunks on demand.",
            "is_active": True,
        },
    )

    for order, spec in enumerate(TOOL_SPECS, start=1):
        config = {
            "callable": spec["callable"],
            "read_only": True,
            "game_tool_category": "context_tool",
            "bind_agent_context": True,
            **spec.get("extra_config", {}),
        }
        defaults = {
            "label": spec["label"],
            "description": spec["description"],
            "tool_kind": "python_callable",
            "input_schema": spec["input_schema"],
            "output_schema": spec["output_schema"],
            "config": config,
            "risk_level": "low",
            "operation_mode": "read",
            "requires_approval": False,
            "is_system_tool": True,
            "is_active": True,
        }
        tool = ToolDefinition.objects.filter(name=spec["name"]).first()
        if tool is not None:
            configured_callable = (tool.config or {}).get("callable")
            if configured_callable != spec["callable"]:
                raise RuntimeError(
                    "Cannot install the built-in Knowledge retrieval tool "
                    f"'{spec['name']}': that name is already used by "
                    f"'{configured_callable}'. Rename the custom tool and retry."
                )
            created = False
        else:
            tool = ToolDefinition.objects.create(name=spec["name"], **defaults)
            created = True
        if not created:
            for field, value in defaults.items():
                setattr(tool, field, value)
            tool.save(update_fields=list(defaults))
        ToolboxTool.objects.get_or_create(
            toolbox=toolbox,
            tool=tool,
            defaults={
                "display_order": order,
                "is_enabled": True,
                "default_enabled": True,
            },
        )

    if renamed_old_list_tool:
        # The old alias was a Core toolbox seed. Retrieval capabilities now
        # belong to Knowledge and are also resolved from collection access.
        ToolboxTool.objects.filter(
            tool_id=old_list_tool.pk,
            toolbox__slug="core-foundation",
        ).delete()

    documents = KnowledgeDocument.objects.exclude(curated_text="").iterator()
    for document in documents:
        if KnowledgeDocumentChunk.objects.filter(document_id=document.pk).exists():
            continue
        content = str(document.curated_text or "").strip()
        if not content:
            continue
        KnowledgeDocumentChunk.objects.create(
            document_id=document.pk,
            chunk_index=1,
            section_title=document.title,
            content=content,
            token_estimate=max(len(content.split()), 1),
            metadata={"ingestion": "initial_curated_text_backfill"},
        )


class Migration(migrations.Migration):
    dependencies = [
        ("ai_hub", "0018_alter_agentprofile_options_and_more"),
    ]

    operations = [
        migrations.RunPython(
            establish_retrieval_foundation,
            migrations.RunPython.noop,
        ),
    ]
