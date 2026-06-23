from dataclasses import dataclass
from decimal import Decimal

from django.db import transaction

from ai_hub.models import (
    AgentProfile,
    AgentToolboxAssignment,
    ModelConfig,
    ProviderConfig,
    ToolDefinition,
    Toolbox,
    ToolboxTool,
)


@dataclass(frozen=True)
class ToolSeed:
    name: str
    label: str
    description: str
    tool_kind: str = ToolDefinition.ToolKind.PROMPT_MACRO
    operation_mode: str = ToolDefinition.OperationMode.READ
    risk_level: str = ToolDefinition.RiskLevel.LOW
    requires_approval: bool = False
    input_schema: dict | None = None
    output_schema: dict | None = None
    config: dict | None = None


CORE_TOOLS = [
    ToolSeed(
        name="list_available_tools",
        label="List available tools",
        description="Explain the tool manifest currently available to the agent.",
        config={"template": "Use the provided tool manifest to summarize available capabilities."},
    ),
    ToolSeed(
        name="list_available_knowledge_libraries",
        label="List knowledge libraries",
        description="List knowledge libraries assigned to the current agent.",
        tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
        input_schema={"required": ["agent_id"], "properties": {"agent_id": {"type": "integer"}}},
        output_schema={"required": ["libraries", "total"]},
        config={"callable": "ai_hub.tools.knowledge.list_knowledge_libraries", "read_only": True},
    ),
    ToolSeed(
        name="get_agent_permissions",
        label="Get agent permissions",
        description="Summarize known agent boundaries from the prompt and manifest.",
        config={"template": "Summarize agent permissions from the active manifest and instructions."},
    ),
    ToolSeed(
        name="validate_input",
        label="Validate input",
        description="Draft validation notes against the expected input contract.",
        config={"template": "Check the supplied context against required fields and report missing values."},
    ),
    ToolSeed(
        name="create_structured_result",
        label="Create structured result",
        description="Draft a structured JSON result that matches the requested schema.",
        operation_mode=ToolDefinition.OperationMode.DRAFT_WRITE,
        config={"template": "Return a draft structured JSON result. Do not apply external changes."},
    ),
]


KNOWLEDGE_TOOLS = [
    ToolSeed(
        name="browse_knowledge_index",
        label="Browse knowledge index",
        description="Browse assigned collections, documents and chunk metadata.",
        tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
        input_schema={"required": ["agent_id"], "properties": {"agent_id": {"type": "integer"}}},
        output_schema={"required": ["collections", "total"]},
        config={"callable": "ai_hub.tools.knowledge.browse_knowledge_index", "read_only": True},
    ),
    ToolSeed(
        name="search_knowledge",
        label="Search knowledge",
        description="Search assigned active knowledge chunks by lexical relevance.",
        tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
        input_schema={
            "required": ["agent_id", "query"],
            "properties": {"agent_id": {"type": "integer"}, "query": {"type": "string"}},
        },
        output_schema={"required": ["query", "results", "total"]},
        config={"callable": "ai_hub.tools.knowledge.search_knowledge", "read_only": True, "limit": 5},
    ),
    ToolSeed(
        name="read_knowledge_chunk",
        label="Read knowledge chunk",
        description="Read one authorized knowledge chunk.",
        tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
        input_schema={
            "required": ["agent_id", "chunk_id"],
            "properties": {"agent_id": {"type": "integer"}, "chunk_id": {"type": "integer"}},
        },
        output_schema={"required": ["chunk_id", "content", "citation"]},
        config={"callable": "ai_hub.tools.knowledge.read_knowledge_chunk", "read_only": True},
    ),
    ToolSeed(
        name="read_document_section",
        label="Read document section",
        description="Read one authorized document section by title or chunk index.",
        tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
        input_schema={
            "required": ["agent_id", "document_id"],
            "properties": {"agent_id": {"type": "integer"}, "document_id": {"type": "integer"}},
        },
        output_schema={"required": ["chunk_id", "content", "citation"]},
        config={"callable": "ai_hub.tools.knowledge.read_document_section", "read_only": True},
    ),
    ToolSeed(
        name="cite_knowledge_source",
        label="Cite knowledge source",
        description="Return citation metadata for one authorized knowledge chunk.",
        tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
        input_schema={
            "required": ["agent_id", "chunk_id"],
            "properties": {"agent_id": {"type": "integer"}, "chunk_id": {"type": "integer"}},
        },
        output_schema={"required": ["citation"]},
        config={"callable": "ai_hub.tools.knowledge.cite_knowledge_source", "read_only": True},
    ),
]


DOCUMENT_TOOLS = [
    ToolSeed(
        name="inspect_file",
        label="Inspect file",
        description="Planning helper for reviewing provided file metadata; does not parse binary files.",
        config={"template": "Review supplied file metadata and propose safe next intake steps."},
    ),
    ToolSeed(
        name="split_large_document",
        label="Split large document",
        description="Draft a chunking plan for provided text; does not modify stored documents.",
        operation_mode=ToolDefinition.OperationMode.DRAFT_WRITE,
        config={"template": "Draft a section/chunk plan for the supplied text."},
    ),
    ToolSeed(
        name="create_source_manifest",
        label="Create source manifest",
        description="Draft a provenance manifest for supplied source metadata.",
        operation_mode=ToolDefinition.OperationMode.DRAFT_WRITE,
        config={"template": "Create a draft source manifest from supplied metadata."},
    ),
]


DRAFT_TOOLS = [
    ToolSeed(
        name="create_draft",
        label="Create draft",
        description="Create a draft artifact in the model response only.",
        operation_mode=ToolDefinition.OperationMode.DRAFT_WRITE,
        config={"template": "Create a draft artifact. Do not apply or publish changes."},
    ),
    ToolSeed(
        name="create_change_proposal",
        label="Create change proposal",
        description="Draft a change proposal for human review.",
        operation_mode=ToolDefinition.OperationMode.DRAFT_WRITE,
        config={"template": "Draft a change proposal with risks, assumptions and acceptance criteria."},
    ),
    ToolSeed(
        name="compare_versions",
        label="Compare versions",
        description="Compare supplied version text and summarize differences.",
        config={"template": "Compare the supplied versions and summarize meaningful differences."},
    ),
    ToolSeed(
        name="submit_for_approval",
        label="Submit for approval",
        description="Prepare an approval request payload; does not approve or apply changes.",
        operation_mode=ToolDefinition.OperationMode.DRAFT_WRITE,
        requires_approval=True,
        config={"template": "Prepare a draft approval request for a human reviewer."},
    ),
]


DEVELOPMENT_TOOLS = [
    ToolSeed(
        name="inspect_repository",
        label="Inspect repository",
        description="Planning helper for repository inspection; does not read local files directly.",
        config={"template": "Use provided repository context to plan inspection steps."},
    ),
    ToolSeed(
        name="search_code",
        label="Search code",
        description="Planning helper for code search based on provided snippets or indexes.",
        config={"template": "Search the supplied code context and report matches."},
    ),
    ToolSeed(
        name="propose_code_patch",
        label="Propose code patch",
        description="Draft a patch proposal for review; does not modify files.",
        operation_mode=ToolDefinition.OperationMode.DRAFT_WRITE,
        config={"template": "Propose a code patch in reviewable text. Do not apply it."},
    ),
]


TOOLBOXES = [
    {
        "name": "Core Foundation",
        "slug": "core-foundation",
        "label": "Core Foundation",
        "description": "Safe context and structure helpers for most agents.",
        "tools": CORE_TOOLS,
    },
    {
        "name": "Knowledge Discovery and Retrieval",
        "slug": "knowledge-discovery-retrieval",
        "label": "Knowledge Discovery and Retrieval",
        "description": "Search, read and cite assigned knowledge chunks on demand.",
        "tools": KNOWLEDGE_TOOLS,
    },
    {
        "name": "Documents and File Intake",
        "slug": "documents-file-intake",
        "label": "Documents and File Intake",
        "description": "Safe draft helpers for source intake planning.",
        "tools": DOCUMENT_TOOLS,
    },
    {
        "name": "Workspace and Draft Artifacts",
        "slug": "workspace-draft-artifacts",
        "label": "Workspace and Draft Artifacts",
        "description": "Draft-only artifact and proposal helpers.",
        "tools": DRAFT_TOOLS,
    },
    {
        "name": "Development and Code Assistance",
        "slug": "development-code-assistance",
        "label": "Development and Code Assistance",
        "description": "Draft-only developer assistance helpers.",
        "tools": DEVELOPMENT_TOOLS,
    },
]


ROLE_TOOLBOXES = {
    "General Assistant": [
        "core-foundation",
        "knowledge-discovery-retrieval",
        "documents-file-intake",
        "workspace-draft-artifacts",
    ],
    "Business Analyst Agent": [
        "core-foundation",
        "documents-file-intake",
        "knowledge-discovery-retrieval",
        "workspace-draft-artifacts",
    ],
    "Project and Delivery Agent": [
        "core-foundation",
        "documents-file-intake",
        "knowledge-discovery-retrieval",
        "workspace-draft-artifacts",
    ],
    "Developer Assistant": [
        "core-foundation",
        "knowledge-discovery-retrieval",
        "development-code-assistance",
        "workspace-draft-artifacts",
    ],
    "Knowledge Curator": [
        "core-foundation",
        "documents-file-intake",
        "knowledge-discovery-retrieval",
        "workspace-draft-artifacts",
    ],
}


ROLE_PROMPT = """You are {name}, a starter AI Hub role.

Use only authorized tools. Prefer read-only retrieval before drafting.
Do not apply irreversible changes. Return clear structured results for review.
"""


def _starter_model_config() -> ModelConfig:
    provider, _ = ProviderConfig.objects.get_or_create(
        name="AI Hub Training Provider",
        defaults={
            "provider_type": ProviderConfig.ProviderType.TRAINING,
            "is_active": True,
        },
    )
    model, _ = ModelConfig.objects.get_or_create(
        provider=provider,
        model_name="training/starter",
        defaults={
            "temperature_default": Decimal("0.30"),
            "max_tokens_default": 1500,
            "supports_tools": False,
            "is_active": True,
        },
    )
    return model


def _upsert_tool(seed: ToolSeed, *, force_update: bool) -> tuple[ToolDefinition, bool]:
    defaults = {
        "label": seed.label,
        "description": seed.description,
        "tool_kind": seed.tool_kind,
        "operation_mode": seed.operation_mode,
        "risk_level": seed.risk_level,
        "requires_approval": seed.requires_approval,
        "input_schema": seed.input_schema or {},
        "output_schema": seed.output_schema or {},
        "config": seed.config or {},
        "is_active": True,
    }
    tool, created = ToolDefinition.objects.get_or_create(name=seed.name, defaults=defaults)
    if not created and force_update:
        for field, value in defaults.items():
            setattr(tool, field, value)
        tool.save(update_fields=[*defaults.keys(), "updated_at"])
    return tool, created


@transaction.atomic
def seed_starter_toolboxes(*, force_update: bool = False, model_config: ModelConfig | None = None) -> dict:
    model_config = model_config or _starter_model_config()
    stats = {
        "tools_created": 0,
        "toolboxes_created": 0,
        "memberships_created": 0,
        "agents_created": 0,
        "assignments_created": 0,
    }
    toolbox_by_slug = {}

    for toolbox_seed in TOOLBOXES:
        toolbox_defaults = {
            "label": toolbox_seed["label"],
            "description": toolbox_seed["description"],
            "is_active": True,
        }
        toolbox, created = Toolbox.objects.get_or_create(
            slug=toolbox_seed["slug"],
            defaults={
                "name": toolbox_seed["name"],
                **toolbox_defaults,
            },
        )
        if created:
            stats["toolboxes_created"] += 1
        elif force_update:
            toolbox.name = toolbox_seed["name"]
            for field, value in toolbox_defaults.items():
                setattr(toolbox, field, value)
            toolbox.save(update_fields=["name", *toolbox_defaults.keys(), "updated_at"])
        toolbox_by_slug[toolbox.slug] = toolbox

        for order, tool_seed in enumerate(toolbox_seed["tools"], start=1):
            tool, tool_created = _upsert_tool(tool_seed, force_update=force_update)
            if tool_created:
                stats["tools_created"] += 1
            membership, membership_created = ToolboxTool.objects.get_or_create(
                toolbox=toolbox,
                tool=tool,
                defaults={
                    "display_order": order,
                    "is_enabled": True,
                    "default_enabled": True,
                },
            )
            if membership_created:
                stats["memberships_created"] += 1
            elif force_update:
                membership.display_order = order
                membership.is_enabled = True
                membership.default_enabled = True
                membership.save(update_fields=["display_order", "is_enabled", "default_enabled"])

    for role_name, toolbox_slugs in ROLE_TOOLBOXES.items():
        agent_defaults = {
            "role": role_name,
            "system_prompt": ROLE_PROMPT.format(name=role_name),
            "model_config": model_config,
            "input_contract": {},
            "output_contract": {"required": ["agent", "llm", "tools"]},
            "is_active": True,
        }
        agent, created = AgentProfile.objects.get_or_create(name=role_name, defaults=agent_defaults)
        if created:
            stats["agents_created"] += 1
        elif force_update:
            for field, value in agent_defaults.items():
                setattr(agent, field, value)
            agent.save(update_fields=[*agent_defaults.keys(), "updated_at"])

        for toolbox_slug in toolbox_slugs:
            assignment, assignment_created = AgentToolboxAssignment.objects.get_or_create(
                agent=agent,
                toolbox=toolbox_by_slug[toolbox_slug],
                defaults={"is_enabled": True},
            )
            if assignment_created:
                stats["assignments_created"] += 1
            elif force_update and not assignment.is_enabled:
                assignment.is_enabled = True
                assignment.save(update_fields=["is_enabled"])

    return stats
