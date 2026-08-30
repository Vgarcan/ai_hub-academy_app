from django.db import transaction

from ai_hub.models import (
    AgentProfile,
    GameActionDefinition,
    GameWorkspace,
    GameWorkspaceAction,
    GameWorkspaceAgent,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ToolDefinition,
)
from ai_hub.services.starter_toolboxes import seed_starter_toolboxes
from ai_hub.services.application_scope import require_single_active_scope


@transaction.atomic
def seed_starter_demo(*, force_update: bool = False) -> dict:
    seed_starter_toolboxes(force_update=force_update)
    stats = {
        "collections_created": 0,
        "documents_created": 0,
        "chunks_created": 0,
        "workspaces_created": 0,
        "actions_created": 0,
        "workspace_actions_created": 0,
        "workspace_agents_created": 0,
    }

    # Explicit ownership: the starter demo seeds into the single active scope
    # and refuses rather than guessing when more than one exists.
    scope = require_single_active_scope()

    collection, created = KnowledgeCollection.objects.get_or_create(
        name="AI Hub Starter Knowledge",
        defaults={
            "description": "Example knowledge library for starter agents.",
            "is_active": True,
            "application_scope": scope,
        },
    )
    if created:
        stats["collections_created"] += 1
    elif force_update:
        collection.description = "Example knowledge library for starter agents."
        collection.is_active = True
        collection.save(update_fields=["description", "is_active", "updated_at"])

    document, created = KnowledgeDocument.objects.get_or_create(
        collection=collection,
        title="Starter operating notes",
        defaults={
            "curated_text": (
                "Starter agents should search assigned knowledge first, draft changes for review, "
                "and avoid external writes unless a workspace policy explicitly allows them."
            ),
            "tags": ["starter", "safety", "workflow"],
            "language": "en",
            "status": KnowledgeDocument.Status.ACTIVE,
        },
    )
    if created:
        stats["documents_created"] += 1
    elif force_update:
        document.curated_text = (
            "Starter agents should search assigned knowledge first, draft changes for review, "
            "and avoid external writes unless a workspace policy explicitly allows them."
        )
        document.tags = ["starter", "safety", "workflow"]
        document.language = "en"
        document.status = KnowledgeDocument.Status.ACTIVE
        document.save(update_fields=["curated_text", "tags", "language", "status", "updated_at"])

    _, created = KnowledgeDocumentChunk.objects.get_or_create(
        document=document,
        chunk_index=1,
        defaults={
            "section_title": "Safe starter workflow",
            "content": document.curated_text,
            "token_estimate": 32,
            "metadata": {"example": True},
        },
    )
    if created:
        stats["chunks_created"] += 1

    workspace, created = GameWorkspace.objects.get_or_create(
        name="AI Hub Starter GAME Workspace",
        defaults={
            "application_scope": scope,
            "description": "Safe example GAME workspace for starter agents.",
            "default_policy": {
                "allowed_actions": ["submit_for_approval"],
                "safety": {
                    "allow_external_writes": False,
                    "require_approval_for_medium_risk": True,
                    "require_approval_for_high_risk": True,
                },
                "budget": {
                    "max_iterations_per_session": 3,
                    "max_action_runs_per_session": 2,
                },
            },
            "default_runtime_config": {
                "max_iterations": 3,
                "use_action_dispatcher": True,
                "strict_response_contract": True,
                "available_actions": ["submit_for_approval"],
            },
            "is_active": True,
        },
    )
    if created:
        stats["workspaces_created"] += 1
    elif force_update:
        workspace.description = "Safe example GAME workspace for starter agents."
        workspace.default_policy = {
            "allowed_actions": ["submit_for_approval"],
            "safety": {
                "allow_external_writes": False,
                "require_approval_for_medium_risk": True,
                "require_approval_for_high_risk": True,
            },
            "budget": {
                "max_iterations_per_session": 3,
                "max_action_runs_per_session": 2,
            },
        }
        workspace.default_runtime_config = {
            "max_iterations": 3,
            "use_action_dispatcher": True,
            "strict_response_contract": True,
            "available_actions": ["submit_for_approval"],
        }
        workspace.is_active = True
        workspace.save(update_fields=["description", "default_policy", "default_runtime_config", "is_active", "updated_at"])

    approval_tool = ToolDefinition.objects.get(name="submit_for_approval")
    action, created = GameActionDefinition.objects.get_or_create(
        name="submit_for_approval",
        defaults={
            "label": "Submit for approval",
            "description": "Example approval-gated GAME action linked to a reusable draft tool.",
            "action_type": GameActionDefinition.ActionType.TOOL,
            "tool": approval_tool,
            "input_contract": {},
            "output_contract": {},
            "risk_level": "medium",
            "requires_approval": True,
            "is_active": True,
        },
    )
    if created:
        stats["actions_created"] += 1
    elif force_update:
        action.label = "Submit for approval"
        action.description = "Example approval-gated GAME action linked to a reusable draft tool."
        action.action_type = GameActionDefinition.ActionType.TOOL
        action.tool = approval_tool
        action.risk_level = "medium"
        action.requires_approval = True
        action.is_active = True
        action.save(update_fields=["label", "description", "action_type", "tool", "risk_level", "requires_approval", "is_active", "updated_at"])

    _, created = GameWorkspaceAction.objects.get_or_create(
        workspace=workspace,
        action=action,
        defaults={"is_enabled": True, "requires_approval_override": True},
    )
    if created:
        stats["workspace_actions_created"] += 1

    for agent_name in ("Business Analyst Agent", "Developer Assistant"):
        agent = AgentProfile.objects.get(name=agent_name)
        agent.knowledge_collections.add(collection)
        _, created = GameWorkspaceAgent.objects.get_or_create(
            workspace=workspace,
            agent=agent,
            defaults={"is_enabled": True},
        )
        if created:
            stats["workspace_agents_created"] += 1

    return stats
