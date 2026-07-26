from dataclasses import dataclass

from django.core.exceptions import ValidationError

from ai_hub.models import AgentProfile, AgentToolGrant, ToolDefinition, ToolboxTool
from ai_hub.services.knowledge_tooling import (
    KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES,
    KNOWLEDGE_RETRIEVAL_TOOL_NAMES,
    is_bound_knowledge_tool,
)
from ai_hub.services.http_tool_policy import build_http_tool_configuration
from ai_hub.services.tools_runtime import ALLOWED_TOOL_KINDS


@dataclass(frozen=True)
class ResolvedTool:
    tool: ToolDefinition
    source: str
    permission_level: str
    requires_approval: bool

    def manifest(self) -> dict:
        return {
            "name": self.tool.name,
            "label": self.tool.label or self.tool.name,
            "description": self.tool.description,
            "operation_mode": self.tool.operation_mode,
            "risk_level": self.tool.risk_level,
            "requires_approval": self.requires_approval,
            "input_schema": self.tool.input_schema or {},
            "output_schema": self.tool.output_schema or {},
        }


@dataclass(frozen=True)
class AgentToolResolution:
    agent: AgentProfile
    tools: tuple[ResolvedTool, ...]

    def manifest(self) -> list[dict]:
        return [resolved_tool.manifest() for resolved_tool in self.tools]

    def tool_names(self) -> list[str]:
        return [resolved_tool.tool.name for resolved_tool in self.tools]


def _policy_names(policy: dict, key: str) -> set[str] | None:
    value = (policy or {}).get(key)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValidationError(f"Policy '{key}' must be a list of non-empty tool names.")
    return {name.strip() for name in value}


def _effective_requires_approval(tool: ToolDefinition, grant: AgentToolGrant | None, workspace) -> bool:
    if grant is not None and grant.requires_approval_override is not None:
        requires_approval = bool(grant.requires_approval_override)
    else:
        requires_approval = bool(tool.requires_approval)
    policy = getattr(workspace, "default_policy", None) or {}
    safety = policy.get("safety", {}) if isinstance(policy, dict) else {}
    if not isinstance(safety, dict):
        raise ValidationError("Policy 'safety' must be a JSON object.")

    if tool.risk_level == ToolDefinition.RiskLevel.HIGH and safety.get("require_approval_for_high_risk"):
        requires_approval = True
    if tool.risk_level == ToolDefinition.RiskLevel.MEDIUM and safety.get("require_approval_for_medium_risk"):
        requires_approval = True

    return requires_approval


def _workspace_allows_tool(tool: ToolDefinition, workspace) -> bool:
    if workspace is None:
        return True

    policy = workspace.default_policy or {}
    if not isinstance(policy, dict):
        raise ValidationError("Workspace policy must be a JSON object.")

    allowed_tools = _policy_names(policy, "allowed_tools")
    if allowed_tools is not None and tool.name not in allowed_tools:
        return False

    blocked_tools = _policy_names(policy, "blocked_tools") or set()
    if tool.name in blocked_tools:
        return False

    safety = policy.get("safety", {})
    if not isinstance(safety, dict):
        raise ValidationError("Policy 'safety' must be a JSON object.")
    if (
        tool.operation_mode == ToolDefinition.OperationMode.EXTERNAL_WRITE
        and not safety.get("allow_external_writes", False)
    ):
        return False

    return True


def _runtime_supports_tool(tool: ToolDefinition) -> bool:
    if tool.tool_kind not in ALLOWED_TOOL_KINDS:
        return False
    if tool.tool_kind == ToolDefinition.ToolKind.HTTP:
        build_http_tool_configuration(tool)
    return True


def _permission_allows_operation(permission_level: str, operation_mode: str) -> bool:
    allowed_modes = {
        AgentToolGrant.PermissionLevel.USE: {
            ToolDefinition.OperationMode.READ,
            ToolDefinition.OperationMode.DRAFT_WRITE,
            ToolDefinition.OperationMode.STATE_WRITE,
            ToolDefinition.OperationMode.EXTERNAL_WRITE,
            ToolDefinition.OperationMode.EXECUTE,
        },
        AgentToolGrant.PermissionLevel.READ_ONLY: {ToolDefinition.OperationMode.READ},
        AgentToolGrant.PermissionLevel.DRAFT_WRITE: {
            ToolDefinition.OperationMode.READ,
            ToolDefinition.OperationMode.DRAFT_WRITE,
        },
        AgentToolGrant.PermissionLevel.STATE_WRITE: {
            ToolDefinition.OperationMode.READ,
            ToolDefinition.OperationMode.DRAFT_WRITE,
            ToolDefinition.OperationMode.STATE_WRITE,
        },
        AgentToolGrant.PermissionLevel.EXTERNAL_WRITE: {
            ToolDefinition.OperationMode.READ,
            ToolDefinition.OperationMode.DRAFT_WRITE,
            ToolDefinition.OperationMode.STATE_WRITE,
            ToolDefinition.OperationMode.EXTERNAL_WRITE,
        },
        AgentToolGrant.PermissionLevel.EXECUTE: {
            ToolDefinition.OperationMode.READ,
            ToolDefinition.OperationMode.EXECUTE,
        },
    }
    return operation_mode in allowed_modes.get(permission_level, set())


def _add_tool(
    resolved: dict[int, ResolvedTool],
    *,
    tool: ToolDefinition,
    source: str,
    permission_level: str,
    grant: AgentToolGrant | None = None,
    workspace=None,
) -> None:
    if not tool.is_active:
        return
    if not _runtime_supports_tool(tool):
        return
    if not _workspace_allows_tool(tool, workspace):
        return
    if not _permission_allows_operation(permission_level, tool.operation_mode):
        return
    resolved[tool.pk] = ResolvedTool(
        tool=tool,
        source=source,
        permission_level=permission_level,
        requires_approval=_effective_requires_approval(tool, grant, workspace),
    )


def resolve_agent_tools(agent: AgentProfile, workspace=None, execution_context=None) -> AgentToolResolution:
    """Resolve the tools an agent is allowed to see.

    This service is intentionally read-only. It is the capability source for
    normal agent calls, Admin manifests and unified GAME tool adapters; actual
    execution remains in the runtime/dispatcher layers.
    """
    resolved: dict[int, ResolvedTool] = {}

    if agent.knowledge_collections.filter(is_active=True).exists():
        knowledge_tools = ToolDefinition.objects.filter(
            name__in=KNOWLEDGE_RETRIEVAL_TOOL_NAMES,
            is_system_tool=True,
            is_active=True,
        )
        for tool in knowledge_tools:
            if not is_bound_knowledge_tool(tool):
                continue
            if (tool.config or {}).get("callable") != KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES[tool.name]:
                continue
            _add_tool(
                resolved,
                tool=tool,
                source="knowledge_retrieval",
                permission_level=AgentToolGrant.PermissionLevel.READ_ONLY,
                workspace=workspace,
            )

    toolbox_entries = (
        ToolboxTool.objects.select_related("tool", "toolbox")
        .filter(
            toolbox__agent_assignments__agent=agent,
            toolbox__agent_assignments__is_enabled=True,
            toolbox__is_active=True,
            is_enabled=True,
            tool__is_active=True,
        )
        .distinct()
    )
    for entry in toolbox_entries:
        _add_tool(
            resolved,
            tool=entry.tool,
            source="toolbox",
            permission_level=AgentToolGrant.PermissionLevel.USE,
            workspace=workspace,
        )

    for tool in agent.tools.filter(is_active=True):
        _add_tool(
            resolved,
            tool=tool,
            source="legacy_direct",
            permission_level=AgentToolGrant.PermissionLevel.USE,
            workspace=workspace,
        )

    grants = {
        grant.tool_id: grant
        for grant in AgentToolGrant.objects.select_related("tool").filter(agent=agent)
    }

    for grant in grants.values():
        if grant.is_enabled:
            if not _permission_allows_operation(grant.permission_level, grant.tool.operation_mode):
                resolved.pop(grant.tool_id, None)
                continue
            _add_tool(
                resolved,
                tool=grant.tool,
                source="agent_grant",
                permission_level=grant.permission_level,
                grant=grant,
                workspace=workspace,
            )
        else:
            resolved.pop(grant.tool_id, None)

    tools = tuple(sorted(resolved.values(), key=lambda resolved_tool: resolved_tool.tool.name))
    return AgentToolResolution(agent=agent, tools=tools)
