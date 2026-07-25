import importlib
import json
from urllib.parse import urlparse

import requests
from django.conf import settings
from django.core.exceptions import ValidationError

from ai_hub.models import ToolDefinition
from ai_hub.services.contracts import validate_payload
from ai_hub.services.knowledge_tooling import (
    BIND_AGENT_CONTEXT_CONFIG_KEY,
    is_bound_knowledge_tool,
)


ALLOWED_TOOL_KINDS = {
    ToolDefinition.ToolKind.HTTP,
    ToolDefinition.ToolKind.PROMPT_MACRO,
    ToolDefinition.ToolKind.PYTHON_CALLABLE,
}

GAME_TOOL_CATEGORY_KEY = "game_tool_category"
GAME_CONTEXT_TOOL = "context_tool"
GAME_ACTION_TOOL = "action_tool"
TOOL_POLICY_ALL = "all"
TOOL_POLICY_GAME_CONTEXT_ONLY = "game_context_only"


def get_game_tool_category(tool: ToolDefinition) -> str:
    config = tool.config or {}
    if config.get(GAME_TOOL_CATEGORY_KEY) != GAME_CONTEXT_TOOL:
        return GAME_ACTION_TOOL
    if tool.tool_kind == ToolDefinition.ToolKind.PROMPT_MACRO:
        return GAME_CONTEXT_TOOL
    if tool.tool_kind == ToolDefinition.ToolKind.PYTHON_CALLABLE and config.get("read_only") is True:
        return GAME_CONTEXT_TOOL
    if tool.tool_kind == ToolDefinition.ToolKind.HTTP and config.get("method", "POST").upper() in {"GET", "HEAD"}:
        return GAME_CONTEXT_TOOL
    return GAME_ACTION_TOOL


def _execute_python_callable(tool: ToolDefinition, payload: dict) -> dict:
    callable_path = (tool.config or {}).get("callable", "")
    if not callable_path:
        raise ValidationError(f"Tool '{tool.name}' is missing 'callable' in config.")
    allowed_callables = set(getattr(settings, "AI_HUB_ALLOWED_TOOL_CALLABLES", ()))
    if callable_path not in allowed_callables:
        raise ValidationError(f"Tool '{tool.name}' callable is not in AI_HUB_ALLOWED_TOOL_CALLABLES.")
    try:
        module_path, func_name = callable_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValidationError(f"Tool '{tool.name}' callable not found: {exc}") from exc
    result = func(payload, tool.config or {})
    if not isinstance(result, dict):
        result = {"result": result}
    return result


def _execute_http_tool(tool: ToolDefinition, payload: dict) -> dict:
    config = tool.config or {}
    url = config.get("url", "")
    if not url:
        raise ValidationError(f"Tool '{tool.name}' is missing 'url' in config.")
    method = config.get("method", "POST").upper()
    parsed_url = urlparse(url)
    allowed_hosts = set(config.get("allowed_hosts") or [])
    if not parsed_url.hostname or parsed_url.hostname not in allowed_hosts:
        raise ValidationError(f"Tool '{tool.name}' HTTP host is not explicitly allowed.")
    headers = config.get("headers", {})
    timeout = min(max(int(config.get("timeout", 30)), 1), 60)
    request_kwargs = {"headers": headers, "timeout": timeout}
    if method in {"GET", "HEAD"}:
        request_kwargs["params"] = payload
    else:
        request_kwargs["json"] = payload
    response = requests.request(method, url, **request_kwargs)
    return {"status_code": response.status_code, "body": response.text[:4000]}


def _ensure_json_serializable(result: dict, tool_name: str) -> dict:
    # Fix #5: reject non-serializable results at the tool boundary so they never
    # reach json.dumps() further up the call stack (e.g. ToolExecutionRun.output_payload
    # or the LLM message), where the failure would be harder to diagnose.
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Tool '{tool_name}' returned a non-JSON-serializable result: {exc}"
        ) from exc
    return result


def bind_tool_runtime_context(tool: ToolDefinition, payload: dict, *, agent=None) -> dict:
    effective_payload = dict(payload or {})
    config = tool.config or {}
    if config.get(BIND_AGENT_CONTEXT_CONFIG_KEY) is not True:
        return effective_payload
    if not is_bound_knowledge_tool(tool):
        raise ValidationError(
            f"Tool '{tool.name}' requests agent binding but is not a registered system knowledge tool."
        )
    if agent is None or not getattr(agent, "pk", None):
        raise ValidationError(
            f"Tool '{tool.name}' requires an authenticated agent runtime context."
        )
    # Never trust a model-supplied identity. The current runtime agent is the
    # authorization principal and always wins.
    effective_payload.pop("agent_name", None)
    effective_payload["agent_id"] = agent.pk
    return effective_payload


def execute_tool(tool: ToolDefinition, payload: dict, *, agent=None) -> dict:
    if tool.tool_kind not in ALLOWED_TOOL_KINDS:
        raise ValidationError(f"Tool kind '{tool.tool_kind}' is not allowed.")
    payload = bind_tool_runtime_context(tool, payload, agent=agent)
    validate_payload(payload, tool.input_schema or {}, f"Tool '{tool.name}' input")

    if tool.tool_kind == ToolDefinition.ToolKind.PYTHON_CALLABLE:
        tool_result = _execute_python_callable(tool, payload)
    elif tool.tool_kind == ToolDefinition.ToolKind.HTTP:
        tool_result = _execute_http_tool(tool, payload)
    elif tool.tool_kind == ToolDefinition.ToolKind.PROMPT_MACRO:
        macro_text = (tool.config or {}).get("template", "")
        tool_result = {"macro": macro_text}
    else:
        tool_result = {"status": "unsupported_kind", "tool_kind": tool.tool_kind}

    validate_payload(tool_result, tool.output_schema or {}, f"Tool '{tool.name}' output")
    return _ensure_json_serializable(tool_result, tool.name)


def execute_tools(tools, payload: dict, *, policy: str = TOOL_POLICY_ALL, agent=None) -> dict:
    if policy not in {TOOL_POLICY_ALL, TOOL_POLICY_GAME_CONTEXT_ONLY}:
        raise ValidationError(f"Unknown tool execution policy '{policy}'.")

    output = {}
    for tool in tools:
        if policy == TOOL_POLICY_GAME_CONTEXT_ONLY and get_game_tool_category(tool) != GAME_CONTEXT_TOOL:
            continue
        output[tool.name] = execute_tool(tool, payload, agent=agent)
    return output
