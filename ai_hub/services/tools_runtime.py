import importlib

import requests
from django.core.exceptions import ValidationError

from ai_hub.models import ToolDefinition
from ai_hub.services.contracts import validate_payload


ALLOWED_TOOL_KINDS = {
    ToolDefinition.ToolKind.HTTP,
    ToolDefinition.ToolKind.PROMPT_MACRO,
    ToolDefinition.ToolKind.PYTHON_CALLABLE,
}


def _execute_python_callable(tool: ToolDefinition, payload: dict) -> dict:
    callable_path = (tool.config or {}).get("callable", "")
    if not callable_path:
        raise ValidationError(f"Tool '{tool.name}' is missing 'callable' in config.")
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
    headers = config.get("headers", {})
    timeout = config.get("timeout", 30)
    response = requests.request(method, url, json=payload, headers=headers, timeout=timeout)
    return {"status_code": response.status_code, "body": response.text[:4000]}


def execute_tools(tools, payload: dict) -> dict:
    output = {}
    for tool in tools:
        if tool.tool_kind not in ALLOWED_TOOL_KINDS:
            raise ValidationError(f"Tool kind '{tool.tool_kind}' is not allowed.")
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
        output[tool.name] = tool_result
    return output
