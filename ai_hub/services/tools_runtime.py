import importlib
import json
from urllib.parse import urljoin, urlparse

import requests
from django.conf import settings
from django.core.exceptions import ValidationError

from ai_hub.models import ToolDefinition
from ai_hub.services.contracts import validate_payload
from ai_hub.services.credential_safety import is_sensitive_credential_name
from ai_hub.services.knowledge_tooling import (
    BIND_AGENT_CONTEXT_CONFIG_KEY,
    is_bound_knowledge_tool,
)
from ai_hub.services.http_tool_policy import (
    HTTP_READ_METHODS,
    HTTP_REDIRECT_STATUSES,
    build_http_tool_configuration,
    validate_http_destination,
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
HTTP_MODEL_FACING_BODY_CHARS = 4000
HTTP_STREAM_READ_CHUNK_BYTES = 64 * 1024


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
    runtime_config = build_http_tool_configuration(tool)
    current_url = runtime_config.url
    current_method = runtime_config.method
    current_headers = runtime_config.headers
    include_payload = True
    redirects_followed = 0

    while True:
        validate_http_destination(
            current_url,
            runtime_config.allowed_hosts,
            tool_name=tool.name,
        )
        request_kwargs = {
            "headers": current_headers,
            "timeout": runtime_config.timeout,
            "allow_redirects": False,
            "stream": True,
        }
        if include_payload:
            if current_method in HTTP_READ_METHODS:
                request_kwargs["params"] = payload
            else:
                request_kwargs["json"] = payload

        response = requests.request(current_method, current_url, **request_kwargs)
        try:
            location = (getattr(response, "headers", {}) or {}).get("Location")
            if response.status_code not in HTTP_REDIRECT_STATUSES or not location:
                return {
                    "status_code": response.status_code,
                    "body": _read_bounded_http_body(
                        response,
                        max_bytes=runtime_config.max_response_bytes,
                        tool_name=tool.name,
                    ),
                }

            if redirects_followed >= runtime_config.max_redirects:
                raise ValidationError(
                    f"Tool '{tool.name}' exceeded its redirect limit "
                    f"({runtime_config.max_redirects})."
                )

            next_url = urljoin(current_url, str(location).strip())
            validate_http_destination(
                next_url,
                runtime_config.allowed_hosts,
                tool_name=tool.name,
            )
            current_headers = _redirect_headers(current_headers, current_url, next_url)
            current_method, include_payload = _redirect_method(
                current_method,
                response.status_code,
                include_payload,
            )
            current_url = next_url
            redirects_followed += 1
        finally:
            response.close()


def _read_bounded_http_body(response, *, max_bytes: int, tool_name: str) -> str:
    headers = getattr(response, "headers", {}) or {}
    raw_content_length = headers.get("Content-Length")
    if raw_content_length is not None:
        try:
            content_length = int(str(raw_content_length).strip())
        except (TypeError, ValueError):
            content_length = None
        if content_length is not None and content_length > max_bytes:
            raise ValidationError(
                f"Tool '{tool_name}' HTTP response exceeds its maximum response "
                f"size of {max_bytes} bytes."
            )

    raw_stream = getattr(response, "raw", None)
    if raw_stream is None or not callable(getattr(raw_stream, "read", None)):
        raise ValidationError(
            f"Tool '{tool_name}' HTTP response does not expose a streaming body."
        )

    body = bytearray()
    while len(body) <= max_bytes:
        remaining = (max_bytes + 1) - len(body)
        read_size = min(HTTP_STREAM_READ_CHUNK_BYTES, remaining)
        chunk = raw_stream.read(read_size, decode_content=True)
        if not chunk:
            break
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ValidationError(
                f"Tool '{tool_name}' HTTP response exceeds its maximum response "
                f"size of {max_bytes} bytes."
            )

    encoding = str(getattr(response, "encoding", "") or "utf-8")
    try:
        decoded = bytes(body).decode(encoding, errors="replace")
    except LookupError:
        decoded = bytes(body).decode("utf-8", errors="replace")
    return decoded[:HTTP_MODEL_FACING_BODY_CHARS]


def _redirect_method(method: str, status_code: int, include_payload: bool) -> tuple[str, bool]:
    if status_code == 303 and method != "HEAD":
        return "GET", False
    if status_code == 302 and method != "HEAD":
        return "GET", False
    if status_code == 301 and method == "POST":
        return "GET", False
    return method, include_payload


def _redirect_headers(headers: dict, source_url: str, target_url: str) -> dict:
    source = urlparse(source_url)
    target = urlparse(target_url)
    source_origin = (source.scheme.lower(), source.hostname, source.port)
    target_origin = (target.scheme.lower(), target.hostname, target.port)
    if source_origin == target_origin:
        return headers

    return {
        name: value
        for name, value in headers.items()
        if not is_sensitive_credential_name(name)
    }


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
