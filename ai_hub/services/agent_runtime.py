import json
import re
import time

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db.models import Count
from django.utils import timezone

from ai_hub.models import AgentProfile, ToolExecutionRun
from ai_hub.services.contracts import validate_payload
from ai_hub.services.litellm_client import completion_call
from ai_hub.services.knowledge_tooling import KNOWLEDGE_RETRIEVAL_TOOL_NAMES
from ai_hub.services.provider_registry import resolve_model_config
from ai_hub.services.tool_resolution import resolve_agent_tools
from ai_hub.services.tools_runtime import (
    GAME_CONTEXT_TOOL,
    TOOL_POLICY_ALL,
    TOOL_POLICY_GAME_CONTEXT_ONLY,
    bind_tool_runtime_context,
    execute_tool,
    execute_tools,
    get_game_tool_category,
)

KNOWLEDGE_PROMPT_MAX_COLLECTIONS = 20
KNOWLEDGE_PROMPT_MAX_DOCUMENTS_PER_COLLECTION = 50
KNOWLEDGE_PROMPT_MAX_DESCRIPTION_CHARS = 500
KNOWLEDGE_PROMPT_MAX_TAGS_PER_DOCUMENT = 20
KNOWLEDGE_PROMPT_MAX_TAG_CHARS = 100
DEFAULT_MAX_TOOL_OBSERVATION_CHARS = 12000


def require_active_agent(agent: AgentProfile) -> None:
    """Fail closed before an inactive Agent reaches Knowledge, Tools or a provider."""
    if not agent.is_active:
        raise ValidationError(f"Agent '{agent.name}' is inactive and cannot execute.")


def get_mapped_value(source: dict, source_key: str):
    current = source
    for part in str(source_key).split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def apply_mapping(source: dict, mapping: dict) -> dict:
    if not mapping:
        return dict(source)
    result = {}
    for target_key, source_key in mapping.items():
        result[target_key] = get_mapped_value(source, source_key)
    return result


def read_document_text(document) -> str:
    if document.curated_text:
        return document.curated_text
    if not document.source_file:
        return ""
    try:
        with document.source_file.open("rb") as source:
            return source.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _bounded_knowledge_tags(tags) -> list[str]:
    if not isinstance(tags, list):
        return []
    return [
        str(tag)[:KNOWLEDGE_PROMPT_MAX_TAG_CHARS]
        for tag in tags[:KNOWLEDGE_PROMPT_MAX_TAGS_PER_DOCUMENT]
    ]


def build_agent_knowledge_context(agent: AgentProfile, *, workspace=None) -> dict:
    max_chars = max(agent.knowledge_max_chars or 0, 0)
    documents = []
    remaining = max_chars
    collections = agent.knowledge_collections.filter(is_active=True).prefetch_related("documents")

    if not getattr(settings, "AI_HUB_LEGACY_EAGER_KNOWLEDGE_CONTEXT_ENABLED", False):
        collection_data = []
        total_collections = collections.count()
        selected_collections = list(
            collections.order_by("name")[:KNOWLEDGE_PROMPT_MAX_COLLECTIONS]
        )
        index_truncated = len(selected_collections) < total_collections
        for collection in selected_collections:
            active_documents = (
                collection.documents.filter(status="active")
                .annotate(chunk_total=Count("chunks"))
                .order_by("title")
            )
            total_documents = active_documents.count()
            selected_documents = list(
                active_documents[:KNOWLEDGE_PROMPT_MAX_DOCUMENTS_PER_COLLECTION]
            )
            if len(selected_documents) < total_documents:
                index_truncated = True
            collection_data.append(
                {
                    "name": collection.name,
                    "description": str(collection.description or "")[
                        :KNOWLEDGE_PROMPT_MAX_DESCRIPTION_CHARS
                    ],
                    "documents": [
                        {
                            "id": document.pk,
                            "title": document.title,
                            "language": document.language,
                            "tags": _bounded_knowledge_tags(document.tags),
                            "chunk_count": document.chunk_total,
                        }
                        for document in selected_documents
                    ],
                    "total_documents": total_documents,
                    "returned_documents": len(selected_documents),
                    "has_more_documents": len(selected_documents) < total_documents,
                }
            )
        resolved_names = set(
            resolve_agent_tools(agent, workspace=workspace).tool_names()
        )
        available_retrieval_tools = [
            name for name in KNOWLEDGE_RETRIEVAL_TOOL_NAMES if name in resolved_names
        ]
        return {
            "collections": [collection["name"] for collection in collection_data],
            "collection_indexes": collection_data,
            "documents": [],
            "text": "",
            "truncated": False,
            "max_chars": max_chars,
            "retrieval_required": bool(total_collections),
            "available_retrieval_tools": available_retrieval_tools,
            "retrieval_available": bool(available_retrieval_tools),
            "total_collections": total_collections,
            "index_truncated": index_truncated,
            "index_limits": {
                "collections": KNOWLEDGE_PROMPT_MAX_COLLECTIONS,
                "documents_per_collection": KNOWLEDGE_PROMPT_MAX_DOCUMENTS_PER_COLLECTION,
            },
        }

    for collection in collections:
        active_documents = collection.documents.filter(status="active").order_by("title")
        for document in active_documents:
            text = read_document_text(document).strip()
            if not text:
                continue
            if max_chars and remaining <= 0:
                break
            selected_text = text[:remaining] if max_chars else text
            if max_chars:
                remaining -= len(selected_text)
            documents.append(
                {
                    "collection": collection.name,
                    "title": document.title,
                    "language": document.language,
                    "tags": document.tags,
                    "content": selected_text,
                }
            )

    text_blocks = [
        f"[{doc['collection']} / {doc['title']}]\n{doc['content']}"
        for doc in documents
    ]
    return {
        "collections": [collection.name for collection in collections],
        "documents": documents,
        "text": "\n\n".join(text_blocks),
        "truncated": bool(max_chars and remaining <= 0),
        "max_chars": max_chars,
    }


def prepare_agent_payload(agent: AgentProfile, context: dict, mapping: dict, *, workspace=None) -> dict:
    require_active_agent(agent)
    payload = apply_mapping(context, mapping or {})
    payload["knowledge_context"] = build_agent_knowledge_context(agent, workspace=workspace)
    return payload


# Matches <think>…</think> and <thinking>…</thinking> blocks emitted by reasoning models
# such as qwen3 and deepseek-r1 before their final answer.
_THINKING_BLOCK_RE = re.compile(r"<think(?:ing)?[^>]*>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _extract_json_object_text(value: str) -> str:
    stripped = value.strip()
    # Strip model-specific thinking blocks before looking for JSON so that any
    # curly braces inside a reasoning preamble don't confuse the extractor.
    stripped = _THINKING_BLOCK_RE.sub("", stripped).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and last > first:
        return stripped[first:last + 1]
    return stripped


def _decode_agent_tool_decision(llm_result: dict) -> dict:
    content = (llm_result or {}).get("content", "")
    if isinstance(content, dict):
        decision = content
    elif isinstance(content, str):
        try:
            decision = json.loads(_extract_json_object_text(content))
        except Exception as exc:
            # Fix #3: catch broadly — json.JSONDecodeError is ValueError but other edge
            # cases (encoding issues, unexpected extractor output) can raise different types.
            raise ValidationError(f"Agent tool response must be valid JSON: {exc}") from exc
    else:
        raise ValidationError("Agent tool response content must be a JSON object or JSON string.")

    if not isinstance(decision, dict):
        raise ValidationError("Agent tool response must decode to a JSON object.")
    response_type = str(decision.get("type") or "").strip()
    if response_type not in {"final", "tool_call"}:
        raise ValidationError("Agent tool response type must be 'final' or 'tool_call'.")
    if response_type == "final" and not isinstance(decision.get("answer", ""), str):
        raise ValidationError("Agent final response must include a string 'answer'.")
    if response_type == "tool_call":
        if not isinstance(decision.get("tool_name"), str) or not decision.get("tool_name").strip():
            raise ValidationError("Agent tool_call response must include 'tool_name'.")
        if not isinstance(decision.get("arguments"), dict):
            raise ValidationError("Agent tool_call response must include object 'arguments'.")
    return decision


def _plain_final_decision(llm_result: dict) -> dict | None:
    """Adapt existing non-tool model responses at the runner boundary.

    A response that already declares ``type`` is never adapted: malformed tool
    protocol must remain an error instead of being mistaken for a final answer.
    """
    content = (llm_result or {}).get("content", "")
    if isinstance(content, dict):
        if "type" in content:
            return None
        answer = json.dumps(content, default=str)
    elif isinstance(content, str):
        try:
            decoded = json.loads(_extract_json_object_text(content))
        except Exception:
            decoded = None
        if isinstance(decoded, dict) and "type" in decoded:
            return None
        answer = content
    else:
        return None
    return {"type": "final", "answer": answer}


def _safe_tool_error(exc: Exception, tool_name: str) -> str:
    # Fix #6: ValidationError messages are already human-readable and safe to surface
    # (they describe contract violations, not internal state). All other exceptions
    # get a generic message so internal paths, API URLs, or stack fragments never
    # reach the model or the API caller.
    if isinstance(exc, ValidationError):
        return str(exc)
    return f"Tool '{tool_name}' failed during execution."


def _bounded_tool_observation(tool_name: str, result: dict) -> dict:
    try:
        max_chars = int(
            getattr(
                settings,
                "AI_HUB_MAX_TOOL_OBSERVATION_CHARS",
                DEFAULT_MAX_TOOL_OBSERVATION_CHARS,
            )
        )
    except (TypeError, ValueError):
        max_chars = DEFAULT_MAX_TOOL_OBSERVATION_CHARS
    max_chars = min(max(max_chars, 256), 100000)
    serialized = json.dumps(result, sort_keys=True)
    if len(serialized) <= max_chars:
        return result

    marker = {
        "truncated": True,
        "tool_name": tool_name,
        "original_chars": len(serialized),
    }
    preview_size = max_chars
    while preview_size > 0:
        marker["json_preview"] = serialized[:preview_size]
        marker_size = len(json.dumps(marker, sort_keys=True))
        if marker_size <= max_chars:
            return marker
        preview_size -= max(marker_size - max_chars, 1)
    marker.pop("json_preview", None)
    return marker


def _structured_tool_system_prompt(agent: AgentProfile) -> str:
    base_prompt = agent.system_prompt or ""
    instructions = (
        "\n\nTool response contract:\n"
        "Return only one JSON object. Do not wrap it in Markdown.\n"
        "To finish: {\"type\":\"final\",\"answer\":\"...\"}\n"
        "To call one tool: {\"type\":\"tool_call\",\"tool_name\":\"tool_name\",\"arguments\":{}}\n"
        "Call at most one tool at a time. Use only listed tool names."
    )
    return f"{base_prompt}{instructions}"


def _execution_context_value(execution_context, key: str):
    if isinstance(execution_context, dict):
        return execution_context.get(key)
    return getattr(execution_context, key, None)


def _create_tool_run(
    *,
    execution_context,
    agent: AgentProfile,
    resolved_tool,
    input_payload: dict,
    status: str,
    approval_state: str,
    error_detail: str = "",
):
    return ToolExecutionRun.objects.create(
        session=_execution_context_value(execution_context, "session"),
        step_run=_execution_context_value(execution_context, "step_run"),
        agent=agent,
        tool=resolved_tool.tool,
        status=status,
        input_payload=input_payload,
        risk_level=resolved_tool.tool.risk_level,
        approval_state=approval_state,
        error_detail=error_detail,
    )


def execute_agent_deliberate(
    agent: AgentProfile,
    payload: dict,
    *,
    workspace=None,
    execution_context=None,
    max_tool_rounds: int | None = None,
    tool_policy: str = TOOL_POLICY_ALL,
    allow_plain_final: bool = False,
    unwrap_final_answer: bool = False,
    allow_approval_requests: bool = True,
) -> dict:
    require_active_agent(agent)
    validate_payload(payload, agent.input_contract or {}, f"Agent '{agent.name}' input")
    model_cfg = resolve_model_config(agent.model_config)
    resolution = resolve_agent_tools(agent, workspace=workspace, execution_context=execution_context)
    if tool_policy not in {TOOL_POLICY_ALL, TOOL_POLICY_GAME_CONTEXT_ONLY}:
        raise ValidationError(f"Unknown tool execution policy '{tool_policy}'.")
    resolved_tools = resolution.tools
    if tool_policy == TOOL_POLICY_GAME_CONTEXT_ONLY:
        resolved_tools = tuple(
            resolved
            for resolved in resolved_tools
            if get_game_tool_category(resolved.tool) == GAME_CONTEXT_TOOL
        )
    if not allow_approval_requests:
        resolved_tools = tuple(
            resolved for resolved in resolved_tools if not resolved.requires_approval
        )
    tool_manifest = [resolved.manifest() for resolved in resolved_tools]
    tools_by_name = {resolved.tool.name: resolved for resolved in resolved_tools}
    max_rounds = max_tool_rounds
    if max_rounds is None:
        max_rounds = int(getattr(settings, "AI_HUB_MAX_TOOL_ROUNDS_PER_AGENT_CALL", 3))
    max_rounds = min(max(max_rounds, 0), 10)

    messages = [
        {"role": "system", "content": _structured_tool_system_prompt(agent)},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "context": payload,
                    "available_tools": tool_manifest,
                },
                default=str,
            ),
        },
    ]
    tools_data = {}
    tool_run_ids = []
    last_llm_result = {}

    for tool_round in range(max_rounds + 1):
        response_mode = "tool_protocol"
        last_llm_result = completion_call(
            provider_type=model_cfg["provider_type"],
            model=model_cfg["model"],
            messages=messages,
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            timeout=model_cfg["timeout"],
            temperature=model_cfg["temperature"],
            max_tokens=model_cfg["max_tokens"],
        )
        try:
            decision = _decode_agent_tool_decision(last_llm_result)
        except ValidationError as exc:
            compatible_decision = _plain_final_decision(last_llm_result) if allow_plain_final else None
            if compatible_decision is not None:
                decision = compatible_decision
                response_mode = "plain_final_compatibility"
            elif tool_round < max_rounds:
                # Retry: inject the raw response as an assistant turn so the model
                # sees its own output, then add a one-line contract correction.
                raw_content = (last_llm_result or {}).get("content", "")
                messages.append({"role": "assistant", "content": raw_content or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. "
                        "Return only a single JSON object with no other text:\n"
                        '{"type":"final","answer":"your answer here"}\n'
                        "or\n"
                        '{"type":"tool_call","tool_name":"exact_tool_name","arguments":{}}'
                    ),
                })
                continue
            else:
                output_payload = {
                    "agent": agent.name,
                    "tools": tools_data,
                    "tool_runs": tool_run_ids,
                    "tool_manifest": tool_manifest,
                    "llm": last_llm_result,
                    "status": "invalid_model_response",
                    "error": str(exc),
                }
                validate_payload(output_payload, agent.output_contract or {}, f"Agent '{agent.name}' output")
                return output_payload

        if decision["type"] == "final":
            llm_payload = last_llm_result
            raw_tool_protocol_llm = None
            if unwrap_final_answer and response_mode == "tool_protocol":
                raw_tool_protocol_llm = last_llm_result
                llm_payload = {
                    **last_llm_result,
                    "content": decision.get("answer", ""),
                }
            output_payload = {
                "agent": agent.name,
                "tools": tools_data,
                "tool_runs": tool_run_ids,
                "tool_manifest": tool_manifest,
                "llm": llm_payload,
                "status": "final",
                "final_answer": decision.get("answer", ""),
                "model_response_mode": response_mode,
            }
            if raw_tool_protocol_llm is not None:
                output_payload["tool_protocol_llm"] = raw_tool_protocol_llm
            validate_payload(output_payload, agent.output_contract or {}, f"Agent '{agent.name}' output")
            return output_payload

        if tool_round >= max_rounds:
            output_payload = {
                "agent": agent.name,
                "tools": tools_data,
                "tool_runs": tool_run_ids,
                "tool_manifest": tool_manifest,
                "llm": last_llm_result,
                "status": "max_tool_rounds",
                "error": f"Agent reached max tool rounds ({max_rounds}).",
            }
            validate_payload(output_payload, agent.output_contract or {}, f"Agent '{agent.name}' output")
            return output_payload

        tool_name = decision["tool_name"].strip()
        resolved_tool = tools_by_name.get(tool_name)
        if resolved_tool is None:
            output_payload = {
                "agent": agent.name,
                "tools": tools_data,
                "tool_runs": tool_run_ids,
                "tool_manifest": tool_manifest,
                "llm": last_llm_result,
                "status": "unauthorised_tool",
                "error": f"Tool '{tool_name}' is not available to agent '{agent.name}'.",
            }
            validate_payload(output_payload, agent.output_contract or {}, f"Agent '{agent.name}' output")
            return output_payload

        arguments = bind_tool_runtime_context(
            resolved_tool.tool,
            decision["arguments"],
            agent=agent,
        )
        if resolved_tool.requires_approval:
            tool_run = _create_tool_run(
                execution_context=execution_context,
                agent=agent,
                resolved_tool=resolved_tool,
                input_payload=arguments,
                status=ToolExecutionRun.Status.WAITING_APPROVAL,
                approval_state=ToolExecutionRun.ApprovalState.REQUIRED,
            )
            tool_run_ids.append(tool_run.pk)
            output_payload = {
                "agent": agent.name,
                "tools": tools_data,
                "tool_runs": tool_run_ids,
                "tool_manifest": tool_manifest,
                "llm": last_llm_result,
                "status": "waiting_approval",
                "requested_tool": tool_name,
            }
            validate_payload(output_payload, agent.output_contract or {}, f"Agent '{agent.name}' output")
            return output_payload

        tool_run = _create_tool_run(
            execution_context=execution_context,
            agent=agent,
            resolved_tool=resolved_tool,
            input_payload=arguments,
            status=ToolExecutionRun.Status.RUNNING,
            approval_state=ToolExecutionRun.ApprovalState.NOT_REQUIRED,
        )
        tool_run.started_at = timezone.now()
        tool_run.save(update_fields=["started_at"])
        start = time.perf_counter()
        try:
            tool_result = execute_tool(resolved_tool.tool, arguments, agent=agent)
            tool_run.status = ToolExecutionRun.Status.SUCCESS
            tool_run.output_payload = tool_result
            prompt_observation = _bounded_tool_observation(tool_name, tool_result)
            tools_data[tool_name] = prompt_observation
            messages.append({"role": "assistant", "content": json.dumps(decision, default=str)})
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "tool_observation": {
                                "tool_name": tool_name,
                                "result": prompt_observation,
                            }
                        },
                        default=str,
                    ),
                }
            )
        except Exception as exc:
            tool_run.status = ToolExecutionRun.Status.FAILED
            tool_run.error_detail = str(exc)  # Full detail stays in DB only
            tool_run.finished_at = timezone.now()
            tool_run.latency_ms = int((time.perf_counter() - start) * 1000)
            tool_run.save(update_fields=["status", "error_detail", "finished_at", "latency_ms"])
            tool_run_ids.append(tool_run.pk)
            output_payload = {
                "agent": agent.name,
                "tools": tools_data,
                "tool_runs": tool_run_ids,
                "tool_manifest": tool_manifest,
                "llm": last_llm_result,
                "status": "tool_error",
                "requested_tool": tool_name,
                "error": _safe_tool_error(exc, tool_name),  # Sanitized for caller / model
            }
            validate_payload(output_payload, agent.output_contract or {}, f"Agent '{agent.name}' output")
            return output_payload
        tool_run.finished_at = timezone.now()
        tool_run.latency_ms = int((time.perf_counter() - start) * 1000)
        tool_run.save(update_fields=["status", "output_payload", "finished_at", "latency_ms"])
        tool_run_ids.append(tool_run.pk)

    output_payload = {
        "agent": agent.name,
        "tools": tools_data,
        "tool_runs": tool_run_ids,
        "tool_manifest": tool_manifest,
        "llm": last_llm_result,
        "status": "max_tool_rounds",
        "error": f"Agent reached max tool rounds ({max_rounds}).",
    }
    validate_payload(output_payload, agent.output_contract or {}, f"Agent '{agent.name}' output")
    return output_payload


def execute_agent(
    agent: AgentProfile,
    payload: dict,
    *,
    tool_policy: str = TOOL_POLICY_ALL,
    workspace=None,
) -> dict:
    require_active_agent(agent)
    validate_payload(payload, agent.input_contract or {}, f"Agent '{agent.name}' input")
    model_cfg = resolve_model_config(agent.model_config)
    direct_tool_ids = set(
        agent.tools.filter(is_active=True).values_list("pk", flat=True)
    )
    resolution = resolve_agent_tools(agent, workspace=workspace)
    executable_direct_tools = [
        resolved.tool
        for resolved in resolution.tools
        if resolved.tool.pk in direct_tool_ids and not resolved.requires_approval
    ]
    tools_data = execute_tools(
        executable_direct_tools,
        payload,
        policy=tool_policy,
        agent=agent,
    )

    # Include tool results in the same LLM call so the agent can reason about them immediately.
    # Without this, tool output only appears in previous_response on the *next* iteration.
    user_content: dict = {"context": payload}
    if tools_data:
        user_content["tool_results"] = {
            tool_name: _bounded_tool_observation(tool_name, tool_result)
            for tool_name, tool_result in tools_data.items()
        }
    user_message = json.dumps(user_content, default=str)

    llm_result = completion_call(
        provider_type=model_cfg["provider_type"],
        model=model_cfg["model"],
        messages=[
            {"role": "system", "content": agent.system_prompt or ""},
            {"role": "user", "content": user_message},
        ],
        api_key=model_cfg["api_key"],
        base_url=model_cfg["base_url"],
        timeout=model_cfg["timeout"],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
    )
    output_payload = {
        "agent": agent.name,
        "tools": tools_data,
        "llm": llm_result,
    }
    validate_payload(output_payload, agent.output_contract or {}, f"Agent '{agent.name}' output")
    return output_payload
