import json
import os
import time
import urllib.error
import urllib.request

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import permission_required
from django.core.cache import cache
from django.shortcuts import get_object_or_404, render

from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    ExecutionStepRun,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    ProviderConfig,
    ToolDefinition,
)
from ai_hub.services.admin_control_center import validate_provider_health_endpoint
from ai_hub.services.game_operational_ux import redact_payload, redact_text


@staff_member_required
@permission_required("ai_hub.view_executionsession", raise_exception=True)
def overview(request):
    """Control room: counts + recent sessions."""
    context = {
        "provider_count": ProviderConfig.objects.count(),
        "model_count": ModelConfig.objects.count(),
        "agent_count": AgentProfile.objects.count(),
        "tool_count": ToolDefinition.objects.count(),
        "pipeline_count": PipelineDefinition.objects.count(),
        "session_count": ExecutionSession.objects.count(),
        "recent_sessions": (
            ExecutionSession.objects.select_related("entry_agent")
            .order_by("-created_at")[:8]
        ),
    }
    return render(request, "dashboard/overview.html", context)


@staff_member_required
@permission_required("ai_hub.view_executionsession", raise_exception=True)
def providers(request):
    """List all ProviderConfigs with their ModelConfigs."""
    provider_list = ProviderConfig.objects.prefetch_related("models").order_by("name")
    return render(request, "dashboard/providers.html", {"provider_list": provider_list})


@staff_member_required
@permission_required("ai_hub.view_executionsession", raise_exception=True)
def provider_detail(request, pk):
    """One provider + its models + agents that use those models."""
    provider = get_object_or_404(ProviderConfig, pk=pk)
    models_qs = provider.models.prefetch_related("agents").order_by("model_name")
    return render(
        request,
        "dashboard/provider_detail.html",
        {"provider": provider, "models_qs": models_qs},
    )


@staff_member_required
@permission_required("ai_hub.view_executionsession", raise_exception=True)
def agents(request):
    """All AgentProfiles with related model/provider/tools."""
    agent_list = (
        AgentProfile.objects.select_related("model_config__provider")
        .prefetch_related("tools")
        .order_by("name")
    )
    return render(request, "dashboard/agents.html", {"agent_list": agent_list})


@staff_member_required
@permission_required("ai_hub.view_executionsession", raise_exception=True)
def agent_detail(request, pk):
    """Full agent config + recent 5 sessions."""
    agent = get_object_or_404(
        AgentProfile.objects.select_related("model_config__provider").prefetch_related("tools"),
        pk=pk,
    )
    recent_sessions = (
        ExecutionSession.objects.filter(entry_agent=agent)
        .order_by("-created_at")[:5]
    )
    return render(
        request,
        "dashboard/agent_detail.html",
        {"agent": agent, "recent_sessions": recent_sessions},
    )


@staff_member_required
@permission_required("ai_hub.view_executionsession", raise_exception=True)
def pipelines(request):
    """All PipelineDefinitions with their steps."""
    pipeline_list = (
        PipelineDefinition.objects.select_related("entry_agent")
        .prefetch_related("steps__agent__model_config__provider")
        .order_by("name")
    )
    return render(request, "dashboard/pipelines.html", {"pipeline_list": pipeline_list})


@staff_member_required
@permission_required("ai_hub.view_executionsession", raise_exception=True)
def pipeline_detail(request, pk):
    """One pipeline + ordered steps."""
    pipeline = get_object_or_404(PipelineDefinition, pk=pk)
    steps = (
        pipeline.steps.select_related("agent__model_config__provider")
        .order_by("order")
    )
    return render(
        request,
        "dashboard/pipeline_detail.html",
        {"pipeline": pipeline, "steps": steps},
    )


@staff_member_required
@permission_required("ai_hub.view_executionsession", raise_exception=True)
def sessions(request):
    """All sessions ordered by -created_at, optional ?status= filter, limited 50."""
    valid_statuses = {c[0] for c in ExecutionSession.Status.choices}
    raw_filter = request.GET.get("status", "").strip()
    status_filter = raw_filter if raw_filter in valid_statuses else ""
    qs = (
        ExecutionSession.objects.select_related("entry_agent")
        .prefetch_related("step_runs")
        .order_by("-created_at")
    )
    if status_filter:
        qs = qs.filter(status=status_filter)
    session_list = qs[:50]
    return render(
        request,
        "dashboard/sessions.html",
        {"session_list": session_list, "status_filter": status_filter},
    )


@staff_member_required
@permission_required("ai_hub.view_executionsession", raise_exception=True)
def session_detail(request, pk):
    """Session + step_runs with parsed GAME payloads."""
    session = get_object_or_404(
        ExecutionSession.objects.select_related("entry_agent", "pipeline", "triggered_by"),
        pk=pk,
    )
    step_runs = session.step_runs.select_related("agent").order_by("order")

    step_data = []
    for step in step_runs:
        obs = redact_payload(step.observation_payload or {})
        resp = redact_payload(step.response_payload or {})
        decision = obs.get("decision", {})
        tools_output = resp.get("tools") or {}
        llm = resp.get("llm") or {}
        step_data.append(
            {
                "step": step,
                "action": obs.get("action", ""),
                "complete": obs.get("complete", False),
                "final_answer": obs.get("final_answer", ""),
                "message": redact_text(decision.get("message", "")),
                "tools_output": tools_output,
                "llm_content": _redact_runtime_value(llm.get("content", "")),
                "llm_model": llm.get("model", ""),
                "contract_valid": obs.get("contract_valid", True),
                "contract_errors": obs.get("contract_errors", []),
                "request_json": (
                    json.dumps(
                        redact_payload(step.request_payload),
                        indent=2,
                        default=str,
                    )
                    if step.request_payload
                    else ""
                ),
            }
        )

    final_answer = redact_text(
        (session.final_context or {}).get("final_answer", "")
    )
    if not final_answer and step_data:
        final_answer = redact_text(step_data[-1].get("final_answer", ""))

    return render(
        request,
        "dashboard/session_detail.html",
        {
            "session": session,
            "step_data": step_data,
            "final_answer": final_answer,
            "goal_text_redacted": redact_text(session.goal_text),
            "error_detail_redacted": redact_text(session.error_detail),
            "initial_context_redacted": redact_payload(session.initial_context),
        },
    )


@staff_member_required
@permission_required("ai_hub.view_executionsession", raise_exception=True)
def api_status(request):
    """Live provider status: ping Ollama, list models + capabilities. Cached 30 s."""
    cached = cache.get("dashboard_api_status")
    if cached is not None:
        return cached

    providers = ProviderConfig.objects.prefetch_related("models").order_by("name")
    provider_data = []

    for provider in providers:
        entry = {
            "provider": provider,
            "status": "unknown",
            "latency_ms": None,
            "error": "",
            "live_models": [],
            "db_models": list(provider.models.filter(is_active=True).order_by("model_name")),
        }

        if provider.provider_type == ProviderConfig.ProviderType.TRAINING:
            entry["status"] = "ready"

        elif provider.provider_type == ProviderConfig.ProviderType.OLLAMA and provider.base_url:
            endpoint_ok, endpoint_reason = validate_provider_health_endpoint(
                provider.base_url
            )
            if not endpoint_ok:
                entry["status"] = "error"
                entry["error"] = endpoint_reason
            else:
                url = f"{provider.base_url.rstrip('/')}/api/tags"
                try:
                    t0 = time.monotonic()
                    req = urllib.request.Request(
                        url,
                        headers={"Accept": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=4) as resp:
                        entry["latency_ms"] = round(
                            (time.monotonic() - t0) * 1000
                        )
                        data = json.loads(resp.read())
                    entry["status"] = "connected"
                    for m in data.get("models", []):
                        details = m.get("details", {})
                        caps = m.get("capabilities") or _infer_capabilities(
                            m.get("name", ""),
                            details.get("families")
                            or [details.get("family", "")],
                        )
                        size_bytes = m.get("size", 0)
                        entry["live_models"].append({
                            "name": m.get("name", ""),
                            "size": _fmt_size(size_bytes),
                            "params": details.get("parameter_size", ""),
                            "quant": details.get("quantization_level", ""),
                            "families": ", ".join(
                                family
                                for family in (
                                    details.get("families")
                                    or [details.get("family", "")]
                                )
                                if family
                            ),
                            "capabilities": caps,
                        })
                except urllib.error.URLError as exc:
                    entry["status"] = "error"
                    entry["error"] = redact_text(exc.reason)
                except Exception as exc:
                    entry["status"] = "error"
                    entry["error"] = redact_text(exc)

        else:
            key_var = provider.api_key_env_var or ""
            if not key_var or os.environ.get(key_var):
                entry["status"] = "configured"
            else:
                entry["status"] = "missing_key"
                entry["error"] = f"Env var {key_var!r} not found in environment"

        provider_data.append(entry)

    response = render(request, "dashboard/api_status.html", {"provider_data": provider_data})
    cache.set("dashboard_api_status", response, 30)
    return response


# ── helpers (not views) ────────────────────────────────────


def _redact_runtime_value(value):
    """Redact structured JSON strings as well as ordinary runtime text."""
    if isinstance(value, (dict, list)):
        return redact_payload(value)
    if isinstance(value, str):
        try:
            return redact_payload(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return redact_text(value)
    return value


def _fmt_size(size_bytes):
    if not size_bytes:
        return ""
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def _infer_capabilities(name, families):
    name_lower = name.lower()
    family_str = " ".join(f.lower() for f in families if f)

    if any(tok in family_str for tok in ("bert", "nomic")):
        return ["embedding"]

    caps = ["completion"]

    if "mllama" in family_str or "llava" in family_str or "vision" in name_lower:
        caps.append("vision")

    # tool-capable families
    if any(tok in family_str for tok in ("llama", "mistral", "gemma")):
        caps.append("tools")
    elif "qwen3" in family_str:
        caps.append("tools")
    elif "qwen2" in family_str and "coder" in name_lower:
        if "base" not in name_lower:
            caps.append("tools")

    if "qwen3" in family_str or "deepseek-r1" in name_lower:
        caps.append("thinking")

    if "coder" in name_lower or "code" in name_lower:
        caps.append("insert")

    return caps
