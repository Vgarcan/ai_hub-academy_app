from __future__ import annotations

from dataclasses import dataclass

import requests
from django.core.cache import cache
from django.db.models import Avg, Count, Max, Prefetch, Q
from django.urls import reverse

from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    ExecutionStepRun,
    GameActionApprovalRequest,
    GameContinuationRequest,
    GameGoal,
    KnowledgeCollection,
    KnowledgeDocument,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    ProviderConfig,
    ToolDefinition,
)

PROVIDER_HEALTH_CACHE_SECONDS = 45

NODE_KIND_LEGEND = [
    {"kind": "provider", "label": "Providers", "description": "AI services and local endpoints."},
    {"kind": "model", "label": "Models", "description": "Configured model names and generation defaults."},
    {"kind": "knowledge", "label": "Knowledge", "description": "Curated context collections available to agents."},
    {"kind": "tool", "label": "Tools", "description": "Optional capabilities agents may call."},
    {"kind": "agent", "label": "Agents", "description": "Specialized prompts, contracts and model choices."},
    {"kind": "pipeline", "label": "Pipelines", "description": "Reusable recipes that connect ordered steps."},
    {"kind": "step", "label": "Steps", "description": "Concrete agent calls in pipeline order."},
]

GAME_KIND_LEGEND = [
    {"kind": "goal", "label": "Goal", "description": "The objective the autonomous session is trying to complete."},
    {"kind": "agent", "label": "Agent", "description": "The entry agent making decisions."},
    {"kind": "decision", "label": "Decision", "description": "One model decision inside the loop."},
    {"kind": "action", "label": "Action", "description": "The action selected by the agent."},
    {"kind": "memory", "label": "Memory", "description": "Context, observations and state carried forward."},
    {"kind": "stop", "label": "Stop", "description": "The finish condition or final answer."},
]

# Models hidden from the index via AIHubHideFromIndexMixin.
# category, admin_model_name, display_label, why_hidden
_HIDDEN_MODEL_CATALOG = [
    ("Bridge tables", "knowledgedocumentchunk", "Knowledge document chunks", "Managed inside their parent document. Direct edits bypass chunking logic."),
    ("Bridge tables", "toolboxtool", "Toolbox → tool assignments", "Managed via the Toolbox record."),
    ("Bridge tables", "agenttoolboxassignment", "Agent → toolbox assignments", "Managed via Agent Composer (Tools tab)."),
    ("Bridge tables", "agenttoolgrant", "Agent → direct tool grants", "Managed via Agent Composer (Tools tab)."),
    ("Bridge tables", "pipelinestep", "Pipeline steps", "Managed inside the Pipeline Designer (Steps tab)."),
    ("GAME structural children", "gamegoaldependency", "Goal dependencies", "Managed inside the Goal detail (Config tab)."),
    ("GAME structural children", "gamegoalplan", "Goal plans", "Auto-created by the GAME planning engine."),
    ("GAME structural children", "gamegoalplanstep", "Goal plan steps", "Auto-created by the GAME planning engine."),
    ("GAME structural children", "gameworkspaceaction", "Workspace actions", "Managed inside the Workspace record."),
    ("GAME structural children", "gameworkspaceagent", "Workspace agents", "Managed inside the Workspace record."),
    ("Runtime / audit", "gamememoryentry", "Memory entries", "Written by the GAME engine each loop iteration."),
    ("Runtime / audit", "executionsteprun", "Execution step runs", "Written by the engine. Inspect via the Session Explorer Timeline tab."),
    ("Runtime / audit", "toolexecutionrun", "Tool execution runs", "Written by the engine on each tool call."),
    ("Runtime / audit", "gameactionrun", "GAME action runs", "Written by the engine per GAME action."),
    ("Runtime / audit", "gamedelegationrun", "GAME delegation runs", "Written by the engine when an agent delegates."),
]

EXAMPLE_TEMPLATES = [
    {
        "title": "Dream interpretation workflow",
        "workspace": "Orchestrator",
        "description": "A fixed sequence that extracts facts, symbols, themes and writes a final interpretation.",
        "steps": "Source reader -> signal extractors -> interpretation writer -> persistence adapter",
    },
    {
        "title": "Research assistant",
        "workspace": "Orchestrator",
        "description": "A predictable research flow that reads context, checks evidence and produces a short brief.",
        "steps": "Question parser -> document reader -> evidence checker -> brief writer",
    },
    {
        "title": "Document analyst",
        "workspace": "Orchestrator",
        "description": "A reusable workflow for classifying, summarizing and extracting structured fields.",
        "steps": "Classifier -> extractor -> risk reviewer -> summary writer",
    },
    {
        "title": "Autonomous planner",
        "workspace": "GAME",
        "description": "A goal-driven agent that decides the next action until the task is complete or stopped.",
        "steps": "Goal -> decision loop -> action/observation -> memory -> final answer",
    },
]


@dataclass(frozen=True)
class ProviderHealth:
    status: str
    detail: str
    models: set[str]


def _admin_url(model_name: str, object_id: int) -> str:
    return reverse(f"admin:ai_hub_{model_name}_change", args=[object_id])


def _admin_changelist_url(model_name: str) -> str:
    return reverse(f"admin:ai_hub_{model_name}_changelist")


def _admin_add_url(model_name: str) -> str:
    return reverse(f"admin:ai_hub_{model_name}_add")


def _node(node_id: str, label: str, kind: str, status: str, detail: str = "", url: str = "", meta: dict | None = None):
    return {
        "id": node_id,
        "label": label,
        "kind": kind,
        "status": status,
        "detail": detail,
        "url": url,
        "meta": meta or {},
    }


def _edge(source: str, target: str, label: str, status: str = "ok"):
    return {
        "source": source,
        "target": target,
        "label": label,
        "status": status,
    }


def _status_summary(nodes: list[dict]) -> dict:
    summary = {"ok": 0, "warning": 0, "error": 0, "inactive": 0, "unknown": 0}
    for node in nodes:
        summary[node["status"] if node["status"] in summary else "unknown"] += 1
    return summary


def _node_trace(node: dict, edges: list[dict], nodes_by_id: dict[str, dict]) -> dict:
    incoming = [edge for edge in edges if edge["target"] == node["id"]]
    outgoing = [edge for edge in edges if edge["source"] == node["id"]]
    return {
        "incoming": [
            {
                "label": edge["label"],
                "node": nodes_by_id[edge["source"]]["label"],
                "kind": nodes_by_id[edge["source"]]["kind"],
                "status": edge["status"],
            }
            for edge in incoming
            if edge["source"] in nodes_by_id
        ],
        "outgoing": [
            {
                "label": edge["label"],
                "node": nodes_by_id[edge["target"]]["label"],
                "kind": nodes_by_id[edge["target"]]["kind"],
                "status": edge["status"],
            }
            for edge in outgoing
            if edge["target"] in nodes_by_id
        ],
    }


def _connected_node_summary(nodes: list[dict], edges: list[dict]) -> list[dict]:
    node_counts = {
        node["id"]: {"node": node, "incoming": 0, "outgoing": 0, "total": 0}
        for node in nodes
    }
    for edge in edges:
        if edge["source"] in node_counts:
            node_counts[edge["source"]]["outgoing"] += 1
            node_counts[edge["source"]]["total"] += 1
        if edge["target"] in node_counts:
            node_counts[edge["target"]]["incoming"] += 1
            node_counts[edge["target"]]["total"] += 1
    return [
        {
            "label": item["node"]["label"],
            "kind": item["node"]["kind"],
            "status": item["node"]["status"],
            "incoming": item["incoming"],
            "outgoing": item["outgoing"],
            "total": item["total"],
            "url": item["node"]["url"],
        }
        for item in sorted(node_counts.values(), key=lambda row: row["total"], reverse=True)[:8]
        if item["total"]
    ]


def _checklist_item(label: str, ok: bool, detail: str, action: str = "") -> dict:
    return {
        "label": label,
        "status": "ok" if ok else "warning",
        "detail": detail,
        "action": action,
    }


def _action(label: str, detail: str, url: str, button: str) -> dict:
    return {"label": label, "detail": detail, "url": url, "button": button}


def _attention_item(
    item_id: str,
    severity: str,
    title: str,
    detail: str,
    url: str = "",
    occurred_at=None,
    source: str = "config",
    relevance: int = 1,
    hover: str = "",
) -> dict:
    return {
        "id": item_id,
        "severity": severity,
        "title": title,
        "detail": detail,
        "url": url,
        "occurred_at": occurred_at,
        "source": source,
        "relevance": relevance,
        "hover": hover or detail,
    }


def _ollama_model_name(model_name: str) -> str:
    return model_name.removeprefix("ollama/")


def _provider_health_cache_key(provider: ProviderConfig) -> str:
    updated_at = provider.updated_at.isoformat() if provider.updated_at else "new"
    return f"ai_hub:provider_health:{provider.pk}:{updated_at}"


def _fetch_provider_health(provider: ProviderConfig) -> ProviderHealth:
    if not provider.is_active:
        return ProviderHealth("inactive", "Provider inactive", set())
    if provider.provider_type != ProviderConfig.ProviderType.OLLAMA:
        return ProviderHealth("unknown", "Live checks are enabled for Ollama providers only.", set())
    if not provider.base_url:
        return ProviderHealth("warning", "Ollama provider has no base_url.", set())

    try:
        response = requests.get(f"{provider.base_url.rstrip('/')}/api/tags", timeout=1.5)
        response.raise_for_status()
        payload = response.json()
    except (ValueError, requests.RequestException) as exc:
        return ProviderHealth("error", str(exc), set())
    if not isinstance(payload, dict):
        return ProviderHealth("error", "Provider health response must be a JSON object.", set())

    models = {item.get("name", "") for item in payload.get("models", []) if item.get("name")}
    return ProviderHealth("ok", f"{len(models)} installed models detected.", models)


def _provider_health(provider: ProviderConfig) -> ProviderHealth:
    cache_key = _provider_health_cache_key(provider)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    health = _fetch_provider_health(provider)
    cache.set(cache_key, health, PROVIDER_HEALTH_CACHE_SECONDS)
    return health


def build_ai_hub_home_context() -> dict:
    provider_count = ProviderConfig.objects.count()
    active_provider_count = ProviderConfig.objects.filter(is_active=True).count()
    model_count = ModelConfig.objects.count()
    active_model_count = ModelConfig.objects.filter(is_active=True, provider__is_active=True).count()
    agent_count = AgentProfile.objects.count()
    active_agent_count = AgentProfile.objects.filter(is_active=True).count()
    active_document_count = KnowledgeDocument.objects.filter(
        status=KnowledgeDocument.Status.ACTIVE,
        collection__is_active=True,
    ).count()
    active_tool_count = ToolDefinition.objects.filter(is_active=True).count()
    active_pipeline_count = PipelineDefinition.objects.filter(is_active=True).count()
    game_session_count = ExecutionSession.objects.filter(runtime_kind=ExecutionSession.RuntimeKind.GAME).count()
    recent_session_count = ExecutionSession.objects.count()

    # ── Action queue: things needing human intervention ──────────────
    pending_approvals = list(
        GameActionApprovalRequest.objects
        .filter(status=GameActionApprovalRequest.Status.PENDING)
        .select_related("action_run", "goal")
        .order_by("created_at")[:5]
    )
    pending_continuations = list(
        GameContinuationRequest.objects
        .filter(status=GameContinuationRequest.Status.PENDING)
        .select_related("goal")
        .order_by("created_at")[:5]
    )
    recent_failed = list(
        ExecutionSession.objects
        .filter(status=ExecutionSession.Status.FAILED)
        .order_by("-created_at")[:3]
    )
    action_queue = []
    for ap in pending_approvals:
        goal_title = ap.goal.title if ap.goal else "–"
        action_name = ap.action_run.action_name if ap.action_run else "–"
        action_queue.append({
            "kind": "approval",
            "label": f"Approve action \"{action_name}\" for goal \"{goal_title}\"",
            "detail": f"Waiting for approval",
            "occurred_at": ap.created_at,
            "url": _admin_url("gameactionapprovalrequest", ap.pk),
            "button": "Review",
        })
    for cont in pending_continuations:
        goal_title = cont.goal.title if cont.goal else "–"
        action_queue.append({
            "kind": "info",
            "label": f"Goal \"{goal_title}\" is paused",
            "detail": cont.get_reason_code_display(),
            "occurred_at": cont.created_at,
            "url": _admin_url("gamecontinuationrequest", cont.pk),
            "button": "Provide",
        })
    for session in recent_failed:
        name = session.source_label or f"Session #{session.pk}"
        action_queue.append({
            "kind": "failed",
            "label": f"{name} failed",
            "detail": (session.error_detail or "Inspect timeline for details")[:80],
            "occurred_at": session.created_at,
            "url": _admin_url("executionsession", session.pk),
            "button": "Inspect",
        })

    # ── Workspace pulse counts ────────────────────────────────────────
    orchestrator_session_count = ExecutionSession.objects.filter(
        runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR,
    ).count()
    orchestrator_active_sessions = ExecutionSession.objects.filter(
        runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR,
        status__in=[ExecutionSession.Status.RUNNING, ExecutionSession.Status.WAITING_ASYNC],
    ).count()
    open_game_goals = GameGoal.objects.filter(
        status__in=[
            GameGoal.Status.QUEUED,
            GameGoal.Status.RUNNING,
            GameGoal.Status.WAITING_INFO,
            GameGoal.Status.WAITING_APPROVAL,
            GameGoal.Status.BLOCKED,
        ]
    ).count()
    live_game_sessions = ExecutionSession.objects.filter(
        runtime_kind=ExecutionSession.RuntimeKind.GAME,
        status__in=[ExecutionSession.Status.RUNNING, ExecutionSession.Status.WAITING_ASYNC],
    ).count()
    running_sessions = ExecutionSession.objects.filter(
        status=ExecutionSession.Status.RUNNING,
    ).count()
    waiting_sessions = ExecutionSession.objects.filter(
        status=ExecutionSession.Status.WAITING_ASYNC,
    ).count()
    live_sessions = running_sessions + waiting_sessions
    failed_session_count = ExecutionSession.objects.filter(
        status=ExecutionSession.Status.FAILED,
    ).count()

    # ── Recent sessions feed ──────────────────────────────────────────
    raw_sessions = (
        ExecutionSession.objects
        .order_by("-created_at")
        .values(
            "id", "source_label", "pipeline__name", "entry_agent__name",
            "runtime_kind", "status", "created_at", "error_detail",
        )[:8]
    )
    recent_sessions_home = [
        {**s, "url": _admin_url("executionsession", s["id"])}
        for s in raw_sessions
    ]

    # ── Light health summary (no full graph build) ────────────────────
    active_nodes = active_provider_count + active_model_count + active_agent_count + active_pipeline_count
    inactive_nodes = (
        ProviderConfig.objects.filter(is_active=False).count()
        + ModelConfig.objects.filter(is_active=False).count()
        + AgentProfile.objects.filter(is_active=False).count()
        + PipelineDefinition.objects.filter(is_active=False).count()
    )
    # Agents missing both contracts count as needing review
    warning_nodes = AgentProfile.objects.filter(
        is_active=True,
        input_contract={},
        output_contract={},
    ).count()
    health_summary = {
        "ok": max(0, active_nodes - warning_nodes),
        "warning": warning_nodes,
        "error": 0,
        "inactive": inactive_nodes,
    }

    checklist = [
        {
            **_checklist_item(
                "Connect an AI provider",
                active_provider_count > 0,
                f"{active_provider_count} active provider(s)",
                "Add a local or external provider before creating models.",
            ),
            "url": _admin_add_url("providerconfig"),
            "button": "Add provider",
        },
        {
            **_checklist_item(
                "Choose at least one model",
                active_model_count > 0,
                f"{active_model_count} active model(s)",
                "Create a model record with the exact provider model name.",
            ),
            "url": _admin_add_url("modelconfig"),
            "button": "Add model",
        },
        {
            **_checklist_item(
                "Create an agent",
                active_agent_count > 0,
                f"{active_agent_count} active agent(s)",
                "Agents are the specialists used by both workspaces.",
            ),
            "url": _admin_add_url("agentprofile"),
            "button": "Add agent",
        },
        {
            **_checklist_item(
                "Add knowledge or tools",
                active_document_count > 0 or active_tool_count > 0,
                f"{active_document_count} active document(s), {active_tool_count} active tool(s)",
                "Optional, but useful once your first agent works.",
            ),
            "url": _admin_changelist_url("knowledgecollection"),
            "button": "Review resources",
        },
        {
            **_checklist_item(
                "Run a first session",
                recent_session_count > 0,
                f"{recent_session_count} execution session(s)",
                "Create one small test run before expanding the system.",
            ),
            "url": _admin_changelist_url("executionsession"),
            "button": "View sessions",
        },
    ]

    if active_provider_count == 0:
        recommended = _action(
            "Start by connecting a provider",
            "AI Hub needs at least one provider before models and agents can run.",
            _admin_add_url("providerconfig"),
            "Connect provider",
        )
    elif active_model_count == 0:
        recommended = _action(
            "Add the first model",
            "Models tell agents which provider model to call and what defaults to use.",
            _admin_add_url("modelconfig"),
            "Add model",
        )
    elif active_agent_count == 0:
        recommended = _action(
            "Create your first agent",
            "An agent is a focused role with one prompt, one model and clear contracts.",
            _admin_add_url("agentprofile"),
            "Create agent",
        )
    elif active_pipeline_count == 0 and game_session_count == 0:
        recommended = _action(
            "Choose a workspace",
            "Use Orchestrator for known steps, or GAME for autonomous goal loops.",
            reverse("admin:ai_hub_workspace_orchestrator"),
            "Build workflow",
        )
    else:
        recommended = _action(
            "Inspect system health",
            "Use the control center to see connections, warnings, sessions and bottlenecks.",
            reverse("admin:ai_hub_control_center"),
            "Open control center",
        )

    resources = [
        {
            "label": "Providers",
            "count": provider_count,
            "description": "AI services and local endpoints.",
            "url": _admin_changelist_url("providerconfig"),
            "add_url": _admin_add_url("providerconfig"),
        },
        {
            "label": "Models",
            "count": model_count,
            "description": "Provider model names and generation defaults.",
            "url": _admin_changelist_url("modelconfig"),
            "add_url": _admin_add_url("modelconfig"),
        },
        {
            "label": "Agents",
            "count": agent_count,
            "description": "Prompts, roles, contracts, knowledge and tools.",
            "url": _admin_changelist_url("agentprofile"),
            "add_url": _admin_add_url("agentprofile"),
        },
        {
            "label": "Knowledge",
            "count": KnowledgeCollection.objects.count(),
            "description": "Curated context groups agents can read.",
            "url": _admin_changelist_url("knowledgecollection"),
            "add_url": _admin_add_url("knowledgecollection"),
        },
        {
            "label": "Tools",
            "count": ToolDefinition.objects.count(),
            "description": "Optional capabilities attached to agents.",
            "url": _admin_changelist_url("tooldefinition"),
            "add_url": _admin_add_url("tooldefinition"),
        },
    ]

    done_count = sum(1 for item in checklist if item["status"] == "ok")
    checklist_pct = round(done_count * 100 / len(checklist)) if checklist else 0

    hidden_models = []
    for category, model_name, label, reason in _HIDDEN_MODEL_CATALOG:
        try:
            url = _admin_changelist_url(model_name)
        except Exception:
            url = ""
        hidden_models.append({"category": category, "label": label, "reason": reason, "url": url})

    return {
        "ai_hub_home": {
            "metrics": {
                "providers": provider_count,
                "models": model_count,
                "agents": agent_count,
                "pipelines": PipelineDefinition.objects.count(),
                "sessions": recent_session_count,
                "game_sessions": game_session_count,
                "orchestrator_sessions": orchestrator_session_count,
                "live": live_sessions,
                "failed": failed_session_count,
            },
            "vitals": {
                "live": live_sessions,
                "running": running_sessions,
                "waiting": waiting_sessions,
                "needs_attention": len(action_queue),
                "open_goals": open_game_goals,
                "sessions": recent_session_count,
                "failed": failed_session_count,
            },
            "orchestrator": {
                "pipelines": PipelineDefinition.objects.count(),
                "active_pipelines": active_pipeline_count,
                "sessions": orchestrator_session_count,
                "active_sessions": orchestrator_active_sessions,
            },
            "game": {
                "open_goals": open_game_goals,
                "live_sessions": live_game_sessions,
                "sessions": game_session_count,
            },
            "action_queue": action_queue,
            "needs_attention_count": len(action_queue),
            "recent_sessions": recent_sessions_home,
            "health_summary": health_summary,
            "checklist": checklist,
            "checklist_pct": checklist_pct,
            "checklist_done": done_count,
            "recommended_action": recommended,
            "resources": resources,
            "examples": EXAMPLE_TEMPLATES,
            "hidden_models": hidden_models,
        }
    }


def _game_node_trace(nodes: list[dict], edges: list[dict]) -> None:
    nodes_by_id = {node["id"]: node for node in nodes}
    for node in nodes:
        node["trace"] = _node_trace(node, edges, nodes_by_id)


def build_game_graph_context() -> dict:
    session = (
        ExecutionSession.objects.select_related("entry_agent")
        .filter(runtime_kind=ExecutionSession.RuntimeKind.GAME)
        .order_by("-created_at")
        .first()
    )
    nodes: list[dict] = []
    edges: list[dict] = []
    title = "How a GAME session thinks"
    subtitle = "A GAME agent receives a goal, decides actions, records observations, updates memory and stops."

    if not session:
        nodes = [
            _node("goal:demo", "Goal", "goal", "ok", "What should be achieved?"),
            _node("agent:demo", "Entry agent", "agent", "ok", "Prompt + model + tools"),
            _node("decision:demo", "Decision", "decision", "ok", "Choose the next action"),
            _node("action:demo", "Action", "action", "ok", "Think, use a tool or finish"),
            _node("memory:demo", "Memory", "memory", "ok", "Carry observations forward"),
            _node("stop:demo", "Stop", "stop", "ok", "Return final answer"),
        ]
        edges = [
            _edge("goal:demo", "agent:demo", "guides"),
            _edge("agent:demo", "decision:demo", "decides"),
            _edge("decision:demo", "action:demo", "acts"),
            _edge("action:demo", "memory:demo", "observes"),
            _edge("memory:demo", "decision:demo", "loops"),
            _edge("memory:demo", "stop:demo", "finishes"),
        ]
    else:
        title = f"Latest GAME session #{session.id}"
        subtitle = session.source_label or session.goal_text or "Latest autonomous goal session"
        goal_detail = session.goal_text[:90] if session.goal_text else "No goal text recorded"
        nodes.append(
            _node(
                f"goal:{session.id}",
                "Goal",
                "goal",
                "ok" if session.goal_text else "warning",
                goal_detail,
                _admin_url("executionsession", session.id),
            )
        )
        agent = session.entry_agent
        agent_label = agent.name if agent else "Entry agent missing"
        nodes.append(
            _node(
                f"agent:{session.id}",
                agent_label,
                "agent",
                "ok" if agent and agent.is_active else "warning",
                agent.role if agent else "No entry agent",
                _admin_url("agentprofile", agent.id) if agent else "",
            )
        )
        edges.append(_edge(f"goal:{session.id}", f"agent:{session.id}", "guides"))

        previous_memory_id = ""
        step_runs = list(session.step_runs.select_related("agent").order_by("order")[:8])
        for step_run in step_runs:
            observation = step_run.observation_payload if isinstance(step_run.observation_payload, dict) else {}
            decision = observation.get("decision") if isinstance(observation.get("decision"), dict) else {}
            action_name = observation.get("action") or step_run.action_name or decision.get("action") or "decision"
            message = decision.get("message") or decision.get("final_answer") or step_run.error_detail or step_run.status
            decision_id = f"decision:{step_run.id}"
            action_id = f"action:{step_run.id}"
            memory_id = f"memory:{step_run.id}"
            status = "error" if step_run.status == ExecutionStepRun.Status.FAILED else "ok"
            if step_run.status == ExecutionStepRun.Status.RUNNING:
                status = "warning"
            nodes.extend(
                [
                    _node(
                        decision_id,
                        f"Decision {step_run.order}",
                        "decision",
                        status,
                        step_run.agent.name if step_run.agent else "No agent",
                        _admin_url("executionsteprun", step_run.id),
                    ),
                    _node(action_id, action_name, "action", status, message[:90], _admin_url("executionsteprun", step_run.id)),
                    _node(memory_id, f"Memory {step_run.order}", "memory", status, f"{step_run.latency_ms or '-'} ms", _admin_url("executionsteprun", step_run.id)),
                ]
            )
            edges.append(_edge(previous_memory_id or f"agent:{session.id}", decision_id, "decides"))
            edges.append(_edge(decision_id, action_id, "acts", status))
            edges.append(_edge(action_id, memory_id, "observes", status))
            previous_memory_id = memory_id

        stop_status = "ok" if session.status == ExecutionSession.Status.SUCCESS else "warning"
        if session.status == ExecutionSession.Status.FAILED:
            stop_status = "error"
        final_answer = (session.final_context or {}).get("final_answer") or session.error_detail or session.status
        nodes.append(
            _node(
                f"stop:{session.id}",
                "Stop / final answer",
                "stop",
                stop_status,
                str(final_answer)[:90],
                _admin_url("executionsession", session.id),
            )
        )
        if previous_memory_id:
            edges.append(_edge(previous_memory_id, f"stop:{session.id}", "finishes", stop_status))
        else:
            edges.append(_edge(f"agent:{session.id}", f"stop:{session.id}", "awaits", stop_status))

    _game_node_trace(nodes, edges)
    return {
        "game_graph": {
            "title": title,
            "subtitle": subtitle,
            "legend": GAME_KIND_LEGEND,
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "kindOrder": ["goal", "agent", "decision", "action", "memory", "stop"],
                "kindLabels": {
                    "goal": "Goal",
                    "agent": "Agent",
                    "decision": "Decision",
                    "action": "Action",
                    "memory": "Memory",
                    "stop": "Stop",
                },
                "kindColors": {
                    "goal": "#2563eb",
                    "agent": "#be123c",
                    "decision": "#7c3aed",
                    "action": "#c2410c",
                    "memory": "#0f766e",
                    "stop": "#047857",
                },
                "relationColors": {
                    "guides": "#38bdf8",
                    "decides": "#f43f5e",
                    "acts": "#fb923c",
                    "observes": "#a78bfa",
                    "loops": "#94a3b8",
                    "finishes": "#34d399",
                    "awaits": "#f59e0b",
                },
            },
        }
    }


def build_control_center_context() -> dict:
    providers = list(ProviderConfig.objects.all().order_by("name"))
    provider_health = {provider.id: _provider_health(provider) for provider in providers}
    provider_by_id = {provider.id: provider for provider in providers}
    models = list(ModelConfig.objects.select_related("provider").order_by("provider__name", "model_name"))
    agents = list(
        AgentProfile.objects.select_related("model_config", "model_config__provider")
        .prefetch_related("knowledge_collections", "tools")
        .order_by("name")
    )
    agent_by_id = {agent.id: agent for agent in agents}
    collections = list(
        KnowledgeCollection.objects.annotate(
            active_documents_count=Count("documents", filter=Q(documents__status=KnowledgeDocument.Status.ACTIVE))
        ).order_by("name")
    )
    tools = list(ToolDefinition.objects.order_by("name"))
    pipelines = list(
        PipelineDefinition.objects.prefetch_related(
            Prefetch(
                "steps",
                queryset=PipelineStep.objects.select_related("agent", "fallback_agent").order_by("order"),
            )
        ).order_by("name")
    )

    nodes = []
    edges = []
    warnings = []
    attention_items = []
    model_catalog = []
    pipeline_summaries = []
    pipeline_scopes = []

    agent_step_metrics = {
        row["agent_id"]: row
        for row in ExecutionStepRun.objects.values("agent_id").annotate(
            total=Count("id"),
            failed=Count("id", filter=Q(status=ExecutionStepRun.Status.FAILED)),
            avg_latency=Avg("latency_ms"),
            last_seen=Max("created_at"),
        )
        if row["agent_id"]
    }
    model_step_metrics = {
        row["agent__model_config_id"]: row
        for row in ExecutionStepRun.objects.values("agent__model_config_id").annotate(
            total=Count("id"),
            failed=Count("id", filter=Q(status=ExecutionStepRun.Status.FAILED)),
            avg_latency=Avg("latency_ms"),
            last_seen=Max("created_at"),
        )
        if row["agent__model_config_id"]
    }

    for provider in providers:
        health = provider_health[provider.id]
        status = health.status if provider.is_active else "inactive"
        nodes.append(
            _node(
                f"provider:{provider.id}",
                provider.name,
                "provider",
                status,
                health.detail,
                _admin_url("providerconfig", provider.id),
                {"type": provider.get_provider_type_display(), "base_url": provider.base_url},
            )
        )
        if status in {"error", "warning"}:
            message = f"Provider '{provider.name}': {health.detail}"
            warnings.append(message)
            attention_items.append(
                _attention_item(
                    f"provider:{provider.id}:{status}",
                    "error" if status == "error" else "warning",
                    f"Provider: {provider.name}",
                    health.detail,
                    _admin_url("providerconfig", provider.id),
                    provider.updated_at,
                    "provider",
                    90 if status == "error" else 60,
                    f"{message} Type: {provider.get_provider_type_display()}. Base URL: {provider.base_url or 'not set'}.",
                )
            )

    for model in models:
        health = provider_health.get(model.provider_id)
        installed = True
        if model.provider.provider_type == ProviderConfig.ProviderType.OLLAMA and health:
            installed = _ollama_model_name(model.model_name) in health.models
        status = "ok" if model.is_active and model.provider.is_active and installed else "warning"
        if not model.is_active or not model.provider.is_active:
            status = "inactive"
        if model.is_active and model.provider.is_active and not installed:
            message = f"Model '{model.model_name}' is configured but was not reported by Ollama."
            warnings.append(message)
            attention_items.append(
                _attention_item(
                    f"model:{model.id}:missing",
                    "warning",
                    f"Model not reported: {model.model_name}",
                    f"Provider {model.provider.name} did not report this configured model.",
                    _admin_url("modelconfig", model.id),
                    model.updated_at,
                    "model",
                    55,
                    f"{message} Provider: {model.provider.name}. Temperature: {model.temperature_default}. Max tokens: {model.max_tokens_default}.",
                )
            )
        model_catalog.append(
            {
                "name": model.model_name,
                "provider": model.provider.name,
                "provider_type": model.provider.get_provider_type_display(),
                "status": status,
                "detail": "Available" if installed else "Not reported by provider",
                "supports_tools": model.supports_tools,
                "temperature": str(model.temperature_default),
                "max_tokens": model.max_tokens_default,
                "url": _admin_url("modelconfig", model.id),
            }
        )
        nodes.append(
            _node(
                f"model:{model.id}",
                model.model_name,
                "model",
                status,
                "Available" if installed else "Not reported by provider",
                _admin_url("modelconfig", model.id),
                {
                    "temperature": str(model.temperature_default),
                    "max_tokens": model.max_tokens_default,
                    "supports_tools": model.supports_tools,
                    "agents": len([agent for agent in agents if agent.model_config_id == model.id]),
                    "runs": model_step_metrics.get(model.id, {}).get("total", 0),
                    "failed_runs": model_step_metrics.get(model.id, {}).get("failed", 0),
                    "avg_latency_ms": (
                        round(model_step_metrics[model.id]["avg_latency"], 1)
                        if model_step_metrics.get(model.id, {}).get("avg_latency")
                        else ""
                    ),
                },
            )
        )
        edges.append(_edge(f"provider:{model.provider_id}", f"model:{model.id}", "serves", status))

    for collection in collections:
        active_docs = collection.active_documents_count
        status = "ok" if collection.is_active and active_docs else "warning"
        if not collection.is_active:
            status = "inactive"
        if collection.is_active and not active_docs:
            message = f"Knowledge collection '{collection.name}' has no active documents."
            warnings.append(message)
            attention_items.append(
                _attention_item(
                    f"knowledge:{collection.id}:empty",
                    "warning",
                    f"Knowledge empty: {collection.name}",
                    "Collection is active but has no active documents.",
                    _admin_url("knowledgecollection", collection.id),
                    collection.updated_at,
                    "knowledge",
                    45,
                    f"{message} Agents using it: {collection.agents.count()}.",
                )
            )
        nodes.append(
            _node(
                f"knowledge:{collection.id}",
                collection.name,
                "knowledge",
                status,
                f"{active_docs} active docs",
                _admin_url("knowledgecollection", collection.id),
                {"active_docs": active_docs, "agents": collection.agents.count()},
            )
        )

    for tool in tools:
        nodes.append(
            _node(
                f"tool:{tool.id}",
                tool.name,
                "tool",
                "ok" if tool.is_active else "inactive",
                tool.get_tool_kind_display(),
                _admin_url("tooldefinition", tool.id),
            )
        )

    for agent in agents:
        status = "ok" if agent.is_active and agent.model_config.is_active and agent.model_config.provider.is_active else "inactive"
        if agent.is_active and (not agent.input_contract or not agent.output_contract):
            status = "warning"
            message = f"Agent '{agent.name}' has incomplete contracts."
            warnings.append(message)
            attention_items.append(
                _attention_item(
                    f"agent:{agent.id}:contracts",
                    "warning",
                    f"Agent contracts: {agent.name}",
                    "Input or output contract is missing.",
                    _admin_url("agentprofile", agent.id),
                    agent.updated_at,
                    "agent",
                    50,
                    f"{message} Role: {agent.role}. Model: {agent.model_config.model_name}.",
                )
            )
        nodes.append(
            _node(
                f"agent:{agent.id}",
                agent.name,
                "agent",
                status,
                agent.role,
                _admin_url("agentprofile", agent.id),
                {
                    "execution_mode": agent.execution_mode,
                    "knowledge_max_chars": agent.knowledge_max_chars,
                    "model": agent.model_config.model_name,
                    "provider": agent.model_config.provider.name,
                    "pipeline_steps": agent.pipeline_steps.count(),
                    "runs": agent_step_metrics.get(agent.id, {}).get("total", 0),
                    "failed_runs": agent_step_metrics.get(agent.id, {}).get("failed", 0),
                    "avg_latency_ms": (
                        round(agent_step_metrics[agent.id]["avg_latency"], 1)
                        if agent_step_metrics.get(agent.id, {}).get("avg_latency")
                        else ""
                    ),
                },
            )
        )
        edges.append(_edge(f"model:{agent.model_config_id}", f"agent:{agent.id}", "runs", status))
        for collection in agent.knowledge_collections.all():
            edges.append(_edge(f"knowledge:{collection.id}", f"agent:{agent.id}", "informs"))
        for tool in agent.tools.all():
            edges.append(_edge(f"tool:{tool.id}", f"agent:{agent.id}", "enables"))

    step_metrics = {
        row["pipeline_step_id"]: row
        for row in ExecutionStepRun.objects.values("pipeline_step_id")
        .annotate(
            total=Count("id"),
            failed=Count("id", filter=Q(status=ExecutionStepRun.Status.FAILED)),
            avg_latency=Avg("latency_ms"),
            last_seen=Max("created_at"),
        )
        if row["pipeline_step_id"]
    }
    pipeline_counts = {
        row["pipeline_id"]: row
        for row in ExecutionSession.objects.filter(runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR).values("pipeline_id")
        .annotate(
            total=Count("id"),
            failed=Count("id", filter=Q(status=ExecutionSession.Status.FAILED)),
        )
        if row["pipeline_id"]
    }

    for pipeline in pipelines:
        run_counts = pipeline_counts.get(pipeline.id, {})
        total_runs = run_counts.get("total", 0)
        failed_runs = run_counts.get("failed", 0)
        status = "ok" if pipeline.is_active else "inactive"
        steps = list(pipeline.steps.all())
        if pipeline.is_active and not steps:
            status = "warning"
            message = f"Pipeline '{pipeline.name}' is active without steps."
            warnings.append(message)
            attention_items.append(
                _attention_item(
                    f"pipeline:{pipeline.id}:no-steps",
                    "warning",
                    f"Pipeline has no steps: {pipeline.name}",
                    "Pipeline is active but no ordered steps are configured.",
                    _admin_url("pipelinedefinition", pipeline.id),
                    pipeline.updated_at,
                    "pipeline",
                    65,
                    f"{message} Add PipelineStep records or deactivate until it is ready.",
                )
            )
        nodes.append(
            _node(
                f"pipeline:{pipeline.id}",
                pipeline.name,
                "pipeline",
                status,
                f"{total_runs} runs, {failed_runs} failed",
                _admin_url("pipelinedefinition", pipeline.id),
                {"active": pipeline.is_active, "runs": total_runs, "failed_runs": failed_runs, "steps": len(steps)},
            )
        )

        previous_step_id = None
        step_summaries = []
        pipeline_scope_nodes = {f"pipeline:{pipeline.id}"}
        for step in steps:
            metric = step_metrics.get(step.id, {})
            avg_latency = metric.get("avg_latency")
            detail = f"step {step.order}"
            if avg_latency:
                detail = f"{detail}, avg {round(avg_latency, 1)} ms"
            step_status = "ok" if step.agent.is_active else "warning"
            if step.on_error == step.OnError.FALLBACK_AGENT and not step.fallback_agent_id:
                step_status = "warning"
            nodes.append(
                _node(
                    f"step:{step.id}",
                    f"{step.order}. {step.agent.name}",
                    "step",
                    step_status,
                    detail,
                    _admin_url("pipelinestep", step.id),
                    {
                        "on_error": step.on_error,
                        "agent": step.agent.name,
                        "model": step.agent.model_config.model_name,
                        "runs": metric.get("total", 0),
                        "failed_runs": metric.get("failed", 0),
                        "avg_latency_ms": round(avg_latency, 1) if avg_latency else "",
                        "last_seen": metric.get("last_seen") or "",
                    },
                )
            )
            edges.append(_edge(f"pipeline:{pipeline.id}", f"step:{step.id}", "contains", step_status))
            edges.append(_edge(f"step:{step.id}", f"agent:{step.agent_id}", "calls", step_status))
            if previous_step_id:
                edges.append(_edge(f"step:{previous_step_id}", f"step:{step.id}", "next"))
            if step.fallback_agent_id:
                edges.append(_edge(f"step:{step.id}", f"agent:{step.fallback_agent_id}", "fallback", "warning"))
            pipeline_scope_nodes.update(
                {
                    f"step:{step.id}",
                    f"agent:{step.agent_id}",
                    f"model:{step.agent.model_config_id}",
                    f"provider:{step.agent.model_config.provider_id}",
                }
            )
            scoped_agent = agent_by_id.get(step.agent_id)
            if scoped_agent:
                pipeline_scope_nodes.update(f"knowledge:{collection.id}" for collection in scoped_agent.knowledge_collections.all())
                pipeline_scope_nodes.update(f"tool:{tool.id}" for tool in scoped_agent.tools.all())
            step_summaries.append(
                {
                    "order": step.order,
                    "agent": step.agent.name,
                    "agent_role": step.agent.role,
                    "model": step.agent.model_config.model_name,
                    "provider": provider_by_id[step.agent.model_config.provider_id].name,
                    "status": step_status,
                    "on_error": step.on_error,
                    "avg_latency": round(avg_latency, 1) if avg_latency else None,
                    "url": _admin_url("pipelinestep", step.id),
                }
            )
            previous_step_id = step.id
        pipeline_summaries.append(
            {
                "id": f"pipeline:{pipeline.id}",
                "name": pipeline.name,
                "status": status,
                "is_active": pipeline.is_active,
                "total_runs": total_runs,
                "failed_runs": failed_runs,
                "steps": step_summaries,
                "url": _admin_url("pipelinedefinition", pipeline.id),
            }
        )
        pipeline_scopes.append(
            {
                "id": f"pipeline:{pipeline.id}",
                "name": pipeline.name,
                "node_ids": sorted(pipeline_scope_nodes),
            }
        )

    orchestrator_sessions = ExecutionSession.objects.filter(runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR)
    run_totals = orchestrator_sessions.aggregate(
        total=Count("id"),
        success=Count("id", filter=Q(status=ExecutionSession.Status.SUCCESS)),
        failed=Count("id", filter=Q(status=ExecutionSession.Status.FAILED)),
        waiting=Count("id", filter=Q(status=ExecutionSession.Status.WAITING_ASYNC)),
    )
    total_runs = run_totals["total"] or 0
    successful_runs = run_totals["success"] or 0
    failed_runs = run_totals["failed"] or 0
    waiting_runs = run_totals["waiting"] or 0
    avg_latency = ExecutionStepRun.objects.aggregate(avg=Avg("latency_ms"))["avg"] or 0
    top_fail_steps = list(
        ExecutionStepRun.objects.filter(status=ExecutionStepRun.Status.FAILED)
        .values("pipeline_step_id", "pipeline_step__pipeline__name", "pipeline_step__order", "pipeline_step__agent__name")
        .annotate(total=Count("id"), last_seen=Max("created_at"))
        .order_by("-last_seen", "-total")[:10]
    )
    for row in top_fail_steps:
        latest_failure = (
            ExecutionStepRun.objects.filter(
                status=ExecutionStepRun.Status.FAILED,
                pipeline_step_id=row["pipeline_step_id"],
            )
            .select_related("session", "pipeline_step", "pipeline_step__pipeline", "agent")
            .order_by("-created_at")
            .first()
        )
        if not latest_failure:
            continue
        pipeline_name = row["pipeline_step__pipeline__name"] or "No pipeline"
        step_order = row["pipeline_step__order"] or latest_failure.order
        agent_name = row["pipeline_step__agent__name"] or (latest_failure.agent.name if latest_failure.agent else "No agent")
        detail = latest_failure.error_detail or latest_failure.session.error_detail or "Latest failed step has no error detail."
        attention_items.append(
            _attention_item(
                f"step-failure:{row['pipeline_step_id'] or 'none'}",
                "error",
                f"{pipeline_name} / step {step_order} / {agent_name}",
                f"{row['total']} failed run(s). Latest: session #{latest_failure.session_id}.",
                _admin_url("executionsteprun", latest_failure.id),
                latest_failure.created_at,
                "step",
                100 + int(row["total"] or 0),
                f"{detail} Open the latest failed step run for request, response, observation and error payloads.",
            )
        )
    session_totals = ExecutionSession.objects.aggregate(
        total=Count("id"),
        orchestrator=Count("id", filter=Q(runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR)),
        game=Count("id", filter=Q(runtime_kind=ExecutionSession.RuntimeKind.GAME)),
        success=Count("id", filter=Q(status=ExecutionSession.Status.SUCCESS)),
        failed=Count("id", filter=Q(status=ExecutionSession.Status.FAILED)),
        waiting=Count("id", filter=Q(status=ExecutionSession.Status.WAITING_ASYNC)),
    )
    generic_avg_latency = ExecutionStepRun.objects.aggregate(avg=Avg("latency_ms"))["avg"] or 0
    recent_sessions = list(
        ExecutionSession.objects.select_related("pipeline", "entry_agent")
        .order_by("-created_at")[:8]
        .values(
            "id",
            "source_label",
            "pipeline__name",
            "entry_agent__name",
            "runtime_kind",
            "runtime_mode",
            "status",
            "created_at",
            "error_detail",
        )
    )
    for session in recent_sessions:
        session["url"] = _admin_url("executionsession", session["id"])

    active_pipelines = [pipeline for pipeline in pipelines if pipeline.is_active]
    configured_ollama_models = [
        model for model in model_catalog if model["provider_type"] == "Ollama" and model["status"] != "inactive"
    ]
    missing_ollama_models = [model for model in configured_ollama_models if model["status"] == "warning"]
    active_docs_total = sum(collection.active_documents_count for collection in collections if collection.is_active)
    checklist = [
        _checklist_item(
            "Active provider",
            any(provider.is_active for provider in providers),
            f"{sum(1 for provider in providers if provider.is_active)} active provider(s)",
            "Create or activate a ProviderConfig.",
        ),
        _checklist_item(
            "Ollama models installed",
            not missing_ollama_models,
            f"{len(missing_ollama_models)} configured model(s) missing from Ollama",
            "Install missing models or update ModelConfig records.",
        ),
        _checklist_item(
            "Active pipeline",
            bool(active_pipelines),
            f"{len(active_pipelines)} active pipeline(s)",
            "Activate one PipelineDefinition after its steps are ready.",
        ),
        _checklist_item(
            "Knowledge ready",
            active_docs_total > 0,
            f"{active_docs_total} active knowledge document(s)",
            "Add active KnowledgeDocument records to feed agent context.",
        ),
        _checklist_item(
            "Recent successful sessions",
            bool(recent_sessions) and any(session["status"] == ExecutionSession.Status.SUCCESS for session in recent_sessions),
            f"{sum(1 for session in recent_sessions if session['status'] == ExecutionSession.Status.SUCCESS)} success in last {len(recent_sessions)} session(s)",
            "Run a pipeline successfully to validate the current config.",
        ),
    ]

    nodes_by_id = {node["id"]: node for node in nodes}
    for node in nodes:
        node["trace"] = _node_trace(node, edges, nodes_by_id)

    if warnings and not attention_items:
        attention_items.extend(
            _attention_item(
                f"warning:{index}",
                "warning",
                warning,
                "Configuration warning.",
                "",
                None,
                "config",
                40,
                warning,
            )
            for index, warning in enumerate(warnings[:12], start=1)
        )

    return {
        "graph": {"nodes": nodes, "edges": edges, "pipelineScopes": pipeline_scopes},
        "metrics": {
            "providers": len(providers),
            "models": len(models),
            "agents": len(agents),
            "pipelines": len(pipelines),
            "runs": total_runs,
            "execution_sessions": session_totals["total"] or 0,
            "orchestrator_sessions": session_totals["orchestrator"] or 0,
            "game_sessions": session_totals["game"] or 0,
            "success_rate": round((successful_runs / total_runs) * 100, 1) if total_runs else 0,
            "failed_runs": failed_runs,
            "waiting_runs": waiting_runs,
            "avg_latency": round(avg_latency, 1),
            "generic_avg_latency": round(generic_avg_latency, 1),
        },
        "warnings": warnings[:12],
        "top_fail_steps": top_fail_steps,
        "attention_items": sorted(
            attention_items,
            key=lambda item: (item["occurred_at"] is not None, item["occurred_at"], item["relevance"]),
            reverse=True,
        )[:40],
        "pipeline_metrics": pipeline_counts,
        "legend": NODE_KIND_LEGEND,
        "status_summary": _status_summary(nodes),
        "model_catalog": model_catalog,
        "pipeline_summaries": pipeline_summaries,
        "critical_nodes": _connected_node_summary(nodes, edges),
        "health_checklist": checklist,
        "recent_sessions": recent_sessions,
    }


def build_operations_inbox_context():
    """Cross-workspace queue of everything that needs a human (IA Step 5).

    Four categories: pending approvals, paused sessions waiting for information,
    failed sessions, and blocked goals. Unbounded for approvals/waiting (those are
    the actionable gates); failures/blocked are capped to the most recent.
    """
    approvals = list(
        GameActionApprovalRequest.objects
        .filter(status=GameActionApprovalRequest.Status.PENDING)
        .select_related("action_run", "goal", "goal__workspace")
        .order_by("created_at")
    )
    waiting = list(
        GameContinuationRequest.objects
        .filter(status=GameContinuationRequest.Status.PENDING)
        .select_related("goal", "goal__workspace")
        .order_by("created_at")
    )
    failures = list(
        ExecutionSession.objects
        .filter(status=ExecutionSession.Status.FAILED)
        .select_related("entry_agent", "pipeline")
        .order_by("-created_at")[:25]
    )
    blocked = list(
        GameGoal.objects
        .filter(status=GameGoal.Status.BLOCKED)
        .select_related("workspace")
        .order_by("-updated_at")[:25]
    )

    approval_items = [
        {
            "id": ap.pk,
            "action_name": ap.action_run.action_name if ap.action_run else "—",
            "goal_title": ap.goal.title if ap.goal else "—",
            "created_at": ap.created_at,
            "expires_at": ap.expires_at,
            "url": _admin_url("gameactionapprovalrequest", ap.pk),
            "approve_url": reverse("admin:ai_hub_gameactionapprovalrequest_approve", args=[ap.pk]),
            "reject_url": reverse("admin:ai_hub_gameactionapprovalrequest_reject", args=[ap.pk]),
        }
        for ap in approvals
    ]
    waiting_items = [
        {
            "id": c.pk,
            "goal_title": c.goal.title if c.goal else "—",
            "reason": c.get_reason_code_display(),
            "created_at": c.created_at,
            "url": _admin_url("gamecontinuationrequest", c.pk),
            "goal_url": _admin_url("gamegoal", c.goal_id) if c.goal_id else None,
        }
        for c in waiting
    ]
    failure_items = [
        {
            "id": s.pk,
            "label": s.source_label or f"Session #{s.pk}",
            "runtime_kind": s.runtime_kind,
            "error": (s.error_detail or "")[:140],
            "created_at": s.created_at,
            "url": _admin_url("executionsession", s.pk),
        }
        for s in failures
    ]
    blocked_items = [
        {
            "id": g.pk,
            "title": g.title,
            "workspace": g.workspace.name if g.workspace else "—",
            "url": _admin_url("gamegoal", g.pk),
        }
        for g in blocked
    ]

    counts = {
        "approvals": len(approval_items),
        "waiting": len(waiting_items),
        "failures": len(failure_items),
        "blocked": len(blocked_items),
    }
    counts["total"] = sum(counts.values())

    return {
        "operations_inbox": {
            "counts": counts,
            "approvals": approval_items,
            "waiting": waiting_items,
            "failures": failure_items,
            "blocked": blocked_items,
        }
    }
