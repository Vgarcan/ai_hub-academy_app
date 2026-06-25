import json
from decimal import Decimal

from django import forms
from django.contrib import admin
from django.contrib import messages
from django.core.exceptions import PermissionDenied, ValidationError
from django.db import transaction
from django.db.models import Count, Q
from django.http import HttpResponseNotAllowed, HttpResponseRedirect, JsonResponse
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _

from .models import (
    AgentProfile,
    AgentToolGrant,
    AgentToolboxAssignment,
    ExecutionSession,
    ExecutionStepRun,
    GameActionApprovalRequest,
    GameActionDefinition,
    GameActionRun,
    GameContinuationRequest,
    GameDelegationRun,
    GameGoal,
    GameGoalDependency,
    GameGoalPlan,
    GameGoalPlanStep,
    GameMemoryEntry,
    GameWorkspace,
    GameWorkspaceAction,
    GameWorkspaceAgent,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    ProviderConfig,
    ToolDefinition,
    Toolbox,
    ToolboxTool,
    ToolExecutionRun,
)
from .services.admin_control_center import (
    build_ai_hub_home_context,
    build_control_center_context,
    build_game_graph_context,
)
from .services.execution_runner import run_execution_session
from .services.game_goal_execution import create_goal_execution_session
from .services.game_operational_ux import redact_payload, redact_text
from .services.game_resume import approve_action_run, reject_action_run
from .services.tool_resolution import resolve_agent_tools


ACTIVE_EXECUTION_STATUSES = (
    ExecutionSession.Status.PENDING,
    ExecutionSession.Status.RUNNING,
    ExecutionSession.Status.WAITING_ASYNC,
)


def _render_redacted_json(payload):
    safe = redact_payload(payload or {})
    return format_html(
        "<pre>{}</pre>",
        json.dumps(safe, indent=2, ensure_ascii=False, default=str),
    )


def _render_redacted_text(value):
    return format_html("<pre>{}</pre>", redact_text(value))
GAME_INPUT_HINTS = {
    "goal",
    "goal_text",
    "iteration",
    "memory",
    "observations",
    "game_response_contract",
}


class GoalPriorityRangeFilter(admin.SimpleListFilter):
    title = _("calculated priority")
    parameter_name = "priority_range"

    def lookups(self, request, model_admin):
        return (("low", _("Low (< 50)")), ("medium", _("Medium (50–99)")), ("high", _("High (100+)")))

    def queryset(self, request, queryset):
        if self.value() == "low":
            return queryset.filter(calculated_priority__lt=50)
        if self.value() == "medium":
            return queryset.filter(calculated_priority__gte=50, calculated_priority__lt=100)
        if self.value() == "high":
            return queryset.filter(calculated_priority__gte=100)
        return queryset


class AIHubFormHelpMixin:
    ai_hub_field_guidance = {}

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        formfield = super().formfield_for_dbfield(db_field, request, **kwargs)
        guidance = self.ai_hub_field_guidance.get(db_field.name, {})
        if not guidance:
            return formfield

        help_text = guidance.get("help_text")
        if help_text:
            formfield.help_text = help_text

        placeholder = guidance.get("placeholder")
        if placeholder and hasattr(formfield.widget, "attrs"):
            input_type = getattr(formfield.widget, "input_type", "")
            if input_type not in {"checkbox", "select", "selectmultiple"}:
                formfield.widget.attrs.setdefault("placeholder", placeholder)
            else:
                formfield.widget.attrs.setdefault("data-placeholder", placeholder)

        rows = guidance.get("rows")
        if rows and hasattr(formfield.widget, "attrs"):
            formfield.widget.attrs.setdefault("rows", rows)

        return formfield


class AIHubListPageMixin(AIHubFormHelpMixin):
    change_list_template = "admin/ai_hub/section_change_list.html"
    change_form_template = "admin/ai_hub/styled_change_form.html"
    ai_hub_section_title = ""
    ai_hub_section_eyebrow = _("AI Hub")
    ai_hub_section_description = ""
    ai_hub_section_note = ""
    ai_hub_section_accent = "provider"
    ai_hub_section_actions = ()

    def get_ai_hub_section_context(self, request):
        return {
            "eyebrow": self.ai_hub_section_eyebrow,
            "title": self.ai_hub_section_title or self.model._meta.verbose_name_plural.title(),
            "description": self.ai_hub_section_description,
            "note": self.ai_hub_section_note,
            "accent": self.ai_hub_section_accent,
            "actions": [
                {
                    "label": action["label"](self) if callable(action["label"]) else action["label"],
                    "url": action["url"](self) if callable(action["url"]) else action["url"],
                    "default": action.get("default", False),
                }
                for action in self.ai_hub_section_actions
            ],
        }

    def changelist_view(self, request, extra_context=None):
        extra_context = {
            **(extra_context or {}),
            "ai_hub_section": self.get_ai_hub_section_context(request),
        }
        return super().changelist_view(request, extra_context=extra_context)


class AIHubHideFromIndexMixin:
    """Keep the model fully registered — URLs, reverse() links, inlines, change pages
    and direct permissions all keep working — but remove it from the admin app index
    and the nav sidebar. Used for bridge tables, structural children, and runtime/audit
    records that belong inside a parent entity or a timeline, not as a top-level
    navigation destination. (Step 3: demote supporting models per the IA blueprint.)"""

    def get_model_perms(self, request):
        return {}


def _status_count(queryset, status):
    return queryset.filter(status=status).count()


def _active_execution_count(queryset):
    return queryset.filter(status__in=ACTIVE_EXECUTION_STATUSES).count()


def _agent_looks_game_ready(agent):
    required_keys = set((agent.input_contract or {}).get("required") or [])
    prompt = (agent.system_prompt or "").lower()
    role = (agent.role or "").lower()
    has_game_contract_hint = bool(required_keys & GAME_INPUT_HINTS)
    has_game_language = "game" in prompt or "game" in role
    has_goal_loop_language = "goal" in prompt and ("action" in prompt or "final_answer" in prompt)
    return has_game_contract_hint or has_game_language or has_goal_loop_language


def _workspace_pipeline_admin(request):
    model_admin = admin.site._registry[PipelineDefinition]
    if (
        not model_admin.has_view_or_change_permission(request)
        or not request.user.has_perm("ai_hub.view_executionsession")
    ):
        raise PermissionDenied
    return model_admin


def ai_hub_orchestrator_workspace_view(request):
    return _workspace_pipeline_admin(request).orchestrator_workspace_view(request)


def ai_hub_game_workspace_view(request):
    return _workspace_pipeline_admin(request).game_workspace_view(request)


def ai_hub_build_wizard_view(request):
    return _workspace_pipeline_admin(request).build_wizard_view(request)


class _WizardRollback(Exception):
    """Raised inside transaction.atomic() to roll back wizard-created objects on validation error."""


def _wizard_build_game(request, data):
    """Create the full GAME chain transactionally. Returns (session, errors_dict)."""
    errors = {}

    # 1. Model config
    engine_mode = data.get("engine_mode", "reuse")
    model_config = None
    if engine_mode == "reuse":
        mid = data.get("engine_reuse_model_id", "")
        try:
            model_config = ModelConfig.objects.get(pk=int(mid))
        except (ModelConfig.DoesNotExist, ValueError, TypeError):
            errors["engine_reuse_model_id"] = _("Select a valid model.")
    else:
        provider_name = (data.get("engine_provider_name") or "").strip()
        if not provider_name:
            errors["engine_provider_name"] = _("Provider name is required.")
        else:
            provider, _ = ProviderConfig.objects.get_or_create(
                name=provider_name,
                defaults={
                    "provider_type": data.get("engine_provider_type") or ProviderConfig.ProviderType.TRAINING,
                    "is_active": True,
                },
            )
            model_name = (data.get("engine_model_name") or "training/starter").strip()
            try:
                temp = Decimal(str(data.get("engine_temperature") or "0.30"))
            except Exception:
                temp = Decimal("0.30")
            model_config, _ = ModelConfig.objects.get_or_create(
                provider=provider,
                model_name=model_name,
                defaults={"temperature_default": temp, "is_active": True},
            )

    if errors:
        return None, errors

    # 2. Agent
    agent_mode = data.get("agent_mode", "reuse")
    agent = None
    if agent_mode == "reuse":
        aid = data.get("agent_reuse_id", "")
        try:
            agent = AgentProfile.objects.get(pk=int(aid))
        except (AgentProfile.DoesNotExist, ValueError, TypeError):
            errors["agent_reuse_id"] = _("Select a valid agent.")
    else:
        agent_name = (data.get("agent_name") or "").strip()
        if not agent_name:
            errors["agent_name"] = _("Agent name is required.")
        else:
            agent = AgentProfile.objects.create(
                name=agent_name,
                role=(data.get("agent_role") or "").strip(),
                system_prompt=(data.get("agent_prompt") or "").strip(),
                model_config=model_config,
                is_active=True,
            )

    if errors:
        return None, errors

    # 3. Toolboxes
    toolbox_ids = data.getlist("agent_toolbox_ids")
    for tb_id in toolbox_ids:
        try:
            tb = Toolbox.objects.get(pk=int(tb_id))
            AgentToolboxAssignment.objects.get_or_create(
                agent=agent, toolbox=tb, defaults={"is_enabled": True}
            )
        except (Toolbox.DoesNotExist, ValueError, TypeError):
            pass

    # 4. Knowledge
    knowledge_mode = data.get("knowledge_mode", "none")
    if knowledge_mode == "reuse":
        try:
            coll = KnowledgeCollection.objects.get(pk=int(data.get("knowledge_collection_id", "")))
            agent.knowledge_collections.add(coll)
        except (KnowledgeCollection.DoesNotExist, ValueError, TypeError):
            pass
    elif knowledge_mode == "create":
        coll_name = (data.get("knowledge_collection_name") or "").strip()
        doc_title  = (data.get("knowledge_doc_title") or "").strip()
        if coll_name and doc_title:
            coll = KnowledgeCollection.objects.create(name=coll_name, is_active=True)
            KnowledgeDocument.objects.create(
                collection=coll,
                title=doc_title,
                curated_text=(data.get("knowledge_doc_content") or "").strip(),
                status=KnowledgeDocument.Status.ACTIVE,
            )
            agent.knowledge_collections.add(coll)

    # 5. Initial context
    raw_ctx = (data.get("initial_context") or "{}").strip()
    try:
        initial_context = json.loads(raw_ctx)
        if not isinstance(initial_context, dict):
            initial_context = {}
    except json.JSONDecodeError:
        initial_context = {}

    # 6. Session
    goal_text = (data.get("goal_text") or "").strip()
    if not goal_text:
        errors["goal_text"] = _("Goal is required.")
        return None, errors

    try:
        max_iter = max(1, min(25, int(data.get("max_iterations") or 3)))
    except (ValueError, TypeError):
        max_iter = 3

    runtime_mode = data.get("runtime_mode") or ExecutionSession.RuntimeMode.ASYNC
    strict = data.get("strict_response_contract") in ("on", "true", "1")
    runtime_config = {"max_iterations": max_iter, "strict_response_contract": strict}

    if data.get("game_flavor") == "advanced":
        # Advanced: workspace → goal → goal-bound session via service.
        # create_goal_execution_session enforces the aihub_unique_active_goal constraint
        # and transitions goal queued → running atomically.
        ws_name = (data.get("workspace_name") or f"Workspace for {agent.name}").strip()
        policy = {
            "allowed_actions": ["submit_for_approval"],
            "safety": {
                "allow_external_writes": False,
                "require_approval_for_medium_risk": data.get("safety_require_approval_medium") in ("on", "true", "1"),
                "require_approval_for_high_risk":   data.get("safety_require_approval_high")   in ("on", "true", "1"),
            },
            "budget": {
                "max_iterations_per_session":  max_iter,
                "max_action_runs_per_session": max(1, int(data.get("budget_max_actions") or 2)),
            },
        }
        workspace, _ = GameWorkspace.objects.get_or_create(
            name=ws_name,
            defaults={"default_policy": policy},
        )
        goal = GameGoal.objects.create(
            workspace=workspace,
            title=goal_text[:200],
            description=goal_text,
        )
        try:
            session = create_goal_execution_session(
                goal=goal,
                entry_agent=agent,
                triggered_by=request.user,
                runtime_config=runtime_config,
            )
        except ValidationError as exc:
            errors["game_flavor"] = str(exc)
            return None, errors
    else:
        # Simple: standalone session, no durable goal record.
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=runtime_mode,
            status=ExecutionSession.Status.PENDING,
            entry_agent=agent,
            triggered_by=request.user,
            source_label=(data.get("source_label") or "").strip(),
            goal_text=goal_text,
            runtime_config=runtime_config,
            initial_context=initial_context,
        )

    return session, {}


def _wizard_build_orchestrator(request, data):
    """Create the full Orchestrator chain transactionally. Returns (pipeline, errors_dict)."""
    errors = {}

    # 1. Model config (same logic as GAME)
    engine_mode = data.get("engine_mode", "reuse")
    model_config = None
    if engine_mode == "reuse":
        mid = data.get("engine_reuse_model_id", "")
        try:
            model_config = ModelConfig.objects.get(pk=int(mid))
        except (ModelConfig.DoesNotExist, ValueError, TypeError):
            errors["engine_reuse_model_id"] = _("Select a valid model.")
    else:
        provider_name = (data.get("engine_provider_name") or "").strip()
        if not provider_name:
            errors["engine_provider_name"] = _("Provider name is required.")
        else:
            provider, _ = ProviderConfig.objects.get_or_create(
                name=provider_name,
                defaults={
                    "provider_type": data.get("engine_provider_type") or ProviderConfig.ProviderType.TRAINING,
                    "is_active": True,
                },
            )
            model_name = (data.get("engine_model_name") or "training/starter").strip()
            try:
                temp = Decimal(str(data.get("engine_temperature") or "0.30"))
            except Exception:
                temp = Decimal("0.30")
            model_config, _ = ModelConfig.objects.get_or_create(
                provider=provider,
                model_name=model_name,
                defaults={"temperature_default": temp, "is_active": True},
            )

    if errors:
        return None, errors

    # 2. Entry agent
    agent_mode = data.get("agent_mode", "reuse")
    entry_agent = None
    if agent_mode == "reuse":
        aid = data.get("agent_reuse_id", "")
        try:
            entry_agent = AgentProfile.objects.get(pk=int(aid))
        except (AgentProfile.DoesNotExist, ValueError, TypeError):
            errors["agent_reuse_id"] = _("Select a valid agent.")
    else:
        agent_name = (data.get("agent_name") or "").strip()
        if not agent_name:
            errors["agent_name"] = _("Agent name is required.")
        else:
            input_contract  = _parse_json_field(data.get("agent_input_contract"), {})
            output_contract = _parse_json_field(data.get("agent_output_contract"), {})
            entry_agent = AgentProfile.objects.create(
                name=agent_name,
                role=(data.get("agent_role") or "").strip(),
                system_prompt=(data.get("agent_prompt") or "").strip(),
                model_config=model_config,
                input_contract=input_contract,
                output_contract=output_contract,
                is_active=True,
            )

    if errors:
        return None, errors

    # 3. Toolboxes + knowledge (same as GAME)
    for tb_id in data.getlist("agent_toolbox_ids"):
        try:
            tb = Toolbox.objects.get(pk=int(tb_id))
            AgentToolboxAssignment.objects.get_or_create(
                agent=entry_agent, toolbox=tb, defaults={"is_enabled": True}
            )
        except (Toolbox.DoesNotExist, ValueError, TypeError):
            pass

    knowledge_mode = data.get("knowledge_mode", "none")
    if knowledge_mode == "reuse":
        try:
            coll = KnowledgeCollection.objects.get(pk=int(data.get("knowledge_collection_id", "")))
            entry_agent.knowledge_collections.add(coll)
        except (KnowledgeCollection.DoesNotExist, ValueError, TypeError):
            pass
    elif knowledge_mode == "create":
        coll_name = (data.get("knowledge_collection_name") or "").strip()
        doc_title  = (data.get("knowledge_doc_title") or "").strip()
        if coll_name and doc_title:
            coll = KnowledgeCollection.objects.create(name=coll_name, is_active=True)
            KnowledgeDocument.objects.create(
                collection=coll,
                title=doc_title,
                curated_text=(data.get("knowledge_doc_content") or "").strip(),
                status=KnowledgeDocument.Status.ACTIVE,
            )
            entry_agent.knowledge_collections.add(coll)

    # 4. Pipeline
    pipeline_name = (data.get("pipeline_name") or "").strip()
    if not pipeline_name:
        errors["pipeline_name"] = _("Pipeline name is required.")
        return None, errors
    if PipelineDefinition.objects.filter(name=pipeline_name).exists():
        errors["pipeline_name"] = _(f'A pipeline named "{pipeline_name}" already exists.')
        return None, errors

    pipeline = PipelineDefinition.objects.create(
        name=pipeline_name,
        description=(data.get("pipeline_description") or "").strip(),
        entry_agent=entry_agent,
        is_active=False,
        input_contract=_parse_json_field(data.get("pipeline_input_contract"), {}),
        output_contract=_parse_json_field(data.get("pipeline_output_contract"), {}),
    )

    # 5. Steps
    step_agents    = data.getlist("step_agent_id")
    step_on_errors = data.getlist("step_on_error")
    step_in_maps   = data.getlist("step_input_mapping")
    step_out_maps  = data.getlist("step_output_mapping")

    for i, agent_id in enumerate(step_agents):
        if not agent_id:
            continue
        try:
            step_agent = AgentProfile.objects.get(pk=int(agent_id))
        except (AgentProfile.DoesNotExist, ValueError, TypeError):
            continue
        PipelineStep.objects.create(
            pipeline=pipeline,
            order=i + 1,
            agent=step_agent,
            on_error=step_on_errors[i] if i < len(step_on_errors) else "fail",
            input_mapping=_parse_json_field(step_in_maps[i] if i < len(step_in_maps) else None, {}),
            output_mapping=_parse_json_field(step_out_maps[i] if i < len(step_out_maps) else None, {}),
        )

    # 6. Activate if requested
    if data.get("pipeline_activate") in ("on", "true", "1"):
        try:
            pipeline.is_active = True
            pipeline.full_clean()
            pipeline.save()
        except Exception:
            pipeline.is_active = False
            pipeline.save()

    return pipeline, {}


def _parse_json_field(raw, default):
    if not raw:
        return default
    try:
        parsed = json.loads(raw.strip())
        return parsed if isinstance(parsed, dict) else default
    except (json.JSONDecodeError, AttributeError):
        return default


# === REUSABLE AI PIPELINE CORE =============================================
# These admin classes manage generic providers, models, tools, agents and
# pipeline definitions.
class GameSessionCreateForm(forms.Form):
    entry_agent = forms.ModelChoiceField(
        queryset=AgentProfile.objects.none(),
        label=_("Entry agent"),
        help_text=_(
            "Choose one active agent. For GAME, pick an agent whose prompt knows how to decide actions, "
            "remember context and finish with a final answer."
        ),
    )
    goal_text = forms.CharField(
        label=_("Goal"),
        widget=forms.Textarea(
            attrs={
                "rows": 5,
                "placeholder": _(
                    "Example: Review this customer request, decide whether more information is needed, "
                    "and finish with the next best response."
                ),
            }
        ),
        help_text=_("Write the outcome you want, the boundaries, and what counts as finished."),
    )
    max_iterations = forms.IntegerField(
        label=_("Max iterations"),
        min_value=1,
        max_value=25,
        initial=3,
        help_text=_("Hard cap for the decision loop. Use 2-3 while testing; increase only after the timeline looks right."),
        widget=forms.NumberInput(attrs={"placeholder": "3"}),
    )
    runtime_mode = forms.ChoiceField(
        label=_("Runtime mode"),
        choices=(
            (ExecutionSession.RuntimeMode.SYNC, _("Sync")),
            (ExecutionSession.RuntimeMode.ASYNC, _("Async")),
        ),
        initial=ExecutionSession.RuntimeMode.ASYNC,
        help_text=_("Async is recommended for long AI runs. Sync is useful only for quick local tests."),
    )
    strict_response_contract = forms.BooleanField(
        label=_("Strict response contract"),
        required=False,
        initial=True,
        help_text=_("Fail the session if the agent does not return the required GAME JSON decision."),
    )
    source_label = forms.CharField(
        label=_("Source label"),
        required=False,
        help_text=_("Optional short name shown in lists and timelines."),
        widget=forms.TextInput(attrs={"placeholder": _("Example: Support triage test #1")}),
    )
    initial_context = forms.CharField(
        label=_("Initial context JSON"),
        required=False,
        initial="{}",
        widget=forms.Textarea(
            attrs={
                "rows": 7,
                "placeholder": '{\n  "customer": {"tier": "pro"},\n  "ticket": "The user cannot upload a file.",\n  "constraints": ["Keep the answer under 120 words"]\n}',
            }
        ),
        help_text=_("Optional JSON object passed to the agent before the first iteration. Use keys the prompt can understand."),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["entry_agent"].queryset = AgentProfile.objects.filter(is_active=True).order_by("name")

    def clean_initial_context(self):
        raw_value = (self.cleaned_data.get("initial_context") or "{}").strip()
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise forms.ValidationError(_("Initial context must be valid JSON.")) from exc
        if not isinstance(parsed, dict):
            raise forms.ValidationError(_("Initial context must be a JSON object."))
        return parsed

    def save(self, *, user):
        session = ExecutionSession(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=self.cleaned_data["runtime_mode"],
            status=ExecutionSession.Status.PENDING,
            entry_agent=self.cleaned_data["entry_agent"],
            triggered_by=user,
            source_label=self.cleaned_data.get("source_label", ""),
            goal_text=self.cleaned_data["goal_text"],
            runtime_config={
                "max_iterations": self.cleaned_data["max_iterations"],
                "strict_response_contract": self.cleaned_data["strict_response_contract"],
            },
            initial_context=self.cleaned_data["initial_context"],
        )
        session.full_clean()
        session.save()
        return session


class PipelineStepInline(AIHubFormHelpMixin, admin.TabularInline):
    model = PipelineStep
    extra = 0
    autocomplete_fields = ("agent", "fallback_agent")
    fields = ("order", "agent", "input_mapping", "output_mapping", "on_error", "fallback_agent")
    show_change_link = True
    verbose_name_plural = "4.1 Pipeline steps - ordered agent connections"
    ai_hub_field_guidance = {
        "order": {
            "placeholder": "1",
            "help_text": _("Use 1, 2, 3 with no gaps."),
        },
        "agent": {
            "help_text": _("The agent that runs at this point in the workflow."),
        },
        "input_mapping": {
            "rows": 5,
            "placeholder": '{"ticket_text": "ticket_text"}',
            "help_text": _("Agent input key on the left, current context path on the right."),
        },
        "output_mapping": {
            "rows": 5,
            "placeholder": '{"triage_result": "agent", "model_text": "llm.content"}',
            "help_text": _("Context key on the left, agent response path on the right."),
        },
        "on_error": {
            "help_text": _("Stop is safest while testing."),
        },
        "fallback_agent": {
            "help_text": _("Only used when on_error is Fallback Agent."),
        },
    }


class KnowledgeDocumentInline(AIHubFormHelpMixin, admin.TabularInline):
    model = KnowledgeDocument
    extra = 0
    fields = ("title", "language", "status", "tags", "updated_at")
    readonly_fields = ("updated_at",)
    show_change_link = True
    verbose_name = "Knowledge document"
    verbose_name_plural = "1.1 Knowledge documents - attach curated context to this group"
    ai_hub_field_guidance = {
        "title": {
            "placeholder": "Example: Refund policy summary",
            "help_text": _("Short searchable title."),
        },
        "language": {
            "placeholder": "en",
            "help_text": _("Optional language code."),
        },
        "tags": {
            "placeholder": '["policy", "support"]',
            "help_text": _("Optional JSON list of tags."),
        },
        "status": {
            "help_text": _("Only active documents are injected into agents."),
        },
    }


class KnowledgeDocumentChunkInline(AIHubFormHelpMixin, admin.TabularInline):
    model = KnowledgeDocumentChunk
    extra = 0
    fields = ("chunk_index", "section_title", "token_estimate", "updated_at")
    readonly_fields = ("updated_at",)
    show_change_link = True
    verbose_name = "Knowledge chunk"
    verbose_name_plural = "1.2 Knowledge chunks - retrievable sections"


@admin.register(ProviderConfig)
class ProviderConfigAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Providers")
    ai_hub_section_description = _(
        "Connect the AI services your agents will call. Keep a provider inactive until the base URL, credentials "
        "and model names are ready."
    )
    ai_hub_section_note = _(
        "This is the first stop in almost every setup. Add one working provider before you create models."
    )
    ai_hub_section_accent = "provider"
    ai_hub_section_actions = (
        {"label": _("Add provider"), "url": lambda self: reverse("admin:ai_hub_providerconfig_add"), "default": True},
        {"label": _("Open control center"), "url": lambda self: reverse("admin:ai_hub_control_center")},
    )
    list_display = ("name", "provider_type", "base_url", "api_key_env_var", "is_active", "default_timeout")
    list_filter = ("provider_type", "is_active")
    search_fields = ("name", "api_key_env_var", "base_url")
    ai_hub_field_guidance = {
        "name": {
            "placeholder": "Example: Ollama LAN or OpenAI Production",
            "help_text": _("A friendly name used in admin lists. It does not need to match the provider company name exactly."),
        },
        "provider_type": {
            "help_text": _("Choose the closest provider type. Use Other for LiteLLM-compatible custom endpoints."),
        },
        "base_url": {
            "placeholder": "Example: http://localhost:11434",
            "help_text": _("Required for local/custom endpoints such as Ollama. Leave blank for providers that use standard API URLs."),
        },
        "api_key_env_var": {
            "placeholder": "Example: OPENAI_API_KEY",
            "help_text": _("Write the environment variable name only. Never paste the secret API key here."),
        },
        "default_timeout": {
            "placeholder": "60",
            "help_text": _("Maximum seconds to wait for a provider call before AI Hub marks the step as failed."),
        },
    }
    fieldsets = (
        ("0.1 Provider identity", {
            "fields": ("name", "provider_type", "is_active"),
            "description": _(
                "Start here. A provider is the external or local AI service account, such as OpenAI, Ollama, "
                "DeepSeek or any LiteLLM-compatible endpoint. Keep inactive until credentials and connection "
                "settings are ready."
            ),
        }),
        ("Connection", {
            "fields": ("base_url", "default_timeout"),
            "description": _(
                "Use base URL only for local/custom providers. Timeout is the maximum seconds an agent call "
                "should wait before failing the step."
            ),
        }),
        ("Credentials (env-only)", {
            "fields": ("api_key_env_var",),
            "description": _(
                "Store only the environment variable name, never the secret itself. Example: OPENAI_API_KEY."
            ),
        }),
    )


@admin.register(ModelConfig)
class ModelConfigAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Models")
    ai_hub_section_description = _(
        "Define the exact model name each agent should use. Keep the provider/model pair consistent so the "
        "runtime stays predictable."
    )
    ai_hub_section_note = _(
        "If a provider is active but no model appears here, agents cannot call it yet."
    )
    ai_hub_section_accent = "model"
    ai_hub_section_actions = (
        {"label": _("Add model"), "url": lambda self: reverse("admin:ai_hub_modelconfig_add"), "default": True},
        {"label": _("View providers"), "url": lambda self: reverse("admin:ai_hub_providerconfig_changelist")},
    )
    list_display = ("provider", "model_name", "supports_tools", "temperature_default", "max_tokens_default", "is_active")
    list_filter = ("provider", "supports_tools", "is_active")
    search_fields = ("model_name", "provider__name")
    autocomplete_fields = ("provider",)
    ai_hub_field_guidance = {
        "provider": {
            "help_text": _("The provider account or endpoint this model belongs to."),
        },
        "model_name": {
            "placeholder": "Examples: gpt-4.1-mini, ollama/qwen3:8b, deepseek-chat",
            "help_text": _(
                "Use the exact model identifier expected by the provider or LiteLLM adapter. "
                "For the Training (stub) provider the name must be 'training' or start with "
                "'training/' (e.g. 'training/assistant') so it routes to the deterministic stub."
            ),
        },
        "temperature_default": {
            "placeholder": "0.20 for extraction, 0.70 for writing",
            "help_text": _("Lower values are more consistent. Higher values are more creative but less predictable."),
        },
        "max_tokens_default": {
            "placeholder": "1000",
            "help_text": _("Upper limit for generated output. Increase for long summaries or detailed final answers."),
        },
        "supports_tools": {
            "help_text": _("Enable only if this model can call tools/function schemas reliably."),
        },
    }
    fieldsets = (
        ("0.2 Model choice", {
            "fields": ("provider", "model_name", "is_active"),
            "description": _(
                "Create models after providers. Agents point here, so keep model names exact for the provider "
                "or LiteLLM adapter you are using."
            ),
        }),
        ("Default generation settings", {
            "fields": ("temperature_default", "max_tokens_default", "supports_tools"),
            "description": _(
                "These defaults apply when an agent uses this model. Lower temperature is better for structured "
                "extraction; higher temperature can help interpretation/writing agents."
            ),
        }),
    )


class ToolboxToolInline(AIHubFormHelpMixin, admin.TabularInline):
    model = ToolboxTool
    extra = 0
    autocomplete_fields = ("tool",)
    fields = ("tool", "is_enabled", "default_enabled", "display_order")


class AgentToolboxAssignmentInline(AIHubFormHelpMixin, admin.TabularInline):
    model = AgentToolboxAssignment
    extra = 0
    autocomplete_fields = ("toolbox",)
    fields = ("toolbox", "is_enabled", "created_at")
    readonly_fields = ("created_at",)


class AgentToolGrantInline(AIHubFormHelpMixin, admin.TabularInline):
    model = AgentToolGrant
    extra = 0
    autocomplete_fields = ("tool",)
    fields = ("tool", "is_enabled", "permission_level", "requires_approval_override", "notes", "created_at")
    readonly_fields = ("created_at",)


@admin.register(ToolDefinition)
class ToolDefinitionAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Tools")
    ai_hub_section_description = _(
        "Optional capabilities attached to agents. Keep tools explicit and small so the runtime remains easy "
        "to debug."
    )
    ai_hub_section_note = _(
        "Tools are attached to agents, not to the whole workspace."
    )
    ai_hub_section_accent = "tool"
    ai_hub_section_actions = (
        {"label": _("Add tool"), "url": lambda self: reverse("admin:ai_hub_tooldefinition_add"), "default": True},
        {"label": _("Open agents"), "url": lambda self: reverse("admin:ai_hub_agentprofile_changelist")},
    )
    list_display = ("name", "label", "tool_kind", "operation_mode", "risk_level", "requires_approval", "is_active")
    list_filter = ("tool_kind", "operation_mode", "risk_level", "requires_approval", "is_system_tool", "is_active")
    search_fields = ("name", "label", "description")
    ai_hub_field_guidance = {
        "name": {
            "placeholder": "Example: fetch_customer_profile",
            "help_text": _("Use a short action-style name. Agents will read this as a capability."),
        },
        "tool_kind": {
            "help_text": _("HTTP for API calls, Python callable for internal functions, Prompt macro for reusable prompt snippets."),
        },
        "operation_mode": {
            "help_text": _("Classify side effects: read, draft write, state write, external write or execute."),
        },
        "risk_level": {
            "help_text": _("Risk classification used by future permission and approval checks."),
        },
        "requires_approval": {
            "help_text": _("Marks this capability as approval-gated when the unified runtime is enabled."),
        },
        "input_schema": {
            "rows": 8,
            "placeholder": '{\n  "required": ["customer_id"],\n  "properties": {\n    "customer_id": {"type": "string"}\n  }\n}',
            "help_text": _("JSON schema-style description of what the tool needs from the agent."),
        },
        "output_schema": {
            "rows": 8,
            "placeholder": '{\n  "required": ["status", "profile"],\n  "properties": {\n    "status": {"type": "string"},\n    "profile": {"type": "object"}\n  }\n}',
            "help_text": _("JSON schema-style description of what the tool returns to the agent."),
        },
        "config": {
            "rows": 8,
            "placeholder": '{\n  "url": "https://api.example.com/customers/{customer_id}",\n  "method": "GET",\n  "timeout": 10\n}',
            "help_text": _("Operational settings for the tool. Keep secrets in environment variables, not here."),
        },
    }
    fieldsets = (
        ("2.0 Tool core", {
            "fields": ("name", "label", "description", "tool_kind", "is_active"),
            "description": _(
                "Optional capabilities that agents may use. Runtime execution is intentionally controlled; "
                "only activate tools you expect an agent to call."
            ),
        }),
        ("Safety metadata", {
            "fields": ("operation_mode", "risk_level", "requires_approval", "is_system_tool"),
            "description": _(
                "Classify operational behavior before a runtime uses this tool. "
                "Side-effecting tools should not be treated as read-only."
            ),
        }),
        ("Tool contracts", {
            "fields": ("input_schema", "output_schema"),
            "description": _(
                "Describe what the tool accepts and returns. Keep required keys explicit so pipeline errors "
                "are visible before they become confusing model behavior."
            ),
        }),
        ("Tool runtime config", {
            "fields": ("config",),
            "description": _(
                "Operational settings for the tool. V1 keeps execution constrained, so use this as the future "
                "place for URLs, callable names, macro settings or allowlisted options."
            ),
        }),
    )


@admin.register(Toolbox)
class ToolboxAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Toolboxes")
    ai_hub_section_description = _(
        "Group reusable tools into named bundles that can be assigned to many agents."
    )
    ai_hub_section_note = _("Toolboxes are additive access groups; agent grants can still allow or deny individual tools.")
    ai_hub_section_accent = "tool"
    list_display = ("name", "label", "is_active", "tool_count", "updated_at")
    list_filter = ("is_active",)
    search_fields = ("name", "slug", "label", "description")
    prepopulated_fields = {"slug": ("name",)}
    inlines = [ToolboxToolInline]

    fieldsets = (
        ("2.1 Toolbox identity", {
            "fields": ("name", "slug", "label", "description", "is_active"),
            "description": _("Create a reusable group of related tools for agent roles."),
        }),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(tools_total=Count("tool_entries", distinct=True))

    def tool_count(self, obj):
        return obj.tools_total

    tool_count.short_description = _("tools")


@admin.register(ToolboxTool)
class ToolboxToolAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Toolbox tools")
    ai_hub_section_description = _("Membership records connecting tools to toolboxes.")
    ai_hub_section_accent = "tool"
    list_display = ("toolbox", "tool", "display_order", "is_enabled", "default_enabled", "created_at")
    list_filter = ("is_enabled", "default_enabled", "toolbox")
    search_fields = ("toolbox__name", "toolbox__label", "tool__name", "tool__label")
    autocomplete_fields = ("toolbox", "tool")


@admin.register(KnowledgeCollection)
class KnowledgeCollectionAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Knowledge collections")
    ai_hub_section_description = _(
        "Group documents by the kind of context your agents should read. Collections are the container, "
        "documents are the content."
    )
    ai_hub_section_note = _(
        "Keep active collections tight and relevant so agents do not receive noisy context."
    )
    ai_hub_section_accent = "knowledge"
    ai_hub_section_actions = (
        {"label": _("Add collection"), "url": lambda self: reverse("admin:ai_hub_knowledgecollection_add"), "default": True},
        {"label": _("View documents"), "url": lambda self: reverse("admin:ai_hub_knowledgedocument_changelist")},
    )
    list_display = ("name", "is_active", "documents_count", "updated_at")
    list_filter = ("is_active",)
    search_fields = ("name", "description", "documents__title", "documents__curated_text")
    inlines = [KnowledgeDocumentInline]
    ai_hub_field_guidance = {
        "name": {
            "placeholder": "Example: Safety rules, Product docs, Jungian symbols",
            "help_text": _("Name the collection by the kind of knowledge agents should receive."),
        },
        "description": {
            "rows": 4,
            "placeholder": "Example: Short product rules that support agents should follow before answering users.",
            "help_text": _("Explain when agents should use this collection and what kind of material it contains."),
        },
        "is_active": {
            "help_text": _("Inactive collections are ignored by the runtime even if agents are linked to them."),
        },
    }
    fieldsets = (
        ("1.0 Knowledge collection", {
            "fields": ("name", "description", "is_active"),
            "description": _(
                "Group documents by the way agents should consume them: general symbols, Jungian references, "
                "cultural context, product rules, safety guidance, etc. Agents attach to collections, not "
                "individual hardcoded files."
            ),
        }),
    )

    @admin.display(description="Documents")
    def documents_count(self, obj):
        return obj.documents.count()


@admin.register(KnowledgeDocument)
class KnowledgeDocumentAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Knowledge documents")
    ai_hub_section_description = _(
        "Write the actual context agents should read: curated text, uploaded references and editorial notes."
    )
    ai_hub_section_note = _(
        "Use draft status while editing; only active documents from active collections are injected into agents."
    )
    ai_hub_section_accent = "knowledge"
    ai_hub_section_actions = (
        {"label": _("Add document"), "url": lambda self: reverse("admin:ai_hub_knowledgedocument_add"), "default": True},
        {"label": _("View collections"), "url": lambda self: reverse("admin:ai_hub_knowledgecollection_changelist")},
    )
    list_display = ("title", "collection", "status", "language", "updated_at")
    list_filter = ("status", "collection", "language")
    search_fields = ("title", "curated_text", "notes", "tags", "collection__name")
    autocomplete_fields = ("collection",)
    inlines = [KnowledgeDocumentChunkInline]
    ai_hub_field_guidance = {
        "title": {
            "placeholder": "Example: Refund policy summary",
            "help_text": _("Use a title that makes the document easy to find from the admin search."),
        },
        "collection": {
            "help_text": _("The knowledge group this document belongs to."),
        },
        "language": {
            "placeholder": "Example: en",
            "help_text": _("Optional language code. Use en, es, fr, etc. Keep it consistent if you filter by language later."),
        },
        "tags": {
            "rows": 4,
            "placeholder": '["refunds", "policy", "support"]',
            "help_text": _("Optional JSON list of labels for searching and organizing documents."),
        },
        "curated_text": {
            "rows": 12,
            "placeholder": "Example: Agents should offer a refund only when the order is less than 30 days old...",
            "help_text": _("The exact knowledge injected into agents. Prefer concise, curated text over long raw files."),
        },
        "source_file": {
            "help_text": _("Optional uploaded reference. Text files can be read when curated text is empty."),
        },
        "notes": {
            "rows": 5,
            "placeholder": "Example: Reviewed by support team on 2026-06-06. Do not use for legal advice.",
            "help_text": _("Private admin notes. These are for humans and should explain provenance or limitations."),
        },
        "status": {
            "help_text": _("Draft is editable and ignored by agents. Active is injected. Archived is kept for history."),
        },
    }
    fieldsets = (
        ("1.1 Document identity", {
            "fields": ("title", "collection", "status", "language", "tags"),
            "description": _(
                "Use status=draft while preparing material. Only active documents from active collections are "
                "injected into agent payloads."
            ),
        }),
        ("Knowledge content", {
            "fields": ("curated_text", "source_file"),
            "description": _(
                "Prefer curated_text for v1 because it is visible, searchable and predictable. source_file is "
                "available for uploaded references; text files are read as UTF-8 when curated_text is empty."
            ),
        }),
        ("Editorial notes", {
            "fields": ("notes",),
            "description": _("Private notes for provenance, quality concerns or how this document should be used."),
        }),
    )


@admin.register(KnowledgeDocumentChunk)
class KnowledgeDocumentChunkAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Knowledge chunks")
    ai_hub_section_description = _("Retrievable sections inside knowledge documents.")
    ai_hub_section_note = _(
        "Agents should search and read selected chunks instead of receiving entire documents by default."
    )
    ai_hub_section_accent = "knowledge"
    list_display = ("document", "chunk_index", "section_title", "token_estimate", "updated_at")
    list_filter = ("document__collection", "document__status")
    search_fields = ("document__title", "section_title", "content")
    autocomplete_fields = ("document",)
    fieldsets = (
        ("1.2 Chunk identity", {
            "fields": ("document", "chunk_index", "section_title", "token_estimate"),
            "description": _("One retrievable section of a knowledge document."),
        }),
        ("Content", {
            "fields": ("content", "metadata"),
            "description": _("Chunk content and source metadata used for retrieval and citation."),
        }),
    )


@admin.register(AgentProfile)
class AgentProfileAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Agents")
    ai_hub_section_description = _(
        "Define a specialist once, then reuse it in Orchestrator, GAME or both. This is where prompts, "
        "knowledge and tools come together."
    )
    ai_hub_section_note = _(
        "The workspace badge shows where each agent is already being used."
    )
    ai_hub_section_accent = "agent"
    ai_hub_section_actions = (
        {"label": _("Add agent"), "url": lambda self: reverse("admin:ai_hub_agentprofile_add"), "default": True},
        {"label": _("Open Orchestrator"), "url": lambda self: reverse("admin:ai_hub_workspace_orchestrator")},
        {"label": _("Open GAME"), "url": lambda self: reverse("admin:ai_hub_workspace_game")},
    )
    list_display = (
        "name",
        "role",
        "model_config",
        "workspace_usage",
        "pipeline_usage_count",
        "game_session_count",
        "is_active",
    )
    list_filter = (
        "is_active",
        "execution_mode",
        "knowledge_collections",
        "model_config__provider__provider_type",
        "model_config__supports_tools",
    )
    search_fields = ("name", "role", "model_config__model_name", "knowledge_collections__name")
    autocomplete_fields = ("model_config", "tools", "knowledge_collections")
    filter_horizontal = ("tools", "knowledge_collections")
    inlines = [AgentToolboxAssignmentInline, AgentToolGrantInline]
    readonly_fields = ("resolved_tool_manifest",)
    ai_hub_field_guidance = {
        "name": {
            "placeholder": "Example: support_ticket_triage or symbol_extractor",
            "help_text": _("Use a short machine-friendly name. This appears in timelines and pipeline steps."),
        },
        "role": {
            "placeholder": "Example: Classifies support tickets and chooses the next support action.",
            "help_text": _("Plain-English job description. A non-technical admin should understand what this agent does."),
        },
        "execution_mode": {
            "help_text": _("Inherit is usually best. Override only when this agent must always run sync or async."),
        },
        "model_config": {
            "help_text": _("The model this agent calls. Choose a reliable model for JSON/contract-heavy agents."),
        },
        "tools": {
            "help_text": _(
                "Legacy direct tools for this agent. Prefer reusable toolboxes and grants for new access."
            ),
        },
        "knowledge_collections": {
            "help_text": _("Optional knowledge groups injected into this agent's prompt context."),
        },
        "knowledge_max_chars": {
            "placeholder": "6000",
            "help_text": _("Maximum knowledge characters passed to this agent. Smaller is faster and easier to debug."),
        },
        "system_prompt": {
            "rows": 14,
            "placeholder": "Example:\\nYou are support_ticket_triage.\\nRead the user request and return valid JSON only.\\nDecide whether the issue is billing, technical, account, or other.\\nDo not write the final customer reply.",
            "help_text": _("Tell the agent its role, boundaries, output format and what it must not do."),
        },
        "input_contract": {
            "rows": 9,
            "placeholder": '{\n  "required": ["ticket_text"],\n  "properties": {\n    "ticket_text": {"type": "string"},\n    "knowledge_context": {"type": "object"}\n  }\n}',
            "help_text": _("JSON schema-style contract for what this agent expects in the context."),
        },
        "output_contract": {
            "rows": 9,
            "placeholder": '{\n  "required": ["category", "priority", "reason"],\n  "properties": {\n    "category": {"type": "string"},\n    "priority": {"type": "string"},\n    "reason": {"type": "string"}\n  }\n}',
            "help_text": _("JSON schema-style contract for what later steps can expect from this agent."),
        },
        "is_active": {
            "help_text": _("Inactive agents cannot be used in active pipelines or new execution sessions."),
        },
    }
    fieldsets = (
        ("3.0 Agent identity", {
            "fields": ("name", "role", "is_active"),
            "description": _(
                "An agent is one specialist in the pipeline. Name it by job, for example symbol_extractor, "
                "knowledge_interpreter or interpretation_writer."
            ),
        }),
        ("Execution", {
            "fields": ("execution_mode", "model_config", "tools"),
            "description": _(
                "Choose the model and optional tools this agent can use. Inherit means the pipeline/run mode "
                "decides whether the step is sync or async."
            ),
        }),
        ("Resolved tool access", {
            "fields": ("resolved_tool_manifest",),
            "description": _(
                "Read-only view of the effective tool manifest after toolboxes, grants, legacy direct tools "
                "and safety metadata have been applied. Workspace policy may restrict this further at runtime."
            ),
        }),
        ("Knowledge", {
            "fields": ("knowledge_collections", "knowledge_max_chars"),
            "description": _(
                "Attach the knowledge groups this agent should read. The runtime injects active documents as "
                "knowledge_context and trims them to knowledge_max_chars."
            ),
        }),
        ("Prompting", {
            "fields": ("system_prompt",),
            "description": _(
                "Tell this agent exactly what role it has, what output shape to produce, and what not to decide. "
                "Keep prompts narrow so each pipeline step remains easy to debug."
            ),
        }),
        ("Contracts", {
            "fields": ("input_contract", "output_contract"),
            "description": _(
                "Use JSON-schema-like required keys to make the data contract explicit. Agents that need DB "
                "knowledge can require knowledge_context in the input contract."
            ),
        }),
    )

    class Media:
        css = {"all": ("ai_hub/admin_control_center.css",)}

    def get_queryset(self, request):
        return (
            super()
            .get_queryset(request)
            .select_related("model_config", "model_config__provider")
            .annotate(
                pipeline_steps_total=Count("pipeline_steps", distinct=True),
                game_sessions_total=Count(
                    "entry_execution_sessions",
                    filter=Q(entry_execution_sessions__runtime_kind=ExecutionSession.RuntimeKind.GAME),
                    distinct=True,
                ),
            )
        )

    @admin.display(description="Workspace")
    def workspace_usage(self, obj):
        pipeline_total = getattr(obj, "pipeline_steps_total", None)
        if pipeline_total is None:
            pipeline_total = obj.pipeline_steps.count()
        game_total = getattr(obj, "game_sessions_total", None)
        if game_total is None:
            game_total = obj.entry_execution_sessions.filter(runtime_kind=ExecutionSession.RuntimeKind.GAME).count()
        in_orchestrator = pipeline_total > 0
        in_game = game_total > 0
        if in_orchestrator and in_game:
            label = "Both"
            status = "both"
        elif in_orchestrator:
            label = "Orchestrator"
            status = "orchestrator"
        elif in_game:
            label = "GAME"
            status = "game"
        else:
            label = "Unused"
            status = "unused"
        return format_html('<span class="ai-admin-badge ai-admin-badge--{}">{}</span>', status, label)

    @admin.display(description="Pipelines")
    def pipeline_usage_count(self, obj):
        pipeline_total = getattr(obj, "pipeline_steps_total", None)
        if pipeline_total is None:
            pipeline_total = obj.pipeline_steps.count()
        return pipeline_total

    @admin.display(description="GAME sessions")
    def game_session_count(self, obj):
        game_total = getattr(obj, "game_sessions_total", None)
        if game_total is None:
            game_total = obj.entry_execution_sessions.filter(runtime_kind=ExecutionSession.RuntimeKind.GAME).count()
        return game_total

    @admin.display(description="Resolved tool manifest")
    def resolved_tool_manifest(self, obj):
        if not obj or not obj.pk:
            return _("Save the agent before inspecting resolved tools.")
        manifest = resolve_agent_tools(obj).manifest()
        if not manifest:
            return _("No active tools are currently resolved for this agent.")
        return format_html(
            "<pre>{}</pre>",
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        )


@admin.register(AgentToolboxAssignment)
class AgentToolboxAssignmentAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Agent toolbox assignments")
    ai_hub_section_description = _("Attach reusable toolboxes to agents without duplicating direct tool lists.")
    ai_hub_section_accent = "agent"
    list_display = ("agent", "toolbox", "is_enabled", "created_at")
    list_filter = ("is_enabled", "toolbox")
    search_fields = ("agent__name", "agent__role", "toolbox__name", "toolbox__label")
    autocomplete_fields = ("agent", "toolbox")


@admin.register(AgentToolGrant)
class AgentToolGrantAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Agent tool grants")
    ai_hub_section_description = _("Allow or deny one specific tool for one agent as an explicit override.")
    ai_hub_section_accent = "agent"
    list_display = (
        "agent",
        "tool",
        "is_enabled",
        "permission_level",
        "requires_approval_override",
        "created_at",
    )
    list_filter = ("is_enabled", "permission_level", "requires_approval_override")
    search_fields = ("agent__name", "agent__role", "tool__name", "tool__label")
    autocomplete_fields = ("agent", "tool")


@admin.register(PipelineDefinition)
class PipelineDefinitionAdmin(AIHubFormHelpMixin, admin.ModelAdmin):
    change_list_template = "admin/ai_hub/pipelinedefinition/change_list.html"
    change_form_template = "admin/ai_hub/styled_change_form.html"
    list_display = ("name", "is_active", "entry_agent", "updated_at")
    list_filter = ("is_active",)
    search_fields = ("name", "description")
    autocomplete_fields = ("entry_agent",)
    inlines = [PipelineStepInline]
    ai_hub_field_guidance = {
        "name": {
            "placeholder": "Example: support_triage_v1",
            "help_text": _("Use a stable name. Add v1/v2 when you make meaningful changes to the workflow."),
        },
        "description": {
            "rows": 4,
            "placeholder": "Example: Reads a support ticket, classifies it, checks policy knowledge, then drafts a response.",
            "help_text": _("Explain the workflow in plain English so another admin can understand when to use it."),
        },
        "is_active": {
            "help_text": _("Keep inactive while building. Activation validates that steps are continuous and agents have contracts."),
        },
        "entry_agent": {
            "help_text": _("Descriptive entry agent for the workflow. The ordered steps below are the execution source of truth."),
        },
        "global_input_contract": {
            "rows": 8,
            "placeholder": '{\n  "required": ["ticket_text"],\n  "properties": {\n    "ticket_text": {"type": "string"},\n    "user_profile": {"type": "object"}\n  }\n}',
            "help_text": _("Optional JSON schema-style contract for the initial context required by the workflow."),
        },
        "global_output_contract": {
            "rows": 8,
            "placeholder": '{\n  "required": ["final_response"],\n  "properties": {\n    "final_response": {"type": "string"},\n    "confidence": {"type": "number"}\n  }\n}',
            "help_text": _("Optional JSON schema-style contract for the final context after all steps complete."),
        },
    }
    fieldsets = (
        ("4.0 Pipeline core", {
            "fields": ("name", "description", "is_active"),
            "description": _(
                "A pipeline is the full recipe. Keep inactive while building steps; activation validates that "
                "steps are continuous and agents have contracts."
            ),
        }),
        ("Orchestration", {
            "fields": ("entry_agent",),
            "description": _(
                "V1 executes steps sequentially by order. entry_agent is descriptive for now; the inline steps "
                "below are the source of truth for execution."
            ),
        }),
        ("Global contracts", {
            "fields": ("global_input_contract", "global_output_contract"),
            "description": _(
                "The runner starts with dream_id, title, content, input_meta, dream_context, user_context and "
                "pipeline. Use output contract only when the final shape is stable."
            ),
        }),
    )

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "orchestrator/",
                self.admin_site.admin_view(self.orchestrator_workspace_view),
                name="ai_hub_orchestrator_workspace",
            ),
            path(
                "game/",
                self.admin_site.admin_view(self.game_workspace_view),
                name="ai_hub_game_workspace",
            ),
            path(
                "control-center/",
                self.admin_site.admin_view(self.control_center_view),
                name="ai_hub_control_center",
            ),
            path(
                "control-center/data/",
                self.admin_site.admin_view(self.control_center_data_view),
                name="ai_hub_control_center_data",
            ),
            path(
                "build/",
                self.admin_site.admin_view(self.build_wizard_view),
                name="ai_hub_pipelinedefinition_build",
            ),
        ]
        return custom_urls + urls

    def orchestrator_workspace_view(self, request):
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied
        session_queryset = ExecutionSession.objects.filter(runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR)
        pipeline_queryset = PipelineDefinition.objects.all()
        context = {
            **self.admin_site.each_context(request),
            "title": _("Orchestrator workspace"),
            "metrics": {
                "pipelines": pipeline_queryset.count(),
                "active_pipelines": pipeline_queryset.filter(is_active=True).count(),
                "sessions": session_queryset.count(),
                "active_sessions": _active_execution_count(session_queryset),
                "failed_sessions": _status_count(session_queryset, ExecutionSession.Status.FAILED),
            },
            "pipelines": pipeline_queryset.prefetch_related("steps").order_by("name"),
            "recent_sessions": ExecutionSession.objects.select_related("pipeline", "entry_agent")
            .filter(runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR)
            .order_by("-created_at")[:8],
        }
        return TemplateResponse(request, "admin/ai_hub/workspaces/orchestrator.html", context)

    def game_workspace_view(self, request):
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied
        session_queryset = ExecutionSession.objects.filter(runtime_kind=ExecutionSession.RuntimeKind.GAME)
        candidate_agents = (
            AgentProfile.objects.select_related("model_config", "model_config__provider")
            .annotate(
                game_sessions_total=Count(
                    "entry_execution_sessions",
                    filter=Q(entry_execution_sessions__runtime_kind=ExecutionSession.RuntimeKind.GAME),
                    distinct=True,
                )
            )
            .filter(is_active=True)
            .order_by("name")
        )
        game_agents = []
        for agent in candidate_agents:
            if agent.game_sessions_total or _agent_looks_game_ready(agent):
                agent.game_fit_label = _("Used in GAME") if agent.game_sessions_total else _("GAME-ready")
                game_agents.append(agent)
            if len(game_agents) >= 12:
                break

        context = {
            **self.admin_site.each_context(request),
            "title": _("GAME workspace"),
            **build_game_graph_context(),
            "metrics": {
                "sessions": session_queryset.count(),
                "active_sessions": _active_execution_count(session_queryset),
                "successful_sessions": _status_count(session_queryset, ExecutionSession.Status.SUCCESS),
                "failed_sessions": _status_count(session_queryset, ExecutionSession.Status.FAILED),
                "recommended_agents": len(game_agents),
            },
            "game_sessions": ExecutionSession.objects.select_related("entry_agent")
            .filter(runtime_kind=ExecutionSession.RuntimeKind.GAME)
            .order_by("-created_at")[:12],
            "game_agents": game_agents,
        }
        return TemplateResponse(request, "admin/ai_hub/workspaces/game.html", context)

    def control_center_view(self, request):
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied
        context = {
            **self.admin_site.each_context(request),
            "title": _("AI Hub Control Center"),
            **build_control_center_context(),
        }
        return TemplateResponse(request, "admin/ai_hub/control_center.html", context)

    def control_center_data_view(self, request):
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied
        context = build_control_center_context()
        return JsonResponse({"graph": context["graph"], "metrics": context["metrics"], "warnings": context["warnings"]})

    def build_wizard_view(self, request):
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied

        kind = request.GET.get("kind", "game")
        if kind not in ("game", "orchestrator"):
            kind = "game"

        errors = {}
        success_url = None

        if request.method == "POST":
            wizard_kind = request.POST.get("wizard_kind", kind)
            try:
                with transaction.atomic():
                    if wizard_kind == "orchestrator":
                        obj, errors = _wizard_build_orchestrator(request, request.POST)
                    else:
                        obj, errors = _wizard_build_game(request, request.POST)
                    if errors:
                        raise _WizardRollback()
                    if wizard_kind == "orchestrator":
                        self.message_user(
                            request,
                            _(f'Pipeline "{obj.name}" created. Add steps and activate when ready.'),
                            level=messages.SUCCESS,
                        )
                        success_url = reverse("admin:ai_hub_pipelinedefinition_change", args=[obj.id])
                    else:
                        self.message_user(
                            request,
                            _(f"GAME session #{obj.id} created. Review it and run it from the session list."),
                            level=messages.SUCCESS,
                        )
                        success_url = reverse("admin:ai_hub_executionsession_change", args=[obj.id])
            except _WizardRollback:
                pass

            if success_url and not errors:
                return HttpResponseRedirect(success_url)

            kind = request.POST.get("wizard_kind", kind)

        context = {
            **self.admin_site.each_context(request),
            "title": _("Build Console"),
            "kind": kind,
            "errors": errors,
            "agents": AgentProfile.objects.filter(is_active=True).order_by("name"),
            "model_configs": ModelConfig.objects.select_related("provider").filter(is_active=True).order_by("provider__name", "model_name"),
            "provider_types": ProviderConfig.ProviderType.choices,
            "toolboxes": Toolbox.objects.prefetch_related("tool_entries__tool").filter(is_active=True).order_by("name"),
            "knowledge_collections": KnowledgeCollection.objects.filter(is_active=True).order_by("name"),
            "pipelines": PipelineDefinition.objects.order_by("name"),
        }
        return TemplateResponse(request, "admin/ai_hub/workspaces/build.html", context)


@admin.register(PipelineStep)
class PipelineStepAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Pipeline steps")
    ai_hub_section_description = _(
        "Each row is one handoff between agents. Keep the order continuous and the input/output mappings clear."
    )
    ai_hub_section_note = _(
        "When a pipeline misbehaves, this is the first place to inspect."
    )
    ai_hub_section_accent = "pipeline"
    ai_hub_section_actions = (
        {"label": _("View pipelines"), "url": lambda self: reverse("admin:ai_hub_pipelinedefinition_changelist"), "default": True},
        {"label": _("Open control center"), "url": lambda self: reverse("admin:ai_hub_control_center")},
    )
    list_display = ("pipeline", "order", "agent", "on_error", "fallback_agent")
    list_filter = ("pipeline", "on_error")
    search_fields = ("pipeline__name", "agent__name")
    autocomplete_fields = ("pipeline", "agent", "fallback_agent")
    ai_hub_field_guidance = {
        "pipeline": {
            "help_text": _("The workflow this step belongs to."),
        },
        "order": {
            "placeholder": "1",
            "help_text": _("Step number. Use 1, 2, 3 with no gaps."),
        },
        "agent": {
            "help_text": _("The specialist agent that runs at this step."),
        },
        "input_mapping": {
            "rows": 8,
            "placeholder": '{\n  "ticket_text": "ticket_text",\n  "knowledge_context": "knowledge_context"\n}',
            "help_text": _("Maps workflow context into the agent input. Left side is agent input key, right side is context path."),
        },
        "output_mapping": {
            "rows": 8,
            "placeholder": '{\n  "triage_result": "agent",\n  "model_text": "llm.content"\n}',
            "help_text": _("Writes agent output back into workflow context. Left side is context key, right side is response path."),
        },
        "on_error": {
            "help_text": _("Stop is safest. Continue is useful for optional enrichment. Fallback agent retries with another agent."),
        },
        "fallback_agent": {
            "help_text": _("Required only when on_error is Fallback Agent."),
        },
    }
    fieldsets = (
        ("4.1 Step connection", {
            "fields": ("pipeline", "order", "agent"),
            "description": _(
                "Each step passes the current context into one agent. Order must be continuous: 1, 2, 3..."
            ),
        }),
        ("Mappings", {
            "fields": ("input_mapping", "output_mapping"),
            "description": _(
                "input_mapping selects context keys for the agent. output_mapping writes agent output back into "
                "context for later steps. Dot paths such as llm.content are supported."
            ),
        }),
        ("Failure behavior", {
            "fields": ("on_error", "fallback_agent"),
            "description": _(
                "Choose whether a failed step stops the run, records the error and continues, or retries with a "
                "fallback agent."
            ),
        }),
    )


class ExecutionStepRunInline(admin.TabularInline):
    model = ExecutionStepRun
    extra = 0
    fields = ("order", "pipeline_step", "agent", "action_name", "status", "latency_ms", "created_at")
    readonly_fields = ("order", "pipeline_step", "agent", "action_name", "status", "latency_ms", "created_at")
    show_change_link = True
    verbose_name = "Execution step run"
    verbose_name_plural = "5.1 Execution step runs - session timeline"


    def has_add_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(GameWorkspace)
class GameWorkspaceAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME workspaces")
    ai_hub_section_description = _("Define the environments that own GAME goals, defaults and policy boundaries.")
    list_display = ("name", "is_active", "goal_count", "dashboard_link", "updated_at")
    list_filter = ("is_active",)
    search_fields = ("name", "description")
    readonly_fields = ("created_at", "updated_at")

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(_goal_count=Count("goals"))

    @admin.display(description=_("Goals"), ordering="_goal_count")
    def goal_count(self, obj):
        return obj._goal_count

    @admin.display(description=_("Dashboard"))
    def dashboard_link(self, obj):
        url = reverse("admin:ai_hub_gameworkspace_dashboard", args=[obj.pk])
        return format_html('<a href="{}">View dashboard</a>', url)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "<int:workspace_id>/dashboard/",
                self.admin_site.admin_view(self.workspace_dashboard_view),
                name="ai_hub_gameworkspace_dashboard",
            ),
        ]
        return custom_urls + urls

    def workspace_dashboard_view(self, request, workspace_id):
        from .services.game_operational_ux import build_workspace_dashboard_context
        if not self.has_view_or_change_permission(request):
            raise PermissionDenied
        if not request.user.has_perm("ai_hub.view_executionsession"):
            raise PermissionDenied
        workspace = self.get_object(request, workspace_id)
        if workspace is None:
            from django.http import Http404
            raise Http404
        context = {
            **self.admin_site.each_context(request),
            "title": f"Dashboard — {workspace.name}",
            "opts": self.model._meta,
            **build_workspace_dashboard_context(workspace, user=request.user),
        }
        return TemplateResponse(request, "admin/ai_hub/gameworkspace/dashboard.html", context)


class GameGoalDependencyInline(admin.TabularInline):
    model = GameGoalDependency
    fk_name = "goal"
    extra = 0
    autocomplete_fields = ("depends_on",)
    fields = ("depends_on", "is_required", "note")
    verbose_name = _("dependency")
    verbose_name_plural = _("Dependencies — this goal depends on…")


@admin.register(GameGoal)
class GameGoalAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME goals")
    ai_hub_section_description = _("Manage durable work items independently from their execution sessions.")
    change_form_template = "admin/ai_hub/gamegoal/change_form.html"
    inlines = (GameGoalDependencyInline,)
    list_display = (
        "title",
        "workspace",
        "status",
        "base_priority",
        "calculated_priority",
        "due_at",
        "updated_at",
    )
    list_filter = ("workspace", "status", "due_at", GoalPriorityRangeFilter)
    search_fields = ("title", "description", "workspace__name")
    autocomplete_fields = ("workspace",)
    readonly_fields = (
        "status",
        "calculated_priority",
        "queued_at",
        "result",
        "transition_metadata",
        "created_at",
        "updated_at",
    )
    actions = (
        "queue_selected_goals",
        "cancel_selected_goals",
        "reopen_selected_goals",
        "resume_selected_goals",
    )

    def change_view(self, request, object_id, form_url="", extra_context=None):
        from .services.game_operational_ux import build_goal_detail_context
        extra_context = extra_context or {}
        goal = self.get_object(request, object_id)
        if goal:
            extra_context.update(build_goal_detail_context(goal, user=request.user))
        return super().change_view(request, object_id, form_url, extra_context=extra_context)

    @admin.action(description=_("Queue selected goals"))
    def queue_selected_goals(self, request, queryset):
        from .services.game_goals import transition_goal_status
        success = 0
        for goal in queryset:
            try:
                transition_goal_status(goal, GameGoal.Status.QUEUED, reason="queued from admin")
                success += 1
            except Exception as exc:
                self.message_user(request, f"Goal #{goal.pk}: {exc}", level=messages.WARNING)
        if success:
            self.message_user(request, f"{success} goal(s) queued.", level=messages.SUCCESS)

    @admin.action(description=_("Cancel selected goals"))
    def cancel_selected_goals(self, request, queryset):
        from .services.game_goals import transition_goal_status
        success = 0
        for goal in queryset:
            try:
                transition_goal_status(goal, GameGoal.Status.CANCELLED, reason="cancelled from admin")
                success += 1
            except Exception as exc:
                self.message_user(request, f"Goal #{goal.pk}: {exc}", level=messages.WARNING)
        if success:
            self.message_user(request, f"{success} goal(s) cancelled.", level=messages.SUCCESS)

    @admin.action(description=_("Reopen selected goals (completed/cancelled → queued)"))
    def reopen_selected_goals(self, request, queryset):
        from .services.game_goals import reopen_goal, transition_goal_status
        success = 0
        for goal in queryset:
            try:
                if goal.status in {GameGoal.Status.COMPLETED, GameGoal.Status.CANCELLED}:
                    reopen_goal(goal, reason="reopened from admin")
                else:
                    transition_goal_status(goal, GameGoal.Status.QUEUED, reason="requeued from admin")
                success += 1
            except Exception as exc:
                self.message_user(request, f"Goal #{goal.pk}: {exc}", level=messages.WARNING)
        if success:
            self.message_user(request, f"{success} goal(s) reopened.", level=messages.SUCCESS)

    @admin.action(description=_("Resume selected goal sessions"))
    def resume_selected_goals(self, request, queryset):
        from .services.game_resume import resume_goal_execution
        success = 0
        for goal in queryset:
            try:
                waiting_session = (
                    ExecutionSession.objects.filter(
                        goal=goal, status=ExecutionSession.Status.WAITING_ASYNC
                    )
                    .order_by("-created_at")
                    .first()
                )
                if not waiting_session:
                    self.message_user(
                        request,
                        f"Goal #{goal.pk}: no waiting session to resume.",
                        level=messages.WARNING,
                    )
                    continue
                resume_goal_execution(session_id=waiting_session.pk, resolved_by=request.user)
                success += 1
            except Exception as exc:
                self.message_user(request, f"Goal #{goal.pk}: {exc}", level=messages.WARNING)
        if success:
            self.message_user(request, f"{success} goal session(s) resumed.", level=messages.SUCCESS)


@admin.register(GameGoalDependency)
class GameGoalDependencyAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME goal dependencies")
    ai_hub_section_description = _("Describe required and optional ordering relationships inside one workspace.")
    list_display = ("goal", "depends_on", "is_required", "note", "created_at")
    list_filter = ("goal__workspace", "is_required")
    search_fields = ("goal__title", "depends_on__title", "note")
    autocomplete_fields = ("goal", "depends_on")
    readonly_fields = ("created_at",)


@admin.register(ExecutionSession)
class ExecutionSessionAdmin(AIHubFormHelpMixin, admin.ModelAdmin):
    change_list_template = "admin/ai_hub/executionsession/change_list.html"
    change_form_template = "admin/ai_hub/executionsession/change_form.html"
    list_display = (
        "id",
        "runtime_kind",
        "source_label",
        "pipeline",
        "entry_agent",
        "goal",
        "status",
        "runtime_mode",
        "triggered_by",
        "created_at",
    )
    list_filter = ("runtime_kind", "status", "runtime_mode", "pipeline", "goal__workspace")
    search_fields = (
        "source_label",
        "goal_text",
        "goal__title",
        "pipeline__name",
        "entry_agent__name",
        "error_detail",
    )
    autocomplete_fields = ("pipeline", "entry_agent", "goal", "triggered_by")
    readonly_fields = (
        "status",
        "runtime_config_redacted",
        "initial_context_redacted",
        "final_context_redacted",
        "goal_outcome_fingerprint",
        "error_detail_redacted",
        "started_at",
        "finished_at",
        "created_at",
        "updated_at",
    )
    inlines = [ExecutionStepRunInline]
    actions = ("run_selected_sessions",)
    ai_hub_field_guidance = {
        "runtime_kind": {
            "help_text": _("Orchestrator uses a fixed pipeline. GAME uses one entry agent with a goal loop."),
        },
        "runtime_mode": {
            "help_text": _("Async is recommended for long AI runs. Sync is mainly for quick local testing."),
        },
        "status": {
            "help_text": _("Current lifecycle state. Pending sessions can be run from the list action."),
        },
        "pipeline": {
            "help_text": _("Required for Orchestrator sessions. Leave empty for GAME sessions."),
        },
        "entry_agent": {
            "help_text": _("Required for GAME sessions. Optional/descriptive for Orchestrator sessions."),
        },
        "goal": {
            "help_text": _("Optional durable GAME goal. Legacy GAME sessions may continue to use goal_text only."),
        },
        "source_label": {
            "placeholder": "Example: Support ticket #4832",
            "help_text": _("Short label shown in admin lists and timelines."),
        },
        "source_object_id": {
            "placeholder": "Example: 4832",
            "help_text": _("Optional ID of the host-project object that triggered this AI run."),
        },
        "goal_text": {
            "rows": 6,
            "placeholder": "Example: Review this ticket and finish with a concise recommended support response.",
            "help_text": _("For GAME, this is the goal. For Orchestrator, use it as a human-readable purpose."),
        },
        "runtime_config": {
            "rows": 8,
            "placeholder": '{\n  "max_iterations": 3,\n  "strict_response_contract": true\n}',
            "help_text": _("Portable JSON settings for the runtime. Keep this small unless the runner explicitly supports more keys."),
        },
        "initial_context": {
            "rows": 10,
            "placeholder": '{\n  "ticket_text": "The user cannot upload a file.",\n  "user_profile": {"tier": "pro"}\n}',
            "help_text": _("JSON object passed into the first step or GAME iteration."),
        },
        "final_context": {
            "rows": 10,
            "placeholder": _("Filled by the runtime after the session finishes."),
            "help_text": _("Final portable JSON state after execution. Usually read-only from an operator perspective."),
        },
        "error_detail": {
            "rows": 5,
            "placeholder": _("Filled when the session fails."),
            "help_text": _("Failure detail captured by the runner."),
        },
    }
    fieldsets = (
        ("5.0 Generic execution", {
            "fields": (
                "runtime_kind",
                "runtime_mode",
                "status",
                "pipeline",
                "entry_agent",
                "goal",
                "triggered_by",
            ),
            "description": _(
                "A reusable execution record. Orchestrator sessions use pipelines; GAME sessions use goals, "
                "actions, memory and an environment policy."
            ),
        }),
        ("Source adapter", {
            "fields": ("source_content_type", "source_object_id", "source_label"),
            "description": _(
                "Optional host-project object. Keep domain data outside AI Hub and point to it through this adapter."
            ),
        }),
        ("Goal and context", {
            "fields": ("goal_text", "runtime_config", "initial_context", "final_context"),
            "description": _(
                "goal_text is the explicit direction for GAME-style runs. Context fields store portable JSON state."
            ),
        }),
        ("Timing and audit", {
            "fields": ("started_at", "finished_at", "created_at", "updated_at", "error_detail"),
            "description": _("Use this area to inspect lifecycle, failures and rollback-safe execution state."),
        }),
    )

    def has_add_permission(self, request):
        from .services.game_feature_flags import is_game_feature_enabled

        return is_game_feature_enabled("AI_HUB_GAME_GOALS_ENABLED") and super().has_add_permission(request)

    def get_actions(self, request):
        from .services.game_feature_flags import is_game_feature_enabled

        actions = super().get_actions(request)
        if not is_game_feature_enabled("AI_HUB_GAME_GOALS_ENABLED"):
            for name in self.actions:
                actions.pop(name, None)
        return actions

    audit_fieldsets = (
        ("5.0 Generic execution", {
            "fields": (
                "runtime_kind", "runtime_mode", "status", "pipeline", "entry_agent",
                "goal", "triggered_by",
            ),
        }),
        ("Source adapter", {
            "fields": ("source_content_type", "source_object_id", "source_label"),
        }),
        ("Goal and context", {
            "fields": (
                "goal_text", "runtime_config_redacted", "initial_context_redacted",
                "final_context_redacted",
            ),
        }),
        ("Timing and audit", {
            "fields": (
                "started_at", "finished_at", "created_at", "updated_at",
                "goal_outcome_fingerprint", "error_detail_redacted",
            ),
        }),
    )

    def get_fieldsets(self, request, obj=None):
        if obj is not None:
            return self.audit_fieldsets
        return super().get_fieldsets(request, obj)

    @admin.display(description=_("Runtime config (redacted)"))
    def runtime_config_redacted(self, obj):
        return _render_redacted_json(obj.runtime_config)

    @admin.display(description=_("Initial context (redacted)"))
    def initial_context_redacted(self, obj):
        return _render_redacted_json(obj.initial_context)

    @admin.display(description=_("Final context (redacted)"))
    def final_context_redacted(self, obj):
        return _render_redacted_json(obj.final_context)

    @admin.display(description=_("Error detail (redacted)"))
    def error_detail_redacted(self, obj):
        return _render_redacted_text(obj.error_detail)

    def get_readonly_fields(self, request, obj=None):
        readonly = list(super().get_readonly_fields(request, obj))
        if obj and obj.status != ExecutionSession.Status.PENDING:
            readonly.extend(
                [
                    "runtime_kind",
                    "runtime_mode",
                    "pipeline",
                    "entry_agent",
                    "goal",
                    "triggered_by",
                    "source_content_type",
                    "source_object_id",
                    "source_label",
                    "goal_text",
                    "runtime_config",
                    "initial_context",
                ]
            )
        return tuple(dict.fromkeys(readonly))

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "game/new/",
                self.admin_site.admin_view(self.create_game_session_view),
                name="ai_hub_executionsession_game_new",
            ),
        ]
        return custom_urls + urls

    def change_view(self, request, object_id, form_url="", extra_context=None):
        from .services.game_operational_ux import build_session_timeline

        extra_context = extra_context or {}
        session = self.get_object(request, object_id)
        if session:
            extra_context["timeline_steps"] = [
                self._timeline_step_context(step_run)
                for step_run in session.step_runs.select_related("agent", "pipeline_step").order_by("order")
            ]
            timeline_events = build_session_timeline(session, user=request.user)
            url_names = {
                "step": "admin:ai_hub_executionsteprun_change",
                "action": "admin:ai_hub_gameactionrun_change",
                "continuation": "admin:ai_hub_gamecontinuationrequest_change",
                "approval": "admin:ai_hub_gameactionapprovalrequest_change",
            }
            for event in timeline_events:
                event["change_url"] = reverse(url_names[event["kind"]], args=[event["pk"]])
            extra_context["timeline_events"] = timeline_events
            extra_context["timeline_summary"] = {
                "total": len(timeline_events),
                "runtime_kind": session.runtime_kind,
                "status": session.status,
                "finish_reason": (session.final_context or {}).get("finish_reason", ""),
                "final_answer": redact_text(
                    (session.final_context or {}).get("final_answer", "")
                ),
            }
        return super().change_view(request, object_id, form_url, extra_context=extra_context)

    def _timeline_step_context(self, step_run):
        observation = step_run.observation_payload or {}
        decision = observation.get("decision") if isinstance(observation, dict) else {}
        if not isinstance(decision, dict):
            decision = {}
        response_payload = step_run.response_payload or {}
        llm_payload = response_payload.get("llm") if isinstance(response_payload, dict) else {}
        if not isinstance(llm_payload, dict):
            llm_payload = {}
        message = (
            decision.get("message")
            or decision.get("final_answer")
            or decision.get("answer")
            or llm_payload.get("content")
            or step_run.error_detail
            or ""
        )
        return {
            "order": step_run.order,
            "agent": step_run.agent,
            "pipeline_step": step_run.pipeline_step,
            "action_name": step_run.action_name,
            "status": step_run.status,
            "latency_ms": step_run.latency_ms,
            "created_at": step_run.created_at,
            "action": observation.get("action", "") if isinstance(observation, dict) else "",
            "complete": observation.get("complete", False) if isinstance(observation, dict) else False,
            "message": message,
            "error_detail": step_run.error_detail,
            "change_url": reverse("admin:ai_hub_executionsteprun_change", args=[step_run.id]),
        }

    def create_game_session_view(self, request):
        if not self.has_add_permission(request):
            raise PermissionDenied

        if request.method == "POST":
            form = GameSessionCreateForm(request.POST)
            if form.is_valid():
                session = form.save(user=request.user)
                self.message_user(
                    request,
                    f"GAME session #{session.id} created. Review it, then run it from the session list.",
                    level=messages.SUCCESS,
                )
                return HttpResponseRedirect(
                    reverse("admin:ai_hub_executionsession_change", args=[session.id])
                )
        else:
            form = GameSessionCreateForm()

        context = {
            **self.admin_site.each_context(request),
            "opts": self.model._meta,
            "title": _("Create GAME session"),
            "form": form,
            "media": self.media + form.media,
            "game_goal_examples": [
                "Review this support ticket and decide the next best response.",
                "Read the uploaded research notes and produce a short action plan.",
                "Explore the context, identify missing information, and finish with a clear recommendation.",
            ],
            "initial_context_example": '{\n  "user_profile": {},\n  "documents": [],\n  "constraints": []\n}',
        }
        return TemplateResponse(request, "admin/ai_hub/executionsession/game_form.html", context)

    @admin.action(description="Run selected execution sessions")
    def run_selected_sessions(self, request, queryset):
        completed = 0
        failed = 0
        blocked = 0
        for session in queryset.select_related("pipeline"):
            try:
                run_execution_session(session.id)
                session.refresh_from_db(fields=["status", "error_detail"])
                if session.status == ExecutionSession.Status.FAILED:
                    failed += 1
                    self.message_user(
                        request,
                        f"Session #{session.id} finished failed: {session.error_detail}",
                        level=messages.WARNING,
                    )
                else:
                    completed += 1
            except Exception as exc:
                blocked += 1
                self.message_user(
                    request,
                    f"Session #{session.id} could not be started: {exc}",
                    level=messages.WARNING,
                )

        if completed:
            self.message_user(request, f"Completed {completed} execution session(s).", level=messages.SUCCESS)
        if failed and not completed:
            self.message_user(request, f"{failed} execution session(s) finished failed.", level=messages.ERROR)
        if blocked and not completed:
            self.message_user(request, f"{blocked} execution session(s) could not be started.", level=messages.ERROR)


@admin.register(ExecutionStepRun)
class ExecutionStepRunAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Execution step runs")
    ai_hub_section_description = _(
        "This is the session timeline at the most detailed level. Use it to inspect a failed or slow step."
    )
    ai_hub_section_note = _(
        "Each record shows the exact request, response and observation payload for one step."
    )
    ai_hub_section_accent = "session"
    ai_hub_section_actions = (
        {"label": _("View sessions"), "url": lambda self: reverse("admin:ai_hub_executionsession_changelist"), "default": True},
        {"label": _("Open control center"), "url": lambda self: reverse("admin:ai_hub_control_center")},
    )
    list_display = ("session", "order", "pipeline_step", "agent", "action_name", "status", "latency_ms", "created_at")
    list_filter = ("status", "session__runtime_kind", "pipeline_step__pipeline", "agent")
    search_fields = ("session__source_label", "session__goal_text", "agent__name", "action_name", "error_detail")
    autocomplete_fields = ("session", "pipeline_step", "agent")
    readonly_fields = (
        "session",
        "order",
        "pipeline_step",
        "agent",
        "action_name",
        "status",
        "latency_ms",
        "created_at",
        "request_payload_redacted",
        "response_payload_redacted",
        "observation_payload_redacted",
        "error_detail_redacted",
    )

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    @admin.display(description=_("Request payload (redacted)"))
    def request_payload_redacted(self, obj):
        return _render_redacted_json(obj.request_payload)

    @admin.display(description=_("Response payload (redacted)"))
    def response_payload_redacted(self, obj):
        return _render_redacted_json(obj.response_payload)

    @admin.display(description=_("Observation payload (redacted)"))
    def observation_payload_redacted(self, obj):
        return _render_redacted_json(obj.observation_payload)

    @admin.display(description=_("Error detail (redacted)"))
    def error_detail_redacted(self, obj):
        return _render_redacted_text(obj.error_detail)
    ai_hub_field_guidance = {
        "session": {
            "help_text": _("Execution session this step belongs to."),
        },
        "order": {
            "placeholder": "1",
            "help_text": _("Position inside the session timeline."),
        },
        "pipeline_step": {
            "help_text": _("The configured pipeline step, when this came from Orchestrator."),
        },
        "agent": {
            "help_text": _("Agent that produced this step response."),
        },
        "action_name": {
            "placeholder": "Example: call_model, decide_next_action, final_answer",
            "help_text": _("Short runtime action label. Useful for reading GAME timelines."),
        },
        "latency_ms": {
            "placeholder": "1250",
            "help_text": _("Duration in milliseconds, captured by the runtime."),
        },
        "request_payload": {
            "rows": 10,
            "placeholder": _("Captured request JSON sent into the agent or tool."),
            "help_text": _("Debug payload. It shows what the runtime actually sent."),
        },
        "response_payload": {
            "rows": 10,
            "placeholder": _("Captured model/tool response JSON."),
            "help_text": _("Debug payload. It shows what came back from the provider or tool."),
        },
        "observation_payload": {
            "rows": 10,
            "placeholder": _("Captured observation JSON for GAME/action loops."),
            "help_text": _("Debug payload. It stores action observations and decision-loop state."),
        },
        "error_detail": {
            "rows": 5,
            "placeholder": _("Filled when this step fails."),
            "help_text": _("Failure detail for this single step."),
        },
    }
    fieldsets = (
        ("5.1 Generic step", {
            "fields": ("session", "order", "pipeline_step", "agent", "action_name", "status", "latency_ms", "created_at"),
            "description": _("One observable step inside a reusable AI Hub execution session."),
        }),
        ("Payloads", {
            "fields": (
                "request_payload_redacted",
                "response_payload_redacted",
                "observation_payload_redacted",
            ),
            "description": _("Request, model/tool response and environment observation stored as portable JSON."),
        }),
        ("Error", {
            "fields": ("error_detail_redacted",),
            "description": _("Failure details for this step, if any."),
        }),
    )


@admin.register(ToolExecutionRun)
class ToolExecutionRunAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("Tool execution runs")
    ai_hub_section_description = _("Generic audit records for individual reusable tool calls.")
    ai_hub_section_note = _("Future runtimes write these records; inspect them here rather than editing them.")
    ai_hub_section_accent = "tool"
    list_display = ("tool", "agent", "session", "status", "risk_level", "approval_state", "latency_ms", "created_at")
    list_filter = ("status", "risk_level", "approval_state", "tool")
    search_fields = ("tool__name", "tool__label", "agent__name", "session__source_label", "error_detail")
    autocomplete_fields = ("session", "step_run", "agent", "tool")
    readonly_fields = (
        "session",
        "step_run",
        "agent",
        "tool",
        "status",
        "input_payload_redacted",
        "output_payload_redacted",
        "error_detail_redacted",
        "latency_ms",
        "risk_level",
        "approval_state",
        "idempotency_key",
        "started_at",
        "finished_at",
        "created_at",
    )

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    @admin.display(description=_("Input payload (redacted)"))
    def input_payload_redacted(self, obj):
        return _render_redacted_json(obj.input_payload)

    @admin.display(description=_("Output payload (redacted)"))
    def output_payload_redacted(self, obj):
        return _render_redacted_json(obj.output_payload)

    @admin.display(description=_("Error detail (redacted)"))
    def error_detail_redacted(self, obj):
        return _render_redacted_text(obj.error_detail)

    fieldsets = (
        ("5.2 Tool call", {
            "fields": (
                "session",
                "step_run",
                "agent",
                "tool",
                "status",
                "risk_level",
                "approval_state",
                "latency_ms",
                "idempotency_key",
                "started_at",
                "finished_at",
                "created_at",
            ),
            "description": _("One reusable tool call, independent from GAME action history."),
        }),
        ("Payloads", {
            "fields": ("input_payload_redacted", "output_payload_redacted"),
            "description": _("Input and output are redacted for safer inspection."),
        }),
        ("Error", {
            "fields": ("error_detail_redacted",),
            "description": _("Failure details for this tool execution, if any."),
        }),
    )


@admin.register(GameActionDefinition)
class GameActionDefinitionAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME action definitions")
    ai_hub_section_description = _(
        "Register the actions the GAME dispatcher can execute. "
        "Only active definitions are available during a goal session."
    )
    ai_hub_section_note = _(
        "Built-in internal actions: finish_goal, update_goal_status, record_memory. "
        "Built-in context tools: search_knowledge, read_document."
    )
    ai_hub_section_accent = "game"
    ai_hub_section_actions = (
        {
            "label": _("Add action"),
            "url": lambda self: reverse("admin:ai_hub_gameactiondefinition_add"),
            "default": True,
        },
    )
    list_display = ("name", "label", "action_type", "risk_level", "is_active", "updated_at")
    list_filter = ("action_type", "risk_level", "requires_approval", "is_active")
    search_fields = ("name", "label", "description")
    autocomplete_fields = ("tool",)
    readonly_fields = ("created_at", "updated_at")
    ai_hub_field_guidance = {
        "name": {
            "placeholder": "Example: finish_goal",
            "help_text": _(
                "Slug name the LLM must write in its action field. "
                "Must exactly match the dispatcher's expected name."
            ),
        },
        "label": {
            "placeholder": "Example: Finish goal",
            "help_text": _("Human-readable label shown in admin lists."),
        },
        "action_type": {
            "help_text": _(
                "Execution mechanism: internal runs built-in Python, context_tool reads knowledge, "
                "tool/http/sub_agent are reserved for future phases."
            ),
        },
        "tool": {
            "help_text": _(
                "Optional reusable ToolDefinition for unified runtime rollout. Leave empty for built-in GAME control actions."
            ),
        },
        "input_contract": {
            "rows": 8,
            "placeholder": (
                '{\n  "required": ["final_answer"],\n'
                '  "properties": {\n'
                '    "final_answer": {"type": "string"},\n'
                '    "message": {"type": "string"}\n'
                "  }\n}"
            ),
            "help_text": _(
                "JSON schema-style contract for the input the LLM must provide. "
                "Leave empty to accept any JSON object."
            ),
        },
        "output_contract": {
            "rows": 8,
            "placeholder": '{\n  "required": ["result"],\n  "properties": {\n    "result": {"type": "string"}\n  }\n}',
            "help_text": _("Optional output contract validated after the handler returns."),
        },
        "risk_level": {
            "placeholder": "low",
            "help_text": _("Risk classification: low, medium, or high. Used by future approval policies."),
        },
        "requires_approval": {
            "help_text": _("Flag for future human-in-the-loop workflows."),
        },
        "config": {
            "rows": 5,
            "placeholder": "{}",
            "help_text": _("Optional handler-specific configuration. Reserved for future use."),
        },
    }
    fieldsets = (
        ("4.8 Action identity", {
            "fields": ("name", "label", "description", "action_type", "tool", "is_active"),
            "description": _(
                "Define one dispatchable action. The name is what the LLM writes; "
                "action_type routes to the correct execution adapter."
            ),
        }),
        ("Contracts", {
            "fields": ("input_contract", "output_contract"),
            "description": _(
                "input_contract is validated before the handler runs. "
                "output_contract is validated after. Leave empty to skip validation."
            ),
        }),
        ("Policy", {
            "fields": ("risk_level", "requires_approval"),
            "description": _("Controls future approval and risk gate flows."),
        }),
        ("Advanced", {
            "fields": ("config", "created_at", "updated_at"),
            "classes": ("collapse",),
            "description": _("Reserved for handler-specific runtime configuration."),
        }),
    )


class GameActionRunInline(admin.TabularInline):
    model = GameActionRun
    extra = 0
    fields = ("iteration", "action_name", "status", "latency_ms", "started_at", "finished_at")
    readonly_fields = ("iteration", "action_name", "status", "latency_ms", "started_at", "finished_at")
    show_change_link = True
    verbose_name = "Action run"
    verbose_name_plural = "4.9 Action runs - dispatcher history for this session"

    def has_add_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(GameActionRun)
class GameActionRunAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME action runs")
    ai_hub_section_description = _(
        "Immutable audit records for every action dispatched during a GAME goal session."
    )
    ai_hub_section_note = _(
        "Records are written by the dispatcher and are read-only. "
        "Use them to inspect what the LLM selected, what input it provided, and what the handler returned."
    )
    ai_hub_section_accent = "game"
    ai_hub_section_actions = (
        {
            "label": _("View sessions"),
            "url": lambda self: reverse("admin:ai_hub_executionsession_changelist"),
            "default": True,
        },
    )
    list_display = (
        "id",
        "session",
        "iteration",
        "action_name",
        "status",
        "latency_ms",
        "started_at",
    )
    list_filter = ("status", "action__action_type", "action")
    search_fields = ("session__source_label", "action_name", "error_detail")
    autocomplete_fields = ("session",)
    readonly_fields = (
        "session",
        "step_run",
        "action",
        "idempotency_key",
        "action_name",
        "iteration",
        "status",
        "input_payload_redacted",
        "output_payload_redacted",
        "observation_payload_redacted",
        "error_detail_redacted",
        "started_at",
        "finished_at",
        "latency_ms",
    )

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def _render_redacted(self, payload):
        return _render_redacted_json(payload)

    @admin.display(description=_("Input payload (redacted)"))
    def input_payload_redacted(self, obj):
        return self._render_redacted(obj.input_payload)

    @admin.display(description=_("Output payload (redacted)"))
    def output_payload_redacted(self, obj):
        return self._render_redacted(obj.output_payload)

    @admin.display(description=_("Observation payload (redacted)"))
    def observation_payload_redacted(self, obj):
        return self._render_redacted(obj.observation_payload)

    @admin.display(description=_("Error detail (redacted)"))
    def error_detail_redacted(self, obj):
        return _render_redacted_text(obj.error_detail)

    fieldsets = (
        ("4.9 Action run", {
            "fields": (
                "session",
                "step_run",
                "action",
                "action_name",
                "iteration",
                "status",
                "idempotency_key",
                "started_at",
                "finished_at",
                "latency_ms",
            ),
            "description": _("Durable record of one dispatcher invocation."),
        }),
        ("Payloads", {
            "fields": (
                "input_payload_redacted",
                "output_payload_redacted",
                "observation_payload_redacted",
            ),
            "description": _(
                "input_payload is the validated JSON the LLM provided. "
                "output_payload is the handler result. "
                "observation_payload is the normalised observation passed to the next iteration. "
                "Known-sensitive keys (api_key, secret, password, token, ...) are redacted in this view."
            ),
        }),
        ("Error", {
            "fields": ("error_detail_redacted",),
            "description": _("Handler or validation error, if any."),
        }),
    )


@admin.register(GameMemoryEntry)
class GameMemoryEntryAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME memory entries")
    ai_hub_section_description = _(
        "Scoped, bounded knowledge entries persisted across GAME iterations. "
        "Workspace, goal, session, and action-result scopes control visibility."
    )
    ai_hub_section_accent = "game"
    ai_hub_section_actions = (
        {
            "label": _("Add entry"),
            "url": lambda self: reverse("admin:ai_hub_gamememoryentry_add"),
            "default": True,
        },
    )
    list_display = ("id", "workspace", "scope_type", "goal", "importance_score", "expires_at", "created_at")
    list_filter = ("scope_type", "workspace", "goal__workspace")
    search_fields = ("content", "workspace__name", "goal__title")
    autocomplete_fields = ("workspace", "goal", "session")
    readonly_fields = ("created_at",)
    fieldsets = (
        ("4.10 Memory entry", {
            "fields": ("workspace", "goal", "session", "scope_type", "is_active_display"),
            "description": _(
                "Scope type determines visibility: workspace memory is shared across goals; "
                "goal memory is visible only to that goal; session memory is run-specific."
            ),
        }),
        ("Content", {
            "fields": ("content", "importance_score", "expires_at"),
            "description": _("Content is the raw text injected into future iterations."),
        }),
        ("Metadata", {
            "fields": ("metadata", "created_at"),
            "classes": ("collapse",),
            "description": _("Source tracking and category labels for audit and compaction."),
        }),
    )

    @admin.display(description="Active")
    def is_active_display(self, obj):
        from django.utils import timezone
        if obj.expires_at and obj.expires_at <= timezone.now():
            return format_html('<span style="color:red">Expired</span>')
        return format_html('<span style="color:green">Active</span>')


@admin.register(GameContinuationRequest)
class GameContinuationRequestAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME continuation requests")
    ai_hub_section_description = _(
        "Durable records for every pause in a GAME session. "
        "Each entry captures why the session stopped and what input is needed to resume."
    )
    ai_hub_section_accent = "game"
    ai_hub_section_actions = (
        {
            "label": _("View sessions"),
            "url": lambda self: reverse("admin:ai_hub_executionsession_changelist"),
            "default": True,
        },
    )
    list_display = ("id", "session", "goal", "reason_code", "status", "created_at", "resolved_at")
    list_filter = ("status", "reason_code", "goal__workspace")
    search_fields = ("detail", "session__source_label", "goal__title")
    autocomplete_fields = ("session", "goal")
    readonly_fields = (
        "session", "goal", "reason_code", "detail_redacted", "payload_redacted",
        "status", "created_at", "resolved_at", "resolved_by",
    )
    fields = readonly_fields

    @admin.display(description=_("Detail (redacted)"))
    def detail_redacted(self, obj):
        return _render_redacted_text(obj.detail)

    @admin.display(description=_("Payload (redacted)"))
    def payload_redacted(self, obj):
        return _render_redacted_json(obj.payload)

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(GameActionApprovalRequest)
class GameActionApprovalRequestAdmin(AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME action approval requests")
    ai_hub_section_description = _(
        "Pending and resolved approval gates for actions that require human sign-off. "
        "Approve or reject from this list to unblock a waiting GAME session."
    )
    ai_hub_section_accent = "game"
    ai_hub_section_actions = (
        {
            "label": _("View sessions"),
            "url": lambda self: reverse("admin:ai_hub_executionsession_changelist"),
            "default": True,
        },
    )
    list_display = (
        "id", "action_run", "goal", "status", "reviewed_by", "created_at", "reviewed_at", "expires_at"
    )
    list_filter = ("status", "goal__workspace")
    search_fields = ("action_run__action_name", "goal__title", "review_note")
    autocomplete_fields = ("goal",)
    change_form_template = "admin/ai_hub/gameactionapprovalrequest/change_form.html"
    readonly_fields = (
        "action_run", "goal", "status", "requested_payload_redacted",
        "reviewed_by", "review_note_redacted", "created_at", "reviewed_at", "expires_at",
    )
    fields = readonly_fields
    actions = ("approve_selected_actions", "reject_selected_actions")

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "<int:object_id>/approve/",
                self.admin_site.admin_view(self.review_approve_view),
                name="ai_hub_gameactionapprovalrequest_approve",
            ),
            path(
                "<int:object_id>/reject/",
                self.admin_site.admin_view(self.review_reject_view),
                name="ai_hub_gameactionapprovalrequest_reject",
            ),
        ]
        return custom_urls + urls

    def change_view(self, request, object_id, form_url="", extra_context=None):
        obj = self.get_object(request, object_id)
        extra_context = extra_context or {}
        extra_context.update(
            {
                "can_review_action": request.user.has_perm("ai_hub.approve_game_action"),
                "approval_is_pending": bool(
                    obj and obj.status == GameActionApprovalRequest.Status.PENDING
                ),
            }
        )
        return super().change_view(
            request, object_id, form_url=form_url, extra_context=extra_context
        )

    def _review_view(self, request, object_id, *, approve):
        if request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        if not request.user.has_perm("ai_hub.approve_game_action"):
            raise PermissionDenied
        approval = self.get_object(request, object_id)
        if approval is None:
            from django.http import Http404
            raise Http404
        review_note = str(request.POST.get("review_note") or "").strip()[:4000]
        if approve:
            approve_action_run(
                action_run_id=approval.action_run_id,
                reviewed_by=request.user,
                review_note=review_note,
            )
            self.message_user(request, _("Action approved and executed."), level=messages.SUCCESS)
        else:
            reject_action_run(
                action_run_id=approval.action_run_id,
                reviewed_by=request.user,
                review_note=review_note,
            )
            self.message_user(request, _("Action rejected."), level=messages.SUCCESS)
        return HttpResponseRedirect(
            reverse("admin:ai_hub_gameactionapprovalrequest_change", args=[approval.pk])
        )

    def review_approve_view(self, request, object_id):
        return self._review_view(request, object_id, approve=True)

    def review_reject_view(self, request, object_id):
        return self._review_view(request, object_id, approve=False)

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    @admin.display(description=_("Requested payload (redacted)"))
    def requested_payload_redacted(self, obj):
        return _render_redacted_json(obj.requested_payload)

    @admin.display(description=_("Review note (redacted)"))
    def review_note_redacted(self, obj):
        return _render_redacted_text(obj.review_note)

    def get_actions(self, request):
        actions = super().get_actions(request)
        if not request.user.has_perm("ai_hub.approve_game_action"):
            actions.pop("approve_selected_actions", None)
            actions.pop("reject_selected_actions", None)
        return actions

    @admin.action(description=_("Approve selected action requests"))
    def approve_selected_actions(self, request, queryset):
        if not request.user.has_perm("ai_hub.approve_game_action"):
            self.message_user(
                request,
                _("You do not have permission to approve GAME actions."),
                level=messages.ERROR,
            )
            return
        success = 0
        for approval_req in queryset.select_related("action_run"):
            try:
                approve_action_run(
                    action_run_id=approval_req.action_run_id,
                    reviewed_by=request.user,
                )
                success += 1
            except Exception as exc:
                self.message_user(request, f"Approval #{approval_req.pk}: {exc}", level=messages.WARNING)
        if success:
            self.message_user(request, f"{success} action(s) approved.", level=messages.SUCCESS)

    @admin.action(description=_("Reject selected action requests"))
    def reject_selected_actions(self, request, queryset):
        if not request.user.has_perm("ai_hub.approve_game_action"):
            self.message_user(
                request,
                _("You do not have permission to reject GAME actions."),
                level=messages.ERROR,
            )
            return
        success = 0
        for approval_req in queryset.select_related("action_run"):
            try:
                reject_action_run(
                    action_run_id=approval_req.action_run_id,
                    reviewed_by=request.user,
                )
                success += 1
            except Exception as exc:
                self.message_user(request, f"Approval #{approval_req.pk}: {exc}", level=messages.WARNING)
        if success:
            self.message_user(request, f"{success} action(s) rejected.", level=messages.SUCCESS)


@admin.register(GameWorkspaceAction)
class GameWorkspaceActionAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME workspace actions")
    ai_hub_section_description = _(
        "Allow-list of actions permitted per workspace. "
        "Disable an entry to block a specific action. "
        "Use requires_approval_override to force human review regardless of the action's default."
    )
    ai_hub_section_accent = "game"
    ai_hub_section_actions = (
        {
            "label": _("View workspaces"),
            "url": lambda self: reverse("admin:ai_hub_gameworkspace_changelist"),
            "default": True,
        },
    )
    list_display = ("id", "workspace", "action", "is_enabled", "requires_approval_override", "created_at")
    list_filter = ("is_enabled", "requires_approval_override", "workspace")
    search_fields = ("workspace__name", "action__name")
    autocomplete_fields = ("workspace", "action")
    readonly_fields = ("created_at",)


@admin.register(GameWorkspaceAgent)
class GameWorkspaceAgentAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME workspace agents")
    ai_hub_section_description = _(
        "Allow-list of agents permitted to run as entry agents within a workspace. "
        "Disable an entry to prevent a specific agent from being used in that workspace."
    )
    ai_hub_section_accent = "game"
    ai_hub_section_actions = (
        {
            "label": _("View workspaces"),
            "url": lambda self: reverse("admin:ai_hub_gameworkspace_changelist"),
            "default": True,
        },
    )
    list_display = ("id", "workspace", "agent", "is_enabled", "created_at")
    list_filter = ("is_enabled", "workspace")
    search_fields = ("workspace__name", "agent__name")
    autocomplete_fields = ("workspace", "agent")
    readonly_fields = ("created_at",)


class GameGoalPlanStepInline(admin.TabularInline):
    model = GameGoalPlanStep
    extra = 0
    fields = ("order", "title", "status", "depends_on_step", "created_at")
    readonly_fields = ("created_at",)
    autocomplete_fields = ()
    ordering = ("order",)


@admin.register(GameGoalPlan)
class GameGoalPlanAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME goal plans")
    ai_hub_section_description = _(
        "Structured execution plans attached to goals. "
        "Each plan holds an ordered set of steps that guide the agent's work."
    )
    ai_hub_section_accent = "game"
    ai_hub_section_actions = (
        {
            "label": _("View goals"),
            "url": lambda self: reverse("admin:ai_hub_gamegoal_changelist"),
            "default": True,
        },
    )
    list_display = ("id", "goal", "status", "version", "created_at", "updated_at")
    list_filter = ("status",)
    search_fields = ("goal__title",)
    readonly_fields = ("version", "revision_history", "created_at", "updated_at")
    inlines = [GameGoalPlanStepInline]

    def get_readonly_fields(self, request, obj=None):
        fields = list(super().get_readonly_fields(request, obj))
        if obj is not None:
            fields.extend(["goal", "summary", "status"])
        return tuple(dict.fromkeys(fields))


@admin.register(GameGoalPlanStep)
class GameGoalPlanStepAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME goal plan steps")
    ai_hub_section_description = _("Individual steps within a goal plan.")
    ai_hub_section_accent = "game"
    ai_hub_section_actions = (
        {
            "label": _("View plans"),
            "url": lambda self: reverse("admin:ai_hub_gamegoalplan_changelist"),
            "default": True,
        },
    )
    list_display = ("id", "plan", "order", "title", "status", "depends_on_step", "created_at")
    list_filter = ("status", "plan__goal__workspace")
    search_fields = ("title", "plan__goal__title")
    readonly_fields = ("created_at",)


@admin.register(GameDelegationRun)
class GameDelegationRunAdmin(AIHubHideFromIndexMixin, AIHubListPageMixin, admin.ModelAdmin):
    ai_hub_section_title = _("GAME delegation runs")
    ai_hub_section_description = _(
        "Records of sub-agent delegation requests. Each entry tracks a parent action, "
        "the target agent, the delegated task, and the outcome."
    )
    ai_hub_section_accent = "game"
    ai_hub_section_actions = (
        {
            "label": _("View goals"),
            "url": lambda self: reverse("admin:ai_hub_gamegoal_changelist"),
            "default": True,
        },
    )
    list_display = (
        "id", "parent_goal", "target_agent", "status", "created_at", "finished_at",
    )
    list_filter = ("status", "target_agent")
    search_fields = ("parent_goal__title", "target_agent__name", "task")
    readonly_fields = (
        "parent_action_run", "parent_goal", "delegated_session", "target_agent",
        "status", "task", "expected_result", "result_summary", "created_at", "finished_at",
    )

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


def _install_ai_hub_workspace_admin_urls():
    if getattr(admin.site, "_ai_hub_workspace_urls_installed", False):
        return

    original_get_urls = admin.site.get_urls
    original_app_index = admin.site.app_index

    def get_urls():
        custom_urls = [
            path(
                "ai_hub/workspaces/orchestrator/",
                admin.site.admin_view(ai_hub_orchestrator_workspace_view),
                name="ai_hub_workspace_orchestrator",
            ),
            path(
                "ai_hub/workspaces/game/",
                admin.site.admin_view(ai_hub_game_workspace_view),
                name="ai_hub_workspace_game",
            ),
            path(
                "ai_hub/workspaces/build/",
                admin.site.admin_view(ai_hub_build_wizard_view),
                name="ai_hub_workspace_build",
            ),
        ]
        return custom_urls + original_get_urls()

    def app_index(request, app_label, extra_context=None):
        if app_label == "ai_hub":
            extra_context = {
                **(extra_context or {}),
                **build_ai_hub_home_context(),
            }
        return original_app_index(request, app_label, extra_context=extra_context)

    admin.site.get_urls = get_urls
    admin.site.app_index = app_index
    admin.site._ai_hub_workspace_urls_installed = True


_install_ai_hub_workspace_admin_urls()
# === END REUSABLE AI PIPELINE CORE =========================================
