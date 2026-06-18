import json

from django import forms
from django.contrib import admin
from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.db.models import Count, Q
from django.http import HttpResponseRedirect, JsonResponse
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.html import format_html
from django.utils.translation import gettext_lazy as _

from .models import (
    AgentProfile,
    ExecutionSession,
    ExecutionStepRun,
    KnowledgeCollection,
    KnowledgeDocument,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    ProviderConfig,
    ToolDefinition,
)
from .services.admin_control_center import (
    build_ai_hub_home_context,
    build_control_center_context,
    build_game_graph_context,
)
from .services.execution_runner import run_execution_session


ACTIVE_EXECUTION_STATUSES = (
    ExecutionSession.Status.PENDING,
    ExecutionSession.Status.RUNNING,
    ExecutionSession.Status.WAITING_ASYNC,
)
GAME_INPUT_HINTS = {
    "goal",
    "goal_text",
    "iteration",
    "memory",
    "observations",
    "game_response_contract",
}


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
    if not model_admin.has_view_or_change_permission(request):
        raise PermissionDenied
    return model_admin


def ai_hub_orchestrator_workspace_view(request):
    return _workspace_pipeline_admin(request).orchestrator_workspace_view(request)


def ai_hub_game_workspace_view(request):
    return _workspace_pipeline_admin(request).game_workspace_view(request)


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
        choices=ExecutionSession.RuntimeMode.choices,
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
            "help_text": _("Use the exact model identifier expected by the provider or LiteLLM adapter."),
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
    list_display = ("name", "tool_kind", "is_active", "updated_at")
    list_filter = ("tool_kind", "is_active")
    search_fields = ("name",)
    ai_hub_field_guidance = {
        "name": {
            "placeholder": "Example: fetch_customer_profile",
            "help_text": _("Use a short action-style name. Agents will read this as a capability."),
        },
        "tool_kind": {
            "help_text": _("HTTP for API calls, Python callable for internal functions, Prompt macro for reusable prompt snippets."),
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
            "fields": ("name", "tool_kind", "is_active"),
            "description": _(
                "Optional capabilities that agents may use. Runtime execution is intentionally controlled; "
                "only activate tools you expect an agent to call."
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
            "help_text": _("Optional capabilities this specific agent can use. Leave empty for normal text-only agents."),
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


@admin.register(PipelineStep)
class PipelineStepAdmin(AIHubListPageMixin, admin.ModelAdmin):
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
    readonly_fields = ("created_at",)
    show_change_link = True
    verbose_name = "Execution step run"
    verbose_name_plural = "5.1 Execution step runs - session timeline"


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
        "status",
        "runtime_mode",
        "triggered_by",
        "created_at",
    )
    list_filter = ("runtime_kind", "status", "runtime_mode", "pipeline")
    search_fields = ("source_label", "goal_text", "pipeline__name", "entry_agent__name", "error_detail")
    autocomplete_fields = ("pipeline", "entry_agent", "triggered_by")
    readonly_fields = ("created_at", "updated_at")
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
        extra_context = extra_context or {}
        session = self.get_object(request, object_id)
        if session:
            extra_context["timeline_steps"] = [
                self._timeline_step_context(step_run)
                for step_run in session.step_runs.select_related("agent", "pipeline_step").order_by("order")
            ]
            extra_context["timeline_summary"] = {
                "total": len(extra_context["timeline_steps"]),
                "runtime_kind": session.runtime_kind,
                "status": session.status,
                "finish_reason": (session.final_context or {}).get("finish_reason", ""),
                "final_answer": (session.final_context or {}).get("final_answer", ""),
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
class ExecutionStepRunAdmin(AIHubListPageMixin, admin.ModelAdmin):
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
    readonly_fields = ("created_at", "request_payload", "response_payload", "observation_payload", "error_detail")
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
            "fields": ("request_payload", "response_payload", "observation_payload"),
            "description": _("Request, model/tool response and environment observation stored as portable JSON."),
        }),
        ("Error", {
            "fields": ("error_detail",),
            "description": _("Failure details for this step, if any."),
        }),
    )


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
