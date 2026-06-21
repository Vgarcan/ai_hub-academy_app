from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


# === REUSABLE AI PIPELINE CORE =============================================
# These models are generic orchestration primitives. They can be copied to
# another Django project without depending on Dreamsreader domain models.
class ProviderConfig(models.Model):
    class ProviderType(models.TextChoices):
        OPENAI = "openai", "OpenAI"
        OLLAMA = "ollama", "Ollama"
        DEEPSEEK = "deepseek", "DeepSeek"
        ANTHROPIC = "anthropic", "Anthropic"
        # Deterministic stub provider for demos and tests — no real API key needed
        TRAINING = "training", "Training (stub)"
        OTHER = "other", "Other"

    name = models.CharField(max_length=120, unique=True)
    provider_type = models.CharField(max_length=30, choices=ProviderType.choices)
    base_url = models.URLField(blank=True)
    api_key_env_var = models.CharField(max_length=100, blank=True)
    default_timeout = models.PositiveIntegerField(default=60)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "0.1 Provider config"
        verbose_name_plural = "0.1 Provider configs - AI service accounts"

    def __str__(self):
        return self.name


class ModelConfig(models.Model):
    provider = models.ForeignKey(ProviderConfig, on_delete=models.CASCADE, related_name="models")
    model_name = models.CharField(max_length=140)
    temperature_default = models.DecimalField(max_digits=4, decimal_places=2, default=0.70)
    max_tokens_default = models.PositiveIntegerField(default=1000)
    supports_tools = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("provider", "model_name")]
        ordering = ["provider__name", "model_name"]
        verbose_name = "0.2 Model config"
        verbose_name_plural = "0.2 Model configs - model choices"

    def __str__(self):
        return f"{self.provider.name} / {self.model_name}"

    def clean(self):
        if self.is_active and not self.provider.is_active:
            raise ValidationError("Cannot activate a model with an inactive provider.")


class ToolDefinition(models.Model):
    class ToolKind(models.TextChoices):
        HTTP = "http", "HTTP"
        PYTHON_CALLABLE = "python_callable", "Python Callable"
        PROMPT_MACRO = "prompt_macro", "Prompt Macro"

    name = models.CharField(max_length=120, unique=True)
    tool_kind = models.CharField(max_length=30, choices=ToolKind.choices)
    input_schema = models.JSONField(default=dict, blank=True)
    output_schema = models.JSONField(default=dict, blank=True)
    config = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "2.0 Tool definition"
        verbose_name_plural = "2.0 Tool definitions - optional agent tools"

    def __str__(self):
        return self.name


class KnowledgeCollection(models.Model):
    name = models.CharField(max_length=140, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "1.0 Knowledge collection"
        verbose_name_plural = "1.0 Knowledge collections - knowledge groups"

    def __str__(self):
        return self.name


class KnowledgeDocument(models.Model):
    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        ACTIVE = "active", "Active"
        ARCHIVED = "archived", "Archived"

    collection = models.ForeignKey(KnowledgeCollection, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=180)
    curated_text = models.TextField(blank=True)
    source_file = models.FileField(upload_to="ai_hub/knowledge/", blank=True)
    tags = models.JSONField(default=list, blank=True)
    language = models.CharField(max_length=20, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["collection__name", "title"]
        verbose_name = "1.1 Knowledge document"
        verbose_name_plural = "1.1 Knowledge documents - curated files and text"
        indexes = [
            models.Index(fields=["collection", "status"], name="ai_hub_kdoc_collect_status_idx"),
            models.Index(fields=["language", "status"], name="ai_hub_kdoc_lang_status_idx"),
        ]

    def __str__(self):
        return f"{self.title} ({self.collection.name})"


class AgentProfile(models.Model):
    class ExecutionMode(models.TextChoices):
        SYNC = "sync", "Sync"
        ASYNC = "async", "Async"
        INHERIT = "inherit", "Inherit"

    name = models.CharField(max_length=140, unique=True)
    role = models.CharField(max_length=140)
    system_prompt = models.TextField(blank=True)
    model_config = models.ForeignKey(ModelConfig, on_delete=models.PROTECT, related_name="agents")
    tools = models.ManyToManyField(ToolDefinition, blank=True, related_name="agents")
    knowledge_collections = models.ManyToManyField(KnowledgeCollection, blank=True, related_name="agents")
    knowledge_max_chars = models.PositiveIntegerField(default=6000)
    input_contract = models.JSONField(default=dict, blank=True)
    output_contract = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    execution_mode = models.CharField(max_length=20, choices=ExecutionMode.choices, default=ExecutionMode.INHERIT)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "3.0 Agent profile"
        verbose_name_plural = "3.0 Agent profiles - prompts and roles"

    def __str__(self):
        return self.name

    def clean(self):
        if self.is_active and (not self.model_config.is_active or not self.model_config.provider.is_active):
            raise ValidationError("Cannot activate an agent with inactive model/provider.")


class PipelineDefinition(models.Model):
    name = models.CharField(max_length=140, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=False)
    entry_agent = models.ForeignKey(
        AgentProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="entry_for_pipelines",
    )
    global_input_contract = models.JSONField(default=dict, blank=True)
    global_output_contract = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "4.0 Pipeline definition"
        verbose_name_plural = "4.0 Pipeline definitions - ordered agent flows"

    def __str__(self):
        return self.name

    def clean(self):
        if self.is_active:
            if not self.pk:
                raise ValidationError("Save the pipeline and add steps before activating it.")
            steps = list(self.steps.order_by("order"))
            if not steps:
                raise ValidationError("Cannot activate pipeline without steps.")
            expected = list(range(1, len(steps) + 1))
            got = [s.order for s in steps]
            if got != expected:
                raise ValidationError("Pipeline step order must be continuous starting at 1.")
            for step in steps:
                if not step.agent.input_contract or not step.agent.output_contract:
                    raise ValidationError(
                        f"Agent '{step.agent.name}' must define input/output contracts before activating pipeline."
                    )


class PipelineStep(models.Model):
    class OnError(models.TextChoices):
        STOP = "stop", "Stop"
        CONTINUE = "continue", "Continue"
        FALLBACK_AGENT = "fallback_agent", "Fallback Agent"

    pipeline = models.ForeignKey(PipelineDefinition, on_delete=models.CASCADE, related_name="steps")
    agent = models.ForeignKey(AgentProfile, on_delete=models.PROTECT, related_name="pipeline_steps")
    order = models.PositiveIntegerField()
    input_mapping = models.JSONField(default=dict, blank=True)
    output_mapping = models.JSONField(default=dict, blank=True)
    on_error = models.CharField(max_length=20, choices=OnError.choices, default=OnError.STOP)
    fallback_agent = models.ForeignKey(
        AgentProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="fallback_steps",
    )

    class Meta:
        unique_together = [("pipeline", "order")]
        ordering = ["pipeline__name", "order"]
        verbose_name = "4.1 Pipeline step"
        verbose_name_plural = "4.1 Pipeline steps - agent connections"

    def __str__(self):
        return f"{self.pipeline.name} - step {self.order}"

    def clean(self):
        if self.agent_id and not self.agent.is_active:
            raise ValidationError("Pipeline step agent must be active.")
        if self.on_error == self.OnError.FALLBACK_AGENT and not self.fallback_agent:
            raise ValidationError("Fallback agent is required when on_error is fallback_agent.")
        if self.fallback_agent_id and not self.fallback_agent.is_active:
            raise ValidationError("Fallback agent must be active.")


class GameWorkspace(models.Model):
    name = models.CharField(max_length=160, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    default_policy = models.JSONField(default=dict, blank=True)
    default_runtime_config = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "4.5 GAME workspace"
        verbose_name_plural = "4.5 GAME workspaces - environments"

    def __str__(self):
        return self.name

    def clean(self):
        if not isinstance(self.default_runtime_config, dict):
            raise ValidationError("GAME workspace default_runtime_config must be a JSON object.")
        from ai_hub.services.game_policy import validate_workspace_policy

        validate_workspace_policy(self.default_policy)


class GameGoal(models.Model):
    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        WAITING_INFO = "waiting_info", "Waiting for information"
        WAITING_APPROVAL = "waiting_approval", "Waiting for approval"
        BLOCKED = "blocked", "Blocked"
        COMPLETED = "completed", "Completed"
        PARTIAL = "partial", "Partial"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    workspace = models.ForeignKey(GameWorkspace, on_delete=models.CASCADE, related_name="goals")
    title = models.CharField(max_length=255)
    description = models.TextField()
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.QUEUED)
    base_priority = models.PositiveIntegerField(default=50, validators=[MaxValueValidator(999900)])
    calculated_priority = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0), MaxValueValidator(999999.99)],
    )
    due_at = models.DateTimeField(null=True, blank=True)
    queued_at = models.DateTimeField(default=timezone.now)
    success_criteria = models.JSONField(default=dict, blank=True)
    context = models.JSONField(default=dict, blank=True)
    result = models.JSONField(default=dict, blank=True)
    transition_metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["workspace__name", "-calculated_priority", "created_at"]
        verbose_name = "4.6 GAME goal"
        verbose_name_plural = "4.6 GAME goals - work items"
        indexes = [
            models.Index(fields=["workspace", "status"], name="aihub_goal_ws_status_idx"),
            models.Index(fields=["workspace", "calculated_priority"], name="aihub_goal_ws_priority_idx"),
            models.Index(fields=["due_at"], name="ai_hub_goal_due_idx"),
        ]

    def __str__(self):
        return f"{self.title} ({self.workspace.name})"


class GameGoalDependency(models.Model):
    goal = models.ForeignKey(GameGoal, on_delete=models.CASCADE, related_name="dependencies")
    depends_on = models.ForeignKey(GameGoal, on_delete=models.CASCADE, related_name="required_by")
    is_required = models.BooleanField(default=True)
    note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["goal_id", "depends_on_id"]
        verbose_name = "4.7 GAME goal dependency"
        verbose_name_plural = "4.7 GAME goal dependencies"
        constraints = [
            models.UniqueConstraint(fields=["goal", "depends_on"], name="ai_hub_unique_goal_dependency"),
            models.CheckConstraint(
                condition=~models.Q(goal=models.F("depends_on")),
                name="ai_hub_goal_dependency_not_self",
            ),
        ]

    def __str__(self):
        return f"{self.goal.title} depends on {self.depends_on.title}"

    def clean(self):
        if not self.goal_id or not self.depends_on_id:
            return
        if self.goal_id == self.depends_on_id:
            raise ValidationError("A goal cannot depend on itself.")
        if self.goal.workspace_id != self.depends_on.workspace_id:
            raise ValidationError("Goal dependencies must belong to the same workspace.")

        pending = [self.depends_on_id]
        visited = set()
        while pending:
            current_id = pending.pop()
            if current_id == self.goal_id:
                raise ValidationError("Circular goal dependencies are not allowed.")
            if current_id in visited:
                continue
            visited.add(current_id)
            next_ids = (
                GameGoalDependency.objects.filter(goal_id=current_id)
                .exclude(pk=self.pk)
                .values_list("depends_on_id", flat=True)
            )
            pending.extend(next_ids)


class GameActionDefinition(models.Model):
    class ActionType(models.TextChoices):
        INTERNAL = "internal", "Internal"
        CONTEXT_TOOL = "context_tool", "Context tool"
        TOOL = "tool", "Tool"
        PYTHON_CALLABLE = "python_callable", "Python callable"
        HTTP = "http", "HTTP"
        SUB_AGENT = "sub_agent", "Sub-agent"
        HUMAN_APPROVAL = "human_approval", "Human approval"

    name = models.SlugField(max_length=120, unique=True)
    label = models.CharField(max_length=160)
    description = models.TextField(blank=True)
    action_type = models.CharField(max_length=30, choices=ActionType.choices)
    input_contract = models.JSONField(default=dict, blank=True)
    output_contract = models.JSONField(default=dict, blank=True)
    config = models.JSONField(default=dict, blank=True)
    risk_level = models.CharField(max_length=20, default="low")
    requires_approval = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "4.8 GAME action definition"
        verbose_name_plural = "4.8 GAME action definitions - dispatcher registry"

    def __str__(self):
        return f"{self.name} ({self.get_action_type_display()})"

    def clean(self):
        for field_name in ("input_contract", "output_contract", "config"):
            if not isinstance(getattr(self, field_name), dict):
                raise ValidationError({field_name: "Must be a JSON object."})
        if self.risk_level not in {"low", "medium", "high"}:
            raise ValidationError({"risk_level": "Risk level must be low, medium, or high."})


class ExecutionSession(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        WAITING_ASYNC = "waiting_async", "Waiting Async Continuation"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    class RuntimeKind(models.TextChoices):
        ORCHESTRATOR = "orchestrator", "Orchestrator"
        GAME = "game", "GAME"

    class RuntimeMode(models.TextChoices):
        SYNC = "sync", "Sync"
        ASYNC = "async", "Async"
        HYBRID = "hybrid", "Hybrid"

    source_content_type = models.ForeignKey(
        ContentType,
        on_delete=models.CASCADE,
        related_name="ai_hub_execution_sessions",
        null=True,
        blank=True,
    )
    source_object_id = models.PositiveBigIntegerField(null=True, blank=True)
    source_object = GenericForeignKey("source_content_type", "source_object_id")
    source_label = models.CharField(max_length=255, blank=True)
    goal = models.ForeignKey(
        GameGoal,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name="execution_sessions",
    )
    pipeline = models.ForeignKey(
        PipelineDefinition,
        on_delete=models.PROTECT,
        related_name="execution_sessions",
        null=True,
        blank=True,
    )
    entry_agent = models.ForeignKey(
        AgentProfile,
        on_delete=models.PROTECT,
        related_name="entry_execution_sessions",
        null=True,
        blank=True,
    )
    triggered_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="triggered_ai_hub_sessions",
    )
    runtime_kind = models.CharField(max_length=20, choices=RuntimeKind.choices, default=RuntimeKind.ORCHESTRATOR)
    runtime_mode = models.CharField(max_length=20, choices=RuntimeMode.choices, default=RuntimeMode.ASYNC)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    goal_text = models.TextField(blank=True)
    runtime_config = models.JSONField(default=dict, blank=True)
    initial_context = models.JSONField(default=dict, blank=True)
    final_context = models.JSONField(default=dict, blank=True)
    goal_outcome_fingerprint = models.CharField(max_length=64, blank=True)
    error_detail = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "5.0 Execution session"
        verbose_name_plural = "5.0 Execution sessions - generic runs"
        indexes = [
            models.Index(fields=["runtime_kind", "status"], name="ai_hub_session_kind_status_idx"),
            models.Index(fields=["source_content_type", "source_object_id"], name="ai_hub_session_source_idx"),
            models.Index(fields=["created_at"], name="ai_hub_session_created_idx"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["goal"],
                condition=models.Q(
                    goal__isnull=False,
                    status__in=["pending", "running", "waiting_async"],
                ),
                name="aihub_unique_active_goal",
            ),
        ]

    def __str__(self):
        label = self.source_label or self.source_object or "no source"
        return f"Session #{self.pk} - {label}"

    def clean(self):
        if self.runtime_kind == self.RuntimeKind.ORCHESTRATOR and not self.pipeline_id:
            raise ValidationError("Orchestrator sessions require a pipeline.")
        if self.runtime_kind == self.RuntimeKind.GAME and not (self.entry_agent_id or self.pipeline_id):
            raise ValidationError("GAME sessions require an entry agent or pipeline.")
        if self.runtime_kind == self.RuntimeKind.GAME and self.runtime_mode == self.RuntimeMode.HYBRID:
            raise ValidationError("GAME Hybrid continuation is not enabled yet. Use sync or async mode.")
        if self.goal_id and self.runtime_kind != self.RuntimeKind.GAME:
            raise ValidationError("Only GAME execution sessions can be linked to a GAME goal.")


class ExecutionStepRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"

    session = models.ForeignKey(ExecutionSession, on_delete=models.CASCADE, related_name="step_runs")
    order = models.PositiveIntegerField()
    pipeline_step = models.ForeignKey(
        PipelineStep,
        on_delete=models.PROTECT,
        related_name="execution_step_runs",
        null=True,
        blank=True,
    )
    agent = models.ForeignKey(
        AgentProfile,
        on_delete=models.PROTECT,
        related_name="execution_step_runs",
        null=True,
        blank=True,
    )
    action_name = models.CharField(max_length=140, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    request_payload = models.JSONField(default=dict, blank=True)
    response_payload = models.JSONField(default=dict, blank=True)
    observation_payload = models.JSONField(default=dict, blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    error_detail = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("session", "order")]
        ordering = ["session_id", "order"]
        verbose_name = "5.1 Execution step run"
        verbose_name_plural = "5.1 Execution step runs - generic step logs"
        indexes = [
            models.Index(fields=["session", "status"], name="ai_hub_step_session_status_idx"),
            models.Index(fields=["agent", "status"], name="ai_hub_step_agent_status_idx"),
        ]

    def __str__(self):
        return f"Session #{self.session_id} - step {self.order}"


class GameActionRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        WAITING_APPROVAL = "waiting_approval", "Waiting approval"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"
        REJECTED = "rejected", "Rejected"

    session = models.ForeignKey(
        ExecutionSession,
        on_delete=models.CASCADE,
        related_name="game_action_runs",
    )
    step_run = models.ForeignKey(
        ExecutionStepRun,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="game_action_runs",
    )
    action = models.ForeignKey(
        GameActionDefinition,
        on_delete=models.PROTECT,
        related_name="game_action_runs",
    )
    idempotency_key = models.CharField(max_length=255, unique=True)
    action_name = models.CharField(max_length=120)
    iteration = models.PositiveIntegerField()
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    input_payload = models.JSONField(default=dict, blank=True)
    output_payload = models.JSONField(default=dict, blank=True)
    observation_payload = models.JSONField(default=dict, blank=True)
    error_detail = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        ordering = ["session_id", "iteration"]
        verbose_name = "4.9 GAME action run"
        verbose_name_plural = "4.9 GAME action runs - dispatcher history"
        indexes = [
            models.Index(fields=["session", "iteration"], name="aihub_action_run_sess_iter_idx"),
            models.Index(fields=["action", "status"], name="aihub_game_run_action_stat_idx"),
        ]

    def __str__(self):
        return f"Action '{self.action_name}' - session #{self.session_id} iter {self.iteration}"


class GameMemoryEntry(models.Model):
    from decimal import Decimal as _Dec

    class ScopeType(models.TextChoices):
        WORKSPACE = "workspace", "Workspace"
        GOAL = "goal", "Goal"
        SESSION = "session", "Session"
        ACTION_RESULT = "action_result", "Action result"

    workspace = models.ForeignKey(GameWorkspace, on_delete=models.CASCADE, related_name="memory_entries")
    goal = models.ForeignKey(
        GameGoal, on_delete=models.CASCADE, null=True, blank=True, related_name="memory_entries"
    )
    session = models.ForeignKey(
        ExecutionSession, on_delete=models.CASCADE, null=True, blank=True, related_name="memory_entries"
    )
    scope_type = models.CharField(max_length=20, choices=ScopeType.choices)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    importance_score = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        default=_Dec("0.50"),
        validators=[MinValueValidator(_Dec("0.00")), MaxValueValidator(_Dec("1.00"))],
    )
    expires_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["workspace_id", "-importance_score", "created_at"]
        verbose_name = "4.10 GAME memory entry"
        verbose_name_plural = "4.10 GAME memory entries - scoped knowledge store"
        indexes = [
            models.Index(fields=["workspace", "scope_type"], name="aihub_mem_ws_scope_idx"),
            models.Index(fields=["goal", "scope_type"], name="aihub_mem_goal_scope_idx"),
            models.Index(fields=["workspace", "importance_score"], name="aihub_mem_ws_import_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(scope_type="workspace", goal__isnull=True, session__isnull=True)
                    | models.Q(scope_type__in=["goal", "action_result"], goal__isnull=False)
                    | models.Q(scope_type="session", session__isnull=False)
                ),
                name="aihub_memory_scope_shape",
            ),
        ]

    def __str__(self):
        return f"{self.scope_type} memory - workspace '{self.workspace.name}'"

    def clean(self):
        if self.scope_type == self.ScopeType.WORKSPACE:
            if self.goal_id is not None or self.session_id is not None:
                raise ValidationError("Workspace-scoped memory must not have a goal or session.")
        elif self.scope_type in {self.ScopeType.GOAL, self.ScopeType.ACTION_RESULT}:
            if not self.goal_id:
                raise ValidationError(
                    f"{self.get_scope_type_display()}-scoped memory requires a goal."
                )
            if self.goal_id and self.goal.workspace_id != self.workspace_id:
                raise ValidationError("Memory entry goal must belong to the same workspace.")
            if self.session_id and self.session.goal_id != self.goal_id:
                raise ValidationError("Goal/action-result memory session must belong to the same goal.")
        elif self.scope_type == self.ScopeType.SESSION:
            if not self.session_id:
                raise ValidationError("Session-scoped memory requires a session.")
            if self.session_id and self.session.goal_id:
                if self.session.goal.workspace_id != self.workspace_id:
                    raise ValidationError("Session memory must belong to the session goal's workspace.")
                if self.goal_id and self.session.goal_id != self.goal_id:
                    raise ValidationError("Memory entry goal must match session.goal when both are set.")
            elif self.session_id and self.goal_id:
                raise ValidationError(
                    "Memory for a legacy session without a goal cannot reference a GAME goal."
                )
        else:
            raise ValidationError({"scope_type": "Unknown GAME memory scope."})


class GameContinuationRequest(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RESOLVED = "resolved", "Resolved"
        EXPIRED = "expired", "Expired"
        CANCELLED = "cancelled", "Cancelled"

    class ReasonCode(models.TextChoices):
        NEEDS_INFORMATION = "needs_information", "Needs information"
        NEEDS_APPROVAL = "needs_approval", "Needs approval"
        EXTERNAL_RESULT_PENDING = "external_result_pending", "External result pending"
        RATE_LIMITED = "rate_limited", "Rate limited"
        MANUAL_REVIEW_REQUIRED = "manual_review_required", "Manual review required"

    session = models.ForeignKey(ExecutionSession, on_delete=models.CASCADE, related_name="continuation_requests")
    goal = models.ForeignKey(GameGoal, on_delete=models.CASCADE, related_name="continuation_requests")
    reason_code = models.CharField(max_length=80, choices=ReasonCode.choices)
    detail = models.TextField(blank=True)
    payload = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.PENDING)
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="resolved_continuation_requests",
    )

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "4.11 GAME continuation request"
        verbose_name_plural = "4.11 GAME continuation requests - pause records"
        indexes = [
            models.Index(fields=["session", "status"], name="aihub_cont_req_sess_stat_idx"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["session"],
                condition=models.Q(status="pending"),
                name="aihub_one_pending_continuation",
            ),
        ]

    def __str__(self):
        return f"Continuation #{self.pk} — {self.reason_code} ({self.status})"

    def clean(self):
        if self.session_id and self.goal_id and self.session.goal_id != self.goal_id:
            raise ValidationError("Continuation goal must match the session goal.")


class GameActionApprovalRequest(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"
        EXPIRED = "expired", "Expired"

    action_run = models.OneToOneField(GameActionRun, on_delete=models.CASCADE, related_name="approval_request")
    goal = models.ForeignKey(GameGoal, on_delete=models.CASCADE, related_name="approval_requests")
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.PENDING)
    requested_payload = models.JSONField(default=dict, blank=True)
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="reviewed_approval_requests",
    )
    review_note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "4.12 GAME action approval request"
        verbose_name_plural = "4.12 GAME action approval requests - gated actions"
        permissions = [("approve_game_action", "Can approve GAME action requests")]
        indexes = [
            models.Index(fields=["goal", "status"], name="aihub_approval_goal_stat_idx"),
        ]

    def __str__(self):
        return f"Approval #{self.pk} — '{self.action_run.action_name}' ({self.status})"

    def clean(self):
        if (
            self.action_run_id
            and self.goal_id
            and self.action_run.session.goal_id != self.goal_id
        ):
            raise ValidationError("Approval goal must match the action run session goal.")


class GameWorkspaceAction(models.Model):
    workspace = models.ForeignKey(
        GameWorkspace, on_delete=models.CASCADE, related_name="workspace_actions"
    )
    action = models.ForeignKey(
        GameActionDefinition, on_delete=models.CASCADE, related_name="workspace_entries"
    )
    is_enabled = models.BooleanField(default=True)
    requires_approval_override = models.BooleanField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["workspace__name", "action__name"]
        verbose_name = "4.13 GAME workspace action"
        verbose_name_plural = "4.13 GAME workspace actions - allow-list"
        constraints = [
            models.UniqueConstraint(fields=["workspace", "action"], name="ai_hub_unique_ws_action"),
        ]

    def __str__(self):
        status = "enabled" if self.is_enabled else "disabled"
        return f"{self.workspace.name} → {self.action.name} ({status})"


class GameWorkspaceAgent(models.Model):
    workspace = models.ForeignKey(
        GameWorkspace, on_delete=models.CASCADE, related_name="workspace_agents"
    )
    agent = models.ForeignKey(
        AgentProfile, on_delete=models.CASCADE, related_name="workspace_entries"
    )
    is_enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["workspace__name", "agent__name"]
        verbose_name = "4.14 GAME workspace agent"
        verbose_name_plural = "4.14 GAME workspace agents - allow-list"
        constraints = [
            models.UniqueConstraint(fields=["workspace", "agent"], name="ai_hub_unique_ws_agent"),
        ]

    def __str__(self):
        status = "enabled" if self.is_enabled else "disabled"
        return f"{self.workspace.name} → {self.agent.name} ({status})"


class GameGoalPlan(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        COMPLETED = "completed", "Completed"
        ABANDONED = "abandoned", "Abandoned"

    goal = models.OneToOneField(GameGoal, on_delete=models.CASCADE, related_name="plan")
    version = models.PositiveIntegerField(default=1)
    summary = models.TextField(blank=True)
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.ACTIVE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "4.15 GAME goal plan"
        verbose_name_plural = "4.15 GAME goal plans - structured execution aids"

    def __str__(self):
        return f"Plan #{self.pk} — {self.goal.title} ({self.status})"


class GameGoalPlanStep(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        IN_PROGRESS = "in_progress", "In progress"
        COMPLETED = "completed", "Completed"
        SKIPPED = "skipped", "Skipped"
        BLOCKED = "blocked", "Blocked"

    plan = models.ForeignKey(GameGoalPlan, on_delete=models.CASCADE, related_name="steps")
    order = models.PositiveIntegerField()
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.PENDING)
    depends_on_step = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="required_by_steps",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["plan_id", "order"]
        verbose_name = "4.16 GAME goal plan step"
        verbose_name_plural = "4.16 GAME goal plan steps"
        constraints = [
            models.UniqueConstraint(
                fields=["plan", "order"], name="ai_hub_unique_plan_step_order"
            ),
        ]

    def __str__(self):
        return f"Step {self.order}: {self.title}"

    def clean(self):
        if self.depends_on_step_id:
            if self.pk and self.depends_on_step_id == self.pk:
                raise ValidationError("A step cannot depend on itself.")
            if self.depends_on_step.plan_id != self.plan_id:
                raise ValidationError("Step dependency must belong to the same plan.")


class GameDelegationRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"

    parent_action_run = models.OneToOneField(
        GameActionRun, on_delete=models.CASCADE, related_name="delegation_run"
    )
    parent_goal = models.ForeignKey(
        GameGoal, on_delete=models.CASCADE, related_name="delegation_runs"
    )
    delegated_session = models.OneToOneField(
        ExecutionSession,
        on_delete=models.PROTECT,
        related_name="delegation_run",
        null=True,
        blank=True,
    )
    target_agent = models.ForeignKey(
        AgentProfile, on_delete=models.PROTECT, related_name="delegation_runs"
    )
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.PENDING)
    task = models.TextField()
    expected_result = models.TextField(blank=True)
    result_summary = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "4.17 GAME delegation run"
        verbose_name_plural = "4.17 GAME delegation runs - sub-agent history"
        indexes = [
            models.Index(
                fields=["parent_goal", "status"], name="aihub_delegation_goal_stat_idx"
            ),
        ]

    def __str__(self):
        return f"Delegation #{self.pk} → {self.target_agent.name} ({self.status})"


# === END REUSABLE AI PIPELINE CORE =========================================
