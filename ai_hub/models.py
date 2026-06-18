from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db import models


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

    def __str__(self):
        label = self.source_label or self.source_object or "no source"
        return f"Session #{self.pk} - {label}"

    def clean(self):
        if self.runtime_kind == self.RuntimeKind.ORCHESTRATOR and not self.pipeline_id:
            raise ValidationError("Orchestrator sessions require a pipeline.")
        if self.runtime_kind == self.RuntimeKind.GAME and not (self.entry_agent_id or self.pipeline_id):
            raise ValidationError("GAME sessions require an entry agent or pipeline.")


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
# === END REUSABLE AI PIPELINE CORE =========================================
