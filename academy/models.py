from django.conf import settings
from django.db import models


class DocumentationSource(models.Model):
    name = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "Documentation source"
        verbose_name_plural = "Documentation sources"

    def __str__(self):
        return self.name


class DocumentationPage(models.Model):
    source = models.ForeignKey(
        DocumentationSource,
        related_name="pages",
        on_delete=models.CASCADE,
    )
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    source_path = models.CharField(max_length=500)
    body_markdown = models.TextField()
    order = models.PositiveIntegerField(default=1)
    is_active = models.BooleanField(default=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["order", "title"]
        verbose_name = "Documentation page"
        verbose_name_plural = "Documentation pages"

    def __str__(self):
        return self.title


class DocumentationChunk(models.Model):
    page = models.ForeignKey(
        DocumentationPage,
        related_name="chunks",
        on_delete=models.CASCADE,
    )
    heading = models.CharField(max_length=300)
    anchor = models.SlugField(max_length=300)
    body_markdown = models.TextField()
    order = models.PositiveIntegerField(default=1)
    # Flattened text for keyword search (no Markdown syntax)
    search_text = models.TextField(blank=True)
    token_estimate = models.PositiveIntegerField(default=0)
    # Semantic embedding vector — populated by embed_docs management command
    embedding = models.JSONField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["page__order", "order"]
        verbose_name = "Documentation chunk"
        verbose_name_plural = "Documentation chunks"

    def __str__(self):
        return f"{self.page.title} — {self.heading}"


class DocumentationChatSession(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="doc_chat_sessions",
    )
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Chat session"
        verbose_name_plural = "Chat sessions"

    def __str__(self):
        return self.title or f"Session #{self.pk}"


class DocumentationChatMessage(models.Model):
    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"
        SYSTEM = "system", "System"

    session = models.ForeignKey(
        DocumentationChatSession,
        related_name="messages",
        on_delete=models.CASCADE,
    )
    role = models.CharField(max_length=20, choices=Role.choices)
    content = models.TextField()
    retrieved_chunks = models.ManyToManyField(
        DocumentationChunk,
        blank=True,
        related_name="chat_messages",
    )
    ai_execution_session = models.ForeignKey(
        "ai_hub.ExecutionSession",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="doc_chat_messages",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        verbose_name = "Chat message"
        verbose_name_plural = "Chat messages"

    def __str__(self):
        return f"{self.role}: {self.content[:60]}"


class TutorialModule(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    order = models.PositiveIntegerField(default=1)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["order"]
        verbose_name = "Tutorial module"
        verbose_name_plural = "Tutorial modules"

    def __str__(self):
        return f"Module {self.order}: {self.title}"


class TutorialMission(models.Model):
    module = models.ForeignKey(
        TutorialModule,
        related_name="missions",
        on_delete=models.CASCADE,
    )
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    order = models.PositiveIntegerField(default=1)
    goal = models.TextField()
    instructions_markdown = models.TextField()
    # Key used to look up the validator function in the registry
    validation_key = models.CharField(max_length=100)
    # Optional key for starter/hint data
    starter_data_key = models.CharField(max_length=100, blank=True)
    related_docs = models.ManyToManyField(
        DocumentationPage,
        blank=True,
        related_name="missions",
    )
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["module__order", "order"]
        verbose_name = "Tutorial mission"
        verbose_name_plural = "Tutorial missions"

    def __str__(self):
        return f"{self.module.title} — {self.title}"


class UserMissionProgress(models.Model):
    class Status(models.TextChoices):
        NOT_STARTED = "not_started", "Not started"
        IN_PROGRESS = "in_progress", "In progress"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="mission_progress",
    )
    mission = models.ForeignKey(
        TutorialMission,
        on_delete=models.CASCADE,
        related_name="progress_records",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.NOT_STARTED,
    )
    attempts = models.PositiveIntegerField(default=0)
    last_feedback = models.TextField(blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [("user", "mission")]
        verbose_name = "Mission progress"
        verbose_name_plural = "Mission progress records"

    def __str__(self):
        return f"{self.user} — {self.mission.title} ({self.status})"


class LabExercise(models.Model):
    class Difficulty(models.TextChoices):
        BEGINNER = "beginner", "Beginner"
        INTERMEDIATE = "intermediate", "Intermediate"
        ADVANCED = "advanced", "Advanced"

    module = models.ForeignKey(
        TutorialModule,
        related_name="lab_exercises",
        on_delete=models.CASCADE,
    )
    mission = models.ForeignKey(
        TutorialMission,
        null=True,
        blank=True,
        related_name="lab_exercises",
        on_delete=models.SET_NULL,
    )
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    order = models.PositiveIntegerField(default=1)
    prompt = models.TextField()
    context = models.TextField(blank=True)
    evaluation_rubric = models.TextField(
        help_text="Instructions the AI evaluator uses to score and comment on the user's answer."
    )
    hint = models.TextField(blank=True)
    difficulty = models.CharField(
        max_length=20, choices=Difficulty.choices, default=Difficulty.BEGINNER
    )
    requires_api = models.BooleanField(
        default=True,
        help_text="Uncheck if this exercise works with the Training provider (no real API key needed).",
    )
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["module__order", "order"]
        verbose_name = "Lab exercise"
        verbose_name_plural = "Lab exercises"

    def __str__(self):
        return f"{self.module.title} — {self.title}"


class LabAttempt(models.Model):
    class Score(models.TextChoices):
        PASS = "pass", "Pass"
        PARTIAL = "partial", "Partial"
        FAIL = "fail", "Fail"
        PENDING = "pending", "Pending"

    exercise = models.ForeignKey(
        LabExercise,
        related_name="attempts",
        on_delete=models.CASCADE,
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="lab_attempts",
    )
    attempt_number = models.PositiveIntegerField(default=1)
    user_input = models.TextField()
    ai_feedback = models.TextField(blank=True)
    ai_score = models.CharField(
        max_length=10, choices=Score.choices, default=Score.PENDING
    )
    follow_up_question = models.TextField(blank=True)
    execution_session = models.ForeignKey(
        "ai_hub.ExecutionSession",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="lab_attempts",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Lab attempt"
        verbose_name_plural = "Lab attempts"

    def __str__(self):
        return f"{self.user} — {self.exercise.title} (attempt {self.attempt_number})"
