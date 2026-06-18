from django.contrib import admin

from .models import (
    DocumentationChunk,
    DocumentationChatMessage,
    DocumentationChatSession,
    DocumentationPage,
    DocumentationSource,
    LabAttempt,
    LabExercise,
    TutorialMission,
    TutorialModule,
    UserMissionProgress,
)


@admin.register(DocumentationSource)
class DocumentationSourceAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "is_active"]
    prepopulated_fields = {"slug": ["name"]}


class DocumentationChunkInline(admin.TabularInline):
    model = DocumentationChunk
    extra = 0
    fields = ["order", "heading", "anchor", "token_estimate", "is_active"]
    readonly_fields = ["token_estimate"]


@admin.register(DocumentationPage)
class DocumentationPageAdmin(admin.ModelAdmin):
    list_display = ["title", "source", "order", "is_active", "updated_at"]
    list_filter = ["source", "is_active"]
    prepopulated_fields = {"slug": ["title"]}
    inlines = [DocumentationChunkInline]


@admin.register(DocumentationChunk)
class DocumentationChunkAdmin(admin.ModelAdmin):
    list_display = ["heading", "page", "order", "token_estimate", "is_active"]
    list_filter = ["page__source", "is_active"]
    search_fields = ["heading", "search_text"]


class DocumentationChatMessageInline(admin.TabularInline):
    model = DocumentationChatMessage
    extra = 0
    fields = ["role", "content", "created_at"]
    readonly_fields = ["created_at"]


@admin.register(DocumentationChatSession)
class DocumentationChatSessionAdmin(admin.ModelAdmin):
    list_display = ["title", "user", "created_at"]
    inlines = [DocumentationChatMessageInline]


class TutorialMissionInline(admin.TabularInline):
    model = TutorialMission
    extra = 0
    fields = ["order", "title", "slug", "validation_key", "is_active"]
    prepopulated_fields = {"slug": ["title"]}


class LabExerciseInline(admin.TabularInline):
    model = LabExercise
    extra = 0
    fields = ["order", "title", "slug", "difficulty", "requires_api", "is_active"]
    prepopulated_fields = {"slug": ["title"]}
    show_change_link = True


@admin.register(TutorialModule)
class TutorialModuleAdmin(admin.ModelAdmin):
    list_display = ["title", "order", "is_active"]
    prepopulated_fields = {"slug": ["title"]}
    inlines = [TutorialMissionInline, LabExerciseInline]


@admin.register(TutorialMission)
class TutorialMissionAdmin(admin.ModelAdmin):
    list_display = ["title", "module", "order", "validation_key", "is_active"]
    list_filter = ["module", "is_active"]
    filter_horizontal = ["related_docs"]


@admin.register(UserMissionProgress)
class UserMissionProgressAdmin(admin.ModelAdmin):
    list_display = ["user", "mission", "status", "attempts", "completed_at"]
    list_filter = ["status"]
    readonly_fields = ["completed_at"]


class LabAttemptInline(admin.TabularInline):
    model = LabAttempt
    extra = 0
    fields = ["user", "attempt_number", "ai_score", "created_at"]
    readonly_fields = ["user", "attempt_number", "ai_score", "created_at"]
    show_change_link = True


@admin.register(LabExercise)
class LabExerciseAdmin(admin.ModelAdmin):
    list_display = ["title", "module", "mission", "difficulty", "requires_api", "is_active"]
    list_filter = ["module", "difficulty", "requires_api", "is_active"]
    prepopulated_fields = {"slug": ["title"]}
    search_fields = ["title", "prompt"]
    inlines = [LabAttemptInline]
    fieldsets = [
        (None, {"fields": ["module", "mission", "title", "slug", "order", "difficulty", "requires_api", "is_active"]}),
        ("Content", {"fields": ["prompt", "context", "hint"]}),
        ("AI Evaluator", {"fields": ["evaluation_rubric"], "description": "These instructions are sent to the AI to score and comment on the user's answer."}),
    ]


@admin.register(LabAttempt)
class LabAttemptAdmin(admin.ModelAdmin):
    list_display = ["user", "exercise", "attempt_number", "ai_score", "created_at"]
    list_filter = ["ai_score", "exercise__module"]
    readonly_fields = ["created_at", "execution_session"]
    search_fields = ["user__username", "exercise__title"]
