from django.contrib import admin

from .models import SupportTicket, TicketAnalysis


class TicketAnalysisInline(admin.StackedInline):
    model = TicketAnalysis
    extra = 0
    readonly_fields = ["category", "priority", "reason", "created_at"]


@admin.register(SupportTicket)
class SupportTicketAdmin(admin.ModelAdmin):
    list_display = ["title", "status", "ai_session", "created_at"]
    list_filter = ["status"]
    readonly_fields = ["ai_session", "created_at", "updated_at"]
    inlines = [TicketAnalysisInline]
    actions = ["run_triage"]

    def run_triage(self, request, queryset):
        from support_demo.services.ai_hub_adapter import run_ticket_triage
        count = 0
        for ticket in queryset.filter(ai_session__isnull=True):
            try:
                run_ticket_triage(ticket, request.user)
                count += 1
            except Exception as exc:
                self.message_user(request, f"Error on '{ticket.title}': {exc}", level="error")
        self.message_user(request, f"Triage started for {count} ticket(s).")
    run_triage.short_description = "Run AI triage on selected tickets"
