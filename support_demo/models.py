from django.db import models


class SupportTicket(models.Model):
    class Status(models.TextChoices):
        NEW = "new", "New"
        ANALYSING = "analysing", "Analysing"
        TRIAGED = "triaged", "Triaged"
        CLOSED = "closed", "Closed"

    title = models.CharField(max_length=200)
    body = models.TextField()
    status = models.CharField(max_length=50, choices=Status.choices, default=Status.NEW)
    ai_session = models.OneToOneField(
        "ai_hub.ExecutionSession",
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="support_ticket",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Support ticket"
        verbose_name_plural = "Support tickets"

    def __str__(self):
        return self.title


class TicketAnalysis(models.Model):
    ticket = models.OneToOneField(
        SupportTicket,
        related_name="analysis",
        on_delete=models.CASCADE,
    )
    category = models.CharField(max_length=100)
    priority = models.CharField(max_length=100)
    reason = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Ticket analysis"
        verbose_name_plural = "Ticket analyses"

    def __str__(self):
        return f"Analysis for: {self.ticket.title}"
