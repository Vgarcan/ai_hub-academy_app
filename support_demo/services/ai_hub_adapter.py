"""
Adapter that connects SupportTicket triage to AI Hub ExecutionSession.

This is the boundary between the support_demo domain and the reusable ai_hub platform.
"""
import json

from ai_hub.models import ExecutionSession, PipelineDefinition
from ai_hub.services.execution_runner import run_execution_session

from support_demo.models import SupportTicket, TicketAnalysis


def run_ticket_triage(ticket: SupportTicket, user=None) -> ExecutionSession:
    """
    Create an ExecutionSession for ticket triage, run it, and save the result.

    The pipeline named 'Ticket Triage Pipeline' must exist and be active.
    """
    pipeline = PipelineDefinition.objects.filter(
        name__icontains="ticket triage",
        is_active=True,
    ).first()

    if not pipeline:
        raise ValueError(
            "No active 'Ticket Triage Pipeline' found. "
            "Create one in Admin → AI Hub → Pipeline definitions."
        )

    ticket.status = SupportTicket.Status.ANALYSING
    ticket.save(update_fields=["status", "updated_at"])

    session = ExecutionSession.objects.create(
        pipeline=pipeline,
        runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR,
        initial_context={
            "ticket_title": ticket.title,
            "ticket_text": ticket.body,
        },
        source_label=f"Ticket: {ticket.title}",
        triggered_by=user if (user and hasattr(user, "is_authenticated") and user.is_authenticated) else None,
    )

    # Link the ticket to its AI session before running
    ticket.ai_session = session
    ticket.save(update_fields=["ai_session", "updated_at"])

    run_execution_session(session.pk)
    session.refresh_from_db()

    if session.status == ExecutionSession.Status.SUCCESS:
        ticket.status = SupportTicket.Status.TRIAGED
        _save_analysis(ticket, session)
    else:
        ticket.status = SupportTicket.Status.NEW

    ticket.save(update_fields=["status", "updated_at"])
    return session


def _save_analysis(ticket: SupportTicket, session: ExecutionSession) -> None:
    """Extract structured output from the final context and save TicketAnalysis."""
    final_context = session.final_context or {}

    # Try to get the last step's output as JSON
    raw_output = final_context.get("final_output") or final_context.get("output") or {}
    if isinstance(raw_output, str):
        try:
            raw_output = json.loads(raw_output)
        except (json.JSONDecodeError, ValueError):
            raw_output = {}

    category = str(raw_output.get("category") or final_context.get("category") or "Unknown")
    priority = str(raw_output.get("priority") or final_context.get("priority") or "Unknown")
    reason = str(raw_output.get("reason") or final_context.get("reason") or "")

    TicketAnalysis.objects.update_or_create(
        ticket=ticket,
        defaults={"category": category, "priority": priority, "reason": reason},
    )
