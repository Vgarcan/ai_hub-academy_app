import logging

from academy.models import DocumentationChatMessage, DocumentationChatSession
from academy.services.documentation_search import search_documentation

logger = logging.getLogger(__name__)


def answer_documentation_question(
    *,
    user,
    chat_session: DocumentationChatSession,
    question: str,
    current_mission=None,
) -> DocumentationChatMessage:
    """
    Handle one user question in a documentation chat session.

    Retrieves relevant chunks, runs an AI Hub ExecutionSession for
    auditability, saves and returns the assistant's response message.
    """
    # Save the user's question
    DocumentationChatMessage.objects.create(
        session=chat_session,
        role=DocumentationChatMessage.Role.USER,
        content=question,
    )

    # Retrieve relevant documentation chunks
    chunks = search_documentation(question, limit=5)

    # Build context for the agent.
    # Deliberately avoids [bracket] format so the AI does not copy citation
    # markers into the response text (sources are shown separately by the UI).
    retrieved_context = "\n\n".join(
        f"--- {chunk.page.title}: {chunk.heading} ---\n{chunk.body_markdown[:1500]}"
        for chunk in chunks
    )
    # dict.fromkeys preserves insertion order while deduplicating
    source_pages = list(dict.fromkeys(chunk.page.title for chunk in chunks))

    initial_context = {
        "question": question,
        "retrieved_context": retrieved_context,
        "source_pages": source_pages,
        # Explicit instruction: the UI already shows source chips — no inline citations needed
        "response_note": (
            "Answer the question directly and clearly. "
            "Do NOT include source references, bracketed citations, or page names inside your response text. "
            "The sources are already displayed to the user by the interface."
        ),
    }

    if current_mission:
        initial_context["current_mission"] = {
            "title": current_mission.title,
            "goal": current_mission.goal,
            "module": current_mission.module.title,
        }

    # Try to run through AI Hub if an agent is configured
    ai_execution_session = None
    answer_text = ""

    try:
        from ai_hub.models import AgentProfile, ExecutionSession
        from ai_hub.services.execution_runner import run_execution_session

        agent = AgentProfile.objects.filter(
            name="AI Hub Documentation Assistant",
            is_active=True,
        ).first()

        if agent:
            session = ExecutionSession.objects.create(
                runtime_kind=ExecutionSession.RuntimeKind.GAME,
                entry_agent=agent,
                goal_text=question,
                initial_context=initial_context,
                triggered_by=user if (user and user.is_authenticated) else None,
                source_label="Documentation Chatbot",
            )
            run_execution_session(session.pk)
            session.refresh_from_db()
            ai_execution_session = session
            answer_text = (session.final_context or {}).get("final_answer", "")

    except ImportError:
        pass  # ai_hub app not installed — expected in some test environments
    except Exception:
        logger.exception("AI execution failed for doc chat session pk=%s", chat_session.pk)

    if not answer_text:
        # Fallback: show clean extracted content (no source headers in the text)
        if chunks:
            clean_sections = "\n\n".join(
                chunk.body_markdown[:600] for chunk in chunks[:3]
            )
            answer_text = clean_sections or "No relevant content found in the documentation."
        else:
            answer_text = (
                "I could not find relevant documentation for your question. "
                "Try searching the docs directly or rephrasing your question."
            )

    # Save the assistant's response
    assistant_msg = DocumentationChatMessage.objects.create(
        session=chat_session,
        role=DocumentationChatMessage.Role.ASSISTANT,
        content=answer_text,
        ai_execution_session=ai_execution_session,
    )
    if chunks:
        assistant_msg.retrieved_chunks.set(chunks)

    return assistant_msg
