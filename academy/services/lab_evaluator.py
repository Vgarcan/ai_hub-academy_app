"""
Lab Exercise Evaluator service.

Calls the 'Lab Exercise Evaluator' AgentProfile to score a student's answer.
Returns a dict: {score, feedback, follow_up_question, ai_session_pk}.
Falls back gracefully when no AI is configured.
"""
import json
import logging

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """You are an educational assistant that evaluates student answers for AI Hub Academy lab exercises.

You receive the following in your context:
- exercise_title: the exercise name
- exercise_prompt: the challenge the student was given
- exercise_context: background information shown to the student
- evaluation_rubric: the criteria you must use to score the answer
- student_answer: what the student wrote

You MUST respond with ONLY this exact JSON (no prose, no markdown fence, nothing else):
{
  "score": "pass",
  "feedback": "Detailed, constructive feedback — what was good, what was missing, how to improve. Be specific. 2-4 sentences.",
  "follow_up_question": "A question to deepen understanding. Leave empty string if not needed."
}

Valid score values: "pass", "partial", "fail"
- pass: student clearly meets all rubric criteria
- partial: student shows some understanding but is missing key elements
- fail: answer is incorrect, too vague, or shows a fundamental misunderstanding

Always be encouraging and educational, never harsh."""


def evaluate_lab_answer(*, exercise, user_input: str, user=None) -> dict:
    """
    Evaluate a student's answer for a LabExercise.

    Returns:
        {
          "score": "pass" | "partial" | "fail" | "pending",
          "feedback": str,
          "follow_up_question": str,
          "ai_session_pk": int | None,
        }
    """
    initial_context = {
        "exercise_title": exercise.title,
        "exercise_prompt": exercise.prompt,
        "exercise_context": exercise.context,
        "evaluation_rubric": exercise.evaluation_rubric,
        "student_answer": user_input,
    }

    ai_session = None
    result = None

    try:
        from ai_hub.models import AgentProfile, ExecutionSession
        from ai_hub.services.execution_runner import run_execution_session

        agent = AgentProfile.objects.filter(
            name="Lab Exercise Evaluator",
            is_active=True,
        ).first()

        if agent:
            session = ExecutionSession.objects.create(
                runtime_kind=ExecutionSession.RuntimeKind.GAME,
                entry_agent=agent,
                goal_text=f"Evaluate student answer for: {exercise.title}",
                initial_context=initial_context,
                triggered_by=user if (user and user.is_authenticated) else None,
                source_label="Academy Lab",
            )
            run_execution_session(session.pk)
            session.refresh_from_db()
            ai_session = session

            final_answer = (session.final_context or {}).get("final_answer", "")
            if final_answer:
                result = _parse_evaluation(final_answer)

    except ImportError:
        pass
    except Exception:
        logger.exception("Lab evaluation failed for exercise pk=%s", exercise.pk)

    if result is None:
        result = {
            "score": "pending",
            "feedback": (
                "AI evaluation is not available right now. "
                "Compare your answer to the criteria below and self-assess.\n\n"
                f"**Evaluation criteria:**\n{exercise.evaluation_rubric}"
            ),
            "follow_up_question": "",
        }

    result["ai_session_pk"] = ai_session.pk if ai_session else None
    return result


def _parse_evaluation(text: str) -> dict | None:
    """Extract {score, feedback, follow_up_question} from AI final_answer."""
    stripped = text.strip()

    # Strip markdown code fences if present
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        stripped = "\n".join(inner)

    try:
        data = json.loads(stripped)
        score_raw = str(data.get("score", "")).lower().strip()
        if score_raw not in ("pass", "partial", "fail"):
            score_raw = "partial"
        return {
            "score": score_raw,
            "feedback": str(data.get("feedback", text)),
            "follow_up_question": str(data.get("follow_up_question", "")),
        }
    except (json.JSONDecodeError, AttributeError):
        pass

    # Heuristic fallback: extract score from prose
    lower = text.lower()
    if "pass" in lower and "fail" not in lower:
        score = "pass"
    elif "fail" in lower and "partial" not in lower:
        score = "fail"
    else:
        score = "partial"

    return {
        "score": score,
        "feedback": text,
        "follow_up_question": "",
    }
