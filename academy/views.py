from django.db.models import Count, F, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from .models import (
    DocumentationChatSession,
    DocumentationPage,
    LabAttempt,
    LabExercise,
    TutorialMission,
    TutorialModule,
    UserMissionProgress,
)
from .services.documentation_assistant import answer_documentation_question
from .services.documentation_search import search_documentation
from .services.lab_evaluator import evaluate_lab_answer
from .services.mission_validators import validate_mission


# ---------------------------------------------------------------------------
# Landing
# ---------------------------------------------------------------------------

def landing(request):
    from ai_hub.models import AgentProfile, ExecutionSession, ProviderConfig
    context = {
        "provider_count": ProviderConfig.objects.filter(is_active=True).count(),
        "agent_count": AgentProfile.objects.filter(is_active=True).count(),
        "session_count": ExecutionSession.objects.count(),
    }
    return render(request, "academy/landing.html", context)


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

def docs_list(request):
    pages = DocumentationPage.objects.filter(
        is_active=True).select_related("source").order_by("order")
    return render(request, "academy/docs_list.html", {"pages": pages})


def docs_detail(request, slug):
    page = get_object_or_404(DocumentationPage, slug=slug, is_active=True)
    chunks = page.chunks.filter(is_active=True).order_by("order")
    all_pages = DocumentationPage.objects.filter(
        is_active=True).order_by("order")
    return render(request, "academy/docs_detail.html", {
        "page": page,
        "chunks": chunks,
        "all_pages": all_pages,
    })


def docs_search(request):
    query = request.GET.get("q", "").strip()
    results = []
    if query:
        results = search_documentation(query, limit=10)
    return render(request, "academy/docs_search.html", {
        "query": query,
        "results": results,
    })


# ---------------------------------------------------------------------------
# Tutorials
# ---------------------------------------------------------------------------

def tutorial_list(request):
    modules = TutorialModule.objects.filter(
        is_active=True).prefetch_related("missions")
    progress_map = {}
    if request.user.is_authenticated:
        for p in UserMissionProgress.objects.filter(user=request.user):
            progress_map[p.mission_id] = p
    return render(request, "academy/tutorial_list.html", {
        "modules": modules,
        "progress_map": progress_map,
    })


def tutorial_module(request, module_slug):
    module = get_object_or_404(
        TutorialModule, slug=module_slug, is_active=True)
    missions = module.missions.filter(is_active=True).order_by("order")
    lab_exercises = module.lab_exercises.filter(
        is_active=True).order_by("order")

    # Django templates can't do dict[variable_key], so pre-compute simple sets/lists
    completed_missions = set()
    started_missions = set()
    lab_score_map = {}  # exercise_pk → score string (internal use only)

    if request.user.is_authenticated:
        for p in UserMissionProgress.objects.filter(user=request.user, mission__module=module):
            if p.status == UserMissionProgress.Status.COMPLETED:
                completed_missions.add(p.mission_id)
            else:
                started_missions.add(p.mission_id)

        for attempt in LabAttempt.objects.filter(
            user=request.user,
            exercise__in=lab_exercises,
        ).order_by("-created_at"):
            if attempt.exercise_id not in lab_score_map:
                lab_score_map[attempt.exercise_id] = attempt.ai_score

    # Pre-annotate lab exercises so the template can access score without dict[key]
    lab_exercise_data = [
        {
            "exercise": ex,
            "score": lab_score_map.get(ex.pk, ""),
            "attempted": ex.pk in lab_score_map,
        }
        for ex in lab_exercises
    ]

    return render(request, "academy/tutorial_module.html", {
        "module": module,
        "missions": missions,
        "completed_missions": completed_missions,
        "started_missions": started_missions,
        "lab_exercise_data": lab_exercise_data,
    })


def tutorial_mission(request, module_slug, mission_slug):
    module = get_object_or_404(
        TutorialModule, slug=module_slug, is_active=True)
    mission = get_object_or_404(
        TutorialMission, slug=mission_slug, module=module, is_active=True)
    related_docs = mission.related_docs.filter(is_active=True)

    progress = None
    if request.user.is_authenticated:
        progress, _ = UserMissionProgress.objects.get_or_create(
            user=request.user,
            mission=mission,
        )
        if progress.status == UserMissionProgress.Status.NOT_STARTED:
            progress.status = UserMissionProgress.Status.IN_PROGRESS
            progress.save(update_fields=["status"])

    suggested_questions = [
        "What should I do in this mission?",
        f"What is {mission.module.title}?",
        "Why did my validation fail?",
        "Show me an example.",
    ]

    return render(request, "academy/tutorial_mission.html", {
        "module": module,
        "mission": mission,
        "progress": progress,
        "related_docs": related_docs,
        "suggested_questions": suggested_questions,
    })


@require_POST
def check_mission(request, module_slug, mission_slug):
    if not request.user.is_authenticated:
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JsonResponse({"passed": False, "feedback": "Login required to track progress."}, status=403)
        return redirect(f"/admin/login/?next={request.path}")

    module = get_object_or_404(
        TutorialModule, slug=module_slug, is_active=True)
    mission = get_object_or_404(
        TutorialMission, slug=mission_slug, module=module, is_active=True)

    passed, feedback = validate_mission(user=request.user, mission=mission)

    if request.user.is_authenticated:
        progress, _ = UserMissionProgress.objects.get_or_create(
            user=request.user,
            mission=mission,
        )
        UserMissionProgress.objects.filter(
            pk=progress.pk).update(attempts=F("attempts") + 1)
        update_fields = {"last_feedback": feedback}
        if passed:
            update_fields["status"] = UserMissionProgress.Status.COMPLETED
            update_fields["completed_at"] = timezone.now()
        else:
            update_fields["status"] = UserMissionProgress.Status.IN_PROGRESS
        UserMissionProgress.objects.filter(
            pk=progress.pk).update(**update_fields)
        progress.refresh_from_db()

    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JsonResponse({"passed": passed, "feedback": feedback})

    return redirect("academy:tutorial_mission", module_slug=module_slug, mission_slug=mission_slug)


def lab_exercise(request, module_slug, exercise_slug):
    module = get_object_or_404(
        TutorialModule, slug=module_slug, is_active=True)
    exercise = get_object_or_404(
        LabExercise, slug=exercise_slug, module=module, is_active=True)

    past_attempts = []
    last_attempt = None
    best_score = "pending"

    if request.user.is_authenticated:
        past_attempts = list(
            LabAttempt.objects.filter(
                user=request.user,
                exercise=exercise,
            ).order_by("-attempt_number")
        )
        if past_attempts:
            last_attempt = past_attempts[0]
            score_rank = {"pass": 3, "partial": 2, "fail": 1, "pending": 0}
            best_score = max(past_attempts, key=lambda a: score_rank.get(
                a.ai_score, 0)).ai_score

    next_exercise = (
        LabExercise.objects.filter(
            module=module, order__gt=exercise.order, is_active=True)
        .order_by("order")
        .first()
    )

    return render(request, "academy/lab_exercise.html", {
        "module": module,
        "exercise": exercise,
        "past_attempts": past_attempts,
        "last_attempt": last_attempt,
        "best_score": best_score,
        "next_exercise": next_exercise,
    })


@require_POST
def lab_submit(request, module_slug, exercise_slug):
    is_ajax = request.headers.get("x-requested-with") == "XMLHttpRequest"

    if not request.user.is_authenticated:
        if is_ajax:
            return JsonResponse({"error": "Login required to submit lab answers."}, status=403)
        return redirect(f"/admin/login/?next={request.path}")

    module = get_object_or_404(
        TutorialModule, slug=module_slug, is_active=True)
    exercise = get_object_or_404(
        LabExercise, slug=exercise_slug, module=module, is_active=True)

    user_input = request.POST.get("user_input", "").strip()
    if not user_input:
        if is_ajax:
            return JsonResponse({"error": "Please write an answer before submitting."}, status=400)
        return redirect("academy:lab_exercise", module_slug=module_slug, exercise_slug=exercise_slug)

    attempt_number = (
        LabAttempt.objects.filter(
            user=request.user, exercise=exercise).count() + 1
    )

    attempt = LabAttempt.objects.create(
        exercise=exercise,
        user=request.user,
        attempt_number=attempt_number,
        user_input=user_input,
        ai_score=LabAttempt.Score.PENDING,
    )

    result = evaluate_lab_answer(
        exercise=exercise, user_input=user_input, user=request.user)

    LabAttempt.objects.filter(pk=attempt.pk).update(
        ai_feedback=result["feedback"],
        ai_score=result["score"],
        follow_up_question=result.get("follow_up_question", ""),
        execution_session_id=result.get("ai_session_pk"),
    )
    attempt.refresh_from_db()

    if is_ajax:
        return JsonResponse({
            "score": attempt.ai_score,
            "score_display": attempt.get_ai_score_display(),
            "feedback": attempt.ai_feedback,
            "follow_up_question": attempt.follow_up_question,
            "attempt_number": attempt.attempt_number,
        })

    return redirect("academy:lab_exercise", module_slug=module_slug, exercise_slug=exercise_slug)


def progress_dashboard(request):
    if not request.user.is_authenticated:
        return redirect("admin:login")

    modules = TutorialModule.objects.filter(
        is_active=True).prefetch_related("missions")
    progress_records = UserMissionProgress.objects.filter(
        user=request.user).select_related("mission")
    progress_map = {p.mission_id: p for p in progress_records}

    total_missions = TutorialMission.objects.filter(
        module__in=modules, is_active=True).count()
    completed = sum(1 for p in progress_records if p.status ==
                    UserMissionProgress.Status.COMPLETED)
    pct = int(completed / total_missions * 100) if total_missions else 0

    return render(request, "academy/progress_dashboard.html", {
        "modules": modules,
        "progress_map": progress_map,
        "total_missions": total_missions,
        "completed": completed,
        "pct": pct,
    })


# ---------------------------------------------------------------------------
# Documentation Chatbot
# ---------------------------------------------------------------------------

def assistant(request):
    session = None
    messages = []

    if request.user.is_authenticated:
        session_id = request.GET.get("session")
        if session_id:
            session = DocumentationChatSession.objects.filter(
                pk=session_id, user=request.user
            ).first()
        if not session:
            session = DocumentationChatSession.objects.filter(
                user=request.user
            ).order_by("-created_at").first()

    if session:
        # Last 50 messages in ascending order (for display)
        messages = list(reversed(list(
            session.messages
            .order_by("-created_at")
            .select_related("ai_execution_session")[:50]
        )))

    current_mission = None
    mission_slug = request.GET.get("mission")
    if mission_slug:
        current_mission = TutorialMission.objects.filter(
            slug=mission_slug, is_active=True).first()

    return render(request, "academy/assistant.html", {
        "session": session,
        "chat_messages": messages,
        "current_mission": current_mission,
    })


@require_POST
def assistant_ask(request):
    is_ajax = request.headers.get("x-requested-with") == "XMLHttpRequest"
    question = request.POST.get("question", "").strip()
    mission_slug = request.POST.get("mission_slug", "")

    if not question:
        if is_ajax:
            return JsonResponse({"error": "empty question"}, status=400)
        return redirect("academy:assistant")

    if request.user.is_authenticated:
        session = DocumentationChatSession.objects.filter(
            user=request.user
        ).order_by("-created_at").first()
        if not session:
            session = DocumentationChatSession.objects.create(
                user=request.user,
                title=question[:100],
            )
    else:
        # Reuse the anonymous session stored in the Django session to prevent
        # unbounded row growth from bots and repeat visitors
        if not request.session.session_key:
            request.session.create()
        anon_pk = request.session.get("doc_chat_session_pk")
        session = DocumentationChatSession.objects.filter(
            pk=anon_pk).first() if anon_pk else None
        if not session:
            session = DocumentationChatSession.objects.create(
                title=question[:100])
            request.session["doc_chat_session_pk"] = session.pk

    current_mission = None
    if mission_slug:
        current_mission = TutorialMission.objects.filter(
            slug=mission_slug, is_active=True).first()

    assistant_msg = answer_documentation_question(
        user=request.user if request.user.is_authenticated else None,
        chat_session=session,
        question=question,
        current_mission=current_mission,
    )

    if is_ajax:
        sources = [
            {
                "page_title": chunk.page.title,
                "heading": chunk.heading,
                "page_slug": chunk.page.slug,
            }
            for chunk in assistant_msg.retrieved_chunks.select_related("page").all()
        ]
        return JsonResponse({
            "answer": assistant_msg.content,
            "sources": sources,
            "session_pk": session.pk,
            "ai_session_pk": assistant_msg.ai_execution_session_id,
        })

    return redirect(f"{reverse('academy:assistant')}?session={session.pk}")
