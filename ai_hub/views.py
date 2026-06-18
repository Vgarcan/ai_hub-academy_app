from django.contrib.admin.views.decorators import staff_member_required
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from ai_hub.models import ExecutionSession
from ai_hub.services.execution_runner import run_execution_session


@staff_member_required
@require_POST
def run_execution_session_endpoint(request):
    try:
        session_id = int(request.POST.get("session_id", "0"))
    except (TypeError, ValueError):
        return JsonResponse({"error": "session_id must be an integer"}, status=400)

    if not session_id:
        return JsonResponse({"error": "session_id is required"}, status=400)

    try:
        run_id = run_execution_session(session_id)
    except ValidationError as exc:
        return JsonResponse({"error": exc.messages}, status=400)
    try:
        session = ExecutionSession.objects.get(pk=run_id)
    except ExecutionSession.DoesNotExist:
        return JsonResponse({"error": f"Session {run_id} not found after run"}, status=404)
    payload = {
        "session_id": run_id,
        "status": session.status,
        "error": session.error_detail,
    }
    status_code = 400 if session.status == ExecutionSession.Status.FAILED else 200
    return JsonResponse(payload, status=status_code)
