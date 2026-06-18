from django.urls import path

from .views import run_execution_session_endpoint

app_name = "ai_hub"

urlpatterns = [
    path("internal/execution-session/run/", run_execution_session_endpoint, name="run_execution_session"),
]
