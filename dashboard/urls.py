from django.urls import path
from . import views

urlpatterns = [
    path("", views.overview, name="dashboard_overview"),
    path("providers/", views.providers, name="dashboard_providers"),
    path("providers/<int:pk>/", views.provider_detail, name="dashboard_provider_detail"),
    path("agents/", views.agents, name="dashboard_agents"),
    path("agents/<int:pk>/", views.agent_detail, name="dashboard_agent_detail"),
    path("pipelines/", views.pipelines, name="dashboard_pipelines"),
    path("pipelines/<int:pk>/", views.pipeline_detail, name="dashboard_pipeline_detail"),
    path("sessions/", views.sessions, name="dashboard_sessions"),
    path("sessions/<int:pk>/", views.session_detail, name="dashboard_session_detail"),
    path("api-status/", views.api_status, name="dashboard_api_status"),
]
