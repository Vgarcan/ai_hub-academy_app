from django.urls import path

from . import views

app_name = "academy"

urlpatterns = [
    # Landing
    path("", views.landing, name="landing"),

    # Documentation
    path("docs/", views.docs_list, name="docs_list"),
    path("docs/search/", views.docs_search, name="docs_search"),
    path("docs/<slug:slug>/", views.docs_detail, name="docs_detail"),

    # Tutorials
    path("tutorials/", views.tutorial_list, name="tutorial_list"),
    path("tutorials/<slug:module_slug>/", views.tutorial_module, name="tutorial_module"),
    path("tutorials/<slug:module_slug>/<slug:mission_slug>/", views.tutorial_mission, name="tutorial_mission"),
    path(
        "tutorials/<slug:module_slug>/<slug:mission_slug>/check/",
        views.check_mission,
        name="check_mission",
    ),
    path("progress/", views.progress_dashboard, name="progress"),

    # Lab exercises
    path(
        "tutorials/<slug:module_slug>/lab/<slug:exercise_slug>/",
        views.lab_exercise,
        name="lab_exercise",
    ),
    path(
        "tutorials/<slug:module_slug>/lab/<slug:exercise_slug>/submit/",
        views.lab_submit,
        name="lab_submit",
    ),

    # Documentation Assistant (chatbot)
    path("assistant/", views.assistant, name="assistant"),
    path("assistant/ask/", views.assistant_ask, name="assistant_ask"),
]
