from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("ai-hub/", include("ai_hub.urls")),
    path("dashboard/", include("dashboard.urls")),
    path("", include("academy.urls")),
]
