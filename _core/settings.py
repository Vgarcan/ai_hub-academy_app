from pathlib import Path
import os

from django.core.exceptions import ImproperlyConfigured

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent

DEV_SECRET_KEY = "django-insecure-ai-hub-academy-dev-key-change-in-production"
SECRET_KEY = os.environ.get("SECRET_KEY", DEV_SECRET_KEY)

DEBUG = os.environ.get("DEBUG", "True") == "True"

if not DEBUG and SECRET_KEY == DEV_SECRET_KEY:
    raise ImproperlyConfigured("SECRET_KEY must be set to a strong environment value when DEBUG=False.")

SECURE_SSL_REDIRECT = os.environ.get("SECURE_SSL_REDIRECT", str(not DEBUG)) == "True"
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", str(not DEBUG)) == "True"
CSRF_COOKIE_SECURE = os.environ.get("CSRF_COOKIE_SECURE", str(not DEBUG)) == "True"
SECURE_HSTS_SECONDS = int(os.environ.get("SECURE_HSTS_SECONDS", "0"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = (
    os.environ.get("SECURE_HSTS_INCLUDE_SUBDOMAINS", "False") == "True"
)
SECURE_HSTS_PRELOAD = os.environ.get("SECURE_HSTS_PRELOAD", "False") == "True"

ALLOWED_HOSTS = os.environ.get(
    "ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

CSRF_TRUSTED_ORIGINS = os.environ.get(
    "CSRF_TRUSTED_ORIGINS", "http://localhost,http://127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # AI Hub — reusable app (plug & play)
    "ai_hub",
    # Academy — documentation, tutorials, chatbot, missions
    "academy",
    # Support Demo — realistic workflow scenario
    "support_demo",
    # Dashboard — read-only visual explorer for all AI Hub entities
    "dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "_core.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "_core.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
# Static assets live in each installed app and are discovered automatically.
STATICFILES_DIRS = []
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "/admin/login/"
LOGIN_REDIRECT_URL = "/"

# Markdown documentation roots imported into the DB.
# AIHUB_DOCS_SOURCE is the canonical reusable-platform doc set (files 01-14);
# ACADEMY_DOCS_SOURCE holds only academy-specific docs (15+). The importer reads
# both, so there is a single source of truth and no duplicated/ drifting copies.
AIHUB_DOCS_SOURCE = BASE_DIR / "ai_hub" / "_docs"
ACADEMY_DOCS_SOURCE = BASE_DIR / "docs_source"

# Python-callable tools are code execution capabilities. Keep the allow-list explicit.
AI_HUB_ALLOWED_TOOL_CALLABLES = tuple(
    item.strip()
    for item in os.environ.get(
        "AI_HUB_ALLOWED_TOOL_CALLABLES",
        "academy.tools.doc_sync.sync_all_docs,academy.tools.doc_search.search_docs",
    ).split(",")
    if item.strip()
)

# Anonymous AI requests can create provider cost. Enable only with external rate limiting.
ACADEMY_ASSISTANT_ALLOW_ANONYMOUS = (
    os.environ.get("ACADEMY_ASSISTANT_ALLOW_ANONYMOUS", "False") == "True"
)

def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ImproperlyConfigured(f"{name} must be a boolean value, got {raw!r}.")


# GAME is enabled by default only for DEBUG/development. Production rollout is
# fail-closed unless each subsystem is explicitly enabled through the environment.
AI_HUB_GAME_GOALS_ENABLED = _env_bool("AI_HUB_GAME_GOALS_ENABLED", DEBUG)
AI_HUB_GAME_SCHEDULER_ENABLED = _env_bool("AI_HUB_GAME_SCHEDULER_ENABLED", DEBUG)
AI_HUB_GAME_ACTION_DISPATCH_ENABLED = _env_bool("AI_HUB_GAME_ACTION_DISPATCH_ENABLED", DEBUG)
AI_HUB_GAME_MEMORY_ENABLED = _env_bool("AI_HUB_GAME_MEMORY_ENABLED", DEBUG)
AI_HUB_GAME_RESUME_ENABLED = _env_bool("AI_HUB_GAME_RESUME_ENABLED", DEBUG)
AI_HUB_GAME_DELEGATION_ENABLED = _env_bool("AI_HUB_GAME_DELEGATION_ENABLED", DEBUG)
