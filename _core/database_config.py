from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlparse

from django.core.exceptions import ImproperlyConfigured


POSTGRES_SCHEMES = {"postgres", "postgresql"}
SQLITE_ENGINES = {"sqlite", "sqlite3"}
POSTGRES_ENGINES = POSTGRES_SCHEMES
POSTGRES_URL_OPTIONS = {
    "sslmode",
    "sslrootcert",
    "sslcert",
    "sslkey",
    "target_session_attrs",
}


def _non_negative_int(value, *, name: str, default: int) -> int:
    raw = str(default if value in (None, "") else value)
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ImproperlyConfigured(f"{name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ImproperlyConfigured(f"{name} must be a non-negative integer.")
    return parsed


def _environment_bool(value, *, default: bool) -> bool:
    if value in (None, ""):
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ImproperlyConfigured(
        "DB_CONN_HEALTH_CHECKS must be a boolean value."
    )


def _postgres_port(value, *, name: str = "POSTGRES_PORT") -> str:
    raw = str(value or "5432").strip()
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ImproperlyConfigured(
            f"{name} must be an integer between 1 and 65535."
        ) from exc
    if not 1 <= parsed <= 65535:
        raise ImproperlyConfigured(
            f"{name} must be an integer between 1 and 65535."
        )
    return str(parsed)


def _postgres_common(environment) -> dict:
    return {
        "CONN_MAX_AGE": _non_negative_int(
            environment.get("DB_CONN_MAX_AGE"),
            name="DB_CONN_MAX_AGE",
            default=0,
        ),
        "CONN_HEALTH_CHECKS": _environment_bool(
            environment.get("DB_CONN_HEALTH_CHECKS"),
            default=True,
        ),
    }


def _postgres_from_url(database_url: str, environment) -> dict:
    parsed = urlparse(database_url)
    if parsed.scheme.lower() not in POSTGRES_SCHEMES:
        raise ImproperlyConfigured(
            "DATABASE_URL must use the postgres:// or postgresql:// scheme."
        )
    database_name = unquote(parsed.path.lstrip("/"))
    if not database_name:
        raise ImproperlyConfigured("DATABASE_URL must include a database name.")
    try:
        port = str(parsed.port or "")
        hostname = parsed.hostname or ""
    except ValueError as exc:
        raise ImproperlyConfigured(
            "DATABASE_URL contains an invalid PostgreSQL port."
        ) from exc

    options = {
        key: value
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key in POSTGRES_URL_OPTIONS
    }
    config = {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": database_name,
        "USER": unquote(parsed.username or ""),
        "PASSWORD": unquote(parsed.password or ""),
        "HOST": hostname,
        "PORT": port,
        **_postgres_common(environment),
    }
    if options:
        config["OPTIONS"] = options
    return config


def _postgres_from_environment(environment) -> dict:
    database_name = (
        environment.get("POSTGRES_DB")
        or environment.get("DB_NAME")
        or ""
    ).strip()
    if not database_name:
        raise ImproperlyConfigured(
            "POSTGRES_DB (or DB_NAME) is required when "
            "DATABASE_ENGINE=postgresql."
        )
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": database_name,
        "USER": environment.get("POSTGRES_USER", environment.get("DB_USER", "")),
        "PASSWORD": environment.get(
            "POSTGRES_PASSWORD",
            environment.get("DB_PASSWORD", ""),
        ),
        "HOST": environment.get("POSTGRES_HOST", environment.get("DB_HOST", "")),
        "PORT": _postgres_port(
            environment.get(
                "POSTGRES_PORT",
                environment.get("DB_PORT", "5432"),
            ),
        ),
        **_postgres_common(environment),
    }


def build_database_config(environment, *, base_dir: Path) -> dict:
    """Return the Django default database config without opening a connection."""
    database_url = str(environment.get("DATABASE_URL", "")).strip()
    if database_url:
        return _postgres_from_url(database_url, environment)

    engine = str(environment.get("DATABASE_ENGINE", "sqlite")).strip().lower()
    if engine in SQLITE_ENGINES:
        sqlite_name = environment.get("SQLITE_NAME")
        sqlite_path = Path(sqlite_name) if sqlite_name else base_dir / "db.sqlite3"
        if not sqlite_path.is_absolute():
            sqlite_path = base_dir / sqlite_path
        return {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": sqlite_path,
        }
    if engine in POSTGRES_ENGINES:
        return _postgres_from_environment(environment)
    raise ImproperlyConfigured(
        "DATABASE_ENGINE must be one of: sqlite, sqlite3, postgres, postgresql."
    )
