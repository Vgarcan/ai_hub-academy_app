from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlparse

from django.core.exceptions import ValidationError


HTTP_READ_METHODS = frozenset({"GET", "HEAD"})
HTTP_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
HTTP_SUPPORTED_METHODS = HTTP_READ_METHODS | HTTP_WRITE_METHODS
HTTP_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
DEFAULT_HTTP_MAX_REDIRECTS = 5
MAX_HTTP_REDIRECTS = 10
DEFAULT_HTTP_MAX_RESPONSE_BYTES = 1024 * 1024
MIN_HTTP_RESPONSE_BYTES = 1024
MAX_HTTP_RESPONSE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class HttpToolConfiguration:
    url: str
    method: str
    allowed_hosts: frozenset[str]
    headers: dict
    timeout: int
    max_redirects: int
    max_response_bytes: int


def normalize_http_hostname(hostname: str) -> str:
    raw = str(hostname or "").strip().lower().rstrip(".")
    if not raw:
        raise ValidationError("HTTP allowed_hosts entries must be non-empty hostnames.")
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    try:
        return ip_address(raw).compressed
    except ValueError:
        pass
    if any(marker in raw for marker in ("://", "/", "?", "#", "@", ":")):
        raise ValidationError(
            "HTTP allowed_hosts entries must contain hostnames only, without scheme, path, or port."
        )
    try:
        return raw.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValidationError(f"Invalid HTTP hostname '{hostname}'.") from exc


def validate_http_destination(url: str, allowed_hosts: frozenset[str], *, tool_name: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValidationError(
            f"Tool '{tool_name}' HTTP URL scheme must be http or https."
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValidationError(
            f"Tool '{tool_name}' HTTP URL must not contain embedded credentials."
        )
    try:
        hostname = normalize_http_hostname(parsed.hostname or "")
        parsed.port
    except ValueError as exc:
        raise ValidationError(f"Tool '{tool_name}' has an invalid HTTP URL.") from exc
    if hostname not in allowed_hosts:
        raise ValidationError(f"Tool '{tool_name}' HTTP host is not explicitly allowed.")


def build_http_tool_configuration(tool) -> HttpToolConfiguration:
    config = tool.config or {}
    if not isinstance(config, dict):
        raise ValidationError("HTTP Tool config must be a JSON object.")

    method = str(config.get("method", "POST") or "").strip().upper()
    if method not in HTTP_SUPPORTED_METHODS:
        supported = ", ".join(sorted(HTTP_SUPPORTED_METHODS))
        raise ValidationError(
            f"Tool '{tool.name}' HTTP method must be one of: {supported}."
        )
    if tool.operation_mode == "read" and method not in HTTP_READ_METHODS:
        raise ValidationError(
            "HTTP tools with operation_mode READ must use GET or HEAD."
        )

    url = str(config.get("url", "") or "").strip()
    if not url:
        raise ValidationError(f"Tool '{tool.name}' is missing 'url' in config.")

    raw_allowed_hosts = config.get("allowed_hosts")
    if not isinstance(raw_allowed_hosts, list) or not raw_allowed_hosts:
        raise ValidationError(
            f"Tool '{tool.name}' HTTP allowed_hosts must be a non-empty list."
        )
    allowed_hosts = frozenset(
        normalize_http_hostname(hostname) for hostname in raw_allowed_hosts
    )
    validate_http_destination(url, allowed_hosts, tool_name=tool.name)

    headers = config.get("headers", {})
    if not isinstance(headers, dict):
        raise ValidationError(f"Tool '{tool.name}' HTTP headers must be a JSON object.")

    try:
        timeout = int(config.get("timeout", 30))
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Tool '{tool.name}' HTTP timeout must be an integer."
        ) from exc
    timeout = min(max(timeout, 1), 60)

    raw_max_redirects = config.get("max_redirects", DEFAULT_HTTP_MAX_REDIRECTS)
    if isinstance(raw_max_redirects, bool):
        raise ValidationError(
            f"Tool '{tool.name}' HTTP max_redirects must be an integer from 0 to "
            f"{MAX_HTTP_REDIRECTS}."
        )
    try:
        max_redirects = int(raw_max_redirects)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Tool '{tool.name}' HTTP max_redirects must be an integer from 0 to "
            f"{MAX_HTTP_REDIRECTS}."
        ) from exc
    if not 0 <= max_redirects <= MAX_HTTP_REDIRECTS:
        raise ValidationError(
            f"Tool '{tool.name}' HTTP max_redirects must be an integer from 0 to "
            f"{MAX_HTTP_REDIRECTS}."
        )

    raw_max_response_bytes = config.get(
        "max_response_bytes",
        DEFAULT_HTTP_MAX_RESPONSE_BYTES,
    )
    if isinstance(raw_max_response_bytes, bool):
        raise ValidationError(
            f"Tool '{tool.name}' HTTP max_response_bytes must be an integer."
        )
    try:
        max_response_bytes = int(raw_max_response_bytes)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Tool '{tool.name}' HTTP max_response_bytes must be an integer."
        ) from exc
    max_response_bytes = min(
        max(max_response_bytes, MIN_HTTP_RESPONSE_BYTES),
        MAX_HTTP_RESPONSE_BYTES,
    )

    return HttpToolConfiguration(
        url=url,
        method=method,
        allowed_hosts=allowed_hosts,
        headers=dict(headers),
        timeout=timeout,
        max_redirects=max_redirects,
        max_response_bytes=max_response_bytes,
    )
