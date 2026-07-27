"""Shared credential-name classification for HTTP and persisted audit boundaries."""

import re
import unicodedata


_NON_ASCII_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_CREDENTIAL_NAME_MARKERS = (
    "authorization",
    "cookie",
    "apikey",
    "secret",
    "password",
    "token",
    "credentials",
    "privatekey",
)


def normalize_credential_name(name) -> str:
    """Return a case- and separator-insensitive canonical field/header name."""
    normalized = unicodedata.normalize("NFKC", str(name or "")).casefold()
    return _NON_ASCII_ALNUM_RE.sub("", normalized)


def is_sensitive_credential_name(name) -> bool:
    """Classify common credential-bearing field and HTTP header name variants."""
    normalized = normalize_credential_name(name)
    return bool(normalized) and any(
        marker in normalized
        for marker in _CREDENTIAL_NAME_MARKERS
    )
