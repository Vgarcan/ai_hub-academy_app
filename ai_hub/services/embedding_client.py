"""HTTP transport for embedding providers. One provider, and nothing else.

This module owns **HTTP only**. It does not decide policy, does not normalize,
does not persist and does not know what a Knowledge chunk is - it is handed a
string and a contract and returns numbers. Orchestration lives in
`embedding_execution.py`.

Deliberately separate from `litellm_client.py` / `provider_registry.py` /
`agent_runtime.py`: completion and embedding are different capabilities with
different contracts, and existing completion behaviour must not move because an
embedding path appeared.

**Transport support is a CAPABILITY axis, not a security axis.** S-20 implements
exactly one real transport, Ollama. A provider an operator has declared LOCAL may
still be refused here with `unsupported_embedding_transport` - that is a
statement about what this slice can execute, never an inference about where the
provider sits. Locality is decided by `ProviderConfig.declared_locality` (S-17)
and by nothing in this file: no URL parsing, no hostname check, no scheme check.

**No credentials.** The Ollama transport reads no `api_key_env_var`, no
environment secret and sends no Authorization header. This contract needs none.
An authenticated local gateway is separate future work.
"""

from dataclasses import dataclass

import requests

from ai_hub.models import ProviderConfig


class ErrorCategory:
    """Bounded, machine-readable transport failures.

    Deliberately coarse. A category never carries submitted content, vector
    values, credentials or a raw provider body - an embedding error must not
    become a way to read back the text that was sent.
    """

    INVALID_PROVIDER_CONFIGURATION = "invalid_provider_configuration"
    UNSUPPORTED_EMBEDDING_TRANSPORT = "unsupported_embedding_transport"
    PROVIDER_UNREACHABLE = "provider_unreachable"
    MODEL_NOT_FOUND = "model_not_found"
    PROVIDER_RETURNED_ERROR = "provider_returned_error"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"


class EmbeddingProviderExecutionError(RuntimeError):
    """A transport failure, carrying a bounded category and no content."""

    def __init__(self, category: str, message: str = ""):
        self.category = category
        super().__init__(message or category)


@dataclass(frozen=True)
class EmbeddingProviderResult:
    """What Core needs back, and nothing more.

    Deliberately excludes the raw response, the submitted input, any provider
    trace body and the HTTP objects. `provider_model` is retained for bounded
    diagnostics only - it must never rewrite `model_name`, `model_revision` or
    `e1`, which remain the operator-declared S-18 contract.
    """

    values: tuple
    provider_type: str
    provider_model: str


def _ollama_model_name(model_name: str) -> str:
    """`ollama/foo` -> `foo`, matching the existing compatibility convention.

    A request-time transformation only. The persisted `EmbeddingModelConfig` is
    never mutated, and this transformed string is NOT part of `e1`: the
    fingerprint uses the operator-declared contract, so how a transport spells a
    model name cannot change vector identity.
    """
    text = str(model_name or "")
    prefix = "ollama/"
    return text[len(prefix):] if text.startswith(prefix) else text


def _validate_values(payload, *, expected_dimension: int) -> tuple:
    """Structural validation of one returned vector. Never repairs it."""
    if not isinstance(payload, (list, tuple)):
        raise EmbeddingProviderExecutionError(
            ErrorCategory.INVALID_PROVIDER_RESPONSE,
            "Provider embedding is not a list.",
        )
    if len(payload) != expected_dimension:
        # Never truncate, never pad. A different dimension is a different
        # vector space, and silently reshaping it would produce a vector that
        # is mathematically meaningless while looking valid.
        raise EmbeddingProviderExecutionError(
            ErrorCategory.INVALID_PROVIDER_RESPONSE,
            f"Provider returned {len(payload)} components; "
            f"the contract requires {expected_dimension}.",
        )
    values = []
    for index, raw in enumerate(payload):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise EmbeddingProviderExecutionError(
                ErrorCategory.INVALID_PROVIDER_RESPONSE,
                f"Provider embedding component {index} is not a number.",
            )
        value = float(raw)
        if value != value or value in (float("inf"), float("-inf")):
            raise EmbeddingProviderExecutionError(
                ErrorCategory.INVALID_PROVIDER_RESPONSE,
                f"Provider embedding component {index} is not finite.",
            )
        values.append(value)
    return tuple(values)


def embed_text_via_ollama(*, provider, contract, text: str) -> EmbeddingProviderResult:
    """POST one string to Ollama's `/api/embed` and return validated numbers.

    `truncate=false` is MANDATORY and load-bearing. Ollama truncates by default,
    which would mean the provider silently embedded something other than the text
    `k1` fingerprints - a vector that claims to represent content it never saw.
    AI Hub enforces its own `max_input_chars` before dispatch AND tells the
    provider not to shorten anything behind our back.
    """
    base_url = str(getattr(provider, "base_url", "") or "").strip()
    if not base_url:
        # A local provider is operator-configured. Never invent
        # localhost:11434 or any other default endpoint.
        raise EmbeddingProviderExecutionError(
            ErrorCategory.INVALID_PROVIDER_CONFIGURATION,
            "The provider has no configured base URL.",
        )

    url = f"{base_url.rstrip('/')}/api/embed"
    body = {
        "model": _ollama_model_name(contract.model_name),
        "input": text,
        "truncate": False,
        "dimensions": contract.vector_dimension,
    }

    try:
        response = requests.post(
            url, json=body, timeout=contract.request_timeout_seconds
        )
    except requests.Timeout as exc:
        raise EmbeddingProviderExecutionError(
            ErrorCategory.PROVIDER_UNREACHABLE, "The provider timed out."
        ) from exc
    except requests.RequestException as exc:
        raise EmbeddingProviderExecutionError(
            ErrorCategory.PROVIDER_UNREACHABLE, "The provider is unreachable."
        ) from exc

    status = getattr(response, "status_code", None)
    if status == 404:
        raise EmbeddingProviderExecutionError(
            ErrorCategory.MODEL_NOT_FOUND,
            "The provider does not have the configured model.",
        )
    if status is None or status >= 400:
        # Deliberately status-only: the provider body may echo the submitted
        # text, so it is never surfaced.
        raise EmbeddingProviderExecutionError(
            ErrorCategory.PROVIDER_RETURNED_ERROR,
            f"The provider returned HTTP {status}.",
        )

    try:
        document = response.json()
    except Exception as exc:  # noqa: BLE001 - any unparseable body is one failure
        raise EmbeddingProviderExecutionError(
            ErrorCategory.INVALID_PROVIDER_RESPONSE,
            "The provider response was not valid JSON.",
        ) from exc

    if not isinstance(document, dict):
        raise EmbeddingProviderExecutionError(
            ErrorCategory.INVALID_PROVIDER_RESPONSE,
            "The provider response was not a JSON object.",
        )

    embeddings = document.get("embeddings")
    if not isinstance(embeddings, (list, tuple)):
        raise EmbeddingProviderExecutionError(
            ErrorCategory.INVALID_PROVIDER_RESPONSE,
            "The provider response has no embeddings array.",
        )
    if len(embeddings) != 1:
        # Exactly one input was submitted, so exactly one embedding is the only
        # coherent answer. Picking the first of several would guess which vector
        # belongs to our text.
        raise EmbeddingProviderExecutionError(
            ErrorCategory.INVALID_PROVIDER_RESPONSE,
            f"The provider returned {len(embeddings)} embeddings for one input.",
        )

    values = _validate_values(
        embeddings[0], expected_dimension=contract.vector_dimension
    )
    return EmbeddingProviderResult(
        values=values,
        provider_type=provider.provider_type,
        # Diagnostics only. Never vector identity.
        provider_model=str(document.get("model") or "")[:140],
    )


#: Transport by provider type. S-20 ships exactly one, deliberately: one
#: production-shaped local path proven end to end before adapters multiply.
EMBEDDING_TRANSPORTS = {
    ProviderConfig.ProviderType.OLLAMA: embed_text_via_ollama,
}


def resolve_embedding_transport(provider):
    """The transport for this provider type, or refuse.

    A capability lookup keyed on `provider_type`. It never consults
    `declared_locality` - choosing HOW to talk to a provider and deciding
    WHETHER we are allowed to are separate questions, and collapsing them is how
    a URL ends up deciding a security outcome.
    """
    transport = EMBEDDING_TRANSPORTS.get(getattr(provider, "provider_type", None))
    if transport is None:
        raise EmbeddingProviderExecutionError(
            ErrorCategory.UNSUPPORTED_EMBEDDING_TRANSPORT,
            "No embedding transport is implemented for this provider type.",
        )
    return transport
