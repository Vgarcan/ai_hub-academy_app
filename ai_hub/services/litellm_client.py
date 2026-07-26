import json

try:
    import litellm
except Exception:  # pragma: no cover
    litellm = None

import requests

from ai_hub.models import ProviderConfig


class ProviderExecutionError(RuntimeError):
    """A provider-boundary failure with a stable, non-secret category."""

    def __init__(self, category: str, message: str):
        self.category = category
        super().__init__(f"{category}: {message}")


def _training_completion_call(*, model: str, messages: list, **kwargs) -> dict:
    """
    Deterministic stub for the 'training' provider.

    Inspects the system prompt to decide what kind of response to return
    so demos and tests can run without any external API. The 'normaliz'
    branch is checked before classification so that a normalizer agent
    whose prompt also mentions "ticket" still returns normalized_* keys.
    """
    system_content = ""
    user_content = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_content = str(msg.get("content", "")).lower()
        elif msg.get("role") == "user":
            user_content = str(msg.get("content", ""))

    # GAME sessions expect a JSON object with action/complete/final_answer
    if "action" in system_content or "game" in system_content or "complete" in system_content:
        content = json.dumps({
            "action": "finish",
            "message": "Training model: goal complete.",
            "complete": True,
            "final_answer": "This is a training stub response. Configure a real provider for production use.",
        })
        return {"status": "ok", "model": model, "content": content, "provider": "training", "stubbed": True}

    # Input normalization expects normalized_title/normalized_body. Checked
    # before classification because a normalizer prompt commonly says "ticket".
    if "normaliz" in system_content:
        content = json.dumps({
            "normalized_title": "Normalized ticket title (training stub).",
            "normalized_body": "Normalized ticket body with HTML stripped and whitespace trimmed (training stub).",
            "word_count": 12,
        })
        return {"status": "ok", "model": model, "content": content, "provider": "training", "stubbed": True}

    # Ticket classification expects category/priority/reason
    if "classif" in system_content or "ticket" in system_content or "triage" in system_content:
        content = json.dumps({
            "category": "Technical Issue",
            "priority": "Medium",
            "reason": "Training stub: ticket text analysed with deterministic response.",
        })
        return {"status": "ok", "model": model, "content": content, "provider": "training", "stubbed": True}

    # Documentation assistant
    if "document" in system_content or "assistant" in system_content:
        content = (
            "This is a training stub response from the documentation assistant. "
            "Configure a real provider to get AI-generated answers."
        )
        return {"status": "ok", "model": model, "content": content, "provider": "training", "stubbed": True}

    # Generic fallback
    content = "Training model stub response. No real API call was made."
    return {"status": "ok", "model": model, "content": content, "provider": "training", "stubbed": True}


def _ollama_model_name(model: str) -> str:
    return model.removeprefix("ollama/")


def _ollama_error_detail(response) -> str:
    try:
        data = response.json()
    except (TypeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("error") or "")[:300]


def _ollama_chat_call(
    *,
    model: str,
    messages: list,
    base_url: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> dict:
    if not (base_url or "").strip():
        raise ProviderExecutionError(
            "invalid_provider_configuration",
            "Ollama provider requires an explicit base URL.",
        )

    endpoint = f"{base_url.rstrip('/')}/api/chat"
    provider_model = _ollama_model_name(model)
    payload = {
        "model": provider_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ProviderExecutionError(
            "provider_unreachable",
            "Ollama is unreachable at the configured base URL.",
        ) from exc
    except requests.RequestException as exc:
        raise ProviderExecutionError(
            "provider_unreachable",
            "Ollama request could not reach the configured base URL.",
        ) from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = _ollama_error_detail(response)
        if (
            response.status_code == 404
            and "model" in detail.lower()
            and "not found" in detail.lower()
        ):
            raise ProviderExecutionError(
                "model_not_found",
                f"Ollama model '{provider_model}' was not found.",
            ) from exc
        raise ProviderExecutionError(
            "provider_returned_error",
            f"Ollama returned HTTP {response.status_code}.",
        ) from exc

    try:
        data = response.json()
    except (TypeError, ValueError) as exc:
        raise ProviderExecutionError(
            "invalid_provider_response",
            "Ollama returned a response that was not valid JSON.",
        ) from exc

    message = data.get("message") if isinstance(data, dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ProviderExecutionError(
            "invalid_provider_response",
            "Ollama response did not contain message.content.",
        )

    return {
        "status": "ok",
        "model": model,
        "provider": ProviderConfig.ProviderType.OLLAMA,
        "provider_model": provider_model,
        "content": message["content"],
        "raw": data,
    }


def completion_call(
    *,
    provider_type: str,
    model: str,
    messages: list,
    api_key: str = "",
    base_url: str = "",
    timeout: int = 60,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> dict:
    if provider_type not in ProviderConfig.ProviderType.values:
        raise ProviderExecutionError(
            "invalid_provider_configuration",
            f"Unsupported provider type '{provider_type}'.",
        )

    if provider_type == ProviderConfig.ProviderType.TRAINING:
        return _training_completion_call(model=model, messages=messages)

    if provider_type == ProviderConfig.ProviderType.OLLAMA:
        return _ollama_chat_call(
            model=model,
            messages=messages,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if litellm is None:
        raise ProviderExecutionError(
            "invalid_provider_configuration",
            "LiteLLM is required for this provider but is not installed.",
        )

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key or None,
            base_url=base_url or None,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise ProviderExecutionError(
            "provider_returned_error",
            f"Provider '{provider_type}' request failed ({type(exc).__name__}).",
        ) from exc

    choices = getattr(response, "choices", None)
    content = getattr(getattr(choices[0], "message", None), "content", None) if choices else None
    if not isinstance(content, str):
        raise ProviderExecutionError(
            "invalid_provider_response",
            f"Provider '{provider_type}' response did not contain text content.",
        )

    return {
        "status": "ok",
        "model": model,
        "provider": provider_type,
        "content": content,
        "raw": response.model_dump() if hasattr(response, "model_dump") else str(response),
    }
