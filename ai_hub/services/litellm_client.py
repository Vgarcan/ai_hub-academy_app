import json

try:
    import litellm
except Exception:  # pragma: no cover
    litellm = None

import requests


def _training_completion_call(*, model: str, messages: list, **kwargs) -> dict:
    """
    Deterministic stub for the 'training' provider.

    Inspects the system prompt to decide what kind of response to return
    so demos and tests can run without any external API.
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


def _is_ollama_call(model: str, base_url: str) -> bool:
    return model.startswith("ollama/") or "11434" in (base_url or "")


def _ollama_chat_call(*, model: str, messages: list, base_url: str, timeout: int, temperature: float, max_tokens: int) -> dict:
    endpoint = f"{(base_url or 'http://localhost:11434').rstrip('/')}/api/chat"
    payload = {
        "model": _ollama_model_name(model),
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return {
        "status": "ok",
        "model": model,
        "content": (data.get("message") or {}).get("content", ""),
        "raw": data,
    }


def completion_call(*, model: str, messages: list, api_key: str = "", base_url: str = "", timeout: int = 60, temperature: float = 0.7, max_tokens: int = 1000) -> dict:
    # Training provider: return a deterministic stub without any API call
    if model.startswith("training/") or model == "training":
        return _training_completion_call(model=model, messages=messages)

    if _is_ollama_call(model, base_url):
        return _ollama_chat_call(
            model=model,
            messages=messages,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if litellm is None:
        return {
            "status": "stubbed",
            "model": model,
            "message": "LiteLLM not installed; returning stub response.",
            "content": "",
        }

    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key or None,
        base_url=base_url or None,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = ""
    if getattr(response, "choices", None):
        content = response.choices[0].message.content or ""

    return {
        "status": "ok",
        "model": model,
        "content": content,
        "raw": response.model_dump() if hasattr(response, "model_dump") else str(response),
    }
