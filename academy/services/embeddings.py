"""
Embedding utilities for semantic documentation search.

Calls the Ollama /api/embed endpoint using the first active Ollama provider.
Falls back gracefully to None when the model or provider is unavailable.
"""
import json
import math
import urllib.error
import urllib.request
from typing import Optional


def get_embedding(text: str, model: str = "bge-m3:latest") -> Optional[list]:
    """
    Return an embedding vector for `text` using `model` on the first active
    Ollama provider. Returns None if unavailable.
    """
    from ai_hub.models import ProviderConfig

    provider = (
        ProviderConfig.objects.filter(
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            is_active=True,
        )
        .exclude(base_url="")
        .first()
    )
    if not provider:
        return None

    import logging
    logging.getLogger(__name__).debug("embed_docs using provider '%s' (%s)", provider.name, provider.base_url)
    base = provider.base_url.rstrip("/")
    return _call_embed(base, model, text)


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors. Uses numpy if available."""
    try:
        import numpy as np
        va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / denom) if denom else 0.0
    except ImportError:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        denom = mag_a * mag_b
        return dot / denom if denom else 0.0


# ── internal helpers ─────────────────────────────────────

def _call_embed(base_url: str, model: str, text: str) -> Optional[list]:
    """Try new Ollama /api/embed, fall back to legacy /api/embeddings."""
    # New API (Ollama ≥ 0.3)
    try:
        payload = json.dumps({"model": model, "input": text}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            embeddings = data.get("embeddings", [])
            if embeddings:
                first = embeddings[0]
                # Handle both [[...]] (nested) and [...] (flat) response shapes
                return first if isinstance(first, list) else embeddings
    except Exception:
        pass

    # Legacy API (Ollama < 0.3)
    try:
        payload = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("embedding")
    except Exception:
        pass

    return None
