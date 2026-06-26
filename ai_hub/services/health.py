"""Reusable runtime-health evaluator for the core configuration entities (P1).

Health logic for providers, models and agents was previously duplicated across
the composed change-page overviews and the Control Center context. This module is
the single source of truth: each ``evaluate_*`` returns a :class:`HealthResult`
with a coarse ``status`` and a list of human-readable :class:`HealthCheck`s.

These checks are **config-level only** — they never make network calls. The live
provider probe (Ollama ``/api/tags``) stays in ``admin_control_center`` and is
layered on top of this when a live result is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ai_hub.models import ProviderConfig

# Status values, ordered worst → best for easy aggregation.
STATUS_INACTIVE = "inactive"
STATUS_WARNING = "warning"
STATUS_OK = "ok"


@dataclass(frozen=True)
class HealthCheck:
    label: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class HealthResult:
    status: str
    checks: list[HealthCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def failing(self) -> list[HealthCheck]:
        return [c for c in self.checks if not c.ok]


def _result(is_active: bool, checks: list[HealthCheck]) -> HealthResult:
    """Derive a status: inactive if turned off, ok if every check passes, else warning."""
    if not is_active:
        return HealthResult(STATUS_INACTIVE, checks)
    if all(c.ok for c in checks):
        return HealthResult(STATUS_OK, checks)
    return HealthResult(STATUS_WARNING, checks)


def evaluate_provider(provider) -> HealthResult:
    """Config health for a provider: active, has models, an active model, credentials."""
    model_qs = provider.models.all()
    has_models = model_qs.exists()
    has_active_model = model_qs.filter(is_active=True).exists()
    is_ollama = provider.provider_type == ProviderConfig.ProviderType.OLLAMA
    if is_ollama:
        creds_label, creds_ok = "Base URL set", bool(provider.base_url)
    else:
        creds_label, creds_ok = "Credentials configured", bool(provider.api_key_env_var)
    checks = [
        HealthCheck("Provider active", provider.is_active),
        HealthCheck("Has models", has_models),
        HealthCheck("Active model available", has_active_model),
        HealthCheck(creds_label, creds_ok),
    ]
    return _result(provider.is_active, checks)


def evaluate_model(model) -> HealthResult:
    """Config health for a model: active, and its provider is active."""
    provider_active = bool(model.provider and model.provider.is_active)
    checks = [
        HealthCheck("Model active", model.is_active),
        HealthCheck("Provider active", provider_active),
    ]
    # An enabled model whose provider is off is a warning, not "inactive".
    return _result(model.is_active, checks)


def evaluate_agent(agent) -> HealthResult:
    """Config health for an agent: model/provider active, prompt set, contracts set."""
    model = agent.model_config
    model_active = bool(model and model.is_active)
    provider_active = bool(model and model.provider and model.provider.is_active)
    checks = [
        HealthCheck("Model active", model_active),
        HealthCheck("Provider active", provider_active),
        HealthCheck("Agent active", agent.is_active),
        HealthCheck("Prompt set", bool((agent.system_prompt or "").strip())),
        HealthCheck("Contracts set", bool(agent.input_contract) and bool(agent.output_contract)),
    ]
    return _result(agent.is_active, checks)
