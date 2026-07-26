import os

from django.core.exceptions import ValidationError

from ai_hub.models import ModelConfig, ProviderConfig


def resolve_model_config(model_config: ModelConfig) -> dict:
    provider = model_config.provider
    if not provider.is_active or not model_config.is_active:
        raise ValidationError("Model/provider is inactive.")
    if provider.provider_type not in ProviderConfig.ProviderType.values:
        raise ValidationError(
            f"Unsupported provider type '{provider.provider_type}' "
            f"for provider '{provider.name}'."
        )

    api_key = ""
    # Training providers are deterministic stubs — no real API key is needed
    if provider.provider_type == ProviderConfig.ProviderType.TRAINING:
        return {
            "provider_name": provider.name,
            "provider_type": provider.provider_type,
            "base_url": "",
            "api_key": "",
            "model": model_config.model_name,
            "timeout": provider.default_timeout,
            "temperature": float(model_config.temperature_default),
            "max_tokens": model_config.max_tokens_default,
            "supports_tools": model_config.supports_tools,
        }
    if provider.api_key_env_var:
        api_key = os.getenv(provider.api_key_env_var, "")
        if not api_key:
            raise ValidationError(
                f"Missing environment variable '{provider.api_key_env_var}' for provider '{provider.name}'."
            )

    return {
        "provider_name": provider.name,
        "provider_type": provider.provider_type,
        "base_url": provider.base_url,
        "api_key": api_key,
        "model": model_config.model_name,
        "timeout": provider.default_timeout,
        "temperature": float(model_config.temperature_default),
        "max_tokens": model_config.max_tokens_default,
        "supports_tools": model_config.supports_tools,
    }
