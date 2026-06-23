from django.conf import settings
from django.core.exceptions import ValidationError

_FLAG_DEFAULTS = {
    "AI_HUB_GAME_GOALS_ENABLED": False,
    "AI_HUB_GAME_SCHEDULER_ENABLED": False,
    "AI_HUB_GAME_ACTION_DISPATCH_ENABLED": False,
    "AI_HUB_GAME_MEMORY_ENABLED": False,
    "AI_HUB_GAME_RESUME_ENABLED": False,
    "AI_HUB_GAME_DELEGATION_ENABLED": False,
    "AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED": False,
}


def is_game_feature_enabled(flag_name: str) -> bool:
    if flag_name not in _FLAG_DEFAULTS:
        raise ValueError(
            f"Unknown GAME feature flag: {flag_name!r}. "
            f"Known flags: {sorted(_FLAG_DEFAULTS)}."
        )
    return bool(getattr(settings, flag_name, _FLAG_DEFAULTS[flag_name]))


def require_game_feature(flag_name: str) -> None:
    """Raise ValidationError when the named GAME feature flag is explicitly disabled.

    Raises ValueError on an unknown flag name so a typo can never silently
    disable the gate (which would leave the feature unprotected).
    """
    if not is_game_feature_enabled(flag_name):
        raise ValidationError(
            f"GAME feature '{flag_name}' is disabled. "
            f"Set {flag_name}=True in Django settings to enable it."
        )
