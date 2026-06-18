import json
from django import template

register = template.Library()


@register.filter
def pretty_json(value):
    """Render a dict, list, or JSON string as indented JSON."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, default=str)
    try:
        parsed = json.loads(value)
        return json.dumps(parsed, indent=2)
    except Exception:
        return str(value)


@register.filter
def action_color(action):
    """Return Bootstrap colour name for a GAME action string."""
    mapping = {
        "finish": "success",
        "think": "info",
        "complete": "success",
        "final": "success",
    }
    return mapping.get(str(action).lower(), "secondary")


@register.filter
def status_color(status):
    """Return Bootstrap colour name for a session/step status string."""
    mapping = {
        "success": "success",
        "failed": "danger",
        "running": "warning",
        "pending": "secondary",
        "created": "light",
    }
    return mapping.get(str(status).lower(), "secondary")
