import json

from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _


JSON_CONTAINER_LABELS = {
    dict: _("object"),
    list: _("array"),
}

JSON_CONTAINER_NAMES = {
    dict: "object",
    list: "array",
}


def expected_json_container(model_field):
    """Return the root container implied by a JSONField default, when explicit."""

    default = model_field.default
    if default is dict or default is list:
        return default
    if isinstance(default, dict):
        return dict
    if isinstance(default, list):
        return list
    return None


class SafeAdminJSONField(forms.JSONField):
    """JSON form field with an optional object/array root contract."""

    default_error_messages = {
        **forms.JSONField.default_error_messages,
        "invalid_root": _("Enter a JSON %(expected)s at the top level."),
    }

    def __init__(self, *args, expected_type=None, **kwargs):
        self.expected_type = expected_type
        super().__init__(*args, **kwargs)

    def clean(self, value):
        cleaned = super().clean(value)

        if cleaned is None and not self.required and self.expected_type:
            return self.expected_type()

        if self.expected_type and not isinstance(cleaned, self.expected_type):
            raise ValidationError(
                self.error_messages["invalid_root"],
                code="invalid_root",
                params={"expected": JSON_CONTAINER_LABELS[self.expected_type]},
            )
        return cleaned


class SafeAdminJSONWidget(forms.Textarea):
    """Progressively enhanced textarea; server-side validation remains authoritative."""

    class Media:
        css = {"all": ("ai_hub/CSS/json-editor.css",)}
        js = ("ai_hub/JS/json-editor.js",)

    def __init__(self, attrs=None, *, expected_type=None):
        expected_name = JSON_CONTAINER_NAMES.get(expected_type, "")
        attrs = dict(attrs or {})
        extra_classes = attrs.pop("class", "")
        defaults = {
            "class": " ".join(
                value
                for value in ("vLargeTextField ai-hub-json-editor", extra_classes)
                if value
            ),
            "data-ai-json-editor": "true",
            "data-json-root": expected_name,
            "rows": 10,
            "spellcheck": "false",
        }
        defaults.update(attrs)
        super().__init__(defaults)

    def format_value(self, value):
        value = super().format_value(value)
        if value in (None, ""):
            return value

        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except (TypeError, ValueError):
            return value

        try:
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return value
