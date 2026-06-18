from django.core.exceptions import ValidationError


def _required_keys(schema: dict) -> list[str]:
    return schema.get("required", []) if isinstance(schema, dict) else []


def _matches_type(value, expected_type: str) -> bool:
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    python_type = type_map.get(expected_type)
    if python_type is None:
        return True
    if expected_type == "integer" and isinstance(value, bool):
        return False
    if expected_type == "number" and isinstance(value, bool):
        return False
    return isinstance(value, python_type)


def validate_payload(payload: dict, schema: dict, label: str) -> None:
    if not isinstance(payload, dict):
        raise ValidationError(f"{label} payload must be a JSON object.")
    if not isinstance(schema, dict):
        raise ValidationError(f"{label} schema must be a JSON object.")

    missing = [key for key in _required_keys(schema) if key not in payload]
    if missing:
        raise ValidationError(f"{label} payload missing required keys: {', '.join(missing)}")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValidationError(f"{label} schema properties must be a JSON object.")
    for key, rules in properties.items():
        if key not in payload or not isinstance(rules, dict):
            continue
        expected_type = rules.get("type")
        if expected_type and not _matches_type(payload[key], expected_type):
            raise ValidationError(f"{label} payload key '{key}' must be of type '{expected_type}'.")
