from django.db import transaction

from ai_hub.models import GameWorkspace


@transaction.atomic
def create_workspace(
    *,
    name: str,
    description: str = "",
    is_active: bool = True,
    default_policy: dict | None = None,
    default_runtime_config: dict | None = None,
) -> GameWorkspace:
    workspace = GameWorkspace(
        name=name,
        description=description,
        is_active=is_active,
        default_policy=default_policy or {},
        default_runtime_config=default_runtime_config or {},
    )
    workspace.full_clean()
    workspace.save()
    return workspace
