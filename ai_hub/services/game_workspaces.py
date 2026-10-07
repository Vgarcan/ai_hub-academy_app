from django.db import transaction

from ai_hub.models import GameWorkspace


@transaction.atomic
def create_workspace(
    *,
    application_scope,
    name: str,
    description: str = "",
    is_active: bool = True,
    default_policy: dict | None = None,
    default_runtime_config: dict | None = None,
) -> GameWorkspace:
    # Ownership is required and has no default. A workspace whose application
    # scope nobody chose is a workspace whose security boundary nobody chose.
    workspace = GameWorkspace(
        application_scope=application_scope,
        name=name,
        description=description,
        is_active=is_active,
        default_policy=default_policy or {},
        default_runtime_config=default_runtime_config or {},
    )
    workspace.full_clean()
    workspace.save()
    return workspace
