"""Stage 2 of 3 — DATA ONLY. Adopt every pre-existing row into one legacy scope.

No schema operation belongs in this file. It sits alone in its own transaction
precisely so that its writes never share one with a DDL statement — the defect
PostgreSQL surfaced as "pending trigger events" when schema and data were
interleaved inside a single atomic migration.

The legacy scope exists for MIGRATION COMPATIBILITY ONLY. It is not a runtime
default: no model field defaults to it, nothing resolves it by name at runtime,
and new resources must supply a scope explicitly. A permanent fallback would be
a fail-open security mechanism wearing a migration's clothes.

Constants and functions are defined HERE rather than imported from 0023.
Migration-time logic must not be shared between migration files: an import
couples two files whose historical model states are different, and a later edit
to one silently changes the other's behaviour.
"""

from django.db import migrations


LEGACY_SCOPE_NAME = "Legacy default"
LEGACY_SCOPE_SLUG = "legacy-default"
LEGACY_SCOPE_DESCRIPTION = (
    "Migration-compatibility scope holding every resource that existed before "
    "application scopes were introduced. Not a runtime default: new resources "
    "must be assigned a scope explicitly."
)

OWNED_MODELS = ("KnowledgeCollection", "AgentProfile", "GameWorkspace")


def assign_legacy_scope(apps, schema_editor):
    """Create the single legacy scope and adopt every pre-existing row."""
    # Never assume "default": run against the database Django is actually
    # migrating. Every query below is pinned to this alias.
    db_alias = schema_editor.connection.alias
    ApplicationScope = apps.get_model("ai_hub", "ApplicationScope")

    orphans = sum(
        apps.get_model("ai_hub", name)
        .objects.using(db_alias)
        .filter(application_scope__isnull=True)
        .count()
        for name in OWNED_MODELS
    )
    if not orphans:
        # A fresh database has nothing to adopt. Do NOT create the legacy scope
        # in that case: a new install should start with no scopes at all rather
        # than inherit a compatibility artifact it never needed.
        return

    scope, _created = ApplicationScope.objects.using(db_alias).get_or_create(
        slug=LEGACY_SCOPE_SLUG,
        defaults={
            "name": LEGACY_SCOPE_NAME,
            "description": LEGACY_SCOPE_DESCRIPTION,
            "is_active": True,
        },
    )
    for name in OWNED_MODELS:
        apps.get_model("ai_hub", name).objects.using(db_alias).filter(
            application_scope__isnull=True
        ).update(application_scope=scope)


def unassign_legacy_scope(apps, schema_editor):
    """Reverse: detach rows and drop the legacy scope if it is empty.

    Reversible on purpose. The forward step adds an owner and creates one row;
    both are undoable without losing Knowledge, agents or workspaces. The scope
    is deleted only when nothing still points at it, so a scope an operator
    created by hand is never removed by a rollback.
    """
    db_alias = schema_editor.connection.alias
    ApplicationScope = apps.get_model("ai_hub", "ApplicationScope")

    for name in OWNED_MODELS:
        apps.get_model("ai_hub", name).objects.using(db_alias).filter(
            application_scope__slug=LEGACY_SCOPE_SLUG
        ).update(application_scope=None)

    scope = (
        ApplicationScope.objects.using(db_alias)
        .filter(slug=LEGACY_SCOPE_SLUG)
        .first()
    )
    if scope is not None and not any(
        apps.get_model("ai_hub", name)
        .objects.using(db_alias)
        .filter(application_scope=scope)
        .exists()
        for name in OWNED_MODELS
    ):
        scope.delete(using=db_alias)


class Migration(migrations.Migration):

    dependencies = [
        ("ai_hub", "0023_application_scope"),
    ]

    operations = [
        migrations.RunPython(assign_legacy_scope, unassign_legacy_scope),
    ]
