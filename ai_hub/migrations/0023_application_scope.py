"""Introduce ApplicationScope, the Core root security boundary.

Staged deliberately, because the final schema requires a non-null owner on
three existing tables that already hold rows:

    1. create ApplicationScope
    2. add the three FKs NULLABLE
    3. create one deterministic legacy scope and assign every existing row
    4. tighten all three FKs to NOT NULL

Step 3 is the only data step. It is written against historical model states,
creates exactly one scope, and assigns rows in bulk.

The legacy scope exists for MIGRATION COMPATIBILITY ONLY. It is not a runtime
default: no model field defaults to it, nothing resolves it by name at runtime,
and new resources must supply a scope explicitly. A permanent fallback would be
a fail-open security mechanism wearing a migration's clothes.
"""

import django.db.models.deletion
from django.db import migrations, models


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
    ApplicationScope = apps.get_model("ai_hub", "ApplicationScope")

    orphans = sum(
        apps.get_model("ai_hub", name).objects.filter(
            application_scope__isnull=True
        ).count()
        for name in OWNED_MODELS
    )
    if not orphans:
        # A fresh database has nothing to adopt. Do NOT create the legacy scope
        # in that case: a new install should start with no scopes at all rather
        # than inherit a compatibility artifact it never needed.
        return

    scope, _created = ApplicationScope.objects.get_or_create(
        slug=LEGACY_SCOPE_SLUG,
        defaults={
            "name": LEGACY_SCOPE_NAME,
            "description": LEGACY_SCOPE_DESCRIPTION,
            "is_active": True,
        },
    )
    for name in OWNED_MODELS:
        apps.get_model("ai_hub", name).objects.filter(
            application_scope__isnull=True
        ).update(application_scope=scope)


def unassign_legacy_scope(apps, schema_editor):
    """Reverse step 3: detach rows and drop the legacy scope if it is empty.

    Reversible on purpose. The forward step adds an owner and creates one row;
    both are undoable without losing Knowledge, agents or workspaces. The scope
    is deleted only when nothing still points at it, so a scope an operator
    created by hand is never removed by a rollback.
    """
    ApplicationScope = apps.get_model("ai_hub", "ApplicationScope")
    for name in OWNED_MODELS:
        apps.get_model("ai_hub", name).objects.filter(
            application_scope__slug=LEGACY_SCOPE_SLUG
        ).update(application_scope=None)
    scope = ApplicationScope.objects.filter(slug=LEGACY_SCOPE_SLUG).first()
    if scope is not None and not any(
        apps.get_model("ai_hub", name).objects.filter(
            application_scope=scope
        ).exists()
        for name in OWNED_MODELS
    ):
        scope.delete()


class Migration(migrations.Migration):

    dependencies = [
        ("ai_hub", "0022_knowledge_lifecycle_event"),
    ]

    operations = [
        # -- 1. the root model ------------------------------------------------
        migrations.CreateModel(
            name="ApplicationScope",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=140, unique=True)),
                ("slug", models.SlugField(max_length=140, unique=True)),
                ("description", models.TextField(blank=True)),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "1.0 Application scope",
                "verbose_name_plural": "1.0 Application scopes - isolation boundaries",
                "ordering": ["name"],
            },
        ),
        # -- 2. nullable ownership, so existing rows survive the add ----------
        migrations.AddField(
            model_name="knowledgecollection",
            name="application_scope",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="knowledge_collections",
                to="ai_hub.applicationscope",
            ),
        ),
        migrations.AddField(
            model_name="agentprofile",
            name="application_scope",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="agent_profiles",
                to="ai_hub.applicationscope",
            ),
        ),
        migrations.AddField(
            model_name="gameworkspace",
            name="application_scope",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="game_workspaces",
                to="ai_hub.applicationscope",
            ),
        ),
        # -- 3. adopt every pre-existing row ----------------------------------
        migrations.RunPython(assign_legacy_scope, unassign_legacy_scope),
        # -- 4. ownership becomes mandatory -----------------------------------
        migrations.AlterField(
            model_name="knowledgecollection",
            name="application_scope",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.PROTECT,
                related_name="knowledge_collections",
                to="ai_hub.applicationscope",
            ),
        ),
        migrations.AlterField(
            model_name="agentprofile",
            name="application_scope",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.PROTECT,
                related_name="agent_profiles",
                to="ai_hub.applicationscope",
            ),
        ),
        migrations.AlterField(
            model_name="gameworkspace",
            name="application_scope",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.PROTECT,
                related_name="game_workspaces",
                to="ai_hub.applicationscope",
            ),
        ),
    ]
