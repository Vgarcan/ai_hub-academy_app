"""Stage 1 of 3 — SCHEMA ONLY. Introduce ApplicationScope and nullable ownership.

The three stages of the application-scope rollout are deliberately three
migrations, not one:

    0023  schema   create the model, add the three FKs NULLABLE
    0024  data     adopt every pre-existing row into one legacy scope
    0025  schema   tighten the three FKs to NOT NULL

They were originally a single atomic migration that interleaved
DDL -> RunPython -> DDL. That is exactly the pattern Django warns against, and
PostgreSQL proved it: reversing it failed with

    cannot ALTER TABLE "ai_hub_applicationscope" because it has pending
    trigger events

because the data step's writes leave deferred FK trigger events queued inside
the same transaction that then tries to alter the table. SQLite never noticed.

Splitting the stages gives each migration its own transaction, so no schema
change ever shares a transaction with a data change. This is a portability fix,
not a suppression: no `atomic = False`, no manual commits, no
`SET CONSTRAINTS`, no backend-specific DDL, and ownership is still mandatory in
the final schema.
"""

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ai_hub", "0022_knowledge_lifecycle_event"),
    ]

    operations = [
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
        # Nullable for now, so existing rows survive the column being added.
        # 0025 makes ownership mandatory, after 0024 has given every row an owner.
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
    ]
