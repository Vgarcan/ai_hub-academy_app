import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ai_hub", "0009_phase08_pause_approval_resume"),
    ]

    operations = [
        migrations.CreateModel(
            name="GameWorkspaceAction",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                (
                    "workspace",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="workspace_actions",
                        to="ai_hub.gameworkspace",
                    ),
                ),
                (
                    "action",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="workspace_entries",
                        to="ai_hub.gameactiondefinition",
                    ),
                ),
                ("is_enabled", models.BooleanField(default=True)),
                ("requires_approval_override", models.BooleanField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "verbose_name": "4.13 GAME workspace action",
                "verbose_name_plural": "4.13 GAME workspace actions - allow-list",
                "ordering": ["workspace__name", "action__name"],
            },
        ),
        migrations.AddConstraint(
            model_name="gameworkspaceaction",
            constraint=models.UniqueConstraint(
                fields=["workspace", "action"], name="ai_hub_unique_ws_action"
            ),
        ),
        migrations.CreateModel(
            name="GameWorkspaceAgent",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                (
                    "workspace",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="workspace_agents",
                        to="ai_hub.gameworkspace",
                    ),
                ),
                (
                    "agent",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="workspace_entries",
                        to="ai_hub.agentprofile",
                    ),
                ),
                ("is_enabled", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "verbose_name": "4.14 GAME workspace agent",
                "verbose_name_plural": "4.14 GAME workspace agents - allow-list",
                "ordering": ["workspace__name", "agent__name"],
            },
        ),
        migrations.AddConstraint(
            model_name="gameworkspaceagent",
            constraint=models.UniqueConstraint(
                fields=["workspace", "agent"], name="ai_hub_unique_ws_agent"
            ),
        ),
    ]
