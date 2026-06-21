import decimal
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ai_hub", "0007_phase06_action_model_refactor"),
    ]

    operations = [
        migrations.CreateModel(
            name="GameMemoryEntry",
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
                (
                    "workspace",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="memory_entries",
                        to="ai_hub.gameworkspace",
                    ),
                ),
                (
                    "goal",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="memory_entries",
                        to="ai_hub.gamegoal",
                    ),
                ),
                (
                    "session",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="memory_entries",
                        to="ai_hub.executionsession",
                    ),
                ),
                (
                    "scope_type",
                    models.CharField(
                        choices=[
                            ("workspace", "Workspace"),
                            ("goal", "Goal"),
                            ("session", "Session"),
                            ("action_result", "Action result"),
                        ],
                        max_length=20,
                    ),
                ),
                ("content", models.TextField()),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "importance_score",
                    models.DecimalField(
                        decimal_places=2,
                        default=decimal.Decimal("0.50"),
                        max_digits=4,
                    ),
                ),
                ("expires_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "verbose_name": "4.10 GAME memory entry",
                "verbose_name_plural": "4.10 GAME memory entries - scoped knowledge store",
                "ordering": ["workspace_id", "-importance_score", "created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="gamememoryentry",
            index=models.Index(
                fields=["workspace", "scope_type"],
                name="aihub_mem_ws_scope_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="gamememoryentry",
            index=models.Index(
                fields=["goal", "scope_type"],
                name="aihub_mem_goal_scope_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="gamememoryentry",
            index=models.Index(
                fields=["workspace", "importance_score"],
                name="aihub_mem_ws_import_idx",
            ),
        ),
    ]
