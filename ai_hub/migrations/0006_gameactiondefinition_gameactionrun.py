import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ai_hub", "0005_executionsession_goal_outcome_fingerprint_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="GameActionDefinition",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.SlugField(max_length=120, unique=True)),
                ("display_name", models.CharField(max_length=160)),
                ("description", models.TextField(blank=True)),
                (
                    "handler",
                    models.CharField(
                        choices=[
                            ("finish_goal", "Finish goal"),
                            ("search_knowledge", "Search knowledge"),
                            ("read_document", "Read document"),
                        ],
                        max_length=60,
                    ),
                ),
                ("input_schema", models.JSONField(blank=True, default=dict)),
                ("is_active", models.BooleanField(default=True)),
                ("config", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "4.8 GAME action definition",
                "verbose_name_plural": "4.8 GAME action definitions - dispatcher registry",
                "ordering": ["name"],
            },
        ),
        migrations.CreateModel(
            name="GameActionRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "session",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="action_runs",
                        to="ai_hub.executionsession",
                    ),
                ),
                (
                    "action_definition",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="runs",
                        to="ai_hub.gameactiondefinition",
                    ),
                ),
                ("idempotency_key", models.CharField(max_length=200, unique=True)),
                ("action_name", models.CharField(max_length=120)),
                ("iteration", models.PositiveIntegerField()),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("running", "Running"),
                            ("success", "Success"),
                            ("failed", "Failed"),
                        ],
                        default="pending",
                        max_length=20,
                    ),
                ),
                ("input_payload", models.JSONField(blank=True, default=dict)),
                ("output_payload", models.JSONField(blank=True, default=dict)),
                ("error_detail", models.TextField(blank=True)),
                ("dispatched_at", models.DateTimeField(auto_now_add=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
            ],
            options={
                "verbose_name": "4.9 GAME action run",
                "verbose_name_plural": "4.9 GAME action runs - dispatcher history",
                "ordering": ["session_id", "iteration"],
            },
        ),
        migrations.AddIndex(
            model_name="gameactionrun",
            index=models.Index(fields=["session", "iteration"], name="aihub_action_run_sess_iter_idx"),
        ),
        migrations.AddIndex(
            model_name="gameactionrun",
            index=models.Index(fields=["action_definition", "status"], name="aihub_action_run_def_stat_idx"),
        ),
    ]
