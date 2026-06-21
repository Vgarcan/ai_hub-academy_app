import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ai_hub", "0006_gameactiondefinition_gameactionrun"),
    ]

    operations = [
        # ── GameActionDefinition: rename fields ──────────────────────────────
        migrations.RenameField(
            model_name="gameactiondefinition",
            old_name="display_name",
            new_name="label",
        ),
        migrations.RenameField(
            model_name="gameactiondefinition",
            old_name="handler",
            new_name="action_type",
        ),
        migrations.RenameField(
            model_name="gameactiondefinition",
            old_name="input_schema",
            new_name="input_contract",
        ),
        # ── GameActionDefinition: update action_type to new choices/max_length ─
        migrations.AlterField(
            model_name="gameactiondefinition",
            name="action_type",
            field=models.CharField(
                max_length=30,
                choices=[
                    ("internal", "Internal"),
                    ("context_tool", "Context tool"),
                    ("tool", "Tool"),
                    ("python_callable", "Python callable"),
                    ("http", "HTTP"),
                    ("sub_agent", "Sub-agent"),
                    ("human_approval", "Human approval"),
                ],
            ),
        ),
        # ── GameActionDefinition: add new fields ─────────────────────────────
        migrations.AddField(
            model_name="gameactiondefinition",
            name="output_contract",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="gameactiondefinition",
            name="risk_level",
            field=models.CharField(default="low", max_length=20),
        ),
        migrations.AddField(
            model_name="gameactiondefinition",
            name="requires_approval",
            field=models.BooleanField(default=False),
        ),
        # ── GameActionRun: drop old index (references action_definition) ─────
        migrations.RemoveIndex(
            model_name="gameactionrun",
            name="aihub_action_run_def_stat_idx",
        ),
        # ── GameActionRun: rename FK action_definition → action ───────────────
        migrations.RenameField(
            model_name="gameactionrun",
            old_name="action_definition",
            new_name="action",
        ),
        # ── GameActionRun: rename timestamp fields ────────────────────────────
        migrations.RenameField(
            model_name="gameactionrun",
            old_name="dispatched_at",
            new_name="started_at",
        ),
        migrations.RenameField(
            model_name="gameactionrun",
            old_name="completed_at",
            new_name="finished_at",
        ),
        # started_at was auto_now_add; make it a plain nullable field
        migrations.AlterField(
            model_name="gameactionrun",
            name="started_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        # ── GameActionRun: expand status choices ─────────────────────────────
        migrations.AlterField(
            model_name="gameactionrun",
            name="status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("running", "Running"),
                    ("waiting_approval", "Waiting approval"),
                    ("success", "Success"),
                    ("failed", "Failed"),
                    ("skipped", "Skipped"),
                    ("rejected", "Rejected"),
                ],
                default="pending",
                max_length=20,
            ),
        ),
        # ── GameActionRun: widen idempotency_key ──────────────────────────────
        migrations.AlterField(
            model_name="gameactionrun",
            name="idempotency_key",
            field=models.CharField(max_length=255, unique=True),
        ),
        # ── GameActionRun: add new fields ────────────────────────────────────
        migrations.AddField(
            model_name="gameactionrun",
            name="step_run",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="game_action_runs",
                to="ai_hub.executionsteprun",
            ),
        ),
        migrations.AddField(
            model_name="gameactionrun",
            name="observation_payload",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="gameactionrun",
            name="latency_ms",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        # ── GameActionRun: recreate index for action + status ─────────────────
        migrations.AddIndex(
            model_name="gameactionrun",
            index=models.Index(
                fields=["action", "status"],
                name="aihub_game_run_action_stat_idx",
            ),
        ),
    ]
