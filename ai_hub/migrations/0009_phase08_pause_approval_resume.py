import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ai_hub", "0008_gamememoryentry"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="GameContinuationRequest",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                (
                    "session",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="continuation_requests",
                        to="ai_hub.executionsession",
                    ),
                ),
                (
                    "goal",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="continuation_requests",
                        to="ai_hub.gamegoal",
                    ),
                ),
                (
                    "resolved_by",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="resolved_continuation_requests",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "reason_code",
                    models.CharField(
                        max_length=80,
                        choices=[
                            ("needs_information", "Needs information"),
                            ("needs_approval", "Needs approval"),
                            ("external_result_pending", "External result pending"),
                            ("rate_limited", "Rate limited"),
                            ("manual_review_required", "Manual review required"),
                        ],
                    ),
                ),
                ("detail", models.TextField(blank=True)),
                ("payload", models.JSONField(blank=True, default=dict)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("resolved", "Resolved"),
                            ("expired", "Expired"),
                            ("cancelled", "Cancelled"),
                        ],
                        default="pending",
                        max_length=30,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("resolved_at", models.DateTimeField(blank=True, null=True)),
            ],
            options={
                "verbose_name": "4.11 GAME continuation request",
                "verbose_name_plural": "4.11 GAME continuation requests - pause records",
                "ordering": ["-created_at"],
            },
        ),
        migrations.AddIndex(
            model_name="gamecontinuationrequest",
            index=models.Index(
                fields=["session", "status"],
                name="aihub_cont_req_sess_stat_idx",
            ),
        ),
        migrations.CreateModel(
            name="GameActionApprovalRequest",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                (
                    "action_run",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="approval_request",
                        to="ai_hub.gameactionrun",
                    ),
                ),
                (
                    "goal",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="approval_requests",
                        to="ai_hub.gamegoal",
                    ),
                ),
                (
                    "reviewed_by",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="reviewed_approval_requests",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("approved", "Approved"),
                            ("rejected", "Rejected"),
                            ("expired", "Expired"),
                        ],
                        default="pending",
                        max_length=30,
                    ),
                ),
                ("requested_payload", models.JSONField(blank=True, default=dict)),
                ("review_note", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("reviewed_at", models.DateTimeField(blank=True, null=True)),
                ("expires_at", models.DateTimeField(blank=True, null=True)),
            ],
            options={
                "verbose_name": "4.12 GAME action approval request",
                "verbose_name_plural": "4.12 GAME action approval requests - gated actions",
                "ordering": ["-created_at"],
                "permissions": [("approve_game_action", "Can approve GAME action requests")],
            },
        ),
        migrations.AddIndex(
            model_name="gameactionapprovalrequest",
            index=models.Index(
                fields=["goal", "status"],
                name="aihub_approval_goal_stat_idx",
            ),
        ),
    ]
