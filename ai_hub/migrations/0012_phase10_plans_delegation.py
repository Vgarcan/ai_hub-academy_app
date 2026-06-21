import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ai_hub', '0011_stabilize_phases_06_09'),
    ]

    operations = [
        migrations.CreateModel(
            name='GameGoalPlan',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('version', models.PositiveIntegerField(default=1)),
                ('summary', models.TextField(blank=True)),
                ('status', models.CharField(
                    choices=[
                        ('active', 'Active'),
                        ('completed', 'Completed'),
                        ('abandoned', 'Abandoned'),
                    ],
                    default='active',
                    max_length=30,
                )),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('goal', models.OneToOneField(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='plan',
                    to='ai_hub.gamegoal',
                )),
            ],
            options={
                'verbose_name': '4.15 GAME goal plan',
                'verbose_name_plural': '4.15 GAME goal plans - structured execution aids',
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='GameGoalPlanStep',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('order', models.PositiveIntegerField()),
                ('title', models.CharField(max_length=255)),
                ('description', models.TextField(blank=True)),
                ('status', models.CharField(
                    choices=[
                        ('pending', 'Pending'),
                        ('in_progress', 'In progress'),
                        ('completed', 'Completed'),
                        ('skipped', 'Skipped'),
                        ('blocked', 'Blocked'),
                    ],
                    default='pending',
                    max_length=30,
                )),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('plan', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='steps',
                    to='ai_hub.gamegoalplan',
                )),
                ('depends_on_step', models.ForeignKey(
                    blank=True,
                    null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='required_by_steps',
                    to='ai_hub.gamegoalplanstep',
                )),
            ],
            options={
                'verbose_name': '4.16 GAME goal plan step',
                'verbose_name_plural': '4.16 GAME goal plan steps',
                'ordering': ['plan_id', 'order'],
            },
        ),
        migrations.AddConstraint(
            model_name='gamegoalplanstep',
            constraint=models.UniqueConstraint(
                fields=('plan', 'order'), name='ai_hub_unique_plan_step_order'
            ),
        ),
        migrations.CreateModel(
            name='GameDelegationRun',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('status', models.CharField(
                    choices=[
                        ('pending', 'Pending'),
                        ('running', 'Running'),
                        ('success', 'Success'),
                        ('failed', 'Failed'),
                    ],
                    default='pending',
                    max_length=30,
                )),
                ('task', models.TextField()),
                ('expected_result', models.TextField(blank=True)),
                ('result_summary', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('finished_at', models.DateTimeField(blank=True, null=True)),
                ('parent_action_run', models.OneToOneField(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='delegation_run',
                    to='ai_hub.gameactionrun',
                )),
                ('parent_goal', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='delegation_runs',
                    to='ai_hub.gamegoal',
                )),
                ('delegated_session', models.OneToOneField(
                    blank=True,
                    null=True,
                    on_delete=django.db.models.deletion.PROTECT,
                    related_name='delegation_run',
                    to='ai_hub.executionsession',
                )),
                ('target_agent', models.ForeignKey(
                    on_delete=django.db.models.deletion.PROTECT,
                    related_name='delegation_runs',
                    to='ai_hub.agentprofile',
                )),
            ],
            options={
                'verbose_name': '4.17 GAME delegation run',
                'verbose_name_plural': '4.17 GAME delegation runs - sub-agent history',
                'ordering': ['-created_at'],
                'indexes': [
                    models.Index(
                        fields=['parent_goal', 'status'],
                        name='aihub_delegation_goal_stat_idx',
                    ),
                ],
            },
        ),
    ]
