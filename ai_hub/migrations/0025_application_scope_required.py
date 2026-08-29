"""Stage 3 of 3 — SCHEMA ONLY. Ownership becomes mandatory.

Safe to run now because 0024 has given every pre-existing row an owner. No data
operation belongs here: keeping this stage pure DDL is what stops a schema
change from sharing a transaction with a write, which is the portability defect
PostgreSQL rejected when all three stages lived in one migration.

The final model state is unchanged by the split: a required, PROTECT-ed FK with
no default and no `blank=True` on all three root-owned models.
"""

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ai_hub", "0024_application_scope_legacy_adoption"),
    ]

    operations = [
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
