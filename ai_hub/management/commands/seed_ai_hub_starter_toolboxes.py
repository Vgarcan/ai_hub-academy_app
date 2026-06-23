from django.core.management.base import BaseCommand, CommandError

from ai_hub.models import ModelConfig
from ai_hub.services.starter_toolboxes import seed_starter_toolboxes


class Command(BaseCommand):
    help = "Seed AI Hub starter toolboxes and starter agent roles."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model-config-id",
            type=int,
            help="Optional ModelConfig id to use for starter agents. Defaults to a training/starter model.",
        )
        parser.add_argument(
            "--force-update",
            action="store_true",
            help="Update existing starter records to match current seed definitions.",
        )

    def handle(self, *args, **options):
        model_config = None
        if options.get("model_config_id"):
            try:
                model_config = ModelConfig.objects.get(pk=options["model_config_id"])
            except ModelConfig.DoesNotExist as exc:
                raise CommandError(f"ModelConfig #{options['model_config_id']} does not exist.") from exc

        stats = seed_starter_toolboxes(
            force_update=options["force_update"],
            model_config=model_config,
        )
        self.stdout.write(
            self.style.SUCCESS(
                "AI Hub starter toolboxes ready: "
                f"{stats['tools_created']} tool(s), "
                f"{stats['toolboxes_created']} toolbox(es), "
                f"{stats['memberships_created']} membership(s), "
                f"{stats['agents_created']} agent(s), "
                f"{stats['assignments_created']} assignment(s) created."
            )
        )
