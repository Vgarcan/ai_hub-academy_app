from django.core.management.base import BaseCommand

from ai_hub.services.starter_demo import seed_starter_demo


class Command(BaseCommand):
    help = "Seed AI Hub starter demo knowledge, GAME workspace, and approval-gated action."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force-update",
            action="store_true",
            help="Update existing starter demo records to match current seed definitions.",
        )

    def handle(self, *args, **options):
        stats = seed_starter_demo(force_update=options["force_update"])
        self.stdout.write(
            self.style.SUCCESS(
                "AI Hub starter demo ready: "
                f"{stats['collections_created']} collection(s), "
                f"{stats['documents_created']} document(s), "
                f"{stats['chunks_created']} chunk(s), "
                f"{stats['workspaces_created']} workspace(s), "
                f"{stats['actions_created']} action(s), "
                f"{stats['workspace_actions_created']} workspace action(s), "
                f"{stats['workspace_agents_created']} workspace agent(s) created."
            )
        )
