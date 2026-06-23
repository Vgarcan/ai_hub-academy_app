"""
Cancel GAME goals stuck in RUNNING with no active execution session.

A goal can stay stuck in RUNNING when its session(s) never reach a terminal state
(interrupted runs, or stub sessions left behind by integration test suites). This
command cancels those orphaned goals so they do not accumulate.

Usage:
    python manage.py cleanup_orphaned_goals
    python manage.py cleanup_orphaned_goals --workspace 3
    python manage.py cleanup_orphaned_goals --older-than-hours 24
    python manage.py cleanup_orphaned_goals --dry-run
"""
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError

from ai_hub.models import GameWorkspace
from ai_hub.services.game_goals import (
    cancel_orphaned_running_goals,
    find_orphaned_running_goals,
)


class Command(BaseCommand):
    help = "Cancel GAME goals stuck in RUNNING that have no active execution session."

    def add_arguments(self, parser):
        parser.add_argument(
            "--workspace",
            type=int,
            default=None,
            help="Limit the sweep to one workspace id.",
        )
        parser.add_argument(
            "--older-than-hours",
            type=float,
            default=None,
            help="Only consider goals not updated within this many hours.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would be cancelled without changing anything.",
        )

    def handle(self, *args, **options):
        workspace = None
        if options["workspace"] is not None:
            try:
                workspace = GameWorkspace.objects.get(pk=options["workspace"])
            except GameWorkspace.DoesNotExist:
                raise CommandError(f"Workspace {options['workspace']} does not exist.")

        older_than = None
        if options["older_than_hours"] is not None:
            older_than = timedelta(hours=options["older_than_hours"])

        candidates = find_orphaned_running_goals(workspace=workspace, older_than=older_than)
        if not candidates:
            self.stdout.write("No orphaned running goals found.")
            return

        if options["dry_run"]:
            self.stdout.write(
                f"[dry-run] {len(candidates)} orphaned running goal(s) would be cancelled:"
            )
            for goal in candidates:
                self.stdout.write(f"  - #{goal.pk} {goal.title} (workspace {goal.workspace_id})")
            return

        cancelled = cancel_orphaned_running_goals(workspace=workspace, older_than=older_than)
        self.stdout.write(
            self.style.SUCCESS(f"Cancelled {len(cancelled)} orphaned running goal(s).")
        )
