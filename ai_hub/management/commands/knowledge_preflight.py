"""
Report structural retrieval readiness and lifecycle state of the Knowledge corpus.

READ-ONLY. This command inspects and reports. It never repairs, backfills,
chunks, adjudicates, regenerates or writes anything, and it has no --fix,
--write, --repair, --backfill, --regenerate or --adjudicate mode by design.

Usage:
    python manage.py knowledge_preflight
    python manage.py knowledge_preflight --collection 3 --collection 7
    python manage.py knowledge_preflight --status active
    python manage.py knowledge_preflight --json
    python manage.py knowledge_preflight --limit 50
"""
import json

from django.core.management.base import BaseCommand, CommandError

from ai_hub.models import KnowledgeDocument
from ai_hub.services.knowledge_preflight import (
    DEFAULT_DOCUMENT_LIMIT,
    ISSUE_CODES,
    SEVERITY_INFO,
    run_knowledge_preflight,
    summarize_preflight,
)


class Command(BaseCommand):
    help = (
        "Report structural retrieval readiness and lifecycle state of the "
        "Knowledge corpus. Read-only: this command never modifies Knowledge data."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--collection",
            type=int,
            action="append",
            dest="collections",
            default=None,
            help="Limit to a collection id. Repeat for several.",
        )
        parser.add_argument(
            "--status",
            action="append",
            dest="statuses",
            default=None,
            choices=[value for value, _label in KnowledgeDocument.Status.choices],
            help="Limit to a document status. Repeat for several.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=DEFAULT_DOCUMENT_LIMIT,
            help=(
                "Maximum documents listed in the detail output. Summary counts "
                f"always cover the full scope. Default {DEFAULT_DOCUMENT_LIMIT}."
            ),
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit the full structured report as JSON.",
        )

    def handle(self, *args, **options):
        try:
            report = run_knowledge_preflight(
                collection_ids=options["collections"],
                statuses=options["statuses"],
                document_limit=options["limit"],
            )
        except Exception as exc:  # surface a clean operator error, not a traceback
            raise CommandError(f"Knowledge preflight failed: {exc}") from exc

        if options["json"]:
            self.stdout.write(json.dumps(report, indent=2, sort_keys=True, default=str))
            return

        for line in summarize_preflight(report):
            self.stdout.write(line)

        blockers = sum(
            count
            for code, count in report["summary"]["issues_by_code"].items()
            if count and ISSUE_CODES[code]["severity"] != SEVERITY_INFO
        )
        if report["scope"]["documents_truncated"]:
            self.stdout.write(
                f"\nDetail list truncated to {report['scope']['document_limit']} "
                "documents. Summary counts cover the full scope; raise --limit "
                "or use --json for more."
            )

        self.stdout.write("")
        if blockers:
            self.stdout.write(
                self.style.WARNING(
                    f"{blockers} actionable issue(s) found — structural and/or "
                    "lifecycle. Nothing was changed: this command is read-only."
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    "No actionable structural or lifecycle issues found. "
                    "Nothing was changed: this command is read-only."
                )
            )
