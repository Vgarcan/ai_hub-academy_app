"""Validate or execute a versioned controlled retrieval-quality suite (S-29)."""

import json

from django.core.management.base import BaseCommand, CommandError

from ai_hub.services.retrieval_quality import (
    RetrievalQualityError,
    execute_suite,
    load_suite,
    preflight_as_dict,
    preflight_suite,
)


class Command(BaseCommand):
    help = (
        "Validate an exact controlled retrieval corpus; with --execute, run "
        "lexical and hybrid evaluation. Report-only by default."
    )

    def add_arguments(self, parser):
        parser.add_argument("--suite", required=True, help="Path to a versioned suite JSON file.")
        parser.add_argument("--agent", required=True, help="Agent name or numeric id.")
        parser.add_argument("--json", action="store_true", help="Emit structured content-free JSON.")
        parser.add_argument(
            "--execute", action="store_true",
            help="Send suite queries through governed hybrid retrieval. No evaluation data is persisted.",
        )

    def handle(self, *args, **options):
        try:
            suite = load_suite(options["suite"])
            preflight = preflight_suite(suite, options["agent"])
            payload = execute_suite(preflight) if options["execute"] else preflight_as_dict(preflight)
        except RetrievalQualityError as exc:
            raise CommandError(exc.category) from exc

        if options["json"]:
            self.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
        else:
            self._write_human(payload)

        if not payload["preflight"]["ready"]:
            raise CommandError("preflight_not_ready")

    def _write_human(self, payload):
        preflight = payload["preflight"]
        self.stdout.write(
            f"Retrieval quality {payload['suite_id']} ({payload['evidence_class']})"
        )
        self.stdout.write(
            f"  Corpus: {preflight['document_count']} exact documents; "
            f"coverage complete={preflight['coverage_complete']}"
        )
        self.stdout.write(
            f"  Cases: {preflight['answerable_case_count']} answerable, "
            f"{preflight['no_answer_case_count']} no-answer"
        )
        if not payload["executed"]:
            self.stdout.write(
                "Report only: no provider was called and nothing was written. "
                "Use --execute to evaluate."
            )
            return
        for path in ("lexical", "hybrid"):
            metrics = payload["metrics"][path]
            self.stdout.write(
                f"  {path}: top1={metrics['top1_count']}/{metrics['answerable_cases']}; "
                f"MRR={metrics['mean_reciprocal_rank']:.4f}; "
                f"no-answer empty={metrics['no_answer_empty_count']}/{metrics['no_answer_cases']}"
            )
