"""
Report, and on request close, embedding coverage of a Knowledge corpus (S-27).

REPORT-ONLY BY DEFAULT. Without --execute this command reads and reports: it
makes no provider call and writes nothing. With --execute it embeds every
MISSING or STALE eligible chunk through S-20's `index_chunk_embedding_local()`,
one chunk at a time, and nothing else.

An ADAPTER. Every rule - authorization, locality, input ceiling, currentness,
the in-flight compare-and-swap, storage - belongs to S-17..S-20 and to
`ai_hub/services/embedding_coverage.py`. This command parses arguments, prints
and sets the exit status.

Why --execute is a plain flag when `knowledge_lifecycle_action` forbids every
automation bypass: lifecycle verbs decide Knowledge AUTHORITY and can destroy
human edits, so a prior human review must be provable. Vectors are derived
index state, rebuildable from the chunk and never authority, and S-17 already
gates what may be sent. That difference is the reason - it is not a precedent.

Output carries ids, counts, bounded codes and contract facts. Never chunk text,
section titles, k1 values, vector values or provider bodies.

Usage:
    python manage.py knowledge_embedding_index --scope legacy-default --embedding-config 3
    python manage.py knowledge_embedding_index --scope legacy-default --embedding-config 3 --collection 7
    python manage.py knowledge_embedding_index --scope legacy-default --embedding-config 3 --execute
    python manage.py knowledge_embedding_index --scope legacy-default --embedding-config 3 --execute --limit 50
    python manage.py knowledge_embedding_index --scope legacy-default --embedding-config 3 --json

Exit status: report mode exits 0 whenever the report could be produced. Execute
mode exits non-zero unless the target ends fully indexed with no refusal, stop,
failure or --limit truncation - a partial index never looks complete to a
script.
"""
import json

from django.core.management.base import BaseCommand, CommandError

from ai_hub.services.embedding_coverage import (
    COVERAGE_STATES,
    IndexTargetError,
    build_coverage_report,
    execute_embedding_index,
    report_as_dict,
    resolve_index_target,
    run_as_dict,
)


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise ValueError("must be at least 1")
    return number


class Command(BaseCommand):
    help = (
        "Report embedding coverage of a Knowledge corpus; with --execute, embed "
        "missing and stale chunks through a LOCAL provider. Report-only by default."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--scope", required=True,
            help="Application scope slug or id. Required; there is no default.",
        )
        parser.add_argument(
            "--embedding-config", required=True, dest="embedding_config",
            help="Embedding model configuration name or id. Required; there is no default.",
        )
        parser.add_argument(
            "--collection", type=int, action="append", dest="collections", default=None,
            help="Narrow to a collection id inside the scope. Repeat for several.",
        )
        parser.add_argument(
            "--limit", type=_positive_int, default=None,
            help="With --execute: maximum chunks attempted. What it leaves is reported.",
        )
        parser.add_argument(
            "--json", action="store_true",
            help="Emit the structured report as JSON.",
        )
        parser.add_argument(
            "--execute", action="store_true",
            help="Embed MISSING and STALE chunks. Without it nothing is sent or written.",
        )

    def handle(self, *args, **options):
        if options["limit"] is not None and not options["execute"]:
            raise CommandError("--limit only applies together with --execute.")

        try:
            target = resolve_index_target(
                scope_ref=options["scope"],
                embedding_config_ref=options["embedding_config"],
                collection_ids=options["collections"],
            )
        except IndexTargetError as exc:
            raise CommandError(f"{exc.category}: {exc}") from exc

        if not options["execute"]:
            report = build_coverage_report(target)
            if options["json"]:
                self._write_json(report_as_dict(report))
                return
            self._write_report(report_as_dict(report), heading="Coverage")
            self.stdout.write("")
            self.stdout.write(
                "Report only: no provider was called and nothing was written. "
                "Use --execute to embed missing and stale chunks."
            )
            return

        run = execute_embedding_index(target, limit=options["limit"])
        payload = run_as_dict(run)
        if options["json"]:
            self._write_json(payload)
        else:
            self._write_run(payload)
        if not run.complete:
            raise CommandError(self._incomplete_reason(payload))

    # -- rendering ----------------------------------------------------------

    def _write_json(self, payload):
        self.stdout.write(json.dumps(payload, indent=2, sort_keys=True))

    def _write_report(self, data, *, heading):
        target = data["target"]
        self.stdout.write(f"{heading} (coverage contract v{data['coverage_contract_version']})")
        self.stdout.write(
            f"  Scope: {target['application_scope_slug']} (id {target['application_scope_id']})"
        )
        self.stdout.write(
            "  Collections: "
            + (", ".join(str(pk) for pk in target["collection_ids"]) or "none")
            + (" (explicit)" if target["collections_explicit"] else " (all active in scope)")
        )
        if target["excluded_inactive_collection_ids"]:
            self.stdout.write(
                "  Inactive collections not targeted: "
                + ", ".join(str(pk) for pk in target["excluded_inactive_collection_ids"])
            )
        contract = data["contract"]
        if contract is not None:
            self.stdout.write(
                f"  Embedding config {target['embedding_model_config_id']}: "
                f"{contract['model_name']} @ {contract['model_revision']}, "
                f"{contract['vector_dimension']}d, {contract['distance_metric']}, "
                f"normalization={contract['normalization']}, "
                f"max_input_chars={contract['max_input_chars']}"
            )
            self.stdout.write(
                f"  Provider {contract['provider_id']}: {contract['provider_type']}, "
                f"declared_locality={contract['declared_locality']}"
            )
            self.stdout.write(f"  e1: {contract['e1']}")

        preconditions = data["preconditions"]
        if preconditions["ok"]:
            self.stdout.write(self.style.SUCCESS("  Preconditions: OK"))
        else:
            self.stdout.write(
                self.style.WARNING(
                    "  Preconditions: REFUSED - "
                    + ", ".join(preconditions["refusal_codes"])
                )
            )
            for decision in preconditions["collections"]:
                if not decision["allowed"]:
                    self.stdout.write(
                        f"    collection {decision['collection_id']}: {decision['code']}"
                    )

        if not data["coverage_available"]:
            self.stdout.write(
                "  Coverage: unavailable - the embedding configuration does not resolve."
            )
            return

        counts = data["counts"]
        self.stdout.write(f"  Eligible chunks (ACTIVE documents): {data['eligible_chunks']}")
        for state in COVERAGE_STATES:
            self.stdout.write(f"    {state:<15} {counts[state]}")
        for code, count in data["stale_reasons"].items():
            self.stdout.write(f"      stale: {code} x{count}")
        for item in data["not_embeddable"]:
            self.stdout.write(f"      not embeddable: chunk {item['chunk_id']} ({item['reason']})")
        if data["outside_eligibility_vectors"]:
            self.stdout.write(
                f"  Vectors outside eligibility (document no longer ACTIVE, kept, "
                f"not deleted): {data['outside_eligibility_vectors']}"
            )

    def _write_run(self, payload):
        self._write_report(payload["before"], heading="Coverage before")
        self.stdout.write("")
        if payload["refused"]:
            self.stdout.write(
                self.style.ERROR("Run REFUSED before any provider call. Nothing was sent.")
            )
            return
        self.stdout.write(f"Attempted: {payload['attempted']}")
        for outcome, count in payload["outcomes"].items():
            self.stdout.write(f"  {outcome:<45} {count}")
        for item in payload["failed_chunks"]:
            self.stdout.write(f"  failed: chunk {item['chunk_id']} ({item['category']})")
        if payload["stopped_by"]:
            self.stdout.write(
                self.style.ERROR(
                    f"Stopped at the first run-fatal failure: {payload['stopped_by']}. "
                    f"{payload['not_attempted_after_stop']} selected chunk(s) not attempted. "
                    "Nothing is retried."
                )
            )
        if payload["remaining_due_to_limit"]:
            self.stdout.write(
                self.style.WARNING(
                    f"--limit left {payload['remaining_due_to_limit']} chunk(s) to embed."
                )
            )
        if payload["after"] is not None:
            self.stdout.write("")
            self._write_report(payload["after"], heading="Coverage after")

    @staticmethod
    def _incomplete_reason(payload):
        if payload["refused"]:
            codes = ", ".join(payload["before"]["preconditions"]["refusal_codes"])
            return f"Indexing refused before any provider call ({codes})."
        if payload["stopped_by"]:
            return f"Indexing stopped: {payload['stopped_by']}."
        if payload["remaining_due_to_limit"]:
            return (
                f"Indexing incomplete: --limit left "
                f"{payload['remaining_due_to_limit']} chunk(s)."
            )
        return "Indexing incomplete: the target is not fully indexed (see report)."
