"""Read-only structural preflight for the reusable Knowledge corpus.

WHAT THIS IS
------------
An operator/corpus-health diagnostic. It inspects `KnowledgeDocument` and
`KnowledgeDocumentChunk` and reports whether the corpus is structurally ready
for canonical (chunk-level) retrieval.

NON-NEGOTIABLE INVARIANT: THIS SERVICE IS READ-ONLY.

It never creates, updates or deletes a row, never touches `metadata`,
`curated_text`, status or timestamps, never generates fingerprints, never
enqueues work and never calls a provider. Running it twice against the same
database returns the same report — the report deliberately contains no
timestamp, so equality is testable.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
The current schema does not persist a lifecycle mode, so this service must not
pretend it can tell a DERIVED chunk set from an EXPLICIT one. Content
inequality between `curated_text` and chunk text is **not** evidence of
staleness — Slice 4 measured that the two cases carry identical evidence and
opposite correct remedies. Documents in that shape are reported as
`UNKNOWN_LEGACY` with an informational issue, never as stale or defective.

Equally, "derivation staleness" is not "domain freshness". This service speaks
only about whether the retrieval representation is structurally usable. Whether
the knowledge is still *true* is a separate concern and out of scope.

SECURITY BOUNDARY
-----------------
This is an **operator** capability, not an Agent one. Retrieval authorization
(Agent -> active collections -> ACTIVE documents -> chunks) and operator corpus
diagnostics are different concerns with different audiences, so this service
takes no Agent and applies no Agent scope.

It must never be registered as a `ToolDefinition`, auto-resolved into an Agent,
exposed through GAME, or called from a retrieval path. Any future Admin or
management surface handles its own authorization.
"""
from django.db.models.functions import Length, Substr

from ai_hub.models import KnowledgeCollection, KnowledgeDocument, KnowledgeDocumentChunk


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------
# Chunk bodies are never loaded. Only a short prefix is read, purely to decide
# whether a chunk carries usable text. See `_chunk_usable()` for the one case
# this deliberately cannot decide, and how it is resolved conservatively.
CHUNK_PROBE_CHARS = 200

# The per-document detail list is bounded. Summary counts are ALWAYS computed
# over the full scope, so truncation never distorts the totals.
DEFAULT_DOCUMENT_LIMIT = 500
MAX_DOCUMENT_LIMIT = 5000

# Bounded evidence attached to an issue.
MAX_ANOMALY_INDEXES = 20


# --------------------------------------------------------------------------
# Classification model
# --------------------------------------------------------------------------
# Exactly one classification per document. Names describe STRUCTURE and, where
# they imply readiness, they say so explicitly.

READY_CANONICAL = "READY_CANONICAL"
UNKNOWN_LEGACY = "UNKNOWN_LEGACY"
SOURCE_WITHOUT_CHUNKS = "SOURCE_WITHOUT_CHUNKS"
UNUSABLE_CHUNKS = "UNUSABLE_CHUNKS"
EMPTY_ACTIVE = "EMPTY_ACTIVE"
NON_ACTIVE = "NON_ACTIVE"

CLASSIFICATIONS = (
    READY_CANONICAL,
    UNKNOWN_LEGACY,
    SOURCE_WITHOUT_CHUNKS,
    UNUSABLE_CHUNKS,
    EMPTY_ACTIVE,
    NON_ACTIVE,
)

CLASSIFICATION_MEANINGS = {
    READY_CANONICAL: (
        "ACTIVE, at least one usable chunk, and no source text. Canonically "
        "retrievable, and the chunk set is unambiguously the authored artifact."
    ),
    UNKNOWN_LEGACY: (
        "ACTIVE, at least one usable chunk, AND source text present. "
        "Canonically retrievable. Whether the chunks are derived from that "
        "source or were authored explicitly cannot be proven by the current "
        "schema. NOT a defect and NOT stale."
    ),
    SOURCE_WITHOUT_CHUNKS: (
        "ACTIVE with a source (curated text and/or an uploaded file) but zero "
        "chunks. Invisible to canonical retrieval: a convergence blocker."
    ),
    UNUSABLE_CHUNKS: (
        "ACTIVE with chunk rows that carry no usable text. Structurally "
        "present, retrievably empty."
    ),
    EMPTY_ACTIVE: (
        "ACTIVE with no usable source and no chunks. Advertised by the "
        "Knowledge index yet unretrievable."
    ),
    NON_ACTIVE: (
        "DRAFT or ARCHIVED. Excluded from retrieval by design; absent chunks "
        "are not a defect here."
    ),
}


# --------------------------------------------------------------------------
# Issue codes — stable, documented, deliberately few
# --------------------------------------------------------------------------

KP001_SOURCE_WITHOUT_CHUNKS = "KP001_SOURCE_WITHOUT_CHUNKS"
KP002_EMPTY_ACTIVE = "KP002_EMPTY_ACTIVE"
KP003_UNUSABLE_CHUNK_CONTENT = "KP003_UNUSABLE_CHUNK_CONTENT"
KP004_UNKNOWN_AUTHORITY = "KP004_UNKNOWN_AUTHORITY"
KP005_SOURCE_FILE_WITHOUT_CHUNKS = "KP005_SOURCE_FILE_WITHOUT_CHUNKS"
KP006_CHUNK_INDEX_ANOMALY = "KP006_CHUNK_INDEX_ANOMALY"

SEVERITY_BLOCKER = "blocker"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

ISSUE_CODES = {
    KP001_SOURCE_WITHOUT_CHUNKS: {
        "severity": SEVERITY_BLOCKER,
        "meaning": (
            "ACTIVE document has source text but no chunks, so canonical "
            "retrieval cannot see it at all."
        ),
    },
    KP002_EMPTY_ACTIVE: {
        "severity": SEVERITY_WARNING,
        "meaning": (
            "ACTIVE document has neither usable source nor chunks. It is listed "
            "in the Knowledge index but nothing can ever be retrieved from it."
        ),
    },
    KP003_UNUSABLE_CHUNK_CONTENT: {
        "severity": SEVERITY_BLOCKER,
        "meaning": "One or more chunks carry empty or whitespace-only content.",
    },
    KP004_UNKNOWN_AUTHORITY: {
        "severity": SEVERITY_INFO,
        "meaning": (
            "Both a source and chunks exist. The current schema cannot prove "
            "whether the chunks are derived or explicitly authored. This is "
            "informational: it is NOT a defect, NOT drift and NOT staleness."
        ),
    },
    KP005_SOURCE_FILE_WITHOUT_CHUNKS: {
        "severity": SEVERITY_BLOCKER,
        "meaning": (
            "An uploaded source_file exists but no chunks do. source_file is "
            "not part of the retrieval-first path, so its content is "
            "unreachable until it is chunked."
        ),
    },
    KP006_CHUNK_INDEX_ANOMALY: {
        "severity": SEVERITY_WARNING,
        "meaning": (
            "Chunk indexes are irregular (do not start at 1, contain gaps, or "
            "include index 0). Duplicates are impossible: a unique constraint "
            "prevents them."
        ),
    },
}

# Historical ingestion markers written by Core. Recognised only as evidence of
# ORIGIN. They are creation-time provenance and are never revisited, so a
# hand-edited chunk still carries the marker it was born with. This is
# KNOWN_DERIVATION_ORIGIN, never CURRENT_DERIVED_AUTHORITY.
KNOWN_INGESTION_MARKERS = (
    "initial_curated_text",
    "initial_curated_text_backfill",
)


def _chunk_usable(probe: str, content_chars: int) -> bool:
    """Decide whether a chunk carries usable retrieval text, without loading it.

    Only the first `CHUNK_PROBE_CHARS` characters are read. That resolves every
    realistic case:

    * zero length          -> unusable;
    * blank probe and the whole body fits in the probe -> unusable;
    * blank probe but the body is longer -> treated as USABLE, because a defect
      has not been proven. Reporting an unprovable defect is worse than missing
      a pathological one.
    """
    if content_chars == 0:
        return False
    if content_chars <= CHUNK_PROBE_CHARS and not probe.strip():
        return False
    return True


def _index_anomalies(indexes: list[int]) -> list[str]:
    """Structural irregularities in a document's chunk numbering."""
    anomalies = []
    if not indexes:
        return anomalies
    ordered = sorted(indexes)
    if 0 in ordered:
        anomalies.append("contains_index_zero")
    if ordered[0] not in (0, 1):
        anomalies.append("does_not_start_at_one")
    if ordered[-1] - ordered[0] + 1 != len(ordered):
        anomalies.append("has_gaps")
    return anomalies


def _bounded_limit(value, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, 1), maximum)


def _collect_chunk_facts(document_ids):
    """One bounded pass over chunks. Never selects a full `content` body.

    Returns {document_id: {counts, index list, metadata marker summary}}.
    """
    facts = {}
    queryset = (
        KnowledgeDocumentChunk.objects.filter(document_id__in=document_ids)
        .annotate(
            content_probe=Substr("content", 1, CHUNK_PROBE_CHARS),
            content_chars=Length("content"),
        )
        .values("document_id", "chunk_index", "content_probe", "content_chars", "metadata")
        .order_by("document_id", "chunk_index")
    )
    for row in queryset.iterator():
        entry = facts.setdefault(
            row["document_id"],
            {
                "chunk_count": 0,
                "usable_chunk_count": 0,
                "indexes": [],
                "markers": {},
                "chunks_without_metadata": 0,
            },
        )
        entry["chunk_count"] += 1
        entry["indexes"].append(row["chunk_index"])
        if _chunk_usable(row["content_probe"] or "", row["content_chars"] or 0):
            entry["usable_chunk_count"] += 1

        metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
        marker = metadata.get("ingestion")
        if marker in KNOWN_INGESTION_MARKERS:
            entry["markers"][marker] = entry["markers"].get(marker, 0) + 1
        elif not metadata:
            entry["chunks_without_metadata"] += 1
    return facts


def _classify(document, facts) -> tuple[str, list[dict]]:
    """Return (classification, issues) for one document. Conservative by design."""
    issues = []
    has_source_text = bool((document.curated_text or "").strip())
    has_source_file = bool(document.source_file)
    chunk_count = facts["chunk_count"]
    usable = facts["usable_chunk_count"]

    if document.status != KnowledgeDocument.Status.ACTIVE:
        return NON_ACTIVE, issues

    anomalies = _index_anomalies(facts["indexes"])
    if anomalies:
        issues.append(
            {
                "code": KP006_CHUNK_INDEX_ANOMALY,
                "detail": {
                    "anomalies": anomalies,
                    "indexes": sorted(facts["indexes"])[:MAX_ANOMALY_INDEXES],
                    "indexes_truncated": len(facts["indexes"]) > MAX_ANOMALY_INDEXES,
                },
            }
        )

    if chunk_count == 0:
        if has_source_text:
            issues.append({"code": KP001_SOURCE_WITHOUT_CHUNKS, "detail": {}})
        if has_source_file:
            issues.append({"code": KP005_SOURCE_FILE_WITHOUT_CHUNKS, "detail": {}})
        if has_source_text or has_source_file:
            return SOURCE_WITHOUT_CHUNKS, issues
        issues.append({"code": KP002_EMPTY_ACTIVE, "detail": {}})
        return EMPTY_ACTIVE, issues

    if usable < chunk_count:
        issues.append(
            {
                "code": KP003_UNUSABLE_CHUNK_CONTENT,
                "detail": {
                    "chunk_count": chunk_count,
                    "usable_chunk_count": usable,
                    "unusable_chunk_count": chunk_count - usable,
                },
            }
        )

    if usable == 0:
        return UNUSABLE_CHUNKS, issues

    if has_source_text:
        # Retrievable, but authority is unprovable. Informational only.
        issues.append({"code": KP004_UNKNOWN_AUTHORITY, "detail": {}})
        return UNKNOWN_LEGACY, issues

    return READY_CANONICAL, issues


def run_knowledge_preflight(
    *,
    collection_ids=None,
    statuses=None,
    document_limit: int = DEFAULT_DOCUMENT_LIMIT,
) -> dict:
    """Inspect the Knowledge corpus and report structural retrieval readiness.

    Read-only. Returns structured data, never formatted strings, and contains no
    timestamp so that repeated runs against unchanged data compare equal.

    Optional scoping:
      * `collection_ids` — restrict to specific collections;
      * `statuses`       — restrict to specific document statuses.

    There is deliberately no Agent scope: this is an operator diagnostic, not a
    retrieval path.
    """
    bounded_document_limit = _bounded_limit(
        document_limit, default=DEFAULT_DOCUMENT_LIMIT, maximum=MAX_DOCUMENT_LIMIT
    )

    documents = KnowledgeDocument.objects.select_related("collection")
    if collection_ids is not None:
        documents = documents.filter(collection_id__in=list(collection_ids))
    if statuses is not None:
        documents = documents.filter(status__in=list(statuses))
    documents = documents.order_by("collection__name", "title", "pk")

    in_scope = list(documents)
    facts_by_document = _collect_chunk_facts([document.pk for document in in_scope])
    empty_facts = {
        "chunk_count": 0,
        "usable_chunk_count": 0,
        "indexes": [],
        "markers": {},
        "chunks_without_metadata": 0,
    }

    by_status = {value: 0 for value, _label in KnowledgeDocument.Status.choices}
    by_classification = {name: 0 for name in CLASSIFICATIONS}
    issues_by_code = {code: 0 for code in ISSUE_CODES}
    collection_rows = {}
    document_rows = []
    issue_rows = []

    chunks_total = 0
    chunks_usable_total = 0
    active_documents = 0
    active_retrievable = 0

    for document in in_scope:
        facts = facts_by_document.get(document.pk, empty_facts)
        classification, issues = _classify(document, facts)

        by_status[document.status] = by_status.get(document.status, 0) + 1
        by_classification[classification] += 1
        chunks_total += facts["chunk_count"]
        chunks_usable_total += facts["usable_chunk_count"]

        is_active = document.status == KnowledgeDocument.Status.ACTIVE
        retrievable = bool(
            is_active
            and document.collection.is_active
            and facts["usable_chunk_count"] > 0
        )
        if is_active:
            active_documents += 1
            if retrievable:
                active_retrievable += 1

        collection_row = collection_rows.setdefault(
            document.collection_id,
            {
                "collection_id": document.collection_id,
                "name": document.collection.name,
                "is_active": document.collection.is_active,
                "documents": 0,
                "by_classification": {name: 0 for name in CLASSIFICATIONS},
                "issue_counts": {},
                "documents_with_issues": 0,
            },
        )
        collection_row["documents"] += 1
        collection_row["by_classification"][classification] += 1

        actionable = [
            issue for issue in issues
            if ISSUE_CODES[issue["code"]]["severity"] != SEVERITY_INFO
        ]
        if actionable:
            collection_row["documents_with_issues"] += 1

        for issue in issues:
            code = issue["code"]
            issues_by_code[code] += 1
            collection_row["issue_counts"][code] = (
                collection_row["issue_counts"].get(code, 0) + 1
            )
            issue_rows.append(
                {
                    "code": code,
                    "severity": ISSUE_CODES[code]["severity"],
                    "document_id": document.pk,
                    "collection_id": document.collection_id,
                    "detail": issue["detail"],
                }
            )

        if len(document_rows) < bounded_document_limit:
            document_rows.append(
                {
                    "document_id": document.pk,
                    "collection_id": document.collection_id,
                    "collection": document.collection.name,
                    "title": document.title,
                    "status": document.status,
                    "collection_is_active": document.collection.is_active,
                    "has_source_text": bool((document.curated_text or "").strip()),
                    "has_source_file": bool(document.source_file),
                    "chunk_count": facts["chunk_count"],
                    "usable_chunk_count": facts["usable_chunk_count"],
                    "classification": classification,
                    "canonically_retrievable": retrievable,
                    "issues": [issue["code"] for issue in issues],
                    "provenance": {
                        # ORIGIN evidence only. Never current authority.
                        "known_derivation_origin": bool(facts["markers"]),
                        "ingestion_markers": dict(facts["markers"]),
                        "chunks_without_metadata": facts["chunks_without_metadata"],
                    },
                }
            )

    total_documents = KnowledgeDocument.objects.count()

    return {
        "scope": {
            "collection_ids": None if collection_ids is None else sorted(collection_ids),
            "statuses": None if statuses is None else sorted(statuses),
            "document_limit": bounded_document_limit,
            "documents_truncated": len(in_scope) > len(document_rows),
        },
        "summary": {
            "documents_total": total_documents,
            "documents_in_scope": len(in_scope),
            "collections_in_scope": len(collection_rows),
            "by_status": by_status,
            "by_classification": by_classification,
            "issues_by_code": issues_by_code,
            "chunks_total": chunks_total,
            "chunks_usable": chunks_usable_total,
            "chunks_unusable": chunks_total - chunks_usable_total,
            "active_documents": active_documents,
            "active_canonically_retrievable": active_retrievable,
            "active_not_retrievable": active_documents - active_retrievable,
        },
        # Facts an operator needs to judge convergence readiness. Deliberately
        # NOT a policy verdict: this service reports, it does not decide whether
        # GAME may switch to canonical retrieval.
        "convergence": {
            "active_source_without_chunks": by_classification[SOURCE_WITHOUT_CHUNKS],
            "active_unusable_chunks": by_classification[UNUSABLE_CHUNKS],
            "active_empty": by_classification[EMPTY_ACTIVE],
            "active_unknown_authority": by_classification[UNKNOWN_LEGACY],
            "collections_with_blockers": sorted(
                row["collection_id"]
                for row in collection_rows.values()
                if row["documents_with_issues"]
            ),
        },
        "collections": [
            collection_rows[key] for key in sorted(collection_rows)
        ],
        "documents": document_rows,
        "issues": issue_rows,
    }


def preflight_collection_choices():
    """Collections available for scoping. Read-only convenience for operators."""
    return list(
        KnowledgeCollection.objects.order_by("name").values("id", "name", "is_active")
    )


def summarize_preflight(report: dict) -> list[str]:
    """Render a concise operator summary. Presentation only; changes nothing."""
    summary = report["summary"]
    lines = [
        f"Documents in scope: {summary['documents_in_scope']} "
        f"(of {summary['documents_total']} total)",
        f"  by status:  " + ", ".join(
            f"{status}={count}" for status, count in sorted(summary["by_status"].items())
        ),
        f"  ACTIVE retrievable: {summary['active_canonically_retrievable']} / "
        f"{summary['active_documents']}",
        f"  chunks: {summary['chunks_total']} total, "
        f"{summary['chunks_unusable']} unusable",
        "Classification:",
    ]
    for name in CLASSIFICATIONS:
        lines.append(f"  {name:<24} {summary['by_classification'][name]}")
    lines.append("Issues:")
    for code in sorted(ISSUE_CODES):
        count = summary["issues_by_code"][code]
        if count:
            lines.append(
                f"  [{ISSUE_CODES[code]['severity']:<8}] {code:<34} {count}"
            )
    if not any(summary["issues_by_code"].values()):
        lines.append("  none")
    return lines
