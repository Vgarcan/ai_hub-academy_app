"""Lifecycle-aware read-only preflight for the reusable Knowledge corpus (V2).

WHAT THIS IS
------------
An operator/corpus-health diagnostic. It inspects `KnowledgeDocument` and
`KnowledgeDocumentChunk` and reports, as **facts**, whether the corpus is
structurally retrievable and whether its persisted lifecycle claims still hold.

NON-NEGOTIABLE INVARIANT: THIS SERVICE IS READ-ONLY.

It never creates, updates or deletes a row; never touches `metadata`,
`curated_text`, status, timestamps or any lifecycle field; never repairs,
regenerates, backfills or adjudicates; never enqueues work; never calls a
provider. Running it twice against the same database returns the same report —
the report deliberately contains no timestamp, so equality is testable.

It computes fingerprints in memory to compare against recorded ones. Computing
is not writing.

THREE INDEPENDENT AXES
----------------------
Evidence naturally has more than one dimension, and collapsing them would
destroy information. A document may simultaneously be structurally usable, mode
DERIVED, have changed inputs, have untouched chunks, and be canonically
retrievable — none of which contradict each other. So the report exposes:

    structural_state        - is there usable retrieval evidence at all?
    authority_mode          - what the document CLAIMS (persisted fact, never inferred)
    lifecycle_state         - does that claim still hold?
    canonically_retrievable - can an Agent actually reach it right now?

plus the individual booleans behind `lifecycle_state`, so a dominant condition
never hides a subordinate fact.

WHAT IT REFUSES TO DO
---------------------
It never infers an authority mode. `UNKNOWN` stays `UNKNOWN` even when content
matches, an ingestion marker exists, timestamps look like generation, or exactly
one chunk exists. It never computes an input fingerprint for a generator
identity Core does not support — it reports "unverifiable" instead, because a
fabricated match or mismatch is worse than an honest gap. And it emits facts,
not repair policy: there is deliberately no `canonical_transition_safe` verdict.

SECURITY BOUNDARY
-----------------
This is an **operator** capability, not an Agent one. Retrieval authorization
and operator corpus diagnostics are different concerns with different audiences,
so this service takes no Agent and applies no Agent scope. It must never be
registered as a `ToolDefinition`, auto-resolved into an Agent, exposed through
GAME, or called from a retrieval path.
"""
from django.db.models.functions import Length, Substr

from ai_hub.models import KnowledgeCollection, KnowledgeDocument, KnowledgeDocumentChunk
from ai_hub.services.knowledge_lifecycle import (
    chunk_set_fingerprint,
    current_generator_version,
    document_generation_input_fingerprint,
    is_supported_generator,
)


# Report contract version. V1 (Slice 5) was structural only; V2 adds the
# lifecycle axes. The only consumers of V1 were this repository's own
# management command and tests, so the shape was evolved rather than dual-run.
PREFLIGHT_CONTRACT_VERSION = 2


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------
# Chunk bodies are never returned. A short prefix is read to decide usability;
# full text is read ONLY for documents claiming DERIVED, where a chunk-set
# fingerprint must be computed, and it is discarded immediately after hashing.
CHUNK_PROBE_CHARS = 200

DEFAULT_DOCUMENT_LIMIT = 500
MAX_DOCUMENT_LIMIT = 5000

MAX_ANOMALY_INDEXES = 20


# --------------------------------------------------------------------------
# Axis 1 - structural state
# --------------------------------------------------------------------------
# Purely about retrieval evidence. Carries no authority claim: in V1 the value
# `UNKNOWN_LEGACY` conflated "has both source and chunks" with "authority
# unprovable". Authority is now its own axis, so the structural axis says only
# what it can see. Nothing is lost - `has_source_text` and `authority_mode` are
# both reported, so the V1 census is still derivable.

READY_CANONICAL = "READY_CANONICAL"
SOURCE_WITHOUT_CHUNKS = "SOURCE_WITHOUT_CHUNKS"
UNUSABLE_CHUNKS = "UNUSABLE_CHUNKS"
EMPTY_ACTIVE = "EMPTY_ACTIVE"
NON_ACTIVE = "NON_ACTIVE"

STRUCTURAL_STATES = (
    READY_CANONICAL,
    SOURCE_WITHOUT_CHUNKS,
    UNUSABLE_CHUNKS,
    EMPTY_ACTIVE,
    NON_ACTIVE,
)

STRUCTURAL_STATE_MEANINGS = {
    READY_CANONICAL: (
        "ACTIVE with at least one usable chunk. Structurally retrievable; says "
        "nothing about who is authoritative for that chunk set."
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
# Axis 3 - lifecycle state
# --------------------------------------------------------------------------
# Does the document's persisted authority claim still hold?

UNKNOWN_AUTHORITY = "UNKNOWN_AUTHORITY"
EXPLICIT_AUTHORITY = "EXPLICIT_AUTHORITY"
DERIVED_CURRENT = "DERIVED_CURRENT"
DERIVED_PROVENANCE_INCOMPLETE = "DERIVED_PROVENANCE_INCOMPLETE"
DERIVED_CHUNKS_MODIFIED = "DERIVED_CHUNKS_MODIFIED"
DERIVED_GENERATOR_UNSUPPORTED = "DERIVED_GENERATOR_UNSUPPORTED"
DERIVED_GENERATOR_VERSION_AHEAD = "DERIVED_GENERATOR_VERSION_AHEAD"
DERIVED_INPUT_CHANGED = "DERIVED_INPUT_CHANGED"
GENERATOR_OUTDATED = "GENERATOR_OUTDATED"

LIFECYCLE_STATES = (
    UNKNOWN_AUTHORITY,
    EXPLICIT_AUTHORITY,
    DERIVED_CURRENT,
    DERIVED_PROVENANCE_INCOMPLETE,
    DERIVED_CHUNKS_MODIFIED,
    DERIVED_GENERATOR_UNSUPPORTED,
    DERIVED_GENERATOR_VERSION_AHEAD,
    DERIVED_INPUT_CHANGED,
    GENERATOR_OUTDATED,
)

# Direction of the stored generator version relative to this Core's current one.
# "Differs" is not one fact: behind and ahead have opposite meanings.
VERSION_CURRENT = "current"
VERSION_OUTDATED = "outdated"  # stored < current
VERSION_AHEAD = "ahead"  # stored > current

LIFECYCLE_STATE_MEANINGS = {
    UNKNOWN_AUTHORITY: (
        "Authority is recorded as UNKNOWN. Never inferred otherwise, whatever "
        "the content, markers or timestamps suggest. Not a defect."
    ),
    EXPLICIT_AUTHORITY: (
        "Authority is EXPLICIT: the chunk set is the authored artifact and is "
        "retrieval-authoritative. curated_text differing is irrelevant and is "
        "NOT staleness."
    ),
    DERIVED_CURRENT: (
        "Authority is DERIVED, provenance is complete, the generator identity is "
        "supported at exactly the recorded version, the recorded inputs still "
        "match and the chunk set is untouched since generation. Says nothing "
        "about retrievability, which is a separate axis."
    ),
    DERIVED_PROVENANCE_INCOMPLETE: (
        "Claims DERIVED but the generation facts needed to verify that claim "
        "are missing. The claim cannot be checked, so it cannot be trusted."
    ),
    DERIVED_CHUNKS_MODIFIED: (
        "Claims DERIVED but the chunk set no longer matches the recorded "
        "generated set. Someone edited it outside a governed path, so the "
        "DERIVED marker is stale metadata, not authority. DOMINANT: no "
        "automatic action is safe regardless of the other conditions."
    ),
    DERIVED_GENERATOR_UNSUPPORTED: (
        "Claims DERIVED under a generator identity this Core cannot compute an "
        "input contract for, so input drift is UNVERIFIABLE. Not a claim that "
        "the inputs did or did not change."
    ),
    DERIVED_GENERATOR_VERSION_AHEAD: (
        "Claims DERIVED under a supported generator identity, but at a version "
        "NEWER than the one this Core implements. The chunks were produced by a "
        "generation contract this code does not know, so the current input "
        "contract cannot be assumed to be the one that produced them: input "
        "drift is UNVERIFIABLE. This is the opposite of GENERATOR_OUTDATED and "
        "must never be collapsed into it - the safe response is to upgrade or "
        "escalate, never to regenerate."
    ),
    DERIVED_INPUT_CHANGED: (
        "Claims DERIVED, chunks untouched, but the current generation inputs "
        "(title and/or curated_text) no longer match the recorded ones. "
        "Derivationally stale: regeneration would now produce something else."
    ),
    GENERATOR_OUTDATED: (
        "Claims DERIVED, inputs match and chunks are untouched, but the recorded "
        "generator version is OLDER than the one this Core implements. NOT "
        "source staleness: the chunks still faithfully represent the recorded "
        "inputs under a superseded segmentation policy. Rechunking is "
        "recommended, not required. Applies only to stored < current."
    ),
}


# --------------------------------------------------------------------------
# Issue codes
# --------------------------------------------------------------------------
# `severity` is retained unchanged for KP001-KP006 so their established meaning
# is preserved. It was, however, an ambiguous single axis, so every code now
# also carries two precise dimensions:
#
#   retrieval_impact  - what it means for an Agent retrieving RIGHT NOW
#   lifecycle_impact  - what it means for a FUTURE governed mutation
#
# A document can be perfectly retrievable and still carry a lifecycle warning.

SEVERITY_BLOCKER = "blocker"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

# retrieval_impact
RETRIEVAL_BLOCKS = "blocks_retrieval"
RETRIEVAL_DEGRADES = "degrades_retrieval"
RETRIEVAL_NONE = "none"

# lifecycle_impact
LIFECYCLE_BLOCKS_SAFE_REGENERATION = "blocks_safe_regeneration"
LIFECYCLE_REGENERATION_RECOMMENDED = "regeneration_recommended"
LIFECYCLE_RECHUNK_RECOMMENDED = "rechunk_recommended"
LIFECYCLE_ADVISORY = "advisory"
LIFECYCLE_NONE = "none"

KP001_SOURCE_WITHOUT_CHUNKS = "KP001_SOURCE_WITHOUT_CHUNKS"
KP002_EMPTY_ACTIVE = "KP002_EMPTY_ACTIVE"
KP003_UNUSABLE_CHUNK_CONTENT = "KP003_UNUSABLE_CHUNK_CONTENT"
KP004_UNKNOWN_AUTHORITY = "KP004_UNKNOWN_AUTHORITY"
KP005_SOURCE_FILE_WITHOUT_CHUNKS = "KP005_SOURCE_FILE_WITHOUT_CHUNKS"
KP006_CHUNK_INDEX_ANOMALY = "KP006_CHUNK_INDEX_ANOMALY"
KP007_DERIVED_CHUNKS_MODIFIED = "KP007_DERIVED_CHUNKS_MODIFIED"
KP008_GENERATOR_OUTDATED = "KP008_GENERATOR_OUTDATED"
KP009_DERIVED_INPUT_CHANGED = "KP009_DERIVED_INPUT_CHANGED"
KP010_LIFECYCLE_FACT_INCONSISTENCY = "KP010_LIFECYCLE_FACT_INCONSISTENCY"
KP011_GENERATOR_UNSUPPORTED = "KP011_GENERATOR_UNSUPPORTED"
KP012_GENERATOR_VERSION_AHEAD = "KP012_GENERATOR_VERSION_AHEAD"

ISSUE_CODES = {
    KP001_SOURCE_WITHOUT_CHUNKS: {
        "severity": SEVERITY_BLOCKER,
        "retrieval_impact": RETRIEVAL_BLOCKS,
        "lifecycle_impact": LIFECYCLE_NONE,
        "meaning": (
            "ACTIVE document has source text but no chunks, so canonical "
            "retrieval cannot see it at all."
        ),
    },
    KP002_EMPTY_ACTIVE: {
        "severity": SEVERITY_WARNING,
        "retrieval_impact": RETRIEVAL_BLOCKS,
        "lifecycle_impact": LIFECYCLE_NONE,
        "meaning": (
            "ACTIVE document has neither usable source nor chunks. It is listed "
            "in the Knowledge index but nothing can ever be retrieved from it."
        ),
    },
    KP003_UNUSABLE_CHUNK_CONTENT: {
        "severity": SEVERITY_BLOCKER,
        "retrieval_impact": RETRIEVAL_DEGRADES,
        "lifecycle_impact": LIFECYCLE_NONE,
        "meaning": "One or more chunks carry empty or whitespace-only content.",
    },
    KP004_UNKNOWN_AUTHORITY: {
        "severity": SEVERITY_INFO,
        "retrieval_impact": RETRIEVAL_NONE,
        "lifecycle_impact": LIFECYCLE_BLOCKS_SAFE_REGENERATION,
        "meaning": (
            "A chunk set exists whose authority is recorded as UNKNOWN. This is "
            "informational: it is NOT a defect, NOT drift and NOT staleness. It "
            "does mean no automatic lifecycle action may touch the document."
        ),
    },
    KP005_SOURCE_FILE_WITHOUT_CHUNKS: {
        "severity": SEVERITY_BLOCKER,
        "retrieval_impact": RETRIEVAL_BLOCKS,
        "lifecycle_impact": LIFECYCLE_NONE,
        "meaning": (
            "An uploaded source_file exists but no chunks do. source_file is "
            "not part of the retrieval-first path, so its content is "
            "unreachable until it is chunked."
        ),
    },
    KP006_CHUNK_INDEX_ANOMALY: {
        "severity": SEVERITY_WARNING,
        "retrieval_impact": RETRIEVAL_NONE,
        "lifecycle_impact": LIFECYCLE_ADVISORY,
        "meaning": (
            "Chunk indexes are irregular (do not start at 1, contain gaps, or "
            "include index 0). Duplicates are impossible: a unique constraint "
            "prevents them."
        ),
    },
    KP007_DERIVED_CHUNKS_MODIFIED: {
        "severity": SEVERITY_WARNING,
        "retrieval_impact": RETRIEVAL_NONE,
        "lifecycle_impact": LIFECYCLE_BLOCKS_SAFE_REGENERATION,
        "meaning": (
            "A DERIVED chunk set no longer matches its recorded generation "
            "fingerprint. Retrieval is unaffected, but the DERIVED marker can "
            "no longer authorize overwrite: regenerating would destroy an edit "
            "somebody made."
        ),
    },
    KP008_GENERATOR_OUTDATED: {
        "severity": SEVERITY_INFO,
        "retrieval_impact": RETRIEVAL_NONE,
        "lifecycle_impact": LIFECYCLE_RECHUNK_RECOMMENDED,
        "meaning": (
            "The recorded generator version is older than the current one. The "
            "chunks still faithfully represent the recorded inputs - this is a "
            "segmentation-policy change, NOT source staleness."
        ),
    },
    KP009_DERIVED_INPUT_CHANGED: {
        "severity": SEVERITY_WARNING,
        "retrieval_impact": RETRIEVAL_NONE,
        "lifecycle_impact": LIFECYCLE_REGENERATION_RECOMMENDED,
        "meaning": (
            "A DERIVED document's current generation inputs (title and/or "
            "curated_text) no longer match the recorded ones. The chunks are "
            "derivationally stale."
        ),
    },
    KP010_LIFECYCLE_FACT_INCONSISTENCY: {
        "severity": SEVERITY_WARNING,
        "retrieval_impact": RETRIEVAL_NONE,
        "lifecycle_impact": LIFECYCLE_BLOCKS_SAFE_REGENERATION,
        "meaning": (
            "The persisted lifecycle facts are internally contradictory for the "
            "recorded authority mode. Reported with a machine-readable list of "
            "reasons. Never repaired automatically."
        ),
    },
    KP011_GENERATOR_UNSUPPORTED: {
        "severity": SEVERITY_WARNING,
        "retrieval_impact": RETRIEVAL_NONE,
        "lifecycle_impact": LIFECYCLE_BLOCKS_SAFE_REGENERATION,
        "meaning": (
            "A DERIVED document records a generator identity this Core has no "
            "input contract for, so input drift cannot be verified. This is an "
            "honest capability gap, not a claim that inputs did or did not "
            "change."
        ),
    },
    KP012_GENERATOR_VERSION_AHEAD: {
        "severity": SEVERITY_WARNING,
        "retrieval_impact": RETRIEVAL_NONE,
        "lifecycle_impact": LIFECYCLE_BLOCKS_SAFE_REGENERATION,
        "meaning": (
            "A DERIVED document records a version of a KNOWN generator that is "
            "newer than the one this Core implements - the database has moved "
            "ahead of the code. Distinct from KP011, where the identity itself "
            "is unknown. Retrieval is unaffected, but safe regeneration is not "
            "PROVABLE here: this Core cannot reproduce, or even verify the "
            "inputs of, a contract it does not implement, so regenerating would "
            "silently downgrade the chunk set."
        ),
    },
}

# Machine-readable reasons attached to KP010. One stable code, many reasons -
# rather than a proliferation of near-identical codes.
INCONSISTENCY_DERIVED_WITHOUT_INPUT_FINGERPRINT = "derived_without_input_fingerprint"
INCONSISTENCY_DERIVED_WITHOUT_CHUNK_SET_FINGERPRINT = (
    "derived_without_chunk_set_fingerprint"
)
INCONSISTENCY_DERIVED_WITHOUT_GENERATOR_IDENTITY = "derived_without_generator_identity"
INCONSISTENCY_DERIVED_WITHOUT_GENERATOR_VERSION = "derived_without_generator_version"
INCONSISTENCY_NON_DERIVED_WITH_PROVENANCE = "non_derived_with_generation_provenance"

# Historical ingestion markers. Recognised only as evidence of ORIGIN - never
# promoted to current authority, because they are creation-time provenance and
# are never revisited.
KNOWN_INGESTION_MARKERS = (
    "initial_curated_text",
    "initial_curated_text_backfill",
)


def _chunk_usable(probe: str, content_chars: int) -> bool:
    """Decide whether a chunk carries usable text, without loading it.

    A blank probe on a body longer than the probe is treated as USABLE, because
    a defect has not been proven. Reporting an unprovable defect is worse than
    missing a pathological one.
    """
    if content_chars == 0:
        return False
    if content_chars <= CHUNK_PROBE_CHARS and not probe.strip():
        return False
    return True


def _index_anomalies(indexes: list[int]) -> list[str]:
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
    """One bounded pass over chunks. Never selects a full `content` body."""
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


def _collect_chunk_set_fingerprints(document_ids):
    """Full chunk text, ONLY for documents claiming DERIVED.

    This is the one place full `content` is read, because a chunk-set
    fingerprint cannot be computed from a prefix. Text is hashed and discarded;
    it never reaches the report. Skipped entirely when nothing claims DERIVED,
    which is every corpus until a governed writer exists.
    """
    if not document_ids:
        return {}
    grouped = {}
    rows = (
        KnowledgeDocumentChunk.objects.filter(document_id__in=document_ids)
        .values("document_id", "chunk_index", "section_title", "content")
        .order_by("document_id", "chunk_index")
    )
    for row in rows.iterator():
        grouped.setdefault(row["document_id"], []).append(row)
    return {
        document_id: chunk_set_fingerprint(chunks)
        for document_id, chunks in grouped.items()
    }


def _structural_state(document, facts) -> tuple[str, list[dict]]:
    """Axis 1. Retrieval evidence only; carries no authority claim."""
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

    return READY_CANONICAL, issues


def _lifecycle_inconsistencies(document) -> list[str]:
    """Contradictions in the persisted facts. Observed, never repaired.

    This is the price of having deliberately added no CheckConstraints: the
    preflight becomes the place these are seen.
    """
    reasons = []
    mode = document.chunk_authority_mode
    modes = KnowledgeDocument.ChunkAuthorityMode

    if mode == modes.DERIVED:
        if not document.generation_input_fingerprint:
            reasons.append(INCONSISTENCY_DERIVED_WITHOUT_INPUT_FINGERPRINT)
        if not document.generation_chunk_set_fingerprint:
            reasons.append(INCONSISTENCY_DERIVED_WITHOUT_CHUNK_SET_FINGERPRINT)
        if not document.generator_identity:
            reasons.append(INCONSISTENCY_DERIVED_WITHOUT_GENERATOR_IDENTITY)
        if document.generator_version is None:
            reasons.append(INCONSISTENCY_DERIVED_WITHOUT_GENERATOR_VERSION)
    else:
        has_provenance = any(
            (
                document.generation_input_fingerprint,
                document.generation_chunk_set_fingerprint,
                document.generator_identity,
                document.generator_version is not None,
            )
        )
        if has_provenance:
            reasons.append(INCONSISTENCY_NON_DERIVED_WITH_PROVENANCE)
    return reasons


def _lifecycle_evaluation(document, facts, current_chunk_set_fingerprint):
    """Axes 2 and 3. Returns (lifecycle_state, lifecycle_facts, issues).

    Precedence for a DERIVED claim:

        1. provenance incomplete   - the claim cannot be checked at all
        2. chunks modified         - DOMINANT (Slice 6): the marker is stale
                                     metadata, so nothing else can license action
        3. generator unsupported   - unknown identity; blocks the INPUT axis
        4. generator version ahead - known identity, unknown NEWER contract;
                                     also blocks the INPUT axis
        5. input changed
        6. generator outdated      - known identity, OLDER version only
        7. derived current

    Steps 2 and 3 are ordered this way because the chunk-set contract is
    generator-independent: tampering is detectable even when the input contract
    is not computable, and it is the more serious finding. Slice 6 listed
    "generator unsupported" implicitly under input evaluation; separating it
    keeps "cannot verify" distinct from "verified and differs".

    Step 4 exists because version difference is not one fact. A version BEHIND
    the current one means this Core knows a better segmentation policy - the
    recorded inputs are still verifiable and the finding is advisory. A version
    AHEAD means the row was written by code this Core does not have; its input
    contract may differ, so computing today's input fingerprint would compare
    two different contracts and manufacture a meaningless verdict. The version
    axis is therefore recorded as a DIRECTION, never as a `current` boolean.
    """
    modes = KnowledgeDocument.ChunkAuthorityMode
    mode = document.chunk_authority_mode
    issues = []

    inconsistencies = _lifecycle_inconsistencies(document)
    if inconsistencies:
        issues.append(
            {
                "code": KP010_LIFECYCLE_FACT_INCONSISTENCY,
                "detail": {"reasons": inconsistencies},
            }
        )

    lifecycle = {
        "authority_mode": mode,
        "generator_identity": document.generator_identity or "",
        "generator_version": document.generator_version,
        "generator_supported": None,
        "generator_version_relation": None,
        "recorded_input_fingerprint": document.generation_input_fingerprint or "",
        "current_input_fingerprint": None,
        "input_matches": None,
        "recorded_chunk_set_fingerprint": document.generation_chunk_set_fingerprint or "",
        "current_chunk_set_fingerprint": current_chunk_set_fingerprint,
        "chunk_set_matches": None,
        "provenance_complete": not any(
            reason.startswith("derived_without") for reason in inconsistencies
        ),
        "inconsistencies": inconsistencies,
    }

    # --- UNKNOWN: never infer anything ---------------------------------
    if mode == modes.UNKNOWN:
        if facts["chunk_count"]:
            issues.append({"code": KP004_UNKNOWN_AUTHORITY, "detail": {}})
        return UNKNOWN_AUTHORITY, lifecycle, issues

    # --- EXPLICIT: chunks are authoritative; no derivation evaluation ---
    if mode == modes.EXPLICIT:
        # curated_text differing is irrelevant here and is NOT staleness.
        return EXPLICIT_AUTHORITY, lifecycle, issues

    # --- DERIVED ---------------------------------------------------------
    if not lifecycle["provenance_complete"]:
        return DERIVED_PROVENANCE_INCOMPLETE, lifecycle, issues

    recorded_chunk_set = document.generation_chunk_set_fingerprint
    lifecycle["chunk_set_matches"] = (
        current_chunk_set_fingerprint == recorded_chunk_set
        if current_chunk_set_fingerprint is not None
        else None
    )

    supported = is_supported_generator(document.generator_identity)
    lifecycle["generator_supported"] = supported
    current_version = None
    version_ahead = False
    if supported:
        current_version = current_generator_version(document.generator_identity)
        if document.generator_version == current_version:
            lifecycle["generator_version_relation"] = VERSION_CURRENT
        elif document.generator_version < current_version:
            lifecycle["generator_version_relation"] = VERSION_OUTDATED
        else:
            lifecycle["generator_version_relation"] = VERSION_AHEAD
            version_ahead = True

    # The input fingerprint is computed only when this Core actually implements
    # the contract that produced the row. An unknown identity, or a known
    # identity at an unknown newer version, both leave the input axis honestly
    # unverifiable rather than fabricating a match or a mismatch. The chunk-set
    # contract is generator-independent, so it is still evaluated above.
    if supported and not version_ahead:
        lifecycle["current_input_fingerprint"] = document_generation_input_fingerprint(
            document, document.generator_identity
        )
        lifecycle["input_matches"] = (
            lifecycle["current_input_fingerprint"]
            == document.generation_input_fingerprint
        )

    # Report every applicable condition as an issue, so a dominant primary state
    # never hides a subordinate fact.
    if lifecycle["chunk_set_matches"] is False:
        issues.append(
            {
                "code": KP007_DERIVED_CHUNKS_MODIFIED,
                "detail": {
                    "recorded": recorded_chunk_set,
                    "current": current_chunk_set_fingerprint,
                },
            }
        )
    if not supported:
        issues.append(
            {
                "code": KP011_GENERATOR_UNSUPPORTED,
                "detail": {"generator_identity": document.generator_identity},
            }
        )
    if version_ahead:
        issues.append(
            {
                "code": KP012_GENERATOR_VERSION_AHEAD,
                "detail": {
                    "generator_identity": document.generator_identity,
                    "recorded_version": document.generator_version,
                    "current_version": current_version,
                },
            }
        )
    if lifecycle["input_matches"] is False:
        issues.append(
            {
                "code": KP009_DERIVED_INPUT_CHANGED,
                "detail": {
                    "recorded": document.generation_input_fingerprint,
                    "current": lifecycle["current_input_fingerprint"],
                },
            }
        )
    if lifecycle["generator_version_relation"] == VERSION_OUTDATED:
        issues.append(
            {
                "code": KP008_GENERATOR_OUTDATED,
                "detail": {
                    "recorded_version": document.generator_version,
                    "current_version": current_version,
                },
            }
        )

    # Primary state, by precedence.
    if lifecycle["chunk_set_matches"] is False:
        return DERIVED_CHUNKS_MODIFIED, lifecycle, issues
    if not supported:
        return DERIVED_GENERATOR_UNSUPPORTED, lifecycle, issues
    if version_ahead:
        return DERIVED_GENERATOR_VERSION_AHEAD, lifecycle, issues
    if lifecycle["input_matches"] is False:
        return DERIVED_INPUT_CHANGED, lifecycle, issues
    if lifecycle["generator_version_relation"] == VERSION_OUTDATED:
        return GENERATOR_OUTDATED, lifecycle, issues
    return DERIVED_CURRENT, lifecycle, issues


def run_knowledge_preflight(
    *,
    collection_ids=None,
    statuses=None,
    document_limit: int = DEFAULT_DOCUMENT_LIMIT,
) -> dict:
    """Inspect the Knowledge corpus and report structural and lifecycle facts.

    Read-only. Returns structured data, never formatted strings, and contains no
    timestamp so that repeated runs against unchanged data compare equal.

    There is deliberately no Agent scope: this is an operator diagnostic, not a
    retrieval path.
    """
    modes = KnowledgeDocument.ChunkAuthorityMode
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
    # Full chunk text only where a DERIVED claim must be verified.
    derived_ids = [
        document.pk
        for document in in_scope
        if document.chunk_authority_mode == modes.DERIVED
    ]
    chunk_set_fingerprints = _collect_chunk_set_fingerprints(derived_ids)

    empty_facts = {
        "chunk_count": 0,
        "usable_chunk_count": 0,
        "indexes": [],
        "markers": {},
        "chunks_without_metadata": 0,
    }

    by_status = {value: 0 for value, _label in KnowledgeDocument.Status.choices}
    by_structural_state = {name: 0 for name in STRUCTURAL_STATES}
    by_authority_mode = {value: 0 for value, _label in modes.choices}
    by_lifecycle_state = {name: 0 for name in LIFECYCLE_STATES}
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
        structural_state, structural_issues = _structural_state(document, facts)
        current_chunk_set = chunk_set_fingerprints.get(document.pk)
        if document.chunk_authority_mode == modes.DERIVED and current_chunk_set is None:
            # DERIVED with zero chunks: the empty set still has a fingerprint.
            current_chunk_set = chunk_set_fingerprint([])
        lifecycle_state, lifecycle, lifecycle_issues = _lifecycle_evaluation(
            document, facts, current_chunk_set
        )
        issues = structural_issues + lifecycle_issues

        by_status[document.status] = by_status.get(document.status, 0) + 1
        by_structural_state[structural_state] += 1
        by_authority_mode[document.chunk_authority_mode] = (
            by_authority_mode.get(document.chunk_authority_mode, 0) + 1
        )
        by_lifecycle_state[lifecycle_state] += 1
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
                "by_structural_state": {name: 0 for name in STRUCTURAL_STATES},
                "by_authority_mode": {value: 0 for value, _l in modes.choices},
                "by_lifecycle_state": {name: 0 for name in LIFECYCLE_STATES},
                "issue_counts": {},
                "documents_with_issues": 0,
            },
        )
        collection_row["documents"] += 1
        collection_row["by_structural_state"][structural_state] += 1
        collection_row["by_authority_mode"][document.chunk_authority_mode] += 1
        collection_row["by_lifecycle_state"][lifecycle_state] += 1

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
                    "retrieval_impact": ISSUE_CODES[code]["retrieval_impact"],
                    "lifecycle_impact": ISSUE_CODES[code]["lifecycle_impact"],
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
                    "structural_state": structural_state,
                    "authority_mode": document.chunk_authority_mode,
                    "lifecycle_state": lifecycle_state,
                    "canonically_retrievable": retrievable,
                    "issues": [issue["code"] for issue in issues],
                    "lifecycle": lifecycle,
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
        "contract_version": PREFLIGHT_CONTRACT_VERSION,
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
            "by_structural_state": by_structural_state,
            "by_authority_mode": by_authority_mode,
            "by_lifecycle_state": by_lifecycle_state,
            "issues_by_code": issues_by_code,
            "chunks_total": chunks_total,
            "chunks_usable": chunks_usable_total,
            "chunks_unusable": chunks_total - chunks_usable_total,
            "active_documents": active_documents,
            "active_canonically_retrievable": active_retrievable,
            "active_not_retrievable": active_documents - active_retrievable,
        },
        # Facts an operator needs to judge convergence readiness. Deliberately
        # NOT a policy verdict.
        "convergence": {
            "active_source_without_chunks": by_structural_state[SOURCE_WITHOUT_CHUNKS],
            "active_unusable_chunks": by_structural_state[UNUSABLE_CHUNKS],
            "active_empty": by_structural_state[EMPTY_ACTIVE],
            "unknown_authority": by_authority_mode.get(modes.UNKNOWN, 0),
            "collections_with_blockers": sorted(
                row["collection_id"]
                for row in collection_rows.values()
                if row["documents_with_issues"]
            ),
        },
        # Lifecycle census. Full scope, never affected by document_limit.
        "lifecycle": {
            "unknown": by_authority_mode.get(modes.UNKNOWN, 0),
            "derived": by_authority_mode.get(modes.DERIVED, 0),
            "explicit": by_authority_mode.get(modes.EXPLICIT, 0),
            "derived_current": by_lifecycle_state[DERIVED_CURRENT],
            "explicit_authority": by_lifecycle_state[EXPLICIT_AUTHORITY],
            "derived_input_changed": by_lifecycle_state[DERIVED_INPUT_CHANGED],
            "derived_chunks_modified": by_lifecycle_state[DERIVED_CHUNKS_MODIFIED],
            "generator_outdated": by_lifecycle_state[GENERATOR_OUTDATED],
            "derived_provenance_incomplete": by_lifecycle_state[
                DERIVED_PROVENANCE_INCOMPLETE
            ],
            "derived_generator_version_ahead": by_lifecycle_state[
                DERIVED_GENERATOR_VERSION_AHEAD
            ],
            "derived_generator_unsupported": by_lifecycle_state[
                DERIVED_GENERATOR_UNSUPPORTED
            ],
            "fact_inconsistencies": issues_by_code[KP010_LIFECYCLE_FACT_INCONSISTENCY],
        },
        "collections": [collection_rows[key] for key in sorted(collection_rows)],
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
    lifecycle = report["lifecycle"]
    lines = [
        f"Preflight contract v{report['contract_version']}",
        f"Documents in scope: {summary['documents_in_scope']} "
        f"(of {summary['documents_total']} total)",
        "  by status:  " + ", ".join(
            f"{status}={count}" for status, count in sorted(summary["by_status"].items())
        ),
        f"  ACTIVE retrievable: {summary['active_canonically_retrievable']} / "
        f"{summary['active_documents']}",
        f"  chunks: {summary['chunks_total']} total, "
        f"{summary['chunks_unusable']} unusable",
        "Structural state:",
    ]
    for name in STRUCTURAL_STATES:
        lines.append(f"  {name:<24} {summary['by_structural_state'][name]}")
    lines.append("Authority mode:")
    for value, _label in KnowledgeDocument.ChunkAuthorityMode.choices:
        lines.append(f"  {value:<24} {summary['by_authority_mode'][value]}")
    lines.append("Lifecycle state:")
    for name in LIFECYCLE_STATES:
        count = summary["by_lifecycle_state"][name]
        if count:
            lines.append(f"  {name:<32} {count}")
    lines.append(
        f"  fact inconsistencies             {lifecycle['fact_inconsistencies']}"
    )
    lines.append("Issues:")
    for code in sorted(ISSUE_CODES):
        count = summary["issues_by_code"][code]
        if count:
            spec = ISSUE_CODES[code]
            lines.append(
                f"  [{spec['severity']:<8}] {code:<34} {count}"
                f"  (retrieval: {spec['retrieval_impact']}, "
                f"lifecycle: {spec['lifecycle_impact']})"
            )
    if not any(summary["issues_by_code"].values()):
        lines.append("  none")
    return lines
