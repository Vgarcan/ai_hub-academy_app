"""Deterministic Knowledge lifecycle primitives (Slice 7, corrected).

This module persists nothing and mutates nothing. It provides the pure,
deterministic facts the lifecycle contract needs:

* a **generation-input fingerprint** answering "are these still the inputs the
  generator was given?";
* a **chunk-set fingerprint** answering "are these still the chunks that
  generator produced?";
* the identity and version of the one generation strategy Core can truthfully
  support today.

WHY BOTH FINGERPRINTS ARE REQUIRED
----------------------------------
Every current Knowledge write path is a raw Django ModelForm save or a raw ORM
call, so nothing can express intent and a persisted ``DERIVED`` marker can go
stale the moment someone hand-edits a chunk. **A persisted mode must therefore
never, by itself, authorize overwrite.** Safe regeneration eventually requires
proving three things: the set was generated; the current generation inputs match
the recorded ones; and the current chunks still match the recorded generated
set. The chunk-set fingerprint is the tamper-evidence for the third.

Both exist for chunk-lifecycle correctness now, not for future embeddings.

WHY "INPUT" AND NOT "SOURCE"
----------------------------
``curated_text_single_chunk`` produces::

    chunk_index   = 1
    section_title = document.title
    content       = curated_text.strip()

so **``document.title`` is a generation input too**. A fingerprint covering only
``curated_text`` could not detect a title change, and regeneration would then
silently produce a different ``section_title`` while the fingerprint still
claimed a match. The contract therefore covers the complete mutable input set
for a specific generator, and is named accordingly.

``i1`` is **not** a universal source-content contract. It is the V1
generation-input contract for the ``curated_text_single_chunk`` shape. A future
generator whose output depends on different inputs must define its own
versioned input contract. The API is deliberately shaped to make that confusion
hard: the fingerprint is reached through a generator identity, and an unknown
identity raises rather than guessing.

CONTRACT VERSIONING
-------------------
Both fingerprints are **self-describing**: the serialized value carries its own
contract version (``i1:``/``c1:``), and the contract name is also inside the
hashed payload, so two contracts can never produce the same digest for different
meanings and an old value can never be silently compared under new rules. A
naked digest is never stored.

The two contracts are versioned **independently**.

NOT IN SCOPE
------------
No regeneration, repair, backfill, adjudication, governed mutation or audit
persistence. Nothing in the runtime writes these facts.
"""
import hashlib
import json
import unicodedata


# --------------------------------------------------------------------------
# Contract versions
# --------------------------------------------------------------------------
# Bump when the canonicalization or hash changes. Never reuse a version.
GENERATION_INPUT_FINGERPRINT_CONTRACT = "i1"
CHUNK_SET_FINGERPRINT_CONTRACT = "c1"


# --------------------------------------------------------------------------
# Generation strategies
# --------------------------------------------------------------------------
# The only strategy Core can truthfully support today: one chunk built from
# curated_text with the document title as its section title, which is exactly
# what `ensure_initial_knowledge_chunk()` does.
#
# `source_file` is deliberately absent. Core has no parser pipeline and does not
# read uploaded files on the retrieval-first path, so a file-only document
# cannot be DERIVED in V1. Claiming otherwise would fabricate provenance about
# content the system has never parsed.
GENERATOR_CURATED_TEXT_SINGLE_CHUNK = "curated_text_single_chunk"
GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION = 1

# Deliberately a small mapping, not a registry or plugin architecture.
SUPPORTED_GENERATORS = {
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK: GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION,
}

# Which input contract each supported generator's inputs are fingerprinted under.
GENERATOR_INPUT_CONTRACTS = {
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK: GENERATION_INPUT_FINGERPRINT_CONTRACT,
}


class UnsupportedGeneratorError(ValueError):
    """Raised when a generator identity has no input contract in this Core.

    Callers must report "unverifiable" rather than guess. Computing some other
    contract's fingerprint and comparing it would manufacture a match or a
    mismatch out of nothing.
    """


class MissingChunkIndexError(ValueError):
    """Raised when a chunk supplied for fingerprinting has no ``chunk_index``.

    Fingerprinting represents exactly what it is given. It does not invent an
    index, and it does not decide whether an index is a valid lifecycle state —
    the preflight owns structural anomaly reporting.
    """


def current_generator_version(identity: str) -> int | None:
    """Current rule version for a known strategy, else None."""
    return SUPPORTED_GENERATORS.get(identity)


def is_supported_generator(identity: str) -> bool:
    return identity in SUPPORTED_GENERATORS


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def _normalize_line_endings(value: str) -> str:
    """CRLF and lone CR become LF. A Windows/Unix round-trip is not an edit."""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_representation(value) -> str:
    """Representation-safe only: text, CRLF/CR -> LF, Unicode NFC.

    No stripping, no collapsing, no case folding.
    """
    text = "" if value is None else str(value)
    return unicodedata.normalize("NFC", _normalize_line_endings(text))


def normalize_title_text(value) -> str:
    """Canonicalize a document title as a generation input.

    **Not stripped**: the generator uses ``document.title`` exactly as
    persisted, so leading or trailing whitespace in the title genuinely changes
    the produced ``section_title``.
    """
    return _normalize_representation(value)


def normalize_curated_text(value) -> str:
    """Canonicalize ``curated_text`` as a generation input.

    **Outer whitespace stripped**, because the generator uses
    ``curated_text.strip()`` — surrounding whitespace cannot change what it
    produces, so treating it as a change would report drift where none exists.

    Internal whitespace and case are preserved: paragraph and indentation
    structure is chunking-relevant, and aggressive normalization would let
    materially different inputs hash identically.
    """
    return _normalize_representation(value).strip()


def normalize_chunk_text(value) -> str:
    """Canonicalize persisted chunk text for fingerprinting.

    Representation-safe only. **Whitespace is NOT stripped or collapsed.** An
    ungoverned whitespace edit to persisted chunk content is still an edit, and
    the chunk-set fingerprint exists to detect edits. The asymmetry with
    `normalize_curated_text()` is deliberate: one describes an input the
    generator will strip anyway, the other describes stored evidence.
    """
    return _normalize_representation(value)


def _digest(payload: dict) -> str:
    """Canonical JSON -> UTF-8 -> SHA-256 hex.

    JSON rather than delimiter concatenation: escaping makes the serialization
    unambiguous, so a value containing a separator cannot forge another field or
    chunk boundary. `sort_keys` and `ensure_ascii` remove every remaining degree
    of freedom (and make the encode step platform-independent), and `separators`
    removes incidental whitespace.
    """
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Generation-input fingerprint  (contract i1)
# --------------------------------------------------------------------------


def curated_text_single_chunk_input_fingerprint(*, title, curated_text) -> str:
    """`i1` fingerprint of the complete mutable input set for one generator.

    Covers **both** inputs `curated_text_single_chunk` reads:

    * ``title``        -> becomes ``section_title``; normalized but NOT stripped;
    * ``curated_text`` -> becomes ``content``;       normalized AND stripped.

    Returns ``"i1:<sha256 hex>"``.

    Named after the generator, and keyword-only, so it is hard to mistake for a
    universal source-content fingerprint.
    """
    payload = {
        "contract": GENERATION_INPUT_FINGERPRINT_CONTRACT,
        "title": normalize_title_text(title),
        "curated_text": normalize_curated_text(curated_text),
    }
    return f"{GENERATION_INPUT_FINGERPRINT_CONTRACT}:{_digest(payload)}"


_INPUT_FINGERPRINT_BUILDERS = {
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK: curated_text_single_chunk_input_fingerprint,
}


def generation_input_fingerprint(identity: str, *, title, curated_text) -> str:
    """Input fingerprint for a named generator identity.

    Raises `UnsupportedGeneratorError` for an unknown identity rather than
    falling back to some other contract — a fabricated match or mismatch is
    worse than an honest "cannot verify".
    """
    builder = _INPUT_FINGERPRINT_BUILDERS.get(identity)
    if builder is None:
        raise UnsupportedGeneratorError(
            f"No generation-input contract for generator identity {identity!r}. "
            f"Supported: {sorted(_INPUT_FINGERPRINT_BUILDERS)}."
        )
    return builder(title=title, curated_text=curated_text)


# --------------------------------------------------------------------------
# Generator output projection
# --------------------------------------------------------------------------
# What a supported generator WOULD produce from a document's current inputs,
# computed without writing anything. Added in Slice 10 so an operator decision
# can ask "is this chunk set already exactly what the generator produces?"
# without running the generator, and therefore without touching the chunks.
#
# This lives beside the input contract on purpose: "what generator X reads" and
# "what generator X emits" are the same piece of knowledge, and splitting them
# across modules is how they drift apart.
#
# CRITICAL: this is a MODEL of `ai_hub.services.knowledge_ingestion.
# ensure_initial_knowledge_chunk`, not the generator itself. The real writer is
# deliberately untouched (it remains an UNKNOWN-producing compatibility path).
# The two are bound by test, not by hope - see
# `test_knowledge_adjudication.ProjectionMatchesRealGeneratorTests`, which runs
# the real writer and compares.


def curated_text_single_chunk_projection(document) -> list[dict]:
    """Chunk set `curated_text_single_chunk` v1 would produce. Pure; no I/O.

    Mirrors the real writer: one chunk at index 1, `section_title` = the title
    verbatim, `content` = `curated_text` stripped. Returns ``[]`` when the
    generator would produce nothing, which is what the writer does for empty
    curated text.

    Only the fields the `c1` chunk-set contract covers are projected.
    `token_estimate` and `metadata` are deliberately absent: `c1` excludes them,
    so including them here would impose a stricter notion of chunk-set identity
    than the lifecycle contract itself uses.
    """
    content = str(getattr(document, "curated_text", "") or "").strip()
    if not content:
        return []
    return [
        {
            "chunk_index": 1,
            "section_title": document.title,
            "content": content,
        }
    ]


_OUTPUT_PROJECTIONS = {
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK: curated_text_single_chunk_projection,
}


def generator_output_projection(identity: str, document) -> list[dict]:
    """Projected chunk set for a named generator identity.

    Raises `UnsupportedGeneratorError` for an unknown identity rather than
    guessing, for the same reason the input contract does: a fabricated match is
    worse than an honest "cannot compute".
    """
    projection = _OUTPUT_PROJECTIONS.get(identity)
    if projection is None:
        raise UnsupportedGeneratorError(
            f"No output projection for generator identity {identity!r}. "
            f"Supported: {sorted(_OUTPUT_PROJECTIONS)}."
        )
    return projection(document)


def document_generation_input_fingerprint(document, identity: str) -> str:
    """Current input fingerprint for a document under a named generator."""
    return generation_input_fingerprint(
        identity,
        title=getattr(document, "title", ""),
        curated_text=getattr(document, "curated_text", ""),
    )


# --------------------------------------------------------------------------
# Chunk-set fingerprint  (contract c1)
# --------------------------------------------------------------------------
# Included, because they define the retrieval evidence:
#     chunk_index, section_title, content
#
# Excluded, because they do not and would raise false tamper signals:
#     primary keys, document id, timestamps, token_estimate, metadata,
#     ingestion markers, audit information.

_MISSING = object()


def _chunk_fields(chunk) -> tuple[int, str, str]:
    """Read the three retrieval-relevant fields from a model or a mapping.

    ``chunk_index`` is required and must not be None. A missing index is an
    error, never a silent zero: ``None`` and ``0`` are different inputs and must
    not fingerprint identically. A real ``0`` is represented faithfully as ``0``
    — deciding whether index 0 is a valid lifecycle state belongs to the
    preflight, not to fingerprinting.
    """
    if isinstance(chunk, dict):
        index = chunk.get("chunk_index", _MISSING)
        section_title = chunk.get("section_title")
        content = chunk.get("content")
    else:
        index = getattr(chunk, "chunk_index", _MISSING)
        section_title = getattr(chunk, "section_title", None)
        content = getattr(chunk, "content", None)

    if index is _MISSING or index is None:
        raise MissingChunkIndexError(
            "chunk_index is required for chunk-set fingerprinting; "
            "None or a missing key is not the same input as 0."
        )

    return (
        int(index),
        normalize_chunk_text(section_title),
        normalize_chunk_text(content),
    )


def chunk_set_fingerprint(chunks) -> str:
    """Versioned fingerprint of a retrieval chunk set.

    Returns ``"c1:<sha256 hex>"``.

    The set is ordered by ``chunk_index``, so the caller's Python ordering is
    irrelevant while a genuine reindex is detected. The canonical form naturally
    encodes chunk count, index changes, reordering, section-title changes and
    content changes.

    Accepts model instances, dicts, or any object exposing the three fields — so
    a proposed replacement set can be fingerprinted before it is persisted.
    """
    entries = sorted((_chunk_fields(chunk) for chunk in chunks), key=lambda row: row[0])
    payload = {
        "contract": CHUNK_SET_FINGERPRINT_CONTRACT,
        "chunks": [
            {"chunk_index": index, "section_title": section_title, "content": content}
            for index, section_title, content in entries
        ],
    }
    return f"{CHUNK_SET_FINGERPRINT_CONTRACT}:{_digest(payload)}"


def document_chunk_set_fingerprint(document) -> str:
    """Fingerprint of a document's currently persisted chunk set."""
    return chunk_set_fingerprint(
        document.chunks.order_by("chunk_index").only(
            "chunk_index", "section_title", "content"
        )
    )


# --------------------------------------------------------------------------
# Contract inspection
# --------------------------------------------------------------------------


def fingerprint_contract(value) -> str:
    """Contract version of a stored fingerprint, or "" if absent/unversioned.

    Lets future code refuse to compare values written under different contracts
    instead of comparing them incorrectly.
    """
    text = str(value or "")
    prefix, separator, _digest_part = text.partition(":")
    return prefix if separator else ""
