"""Safe regeneration of a DERIVED Knowledge chunk set (Slice 11, RC-002 3e).

WHAT THIS IS
------------
One operation-specific Core capability:

    regenerate_derived_chunk_set(document_id, *, expected, principal, reason_code="")

It replaces a DERIVED chunk set with fresh generator output, but **only where
Core can prove it is destroying nothing it did not itself produce**.

This is the first governed operation that can delete retrievable content. Every
lifecycle operation before it was read-only (Preflight V2) or metadata-only
(adjudication). That raises the cost of a boundary failure from "wrong label" to
"content gone", which is why the eligibility proof below is a conjunction with no
policy toggles.

THE SAFETY PROOF
----------------
All four must hold, read from RECORDED facts under the transaction lock:

    1. `chunk_authority_mode == DERIVED`   - a recorded claim, never inferred
    2. provenance is complete              - all four generation facts present
    3. the generator is supported at a version <= the one this Core implements
    4. observed `c1` == recorded `c1`      - the chunks are untouched since
                                             generation

Condition 4 is load-bearing. It is what makes replacement safe rather than
destructive: if the chunk set still hashes to what Core recorded generating,
then Core wrote every byte it is about to replace. The moment that stops being
true - `DERIVED_CHUNKS_MODIFIED` - somebody edited outside governance, and Core
cannot tell whether the edit or the derivation is authoritative. It refuses
rather than guessing that the edit was accidental.

`DERIVED_INPUT_CHANGED` and `GENERATOR_OUTDATED` are the *reasons to* regenerate,
not bars to it: neither disturbs the four conditions.

EXPLICIT AND UNKNOWN NEVER ENTER HERE
-------------------------------------
`EXPLICIT` asserts the chunk set IS the authored artifact; regenerating it is
unconditional data loss. `UNKNOWN` carries no evidence the chunks were ever
generated, and Slice 4 measured that question unanswerable. Both are hard
refusals. In particular this module does **not** provide an alternate route from
`UNKNOWN` to a generated chunk set - that path runs through Slice 10
adjudication, and only through it.

TWO BRANCHES, DECIDED BY `c1` (D-3e-2)
--------------------------------------
    c1(candidate) != c1(current)  -> replace the chunk set; new chunk identities
    c1(candidate) == c1(current)  -> touch NO chunk row; only lifecycle facts move

The second branch matters more than it looks. A generator-version advancement
whose output is unchanged, or a re-run on an already-current document, must not
churn chunk primary keys for nothing: `chunk_id` is a live Agent-facing handle
(`read_knowledge_chunk`, `cite_knowledge_source`) and is embedded in historical
citations. Needless identity churn breaks handles to prove a point.

When `c1` genuinely changes, replacement is correct and the old identities go
with the old artifact. A dangling handle is better than a handle that silently
resolves to different Knowledge.

WHAT THE AUDIT DOES AND DOES NOT PRESERVE
-----------------------------------------
The lifecycle event records the transition truthfully - authority, status, both
recorded fingerprints, both observed fingerprints, generator identity/version and
chunk counts, before and after.

It does **not** preserve the replaced content, and the fingerprints do not
reconstruct it. A fingerprint verifies equality; it stores nothing. Lifecycle
audit is reference-first transition history, not Knowledge-body version storage.
Do not describe it as recoverable.

SECURITY BOUNDARY
-----------------
Operator/system infrastructure. Never an Agent capability: no `ToolDefinition`,
no auto-resolution, no GAME, Admin, Build Console or Orchestrator exposure, no
`save()` hook, no signal, no background trigger. A model must never be able to
regenerate Knowledge or nominate the principal.
"""
from ai_hub.models import KnowledgeDocument, KnowledgeDocumentChunk
from ai_hub.services.knowledge_lifecycle import (
    SUPPORTED_GENERATORS,
    chunk_set_fingerprint,
    document_chunk_set_fingerprint,
    document_generation_input_fingerprint,
    generator_output_projection,
    is_supported_generator,
)
from ai_hub.services.knowledge_mutation import _governed_knowledge_mutation

# Operation slug recorded on the lifecycle event.
OPERATION_REGENERATE_DERIVED_CHUNK_SET = "regenerate_derived_chunk_set"


class KnowledgeRegenerationError(RuntimeError):
    """Regeneration was refused. Nothing was changed."""


class AuthorityNotDerivedError(KnowledgeRegenerationError):
    """Only a DERIVED document may be regenerated.

    `EXPLICIT` is the authored artifact and regenerating it is data loss.
    `UNKNOWN` has no evidence of generation at all - it must go through Slice 10
    adjudication, and this module deliberately offers no shortcut.
    """


class IncompleteProvenanceError(KnowledgeRegenerationError):
    """The DERIVED claim cannot be verified, so it cannot license replacement.

    Without the recorded chunk-set fingerprint there is no way to prove the
    current chunks are the ones Core generated, which is the entire basis for
    being allowed to destroy them.
    """


class UnsupportedGeneratorForRegenerationError(KnowledgeRegenerationError):
    """This Core implements no contract for the recorded generator identity."""


class GeneratorVersionAheadError(KnowledgeRegenerationError):
    """The recorded version is NEWER than the one this Core implements.

    Regenerating would silently downgrade the chunk set to an older contract's
    output. The remedy is to upgrade Core, never to rewrite the data.
    """


class ChunkSetModifiedError(KnowledgeRegenerationError):
    """The chunks no longer match what Core recorded generating.

    Somebody edited them outside the governed boundary. Core cannot tell whether
    the edit or the derivation is authoritative, so it refuses rather than
    destroying an edit on the assumption it was accidental. Resolving this is a
    lifecycle repair decision (step 3e, later slice), not a regeneration.
    """

    def __init__(self, *, document_id, recorded_fingerprint, observed_fingerprint):
        self.document_id = document_id
        self.recorded_fingerprint = recorded_fingerprint
        self.observed_fingerprint = observed_fingerprint
        super().__init__(
            f"Knowledge document #{document_id} has chunks that no longer match "
            f"the recorded generated set (recorded {recorded_fingerprint!r}, "
            f"observed {observed_fingerprint!r}). Regenerating would destroy an "
            "edit Core did not make. The chunks were NOT modified."
        )


class InvalidCandidateError(KnowledgeRegenerationError):
    """The generator produced nothing usable, so there is nothing to install.

    Refused BEFORE any destructive change. Replacing a valid chunk set with an
    empty or malformed one would be a governed way to lose Knowledge.
    """


def _validate_candidate(document_id, candidate):
    """Reject a candidate before it can replace anything."""
    if not candidate:
        raise InvalidCandidateError(
            f"The supported generator produced no chunks for Knowledge document "
            f"#{document_id}; there is nothing to regenerate into."
        )

    indexes = []
    for position, chunk in enumerate(candidate):
        missing = {"chunk_index", "section_title", "content"} - set(chunk)
        if missing:
            raise InvalidCandidateError(
                f"Candidate chunk {position} for document #{document_id} is "
                f"missing {sorted(missing)}."
            )
        index = chunk["chunk_index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise InvalidCandidateError(
                f"Candidate chunk {position} for document #{document_id} has a "
                f"non-integer or negative chunk_index {index!r}."
            )
        if not str(chunk["content"] or "").strip():
            raise InvalidCandidateError(
                f"Candidate chunk {index} for document #{document_id} carries no "
                "usable content."
            )
        indexes.append(index)

    if len(set(indexes)) != len(indexes):
        raise InvalidCandidateError(
            f"Candidate chunk indexes for document #{document_id} are not unique: "
            f"{sorted(indexes)}. A unique constraint would reject them."
        )
    return candidate


def _require_regenerable(document):
    """The four-part safety proof, evaluated against the LOCKED document.

    Returns the generator identity the document is regenerated under - its own
    recorded one, never a substituted default.
    """
    modes = KnowledgeDocument.ChunkAuthorityMode

    if document.chunk_authority_mode != modes.DERIVED:
        raise AuthorityNotDerivedError(
            f"Knowledge document #{document.pk} has authority mode "
            f"{document.chunk_authority_mode!r}. Only {modes.DERIVED!r} may be "
            "regenerated: EXPLICIT is the authored artifact, and UNKNOWN must go "
            "through operator adjudication first."
        )

    identity = document.generator_identity or ""
    # `generator_version` is tested for None, NOT falsiness: version 0 is a real
    # recorded version and must not be mistaken for absent provenance. This is
    # the same None-vs-0 distinction the chunk-index contract enforces, and it is
    # how the preflight already evaluates the same fact.
    missing = sorted(
        [
            name
            for name, value in (
                ("generation_input_fingerprint", document.generation_input_fingerprint),
                ("generation_chunk_set_fingerprint", document.generation_chunk_set_fingerprint),
                ("generator_identity", identity),
            )
            if not value
        ]
        + (["generator_version"] if document.generator_version is None else [])
    )
    if missing:
        raise IncompleteProvenanceError(
            f"Knowledge document #{document.pk} claims DERIVED but is missing "
            f"{', '.join(missing)}. The claim cannot be verified, so it cannot "
            "authorize replacing the chunks."
        )

    if not is_supported_generator(identity):
        raise UnsupportedGeneratorForRegenerationError(
            f"Knowledge document #{document.pk} records generator identity "
            f"{identity!r}, which this Core has no contract for. Supported: "
            f"{sorted(SUPPORTED_GENERATORS)}."
        )

    current_version = SUPPORTED_GENERATORS[identity]
    if document.generator_version > current_version:
        raise GeneratorVersionAheadError(
            f"Knowledge document #{document.pk} records generator {identity!r} at "
            f"version {document.generator_version}, newer than the version "
            f"{current_version} this Core implements. Regenerating would "
            "downgrade the chunk set; upgrade Core instead."
        )

    observed = document_chunk_set_fingerprint(document)
    if observed != document.generation_chunk_set_fingerprint:
        raise ChunkSetModifiedError(
            document_id=document.pk,
            recorded_fingerprint=document.generation_chunk_set_fingerprint,
            observed_fingerprint=observed,
        )

    return identity, current_version, observed


def _replace_chunk_set(document, candidate):
    """Delete-then-insert, inside the caller's transaction.

    Delete-then-insert rather than update-in-place because
    `UniqueConstraint(document, chunk_index)` makes renumbering collide
    mid-statement. Inside `atomic()` the two are equally safe, and this one is
    simpler to reason about.
    """
    document.chunks.all().delete()
    KnowledgeDocumentChunk.objects.bulk_create(
        [
            KnowledgeDocumentChunk(
                document=document,
                chunk_index=chunk["chunk_index"],
                section_title=chunk["section_title"],
                content=chunk["content"],
                token_estimate=max(len(str(chunk["content"]).split()), 1),
                # Deliberately no ingestion marker. The only marker Core writes
                # means "created by the initial curated-text ingestion fallback",
                # which a governed regeneration is not; writing it would be a
                # false provenance claim. Real provenance now lives in the
                # document's lifecycle fields and the audit event, so the marker
                # has no remaining job. `metadata` is excluded from `c1`, so this
                # cannot affect equality either way.
                metadata={},
            )
            for chunk in sorted(candidate, key=lambda c: c["chunk_index"])
        ]
    )


def regenerate_derived_chunk_set(
    document_id, *, expected, principal, reason_code=""
):
    """Regenerate a DERIVED chunk set, safely or not at all.

    `expected` MUST be the state the operator ACTUALLY REVIEWED - not a fresh
    snapshot taken at call time. Compare-and-swap can only detect drift relative
    to the state it is given, so re-snapshotting satisfies it vacuously. That
    obligation matters more here than anywhere previous: this operation deletes
    retrievable content, and the reviewed snapshot is what ties the deletion to
    a decision somebody actually made.

    Eligibility is the four-part proof in `_require_regenerable`, evaluated
    against the locked row. Everything else refuses with a typed error and
    changes nothing.

    There is exactly ONE correctness-authoritative projection, and it happens
    inside the governed transaction after the lock and the compare-and-swap:

        lock + CAS -> read authoritative locked state -> project candidate
                   -> validate candidate -> apply or no-op

    Nothing is projected, read or validated before the transaction opens. An
    earlier revision did that as a "fail-fast" optimisation and it was actively
    harmful: it let a stale reviewed state surface as a candidate/generator
    error instead of `KnowledgeMutationConflict`, it could refuse an operation
    that the locked state would have permitted, and it let candidate failures be
    raised before the transaction ever opened - which silently hollowed out the
    rollback tests that matter most.

    Two outcomes, decided by comparing the candidate against the current chunk
    set under `c1`:

    * **changed** - the chunk set is replaced and chunk identities change with
      it. Live `chunk_id` handles for this document stop resolving; that is the
      intended consequence of the artifact genuinely changing.
    * **unchanged** - no chunk row is touched at all. Primary keys, timestamps,
      content, `metadata` and `token_estimate` all survive. Only lifecycle facts
      move, which is what a version advancement with identical output should
      cost.

    Either way exactly one `KnowledgeLifecycleEvent` is written, in the same
    transaction. Any failure - refusal, invalid candidate, generator exception,
    CAS conflict or audit failure - rolls back the whole thing.

    Returns the reloaded document so the caller sees committed state.
    """
    with _governed_knowledge_mutation(
        document_id,
        expected=expected,
        operation=OPERATION_REGENERATE_DERIVED_CHUNK_SET,
        principal=principal,
        reason_code=reason_code,
    ) as mutation:
        document = mutation.document
        identity, current_version, observed = _require_regenerable(document)

        candidate = _validate_candidate(
            document.pk, generator_output_projection(identity, document)
        )
        candidate_fingerprint = chunk_set_fingerprint(candidate)

        if candidate_fingerprint != observed:
            _replace_chunk_set(document, candidate)

        document.generation_input_fingerprint = document_generation_input_fingerprint(
            document, identity
        )
        document.generation_chunk_set_fingerprint = candidate_fingerprint
        document.generator_identity = identity
        document.generator_version = current_version
        document.save(
            update_fields=[
                "generation_input_fingerprint",
                "generation_chunk_set_fingerprint",
                "generator_identity",
                "generator_version",
            ]
        )

    return KnowledgeDocument.objects.get(pk=document_id)
