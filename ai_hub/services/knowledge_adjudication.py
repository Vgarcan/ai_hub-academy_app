"""Operator adjudication of UNKNOWN Knowledge authority (Slice 10, RC-002 3d).

WHAT THIS IS
------------
The first consumers of the Slice 9 governed mutation boundary, and the first
real lifecycle operations in Core. Two of them, both operator-initiated, both
draining the `UNKNOWN` backlog that Slice 7's migration created:

    adjudicate_unknown_as_explicit(...)   UNKNOWN -> EXPLICIT
    adopt_unknown_as_derived(...)         UNKNOWN -> DERIVED  (forward only)

Each is a narrow, operation-specific public verb, which is exactly what Slice 9
said would replace the generic mutator it kept private. Neither duplicates
transaction, locking, compare-and-swap or audit logic: all of that comes from
`_governed_knowledge_mutation`, so there is one governed-write path, not two.

THE TWO DECISIONS ARE NOT SYMMETRIC
-----------------------------------
`UNKNOWN -> EXPLICIT` is a **declaration**. An operator asserts that the chunk
set is the authored artifact and is authoritative for retrieval. No evidence can
prove that - authorship is a fact about intent, not about bytes - so Core does
not try. It records who decided, and when.

`UNKNOWN -> DERIVED` is a **claim about history**, and Slice 4 measured that
history to be unrecoverable: ingestion markers are creation-time only and
written by 2 of 3 derived paths, marker absence is uninformative, `section_title
== title` breaks on rename, content equality is direction-blind, and timestamps
fail in both directions. So this operation **never guesses the past**. It is
FORWARD ADOPTION only: it asks whether the chunk set that exists *right now* is
exactly what the supported generator would produce from the inputs that exist
*right now*. If yes, DERIVED becomes true from this moment forward, and the
lifecycle event marks that moment. If no, it is REJECTED - never repaired.

WHAT ADOPTION DOES NOT DO
-------------------------
It does not regenerate. It does not modify, reorder, delete or create a single
chunk. A successful adoption writes only lifecycle fields on the document; the
chunk rows are byte-identical BEFORE AND AFTER THE OPERATION, which is asserted
by test. An adoption that would need to change a chunk to succeed is not an
adoption, it is a repair, and repair is step 3e.

Do not read that as "the chunks are byte-identical to generator output". Those
are different claims. Adoption's precondition is equality under the `c1`
contract, which normalizes and excludes several things - see "WHAT 'EXACTLY
REPRODUCIBLE' MEANS HERE" in `adopt_unknown_as_derived`.

WHAT THE EVENT MEANS
--------------------
`KnowledgeLifecycleEvent.created_at` is the point from which DERIVED authority
is **known**, not a claim about when the chunks were generated. Core has no
evidence of the latter and does not pretend otherwise. The event's
`previous_authority_mode` is `unknown` precisely because that was the truth
until the operator decided.

THE `expected` SNAPSHOT IS A CALLER OBLIGATION
----------------------------------------------
Both operations take an `ExpectedKnowledgeState`. It MUST represent the state
the human operator **actually reviewed**. A caller MUST NOT build a fresh
`ExpectedKnowledgeState.from_snapshot(build_snapshot(document))` immediately
before calling merely to satisfy the API.

Why this matters, precisely:

* compare-and-swap can only detect changes **relative to the expected state it
  is given**. It has no independent notion of "what the operator saw".
* Re-snapshotting at call time therefore compares current state against current
  state. The CAS passes **vacuously** - it is not weakened, it is answered with
  the wrong question.
* For `UNKNOWN -> EXPLICIT` this is the whole protection. That transition is
  deliberately a **declaration**, not a reproducibility-tested inference, so
  there is no second check behind the CAS. A caller that re-snapshots will
  successfully grant EXPLICIT authority over content the operator never saw.
  `UNKNOWN -> DERIVED` happens to have accidental cover, because the
  reproducibility test still runs - but relying on that is relying on an
  accident of one operation.

This is a **caller / operator-surface responsibility, by design.** The service
deliberately does NOT take its own snapshot on the caller's behalf: doing so
would silently guarantee the vacuous comparison for every caller, and would hide
the obligation instead of stating it.

A future operator surface MUST capture the snapshot at the moment the operator
reviews the document, hold it across the review, and submit **that same**
expected state when the operator confirms. If the document moved in between, the
CAS is supposed to reject the decision - that is the feature, not a failure.

SECURITY BOUNDARY
-----------------
Operator/system capability. Never an Agent one: no `ToolDefinition`, no
auto-resolution, no GAME, Admin or Orchestrator exposure. A model must never be
able to adjudicate authority or nominate the principal. There is no Admin UI and
no Build Console integration in this slice - deliberately, because an
adjudication surface is a design question of its own.
"""
from ai_hub.models import KnowledgeDocument
from ai_hub.services.knowledge_lifecycle import (
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK,
    SUPPORTED_GENERATORS,
    chunk_set_fingerprint,
    document_chunk_set_fingerprint,
    document_generation_input_fingerprint,
    generator_output_projection,
)
from ai_hub.services.knowledge_mutation import _governed_knowledge_mutation

# Operation slugs recorded on the lifecycle event. Validated by the mutation
# foundation against its slug pattern; named for what an operator decided, not
# for what the code did.
OPERATION_ADJUDICATE_UNKNOWN_AS_EXPLICIT = "adjudicate_unknown_as_explicit"
OPERATION_ADOPT_UNKNOWN_AS_DERIVED = "adopt_unknown_as_derived"

# The generator an adoption is checked against. Pinned to the one contract this
# Core implements; `SUPPORTED_GENERATORS` supplies its current version.
ADOPTION_GENERATOR = GENERATOR_CURATED_TEXT_SINGLE_CHUNK


class KnowledgeAdjudicationError(RuntimeError):
    """A lifecycle adjudication was refused. Nothing was changed."""


class AuthorityNotUnknownError(KnowledgeAdjudicationError):
    """The document is not `UNKNOWN`, so there is nothing to adjudicate.

    Adjudication drains the legacy backlog. Re-deciding authority for a document
    that already has some is a different operation with different hazards -
    notably DERIVED -> EXPLICIT operator takeover - and is not implemented.
    """


class MissingChunkSetError(KnowledgeAdjudicationError):
    """`EXPLICIT` means the chunk set IS the authored artifact.

    A document with no chunks has no artifact to declare authoritative, so the
    declaration would be vacuous and would block later derived generation.

    Note this checks EXISTENCE, not usability. Whether the chunks carry usable
    text is the STRUCTURAL axis (Preflight `KP003`/`UNUSABLE_CHUNKS`), which
    Slice 8 deliberately kept orthogonal to authority. Refusing to record a
    truthful authority fact because a chunk is blank would collapse two axes
    that were separated on purpose.
    """


class UnexpectedProvenanceError(KnowledgeAdjudicationError):
    """The `UNKNOWN` document already carries generation provenance.

    That combination is a recorded inconsistency (Preflight `KP010`,
    `non_derived_with_generation_provenance`), not a normal legacy row.
    Adjudicating it either way would overwrite or entrench evidence of the
    inconsistency, so both operations refuse. Resolving contradictory lifecycle
    facts is repair, and repair is step 3e.
    """


class ChunkSetNotReproducibleError(KnowledgeAdjudicationError):
    """The chunk set is not what the supported generator would produce now.

    Carries the two fingerprints so an operator can see the disagreement.

    This is a REJECTION, never a repair instruction. The chunk set may be
    perfectly good - hand-improved, differently segmented, or produced by a
    generator this Core does not implement. Core simply cannot call it DERIVED,
    because DERIVED asserts a reproducibility relationship that does not hold,
    and acting on that assertion later would destroy someone's work.
    """

    def __init__(self, *, document_id, expected_fingerprint, actual_fingerprint, reason):
        self.document_id = document_id
        self.expected_fingerprint = expected_fingerprint
        self.actual_fingerprint = actual_fingerprint
        self.reason = reason
        super().__init__(
            f"Knowledge document #{document_id} cannot be adopted as DERIVED: "
            f"{reason} Generator would produce {expected_fingerprint!r}; the "
            f"current chunk set is {actual_fingerprint!r}. The chunks were NOT "
            "modified. Adopt only a chunk set the generator already reproduces."
        )


class EmptyGeneratorOutputError(ChunkSetNotReproducibleError):
    """The generator would produce nothing from the current inputs.

    An empty projection trivially "matches" an empty chunk set, which would let
    a document with no source and no chunks be adopted as DERIVED - a document
    claiming derived authority over nothing. Refused explicitly rather than
    allowed to fall out of a fingerprint comparison.
    """


def _require_unknown_and_unprovenanced(document):
    """Shared preconditions. Evaluated inside the governed transaction."""
    modes = KnowledgeDocument.ChunkAuthorityMode
    if document.chunk_authority_mode != modes.UNKNOWN:
        raise AuthorityNotUnknownError(
            f"Knowledge document #{document.pk} has authority mode "
            f"{document.chunk_authority_mode!r}, not {modes.UNKNOWN!r}. "
            "Adjudication applies only to the legacy UNKNOWN backlog."
        )
    carried = {
        "generation_input_fingerprint": document.generation_input_fingerprint,
        "generation_chunk_set_fingerprint": document.generation_chunk_set_fingerprint,
        "generator_identity": document.generator_identity,
        "generator_version": document.generator_version,
    }
    present = sorted(name for name, value in carried.items() if value)
    if present:
        raise UnexpectedProvenanceError(
            f"Knowledge document #{document.pk} is UNKNOWN but already carries "
            f"generation provenance ({', '.join(present)}). That is a recorded "
            "lifecycle inconsistency, not a legacy row; resolve it through "
            "repair rather than adjudication."
        )


def adjudicate_unknown_as_explicit(
    document_id, *, expected, principal, reason_code=""
):
    """Record an operator's decision that the chunk set is authoritative.

    A DECLARATION, not an inference. Core cannot prove authorship and does not
    try; it records who decided and when. Nothing about the chunks or the
    document's content is changed - only `chunk_authority_mode`.

    `expected` MUST be the state the operator ACTUALLY REVIEWED - not a fresh
    snapshot taken at call time. See "THE `expected` SNAPSHOT IS A CALLER
    OBLIGATION" in the module docstring. This operation is a declaration with no
    reproducibility test behind it, so the reviewed snapshot is the ONLY thing
    preventing authority being granted over content the operator never saw. A
    caller that re-snapshots satisfies the compare-and-swap vacuously.

    If the document changed since review, the Slice 9 boundary raises
    `KnowledgeMutationConflict` and nothing is applied.

    Raises `AuthorityNotUnknownError`, `MissingChunkSetError` or
    `UnexpectedProvenanceError`; every one of them leaves zero changes and zero
    lifecycle events.
    """
    modes = KnowledgeDocument.ChunkAuthorityMode
    with _governed_knowledge_mutation(
        document_id,
        expected=expected,
        operation=OPERATION_ADJUDICATE_UNKNOWN_AS_EXPLICIT,
        principal=principal,
        reason_code=reason_code,
    ) as mutation:
        document = mutation.document
        _require_unknown_and_unprovenanced(document)

        if not document.chunks.exists():
            raise MissingChunkSetError(
                f"Knowledge document #{document.pk} has no chunks, so there is "
                "no authored artifact to declare authoritative."
            )

        document.chunk_authority_mode = modes.EXPLICIT
        document.save(update_fields=["chunk_authority_mode"])

    return _reload(document_id)


def adopt_unknown_as_derived(document_id, *, expected, principal, reason_code=""):
    """Adopt DERIVED authority **forward** when the chunks already reproduce.

    `expected` MUST be the state the operator ACTUALLY REVIEWED - not a fresh
    snapshot taken at call time. See "THE `expected` SNAPSHOT IS A CALLER
    OBLIGATION" in the module docstring.

    Never a guess about history. The test is entirely present-tense: would the
    supported generator, run against the inputs as they are now, produce a chunk
    set EQUAL UNDER `c1` to the one that exists now?

    WHAT "EXACTLY REPRODUCIBLE" MEANS HERE
    --------------------------------------
    Equal under the **`c1` chunk-set contract** - NOT raw-byte-identical text.
    `c1` covers `chunk_index`, `section_title` and `content`, each normalized,
    so adoption currently tolerates:

        * `token_estimate` differences        (excluded from `c1`)
        * `metadata` differences              (excluded from `c1`)
        * line-ending differences             (CRLF/CR normalized to LF)
        * Unicode normalization differences   (normalized to NFC)

    That is deliberate, and the reason is consistency rather than leniency: the
    equality contract used to ADOPT must be the one used later to JUDGE. `c1` is
    what Preflight V2 and any future regeneration evaluate against, so a document
    adopted under `c1` is immediately `DERIVED_CURRENT`. A stricter comparison
    here would let a document be adopted under one notion of sameness and judged
    under another - the exact incoherence this contract exists to prevent.

    Two statements that must not be conflated:

        * "adoption does not modify the chunks" - TRUE, unconditionally. The
          success path writes five document fields and touches no chunk row.
        * "the stored chunks are byte-identical to generator output" - NOT
          implied. They are equal under `c1`. A later governed regeneration
          could therefore rewrite raw bytes (line endings, Unicode form,
          `token_estimate`, `metadata`) while `c1` reports no chunk-set change
          throughout.

    On success it writes ONLY lifecycle fields. The chunk rows are untouched:
    not regenerated, not normalized, not reordered. The event records the moment
    DERIVED became *known*, and claims nothing about when the chunks were made.

    On any mismatch it raises `ChunkSetNotReproducibleError` and changes nothing.
    It does not repair the difference, and it does not tell the caller how to -
    that is step 3e.
    """
    modes = KnowledgeDocument.ChunkAuthorityMode
    with _governed_knowledge_mutation(
        document_id,
        expected=expected,
        operation=OPERATION_ADOPT_UNKNOWN_AS_DERIVED,
        principal=principal,
        reason_code=reason_code,
    ) as mutation:
        document = mutation.document
        _require_unknown_and_unprovenanced(document)

        projected = generator_output_projection(ADOPTION_GENERATOR, document)
        projected_fingerprint = chunk_set_fingerprint(projected)
        actual_fingerprint = document_chunk_set_fingerprint(document)

        if not projected:
            raise EmptyGeneratorOutputError(
                document_id=document.pk,
                expected_fingerprint=projected_fingerprint,
                actual_fingerprint=actual_fingerprint,
                reason=(
                    "the generator would produce no chunks at all from the "
                    "current inputs, so there is no derivation to adopt."
                ),
            )

        if projected_fingerprint != actual_fingerprint:
            raise ChunkSetNotReproducibleError(
                document_id=document.pk,
                expected_fingerprint=projected_fingerprint,
                actual_fingerprint=actual_fingerprint,
                reason=(
                    "the current chunk set is not what the supported generator "
                    "produces from the current inputs."
                ),
            )

        # Reproducibility holds. Record it - and ONLY it.
        document.chunk_authority_mode = modes.DERIVED
        document.generation_input_fingerprint = document_generation_input_fingerprint(
            document, ADOPTION_GENERATOR
        )
        document.generation_chunk_set_fingerprint = actual_fingerprint
        document.generator_identity = ADOPTION_GENERATOR
        document.generator_version = SUPPORTED_GENERATORS[ADOPTION_GENERATOR]
        document.save(
            update_fields=[
                "chunk_authority_mode",
                "generation_input_fingerprint",
                "generation_chunk_set_fingerprint",
                "generator_identity",
                "generator_version",
            ]
        )

    return _reload(document_id)


def _reload(document_id):
    return KnowledgeDocument.objects.get(pk=document_id)
