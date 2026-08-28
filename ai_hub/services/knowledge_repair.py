"""Narrow governed Knowledge repair decisions (Slice 12, RC-002 3e).

WHAT THIS IS
------------
Two operation-specific Core capabilities, both **human decisions**, both
resolving lifecycle states Core deliberately refuses to resolve on its own:

    accept_current_chunks_as_explicit(...)       authority transfer
    discard_modified_chunks_and_regenerate(...)  authorized destruction

WHY THESE TWO, AND ONLY THESE TWO
---------------------------------
`DERIVED_CHUNKS_MODIFIED` means the recorded generated fingerprint and the
observed one disagree: somebody edited the chunks outside the governed boundary.
Core cannot tell whether that edit was deliberate authorship or accidental
corruption - Slice 4 measured that this class of question is unanswerable from
stored evidence - so Slice 11 correctly refuses to act.

Two opposite outcomes are both truthful, and which one is right is a fact about
human intent, not about bytes. That is what makes this the one lifecycle state
needing a *pair* of verbs. With only "accept", an operator who knows the edit was
corruption would be coerced into blessing it to escape the state. With only
"discard", deliberate authorship would be destroyed. Neither alone is a decision.

REPAIR IS NOT NORMALIZATION
---------------------------
Neither operation exists to make Preflight green. A lifecycle inconsistency can
be the most truthful description available, and this module leaves several
standing on purpose:

* `KP010` (UNKNOWN or EXPLICIT carrying generation provenance) is **not**
  repaired. The provenance is anomalous but not proven false, and it may be the
  only surviving trace of how a document really came to be. It costs nothing
  operationally (`retrieval_impact: none`) and is worth more as forensic
  evidence than as a silenced warning.
* `DERIVED_GENERATOR_UNSUPPORTED` and `DERIVED_GENERATOR_VERSION_AHEAD` get no
  exit here **on their own**. Those are defects in *this Core's capability*, not
  in the data: the persisted claim may be perfectly valid and simply unreadable
  by the running code. Offering a repair verb for them would invite an operator
  on an old binary to erase true provenance. The remedy is a Core that
  understands the generator.

  But `DERIVED_CHUNKS_MODIFIED` is **dominant**, exactly as Preflight documents
  it, and that dominance holds here too. Once the current chunks no longer match
  the recorded generated set, the DERIVED claim no longer describes the current
  artifact **regardless of whether this Core understands the historic
  generator** - so an operator may accept the reviewed current chunks as
  EXPLICIT. See the accept/discard asymmetry below for why only one verb can.
* There is deliberately no `reset_unverifiable_derived_to_unknown`. It would be
  a laundering primitive - a route from any awkward state back to UNKNOWN and
  then onward to a convenient new claim - and a truthful unresolved state is
  better than erased uncertainty. `DERIVED_PROVENANCE_INCOMPLETE` may therefore
  remain visible indefinitely if the operator declines to make the authority
  decision this module offers.

There is also no generic `repair_knowledge(...)`. Each verb is named for the
decision a human actually made, so the audit row can be read without reading the
code that wrote it.

THE ACCEPT / DISCARD ASYMMETRY
------------------------------
An architectural distinction, not an implementation detail:

    accept_current_chunks_as_explicit   does NOT need the old generator.
        The operator is taking authority over the CURRENT chunks. Core neither
        executes nor interprets the recorded generator to honour that decision,
        so an unsupported identity or a version this Core cannot reproduce is
        no obstacle - provided the chunks are genuinely modified, which is what
        made the derived claim stale in the first place.

    discard_modified_chunks_and_regenerate  DOES need the generator.
        Core must CONSTRUCT a new truthful DERIVED artifact, which is impossible
        without a contract it implements at a version it understands. It
        therefore keeps refusing unsupported and version-ahead generators even
        when the chunks are modified.

So the same document can be acceptable to one verb and refused by the other, and
that is correct rather than an inconsistency.

SECURITY BOUNDARY
-----------------
Operator/system infrastructure. Never an Agent capability: no `ToolDefinition`,
no auto-resolution, no GAME, Admin, Build Console, Orchestrator or management
command, no signal, no `save()` hook, no background trigger.

Of everything built in this track, this is the invariant that matters most:

    A model must never autonomously choose to discard human Knowledge edits.
"""
from ai_hub.models import KnowledgeDocument
from ai_hub.services.knowledge_lifecycle import (
    SUPPORTED_GENERATORS,
    chunk_set_fingerprint,
    document_chunk_set_fingerprint,
    document_generation_input_fingerprint,
    generator_output_projection,
    is_supported_generator,
)
from ai_hub.services.knowledge_mutation import _governed_knowledge_mutation

# Deterministic helpers shared with Slice 11. Private on both sides: this is an
# in-Core collaboration between two operation-specific services, not an API.
# `_require_derived_with_usable_generator` is the part of the safety proof the
# two operations agree on; each then applies its own, opposite, chunk-set
# condition. Extracted rather than copied so the shared rules cannot drift.
from ai_hub.services.knowledge_regeneration import (
    _replace_chunk_set,
    _require_derived_with_usable_generator,
    _validate_candidate,
)

OPERATION_ACCEPT_CURRENT_CHUNKS_AS_EXPLICIT = "accept_current_chunks_as_explicit"
OPERATION_DISCARD_MODIFIED_CHUNKS_AND_REGENERATE = (
    "discard_modified_chunks_and_regenerate"
)


class KnowledgeRepairError(RuntimeError):
    """A repair decision was refused. Nothing was changed."""


class IneligibleLifecycleStateError(KnowledgeRepairError):
    """The document is not in a state this decision applies to.

    Carries a machine-readable `reason` so a caller - notably a future operator
    surface - can decide what to offer next WITHOUT parsing the message. The
    vocabulary is deliberately tiny and closed, and describes actual refusal
    paths only:

        not_derived        authority is UNKNOWN or EXPLICIT; a different verb
                           owns those, or none does
        claim_verifies     the DERIVED claim still checks out, so there is no
                           authority ambiguity for a human to resolve
        capability_defect  the claim checks out but THIS Core cannot interpret
                           the recorded generator (unsupported identity, or a
                           version ahead of ours). The remedy is upgrading Core

    `claim_verifies` and `capability_defect` refuse identically; they are
    separated because the operator's next step differs - "nothing to repair"
    versus "upgrade Core" - and that distinction was previously only conveyed in
    prose.

    NOTE: `capability_defect` is never reported when the chunks are modified.
    `DERIVED_CHUNKS_MODIFIED` is dominant, so such a document is ELIGIBLE for
    accept-as-EXPLICIT and raises nothing at all.
    """

    REASON_NOT_DERIVED = "not_derived"
    REASON_CLAIM_VERIFIES = "claim_verifies"
    REASON_CAPABILITY_DEFECT = "capability_defect"

    REASONS = frozenset(
        {REASON_NOT_DERIVED, REASON_CLAIM_VERIFIES, REASON_CAPABILITY_DEFECT}
    )

    def __init__(self, message, *, reason):
        if reason not in self.REASONS:
            raise ValueError(f"Unknown ineligibility reason {reason!r}.")
        self.reason = reason
        super().__init__(message)


class MissingReasonCodeError(KnowledgeRepairError):
    """These decisions require a recorded justification.

    Unlike the older lifecycle verbs, both operations here either transfer
    authority permanently or destroy reviewed human work. The lifecycle event
    carries no free text by design, so the bounded reason code is the ONLY
    carrier of *why* - and a future reader asking "who decided to discard that
    editor's work, and on what grounds?" has to be able to answer it from the
    row.
    """


class MissingChunkSetForExplicitError(KnowledgeRepairError):
    """`EXPLICIT` means the chunk set IS the authored artifact.

    With no chunk rows the declaration would be vacuous and would block later
    derived generation. Same rule Slice 10 applies to `UNKNOWN -> EXPLICIT`.

    This checks EXISTENCE, not usability. Whether the chunks carry usable text is
    the STRUCTURAL axis (`KP003`/`UNUSABLE_CHUNKS`), kept orthogonal to authority
    since Slice 8; Preflight goes on reporting structural problems separately.
    """


class ChunkSetNotModifiedError(KnowledgeRepairError):
    """There are no ungoverned modifications to discard.

    The chunks still match what Core recorded generating, so this is an ordinary
    regeneration and belongs to `regenerate_derived_chunk_set`. Routing it here
    would let a human decision stand in for a proof Core can make by itself.
    """

    def __init__(self, *, document_id, fingerprint):
        self.document_id = document_id
        self.fingerprint = fingerprint
        super().__init__(
            f"Knowledge document #{document_id} has chunks that still match the "
            f"recorded generated set ({fingerprint!r}). There is nothing to "
            "discard; use ordinary regeneration."
        )


def _require_reason_code(reason_code):
    """Both decisions require a justification. Shape is validated downstream."""
    if not str(reason_code or "").strip():
        raise MissingReasonCodeError(
            "A non-empty reason_code is required: this decision either transfers "
            "authority permanently or destroys reviewed human modifications, and "
            "the lifecycle event has no other place to record why."
        )


def _core_cannot_interpret_generator(document):
    """Is the recorded generator unreadable by THIS Core?

    True for an unknown identity, or a known one at a version newer than the one
    this Core implements. Both mean the same thing operationally: upgrading Core
    would fix it, and rewriting the data would not.

    Used only to choose a refusal REASON, never to change an outcome. Whether a
    document is eligible is decided entirely by the chunk-set and provenance
    conditions above.
    """
    identity = document.generator_identity
    if not is_supported_generator(identity):
        return True
    return document.generator_version > SUPPORTED_GENERATORS[identity]


def _require_derived(document):
    modes = KnowledgeDocument.ChunkAuthorityMode
    if document.chunk_authority_mode != modes.DERIVED:
        raise IneligibleLifecycleStateError(
            f"Knowledge document #{document.pk} has authority mode "
            f"{document.chunk_authority_mode!r}. Repair decisions apply only to "
            f"{modes.DERIVED!r}: UNKNOWN goes through operator adjudication, and "
            "EXPLICIT is already the authored artifact.",
            reason=IneligibleLifecycleStateError.REASON_NOT_DERIVED,
        )


def accept_current_chunks_as_explicit(
    document_id, *, expected, principal, reason_code
):
    """Record that the CURRENT chunk set is authoritative from now on.

    A forward authority transfer, not a claim about the past. It says only:

        from this governed decision onward, the chunk set the operator reviewed
        is the authoritative artifact.

    It does NOT claim the chunks were always explicit, that they were never
    generated, that the previous provenance was false, or that they match
    `curated_text`. Slice 4's conclusion was to *record ownership when it
    transfers* rather than infer authority afterwards; this is that record.

    Eligible from `DERIVED_CHUNKS_MODIFIED` and `DERIVED_PROVENANCE_INCOMPLETE`
    only - the two states whose defect is in the DATA and would look identical
    under any Core version. Everything else refuses, with a machine-readable
    `reason` on `IneligibleLifecycleStateError`.

    Because `DERIVED_CHUNKS_MODIFIED` is dominant, a document that ALSO carries
    an unsupported or version-ahead generator is eligible here: the recorded
    claim has stopped describing the current chunks whatever this Core makes of
    the historic generator, and honouring the operator's decision requires
    neither executing nor interpreting it. The same document is refused by
    `discard_modified_chunks_and_regenerate`, which does need a usable
    generator. That asymmetry is intentional.

    **Irreversible.** No `EXPLICIT -> anything` verb exists, deliberately: one
    would be a laundering route around the protection EXPLICIT provides. An
    operator who blesses corrupted chunks has no route back, so the reviewed
    snapshot obligation below is not a formality.

    `expected` MUST be the state the operator ACTUALLY REVIEWED - specifically
    the chunk bodies being blessed. Re-snapshotting at call time satisfies the
    compare-and-swap vacuously and removes the only protection this operation
    has.

    Not one chunk row is written. Generation provenance is cleared, because
    EXPLICIT carrying provenance is itself a `KP010` inconsistency and this
    operation must not create the anomaly it declines to repair.
    """
    modes = KnowledgeDocument.ChunkAuthorityMode
    _require_reason_code(reason_code)

    with _governed_knowledge_mutation(
        document_id,
        expected=expected,
        operation=OPERATION_ACCEPT_CURRENT_CHUNKS_AS_EXPLICIT,
        principal=principal,
        reason_code=reason_code,
    ) as mutation:
        document = mutation.document
        _require_derived(document)

        # Eligible iff the DERIVED claim is not verifiable-and-current: either
        # the provenance cannot verify itself, or it can and the chunks disagree
        # with it. A document whose claim checks out has nothing to repair.
        recorded = document.generation_chunk_set_fingerprint
        provenance_incomplete = not all(
            (
                document.generation_input_fingerprint,
                recorded,
                document.generator_identity,
            )
        ) or document.generator_version is None

        if not provenance_incomplete:
            observed = document_chunk_set_fingerprint(document)
            if observed == recorded:
                # The chunks still match the recorded generated set, so the
                # DERIVED claim describes the current artifact accurately and
                # there is no authority ambiguity for a human to resolve.
                #
                # Both branches below refuse identically; they differ only in
                # what the operator should do next, which is why the reason is
                # machine-readable rather than buried in prose.
                if _core_cannot_interpret_generator(document):
                    identity = document.generator_identity
                    raise IneligibleLifecycleStateError(
                        f"Knowledge document #{document.pk} records generator "
                        f"{identity!r} at version {document.generator_version}, "
                        "which this Core cannot interpret, and its chunks still "
                        "match the recorded generated set. That is a defect in "
                        "this Core's capability, not in the data: upgrade Core "
                        "rather than erasing truthful provenance. (Had the "
                        "chunks been modified, the DERIVED claim would no longer "
                        "describe them and this decision would be available.)",
                        reason=IneligibleLifecycleStateError.REASON_CAPABILITY_DEFECT,
                    )
                raise IneligibleLifecycleStateError(
                    f"Knowledge document #{document.pk} has a DERIVED claim that "
                    "still verifies: the chunks match the recorded generated set. "
                    "There is no authority ambiguity to resolve.",
                    reason=IneligibleLifecycleStateError.REASON_CLAIM_VERIFIES,
                )

        if not document.chunks.exists():
            raise MissingChunkSetForExplicitError(
                f"Knowledge document #{document.pk} has no chunks, so there is no "
                "authored artifact to declare authoritative."
            )

        document.chunk_authority_mode = modes.EXPLICIT
        document.generation_input_fingerprint = ""
        document.generation_chunk_set_fingerprint = ""
        document.generator_identity = ""
        document.generator_version = None
        document.save(
            update_fields=[
                "chunk_authority_mode",
                "generation_input_fingerprint",
                "generation_chunk_set_fingerprint",
                "generator_identity",
                "generator_version",
            ]
        )

    return KnowledgeDocument.objects.get(pk=document_id)


def discard_modified_chunks_and_regenerate(
    document_id, *, expected, principal, reason_code
):
    """Discard reviewed ungoverned modifications and restore generated truth.

    **This is not ordinary regeneration.** `regenerate_derived_chunk_set` refuses
    `DERIVED_CHUNKS_MODIFIED` and continues to, unwidened - its refusal is the
    reason this verb exists. The destructive permission here comes from an
    explicit human decision to discard the state they reviewed, not from a proof
    Core can make.

    Eligible only from `DERIVED_CHUNKS_MODIFIED`, and only while the rest of the
    DERIVED contract remains usable: complete provenance, a supported generator,
    and a recorded version not ahead of this Core's. Without those there is
    nothing to regenerate *into*.

    This is where the two repair verbs diverge. Core must CONSTRUCT a new
    truthful DERIVED artifact here, which is impossible without a generator
    contract it implements at a version it understands - so unsupported and
    version-ahead generators keep being refused even when the chunks are
    modified, while `accept_current_chunks_as_explicit` accepts exactly those
    documents. The operator taking authority over current chunks needs no
    generator; Core rebuilding an artifact does.

    No fingerprint is ever fabricated to reach this path. An implementation that
    rewrote the recorded fingerprint to match the observed one and then called
    ordinary regeneration would assert - even transiently - that Core generated
    chunks it did not generate. This operation instead runs its own governed
    mutation with its own slug, so one human decision produces exactly one
    lifecycle event.

    `expected` MUST be the state the operator ACTUALLY REVIEWED: both the
    modifications being destroyed and the generation inputs that will replace
    them. The CAS binds both through `observed_chunk_set_fingerprint` and
    `observed_input_fingerprint`.

    Chunk identity follows D-3e-2 against the CURRENT set. Note the edge case
    that makes this worth stating: `DERIVED_CHUNKS_MODIFIED` means observed
    differs from RECORDED - it does not imply the modified chunks differ from
    what the generator would produce NOW. If they happen to be equal, "discard"
    is vacuous, no row is touched and no primary key churns; only the stale
    provenance is corrected.
    """
    _require_reason_code(reason_code)

    with _governed_knowledge_mutation(
        document_id,
        expected=expected,
        operation=OPERATION_DISCARD_MODIFIED_CHUNKS_AND_REGENERATE,
        principal=principal,
        reason_code=reason_code,
    ) as mutation:
        document = mutation.document
        # Deliberately duplicates the authority check inside the shared helper
        # below: raising the repair-specific error first gives a caller the
        # machine-readable `reason`, which Slice 11's AuthorityNotDerivedError
        # does not carry. Not redundant - error shaping.
        _require_derived(document)

        # Shared with Slice 11: DERIVED + complete provenance + supported
        # generator + version not ahead. Raises Slice 11's typed errors, which
        # describe the same facts and need no restatement here.
        identity, current_version = _require_derived_with_usable_generator(document)

        observed = document_chunk_set_fingerprint(document)
        if observed == document.generation_chunk_set_fingerprint:
            raise ChunkSetNotModifiedError(
                document_id=document.pk, fingerprint=observed
            )

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
