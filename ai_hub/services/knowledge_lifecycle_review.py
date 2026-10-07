"""Coherent operator-review capture for Knowledge lifecycle decisions (Slice 13).

WHAT THIS IS
------------
The evidence half of a governed lifecycle decision. It captures, in one
consistent read, everything a human must see before authorising an authority
transfer, a regeneration or a destruction - together with the exact
`ExpectedKnowledgeState` that will later be submitted to the Core verb.

It decides nothing. It mutates nothing. It contains no lifecycle policy: which
states are eligible, what refuses and what an operation writes all stay in the
operation-specific services, which re-derive and re-validate everything under
their own lock.

WHY THIS EXISTS IN CORE RATHER THAN IN THE COMMAND
--------------------------------------------------
Because coherence is a safety property, and a safety property implemented in an
adapter is not inherited by the next adapter.

The hazard is precise. If the snapshot is computed by one query and the chunk
bodies shown to the operator are fetched by another, a document that changes and
changes back between them lets the operator review one chunk set while the
compare-and-swap binds - and commits - a different one:

    query 1  -> snapshot of A          CAS will bind A
    ...      -> document becomes B
    query 2  -> bodies displayed: B    operator reviews B
    ...      -> document reverts to A
    confirm  -> CAS passes on A        operator authorised B

Note what is NOT the hazard: the document changing and changing back on its own
is harmless, because the state the operator reviewed is the state that commits.
The bug is only ever the mismatch between *displayed bytes* and *hashed bytes*.
Reading the chunk rows exactly once removes it entirely.

NO LOCKS, NO TRANSACTION
------------------------
Capture takes no row locks and opens no transaction, so a human never holds a
lock while deciding. It does not need to: `_snapshot_from_chunk_rows` is fed the
same in-memory rows that are displayed, which is the whole requirement.

The document row and the chunk rows are still two statements, so their *metadata*
could in principle straddle a change. That is deliberately not defended against,
because it cannot produce an unsafe commit: the compare-and-swap compares all
twelve fields against one consistent locked read at confirmation, so an
incoherent capture can only yield a spurious conflict. A spurious conflict is
acceptable; a mutation against content the operator did not review is not.

EPHEMERAL
---------
The review bundle is process-local, immutable, never persisted and never
Agent-facing. It is evidence of what a human saw, not a source of lifecycle
truth, and nothing downstream is allowed to trust it: the target service
recomputes its own projection and its own snapshot under its own transaction.
"""
from dataclasses import dataclass

from ai_hub.models import KnowledgeDocument
from ai_hub.services.knowledge_adjudication import (
    ADOPTION_GENERATOR,
    OPERATION_ADJUDICATE_UNKNOWN_AS_EXPLICIT,
    OPERATION_ADOPT_UNKNOWN_AS_DERIVED,
)
from ai_hub.services.knowledge_lifecycle import (
    SUPPORTED_GENERATORS,
    UnsupportedGeneratorError,
    chunk_set_fingerprint,
    generator_output_projection,
)
from ai_hub.services.knowledge_mutation import (
    ExpectedKnowledgeState,
    _snapshot_from_chunk_rows,
)
from ai_hub.services.knowledge_regeneration import (
    OPERATION_REGENERATE_DERIVED_CHUNK_SET,
)
from ai_hub.services.knowledge_repair import (
    OPERATION_ACCEPT_CURRENT_CHUNKS_AS_EXPLICIT,
    OPERATION_DISCARD_MODIFIED_CHUNKS_AND_REGENERATE,
)

# Actions whose review is incomplete without showing what the generator would
# produce. The operator cannot judge an adoption, a regeneration or a discard
# without seeing the artifact that would replace or justify the current one.
CANDIDATE_ACTIONS = frozenset(
    {
        OPERATION_ADOPT_UNKNOWN_AS_DERIVED,
        OPERATION_REGENERATE_DERIVED_CHUNK_SET,
        OPERATION_DISCARD_MODIFIED_CHUNKS_AND_REGENERATE,
    }
)

REVIEWABLE_ACTIONS = frozenset(
    {
        OPERATION_ADJUDICATE_UNKNOWN_AS_EXPLICIT,
        OPERATION_ADOPT_UNKNOWN_AS_DERIVED,
        OPERATION_REGENERATE_DERIVED_CHUNK_SET,
        OPERATION_ACCEPT_CURRENT_CHUNKS_AS_EXPLICIT,
        OPERATION_DISCARD_MODIFIED_CHUNKS_AND_REGENERATE,
    }
)


class ReviewEvidenceUnavailable(RuntimeError):
    """The evidence a truthful review requires could not be constructed.

    Raised BEFORE any confirmation prompt, so nothing is decided and nothing is
    mutated. This is a failure to assemble review material, NOT a lifecycle
    verdict: it makes no claim about whether the operation would have been
    permitted, and it never substitutes another generator or another action.

    The commonest cause is an operation whose candidate depends on a generator
    contract this Core does not implement.
    """


@dataclass(frozen=True)
class CapturedChunk:
    """One chunk row, frozen at capture.

    Exposes exactly the three fields the `c1` contract covers, as attributes, so
    the very same objects shown to the operator can be handed to
    `_snapshot_from_chunk_rows` - the chunk-set contract accepts mappings or
    attribute-bearing objects alike.
    """

    chunk_index: int
    section_title: str
    content: str


@dataclass(frozen=True)
class KnowledgeLifecycleReview:
    """What the operator saw, and the exact expectation bound to it.

    `expected` is the load-bearing field: it must travel unchanged from here to
    the Core verb. Re-deriving it at confirmation time would compare current
    state against current state and satisfy the compare-and-swap vacuously.
    """

    action: str
    snapshot: object
    expected: ExpectedKnowledgeState

    document_id: int
    title: str
    collection_id: int
    collection_name: str
    status: str
    authority_mode: str

    recorded_input_fingerprint: str
    recorded_chunk_set_fingerprint: str
    generator_identity: str
    generator_version: "int | None"

    curated_text: str
    chunks: tuple

    candidate: "tuple | None" = None
    candidate_fingerprint: "str | None" = None
    candidate_generator_identity: "str | None" = None
    # The version the candidate is produced under and that the service will
    # record - NOT `generator_version` above, which is what the document
    # currently claims. When the two differ the operator is authorising a
    # version advance and must be shown both.
    candidate_generator_version: "int | None" = None

    @property
    def candidate_matches_current(self):
        """Would the generator output equal the current chunk set under `c1`?

        Decides whether chunk identities are expected to survive. `None` when no
        candidate applies to this action.
        """
        if self.candidate_fingerprint is None:
            return None
        return self.candidate_fingerprint == self.snapshot.observed_chunk_set_fingerprint


def _candidate_generator_for(action, document):
    """Which generator the TARGET operation would use - never a substitute.

    Adoption pins the one contract this Core implements, because an `UNKNOWN`
    document records no identity of its own. Regeneration and discard use the
    document's own recorded identity, exactly as those services do.
    """
    if action == OPERATION_ADOPT_UNKNOWN_AS_DERIVED:
        return ADOPTION_GENERATOR
    return document.generator_identity or ""


def capture_lifecycle_review(document_id, *, action):
    """Capture a coherent review bundle. Reads only; takes no locks.

    Raises `KnowledgeDocument.DoesNotExist` for an unknown document and
    `ReviewEvidenceUnavailable` when a required candidate cannot be produced.
    """
    if action not in REVIEWABLE_ACTIONS:
        raise ValueError(f"Unknown lifecycle action {action!r}.")

    document = KnowledgeDocument.objects.select_related("collection").get(pk=document_id)

    # ONE read of the chunk rows. These exact objects are hashed into the
    # snapshot AND shown to the operator; nothing re-queries them.
    chunks = tuple(
        CapturedChunk(
            chunk_index=row["chunk_index"],
            section_title=row["section_title"],
            content=row["content"],
        )
        for row in document.chunks.order_by("chunk_index").values(
            "chunk_index", "section_title", "content"
        )
    )

    snapshot = _snapshot_from_chunk_rows(document, chunks)

    candidate = candidate_fingerprint = None
    candidate_identity = candidate_version = None
    if action in CANDIDATE_ACTIONS:
        candidate_identity = _candidate_generator_for(action, document)
        try:
            projected = generator_output_projection(candidate_identity, document)
        except UnsupportedGeneratorError as exc:
            detail = (
                "the document records no generator identity"
                if not candidate_identity
                else f"this Core implements no contract for generator {candidate_identity!r}"
            )
            raise ReviewEvidenceUnavailable(
                f"Cannot show a candidate for {action!r} on Knowledge document "
                f"#{document_id}: {detail}. Without the candidate there is "
                "nothing truthful to review, so no decision is offered. This is "
                "a failure to assemble review evidence, NOT a verdict on whether "
                "the operation would have been permitted. Nothing was changed."
            ) from exc
        candidate = tuple(
            CapturedChunk(
                chunk_index=row["chunk_index"],
                section_title=row["section_title"],
                content=row["content"],
            )
            for row in projected
        )
        candidate_fingerprint = chunk_set_fingerprint(projected)
        # The version that will actually produce and be RECORDED for this
        # candidate - the contract this Core implements - not the version the
        # document happens to record. Those are different facts, and conflating
        # them made the review evidence untrue in every generator-backed case:
        # adoption showed None while the service would record v1, and an
        # outdated document showed its stale version rather than the one the
        # regeneration would advance to. The projection above already succeeded,
        # so the identity is known-supported and this lookup cannot fail.
        candidate_version = SUPPORTED_GENERATORS[candidate_identity]

    return KnowledgeLifecycleReview(
        action=action,
        snapshot=snapshot,
        expected=ExpectedKnowledgeState.from_snapshot(snapshot),
        document_id=document.pk,
        title=document.title,
        collection_id=document.collection_id,
        collection_name=document.collection.name,
        status=document.status,
        authority_mode=document.chunk_authority_mode,
        recorded_input_fingerprint=document.generation_input_fingerprint or "",
        recorded_chunk_set_fingerprint=document.generation_chunk_set_fingerprint or "",
        generator_identity=document.generator_identity or "",
        generator_version=document.generator_version,
        curated_text=document.curated_text or "",
        chunks=chunks,
        candidate=candidate,
        candidate_fingerprint=candidate_fingerprint,
        candidate_generator_identity=candidate_identity,
        candidate_generator_version=candidate_version,
    )
