"""Governed Knowledge mutation foundation (Slice 9, RC-002 step 3c).

INTERNAL FOUNDATION — NOT HOST API.

WHAT THIS IS
------------
The trusted mechanism through which a Knowledge lifecycle change is allowed to
happen. It does NOT decide what any change should be. It guarantees that when
one does happen it is:

    atomic              - the state change and its audit record commit together
    stale-review-safe   - it applies to the EXACT state that was reviewed
    principal-bound     - the initiator is supplied by trusted server code
    durably auditable   - a queryable, reference-first event row survives it

WHAT THIS IS NOT
----------------
There is deliberately no `adjudicate()`, `repair()`, `regenerate()`,
`backfill()` or `set_authority()` here. Those are domain operations belonging to
RC-002 steps 3d and 3e. Shipping a convenient one now would implement the next
slice's decision under this slice's name, and the decision about what legacy
`UNKNOWN` *means* has not been made.

There is also **no public generic mutation API**. `_governed_knowledge_mutation`
is private on purpose: a supported public entry point whose contract is "mutate
anything inside this KnowledgeDocument transaction, under any operation slug you
like" would be a business API pretending to be infrastructure. Future PUBLIC
write APIs must be operation-specific and must each carry their own validation -
for example a later step 3d might expose something like
`adjudicate_unknown_as_explicit(...)`, which is NOT implemented here.

    PUBLIC    KnowledgeMutationPrincipal, KnowledgeMutationSnapshot,
              ExpectedKnowledgeState, KnowledgeMutationConflict and the two
              read-only helpers `build_snapshot` / `verify_expected_state`.
    INTERNAL  `_governed_knowledge_mutation`, `_GovernedMutation`,
              `_record_event`. Callable only by future in-Core, operation-
              specific services. Not a host API, and nothing imports it today.

SECURITY BOUNDARY
-----------------
Operator/system infrastructure. Never an Agent capability: no `ToolDefinition`,
no auto-resolution, no GAME, Admin or Orchestrator exposure, no model-facing
surface. A model must never be able to declare itself the initiating principal,
change Knowledge authority, or create a lifecycle event. The principal is
server-bound by the caller and is never read from model output, a GAME payload
or tool arguments.

WHAT THE LOCKS DO AND DO NOT PROVE
----------------------------------
`select_for_update()` serializes *governed* writers against each other. It
cannot make an ungoverned raw-ORM writer safe: raw ORM remains an intentional
escape hatch, and a writer that never enters this boundary never takes the lock.
That is accepted, not papered over. The defence against ungoverned edits is
evidence, not exclusion - fingerprints make them *detectable* afterwards through
Preflight V2, and the compare-and-swap below makes them *fatal* to a governed
mutation planned against stale evidence.

SQLite has no real row locks (`connection.features.has_select_for_update` is
False), so on SQLite `select_for_update()` is effectively a no-op and this
module's serialization claims are unproven. Only the PostgreSQL CI job exercises
them. The compare-and-swap check, by contrast, is pure application logic and
holds identically on every backend.
"""
import re
from contextlib import contextmanager
from dataclasses import dataclass, fields as dataclass_fields, replace

from django.db import transaction

from ai_hub.models import KnowledgeDocument, KnowledgeLifecycleEvent
from ai_hub.services.knowledge_lifecycle import (
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK,
    chunk_set_fingerprint,
    curated_text_single_chunk_input_fingerprint,
)


# The input contract used to compute the OBSERVED input fingerprint in a
# snapshot. Pinned to the one contract this Core implements. A snapshot says
# "under `curated_text_single_chunk` v1 the inputs currently hash to this"; it
# never claims the document IS derived, and computing it for an `UNKNOWN`
# document is a neutral measurement, not a reclassification.
SNAPSHOT_INPUT_GENERATOR = GENERATOR_CURATED_TEXT_SINGLE_CHUNK

# Operation slugs are validated here rather than by model `choices`, so the
# vocabulary can grow in later slices without an AlterField migration for a
# column whose storage never changes. Shape only - this module deliberately
# holds no list of operations, because none exist yet and inventing speculative
# ones to fill an enum would be fabrication.
_OPERATION_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


class KnowledgeMutationConflict(RuntimeError):
    """The document changed between review and commit; nothing was applied.

    Raised before any mutation runs, inside the transaction, so a caller that
    lets it propagate leaves zero Knowledge changes and zero lifecycle events.
    Never recovered from by silently recomputing the expectation and continuing:
    the operator reviewed evidence that no longer exists, so their decision has
    to be made again against what is actually there.
    """

    def __init__(self, *, document_id, field, expected, actual):
        self.document_id = document_id
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Knowledge document #{document_id} changed since it was reviewed: "
            f"expected {field} {expected!r}, found {actual!r}. "
            "Re-read the document and decide again."
        )


class KnowledgeMutationPrincipalError(ValueError):
    """The initiating principal is missing or malformed."""


class KnowledgeMutationOperationError(ValueError):
    """The operation slug or reason code is missing or malformed."""


class UnauditableMutationError(RuntimeError):
    """The mutation changed a fact the lifecycle event cannot describe.

    The foundation refuses rather than committing a change its audit row would
    silently misrepresent. Extend the event schema before permitting the change.
    """


@dataclass(frozen=True)
class KnowledgeMutationPrincipal:
    """WHO initiated a lifecycle mutation. Immutable, server-bound, tiny.

    Not an `AgentProfile` and not derived from one. Lifecycle mutation is
    operator/system activity: an Agent has no standing to change what the corpus
    claims about itself, and accepting a principal from model output, a GAME
    payload or tool arguments would let a model author its own audit trail. The
    caller - trusted server code in a host adapter, a management command or an
    admin view - is responsible for binding this from an authenticated context.

    A snapshot rather than a foreign key: history must stay readable after a
    host deletes the user, and Core must not depend on a host's user model.

    `identifier` must be a stable opaque handle such as a primary key or
    username. Do NOT pass an email address, personal name or any other contact
    detail: audit rows are retained indefinitely. There is deliberately no
    display-label field - a host that wants to show a name resolves it from the
    identifier at display time rather than freezing it into history.
    """

    kind: str
    identifier: str

    HUMAN = KnowledgeLifecycleEvent.PrincipalKind.HUMAN
    SYSTEM = KnowledgeLifecycleEvent.PrincipalKind.SYSTEM

    @classmethod
    def human(cls, identifier):
        return cls(kind=cls.HUMAN, identifier=identifier)

    @classmethod
    def system(cls, identifier):
        return cls(kind=cls.SYSTEM, identifier=identifier)

    def validate(self):
        valid_kinds = set(KnowledgeLifecycleEvent.PrincipalKind.values)
        if self.kind not in valid_kinds:
            raise KnowledgeMutationPrincipalError(
                f"Unknown principal kind {self.kind!r}; expected one of {sorted(valid_kinds)}."
            )
        identifier = (self.identifier or "").strip()
        if not identifier:
            raise KnowledgeMutationPrincipalError(
                "A lifecycle mutation requires a principal identifier; anonymous "
                "governed writes are not supported."
            )
        if len(identifier) > 150:
            raise KnowledgeMutationPrincipalError(
                "Principal identifier is bounded to 150 characters."
            )
        return replace(self, identifier=identifier)


# Every correctness-relevant fact a reviewed decision depends on. Declared once
# so the snapshot, the expectation and the comparison can never drift apart: add
# a field here and all three pick it up.
CAS_FIELDS = (
    "document_id",
    "collection_id",
    "status",
    "authority_mode",
    "chunk_count",
    "observed_input_generator",
    "observed_input_fingerprint",
    "observed_chunk_set_fingerprint",
    "recorded_input_fingerprint",
    "recorded_chunk_set_fingerprint",
    "generator_identity",
    "generator_version",
)


@dataclass(frozen=True)
class KnowledgeMutationSnapshot:
    """Immutable measurement of one document at one instant.

    Two jobs. It is the stale-review token a caller compares against before
    committing a decision, and it is the before/after evidence recorded in the
    lifecycle event.

    It carries both what the document *records* about its generation and what
    its rows *currently hash to*. Those are different facts and collapsing them
    would destroy the tamper evidence: `recorded_chunk_set_fingerprint` is a
    claim, `observed_chunk_set_fingerprint` is a measurement, and a mismatch
    between them is exactly what Preflight V2 reports as
    `DERIVED_CHUNKS_MODIFIED`.

    It infers nothing. `observed_input_fingerprint` is computed under the
    supported V1 input contract for every document regardless of authority mode;
    for an `UNKNOWN` document that is a neutral measurement and emphatically not
    a claim that the document is DERIVED. `observed_input_generator` is carried
    alongside it because a fingerprint is meaningless without the contract that
    produced it.
    """

    document_id: int
    collection_id: int
    status: str
    authority_mode: str
    chunk_count: int
    observed_input_generator: str
    observed_input_fingerprint: str
    observed_chunk_set_fingerprint: str
    recorded_input_fingerprint: str
    recorded_chunk_set_fingerprint: str
    generator_identity: str
    generator_version: "int | None"


@dataclass(frozen=True)
class ExpectedKnowledgeState:
    """The COMPLETE reviewed state a decision was made against.

    Every field is required, and the only supported constructor is
    `from_snapshot()`. That is the correction: an earlier design made most
    fields optional with only the chunk-set fingerprint required, which let a
    caller silently downgrade an exact reviewed snapshot into a partial
    compare-and-swap.

    Fingerprints alone are not sufficient. A document can change in ways that
    move neither `i1` nor `c1` and still invalidate the decision - most
    importantly a **collection move**, which leaves title, `curated_text` and
    every chunk untouched while changing the retrieval authorization boundary.
    Status and the recorded generation facts are the same kind of hazard. So the
    expectation binds the whole correctness-relevant snapshot.

    Deliberately NOT `updated_at`. Slice 4 measured that timestamps fail in both
    directions: `updated_at` advances for edits that change nothing relevant,
    and `save(update_fields=[...])` can change content while leaving no trace a
    reader can rely on. Fingerprints and facts answer "is this the same state?"
    directly.

    A future operation-specific service MAY deliberately take a fresh snapshot
    and act on that - but it must do so explicitly, not by omitting fields.
    """

    document_id: int
    collection_id: int
    status: str
    authority_mode: str
    chunk_count: int
    observed_input_generator: str
    observed_input_fingerprint: str
    observed_chunk_set_fingerprint: str
    recorded_input_fingerprint: str
    recorded_chunk_set_fingerprint: str
    generator_identity: str
    generator_version: "int | None"

    @classmethod
    def from_snapshot(cls, snapshot):
        if not isinstance(snapshot, KnowledgeMutationSnapshot):
            raise TypeError("from_snapshot requires a KnowledgeMutationSnapshot.")
        return cls(**{name: getattr(snapshot, name) for name in CAS_FIELDS})


@dataclass(frozen=True)
class _GovernedMutation:
    """Handed to the caller inside the transaction. INTERNAL.

    `document` is the locked row. The caller edits and saves it; this object
    provides no verb that would amount to a domain operation.
    """

    document: KnowledgeDocument
    before: KnowledgeMutationSnapshot
    operation: str
    principal: KnowledgeMutationPrincipal


def _snapshot_from_chunk_rows(document, chunks):
    """Build the snapshot from chunk rows the CALLER already holds.

    Deterministic and pure: given the same document instance and the same rows,
    it always produces the same snapshot. It performs no query of its own.

    Extracted so an operator-review surface can read the chunk rows ONCE, retain
    them to show a human, and derive the snapshot from those exact values. If the
    displayed bytes and the hashed bytes came from two separate reads, a document
    that changed and changed back between them would let an operator review one
    chunk set while the compare-and-swap bound another - the review would be of
    content that never commits. One read closes that.

    `chunks` may be mappings (`.values()` rows) or objects exposing
    `chunk_index` / `section_title` / `content`; the chunk-set contract accepts
    both.
    """
    return KnowledgeMutationSnapshot(
        document_id=document.pk,
        collection_id=document.collection_id,
        status=document.status,
        authority_mode=document.chunk_authority_mode,
        chunk_count=len(chunks),
        observed_input_generator=SNAPSHOT_INPUT_GENERATOR,
        observed_input_fingerprint=curated_text_single_chunk_input_fingerprint(
            title=document.title, curated_text=document.curated_text
        ),
        observed_chunk_set_fingerprint=chunk_set_fingerprint(chunks),
        recorded_input_fingerprint=document.generation_input_fingerprint or "",
        recorded_chunk_set_fingerprint=document.generation_chunk_set_fingerprint or "",
        generator_identity=document.generator_identity or "",
        generator_version=document.generator_version,
    )


def build_snapshot(document):
    """Measure `document` as it currently is. Pure read; writes nothing.

    Reads chunk bodies to hash them and discards them immediately - the digest
    is kept, the text is not. Behaviour is unchanged: this is exactly the same
    query it always issued, with the deterministic construction delegated to
    `_snapshot_from_chunk_rows` so a review surface can reuse it with rows it
    already holds.
    """
    chunks = list(
        document.chunks.order_by("chunk_index").values(
            "chunk_index", "section_title", "content"
        )
    )
    return _snapshot_from_chunk_rows(document, chunks)


def verify_expected_state(snapshot, expected):
    """Compare-and-swap over the whole reviewed state.

    Raises `KnowledgeMutationConflict` on the first divergence, in `CAS_FIELDS`
    order. Pure application logic, so it behaves identically on SQLite and
    PostgreSQL.
    """
    if not isinstance(expected, ExpectedKnowledgeState):
        raise TypeError("expected must be an ExpectedKnowledgeState.")

    for name in CAS_FIELDS:
        want = getattr(expected, name)
        got = getattr(snapshot, name)
        if want != got:
            raise KnowledgeMutationConflict(
                document_id=snapshot.document_id, field=name, expected=want, actual=got
            )


def _validate_operation(operation, reason_code):
    if not _OPERATION_PATTERN.match(operation or ""):
        raise KnowledgeMutationOperationError(
            f"Invalid lifecycle operation {operation!r}: expected a lowercase "
            "slug of 3-64 characters matching [a-z][a-z0-9_]*."
        )
    if reason_code and not _REASON_CODE_PATTERN.match(reason_code):
        raise KnowledgeMutationOperationError(
            f"Invalid lifecycle reason code {reason_code!r}: expected a lowercase "
            "slug of 3-64 characters matching [a-z][a-z0-9_]*."
        )


def _lock_document(document_id):
    """Lock the document row and its chunk rows for the rest of the transaction.

    Chunks are locked as well as the document because the chunk-set fingerprint
    is computed from them: locking only the document would let another governed
    writer change the very rows this snapshot is measuring.
    """
    document = (
        KnowledgeDocument.objects.select_for_update()
        .select_related("collection")
        .get(pk=document_id)
    )
    # Materialize under the lock. The values are discarded; the point is the lock.
    list(document.chunks.select_for_update().values_list("pk", flat=True))
    return document


def _assert_auditable(before, after):
    """Refuse a change the lifecycle event cannot describe truthfully.

    The event carries no before/after collection, so a collection move would
    commit with an audit row that silently reports only the destination. A
    collection move is also an authorization-boundary change and a separate,
    security-sensitive future operation - so it is refused here rather than
    mis-recorded. Extend the event schema before permitting one.
    """
    if after.document_id != before.document_id:
        raise UnauditableMutationError(
            "Governed mutation changed the document identity; refusing to record "
            "a lifecycle event for a different document."
        )
    if after.collection_id != before.collection_id:
        raise UnauditableMutationError(
            f"Governed mutation moved document #{before.document_id} from "
            f"collection {before.collection_id} to {after.collection_id}. The "
            "lifecycle event cannot describe a collection move, and a move "
            "changes the retrieval authorization boundary, so it is not a "
            "governed operation in this foundation."
        )


@contextmanager
def _governed_knowledge_mutation(
    document_id, *, expected, operation, principal, reason_code=""
):
    """The governed write boundary. INTERNAL FOUNDATION — NOT HOST API.

        BEGIN
          lock document + chunks
          verify the FULL reviewed state  -> KnowledgeMutationConflict, nothing applied
          capture BEFORE
          << caller's Knowledge-specific mutation runs here >>
          capture AFTER, refuse anything unauditable
          create KnowledgeLifecycleEvent
        COMMIT

    If the caller's block raises, the transaction rolls back and no event
    exists. If event creation fails, the caller's mutation rolls back with it.
    Neither half can survive the other.

    Private because "mutate anything in this document's transaction under any
    slug" is not a contract Core should support. Intended callers are future
    in-Core, operation-specific services, each of which owns its own domain
    validation and exposes its own narrow public verb::

        with _governed_knowledge_mutation(
            doc.pk, expected=expected, operation="...", principal=principal
        ) as mutation:
            mutation.document.chunk_authority_mode = ...
            mutation.document.save(update_fields=[...])

    No such service exists yet, and this module intentionally provides none.
    """
    _validate_operation(operation, reason_code)
    if principal is None:
        raise KnowledgeMutationPrincipalError(
            "A lifecycle mutation requires a KnowledgeMutationPrincipal."
        )
    bound_principal = principal.validate()

    with transaction.atomic():
        document = _lock_document(document_id)
        before = build_snapshot(document)
        verify_expected_state(before, expected)

        yield _GovernedMutation(
            document=document,
            before=before,
            operation=operation,
            principal=bound_principal,
        )

        # Re-read rather than trusting the in-memory instance: the caller may
        # have used queryset updates, and the AFTER facts must describe what the
        # database will actually commit.
        applied = KnowledgeDocument.objects.get(pk=document_id)
        after = build_snapshot(applied)
        _assert_auditable(before, after)

        _record_event(before=before, after=after, operation=operation,
                      principal=bound_principal, reason_code=reason_code)


def _record_event(*, before, after, operation, principal, reason_code):
    """Write the committed-state event. INTERNAL: there is no public writer.

    Called only from inside `_governed_knowledge_mutation`'s transaction, which
    is what makes "no committed mutation without its event" structural rather
    than a convention a caller could forget.
    """
    return KnowledgeLifecycleEvent.objects.create(
        document_id=after.document_id,
        document_id_snapshot=after.document_id,
        collection_id_snapshot=after.collection_id,
        operation=operation,
        reason_code=reason_code,
        principal_kind=principal.kind,
        principal_identifier=principal.identifier,
        previous_authority_mode=before.authority_mode,
        new_authority_mode=after.authority_mode,
        previous_status=before.status,
        new_status=after.status,
        previous_generation_input_fingerprint=before.recorded_input_fingerprint,
        new_generation_input_fingerprint=after.recorded_input_fingerprint,
        previous_generation_chunk_set_fingerprint=before.recorded_chunk_set_fingerprint,
        new_generation_chunk_set_fingerprint=after.recorded_chunk_set_fingerprint,
        previous_generator_identity=before.generator_identity,
        new_generator_identity=after.generator_identity,
        previous_generator_version=before.generator_version,
        new_generator_version=after.generator_version,
        previous_chunk_count=before.chunk_count,
        new_chunk_count=after.chunk_count,
        previous_observed_input_fingerprint=before.observed_input_fingerprint,
        new_observed_input_fingerprint=after.observed_input_fingerprint,
        previous_observed_chunk_set_fingerprint=before.observed_chunk_set_fingerprint,
        new_observed_chunk_set_fingerprint=after.observed_chunk_set_fingerprint,
    )
