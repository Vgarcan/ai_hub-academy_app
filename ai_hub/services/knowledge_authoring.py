"""Operation-specific authoring of an EXPLICIT Knowledge chunk set (T2).

The first public write path for Knowledge *content*. It lets trusted host code
replace the chunk set of ONE existing document with a caller-supplied sequence,
under the governed mutation foundation, and nothing else.

    author_explicit_chunk_set(document_id, chunks, *, application_scope,
                              expected, principal, reason_code="")

WHAT IT GUARANTEES
------------------
* **Scope-bound.** The document's collection must belong to `application_scope`,
  and that scope must be active. This is checked before anything else and again
  inside the governed transaction. The active scope row is locked before the
  governed foundation locks the document and chunks, so deactivation and authoring
  have one serial order. A missing document, a document of another scope and an
  inactive scope are indistinguishable: one error, one message, and it takes
  precedence over every other argument or payload error.
* **Governed and audited.** A change runs inside the mutation foundation: lock,
  compare-and-swap against `expected`, one `KnowledgeLifecycleEvent` in the same
  transaction. If any step or the event write fails, everything rolls back.
* **Deterministic identity.** A chunk whose canonical section title and content
  are unchanged keeps its row, its `chunk_id`, its `k1` and its stored vector.
  Anything else is a new row with a new id. An id that was cited is therefore
  either still the same text or gone; it is never silently reused for different
  text. Ids are not reused by the database.
* **Verified result.** The stored set is contiguous (1..N, assigned here) and its
  `c1` fingerprint is compared with the fingerprint of what was requested before
  the transaction may commit.

WHAT IT DOES NOT DO
-------------------
* It never changes `status`. Activating a document is a separate, host-owned
  step, and that step is not audited. First authoring requires a DRAFT document,
  so new content stays invisible until the host activates it.
* It never runs or schedules embeddings. Kept chunks keep current vectors; new
  chunks have none until the operator runs `knowledge_embedding_index`.
* It accepts no chunk metadata and no token estimate, and it never edits either
  on an existing row. `c1` and the mutation snapshot do not cover them, so the
  lifecycle event could not describe such a change. A document whose chunks carry
  non-empty metadata is refused rather than silently rewritten or dropped.
* It does not parse files, split text, or know any host identifier.

WHO MAY CALL IT
---------------
Trusted server-side host code only. `application_scope` and `principal` must come
from the host's own configuration or authenticated context, never from a request,
a model output, a GAME payload or tool arguments. `principal` is an audit
attribution, NOT an authorization grant; Core cannot authenticate the caller, so
any code in the process that holds a scope object and a document id can call this.
It is deliberately not a `ToolDefinition`, not reachable from an Agent, and not
part of the operator lifecycle command.

THE `unchanged` RESULT, PRECISELY
---------------------------------
Every call, including one that turns out to change nothing, is decided inside the
governed boundary: the document is locked, `expected` is verified against that locked
state, and only then are the stored chunks read and compared with the request. A stale
`expected` is therefore always a `KnowledgeMutationConflict`, never `unchanged`, even
when the request happens to match what is stored now. When the locked state matches
`expected` and the request equals the stored set, the transaction is rolled back
untouched: no write and no lifecycle event. `unchanged` is not a committed mutation and
carries no audit event, but it is decided against the same locked state as a change
(on PostgreSQL; SQLite has no row locks, so that serialisation is not exercised here).

LIMITS ARE BOUNDED V1 OPERATING LIMITS
--------------------------------------
`MAX_AUTHORED_CHUNKS`, `MAX_CHUNK_CONTENT_CHARS` and `MAX_AUTHORED_TOTAL_CHARS`
bound transaction size and per-call memory. They are not derived from any search,
embedding or scalability measurement and are not evidence about corpus size.

PostgreSQL concurrency, database-level length enforcement and the ANN shadow
backend are exercised only by the Linux CI; SQLite has no row locks.
"""
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
import re

from django.db import transaction
from django.utils import timezone

from ai_hub.models import ApplicationScope, KnowledgeDocument, KnowledgeDocumentChunk
from ai_hub.services.knowledge_lifecycle import chunk_set_fingerprint, normalize_chunk_text
from ai_hub.services.knowledge_mutation import (
    ExpectedKnowledgeState,
    KnowledgeMutationOperationError,
    KnowledgeMutationPrincipal,
    KnowledgeMutationPrincipalError,
    _governed_knowledge_mutation,
)

OPERATION_AUTHOR_EXPLICIT_CHUNK_SET = "author_explicit_chunk_set"

#: Bounded v1 operating limits (see the module docstring). Not scalability evidence.
MAX_AUTHORED_CHUNKS = 500
MAX_CHUNK_CONTENT_CHARS = 20_000
MAX_AUTHORED_TOTAL_CHARS = 2_000_000
MAX_SECTION_TITLE_CHARS = KnowledgeDocumentChunk._meta.get_field("section_title").max_length

OUTCOME_CHANGED = "changed"
OUTCOME_UNCHANGED = "unchanged"

NOT_AUTHORIZED_MESSAGE = (
    "Knowledge document not found or not accessible for this application scope."
)

# Mirrors the foundation's reason-code shape for early control-argument validation.
# Every authorized call, including `unchanged`, still enters the foundation.
_REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


# --------------------------------------------------------------------------
# Public types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AuthoredChunk:
    """One chunk of the requested set. Position in the sequence is its index."""

    content: str
    section_title: str = ""


@dataclass(frozen=True)
class AuthoringResult:
    """What a call did. `kept` includes `moved`. No event id: the foundation does
    not return one, and the latest event is never assumed to be this call's."""

    outcome: str
    document_id: int
    chunk_count: int
    kept: int
    inserted: int
    removed: int
    moved: int
    chunk_set_fingerprint: str


class KnowledgeAuthoringError(RuntimeError):
    """An authoring request was refused. Nothing was changed."""


class KnowledgeAuthoringNotAuthorized(KnowledgeAuthoringError):
    """Missing document, document of another scope, or inactive scope.

    Deliberately one type with one message, so the error cannot be used to learn
    whether a document id exists.
    """


class KnowledgeAuthoringInvalidInput(KnowledgeAuthoringError):
    """The requested chunk sequence is malformed or over a limit."""


class AuthoringStateRefused(KnowledgeAuthoringError):
    """The document is not in a state this operation may author."""


class AuthorityRefused(AuthoringStateRefused):
    """DERIVED, or UNKNOWN that already has chunks (adjudicate it first)."""


class StatusRefused(AuthoringStateRefused):
    """First authoring requires a DRAFT document."""


class ProvenanceRefused(AuthoringStateRefused):
    """The document carries generation facts, which an authored set must not."""


class SourceRepresentationRefused(AuthoringStateRefused):
    """The document has source text or a source file this set would not sync."""


class ChunkMetadataRefused(AuthoringStateRefused):
    """An existing chunk carries metadata this operation can neither sync nor audit."""


class AuthoringIntegrityError(KnowledgeAuthoringError):
    """The stored set did not match the requested one; the transaction rolls back."""


class _NothingToDo(Exception):
    """Internal: under the lock the request turned out to equal the stored set."""


# --------------------------------------------------------------------------
# Validation (pure; no database)
# --------------------------------------------------------------------------

def _validate_reason_code(reason_code):
    if reason_code and not _REASON_CODE_PATTERN.match(reason_code):
        raise KnowledgeMutationOperationError(
            f"Invalid lifecycle reason code {reason_code!r}: expected a lowercase "
            "slug of 3-64 characters matching [a-z][a-z0-9_]*."
        )


def _is_clean_text(value) -> bool:
    if not isinstance(value, str) or "\x00" in value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _validate_chunks(chunks) -> list:
    """Return the requested chunks as a list, or raise. Never echoes content."""
    if isinstance(chunks, (str, bytes, bytearray)) or not isinstance(chunks, Sequence):
        raise KnowledgeAuthoringInvalidInput(
            "chunks must be a list or tuple of AuthoredChunk."
        )
    desired = list(chunks)
    if not desired:
        raise KnowledgeAuthoringInvalidInput(
            "chunks must not be empty. To remove a document's content, change its "
            "status or delete the document."
        )
    if len(desired) > MAX_AUTHORED_CHUNKS:
        raise KnowledgeAuthoringInvalidInput(
            f"{len(desired)} chunks exceed the limit of {MAX_AUTHORED_CHUNKS}."
        )
    total = 0
    for position, chunk in enumerate(desired, 1):
        if not isinstance(chunk, AuthoredChunk):
            raise KnowledgeAuthoringInvalidInput(
                f"chunk {position} is not an AuthoredChunk."
            )
        if not _is_clean_text(chunk.content):
            raise KnowledgeAuthoringInvalidInput(
                f"chunk {position}: content must be text without NUL or unencodable characters."
            )
        if not _is_clean_text(chunk.section_title):
            raise KnowledgeAuthoringInvalidInput(
                f"chunk {position}: section_title must be text without NUL or unencodable characters."
            )
        if not chunk.content.strip():
            raise KnowledgeAuthoringInvalidInput(f"chunk {position}: content is empty.")
        if len(chunk.content) > MAX_CHUNK_CONTENT_CHARS:
            raise KnowledgeAuthoringInvalidInput(
                f"chunk {position}: content exceeds {MAX_CHUNK_CONTENT_CHARS} characters."
            )
        if len(chunk.section_title) > MAX_SECTION_TITLE_CHARS:
            raise KnowledgeAuthoringInvalidInput(
                f"chunk {position}: section_title exceeds {MAX_SECTION_TITLE_CHARS} characters."
            )
        total += len(chunk.content) + len(chunk.section_title)
    if total > MAX_AUTHORED_TOTAL_CHARS:
        raise KnowledgeAuthoringInvalidInput(
            f"the request exceeds {MAX_AUTHORED_TOTAL_CHARS} characters in total."
        )
    return desired


# --------------------------------------------------------------------------
# Authorization and eligibility
# --------------------------------------------------------------------------

def _scope_is_active(application_scope) -> bool:
    return ApplicationScope.objects.filter(pk=application_scope.pk, is_active=True).exists()


def _lock_active_scope(application_scope):
    """Lock the active authorization root for the rest of the transaction.

    T2's lock order is root-to-leaf: ApplicationScope first, then the governed
    foundation's document and chunk locks. If deactivation already owns the scope
    lock, PostgreSQL waits and re-evaluates `is_active` after it commits; if T2 owns
    it first, deactivation waits until authoring commits or rolls back. SQLite keeps
    the fresh active-state check but provides no row-lock serialization.

    The ORM's exclusive row lock also serializes T2 calls within one scope. That is
    deliberate here: bounded payload validation happens before the lock, and the
    locked region performs database work only (no parsing, embeddings or I/O).
    """
    locked = (
        ApplicationScope.objects.select_for_update()
        .only("pk")
        .filter(pk=application_scope.pk, is_active=True)
        .first()
    )
    if locked is None:
        raise KnowledgeAuthoringNotAuthorized(NOT_AUTHORIZED_MESSAGE)
    return locked


def _owns(document, application_scope) -> bool:
    return document.collection.application_scope_id == application_scope.pk


def _authorized_document(document_id, application_scope):
    """The document, or the one uniform refusal. Reads fresh state."""
    if (
        isinstance(document_id, bool)
        or not isinstance(document_id, int)
        or not isinstance(application_scope, ApplicationScope)
        or application_scope.pk is None
    ):
        raise KnowledgeAuthoringNotAuthorized(NOT_AUTHORIZED_MESSAGE)
    document = (
        KnowledgeDocument.objects.select_related("collection").filter(pk=document_id).first()
    )
    if document is None or not _owns(document, application_scope) or not _scope_is_active(application_scope):
        raise KnowledgeAuthoringNotAuthorized(NOT_AUTHORIZED_MESSAGE)
    return document


def _require_owned_under_lock(document, application_scope):
    if not _owns(document, application_scope) or not _scope_is_active(application_scope):
        raise KnowledgeAuthoringNotAuthorized(NOT_AUTHORIZED_MESSAGE)


def _load_existing(document) -> list:
    return list(
        document.chunks.order_by("chunk_index", "pk").values(
            "pk", "chunk_index", "section_title", "content", "metadata"
        )
    )


def _require_eligible(document, existing):
    modes = KnowledgeDocument.ChunkAuthorityMode
    mode = document.chunk_authority_mode

    if mode == modes.DERIVED:
        raise AuthorityRefused(
            f"Knowledge document #{document.pk} is DERIVED. Its chunks belong to the "
            "generator; use regeneration or repair, not authoring."
        )
    if mode not in (modes.UNKNOWN, modes.EXPLICIT):
        raise AuthorityRefused(
            f"Knowledge document #{document.pk} has an unsupported authority mode."
        )
    # `generator_version` is null when no generator applies, so 0 is a stored
    # provenance claim, not an absence: test it for None, never for truthiness.
    if (
        document.generation_input_fingerprint
        or document.generation_chunk_set_fingerprint
        or document.generator_identity
        or document.generator_version is not None
    ):
        raise ProvenanceRefused(
            f"Knowledge document #{document.pk} carries generation facts, which an "
            "authored chunk set must not; resolve it through repair first."
        )
    if mode == modes.UNKNOWN:
        if existing:
            raise AuthorityRefused(
                f"Knowledge document #{document.pk} is UNKNOWN and already has chunks. "
                "That is a decision for governed adjudication, not for authoring."
            )
        if document.status != KnowledgeDocument.Status.DRAFT:
            raise StatusRefused(
                f"Knowledge document #{document.pk} must be DRAFT for first authoring; "
                "set it to DRAFT, author, then activate it."
            )
    if (document.curated_text or "").strip() or document.source_file:
        raise SourceRepresentationRefused(
            f"Knowledge document #{document.pk} has source text or a source file. "
            "Authoring cannot keep a separate source representation in sync with the "
            "chunks, so it refuses such documents."
        )
    for row in existing:
        if row["metadata"] != {}:
            raise ChunkMetadataRefused(
                f"Knowledge document #{document.pk} has chunks with metadata. Authoring "
                "neither synchronises nor audits chunk metadata; clear it deliberately "
                "first, or keep managing this document directly."
            )


# --------------------------------------------------------------------------
# Identity plan (pure) and application
# --------------------------------------------------------------------------

def _key(section_title, content):
    """Canonical identity of a chunk inside one document: the `c1` normalisation of
    its two fields. Finer than `k1`, which can coincide for different title/content
    splits of the same rendered text."""
    return normalize_chunk_text(section_title), normalize_chunk_text(content)


@dataclass(frozen=True)
class _Plan:
    kept: tuple      # (pk, old_index, new_index)
    inserted: tuple  # (new_index, AuthoredChunk)
    removed: tuple   # pks

    @property
    def moved(self):
        return tuple(item for item in self.kept if item[1] != item[2])

    @property
    def is_noop(self):
        return not self.inserted and not self.removed and not self.moved


def _plan(existing, desired) -> _Plan:
    """Deterministic matching. Existing rows (by index, then pk) wait in a FIFO queue
    per canonical key; each requested chunk, in order, takes the oldest waiting row
    with its key or is inserted. Rows left over are removed."""
    waiting = defaultdict(deque)
    for row in existing:
        waiting[_key(row["section_title"], row["content"])].append(row)
    kept, inserted = [], []
    for position, chunk in enumerate(desired, 1):
        queue = waiting.get(_key(chunk.section_title, chunk.content))
        if queue:
            row = queue.popleft()
            kept.append((row["pk"], row["chunk_index"], position))
        else:
            inserted.append((position, chunk))
    removed = sorted(row["pk"] for queue in waiting.values() for row in queue)
    return _Plan(tuple(kept), tuple(inserted), tuple(removed))


def _desired_rows(desired):
    return [
        {"chunk_index": position, "section_title": chunk.section_title, "content": chunk.content}
        for position, chunk in enumerate(desired, 1)
    ]


def _delete_rows(plan):
    if plan.removed:
        KnowledgeDocumentChunk.objects.filter(pk__in=list(plan.removed)).delete()


def _move_rows(plan, existing, total):
    moves = plan.moved
    if not moves:
        return
    # The unique (document, chunk_index) constraint forbids swapping in place, so
    # moving rows pass through indexes above every existing and every final one.
    ceiling = max([row["chunk_index"] for row in existing] + [total])
    now = timezone.now()
    for offset, (pk, _old, _new) in enumerate(moves, 1):
        KnowledgeDocumentChunk.objects.filter(pk=pk).update(
            chunk_index=ceiling + offset, updated_at=now
        )
    for pk, _old, new_index in moves:
        KnowledgeDocumentChunk.objects.filter(pk=pk).update(
            chunk_index=new_index, updated_at=now
        )


def _insert_rows(document, plan):
    for position, chunk in plan.inserted:
        KnowledgeDocumentChunk.objects.create(
            document=document,
            chunk_index=position,
            section_title=chunk.section_title,
            content=chunk.content,
            token_estimate=max(len(chunk.content.split()), 1),
            metadata={},
        )


def _declare_explicit(document):
    document.chunk_authority_mode = KnowledgeDocument.ChunkAuthorityMode.EXPLICIT
    document.save(update_fields=["chunk_authority_mode", "updated_at"])


def _verify_result(document, desired):
    """Post-condition: the stored set must equal the requested one under `c1`."""
    stored = list(
        document.chunks.order_by("chunk_index").values("chunk_index", "section_title", "content")
    )
    expected_rows = _desired_rows(desired)
    if (
        [row["chunk_index"] for row in stored] != [row["chunk_index"] for row in expected_rows]
        or chunk_set_fingerprint(stored) != chunk_set_fingerprint(expected_rows)
    ):
        raise AuthoringIntegrityError(
            f"Knowledge document #{document.pk}: the stored chunk set does not match the "
            "requested one; nothing was committed."
        )


def _result(outcome, document_id, desired, plan, fingerprint):
    return AuthoringResult(
        outcome=outcome,
        document_id=document_id,
        chunk_count=len(desired),
        kept=len(plan.kept),
        inserted=len(plan.inserted),
        removed=len(plan.removed),
        moved=len(plan.moved),
        chunk_set_fingerprint=fingerprint,
    )


# --------------------------------------------------------------------------
# The operation
# --------------------------------------------------------------------------

def author_explicit_chunk_set(
    document_id,
    chunks,
    *,
    application_scope,
    expected,
    principal,
    reason_code="",
) -> AuthoringResult:
    """Make the chunk set of one document equal `chunks`, or change nothing.

    `expected` MUST be the `ExpectedKnowledgeState` the host observed when it planned
    this update (`ExpectedKnowledgeState.from_snapshot(build_snapshot(document))`). It
    is required so that an unreviewed edit made elsewhere (for example in the Admin)
    is reported as `KnowledgeMutationConflict` instead of being overwritten.

    Accepted: an UNKNOWN, DRAFT document with no chunks, no generation facts and no
    source text or file (first authoring); an EXPLICIT document with no generation
    facts, no source text or file and no chunk metadata (update).
    Refused: DERIVED; UNKNOWN with chunks; UNKNOWN that is not DRAFT; anything
    carrying generation facts, a source representation or chunk metadata.

    Raises `KnowledgeAuthoringNotAuthorized` (uniform), `KnowledgeAuthoringInvalidInput`,
    `AuthoringStateRefused` and its subclasses, `KnowledgeMutationConflict`,
    `AuthoringIntegrityError`, and the foundation's principal/operation errors. Every
    failure leaves zero changes and zero lifecycle events.
    """
    # Authorization is decided first, so that a caller who cannot see a document gets
    # the one uniform refusal whatever else is wrong with the call. `principal` is
    # audit attribution, never authorization.
    document = _authorized_document(document_id, application_scope)

    if not isinstance(expected, ExpectedKnowledgeState):
        raise TypeError("expected must be an ExpectedKnowledgeState.")
    if not isinstance(principal, KnowledgeMutationPrincipal):
        raise KnowledgeMutationPrincipalError(
            "A lifecycle mutation requires a KnowledgeMutationPrincipal."
        )
    principal.validate()
    _validate_reason_code(reason_code)

    desired = _validate_chunks(chunks)
    desired_fingerprint = chunk_set_fingerprint(_desired_rows(desired))

    # There is deliberately no unlocked early return for an identical request. The
    # active scope is locked first, then the governed foundation locks the document
    # and chunks. This root-to-leaf order serializes scope deactivation without
    # changing the shared foundation contract. The compare-and-swap check and the
    # chunk read then see the same locked state, so even a no-op is decided inside
    # the governed boundary. A genuine no-op raises `_NothingToDo`, which rolls the
    # (empty) outer transaction back: no write, no event.
    try:
        with transaction.atomic():
            _lock_active_scope(application_scope)
            with _governed_knowledge_mutation(
                document.pk,
                expected=expected,
                operation=OPERATION_AUTHOR_EXPLICIT_CHUNK_SET,
                principal=principal,
                reason_code=reason_code,
            ) as mutation:
                locked = mutation.document
                _require_owned_under_lock(locked, application_scope)
                existing = _load_existing(locked)
                _require_eligible(locked, existing)
                plan = _plan(existing, desired)
                if plan.is_noop:
                    raise _NothingToDo
                _delete_rows(plan)
                _move_rows(plan, existing, len(desired))
                _insert_rows(locked, plan)
                _declare_explicit(locked)
                _verify_result(locked, desired)
    except _NothingToDo:
        # Rolled back: no write, no event.
        return _result(OUTCOME_UNCHANGED, document.pk, desired, plan, desired_fingerprint)

    return _result(OUTCOME_CHANGED, document.pk, desired, plan, desired_fingerprint)
