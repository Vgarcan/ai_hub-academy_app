"""Operator surface for governed Knowledge lifecycle decisions (Slice 13).

WHAT THIS IS
------------
An ADAPTER. It makes the five existing governed Core operations reachable by a
human, and adds no lifecycle intelligence of its own.

    review  ->  the exact reviewed state is retained  ->  explicit decision
            ->  the SAME ExpectedKnowledgeState       ->  CAS
            ->  governed mutation                     ->  audit

Every eligibility rule, refusal, transaction, lock, compare-and-swap and audit
row belongs to the Core services. This command never pre-checks eligibility,
never decides which action to take, and never retries.

THE ONE RULE THAT MATTERS
-------------------------
The `ExpectedKnowledgeState` submitted at confirmation is the one captured at
review time. It is never rebuilt from the current document afterwards. Doing so
would compare current state against current state, satisfying the compare-and-
swap vacuously and reducing "the state the operator reviewed" to a fiction.

If the document changed in between, `KnowledgeMutationConflict` is the correct
outcome and the operator starts again. There is deliberately no automatic
refresh and no retry.

DISPATCH IS A CLOSED TABLE
--------------------------
`--action` accepts exactly the five Core verb names, mapped by an explicit
literal dict - never `getattr`, never dynamic import. This is a dispatcher at the
adapter layer only; Core still exposes no generic mutation API, and this command
reaches only the public operation-specific verbs - never the internal governed
mutator that backs them.

(That last sentence deliberately avoids naming the internal helper: the Slice 9
boundary test scans this file's raw text, so even a prose mention would read as
a violation.)

SECURITY
--------
Running a management command implies server/runtime access, and that IS the
authorization boundary for this slice - such an operator could bypass every
governed path with raw ORM anyway. The command deliberately does not invent a
second permissions framework.

It remains unreachable from any Agent path: no `ToolDefinition`, no Orchestrator
or GAME dispatch, no signal, no background task.

    A model must never autonomously decide Knowledge authority or discard
    human Knowledge edits.
"""
from django.core.management.base import BaseCommand, CommandError

from ai_hub.models import KnowledgeDocument
from ai_hub.services.knowledge_adjudication import (
    OPERATION_ADJUDICATE_UNKNOWN_AS_EXPLICIT,
    OPERATION_ADOPT_UNKNOWN_AS_DERIVED,
    KnowledgeAdjudicationError,
    adjudicate_unknown_as_explicit,
    adopt_unknown_as_derived,
)
from ai_hub.services.knowledge_lifecycle_review import (
    ReviewEvidenceUnavailable,
    capture_lifecycle_review,
)
from ai_hub.services.knowledge_mutation import (
    KnowledgeMutationConflict,
    KnowledgeMutationOperationError,
    KnowledgeMutationPrincipal,
    KnowledgeMutationPrincipalError,
)
from ai_hub.services.knowledge_regeneration import (
    OPERATION_REGENERATE_DERIVED_CHUNK_SET,
    KnowledgeRegenerationError,
    regenerate_derived_chunk_set,
)
from ai_hub.services.knowledge_repair import (
    OPERATION_ACCEPT_CURRENT_CHUNKS_AS_EXPLICIT,
    OPERATION_DISCARD_MODIFIED_CHUNKS_AND_REGENERATE,
    IneligibleLifecycleStateError,
    KnowledgeRepairError,
    accept_current_chunks_as_explicit,
    discard_modified_chunks_and_regenerate,
)

# Explicit literal mapping. No getattr, no import_module, no registry: an action
# that is not written here cannot be reached, and adding one is a visible diff.
ACTIONS = {
    OPERATION_ADJUDICATE_UNKNOWN_AS_EXPLICIT: adjudicate_unknown_as_explicit,
    OPERATION_ADOPT_UNKNOWN_AS_DERIVED: adopt_unknown_as_derived,
    OPERATION_REGENERATE_DERIVED_CHUNK_SET: regenerate_derived_chunk_set,
    OPERATION_ACCEPT_CURRENT_CHUNKS_AS_EXPLICIT: accept_current_chunks_as_explicit,
    OPERATION_DISCARD_MODIFIED_CHUNKS_AND_REGENERATE: discard_modified_chunks_and_regenerate,
}

CONTENT_WARNING = (
    "WARNING: this command prints full Knowledge content for operator review.\n"
    "Use only in a trusted terminal. Do not run it in CI or shared/logged "
    "sessions."
)

IRREVERSIBLE_ACTIONS = {
    OPERATION_ADJUDICATE_UNKNOWN_AS_EXPLICIT,
    OPERATION_ACCEPT_CURRENT_CHUNKS_AS_EXPLICIT,
}

IRREVERSIBLE_WARNING = (
    "IRREVERSIBLE: this transfers authority to the chunk set shown above. "
    "There is currently no EXPLICIT -> anything lifecycle verb, deliberately, "
    "so this decision cannot be undone through Core."
)

DESTRUCTIVE_WARNING = (
    "DESTRUCTIVE: the current human-edited chunks shown above may be "
    "permanently destroyed and replaced by the candidate. Core does not retain "
    "their content anywhere."
)

# What each decision means, in the operator's terms rather than the code's.
ACTION_MEANING = {
    OPERATION_ADJUDICATE_UNKNOWN_AS_EXPLICIT: (
        "This declares the reviewed current chunks authoritative from now on. "
        "It does not prove historical authorship, and Core makes no such claim."
    ),
    OPERATION_ADOPT_UNKNOWN_AS_DERIVED: (
        "Forward adoption establishes DERIVED authority from this decision "
        "onward. Reproducibility now is NOT historical proof that these chunks "
        "were generated."
    ),
    OPERATION_REGENERATE_DERIVED_CHUNK_SET: (
        "Replaces the chunk set with the generator's current output. Permitted "
        "only because the chunks still match what Core recorded generating."
    ),
    OPERATION_ACCEPT_CURRENT_CHUNKS_AS_EXPLICIT: (
        "Takes authority over the reviewed current chunks and clears the "
        "generation provenance. It does not claim the chunks were always "
        "explicit or that the previous provenance was false."
    ),
    OPERATION_DISCARD_MODIFIED_CHUNKS_AND_REGENERATE: (
        "Discards the ungoverned modifications shown above and restores the "
        "generator's current output."
    ),
}

# Operator guidance per structural ineligibility reason. No message parsing.
REASON_GUIDANCE = {
    IneligibleLifecycleStateError.REASON_NOT_DERIVED: (
        "The document is not DERIVED. UNKNOWN documents go through adjudication "
        "(adjudicate_unknown_as_explicit / adopt_unknown_as_derived); EXPLICIT "
        "is already the authored artifact."
    ),
    IneligibleLifecycleStateError.REASON_CLAIM_VERIFIES: (
        "The DERIVED claim still verifies, so there is no authority ambiguity "
        "for a human to resolve. Nothing to repair."
    ),
    IneligibleLifecycleStateError.REASON_CAPABILITY_DEFECT: (
        "This Core cannot interpret the recorded generator. Upgrade Core rather "
        "than erasing truthful provenance."
    ),
}


class Command(BaseCommand):
    help = (
        "Review one Knowledge document and apply ONE governed lifecycle "
        "decision to it. Prints full Knowledge content for review, then "
        "requires exact typed confirmation. Interactive only: there is no "
        "--yes, --force or batch mode, deliberately. Run knowledge_preflight "
        "first to find documents, and again afterwards to verify the result."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--document", type=int, required=True,
            help="KnowledgeDocument id. Exactly one document per invocation.",
        )
        parser.add_argument(
            "--action", choices=sorted(ACTIONS), required=True,
            help="The governed Core operation to apply.",
        )
        parser.add_argument(
            "--principal-kind", choices=["human", "system"], required=True,
            help="Who is deciding. Never inferred and never defaulted.",
        )
        parser.add_argument(
            "--principal-id", required=True,
            help=(
                "A STABLE OPAQUE handle for the principal, such as a primary key "
                "or username. Audit rows are retained indefinitely: do NOT pass "
                "an email address, personal name or any other contact detail."
            ),
        )
        parser.add_argument(
            "--reason-code", required=True,
            help=(
                "Lowercase slug recording WHY, e.g. 'accidental_edit'. Required "
                "for every action here, which is stricter than some underlying "
                "services: a CLI invocation is always a deliberate decision, and "
                "the lifecycle event has no free-text field."
            ),
        )

    # Separate method so tests can drive confirmation without a terminal.
    def _read_confirmation(self):
        try:
            return input()
        except EOFError:
            return ""

    def handle(self, *args, **options):
        action = options["action"]
        document_id = options["document"]

        principal = self._build_principal(options)
        review = self._capture(document_id, action)

        self._render(review, principal, options["reason_code"])
        self._confirm(review)

        return self._apply(review, principal, options["reason_code"])

    # -- inputs ------------------------------------------------------------

    def _build_principal(self, options):
        principal = KnowledgeMutationPrincipal(
            kind=options["principal_kind"], identifier=options["principal_id"]
        )
        try:
            return principal.validate()
        except KnowledgeMutationPrincipalError as exc:
            raise CommandError(f"Invalid principal: {exc}") from exc

    def _capture(self, document_id, action):
        try:
            return capture_lifecycle_review(document_id, action=action)
        except KnowledgeDocument.DoesNotExist as exc:
            raise CommandError(f"Knowledge document #{document_id} does not exist.") from exc
        except ReviewEvidenceUnavailable as exc:
            raise CommandError(str(exc)) from exc

    # -- review ------------------------------------------------------------

    def _render(self, review, principal, reason_code):
        out = self.stdout
        out.write(self.style.WARNING(CONTENT_WARNING))
        out.write("")
        out.write(f"ACTION      {review.action}")
        out.write(f"            {ACTION_MEANING[review.action]}")
        out.write("")
        out.write(f"DOCUMENT    #{review.document_id}  {review.title}")
        out.write(f"COLLECTION  #{review.collection_id}  {review.collection_name}")
        out.write(f"STATUS      {review.status}")
        out.write(f"AUTHORITY   {review.authority_mode}")
        out.write("")
        out.write("RECORDED GENERATION PROVENANCE")
        out.write(f"  input fingerprint      {review.recorded_input_fingerprint or '(none)'}")
        out.write(f"  chunk-set fingerprint  {review.recorded_chunk_set_fingerprint or '(none)'}")
        out.write(f"  generator identity     {review.generator_identity or '(none)'}")
        out.write(f"  generator version      {review.generator_version}")
        out.write("")
        out.write("OBSERVED NOW")
        out.write(f"  input fingerprint      {review.snapshot.observed_input_fingerprint}")
        out.write(f"  chunk-set fingerprint  {review.snapshot.observed_chunk_set_fingerprint}")
        out.write(f"  chunk count            {review.snapshot.chunk_count}")
        out.write("")

        source_label = "SOURCE (curated_text)"
        if review.action in IRREVERSIBLE_ACTIONS:
            source_label += "  - becomes NON-AUTHORITATIVE relative to EXPLICIT chunks"
        out.write(source_label)
        out.write(review.curated_text or "(empty)")
        out.write("")

        out.write(f"CURRENT CHUNKS ({len(review.chunks)}) - full content")
        if not review.chunks:
            out.write("  (none)")
        for chunk in review.chunks:
            out.write(f"  --- chunk_index={chunk.chunk_index}  section={chunk.section_title!r}")
            out.write(chunk.content)
        out.write("")

        if review.candidate is not None:
            out.write(
                f"CANDIDATE - PREVIEW ONLY ({len(review.candidate)}) - generator "
                f"{review.candidate_generator_identity} "
                f"v{review.candidate_generator_version}"
            )
            out.write(
                "  Recomputed authoritatively by the service under its own lock; "
                "this preview is review evidence, not committed output."
            )
            # The recorded version and the version that will produce the
            # candidate are different facts. Show both whenever they differ, so
            # the operator can see they are authorising a version advance -
            # never presented as an automatic migration or a correction of
            # history, only as the contract the candidate is generated under.
            if review.generator_version is None:
                out.write(
                    f"  Generator version: establishing "
                    f"v{review.candidate_generator_version} "
                    "(the document records none)."
                )
            elif review.generator_version != review.candidate_generator_version:
                out.write(
                    f"  Generator version advance: recorded "
                    f"v{review.generator_version} -> candidate "
                    f"v{review.candidate_generator_version}. The candidate is "
                    "generated under the version this Core currently supports."
                )
            else:
                out.write(
                    f"  Generator version: v{review.candidate_generator_version}, "
                    "unchanged from the recorded version."
                )
            for chunk in review.candidate:
                out.write(f"  --- chunk_index={chunk.chunk_index}  section={chunk.section_title!r}")
                out.write(chunk.content)
            out.write(f"  candidate fingerprint  {review.candidate_fingerprint}")
            if review.candidate_matches_current:
                out.write(
                    "  candidate EQUALS the current chunk set under c1: chunk rows "
                    "and their ids are expected to be PRESERVED."
                )
            else:
                out.write(
                    "  candidate DIFFERS from the current chunk set under c1: chunk "
                    "rows will be replaced and their ids will change. Existing "
                    "chunk_id references to this document stop resolving."
                )
            out.write("")

        if review.action in IRREVERSIBLE_ACTIONS:
            out.write(self.style.WARNING(IRREVERSIBLE_WARNING))
        if review.action == OPERATION_DISCARD_MODIFIED_CHUNKS_AND_REGENERATE:
            out.write(self.style.WARNING(DESTRUCTIVE_WARNING))
        out.write("")
        out.write(f"PRINCIPAL   {principal.kind}:{principal.identifier}")
        out.write(f"REASON      {reason_code}")
        out.write("")

    def _confirm(self, review):
        token = f"{review.action}:{review.document_id}"
        self.stdout.write(f"Type exactly:  {token}")
        if self._read_confirmation().strip() != token:
            raise CommandError(
                "Confirmation did not match. Nothing was changed. Re-run the "
                "command to review the document again."
            )

    # -- apply -------------------------------------------------------------

    def _apply(self, review, principal, reason_code):
        operation = ACTIONS[review.action]
        try:
            # The captured expectation, unchanged. Never rebuilt here.
            document = operation(
                review.document_id,
                expected=review.expected,
                principal=principal,
                reason_code=reason_code,
            )
        except KnowledgeMutationConflict as exc:
            raise CommandError(
                "The Knowledge document changed after you reviewed it "
                f"({exc.field} differs). Nothing was changed. Run the command "
                "again and review the new state."
            ) from exc
        except IneligibleLifecycleStateError as exc:
            raise CommandError(
                f"{review.action} is not available for this document "
                f"({exc.reason}). {REASON_GUIDANCE[exc.reason]} Nothing was "
                "changed, and no other action was taken on your behalf."
            ) from exc
        except (
            KnowledgeAdjudicationError,
            KnowledgeRegenerationError,
            KnowledgeRepairError,
            KnowledgeMutationOperationError,
            KnowledgeMutationPrincipalError,
        ) as exc:
            raise CommandError(
                f"{review.action} was refused: {exc} Nothing was changed. If "
                "another decision is appropriate, run the command again for that "
                "action and review the document afresh."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            raise CommandError(
                f"{review.action} failed unexpectedly: {exc!r}. Nothing is "
                "assumed about the outcome; inspect the document before retrying."
            ) from exc

        self.stdout.write(self.style.SUCCESS(f"{review.action} committed."))
        # Post-commit observation only. Deliberately NOT reused for any further
        # decision, and no lifecycle event id is inferred: querying for "the
        # latest event" could race another writer and print a misleading id.
        self.stdout.write(
            f"Observed after commit: authority={document.chunk_authority_mode} "
            f"generator={document.generator_identity or '(none)'} "
            f"version={document.generator_version} "
            f"chunks={document.chunks.count()}"
        )
        return None
