"""Tests for the Knowledge lifecycle operator surface (Slice 13).

The surface adds no lifecycle intelligence, so these tests are weighted almost
entirely toward the properties that make an operator decision *trustworthy*:

* the snapshot the operator reviewed is the one submitted — never a fresh one;
* the bytes displayed are the bytes the snapshot hashed;
* a stale review conflicts and is never retried;
* nothing happens without exact typed confirmation;
* one action never silently becomes another.
"""
from io import StringIO
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from ai_hub.models import (
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    KnowledgeLifecycleEvent,
)
from ai_hub.services.knowledge_ingestion import ensure_initial_knowledge_chunk
from ai_hub.services.knowledge_lifecycle import (
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK,
    GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION,
    SUPPORTED_GENERATORS,
    chunk_set_fingerprint,
)
from ai_hub.services.knowledge_lifecycle_review import (
    CANDIDATE_ACTIONS,
    REVIEWABLE_ACTIONS,
    ReviewEvidenceUnavailable,
    capture_lifecycle_review,
)
from ai_hub.services.knowledge_mutation import (
    ExpectedKnowledgeState,
    KnowledgeMutationConflict,
    build_snapshot,
)
from ai_hub.services.knowledge_preflight import run_knowledge_preflight
from ai_hub.test_knowledge_regeneration import make_derived, preflight_row
from ai_hub.test_knowledge_repair import make_modified
from ai_hub.test_application_scope_helpers import test_scope

MODES = KnowledgeDocument.ChunkAuthorityMode
COMMAND = "knowledge_lifecycle_action"

ALL_ACTIONS = (
    "adjudicate_unknown_as_explicit",
    "adopt_unknown_as_derived",
    "regenerate_derived_chunk_set",
    "accept_current_chunks_as_explicit",
    "discard_modified_chunks_and_regenerate",
)


def run_command(*, document, action, confirm=None, principal_kind="human",
                principal_id="operator-1", reason_code="operator_reviewed",
                **extra):
    """Drive the command, answering the confirmation prompt with `confirm`."""
    if confirm is None:
        confirm = f"{action}:{document}"
    out, err = StringIO(), StringIO()
    with mock.patch("builtins.input", return_value=confirm):
        call_command(
            COMMAND, document=document, action=action,
            principal_kind=principal_kind, principal_id=principal_id,
            reason_code=reason_code, stdout=out, stderr=err, **extra,
        )
    return out.getvalue()


def make_unknown(collection, title="Unknown", *, curated_text="body", chunks=None):
    document = KnowledgeDocument.objects.create(
        collection=collection, title=title, curated_text=curated_text,
        status=KnowledgeDocument.Status.ACTIVE,
    )
    if chunks is None:
        ensure_initial_knowledge_chunk(document)
    else:
        for index, content in chunks:
            KnowledgeDocumentChunk.objects.create(
                document=document, chunk_index=index,
                section_title=f"S{index}", content=content,
            )
    document.refresh_from_db()
    return document


# ---------------------------------------------------------------------------
# THE load-bearing property: the reviewed snapshot is the submitted one
# ---------------------------------------------------------------------------

class ReviewedSnapshotIsSubmittedTests(TestCase):
    """If this fails, the whole reviewed-state guarantee is theatre."""

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Submitted")
        self.document = make_unknown(self.collection, "Submitted Doc")

    def test_the_captured_expected_state_is_what_the_service_receives(self):
        captured = capture_lifecycle_review(
            self.document.pk, action="adjudicate_unknown_as_explicit"
        )

        seen = {}
        real = __import__(
            "ai_hub.services.knowledge_adjudication", fromlist=["x"]
        ).adjudicate_unknown_as_explicit

        def spy(document_id, *, expected, principal, reason_code=""):
            seen["expected"] = expected
            return real(
                document_id, expected=expected, principal=principal,
                reason_code=reason_code,
            )

        with mock.patch.dict(
            "ai_hub.management.commands.knowledge_lifecycle_action.ACTIONS",
            {"adjudicate_unknown_as_explicit": spy},
        ):
            run_command(document=self.document.pk, action="adjudicate_unknown_as_explicit")

        self.assertEqual(seen["expected"], captured.expected)

    def test_exactly_one_expected_state_is_ever_constructed(self):
        """The vacuous-CAS bug stated as a counting property.

        `ExpectedKnowledgeState.from_snapshot` must be called exactly once, at
        capture. A second construction anywhere on this path would mean a fresh
        expectation was derived from current state and compared against itself.

        Note this deliberately does NOT count `build_snapshot`: the mutation
        foundation legitimately calls it inside its own transaction to compute
        the audit event's before/after facts. That call is not a re-review — it
        is under the lock, and its result is never used as the expectation.
        """
        from ai_hub.services import knowledge_mutation as mutation_module

        real_from_snapshot = mutation_module.ExpectedKnowledgeState.from_snapshot
        calls = []

        def counting_from_snapshot(snapshot):
            calls.append(snapshot.document_id)
            return real_from_snapshot(snapshot)

        with mock.patch.object(
            mutation_module.ExpectedKnowledgeState, "from_snapshot",
            staticmethod(counting_from_snapshot),
        ):
            run_command(
                document=self.document.pk, action="adjudicate_unknown_as_explicit"
            )

        self.assertEqual(calls, [self.document.pk])
        self.document.refresh_from_db()
        self.assertEqual(self.document.chunk_authority_mode, MODES.EXPLICIT)

    def test_capture_happens_exactly_once_per_invocation(self):
        import ai_hub.management.commands.knowledge_lifecycle_action as cmd

        real_capture = cmd.capture_lifecycle_review
        captures = []

        def counting_capture(document_id, *, action):
            captures.append(action)
            return real_capture(document_id, action=action)

        with mock.patch.object(cmd, "capture_lifecycle_review", counting_capture):
            run_command(
                document=self.document.pk, action="adjudicate_unknown_as_explicit"
            )

        self.assertEqual(captures, ["adjudicate_unknown_as_explicit"])

    def test_the_command_never_constructs_expected_state_itself(self):
        """Only the review service may build it, from the captured snapshot."""
        import ast
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parent
            / "management" / "commands" / f"{COMMAND}.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for forbidden in ("ExpectedKnowledgeState", "build_snapshot", "from_snapshot"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(forbidden, names)


# ---------------------------------------------------------------------------
# Review-body / snapshot coherence
# ---------------------------------------------------------------------------

class ReviewCoherenceTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Coherence")

    def test_displayed_chunks_hash_to_the_snapshot_fingerprint(self):
        document = make_unknown(
            self.collection, "Coherent", chunks=((1, "alpha"), (2, "beta")),
        )
        review = capture_lifecycle_review(
            document.pk, action="adjudicate_unknown_as_explicit"
        )
        self.assertEqual(
            chunk_set_fingerprint(review.chunks),
            review.snapshot.observed_chunk_set_fingerprint,
        )

    def test_the_bundle_chunks_are_what_the_command_prints(self):
        document = make_unknown(
            self.collection, "Printed", chunks=((1, "UNIQUE-ALPHA"), (2, "UNIQUE-BETA")),
        )
        output = run_command(document=document.pk, action="adjudicate_unknown_as_explicit")
        for body in ("UNIQUE-ALPHA", "UNIQUE-BETA"):
            self.assertIn(body, output)

    def test_a_change_between_capture_and_display_cannot_occur(self):
        """Regression: display content must come from the captured rows.

        Mutating the database after capture must not change what the bundle
        holds. If display re-fetched, this would drift.
        """
        document = make_unknown(self.collection, "Frozen", chunks=((1, "ORIGINAL"),))
        review = capture_lifecycle_review(
            document.pk, action="adjudicate_unknown_as_explicit"
        )
        chunk = document.chunks.get()
        chunk.content = "CHANGED AFTER CAPTURE"
        chunk.save(update_fields=["content"])

        self.assertEqual(review.chunks[0].content, "ORIGINAL")
        self.assertEqual(
            chunk_set_fingerprint(review.chunks),
            review.snapshot.observed_chunk_set_fingerprint,
        )

    def test_capture_takes_no_locks_and_writes_nothing(self):
        document = make_unknown(self.collection, "NoLocks")
        before = list(
            KnowledgeDocument.objects.values("pk", "chunk_authority_mode")
        )
        capture_lifecycle_review(document.pk, action="adjudicate_unknown_as_explicit")
        self.assertEqual(
            list(KnowledgeDocument.objects.values("pk", "chunk_authority_mode")), before
        )
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_the_review_bundle_is_immutable(self):
        document = make_unknown(self.collection, "Immutable")
        review = capture_lifecycle_review(
            document.pk, action="adjudicate_unknown_as_explicit"
        )
        with self.assertRaises(Exception):
            review.action = "something_else"
        with self.assertRaises(Exception):
            review.chunks[0].content = "tampered"

    def test_capture_is_never_persisted(self):
        from django.apps import apps

        names = {m.__name__ for m in apps.get_app_config("ai_hub").get_models()}
        for forbidden in ("KnowledgeLifecycleReview", "CapturedChunk"):
            self.assertNotIn(forbidden, names)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class ActionDispatchTests(TestCase):
    def test_the_action_set_is_exactly_the_five_core_verbs(self):
        from ai_hub.management.commands.knowledge_lifecycle_action import ACTIONS

        self.assertEqual(set(ACTIONS), set(ALL_ACTIONS))
        self.assertEqual(set(ACTIONS), REVIEWABLE_ACTIONS)

    def test_each_action_maps_to_its_own_core_verb(self):
        from ai_hub.management.commands.knowledge_lifecycle_action import ACTIONS

        for name, callable_ in ACTIONS.items():
            with self.subTest(action=name):
                self.assertEqual(callable_.__name__, name)

    def test_dispatch_is_a_literal_table_not_dynamic_lookup(self):
        import ast
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parent
            / "management" / "commands" / f"{COMMAND}.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for forbidden in ("getattr", "__import__", "import_module", "eval", "exec"):
            with self.subTest(builtin=forbidden):
                self.assertNotIn(forbidden, called)

    def test_an_unknown_action_is_rejected_by_argparse(self):
        with self.assertRaises(CommandError):
            call_command(
                COMMAND, document=1, action="repair_knowledge",
                principal_kind="human", principal_id="x", reason_code="y",
                stdout=StringIO(),
            )


# ---------------------------------------------------------------------------
# All five actions actually work end to end
# ---------------------------------------------------------------------------

class EndToEndActionTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="E2E")

    def test_adjudicate_unknown_as_explicit(self):
        document = make_unknown(self.collection, "E2E Explicit")
        run_command(document=document.pk, action="adjudicate_unknown_as_explicit")
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, MODES.EXPLICIT)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 1)

    def test_adopt_unknown_as_derived(self):
        document = make_unknown(self.collection, "E2E Derived")
        run_command(document=document.pk, action="adopt_unknown_as_derived")
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, MODES.DERIVED)
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "DERIVED_CURRENT")

    def test_regenerate_derived_chunk_set(self):
        document = make_derived(self.collection, "E2E Regen", curated_text="original")
        KnowledgeDocument.objects.filter(pk=document.pk).update(curated_text="a new body")
        document.refresh_from_db()
        run_command(document=document.pk, action="regenerate_derived_chunk_set")
        self.assertEqual(document.chunks.get().content, "a new body")
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "DERIVED_CURRENT")

    def test_accept_current_chunks_as_explicit(self):
        document = make_modified(self.collection, "E2E Accept")
        run_command(document=document.pk, action="accept_current_chunks_as_explicit")
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, MODES.EXPLICIT)
        self.assertEqual(document.generator_identity, "")

    def test_discard_modified_chunks_and_regenerate(self):
        document = make_modified(self.collection, "E2E Discard", curated_text="true body")
        run_command(document=document.pk, action="discard_modified_chunks_and_regenerate")
        self.assertEqual(document.chunks.get().content, "true body")
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "DERIVED_CURRENT")


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------

class ConfirmationContractTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Confirm")
        self.document = make_unknown(self.collection, "Confirm Doc")

    def _assert_aborted(self, confirm):
        with self.assertRaises(CommandError) as caught:
            run_command(
                document=self.document.pk,
                action="adjudicate_unknown_as_explicit", confirm=confirm,
            )
        self.assertIn("Nothing was changed", str(caught.exception))
        self.document.refresh_from_db()
        self.assertEqual(self.document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_yes_style_answers_are_refused(self):
        for answer in ("y", "yes", "Y", "YES", "ok", "1"):
            with self.subTest(answer=answer):
                self._assert_aborted(answer)

    def test_empty_input_aborts(self):
        self._assert_aborted("")

    def test_a_token_for_a_different_document_aborts(self):
        self._assert_aborted(f"adjudicate_unknown_as_explicit:{self.document.pk + 1}")

    def test_a_token_for_a_different_action_aborts(self):
        self._assert_aborted(f"adopt_unknown_as_derived:{self.document.pk}")

    def test_the_exact_token_is_shown_to_the_operator(self):
        output = run_command(
            document=self.document.pk, action="adjudicate_unknown_as_explicit"
        )
        self.assertIn(f"adjudicate_unknown_as_explicit:{self.document.pk}", output)

    def test_surrounding_whitespace_is_tolerated(self):
        run_command(
            document=self.document.pk, action="adjudicate_unknown_as_explicit",
            confirm=f"  adjudicate_unknown_as_explicit:{self.document.pk}  ",
        )
        self.document.refresh_from_db()
        self.assertEqual(self.document.chunk_authority_mode, MODES.EXPLICIT)


# ---------------------------------------------------------------------------
# Stale review
# ---------------------------------------------------------------------------

class StaleReviewTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Stale")

    def test_a_change_after_capture_conflicts_and_is_not_retried(self):
        document = make_unknown(self.collection, "Stale Doc", chunks=((1, "reviewed"),))

        import ai_hub.management.commands.knowledge_lifecycle_action as cmd
        real_capture = cmd.capture_lifecycle_review
        attempts = []

        def capture_then_mutate(document_id, *, action):
            review = real_capture(document_id, action=action)
            # The database moves on AFTER the operator reviewed it.
            chunk = KnowledgeDocumentChunk.objects.get(document_id=document_id)
            chunk.content = "changed after review"
            chunk.save(update_fields=["content"])
            attempts.append(review)
            return review

        with mock.patch.object(cmd, "capture_lifecycle_review", capture_then_mutate):
            with self.assertRaises(CommandError) as caught:
                run_command(
                    document=document.pk, action="adjudicate_unknown_as_explicit"
                )

        message = str(caught.exception)
        self.assertIn("changed after you reviewed it", message)
        self.assertIn("Nothing was changed", message)
        self.assertIsInstance(caught.exception.__cause__, KnowledgeMutationConflict)
        # Captured exactly once: no automatic re-review, no retry.
        self.assertEqual(len(attempts), 1)
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)


# ---------------------------------------------------------------------------
# Principal and reason
# ---------------------------------------------------------------------------

class PrincipalAndReasonTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Principal")
        self.document = make_unknown(self.collection, "Principal Doc")

    def _required(self, **omit):
        options = dict(
            document=self.document.pk, action="adjudicate_unknown_as_explicit",
            principal_kind="human", principal_id="operator-1",
            reason_code="operator_reviewed",
        )
        options.update(omit)
        options = {k: v for k, v in options.items() if v is not None}
        with mock.patch("builtins.input", return_value="x"):
            call_command(COMMAND, stdout=StringIO(), **options)

    def test_principal_kind_is_required(self):
        with self.assertRaises(CommandError):
            self._required(principal_kind=None)

    def test_principal_id_is_required(self):
        with self.assertRaises(CommandError):
            self._required(principal_id=None)

    def test_reason_code_is_required(self):
        with self.assertRaises(CommandError):
            self._required(reason_code=None)

    def test_there_is_no_default_human_principal(self):
        from ai_hub.management.commands.knowledge_lifecycle_action import Command
        import argparse

        parser = argparse.ArgumentParser()
        Command().add_arguments(parser)
        for action in parser._actions:
            if action.dest in {"principal_kind", "principal_id", "reason_code", "action", "document"}:
                with self.subTest(option=action.dest):
                    self.assertTrue(action.required)
                    self.assertIsNone(action.default)

    def test_a_blank_principal_identifier_is_refused(self):
        with self.assertRaises(CommandError) as caught:
            run_command(
                document=self.document.pk,
                action="adjudicate_unknown_as_explicit", principal_id="   ",
            )
        self.assertIn("Invalid principal", str(caught.exception))
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_a_malformed_reason_code_is_refused_before_mutation(self):
        with self.assertRaises(CommandError):
            run_command(
                document=self.document.pk,
                action="adjudicate_unknown_as_explicit",
                reason_code="Not A Slug",
            )
        self.document.refresh_from_db()
        self.assertEqual(self.document.chunk_authority_mode, MODES.UNKNOWN)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_the_reason_reaches_the_lifecycle_event(self):
        run_command(
            document=self.document.pk, action="adjudicate_unknown_as_explicit",
            reason_code="hand_authored",
        )
        self.assertEqual(KnowledgeLifecycleEvent.objects.get().reason_code, "hand_authored")

    def test_a_system_principal_is_recorded_truthfully(self):
        run_command(
            document=self.document.pk, action="adjudicate_unknown_as_explicit",
            principal_kind="system", principal_id="operator-console",
        )
        event = KnowledgeLifecycleEvent.objects.get()
        self.assertEqual(event.principal_kind, "system")
        self.assertEqual(event.principal_identifier, "operator-console")

    def test_the_help_warns_against_personal_identifiers(self):
        from ai_hub.management.commands.knowledge_lifecycle_action import Command
        import argparse

        parser = argparse.ArgumentParser()
        Command().add_arguments(parser)
        help_text = next(
            a.help for a in parser._actions if a.dest == "principal_id"
        ).lower()
        for token in ("opaque", "email"):
            self.assertIn(token, help_text)

    def test_the_cli_requires_a_reason_even_where_core_does_not(self):
        """Slice 10/11 services keep reason optional; the CLI does not."""
        import inspect

        from ai_hub.services.knowledge_adjudication import adjudicate_unknown_as_explicit

        signature = inspect.signature(adjudicate_unknown_as_explicit)
        self.assertEqual(signature.parameters["reason_code"].default, "")


# ---------------------------------------------------------------------------
# Candidate preview
# ---------------------------------------------------------------------------

class CandidatePreviewTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Preview")

    def test_only_generator_backed_actions_carry_a_candidate(self):
        self.assertEqual(
            CANDIDATE_ACTIONS,
            frozenset({
                "adopt_unknown_as_derived",
                "regenerate_derived_chunk_set",
                "discard_modified_chunks_and_regenerate",
            }),
        )

    def test_candidate_content_and_preview_label_appear(self):
        document = make_modified(self.collection, "Preview Shown", curated_text="CANDIDATE-BODY")
        out = StringIO()
        with mock.patch("builtins.input", return_value="nope"):
            try:
                call_command(
                    COMMAND, document=document.pk,
                    action="discard_modified_chunks_and_regenerate",
                    principal_kind="human", principal_id="op",
                    reason_code="accidental_edit", stdout=out,
                )
            except CommandError:
                pass
        output = out.getvalue()
        self.assertIn("CANDIDATE-BODY", output)
        self.assertIn("PREVIEW ONLY", output)

    def test_the_preview_is_never_passed_to_the_service(self):
        """The service must recompute; the preview is review evidence only."""
        import inspect

        from ai_hub.services import knowledge_repair

        signature = inspect.signature(knowledge_repair.discard_modified_chunks_and_regenerate)
        self.assertEqual(
            set(signature.parameters),
            {"document_id", "expected", "principal", "reason_code"},
        )

    def test_candidate_equality_is_reported_for_identity_consequences(self):
        document = make_derived(self.collection, "Equal", curated_text="same")
        review = capture_lifecycle_review(
            document.pk, action="regenerate_derived_chunk_set"
        )
        self.assertTrue(review.candidate_matches_current)

        changed = make_derived(self.collection, "Differs", curated_text="before")
        KnowledgeDocument.objects.filter(pk=changed.pk).update(curated_text="after")
        changed.refresh_from_db()
        self.assertFalse(
            capture_lifecycle_review(
                changed.pk, action="regenerate_derived_chunk_set"
            ).candidate_matches_current
        )

    def test_actions_without_a_candidate_have_none(self):
        document = make_unknown(self.collection, "No Candidate")
        review = capture_lifecycle_review(
            document.pk, action="adjudicate_unknown_as_explicit"
        )
        self.assertIsNone(review.candidate)
        self.assertIsNone(review.candidate_matches_current)

    def test_an_unbuildable_candidate_stops_before_confirmation(self):
        """Failure to assemble evidence is not a lifecycle verdict."""
        document = make_modified(self.collection, "Unsupported")
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            generator_identity="some_future_generator"
        )
        document.refresh_from_db()

        with self.assertRaises(ReviewEvidenceUnavailable):
            capture_lifecycle_review(
                document.pk, action="discard_modified_chunks_and_regenerate"
            )

        with mock.patch("builtins.input") as prompt:
            with self.assertRaises(CommandError) as caught:
                run_command(
                    document=document.pk,
                    action="discard_modified_chunks_and_regenerate",
                )
        prompt.assert_not_called()
        self.assertIn("Nothing was changed", str(caught.exception))
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)


class CandidateGeneratorVersionTests(TestCase):
    """The candidate version must describe the contract that will RECORD it.

    Not the version the document currently claims. Those are different facts,
    and conflating them made the evidence untrue in every generator-backed case:
    adoption showed `None` while the service would record v1, and an outdated
    document showed its stale version rather than the one regeneration advances
    to. The operator has to be able to see which contract version they are
    establishing.
    """

    CURRENT = GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Candidate Version")

    def _rendered(self, document, action):
        out = StringIO()
        with mock.patch("builtins.input", return_value="abort"):
            try:
                call_command(
                    COMMAND, document=document.pk, action=action,
                    principal_kind="human", principal_id="op",
                    reason_code="operator_reviewed", stdout=out,
                )
            except CommandError:
                pass
        return out.getvalue()

    # -- adoption -----------------------------------------------------------

    def test_adoption_reports_the_version_being_established_not_none(self):
        document = make_unknown(self.collection, "Adopt Version")
        self.assertIsNone(document.generator_version)

        review = capture_lifecycle_review(
            document.pk, action="adopt_unknown_as_derived"
        )

        self.assertIsNotNone(review.candidate_generator_version)
        self.assertEqual(
            review.candidate_generator_version,
            SUPPORTED_GENERATORS[review.candidate_generator_identity],
        )
        self.assertEqual(review.candidate_generator_version, self.CURRENT)
        # The document's own recorded fact is untouched and still absent.
        self.assertIsNone(review.generator_version)

    def test_adoption_renders_the_version_and_says_it_is_being_established(self):
        document = make_unknown(self.collection, "Adopt Render")
        output = self._rendered(document, "adopt_unknown_as_derived")
        self.assertIn(f"curated_text_single_chunk v{self.CURRENT}", output)
        self.assertIn(f"establishing v{self.CURRENT}", output)
        self.assertIn("the document records none", output)

    def test_the_adopted_version_is_the_one_the_service_records(self):
        """Evidence and outcome must agree."""
        document = make_unknown(self.collection, "Adopt Agrees")
        review = capture_lifecycle_review(
            document.pk, action="adopt_unknown_as_derived"
        )
        run_command(document=document.pk, action="adopt_unknown_as_derived")
        document.refresh_from_db()
        self.assertEqual(document.generator_version, review.candidate_generator_version)

    # -- outdated regeneration ---------------------------------------------

    def test_outdated_regeneration_reports_both_versions(self):
        document = make_derived(
            self.collection, "Outdated Version", version=self.CURRENT - 1
        )
        review = capture_lifecycle_review(
            document.pk, action="regenerate_derived_chunk_set"
        )
        self.assertEqual(review.generator_version, self.CURRENT - 1)
        self.assertEqual(review.candidate_generator_version, self.CURRENT)

    def test_outdated_regeneration_renders_the_advance(self):
        document = make_derived(
            self.collection, "Outdated Render", version=self.CURRENT - 1
        )
        output = self._rendered(document, "regenerate_derived_chunk_set")
        self.assertIn(
            f"advance: recorded v{self.CURRENT - 1} -> candidate v{self.CURRENT}",
            output,
        )
        self.assertIn("version this Core currently supports", output)
        # Never framed as a migration or a correction of history.
        for misleading in ("migration", "historical correction", "auto-upgrade"):
            self.assertNotIn(misleading, output.lower())

    def test_the_advanced_version_is_the_one_the_service_records(self):
        document = make_derived(
            self.collection, "Advance Agrees", version=self.CURRENT - 1
        )
        review = capture_lifecycle_review(
            document.pk, action="regenerate_derived_chunk_set"
        )
        run_command(document=document.pk, action="regenerate_derived_chunk_set")
        document.refresh_from_db()
        self.assertEqual(document.generator_version, review.candidate_generator_version)

    # -- current version ----------------------------------------------------

    def test_a_current_version_document_claims_no_upgrade(self):
        document = make_derived(self.collection, "Current Version")
        review = capture_lifecycle_review(
            document.pk, action="regenerate_derived_chunk_set"
        )
        self.assertEqual(review.generator_version, review.candidate_generator_version)

        output = self._rendered(document, "regenerate_derived_chunk_set")
        self.assertIn(f"Generator version: v{self.CURRENT}, unchanged", output)
        self.assertNotIn("advance", output.lower())

    # -- discard ------------------------------------------------------------

    def test_discard_reports_the_recording_version_too(self):
        document = make_modified(
            self.collection, "Discard Version", curated_text="true body"
        )
        review = capture_lifecycle_review(
            document.pk, action="discard_modified_chunks_and_regenerate"
        )
        self.assertEqual(
            review.candidate_generator_version,
            SUPPORTED_GENERATORS[review.candidate_generator_identity],
        )

        output = self._rendered(document, "discard_modified_chunks_and_regenerate")
        self.assertIn(f"curated_text_single_chunk v{self.CURRENT}", output)

    def test_discard_on_an_outdated_document_shows_the_advance(self):
        document = make_modified(
            self.collection, "Discard Outdated", curated_text="true body"
        )
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            generator_version=self.CURRENT - 1
        )
        document.refresh_from_db()
        review = capture_lifecycle_review(
            document.pk, action="discard_modified_chunks_and_regenerate"
        )
        self.assertEqual(review.generator_version, self.CURRENT - 1)
        self.assertEqual(review.candidate_generator_version, self.CURRENT)
        self.assertIn(
            f"advance: recorded v{self.CURRENT - 1} -> candidate v{self.CURRENT}",
            self._rendered(document, "discard_modified_chunks_and_regenerate"),
        )

    # -- the version is never displayed without its identity ----------------

    def test_every_candidate_action_renders_identity_and_version_together(self):
        cases = (
            ("adopt_unknown_as_derived", make_unknown(self.collection, "V Adopt")),
            ("regenerate_derived_chunk_set", make_derived(self.collection, "V Regen")),
            (
                "discard_modified_chunks_and_regenerate",
                make_modified(self.collection, "V Discard"),
            ),
        )
        for action, document in cases:
            with self.subTest(action=action):
                output = self._rendered(document, action)
                self.assertIn(
                    f"curated_text_single_chunk v{self.CURRENT}", output,
                    "identity must never be shown without its version",
                )

    def test_the_preview_version_is_never_passed_to_the_service(self):
        """Evidence only: the service resolves its own version under the lock."""
        import inspect

        from ai_hub.services import knowledge_adjudication, knowledge_regeneration

        for verb in (
            knowledge_adjudication.adopt_unknown_as_derived,
            knowledge_regeneration.regenerate_derived_chunk_set,
        ):
            with self.subTest(verb=verb.__name__):
                self.assertEqual(
                    set(inspect.signature(verb).parameters),
                    {"document_id", "expected", "principal", "reason_code"},
                )


class CandidateVersionDirectionTests(TestCase):
    """The rendered DIRECTION between recorded and candidate must be true.

    There are four relationships, not three. An earlier revision collapsed
    every unequal pair into "advance", which stated the direction falsely
    whenever the document recorded a version this Core has not reached:
    `recorded v2 -> candidate v1` is not an advance.

    Nothing about eligibility is tested here - these are display facts. The
    service remains the only thing that decides whether an operation is
    permitted, and the later tests in this class pin that it still refuses the
    version-ahead relationship exactly as before.
    """

    CURRENT = GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION
    AHEAD = GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION + 1
    BEHIND = GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION - 1

    ADVANCE = "Generator version advance"
    LOWER = "Candidate generator version is LOWER than recorded"
    ESTABLISHING = "Generator version: establishing"
    UNCHANGED = "unchanged from the recorded version"

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Version Direction")

    def _rendered(self, document, action):
        """Render the review, then abort. No service call, no writes."""
        out = StringIO()
        with mock.patch("builtins.input", return_value="abort"):
            try:
                call_command(
                    COMMAND, document=document.pk, action=action,
                    principal_kind="human", principal_id="op",
                    reason_code="operator_reviewed", stdout=out,
                )
            except CommandError:
                pass
        return out.getvalue()

    def _at_version(self, document, version):
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            generator_version=version
        )
        document.refresh_from_db()
        return document

    def _facts(self, document_id):
        return KnowledgeDocument.objects.filter(pk=document_id).values(
            "chunk_authority_mode", "status", "generator_identity",
            "generator_version", "generation_input_fingerprint",
            "generation_chunk_set_fingerprint",
        ).first()

    # -- 1. no recorded version --------------------------------------------

    def test_no_recorded_version_establishes_and_claims_no_direction(self):
        document = make_unknown(self.collection, "Direction None")
        self.assertIsNone(document.generator_version)

        output = self._rendered(document, "adopt_unknown_as_derived")

        self.assertIn(f"{self.ESTABLISHING} v{self.CURRENT}", output)
        self.assertIn("the document records none", output)
        # A first version is neither an advance nor a reduction.
        self.assertNotIn(self.ADVANCE, output)
        self.assertNotIn(self.LOWER, output)

    # -- 2. recorded older than candidate ----------------------------------

    def test_older_recorded_version_renders_an_advance(self):
        document = make_derived(self.collection, "Direction Older", version=self.BEHIND)

        output = self._rendered(document, "regenerate_derived_chunk_set")

        self.assertIn(
            f"{self.ADVANCE}: recorded v{self.BEHIND} -> candidate v{self.CURRENT}",
            output,
        )
        self.assertNotIn(self.LOWER, output)

    # -- 3. equal ----------------------------------------------------------

    def test_equal_versions_render_unchanged_and_no_direction(self):
        document = make_derived(self.collection, "Direction Equal")
        self.assertEqual(document.generator_version, self.CURRENT)

        output = self._rendered(document, "regenerate_derived_chunk_set")

        self.assertIn(f"Generator version: v{self.CURRENT}, {self.UNCHANGED}", output)
        self.assertNotIn(self.ADVANCE, output)
        self.assertNotIn(self.LOWER, output)

    # -- 4. recorded ahead of candidate ------------------------------------
    #
    # Pinned for BOTH regenerating verbs: each renders candidate evidence
    # before the service subsequently refuses the version-ahead relationship.

    def _ahead_documents(self):
        clean = self._at_version(
            make_derived(self.collection, "Direction Ahead Clean"), self.AHEAD
        )
        modified = self._at_version(
            make_modified(self.collection, "Direction Ahead Modified"), self.AHEAD
        )
        return (
            ("regenerate_derived_chunk_set", clean),
            ("discard_modified_chunks_and_regenerate", modified),
        )

    def test_ahead_recorded_version_is_never_called_an_advance(self):
        for action, document in self._ahead_documents():
            with self.subTest(action=action):
                output = self._rendered(document, action)
                self.assertNotIn(self.ADVANCE, output)
                # Not merely the exact phrase: the WORD must not appear at all
                # on this branch, in any casing.
                self.assertNotIn("advance", output.lower())

    def test_ahead_recorded_version_shows_both_versions_and_the_direction(self):
        for action, document in self._ahead_documents():
            with self.subTest(action=action):
                output = self._rendered(document, action)
                self.assertIn(
                    f"{self.LOWER}: recorded v{self.AHEAD} -> candidate "
                    f"v{self.CURRENT}",
                    output,
                )
                # Both versions are legible on their own, not only inside the
                # combined sentence.
                self.assertIn(f"v{self.AHEAD}", output)
                self.assertIn(f"v{self.CURRENT}", output)

    def test_ahead_render_says_the_preview_is_not_the_decision(self):
        for action, document in self._ahead_documents():
            with self.subTest(action=action):
                output = self._rendered(document, action)
                self.assertIn(
                    "the selected Core service determines whether the operation "
                    "is permitted",
                    output,
                )
                self.assertIn(
                    "this preview is review evidence, not committed output",
                    output,
                )
                # The command must not present itself as having decided or
                # performed anything.
                for overclaim in (
                    "downgrading", "core has downgraded", "will downgrade the",
                ):
                    self.assertNotIn(overclaim, output.lower())

    # -- the display change did not touch eligibility ----------------------

    def test_version_ahead_is_still_refused_by_the_service_for_both_verbs(self):
        for action, document in self._ahead_documents():
            with self.subTest(action=action):
                before = self._facts(document.pk)
                events = KnowledgeLifecycleEvent.objects.count()

                with self.assertRaises(CommandError) as raised:
                    run_command(document=document.pk, action=action)

                self.assertIn("newer than the version", str(raised.exception))
                self.assertEqual(KnowledgeLifecycleEvent.objects.count(), events)
                self.assertEqual(self._facts(document.pk), before)

    def test_modified_chunks_survive_the_refused_discard(self):
        """Version-ahead refusal must not destroy the human edit it previewed."""
        document = self._at_version(
            make_modified(
                self.collection, "Ahead Modified Survives",
                edit="a human edit that must survive the refusal",
            ),
            self.AHEAD,
        )
        chunk = document.chunks.get()

        with self.assertRaises(CommandError):
            run_command(
                document=document.pk,
                action="discard_modified_chunks_and_regenerate",
            )

        chunk.refresh_from_db()
        self.assertEqual(chunk.content, "a human edit that must survive the refusal")

    def test_accept_as_explicit_still_ignores_the_direction_rendering(self):
        """Chunk-modification dominance for `accept` is untouched.

        `accept_current_chunks_as_explicit` never executes the generator, so a
        version-ahead document with MODIFIED chunks remains acceptable. That is
        the Slice 12 dominance rule, and a display correction must not disturb
        it. The verb also renders no candidate at all, so no direction line can
        appear on its path.
        """
        document = self._at_version(
            make_modified(self.collection, "Ahead Modified Accept"), self.AHEAD
        )

        output = self._rendered(document, "accept_current_chunks_as_explicit")
        self.assertNotIn("CANDIDATE", output)
        self.assertNotIn(self.ADVANCE, output)
        self.assertNotIn(self.LOWER, output)

        run_command(
            document=document.pk, action="accept_current_chunks_as_explicit"
        )
        document.refresh_from_db()
        self.assertEqual(
            document.chunk_authority_mode,
            KnowledgeDocument.ChunkAuthorityMode.EXPLICIT,
        )

    # -- every branch is exercised, and exactly one fires ------------------

    def test_exactly_one_direction_line_is_rendered_per_review(self):
        cases = (
            ("adopt_unknown_as_derived",
             make_unknown(self.collection, "One Line None")),
            ("regenerate_derived_chunk_set",
             make_derived(self.collection, "One Line Older", version=self.BEHIND)),
            ("regenerate_derived_chunk_set",
             make_derived(self.collection, "One Line Equal")),
            ("regenerate_derived_chunk_set",
             self._at_version(
                 make_derived(self.collection, "One Line Ahead"), self.AHEAD)),
        )
        for action, document in cases:
            with self.subTest(document=document.title):
                output = self._rendered(document, action)
                fired = [
                    marker for marker in (
                        self.ESTABLISHING, self.ADVANCE, self.LOWER, self.UNCHANGED,
                    )
                    if marker in output
                ]
                self.assertEqual(len(fired), 1, fired)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

class WarningTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Warnings")

    def _output(self, document, action):
        out = StringIO()
        with mock.patch("builtins.input", return_value="abort"):
            try:
                call_command(
                    COMMAND, document=document.pk, action=action,
                    principal_kind="human", principal_id="op",
                    reason_code="operator_reviewed", stdout=out,
                )
            except CommandError:
                pass
        return out.getvalue()

    def test_the_full_content_terminal_warning_is_always_shown(self):
        document = make_unknown(self.collection, "Warned")
        output = self._output(document, "adjudicate_unknown_as_explicit")
        self.assertIn("prints full Knowledge content", output)
        self.assertIn("Do not run it in CI", output)

    def test_irreversible_warning_for_authority_transfers(self):
        for title, action, factory in (
            ("Irr Adjudicate", "adjudicate_unknown_as_explicit", make_unknown),
            ("Irr Accept", "accept_current_chunks_as_explicit", make_modified),
        ):
            with self.subTest(action=action):
                document = factory(self.collection, title)
                output = self._output(document, action)
                self.assertIn("IRREVERSIBLE", output)
                self.assertIn("no EXPLICIT -> anything", output)

    def test_destructive_warning_for_discard(self):
        document = make_modified(self.collection, "Destructive")
        output = self._output(document, "discard_modified_chunks_and_regenerate")
        self.assertIn("DESTRUCTIVE", output)
        self.assertIn("permanently destroyed", output)

    def test_adoption_states_it_is_not_historical_proof(self):
        document = make_unknown(self.collection, "Forward")
        output = self._output(document, "adopt_unknown_as_derived")
        self.assertIn("NOT historical proof", output)

    def test_source_is_labelled_non_authoritative_for_explicit_transfers(self):
        document = make_modified(self.collection, "Labelled")
        output = self._output(document, "accept_current_chunks_as_explicit")
        self.assertIn("NON-AUTHORITATIVE", output)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorHandlingTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Errors")

    def test_a_missing_document_fails_cleanly(self):
        with self.assertRaises(CommandError) as caught:
            run_command(document=999999, action="adjudicate_unknown_as_explicit")
        self.assertIn("does not exist", str(caught.exception))

    def test_an_ineligible_state_uses_the_structural_reason(self):
        document = make_derived(self.collection, "Verifies")
        with self.assertRaises(CommandError) as caught:
            run_command(
                document=document.pk, action="accept_current_chunks_as_explicit"
            )
        message = str(caught.exception)
        self.assertIn("claim_verifies", message)
        self.assertIn("Nothing to repair", message)
        self.assertIn("no other action was taken on your behalf", message)

    def test_the_capability_defect_reason_says_upgrade_core(self):
        document = make_derived(self.collection, "Capability")
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            generator_identity="some_future_generator"
        )
        document.refresh_from_db()
        with self.assertRaises(CommandError) as caught:
            run_command(
                document=document.pk, action="accept_current_chunks_as_explicit"
            )
        self.assertIn("capability_defect", str(caught.exception))
        self.assertIn("Upgrade Core", str(caught.exception))

    def test_a_typed_lifecycle_refusal_is_a_clean_command_error(self):
        """The review builds fine; the SERVICE is what refuses."""
        document = make_unknown(
            self.collection, "Not Reproducible",
            chunks=((1, "hand written, not generator output"),),
        )
        with self.assertRaises(CommandError) as caught:
            run_command(document=document.pk, action="adopt_unknown_as_derived")
        self.assertIn("was refused", str(caught.exception))
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_missing_review_evidence_stops_without_a_lifecycle_verdict(self):
        """An UNKNOWN document has no generator identity to preview with.

        The command must say it cannot assemble evidence, and must NOT assert
        anything about whether the operation would have been allowed - that
        judgement belongs to the service, which is never reached.
        """
        document = make_unknown(self.collection, "No Identity")
        with mock.patch("builtins.input") as prompt:
            with self.assertRaises(CommandError) as caught:
                run_command(
                    document=document.pk, action="regenerate_derived_chunk_set"
                )
        prompt.assert_not_called()
        message = str(caught.exception)
        self.assertIn("no generator identity", message)
        self.assertIn("NOT a verdict", message)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)

    def test_an_unexpected_exception_is_not_swallowed(self):
        document = make_unknown(self.collection, "Boom")
        with mock.patch.dict(
            "ai_hub.management.commands.knowledge_lifecycle_action.ACTIONS",
            {"adjudicate_unknown_as_explicit": mock.Mock(
                side_effect=RuntimeError("kaboom")
            )},
        ):
            with self.assertRaises(CommandError) as caught:
                run_command(
                    document=document.pk, action="adjudicate_unknown_as_explicit"
                )
        self.assertIn("failed unexpectedly", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_no_automatic_fallback_to_another_verb(self):
        """A refusal must never become a different lifecycle decision."""
        document = make_modified(self.collection, "No Fallback")
        KnowledgeDocument.objects.filter(pk=document.pk).update(
            generator_version=GENERATOR_CURATED_TEXT_SINGLE_CHUNK_VERSION + 1
        )
        document.refresh_from_db()
        with self.assertRaises(CommandError):
            run_command(
                document=document.pk,
                action="discard_modified_chunks_and_regenerate",
                reason_code="accidental_edit",
            )
        document.refresh_from_db()
        self.assertEqual(document.chunk_authority_mode, MODES.DERIVED)
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)


# ---------------------------------------------------------------------------
# CLI safety and isolation
# ---------------------------------------------------------------------------

class OperatorSurfaceSafetyTests(TestCase):
    def _parser_options(self):
        import argparse

        from ai_hub.management.commands.knowledge_lifecycle_action import Command

        parser = argparse.ArgumentParser()
        Command().add_arguments(parser)
        return {opt for action in parser._actions for opt in action.option_strings}

    def test_there_is_no_automation_bypass(self):
        options = self._parser_options()
        for forbidden in ("--yes", "--force", "--noinput", "--no-input",
                          "--non-interactive", "--all", "--batch", "--collection"):
            with self.subTest(option=forbidden):
                self.assertNotIn(forbidden, options)

    def test_only_one_document_can_be_targeted(self):
        import argparse

        from ai_hub.management.commands.knowledge_lifecycle_action import Command

        parser = argparse.ArgumentParser()
        Command().add_arguments(parser)
        document = next(a for a in parser._actions if a.dest == "document")
        self.assertEqual(document.type, int)
        self.assertIsNone(document.nargs)

    def test_the_command_does_not_import_the_internal_mutator(self):
        """AST-checked, so prose in a docstring can neither pass nor fail it."""
        import ast
        import pathlib

        tree = ast.parse(
            (
                pathlib.Path(__file__).resolve().parent
                / "management" / "commands" / f"{COMMAND}.py"
            ).read_text(encoding="utf-8")
        )
        imported = {
            alias.name for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) for alias in node.names
        }
        referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        referenced |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for forbidden in ("_governed_knowledge_mutation", "_GovernedMutation"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(forbidden, imported)
                self.assertNotIn(forbidden, referenced)

    def test_no_agent_facing_module_imports_the_operator_surface(self):
        import pathlib

        base = pathlib.Path(__file__).resolve().parent
        modules = (
            "services/knowledge_retrieval.py",
            "services/game_action_dispatcher.py",
            "services/agent_runtime.py",
            "services/tools_runtime.py",
            "services/tool_resolution.py",
            "services/knowledge_ingestion.py",
            "services/knowledge_preflight.py",
            "admin.py",
            "services/build_console.py",
        )
        for relative in modules:
            path = base / relative
            self.assertTrue(path.exists(), f"{relative} missing; scan would lapse")
            with self.subTest(module=relative):
                source = path.read_text(encoding="utf-8")
                for symbol in (
                    "knowledge_lifecycle_action", "knowledge_lifecycle_review",
                    "capture_lifecycle_review",
                ):
                    self.assertNotIn(symbol, source)

    def test_no_seeded_tool_definition_exposes_the_operator_surface(self):
        from ai_hub.models import ToolDefinition
        from ai_hub.services.starter_toolboxes import seed_starter_toolboxes

        test_scope()  # the seed requires an existing scope
        seed_starter_toolboxes()
        self.assertGreater(ToolDefinition.objects.count(), 0)
        for tool in ToolDefinition.objects.all():
            with self.subTest(tool=tool.name):
                blob = " ".join(
                    str(p) for p in (tool.name, tool.label, tool.description, tool.config)
                ).lower()
                for forbidden in ("lifecycle_action", "lifecycle_review", "adjudicat"):
                    self.assertNotIn(forbidden, blob)


# ---------------------------------------------------------------------------
# Preflight separation and regressions
# ---------------------------------------------------------------------------

class PreflightSeparationTests(TestCase):
    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Preflight Sep")

    def test_preflight_has_no_mutation_flags(self):
        import argparse

        from ai_hub.management.commands.knowledge_preflight import Command

        parser = argparse.ArgumentParser()
        Command().add_arguments(parser)
        options = {opt for a in parser._actions for opt in a.option_strings}
        for forbidden in ("--fix", "--repair", "--adjudicate", "--regenerate",
                          "--accept", "--discard", "--apply", "--write"):
            with self.subTest(option=forbidden):
                self.assertNotIn(forbidden, options)

    def test_the_action_command_does_not_depend_on_preflight(self):
        """Eligibility is owned by the Core operation, never by the diagnostic.

        AST-checked: the command's prose may mention the preflight workflow,
        but it must not import or call it.
        """
        import ast
        import pathlib

        tree = ast.parse(
            (
                pathlib.Path(__file__).resolve().parent
                / "management" / "commands" / f"{COMMAND}.py"
            ).read_text(encoding="utf-8")
        )
        modules = {
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertNotIn("ai_hub.services.knowledge_preflight", modules)
        self.assertNotIn("run_knowledge_preflight", referenced)

    def test_preflight_still_reports_the_state_after_an_action(self):
        """The intended workflow: preflight -> action -> preflight."""
        document = make_unknown(self.collection, "Workflow")
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "UNKNOWN_AUTHORITY")
        run_command(document=document.pk, action="adjudicate_unknown_as_explicit")
        self.assertEqual(preflight_row(document.pk)["lifecycle_state"], "EXPLICIT_AUTHORITY")

    def test_no_new_migration_is_required(self):
        from django.apps import apps
        from django.db.migrations.autodetector import MigrationAutodetector
        from django.db.migrations.loader import MigrationLoader
        from django.db.migrations.questioner import NonInteractiveMigrationQuestioner
        from django.db.migrations.state import ProjectState

        loader = MigrationLoader(None, ignore_no_migrations=True)
        autodetector = MigrationAutodetector(
            loader.project_state(), ProjectState.from_apps(apps),
            NonInteractiveMigrationQuestioner(specified_apps=set(), dry_run=True),
        )
        self.assertEqual(autodetector.changes(graph=loader.graph).get("ai_hub", []), [])


class BuildSnapshotUnchangedTests(TestCase):
    """The D-13-1 extraction must not have altered Slice 9 semantics."""

    def setUp(self):
        self.collection = KnowledgeCollection.objects.create(application_scope=test_scope(), name="Snapshot Parity")

    def test_build_snapshot_equals_the_row_based_constructor(self):
        from ai_hub.services.knowledge_mutation import _snapshot_from_chunk_rows

        document = make_derived(self.collection, "Parity", curated_text="body")
        rows = list(
            document.chunks.order_by("chunk_index").values(
                "chunk_index", "section_title", "content"
            )
        )
        self.assertEqual(
            build_snapshot(document), _snapshot_from_chunk_rows(document, rows)
        )

    def test_captured_rows_and_values_rows_produce_the_same_snapshot(self):
        document = make_derived(self.collection, "Both Shapes", curated_text="body")
        review = capture_lifecycle_review(
            document.pk, action="regenerate_derived_chunk_set"
        )
        self.assertEqual(review.snapshot, build_snapshot(document))
        self.assertEqual(
            review.expected, ExpectedKnowledgeState.from_snapshot(build_snapshot(document))
        )
