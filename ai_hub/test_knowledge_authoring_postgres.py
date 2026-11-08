"""T2 - PostgreSQL-only regression tests for `author_explicit_chunk_set`.

What SQLite cannot prove, and why these tests exist:

* `select_for_update()` is a no-op on SQLite, so the compare-and-swap ordering of
  a contended authoring call (lock, then verify `expected`, then decide) can only
  be exercised against real row locks.
* The ANN generation triggers and the mirrored HNSW leaves exist only on
  PostgreSQL with pgvector.

Every class here is skipped on any other backend, with a reason that says so. A
skipped test is NOT a passed PostgreSQL validation. The PostgreSQL gate for T2
stays open until the Linux/PostgreSQL CI job has actually executed this module.

The concurrency tests follow the repository's existing convention
(`GovernedMutationConcurrencyTests`, `ProvisioningConcurrencyTests`): a
`TransactionTestCase`, one real connection per thread, the server's own
`pg_backend_pid()` to prove the connections differ, and no mocking of the lock.
They use explicit events instead of sleeps, and observe the waiting writer through
`pg_stat_activity` rather than assuming it waits. Every wait is bounded.
"""
import contextlib
import threading
import time
from collections import defaultdict
from unittest import mock, skipUnless

from django.db import connection, connections
from django.test import TestCase, TransactionTestCase

from ai_hub.models import (
    KnowledgeChunkEmbedding,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    KnowledgeLifecycleEvent,
)
from ai_hub.services import knowledge_authoring as authoring
from ai_hub.services.knowledge_authoring import (
    AuthoredChunk,
    author_explicit_chunk_set,
)
from ai_hub.services.knowledge_authorization import resolve_effective_knowledge_scope
from ai_hub.services.knowledge_mutation import (
    ExpectedKnowledgeState,
    KnowledgeMutationConflict,
    KnowledgeMutationPrincipal,
    _governed_knowledge_mutation,
    build_snapshot,
)
from ai_hub.services.pgvector_ann import (
    PgvectorAnnError,
    PgvectorFailureCategory,
    current_generation,
    leaf_readiness,
    search_pgvector_ann_with_scope,
)
from ai_hub.services.vector_store import store_chunk_vector
from ai_hub.test_application_scope_helpers import test_scope
from ai_hub.test_pgvector_ann import PgvectorFixtureMixin

Status = KnowledgeDocument.Status
Mode = KnowledgeDocument.ChunkAuthorityMode

POSTGRES = connection.vendor == "postgresql"
SKIP_REASON = (
    "T2 PostgreSQL regression test: needs real row locks and the pgvector ANN "
    "triggers, which only a PostgreSQL backend provides. A skip is not a pass."
)
requires_postgres = skipUnless(POSTGRES, SKIP_REASON)

#: Bound for every wait in this module, in seconds.
WAIT = 30

#: The only helpers that write chunk or document rows inside the governed block.
MUTATING_HELPERS = ("_delete_rows", "_move_rows", "_insert_rows", "_declare_explicit")


def C(content, title=""):
    return AuthoredChunk(content=content, section_title=title)


# ---------------------------------------------------------------------------
# Runs on every backend: the module imports and skips for the right reason
# ---------------------------------------------------------------------------

class SkipContractTests(TestCase):
    """Never skipped. Guards against a PostgreSQL class that silently never runs."""

    CLASSES = ()  # filled below, after the classes exist

    def test_every_postgres_class_is_gated_on_the_vendor_and_says_why(self):
        self.assertTrue(self.CLASSES)
        for cls in self.CLASSES:
            with self.subTest(cls=cls.__name__):
                skipped = getattr(cls, "__unittest_skip__", False)
                self.assertEqual(skipped, not POSTGRES)
                if skipped:
                    reason = cls.__unittest_skip_why__
                    self.assertIn("PostgreSQL", reason)
                    self.assertIn("not a pass", reason)

    def test_the_gate_follows_the_connection_vendor(self):
        self.assertEqual(POSTGRES, connection.vendor == "postgresql")


# ---------------------------------------------------------------------------
# A / B: contended authoring on real row locks
# ---------------------------------------------------------------------------

@requires_postgres
class ContendedAuthoringTests(TransactionTestCase):
    """Two real connections, one document, the same starting expectation."""

    reset_sequences = True

    def setUp(self):
        self.scope = test_scope("Contention Host", slug="contention-host")
        self.collection = KnowledgeCollection.objects.create(
            application_scope=self.scope, name="Contention Docs"
        )
        # Starting state "A", created directly as an EXPLICIT document so that the
        # only lifecycle events in these tests are the ones under examination.
        self.document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Contended", status=Status.ACTIVE,
            chunk_authority_mode=Mode.EXPLICIT,
        )
        self.seed = KnowledgeDocumentChunk.objects.create(
            document=self.document, chunk_index=1, content="alpha"
        )
        self.expected = ExpectedKnowledgeState.from_snapshot(build_snapshot(self.document))
        self.stray = []

    # -- harness ------------------------------------------------------------

    def contents(self):
        return list(
            KnowledgeDocumentChunk.objects.filter(document=self.document)
            .order_by("chunk_index").values_list("content", flat=True)
        )

    def authoring_events(self):
        return KnowledgeLifecycleEvent.objects.filter(
            operation=authoring.OPERATION_AUTHOR_EXPLICIT_CHUNK_SET
        )

    def blocked_backends(self):
        """Other backends of this database currently waiting on a lock."""
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid() "
                "AND wait_event_type = 'Lock'"
            )
            return cursor.fetchone()[0]

    def wait_until_blocked(self):
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            if self.blocked_backends() >= 1:
                return True
            time.sleep(0.05)
        return False

    def spawn(self, name, target):
        thread = threading.Thread(target=target, name=name, daemon=True)
        self.stray.append(thread)
        thread.start()
        return thread

    def join_all(self):
        for thread in self.stray:
            thread.join(timeout=WAIT)
        alive = [t.name for t in self.stray if t.is_alive()]
        self.assertEqual(alive, [], f"threads still running (deadlock?): {alive}")

    def author_in_thread(self, results, label, texts, *, principal, pids):
        """One independent connection, recorded by the server's own pid."""
        try:
            with connections["default"].cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                pids[label] = cursor.fetchone()[0]
            results[label] = author_explicit_chunk_set(
                self.document.pk, [C(text) for text in texts],
                application_scope=self.scope, expected=self.expected,
                principal=KnowledgeMutationPrincipal.system(principal),
            )
        except BaseException as exc:  # noqa: BLE001 - reported to the test, never swallowed
            results[label] = exc
        finally:
            connections["default"].close()

    @contextlib.contextmanager
    def instrumented(self, *, hold_thread=None, locked=None, release=None):
        """Record which thread calls each mutating helper; optionally make
        `hold_thread` pause inside the governed block, after the lock and the
        compare-and-swap, and before it writes."""
        calls = defaultdict(list)

        def wrap(name, real):
            def wrapper(*args, **kwargs):
                who = threading.current_thread().name
                calls[who].append(name)
                if name == "_delete_rows" and who == hold_thread:
                    locked.set()
                    if not release.wait(WAIT):
                        raise AssertionError("the lock holder was never released")
                return real(*args, **kwargs)
            return wrapper

        with contextlib.ExitStack() as stack:
            for name in MUTATING_HELPERS:
                stack.enter_context(
                    mock.patch.object(authoring, name, wrap(name, getattr(authoring, name)))
                )
            yield calls

    def race_against_a_committing_holder(self, second_texts):
        """Holder: A -> B, paused while holding the lock. Second: same `expected`.
        Returns (results, calls, pids)."""
        locked, release = threading.Event(), threading.Event()
        results, pids = {}, {}
        with self.instrumented(hold_thread="t2-holder", locked=locked, release=release) as calls:
            try:
                self.spawn("t2-holder", lambda: self.author_in_thread(
                    results, "holder", ["beta"], principal="holder", pids=pids))
                self.assertTrue(locked.wait(WAIT), "the holder never reached its write phase")

                self.spawn("t2-second", lambda: self.author_in_thread(
                    results, "second", second_texts, principal="second", pids=pids))
                self.assertTrue(
                    self.wait_until_blocked(),
                    "the second author never appeared as waiting on a row lock",
                )
                self.assertNotIn("second", results, "the second author must still be waiting")
            finally:
                release.set()  # always, so no thread outlives a failed assertion
            self.join_all()
        return results, calls, pids

    # -- A: the contended race ---------------------------------------------

    def test_a_waiting_author_with_a_stale_expectation_conflicts_after_the_commit(self):
        results, calls, pids = self.race_against_a_committing_holder(["gamma"])

        self.assertNotEqual(pids["holder"], pids["second"], "needs two real connections")
        self.assertEqual(results["holder"].outcome, authoring.OUTCOME_CHANGED)
        self.assertIsInstance(results["second"], KnowledgeMutationConflict)

        self.assertEqual(self.contents(), ["beta"])  # the holder's B, nothing of the second
        self.assertEqual(KnowledgeDocumentChunk.objects.filter(document=self.document).count(), 1)
        self.assertFalse(KnowledgeDocumentChunk.objects.filter(pk=self.seed.pk).exists())

        events = list(self.authoring_events())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].principal_identifier, "holder")
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 1)

        self.assertEqual(calls["t2-second"], [], "the loser performed no writes")
        self.assertTrue(calls["t2-holder"])

    def test_a_waiting_identical_request_with_a_stale_expectation_is_a_conflict_not_unchanged(self):
        """The audited defect, on real locks: after the wait, B equals what is stored,
        but the caller planned against A. It must conflict, never report `unchanged`."""
        results, calls, pids = self.race_against_a_committing_holder(["beta"])

        self.assertNotEqual(pids["holder"], pids["second"])
        self.assertEqual(results["holder"].outcome, authoring.OUTCOME_CHANGED)
        self.assertIsInstance(results["second"], KnowledgeMutationConflict)
        self.assertEqual(self.contents(), ["beta"])
        self.assertEqual(self.authoring_events().count(), 1)
        self.assertEqual(calls["t2-second"], [])

    # -- B: a no-op takes part in the locking ------------------------------

    def test_an_identical_request_waits_for_the_lock_and_is_unchanged_after_a_compatible_release(self):
        """The holder takes the same governed lock and gives it back unchanged, so
        the waiting caller's `expected` is still exact. Its identical request must
        have waited, then return `unchanged` with no write and no event."""
        locked, release = threading.Event(), threading.Event()
        results, pids = {}, {}

        class Abort(Exception):
            pass

        def holder():
            try:
                with connections["default"].cursor() as cursor:
                    cursor.execute("SELECT pg_backend_pid()")
                    pids["holder"] = cursor.fetchone()[0]
                with _governed_knowledge_mutation(
                    self.document.pk, expected=self.expected,
                    operation="t2_lock_holder",
                    principal=KnowledgeMutationPrincipal.system("holder"),
                ):
                    locked.set()
                    if not release.wait(WAIT):
                        raise AssertionError("the lock holder was never released")
                    raise Abort  # roll back: lock released, document untouched, no event
            except Abort:
                results["holder"] = "released unchanged"
            except BaseException as exc:  # noqa: BLE001
                results["holder"] = exc
            finally:
                connections["default"].close()

        before = list(KnowledgeDocumentChunk.objects.filter(document=self.document).values())
        document_updated_at = KnowledgeDocument.objects.get(pk=self.document.pk).updated_at

        with self.instrumented() as calls:
            try:
                self.spawn("t2-holder", holder)
                self.assertTrue(locked.wait(WAIT), "the holder never took the lock")
                self.spawn("t2-second", lambda: self.author_in_thread(
                    results, "second", ["alpha"], principal="second", pids=pids))
                self.assertTrue(
                    self.wait_until_blocked(),
                    "an identical request must wait on the row lock, like any other",
                )
                self.assertNotIn("second", results, "it must not answer while the lock is held")
            finally:
                release.set()
            self.join_all()

        self.assertNotEqual(pids["holder"], pids["second"])
        self.assertEqual(results["holder"], "released unchanged")
        self.assertEqual(results["second"].outcome, authoring.OUTCOME_UNCHANGED)
        self.assertEqual((results["second"].kept, results["second"].inserted,
                          results["second"].removed, results["second"].moved), (1, 0, 0, 0))
        self.assertEqual(calls["t2-second"], [])
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)
        self.assertEqual(
            list(KnowledgeDocumentChunk.objects.filter(document=self.document).values()), before
        )
        self.assertEqual(
            KnowledgeDocument.objects.get(pk=self.document.pk).updated_at, document_updated_at
        )

    def test_the_governed_lock_is_a_real_row_lock_on_the_document(self):
        """The locking the tests above rely on is `FOR UPDATE`, not a stand-in."""
        from django.test.utils import CaptureQueriesContext

        with CaptureQueriesContext(connection) as captured:
            author_explicit_chunk_set(
                self.document.pk, [C("alpha")], application_scope=self.scope,
                expected=self.expected,
                principal=KnowledgeMutationPrincipal.system("probe"),
            )
        locking = [q["sql"] for q in captured.captured_queries if "FOR UPDATE" in q["sql"].upper()]
        self.assertGreaterEqual(len(locking), 1, "even a no-op must take the lock")


# ---------------------------------------------------------------------------
# C: authored chunk sets and the ANN generation / leaf
# ---------------------------------------------------------------------------

#: Distinct, non-zero 4-d vectors (the fixture contract is dimension 4, cosine).
VECTORS = {
    "alpha": (0.9, 0.1, 0.0, 0.0),
    "bravo": (0.5, 0.5, 0.0, 0.0),
    "charlie": (0.1, 0.9, 0.0, 0.0),
    "delta": (0.3, 0.3, 0.3, 0.3),
}
QUERY = (1.0, 0.0, 0.0, 0.0)


@requires_postgres
class AuthoredChunkSetsAndAnnTests(PgvectorFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()
        self.principal = KnowledgeMutationPrincipal.system("t2-ann")
        # A separate collection of the same scope holds the authored document, and a
        # neighbouring collection proves invalidation is not scope-wide.
        self.coll_t2 = self._collection(self.scope_a, "T2 Authored")
        self.agent.knowledge_collections.add(self.coll_t2)
        self.scope = resolve_effective_knowledge_scope(self.agent)
        self.provision_and_rebuild(self.coll_a1)  # neighbour leaf, untouched by T2

    # -- helpers ------------------------------------------------------------

    def author(self, document, texts):
        document.refresh_from_db()
        return author_explicit_chunk_set(
            document.pk, [C(text) for text in texts],
            application_scope=self.scope_a,
            expected=ExpectedKnowledgeState.from_snapshot(build_snapshot(document)),
            principal=self.principal,
        )

    def seed(self, texts, title="Authored"):
        """Author, activate (host-owned step), index every chunk, build the leaf."""
        document = KnowledgeDocument.objects.create(
            collection=self.coll_t2, title=title, status=Status.DRAFT
        )
        self.author(document, texts)
        KnowledgeDocument.objects.filter(pk=document.pk).update(status=Status.ACTIVE)
        for chunk in document.chunks.all():
            store_chunk_vector(
                application_scope=self.scope_a, chunk=chunk,
                embedding_model_config=self.config, vector=VECTORS[chunk.content],
            )
        self.provision_and_rebuild(self.coll_t2)
        return document

    def ids(self, document):
        return {row.content: row.pk for row in document.chunks.all()}

    def embeddings(self, document):
        return list(
            KnowledgeChunkEmbedding.objects.filter(chunk__document=document)
            .order_by("chunk_id").values_list("pk", "chunk_id", "k1")
        )

    def leaf_chunk_ids(self, collection):
        identity = self.leaf_names(collection)
        with connection.cursor() as cursor:
            cursor.execute(f'SELECT chunk_id FROM "{identity.leaf_table}" ORDER BY chunk_id')
            return [row[0] for row in cursor.fetchall()]

    def generation(self, collection):
        return current_generation(self.scope_a.pk, collection.pk)

    def readiness(self, collection):
        return leaf_readiness(self.scope_a.pk, collection.pk, self.contract)

    def search_ids(self):
        result = search_pgvector_ann_with_scope(
            self.scope, query_values=QUERY, embedding_model_config=self.config,
            collection_id=self.coll_t2.pk, limit=10,
        )
        return [match.chunk_id for match in result.matches]

    def assertStaleAndRefused(self, collection):
        readiness = self.readiness(collection)
        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "stale_generation")
        with self.assertRaises(PgvectorAnnError) as raised:
            self.search_ids()
        self.assertEqual(raised.exception.category, PgvectorFailureCategory.LEAF_NOT_READY)

    # -- tests --------------------------------------------------------------

    def test_the_seeded_leaf_is_ready_and_serves_the_authored_chunks(self):
        document = self.seed(["alpha", "bravo", "charlie"])
        self.assertTrue(self.readiness(self.coll_t2).ready)
        self.assertEqual(sorted(self.search_ids()), sorted(self.ids(document).values()))

    def test_removing_a_chunk_invalidates_only_its_collection_and_a_stale_leaf_is_never_served(self):
        document = self.seed(["alpha", "bravo", "charlie"])
        ids = self.ids(document)
        own_before = self.generation(self.coll_t2)
        neighbour_before = self.generation(self.coll_a1)

        result = self.author(document, ["alpha", "charlie"])

        self.assertEqual((result.outcome, result.removed), (authoring.OUTCOME_CHANGED, 1))
        self.assertGreater(self.generation(self.coll_t2), own_before)
        self.assertEqual(self.generation(self.coll_a1), neighbour_before)
        self.assertTrue(self.readiness(self.coll_a1).ready)

        # The mirror row of the removed chunk is still physically there (there is no
        # cascade into the derived leaf), which is exactly why it must not be served.
        self.assertIn(ids["bravo"], self.leaf_chunk_ids(self.coll_t2))
        self.assertStaleAndRefused(self.coll_t2)

        self.provision_and_rebuild(self.coll_t2)
        canonical = set(
            KnowledgeChunkEmbedding.objects.filter(
                collection_id=self.coll_t2.pk, e1=self.contract.e1
            ).values_list("chunk_id", flat=True)
        )
        self.assertEqual(canonical, {ids["alpha"], ids["charlie"]})
        self.assertEqual(set(self.leaf_chunk_ids(self.coll_t2)), canonical)
        self.assertTrue(self.readiness(self.coll_t2).ready)
        served = self.search_ids()
        self.assertEqual(set(served), canonical)
        self.assertNotIn(ids["bravo"], served)

    def test_changed_content_replaces_the_chunk_and_its_vector_and_stales_the_leaf(self):
        document = self.seed(["alpha", "bravo", "charlie"])
        ids = self.ids(document)
        before = self.generation(self.coll_t2)

        self.author(document, ["alpha", "bravo, revised", "charlie"])

        self.assertGreater(self.generation(self.coll_t2), before)
        self.assertFalse(KnowledgeDocumentChunk.objects.filter(pk=ids["bravo"]).exists())
        self.assertFalse(KnowledgeChunkEmbedding.objects.filter(chunk_id=ids["bravo"]).exists())
        replacement = document.chunks.get(content="bravo, revised")
        self.assertNotIn(replacement.pk, ids.values())
        self.assertFalse(replacement.embeddings.exists())
        self.assertStaleAndRefused(self.coll_t2)

        self.provision_and_rebuild(self.coll_t2)
        self.assertEqual(
            set(self.leaf_chunk_ids(self.coll_t2)), {ids["alpha"], ids["charlie"]}
        )
        self.assertNotIn(replacement.pk, self.search_ids())  # unindexed, so absent

    def test_new_chunks_have_no_embedding_and_preserved_chunks_keep_ids_and_vectors(self):
        document = self.seed(["alpha", "bravo", "charlie"])
        ids = self.ids(document)
        before = self.embeddings(document)
        self.assertEqual(len(before), 3)

        result = self.author(document, ["delta", "alpha", "bravo", "charlie"])  # new first

        self.assertEqual((result.kept, result.inserted, result.removed), (3, 1, 0))
        after_ids = self.ids(document)
        for content in ("alpha", "bravo", "charlie"):
            self.assertEqual(after_ids[content], ids[content])  # same chunk ids
        new_chunk = document.chunks.get(content="delta")
        self.assertFalse(new_chunk.embeddings.exists())  # not indexed until the host does it
        self.assertEqual(self.embeddings(document), before)  # same rows, same k1

        self.provision_and_rebuild(self.coll_t2)
        self.assertEqual(set(self.leaf_chunk_ids(self.coll_t2)), set(ids.values()))
        self.assertNotIn(new_chunk.pk, self.search_ids())

    def test_a_reorder_only_change_keeps_vector_identity_and_does_not_invalidate_the_leaf(self):
        document = self.seed(["alpha", "bravo", "charlie"])
        ids = self.ids(document)
        embeddings_before = self.embeddings(document)
        generation_before = self.generation(self.coll_t2)

        result = self.author(document, ["charlie", "bravo", "alpha"])

        self.assertEqual(result.outcome, authoring.OUTCOME_CHANGED)
        self.assertEqual((result.kept, result.inserted, result.removed), (3, 0, 0))
        self.assertGreater(result.moved, 0)
        self.assertEqual(
            list(document.chunks.order_by("chunk_index").values_list("content", flat=True)),
            ["charlie", "bravo", "alpha"],
        )
        self.assertEqual(self.ids(document), ids)
        self.assertEqual(self.embeddings(document), embeddings_before)

        # chunk_index is not part of what the trigger watches, so nothing went stale.
        self.assertEqual(self.generation(self.coll_t2), generation_before)
        self.assertTrue(self.readiness(self.coll_t2).ready)
        self.assertEqual(sorted(self.search_ids()), sorted(ids.values()))


SkipContractTests.CLASSES = (ContendedAuthoringTests, AuthoredChunkSetsAndAnnTests)
