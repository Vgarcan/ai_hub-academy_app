"""T2 - PostgreSQL-only regression tests for `author_explicit_chunk_set`.

What SQLite cannot prove, and why these tests exist:

* `select_for_update()` is a no-op on SQLite, so the compare-and-swap ordering of
  a contended authoring call (lock, then verify `expected`, then decide) can only
  be exercised against real row locks.
* The ANN generation triggers and the mirrored HNSW leaves exist only on
  PostgreSQL with pgvector.

Every PostgreSQL class here is skipped on any other backend, with a reason that
says so. A skipped test is NOT a passed PostgreSQL validation. The PostgreSQL gate
for T2 stays open until a Linux/PostgreSQL job has actually executed this module.

HOW THE CONCURRENCY TESTS STAY SAFE
-----------------------------------
All of them run their workers through `_Harness`:

* Workers are ordinary (non-daemon) threads that own their database connection:
  each one opens it, bounds its own session (`lock_timeout`, `statement_timeout`,
  `idle_in_transaction_session_timeout`), publishes the server's `pg_backend_pid()`
  and closes the connection itself in its own `finally`.
* The harness is a context manager. Whatever happens inside the `with` block,
  including a failed synchronisation assertion, its exit (1) releases every gate,
  (2) joins every started worker, (3) only if one is still running after a bounded
  wait, terminates exactly the backend PIDs those workers published, and only when
  they belong to this database and user, (4) removes the instrumentation last, so
  patched code stays in place until no worker can still call it, and (5) checks
  that the sessions are gone. An in-flight test exception is never replaced by a
  cleanup message; the cleanup diagnostics are attached to it as a note.
* "The second author is blocked by the first" is established from the server, not
  assumed: the second worker's own PID must be waiting with
  `wait_event_type = 'Lock'` and the holder's PID must appear in
  `pg_blocking_pids(second_pid)`. An unrelated waiting backend proves nothing.

Following the repository's convention (`GovernedMutationConcurrencyTests`,
`ProvisioningConcurrencyTests`): `TransactionTestCase`, no mocking of the lock.
"""
import contextlib
import re
import sys
import threading
import time
from collections import defaultdict
from unittest import mock, skipUnless

from django.db import connection, connections
from django.test import TestCase, TransactionTestCase
from django.test.utils import CaptureQueriesContext

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

#: Bound for every Python-side wait in this module, in seconds.
WAIT = 30

#: Bound for the join that follows an emergency backend termination, in seconds.
TERMINATE_JOIN = 10

#: Workers that outlived a failed cleanup. While any is alive, no new harness starts.
_ORPHANS = []

#: Session bounds applied by each worker to its own connection, in milliseconds.
#: They are longer than any wait the tests make on purpose: they exist so that a
#: broken test ends a database session, never so that a healthy one is cut short.
LOCK_TIMEOUT_MS = 45_000
STATEMENT_TIMEOUT_MS = 60_000
IDLE_IN_TRANSACTION_TIMEOUT_MS = 90_000

#: The only helpers that write chunk or document rows inside the governed block.
MUTATING_HELPERS = ("_delete_rows", "_move_rows", "_insert_rows", "_declare_explicit")


def C(content, title=""):
    return AuthoredChunk(content=content, section_title=title)


class _Abort(Exception):
    """Raised by a lock holder to roll its transaction back unchanged."""


# ---------------------------------------------------------------------------
# Worker and harness
# ---------------------------------------------------------------------------

class _Worker:
    """One thread, one owned connection, one published backend PID."""

    def __init__(self, name, fn):
        self.name = name
        self._fn = fn
        self.pid = None
        self.result = None
        self.error = None
        self.close_error = None
        self.ready = threading.Event()     # connection initialised, PID published
        self.finished = threading.Event()  # connection closed by its owner
        self.thread = threading.Thread(target=self._run, name=name)  # not a daemon

    def _run(self):
        try:
            if POSTGRES:
                with connections["default"].cursor() as cursor:
                    cursor.execute(f"SET lock_timeout = {LOCK_TIMEOUT_MS}")
                    cursor.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
                    cursor.execute(
                        f"SET idle_in_transaction_session_timeout = {IDLE_IN_TRANSACTION_TIMEOUT_MS}"
                    )
                    cursor.execute("SELECT pg_backend_pid()")
                    self.pid = cursor.fetchone()[0]
            self.ready.set()
            self.result = self._fn()
        except BaseException as exc:  # noqa: BLE001 - reported to the test, never swallowed
            self.error = exc
        finally:
            self.ready.set()  # never leave the harness waiting on a worker that failed early
            try:
                connections["default"].close()  # closed by the thread that owns it
            except Exception as exc:  # noqa: BLE001
                self.close_error = exc
            self.finished.set()

    def outcome(self):
        """The worker's result, or its original exception."""
        if self.error is not None:
            raise self.error
        return self.result

    @property
    def alive(self):
        return self.thread.is_alive()


class _Harness:
    """Runs workers and guarantees they are finished, whatever the test does."""

    def __init__(self):
        self.locked = threading.Event()   # a holder reached its critical point
        self.release = threading.Event()  # let the holder go
        self.workers = []
        self.calls = defaultdict(list)
        self._stack = contextlib.ExitStack()

    def __enter__(self):
        _ORPHANS[:] = [w for w in _ORPHANS if w.alive]
        if _ORPHANS:
            raise AssertionError(
                "a previous test left worker(s) still running after its cleanup failed: "
                f"{[w.name for w in _ORPHANS]}. Refusing to start another concurrency test "
                "beside them. Run this module in a fresh process."
            )
        return self

    # -- instrumentation ----------------------------------------------------

    def instrument(self, *, hold=None):
        """Record which worker calls each mutating helper. If `hold` is the name of
        a worker, that worker pauses inside the governed block, after the lock and
        the compare-and-swap and before its first write, until `release` is set."""
        def wrap(name, real):
            def wrapper(*args, **kwargs):
                who = threading.current_thread().name
                self.calls[who].append(name)
                if hold is not None and name == "_delete_rows" and who == hold:
                    self.locked.set()
                    if not self.release.wait(WAIT):
                        raise AssertionError("the lock holder was never released")
                return real(*args, **kwargs)
            return wrapper

        for name in MUTATING_HELPERS:
            self._stack.enter_context(
                mock.patch.object(authoring, name, wrap(name, getattr(authoring, name)))
            )
        return self.calls

    # -- workers ------------------------------------------------------------

    def start(self, name, fn):
        worker = _Worker(name, fn)
        self.workers.append(worker)
        worker.thread.start()
        if not worker.ready.wait(WAIT):
            raise AssertionError(f"worker {name!r} never initialised its connection")
        return worker

    def wait_blocked(self, waiter, holder):
        """Establish from the server that `waiter` is waiting on a lock held by
        `holder`. Bounded; fails with what was actually observed."""
        if not (waiter.pid and holder.pid):
            raise AssertionError(f"backend pids were not published: {waiter.pid=} {holder.pid=}")
        if waiter.pid == holder.pid:
            raise AssertionError("the two workers share one backend; this proves nothing")
        deadline = time.monotonic() + WAIT
        observed = None
        while time.monotonic() < deadline:
            if waiter.finished.is_set():
                raise AssertionError(
                    f"{waiter.name} finished instead of waiting: result={waiter.result!r} "
                    f"error={waiter.error!r}"
                )
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT wait_event_type, wait_event, state, pg_blocking_pids(pid) "
                    "FROM pg_stat_activity "
                    "WHERE pid = %s AND datname = current_database()",
                    [waiter.pid],
                )
                row = cursor.fetchone()
            observed = row
            if row is not None and row[0] == "Lock" and holder.pid in (row[3] or []):
                return
            time.sleep(0.05)
        raise AssertionError(
            f"{waiter.name} (pid {waiter.pid}) was never seen waiting on a lock held by "
            f"{holder.name} (pid {holder.pid}); last observation "
            f"(wait_event_type, wait_event, state, blocking_pids) = {observed!r}"
        )

    # -- cleanup ------------------------------------------------------------

    def _terminate_own_backends(self, workers):
        """Defensive, last resort: only PIDs these workers published, only in this
        database, only under this user, never this session. Returns the PIDs a
        termination was *requested* for. That is a request, not proof that a worker
        stopped: only the worker's thread actually ending proves that."""
        pids = [w.pid for w in workers if w.pid]
        if not (POSTGRES and pids):
            return []
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pid, pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE pid = ANY(%s) AND datname = current_database() "
                "AND usename = current_user AND pid <> pg_backend_pid()",
                [pids],
            )
            return [row[0] for row in cursor.fetchall()]

    def _sessions_remaining(self):
        pids = [w.pid for w in self.workers if w.pid]
        if not (POSTGRES and pids):
            return 0
        deadline = time.monotonic() + 10
        while True:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE pid = ANY(%s)", [pids]
                )
                remaining = cursor.fetchone()[0]
            if remaining == 0 or time.monotonic() > deadline:
                return remaining
            time.sleep(0.05)

    @staticmethod
    def _attempt(problems, label, step, default=None):
        """Run one cleanup phase. Its failure is recorded, never raised, so that the
        remaining phases still run and the test's own exception stays primary."""
        try:
            return step()
        except Exception as exc:  # noqa: BLE001 - collected as a diagnostic, not dropped
            problems.append(f"{label} raised {exc!r}")
            return default

    def _survivors(self, problems):
        # If even the inspection fails, assume the worst: every worker is alive.
        return self._attempt(
            problems, "inspecting workers",
            lambda: [w for w in self.workers if w.alive], default=list(self.workers),
        )

    def __exit__(self, exc_type, exc, tb):
        problems = []
        attempt = self._attempt

        attempt(problems, "releasing the holder", self.release.set)
        for worker in self.workers:
            attempt(problems, f"joining {worker.name}", lambda w=worker: w.thread.join(timeout=WAIT))

        alive = self._survivors(problems)
        survivors = alive
        if alive:
            problems.append(
                "workers did not finish in time: "
                + ", ".join(f"{w.name} (pid {w.pid})" for w in alive)
            )
            requested = attempt(
                problems, "terminating worker backends",
                lambda: self._terminate_own_backends(alive), default=[],
            )
            problems.append(
                f"termination requested for backend pids {requested} (a request, not proof "
                "that a worker stopped)"
            )
            for worker in alive:
                attempt(problems, f"joining {worker.name} after termination",
                        lambda w=worker: w.thread.join(timeout=TERMINATE_JOIN))
            survivors = self._survivors(problems)

        if survivors:
            # FAIL CLOSED. A Python thread cannot be killed, so a worker that outlived
            # its backend termination may still run. The instrumentation it could still
            # call is therefore left in place (it is pass-through once `release` is
            # set), and the worker is registered so that the next harness refuses to
            # start instead of running beside it.
            _ORPHANS.extend(w for w in survivors if w not in _ORPHANS)
            problems.append(
                f"CLEANUP INCOMPLETE: workers STILL running: {[w.name for w in survivors]}; "
                "instrumentation was left in place and later harnesses will refuse to start"
            )
        else:
            attempt(problems, "removing the instrumentation", self._stack.close)

        for worker in self.workers:
            if worker.close_error is not None:
                problems.append(f"{worker.name} could not close its connection: {worker.close_error!r}")
        remaining = attempt(problems, "checking for remaining sessions", self._sessions_remaining, default=0)
        if remaining:
            problems.append(f"{remaining} worker backend session(s) still present in pg_stat_activity")

        if problems:
            message = "T2 test cleanup problem: " + "; ".join(problems)
            if exc is not None:
                # The original failure stays the primary exception; the diagnostics ride
                # along on it. (`add_note` needs Python 3.11+; the fallback keeps them.)
                if hasattr(exc, "add_note"):
                    exc.add_note(message)
                else:
                    exc.cleanup_diagnostics = message
                return False
            raise AssertionError(message)
        return False


# ---------------------------------------------------------------------------
# Runs on every backend: gating contract and harness cleanup
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


class HarnessCleanupTests(TestCase):
    """Runs on every backend: a failed synchronisation must not strand a worker."""

    class Injected(Exception):
        pass

    def test_a_failing_test_body_still_releases_and_joins_the_workers(self):
        started = threading.Event()
        harness = _Harness()
        workers = []

        def hold():
            started.set()
            # Waits for the gate; only the harness exit can open it here.
            if not harness.release.wait(WAIT):
                raise AssertionError("never released")
            return "released"

        with self.assertRaises(self.Injected):
            with harness:
                workers.append(harness.start("t2-holder", hold))
                self.assertTrue(started.wait(WAIT))
                workers.append(harness.start("t2-second", hold))
                raise self.Injected("a synchronisation assertion failed")

        for worker in workers:
            self.assertFalse(worker.alive, f"{worker.name} was left running")
            self.assertTrue(worker.finished.is_set(), "the owning thread closes its connection")
            self.assertEqual(worker.outcome(), "released")

    def test_the_original_exception_survives_a_cleanup_problem(self):
        harness = _Harness()
        harness.start("t2-w", lambda: "done")
        with mock.patch.object(_Harness, "_sessions_remaining", return_value=2):
            with self.assertRaises(self.Injected) as raised:
                with harness:
                    raise self.Injected("the real failure")
        self.assertEqual(str(raised.exception), "the real failure")
        if hasattr(raised.exception, "__notes__"):
            self.assertTrue(any("cleanup problem" in n for n in raised.exception.__notes__))

    def test_a_cleanup_problem_alone_is_reported_not_hidden(self):
        harness = _Harness()
        harness.start("t2-w", lambda: "done")
        with mock.patch.object(_Harness, "_sessions_remaining", return_value=1):
            with self.assertRaisesRegex(AssertionError, "cleanup problem"):
                with harness:
                    pass

    def test_instrumentation_stays_until_every_worker_has_finished(self):
        harness = _Harness()
        seen = []

        def late_call():
            harness.release.wait(WAIT)
            seen.append(authoring._delete_rows.__name__)  # touches the patched name
            return "ok"

        with harness:
            harness.instrument()
            harness.start("t2-late", late_call)
        self.assertEqual(seen, ["wrapper"])  # still the instrumented wrapper when it ran
        self.assertEqual(authoring._delete_rows.__name__, "_delete_rows")  # restored after


class HarnessCleanupFailureTests(TestCase):
    """Runs on every backend: cleanup that itself fails, or cannot finish.

    Waits are shortened for these tests, and every stubborn worker is bounded and is
    released and joined by the test's own cleanup. No thread here can block forever.
    """

    class Injected(Exception):
        pass

    def setUp(self):
        module = sys.modules[__name__]
        for name, value in (("WAIT", 0.5), ("TERMINATE_JOIN", 0.3)):
            patcher = mock.patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(_ORPHANS.clear)  # registered early, so it runs late

    def stubborn(self, harness):
        """A worker that outlives the (shortened) joins, but only for a bounded time."""
        gate = threading.Event()

        def work():
            gate.wait(5)
            return "late"

        worker = harness.start("t2-stubborn", work)

        def reap():
            gate.set()
            worker.thread.join(timeout=10)
            harness._stack.close()

        self.addCleanup(reap)
        return worker

    @staticmethod
    def notes_of(exc):
        return list(getattr(exc, "__notes__", None) or [getattr(exc, "cleanup_diagnostics", "")])

    def assertNote(self, exc, fragment):
        joined = " | ".join(self.notes_of(exc))
        self.assertIn(fragment, joined)

    def test_a_failing_session_check_does_not_replace_the_original_exception(self):
        harness = _Harness()
        with mock.patch.object(_Harness, "_sessions_remaining", side_effect=RuntimeError("pg gone")):
            with self.assertRaises(self.Injected) as raised:
                with harness:
                    raise self.Injected("the real failure")
        self.assertEqual(str(raised.exception), "the real failure")
        self.assertNote(raised.exception, "checking for remaining sessions raised")
        self.assertNote(raised.exception, "pg gone")

    def test_a_failing_termination_does_not_replace_the_original_exception(self):
        harness = _Harness()
        with mock.patch.object(_Harness, "_terminate_own_backends", side_effect=RuntimeError("no terminate")), \
                mock.patch.object(_Harness, "_sessions_remaining", return_value=0) as sessions:
            with self.assertRaises(self.Injected) as raised:
                with harness:
                    self.stubborn(harness)
                    raise self.Injected("the real failure")
        self.assertEqual(str(raised.exception), "the real failure")
        self.assertNote(raised.exception, "terminating worker backends raised")
        self.assertNote(raised.exception, "STILL running")
        self.assertTrue(sessions.called, "later phases must still run")

    def test_a_failing_instrumentation_removal_does_not_replace_the_original_exception(self):
        harness = _Harness()
        harness._stack = mock.Mock()
        harness._stack.close.side_effect = RuntimeError("unpatch failed")
        with mock.patch.object(_Harness, "_sessions_remaining", return_value=0) as sessions:
            with self.assertRaises(self.Injected) as raised:
                with harness:
                    raise self.Injected("the real failure")
        self.assertEqual(str(raised.exception), "the real failure")
        self.assertNote(raised.exception, "removing the instrumentation raised")
        self.assertTrue(sessions.called, "a failed phase must not stop the next one")

    def test_several_cleanup_failures_are_all_reported_and_every_phase_runs(self):
        harness = _Harness()
        worker = harness.start("t2-w", lambda: "done")
        worker.thread.join(timeout=5)
        worker.close_error = RuntimeError("close failed")
        harness._stack = mock.Mock()
        harness._stack.close.side_effect = RuntimeError("unpatch failed")
        with mock.patch.object(_Harness, "_sessions_remaining", side_effect=RuntimeError("pg gone")) as sessions:
            with self.assertRaises(self.Injected) as raised:
                with harness:
                    raise self.Injected("the real failure")
        self.assertEqual(str(raised.exception), "the real failure")
        for fragment in ("removing the instrumentation raised", "unpatch failed",
                         "could not close its connection", "close failed",
                         "checking for remaining sessions raised", "pg gone"):
            self.assertNote(raised.exception, fragment)
        harness._stack.close.assert_called_once()
        sessions.assert_called_once()

    def test_failures_with_a_surviving_worker_are_all_reported_together(self):
        harness = _Harness()
        with mock.patch.object(_Harness, "_terminate_own_backends", side_effect=RuntimeError("no terminate")), \
                mock.patch.object(_Harness, "_sessions_remaining", side_effect=RuntimeError("pg gone")):
            with self.assertRaises(self.Injected) as raised:
                with harness:
                    self.stubborn(harness)
                    raise self.Injected("the real failure")
        self.assertEqual(str(raised.exception), "the real failure")
        for fragment in ("terminating worker backends raised", "no terminate",
                         "CLEANUP INCOMPLETE", "checking for remaining sessions raised"):
            self.assertNote(raised.exception, fragment)

    def test_without_an_original_exception_every_cleanup_failure_is_raised(self):
        harness = _Harness()
        harness._stack = mock.Mock()
        harness._stack.close.side_effect = RuntimeError("unpatch failed")
        with mock.patch.object(_Harness, "_sessions_remaining", side_effect=RuntimeError("pg gone")):
            with self.assertRaises(AssertionError) as raised:
                with harness:
                    pass
        message = str(raised.exception)
        self.assertIn("removing the instrumentation raised", message)
        self.assertIn("checking for remaining sessions raised", message)

    def test_a_surviving_worker_fails_closed_and_is_never_reported_as_stopped(self):
        harness = _Harness()
        with mock.patch.object(_Harness, "_terminate_own_backends", return_value=[4242]) as terminate:
            with self.assertRaises(AssertionError) as raised:
                with harness:
                    harness.instrument()
                    worker = self.stubborn(harness)
        message = str(raised.exception)

        terminate.assert_called_once()
        self.assertTrue(worker.alive, "the simulated worker survived the first cleanup attempt")
        self.assertIn("CLEANUP INCOMPLETE", message)
        self.assertIn("STILL running", message)
        self.assertIn("t2-stubborn", message)
        self.assertIn("a request, not proof", message)  # termination is never reported as success
        self.assertNotIn("terminated successfully", message)

        # Fail closed: the instrumentation the survivor could still call stays in place...
        self.assertEqual(authoring._delete_rows.__name__, "wrapper")
        # ...and the next harness refuses to start beside it.
        self.assertIn(worker, _ORPHANS)
        with self.assertRaisesRegex(AssertionError, "previous test left worker"):
            _Harness().__enter__()

    def test_a_finished_orphan_no_longer_blocks_a_new_harness(self):
        done = _Worker("t2-done", lambda: "x")
        done.thread.start()
        done.thread.join(timeout=5)
        _ORPHANS.append(done)
        with _Harness():
            pass
        self.assertEqual(_ORPHANS, [])


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

    def contents(self):
        return list(
            KnowledgeDocumentChunk.objects.filter(document=self.document)
            .order_by("chunk_index").values_list("content", flat=True)
        )

    def authoring_events(self):
        return KnowledgeLifecycleEvent.objects.filter(
            operation=authoring.OPERATION_AUTHOR_EXPLICIT_CHUNK_SET
        )

    def author(self, texts, principal):
        return author_explicit_chunk_set(
            self.document.pk, [C(text) for text in texts],
            application_scope=self.scope, expected=self.expected,
            principal=KnowledgeMutationPrincipal.system(principal),
        )

    def race_against_a_committing_holder(self, second_texts):
        """Holder: A -> B, paused while holding the lock. Second: the same `expected`.
        Returns (holder, second, calls); every worker is finished on return."""
        with _Harness() as harness:
            calls = harness.instrument(hold="t2-holder")
            holder = harness.start("t2-holder", lambda: self.author(["beta"], "holder"))
            self.assertTrue(harness.locked.wait(WAIT), "the holder never reached its write phase")
            second = harness.start("t2-second", lambda: self.author(second_texts, "second"))
            harness.wait_blocked(second, holder)
            self.assertFalse(second.finished.is_set(), "the second author must still be waiting")
            harness.release.set()
        return holder, second, calls

    # -- A: the contended race ---------------------------------------------

    def test_a_waiting_author_with_a_stale_expectation_conflicts_after_the_commit(self):
        holder, second, calls = self.race_against_a_committing_holder(["gamma"])

        self.assertNotEqual(holder.pid, second.pid, "needs two real connections")
        self.assertEqual(holder.outcome().outcome, authoring.OUTCOME_CHANGED)
        self.assertIsInstance(second.error, KnowledgeMutationConflict)

        self.assertEqual(self.contents(), ["beta"])  # the holder's B, nothing of the second
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
        holder, second, calls = self.race_against_a_committing_holder(["beta"])

        self.assertNotEqual(holder.pid, second.pid)
        self.assertEqual(holder.outcome().outcome, authoring.OUTCOME_CHANGED)
        self.assertIsInstance(second.error, KnowledgeMutationConflict)
        self.assertEqual(self.contents(), ["beta"])
        self.assertEqual(self.authoring_events().count(), 1)
        self.assertEqual(calls["t2-second"], [])

    # -- B: a no-op takes part in the locking ------------------------------

    def test_an_identical_request_waits_for_the_lock_and_is_unchanged_after_a_compatible_release(self):
        """The holder takes the same governed lock and gives it back unchanged, so
        the waiting caller's `expected` is still exact. Its identical request must
        have waited, then return `unchanged` with no write and no event."""
        before = list(KnowledgeDocumentChunk.objects.filter(document=self.document).values())
        document_updated_at = KnowledgeDocument.objects.get(pk=self.document.pk).updated_at

        with _Harness() as harness:
            calls = harness.instrument()

            def hold():
                try:
                    with _governed_knowledge_mutation(
                        self.document.pk, expected=self.expected,
                        operation="t2_lock_holder",
                        principal=KnowledgeMutationPrincipal.system("holder"),
                    ):
                        harness.locked.set()
                        if not harness.release.wait(WAIT):
                            raise AssertionError("the lock holder was never released")
                        raise _Abort  # roll back: lock released, document untouched, no event
                except _Abort:
                    return "released unchanged"

            holder = harness.start("t2-holder", hold)
            self.assertTrue(harness.locked.wait(WAIT), "the holder never took the lock")
            second = harness.start("t2-second", lambda: self.author(["alpha"], "second"))
            harness.wait_blocked(second, holder)
            self.assertFalse(second.finished.is_set(), "it must not answer while the lock is held")
            harness.release.set()

        self.assertNotEqual(holder.pid, second.pid)
        self.assertEqual(holder.outcome(), "released unchanged")
        result = second.outcome()
        self.assertEqual(result.outcome, authoring.OUTCOME_UNCHANGED)
        self.assertEqual((result.kept, result.inserted, result.removed, result.moved), (1, 0, 0, 0))
        self.assertEqual(calls["t2-second"], [])
        self.assertEqual(KnowledgeLifecycleEvent.objects.count(), 0)
        self.assertEqual(
            list(KnowledgeDocumentChunk.objects.filter(document=self.document).values()), before
        )
        self.assertEqual(
            KnowledgeDocument.objects.get(pk=self.document.pk).updated_at, document_updated_at
        )

    # -- the lock itself ----------------------------------------------------

    def test_even_a_no_op_locks_the_document_and_its_chunks(self):
        """The locking the tests above rely on is `FOR UPDATE` on BOTH the document
        row and its chunk rows, checked against the SQL Django actually generated."""
        with CaptureQueriesContext(connection) as captured:
            result = author_explicit_chunk_set(
                self.document.pk, [C("alpha")], application_scope=self.scope,
                expected=self.expected,
                principal=KnowledgeMutationPrincipal.system("probe"),
            )
        self.assertEqual(result.outcome, authoring.OUTCOME_UNCHANGED)

        locking = [
            re.sub(r"\s+", " ", q["sql"]) for q in captured.captured_queries
            if re.search(r"\bFOR UPDATE\b", q["sql"], re.IGNORECASE)
        ]
        for model in (KnowledgeDocument, KnowledgeDocumentChunk):
            table = connection.ops.quote_name(model._meta.db_table)
            with self.subTest(table=model._meta.db_table):
                self.assertTrue(
                    any(re.search(r"\bFROM " + re.escape(table), sql) for sql in locking),
                    f"no FOR UPDATE query selects FROM {table}; locking queries: {locking}",
                )


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

    def test_inserting_an_unembedded_chunk_does_not_invalidate_a_current_leaf(self):
        document = self.seed(["alpha", "bravo", "charlie"])
        ids = self.ids(document)
        embeddings_before = self.embeddings(document)
        self.assertEqual(len(embeddings_before), 3)
        generation_before = self.generation(self.coll_t2)
        readiness_before = self.readiness(self.coll_t2)
        self.assertTrue(readiness_before.ready, readiness_before.reason)

        result = self.author(document, ["delta", "alpha", "bravo", "charlie"])  # new first

        self.assertEqual((result.kept, result.inserted, result.removed), (3, 1, 0))
        after_ids = self.ids(document)
        for content in ("alpha", "bravo", "charlie"):
            self.assertEqual(after_ids[content], ids[content])  # same chunk ids
        new_chunk = document.chunks.get(content="delta")
        self.assertFalse(new_chunk.embeddings.exists())  # not indexed until the host does it
        self.assertEqual(self.embeddings(document), embeddings_before)  # same rows, same k1

        # Nothing the leaf mirrors changed, so it is still current BEFORE any rebuild,
        # and the preserved vectors are still served from it.
        self.assertEqual(self.generation(self.coll_t2), generation_before)
        readiness_after = self.readiness(self.coll_t2)
        self.assertTrue(readiness_after.ready, readiness_after.reason)
        self.assertEqual(sorted(self.search_ids()), sorted(ids.values()))
        self.assertNotIn(new_chunk.pk, self.search_ids())

        # Existing post-rebuild expectations still hold.
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


@requires_postgres
class HarnessSessionCleanupTests(TransactionTestCase):
    """PostgreSQL only: the harness leaves no backend behind when the body fails."""

    class Injected(Exception):
        pass

    def test_a_failed_synchronisation_leaves_no_worker_session_behind(self):
        workers = []
        with self.assertRaises(self.Injected):
            with _Harness() as harness:
                def hold():
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT 1")
                    harness.locked.set()
                    harness.release.wait(WAIT)
                    return "released"

                workers.append(harness.start("t2-holder", hold))
                self.assertTrue(harness.locked.wait(WAIT))
                raise self.Injected("a synchronisation assertion failed")

        (worker,) = workers
        self.assertIsNotNone(worker.pid)
        self.assertFalse(worker.alive)
        self.assertEqual(worker.outcome(), "released")
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE pid = %s", [worker.pid])
            self.assertEqual(cursor.fetchone()[0], 0, "the worker's backend session must be gone")


SkipContractTests.CLASSES = (
    ContendedAuthoringTests, AuthoredChunkSetsAndAnnTests, HarnessSessionCleanupTests,
)
