"""Characterization of the GAME compatibility Knowledge retrieval path (Slice 2).

ARCHITECTURAL CONTEXT
---------------------
Convergence is already the decided direction (ADR-N1): AI Hub has ONE canonical
reusable Knowledge retrieval capability, and GAME and Orchestrator must
eventually consume the same one. The GAME internal handlers characterized here
are a CURRENT compatibility path, not a second permanent architecture.

This module does NOT decide whether to converge. It measures what exists today
so a future convergence slice knows what it must preserve, what it must not
preserve, and what regression tests it will need.

THE TWO PATHS
-------------
Path 1 - canonical, chunk-level:
    ai_hub.services.knowledge_retrieval.search_knowledge()

Path 2 - GAME selected-action compatibility, document-level:
    ai_hub.services.game_action_dispatcher._handle_search_knowledge()
    ai_hub.services.game_action_dispatcher._handle_read_document()

Path 2 is the DEFAULT GAME selected-action path, because
AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED defaults to False.

CORPUS EQUIVALENCE
------------------
The two paths read DIFFERENT FIELDS. Path 1 searches chunks; Path 2 searches
KnowledgeDocument.curated_text. The Slice 1 corpus populates chunks only, so it
cannot fairly exercise Path 2.

This module therefore builds an EQUIVALENT corpus from the same Slice 1 spec,
additionally populating curated_text with the same text the chunks carry. Both
paths then see the same information, and every measured difference is
attributable to retrieval logic rather than to data population.

That the two paths read different fields at all is itself a headline finding,
and is measured separately in KnowledgeFieldDivergenceTests.

NO PRODUCTION CODE IS CHANGED BY THIS MODULE.
"""
from dataclasses import dataclass

from django.core.exceptions import ValidationError
from django.test import TestCase, override_settings

from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    GameActionDefinition,
    GameActionRun,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    ProviderConfig,
    ToolExecutionRun,
)
from ai_hub.services.game_action_dispatcher import (
    _handle_read_document,
    _handle_search_knowledge,
    execute_game_action,
)
from ai_hub.services.game_goals import create_goal
from ai_hub.services.game_workspaces import create_workspace
from ai_hub.services.knowledge_retrieval import search_knowledge
from ai_hub.test_retrieval_baseline import (
    CORPUS,
    FORBIDDEN_CHUNK_KEYS,
    GOLDEN_QUERIES,
    RECORDED_BASELINE,
    SEARCH_LIMIT,
    evaluate_golden_set,
)


# Documents that must never be reachable, projected from the Slice 1 chunk set.
FORBIDDEN_DOCUMENT_KEYS = frozenset(
    key.split("/")[0] for key in FORBIDDEN_CHUNK_KEYS
)

# Path 2's own internal bounds, read from the implementation. Recorded here so a
# change to either constant is visible in this module's diff.
PATH2_SEARCH_CHAR_BUDGET = 4000
PATH2_READ_DOCUMENT_CHAR_CAP = 8000


# ---------------------------------------------------------------------------
# Comparability classification
# ---------------------------------------------------------------------------
# The two paths expose different semantics. Inventing equivalent numbers where
# they do not exist would produce a tidy table and a false comparison, so each
# dimension is classified explicitly.

SUPPORTED = "SUPPORTED COMPARISON"
PARTIAL = "PARTIAL COMPARISON"
NOT_COMPARABLE = "NOT COMPARABLE"

COMPARABILITY = (
    ("authorization scope", SUPPORTED,
     "Both enforce Agent collections plus active collection/document filters."),
    ("no-answer (returned nothing at all)", SUPPORTED,
     "Both either return results or do not."),
    ("answerable-query hit rate", PARTIAL,
     "Only at DOCUMENT granularity. Path 1 chunk hits are projected to their "
     "parent document; Path 2 has no chunk concept."),
    ("precision", PARTIAL,
     "Document granularity only, and the denominators differ: Path 1 is capped "
     "at top-K, Path 2 has no result cap at all."),
    ("multi-document behavior", SUPPORTED,
     "Both can return several documents."),
    ("returned text volume", SUPPORTED,
     "Both return characters; the budgets and units differ and are recorded."),
    ("evidence granularity", NOT_COMPARABLE,
     "Path 1 returns chunks with section titles; Path 2 returns document-level "
     "curated_text prefixes. Different objects."),
    ("relevance score", NOT_COMPARABLE,
     "Path 2 computes no score of any kind."),
    ("ranking semantics", NOT_COMPARABLE,
     "Path 1 sorts by score then (title, chunk_index). Path 2 does not rank: "
     "output order is collection name then document title, always."),
    ("rank of expected evidence", NOT_COMPARABLE,
     "Position exists in both outputs, but in Path 2 it carries no relevance "
     "signal, so comparing the numbers would be meaningless."),
    ("chunk identity", NOT_COMPARABLE,
     "Path 2 returns no chunk_id and cannot address a chunk."),
    ("citation / provenance metadata", NOT_COMPARABLE,
     "Path 1 returns an 8-field citation. Path 2 returns no citation object."),
    ("candidate counters / truncation reporting", NOT_COMPARABLE,
     "Path 1 reports candidates_scanned / candidate_limit / candidates_truncated. "
     "Path 2 reports none, and silently stops at its character budget."),
    ("query tokenization", NOT_COMPARABLE,
     "Path 1 tokenizes into up to 20 words. Path 2 matches the ENTIRE query "
     "string as one substring."),
    ("audit record", PARTIAL,
     "Both are audited, by different models: Path 1 via ToolExecutionRun when "
     "invoked through the Tool runtime, Path 2 via GameActionRun."),
)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def _curated_text_for(document_spec) -> str:
    """Reconstruct document-level text from the same chunks Path 1 will index.

    Section titles are included so both paths have access to the same words.
    """
    return "\n\n".join(
        f"{section_title}\n{content}"
        for _index, section_title, content in document_spec["chunks"]
    )


def build_dual_path_corpus(agent, *, populate_curated_text=True):
    """Build the Slice 1 corpus, optionally with curated_text populated.

    Returns (chunk_key_by_id, document_key_by_id, document_id_by_key).
    """
    chunk_key_by_id = {}
    document_key_by_id = {}
    document_id_by_key = {}

    for collection_spec in CORPUS:
        collection = KnowledgeCollection.objects.create(
            name=collection_spec["name"],
            description=collection_spec["description"],
            is_active=collection_spec["is_active"],
        )
        if collection_spec["attached"]:
            agent.knowledge_collections.add(collection)

        for document_spec in collection_spec["documents"]:
            document = KnowledgeDocument.objects.create(
                collection=collection,
                title=document_spec["title"],
                curated_text=(
                    _curated_text_for(document_spec) if populate_curated_text else ""
                ),
                tags=list(document_spec["tags"]),
                language=document_spec["language"],
                status=document_spec["status"],
            )
            document_key_by_id[document.pk] = document_spec["key"]
            document_id_by_key[document_spec["key"]] = document.pk

            for index, section_title, content in document_spec["chunks"]:
                chunk = KnowledgeDocumentChunk.objects.create(
                    document=document,
                    chunk_index=index,
                    section_title=section_title,
                    content=content,
                    token_estimate=len(content.split()),
                )
                chunk_key_by_id[chunk.pk] = f"{document_spec['key']}/{index}"

    return chunk_key_by_id, document_key_by_id, document_id_by_key


def expected_documents_for(golden) -> set:
    """Project a golden query's expected chunks onto their parent documents."""
    return {key.split("/")[0] for key in golden.expected}


# ---------------------------------------------------------------------------
# Path 2 invocation
# ---------------------------------------------------------------------------


@dataclass
class _ActionRunStub:
    """Minimal stand-in for GameActionRun when calling the handler directly.

    The handler reads only ``_resolved_effective_agent`` (absent here, so the
    real ``resolve_game_entry_agent`` runs) and ``session``. Using a stub keeps
    the retrieval characterization free of policy, budget and approval
    machinery, which is exercised separately in GameRetrievalAuditTests.
    """

    session: ExecutionSession


def run_path2_search(session, query: str) -> dict:
    """Invoke the CURRENT GAME compatibility search handler unchanged."""
    return _handle_search_knowledge(
        _ActionRunStub(session=session), None, None, {"query": query}
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _round(value):
    return round(float(value), 4)


def evaluate_path2_query(session, document_key_by_id, golden) -> dict:
    result = run_path2_search(session, golden.query)
    rows = result["knowledge_context"]
    returned = [document_key_by_id[row["document_id"]] for row in rows]
    expected = expected_documents_for(golden)

    hits = [key for key in returned if key in expected]
    unauthorized = [key for key in returned if key in FORBIDDEN_DOCUMENT_KEYS]
    returned_chars = sum(len(row["snippet"]) for row in rows)

    if not returned:
        outcome = "EMPTY"
    elif hits:
        outcome = "HIT"
    else:
        outcome = "NOISE"

    return {
        "query_id": golden.query_id,
        "category": golden.category,
        "query": golden.query,
        "outcome": outcome,
        "returned": len(returned),
        "returned_keys": tuple(returned),
        "expected_count": len(expected),
        "hit_count": len(hits),
        "recall_at_docs": _round(len(hits) / len(expected)) if expected else None,
        "precision_at_docs": _round(len(hits) / len(returned)) if returned else None,
        "unauthorized_hits": len(unauthorized),
        "returned_chars": returned_chars,
        "matched_documents": result["matched_documents"],
    }


def evaluate_path1_query_at_document_level(agent, chunk_key_by_id, golden) -> dict:
    """Path 1, projected onto documents so the two paths can be compared."""
    result = search_knowledge(agent, query=golden.query, limit=SEARCH_LIMIT)
    ordered = []
    for row in result["results"]:
        document_key = chunk_key_by_id[row["chunk_id"]].split("/")[0]
        if document_key not in ordered:
            ordered.append(document_key)
    expected = expected_documents_for(golden)
    hits = [key for key in ordered if key in expected]
    returned_chars = sum(len(row["snippet"]) for row in result["results"])

    if not ordered:
        outcome = "EMPTY"
    elif hits:
        outcome = "HIT"
    else:
        outcome = "NOISE"

    return {
        "query_id": golden.query_id,
        "outcome": outcome,
        "returned": len(ordered),
        "returned_keys": tuple(ordered),
        "expected_count": len(expected),
        "hit_count": len(hits),
        "recall_at_docs": _round(len(hits) / len(expected)) if expected else None,
        "precision_at_docs": _round(len(hits) / len(ordered)) if ordered else None,
        "unauthorized_hits": sum(1 for key in ordered if key in FORBIDDEN_DOCUMENT_KEYS),
        "returned_chars": returned_chars,
        "chunks_returned": len(result["results"]),
    }


def _aggregate(records, *, recall_key, precision_key):
    answerable = [r for r in records if r["expected_count"]]
    non_empty = [r for r in records if r["returned"]]
    unanswerable = [r for r in records if not r["expected_count"]]
    total_results = sum(r["returned"] for r in records)
    return {
        "queries": len(records),
        "answerable_queries": len(answerable),
        "total_results_returned": total_results,
        "mean_recall_at_docs": _round(
            sum(r[recall_key] for r in answerable) / len(answerable)
        ),
        "mean_precision_at_docs": _round(
            sum(r[precision_key] for r in non_empty) / len(non_empty)
        )
        if non_empty
        else None,
        "queries_returning_nothing": sum(1 for r in records if not r["returned"]),
        "unanswerable_queries_returning_results": sum(
            1 for r in unanswerable if r["returned"]
        ),
        "answerable_queries_fully_missed": sum(
            1 for r in answerable if not r["hit_count"]
        ),
        "unauthorized_hits": sum(r["unauthorized_hits"] for r in records),
        "total_returned_chars": sum(r["returned_chars"] for r in records),
    }


def evaluate_path2_golden_set(session, document_key_by_id):
    records = [
        evaluate_path2_query(session, document_key_by_id, golden)
        for golden in GOLDEN_QUERIES
    ]
    return records, _aggregate(
        records, recall_key="recall_at_docs", precision_key="precision_at_docs"
    )


def evaluate_path1_golden_set_at_document_level(agent, chunk_key_by_id):
    records = [
        evaluate_path1_query_at_document_level(agent, chunk_key_by_id, golden)
        for golden in GOLDEN_QUERIES
    ]
    return records, _aggregate(
        records, recall_key="recall_at_docs", precision_key="precision_at_docs"
    )


def format_comparison(path1_records, path1_aggregate, path2_records, path2_aggregate):
    by_id = {record["query_id"]: record for record in path1_records}
    lines = [
        "",
        "=" * 104,
        "PATH 1 (canonical chunk retrieval) vs PATH 2 (GAME compatibility retrieval)",
        "document-granularity comparison; see COMPARABILITY for what cannot be compared",
        "=" * 104,
        f"{'ID':<5}{'CATEGORY':<26}{'P1':<7}{'P2':<7}{'P1 DOCS':>8}{'P2 DOCS':>8}"
        f"{'P1 CHR':>8}{'P2 CHR':>8}  QUERY",
        "-" * 104,
    ]
    for record in path2_records:
        one = by_id[record["query_id"]]
        lines.append(
            f"{record['query_id']:<5}"
            f"{record['category']:<26}"
            f"{one['outcome']:<7}"
            f"{record['outcome']:<7}"
            f"{one['returned']:>8}"
            f"{record['returned']:>8}"
            f"{one['returned_chars']:>8}"
            f"{record['returned_chars']:>8}"
            f"  {record['query']}"
        )
    lines.append("-" * 104)
    lines.append("  PATH 1 (document granularity)")
    for key, value in path1_aggregate.items():
        lines.append(f"    {key:<42} {value}")
    lines.append("  PATH 2")
    for key, value in path2_aggregate.items():
        lines.append(f"    {key:<42} {value}")
    lines.append("=" * 104)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recorded baselines
# ---------------------------------------------------------------------------
# CHARACTERIZATION METRICS, not architectural requirements. They describe
# CURRENT behavior of both paths on the shared corpus so that convergence work
# can be measured as a diff. `unauthorized_hits` is the exception: it is an
# ARCHITECTURAL INVARIANT and must stay 0 on both paths forever.

PATH1_DOCUMENT_LEVEL_BASELINE = {
    "queries": 31,
    "answerable_queries": 20,
    "total_results_returned": 42,
    "mean_recall_at_docs": 0.9,
    "mean_precision_at_docs": 0.6449,
    "queries_returning_nothing": 8,
    "unanswerable_queries_returning_results": 4,
    "answerable_queries_fully_missed": 2,
    "unauthorized_hits": 0,  # INVARIANT
    "total_returned_chars": 3859,
}

PATH2_RECORDED_BASELINE = {
    "queries": 31,
    "answerable_queries": 20,
    # Path 2 recovers the expected document for far fewer queries...
    "mean_recall_at_docs": 0.65,
    # ...but what it does return is more often relevant, because whole-string
    # matching either hits exactly or returns nothing. High precision here is a
    # side effect of low sensitivity, not of good ranking.
    "mean_precision_at_docs": 0.8125,
    "total_results_returned": 27,
    "queries_returning_nothing": 15,
    "unanswerable_queries_returning_results": 3,
    "answerable_queries_fully_missed": 7,
    "unauthorized_hits": 0,  # INVARIANT
    # More characters from fewer results: Path 2 returns whole documents.
    "total_returned_chars": 4348,
}

PATH2_RECORDED_OUTCOMES = {
    "G01": "HIT",    # refund policy
    "G02": "HIT",    # gift cards
    "G03": "HIT",    # secret storage
    "G04": "HIT",    # escalation path
    "G05": "HIT",    # PROD-10252-01
    "G06": "HIT",    # PROD-10251-99
    "G07": "EMPTY",  # exact id + one word -> whole-string match fails entirely
    "G08": "HIT",    # contracts
    "G09": "HIT",    # incidents
    "G10": "EMPTY",  # DRAFT document correctly excluded
    "G11": "EMPTY",  # synonym
    "G12": "EMPTY",  # synonym
    "G13": "EMPTY",  # conceptual
    "G14": "EMPTY",  # conceptual
    "G15": "EMPTY",  # natural-language question
    "G16": "NOISE",  # the
    "G17": "HIT",    # identifier
    "G18": "HIT",    # days
    "G19": "EMPTY",  # no genuine match
    "G20": "EMPTY",  # no genuine match
    "G21": "NOISE",  # ion
    "G22": "NOISE",  # act
    "G23": "HIT",    # cred
    "G24": "EMPTY",  # restricted collection
    "G25": "EMPTY",  # restricted collection
    "G26": "EMPTY",  # restricted answer; no substitute offered (see findings)
    "G27": "EMPTY",  # DRAFT document
    "G28": "EMPTY",  # inactive collection
    "G29": "EMPTY",  # multi-word query
    "G30": "HIT",    # record
    "G31": "HIT",    # within
}


class GameRetrievalCharacterizationTests(TestCase):
    """Characterizes Path 2 against Path 1 on one shared, equivalent corpus."""

    @classmethod
    def setUpTestData(cls):
        provider = ProviderConfig.objects.create(
            name="game-baseline-provider",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        cls.agent = AgentProfile.objects.create(
            name="game-baseline-agent",
            role="GAME retrieval characterization",
            model_config=model,
        )
        (
            cls.chunk_key_by_id,
            cls.document_key_by_id,
            cls.document_id_by_key,
        ) = build_dual_path_corpus(cls.agent)
        cls.session = ExecutionSession.objects.create(
            entry_agent=cls.agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            status=ExecutionSession.Status.RUNNING,
            goal_text="Characterize GAME compatibility retrieval.",
        )

    # -- corpus equivalence ----------------------------------------------

    def test_path1_behaves_identically_on_the_slice2_corpus(self):
        """Adding curated_text must not change Path 1 at all.

        Path 1 searches chunk content/section and document title/tags; it never
        reads curated_text. Asserting the Slice 1 aggregate reproduces exactly
        proves the comparison below is apples-to-apples.
        """
        _records, aggregate = evaluate_golden_set(self.agent, self.chunk_key_by_id)
        self.assertEqual(aggregate, RECORDED_BASELINE)

    def test_every_document_carries_both_chunk_and_curated_text(self):
        for document in KnowledgeDocument.objects.all():
            with self.subTest(document=document.title):
                self.assertTrue(document.curated_text.strip())
                self.assertTrue(document.chunks.exists())

    # -- recorded baselines ----------------------------------------------

    def test_path1_document_level_baseline(self):
        _records, aggregate = evaluate_path1_golden_set_at_document_level(
            self.agent, self.chunk_key_by_id
        )
        self.assertEqual(aggregate, PATH1_DOCUMENT_LEVEL_BASELINE)

    def test_path2_recorded_baseline(self):
        _records, aggregate = evaluate_path2_golden_set(
            self.session, self.document_key_by_id
        )
        self.assertEqual(aggregate, PATH2_RECORDED_BASELINE)

    def test_path2_per_query_outcomes_are_pinned(self):
        records, _aggregate = evaluate_path2_golden_set(
            self.session, self.document_key_by_id
        )
        observed = {record["query_id"]: record["outcome"] for record in records}
        self.assertEqual(observed, PATH2_RECORDED_OUTCOMES)

    # -- ARCHITECTURAL INVARIANTS (must never regress) --------------------

    def test_invariant_path2_never_returns_unauthorized_knowledge(self):
        """Restricted, inactive-collection and DRAFT documents never surface."""
        records, aggregate = evaluate_path2_golden_set(
            self.session, self.document_key_by_id
        )
        self.assertEqual(aggregate["unauthorized_hits"], 0)
        for record in records:
            for key in record["returned_keys"]:
                self.assertNotIn(key, FORBIDDEN_DOCUMENT_KEYS, msg=record["query_id"])

    def test_invariant_path2_excludes_each_out_of_scope_mechanism(self):
        for query in (
            "zephyr-restricted-marker",  # collection not attached to the agent
            "archived-collection-marker",  # collection attached but inactive
            "draft-status-marker",  # document status is DRAFT
        ):
            with self.subTest(query=query):
                result = run_path2_search(self.session, query)
                self.assertEqual(result["knowledge_context"], [])
                self.assertEqual(result["matched_documents"], 0)

    def test_invariant_path2_read_document_blocks_out_of_scope_documents(self):
        for key in sorted(FORBIDDEN_DOCUMENT_KEYS):
            with self.subTest(document=key):
                with self.assertRaisesMessage(
                    ValidationError, "not found or not accessible"
                ):
                    _handle_read_document(
                        _ActionRunStub(session=self.session),
                        None,
                        None,
                        {"document_id": self.document_id_by_key[key]},
                    )

    def test_invariant_path2_requires_a_resolvable_effective_agent(self):
        orphan_session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            status=ExecutionSession.Status.RUNNING,
            entry_agent=self.agent,
            goal_text="Orphan.",
        )
        orphan_session.entry_agent = None
        with self.assertRaisesMessage(ValidationError, "effective GAME agent"):
            run_path2_search(orphan_session, "refund")

    # -- measured differences (CHARACTERIZATION) --------------------------

    def test_path2_matches_the_whole_query_string_not_tokens(self):
        """The single largest behavioral difference between the two paths.

        Path 1 tokenizes into words. Path 2 tests `query.lower() in field`, so a
        multi-word natural-language query matches only if that exact string is
        present. Most such queries therefore return nothing at all.
        """
        tokenized = search_knowledge(
            self.agent, query="what is the refund process", limit=SEARCH_LIMIT
        )
        whole_string = run_path2_search(self.session, "what is the refund process")
        self.assertTrue(tokenized["results"])
        self.assertEqual(whole_string["knowledge_context"], [])

    def test_path2_applies_no_ranking_at_all(self):
        """Output order is (collection name, document title). Always.

        No score is computed, so ordering carries no relevance signal.
        """
        result = run_path2_search(self.session, "refund")
        returned = [
            (row["collection"], row["title"]) for row in result["knowledge_context"]
        ]
        self.assertEqual(returned, sorted(returned))

    def test_path2_has_no_result_count_limit(self):
        """Path 1 caps at top-K; Path 2 stops only at its character budget.

        With a common term Path 2 reaches more documents than Path 1 can, and
        the caller has no way to ask for fewer - a `limit` in the payload is
        silently ignored because the handler never reads one.
        """
        path1 = search_knowledge(self.agent, query="the", limit=SEARCH_LIMIT)
        path1_documents = {
            self.chunk_key_by_id[row["chunk_id"]].split("/")[0]
            for row in path1["results"]
        }
        path2 = run_path2_search(self.session, "the")

        self.assertEqual(len(path1["results"]), SEARCH_LIMIT)
        self.assertGreater(len(path2["knowledge_context"]), len(path1_documents))

        capped = _handle_search_knowledge(
            _ActionRunStub(session=self.session), None, None, {"query": "the", "limit": 1}
        )
        self.assertEqual(
            len(capped["knowledge_context"]), len(path2["knowledge_context"])
        )

    def test_path2_returns_whole_documents_not_targeted_evidence(self):
        """Path 2 hands back the full curated_text prefix, not a focused chunk."""
        path1 = search_knowledge(self.agent, query="refund", limit=SEARCH_LIMIT)
        path2 = run_path2_search(self.session, "refund")
        refund_document = KnowledgeDocument.objects.get(title="Refund Policy")
        path2_refund = next(
            row
            for row in path2["knowledge_context"]
            if row["document_id"] == refund_document.pk
        )
        # Path 2 returns the entire document text; Path 1 returns a bounded
        # window around the match.
        self.assertEqual(path2_refund["snippet"], refund_document.curated_text)
        for row in path1["results"]:
            self.assertLessEqual(len(row["snippet"]), 500)

    def test_path2_exposes_no_chunk_identity_score_or_citation(self):
        result = run_path2_search(self.session, "refund")
        self.assertTrue(result["knowledge_context"])
        for row in result["knowledge_context"]:
            self.assertEqual(
                set(row), {"document_id", "title", "collection", "snippet"}
            )
        self.assertEqual(
            set(result), {"action_name", "query", "knowledge_context", "matched_documents"}
        )

    def test_path2_reports_no_candidate_or_truncation_counters(self):
        """Path 2 can silently stop at its character budget with no signal."""
        result = run_path2_search(self.session, "refund")
        for key in ("candidates_scanned", "candidate_limit", "candidates_truncated"):
            self.assertNotIn(key, result)

    def test_path2_exact_identifier_behavior(self):
        """Exact identifiers are found, at document granularity, unranked."""
        result = run_path2_search(self.session, "PROD-10252-01")
        returned = [
            self.document_key_by_id[row["document_id"]]
            for row in result["knowledge_context"]
        ]
        self.assertEqual(returned, ["release"])
        # The whole document comes back, not the matching section.
        self.assertIn("PROD-10251-99", result["knowledge_context"][0]["snippet"])

    def test_path2_substring_overmatching_is_also_present(self):
        """Path 2 shares Path 1's substring behavior, without any score to show."""
        for query in ("act", "the", "ion"):
            with self.subTest(query=query):
                result = run_path2_search(self.session, query)
                self.assertTrue(result["knowledge_context"])

    def test_path2_no_answer_returns_a_shaped_empty_result(self):
        result = run_path2_search(self.session, "quantum photolithography calibration")
        self.assertEqual(result["knowledge_context"], [])
        self.assertEqual(result["matched_documents"], 0)
        self.assertEqual(result["query"], "quantum photolithography calibration")

    def test_path2_rejects_an_empty_query(self):
        """One behavior Path 2 shares with Path 1: empty queries are refused."""
        with self.assertRaisesMessage(ValidationError, "non-empty 'query'"):
            run_path2_search(self.session, "   ")

    def test_path2_does_not_bound_the_query_length(self):
        """Path 1 caps the query at 1,000 chars / 20 words. Path 2 caps nothing.

        A very long query is harmless here only because a long string is
        unlikely to be a substring of anything.
        """
        long_query = "refund " * 500
        result = run_path2_search(self.session, long_query)
        self.assertEqual(result["query"], long_query.strip())
        self.assertEqual(result["knowledge_context"], [])

    # -- the security / no-answer scenario that must survive convergence --

    def test_restricted_answer_with_plausible_authorized_substitutes(self):
        """The G26 scenario, on Path 2. Long-term regression scenario.

        The true answer to `credential resets` exists only in a collection the
        Agent is not authorized for. Authorization holds - the restricted
        document is correctly withheld - but the Agent is then handed whole
        authorized documents with no signal that its question went unanswered.

        Path 2 makes this worse than Path 1 in one specific way: it returns the
        ENTIRE substitute document rather than a bounded snippet, so the
        irrelevant material reaching the model is larger.
        """
        path1 = search_knowledge(self.agent, query="credential", limit=SEARCH_LIMIT)
        path2 = run_path2_search(self.session, "credential")

        path1_keys = {
            self.chunk_key_by_id[row["chunk_id"]].split("/")[0]
            for row in path1["results"]
        }
        path2_keys = {
            self.document_key_by_id[row["document_id"]]
            for row in path2["knowledge_context"]
        }

        # INVARIANT: the restricted document is withheld on both paths.
        self.assertNotIn("restricted_matrix", path1_keys)
        self.assertNotIn("restricted_matrix", path2_keys)

        # CHARACTERIZATION: both return plausible substitutes instead of
        # signalling that the question cannot be answered in scope.
        self.assertEqual(path1_keys, {"credentials"})
        self.assertEqual(path2_keys, {"credentials"})

        # Path 2 hands over the WHOLE substitute document; Path 1 hands over
        # bounded windows. The irrelevant material reaching the model is larger.
        credential_document = KnowledgeDocument.objects.get(
            title="Credential Management Standard"
        )
        self.assertEqual(
            path2["knowledge_context"][0]["snippet"],
            credential_document.curated_text,
        )

    # -- read_document ----------------------------------------------------

    def test_path2_read_document_returns_a_capped_whole_document(self):
        document_id = self.document_id_by_key["refund"]
        result = _handle_read_document(
            _ActionRunStub(session=self.session), None, None, {"document_id": document_id}
        )
        self.assertEqual(
            set(result), {"action_name", "document_id", "title", "content"}
        )
        document = KnowledgeDocument.objects.get(pk=document_id)
        self.assertEqual(result["content"], document.curated_text[:PATH2_READ_DOCUMENT_CHAR_CAP])
        # No chunk addressing, no section, no citation.
        self.assertNotIn("chunk_id", result)
        self.assertNotIn("citation", result)

    def test_path2_read_document_rejects_a_non_numeric_id(self):
        with self.assertRaisesMessage(ValidationError, "numeric 'document_id'"):
            _handle_read_document(
                _ActionRunStub(session=self.session), None, None, {"document_id": "abc"}
            )

    # -- report -----------------------------------------------------------

    def test_comparison_report_renders(self):
        path1_records, path1_aggregate = evaluate_path1_golden_set_at_document_level(
            self.agent, self.chunk_key_by_id
        )
        path2_records, path2_aggregate = evaluate_path2_golden_set(
            self.session, self.document_key_by_id
        )
        report = format_comparison(
            path1_records, path1_aggregate, path2_records, path2_aggregate
        )
        self.assertIn("PATH 1", report)
        for golden in GOLDEN_QUERIES:
            self.assertIn(golden.query_id, report)

    def test_comparability_classification_is_complete(self):
        self.assertTrue(COMPARABILITY)
        for dimension, classification, reason in COMPARABILITY:
            with self.subTest(dimension=dimension):
                self.assertIn(classification, {SUPPORTED, PARTIAL, NOT_COMPARABLE})
                self.assertTrue(reason)


class KnowledgeFieldDivergenceTests(TestCase):
    """The two paths read DIFFERENT FIELDS. This is a convergence blocker.

    Path 1 indexes KnowledgeDocumentChunk. Path 2 reads
    KnowledgeDocument.curated_text. A document populated for one path can be
    entirely invisible to the other, in both directions.
    """

    @classmethod
    def setUpTestData(cls):
        provider = ProviderConfig.objects.create(
            name="divergence-provider",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        cls.agent = AgentProfile.objects.create(
            name="divergence-agent",
            role="Field divergence characterization",
            model_config=model,
        )
        collection = KnowledgeCollection.objects.create(name="Divergence Collection")
        cls.agent.knowledge_collections.add(collection)
        cls.session = ExecutionSession.objects.create(
            entry_agent=cls.agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            status=ExecutionSession.Status.RUNNING,
            goal_text="Field divergence.",
        )

        # Chunks only: the shape produced by curated chunk authoring, and the
        # shape the Slice 1 corpus uses.
        chunk_only = KnowledgeDocument.objects.create(
            collection=collection,
            title="Chunked Only Document",
            curated_text="",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        KnowledgeDocumentChunk.objects.create(
            document=chunk_only,
            chunk_index=1,
            section_title="Chunk Section",
            content="chunkonlymarker is reachable through chunk retrieval.",
        )

        # curated_text only: the shape migration 0019 backfills from, before
        # any explicit chunking, and the only shape Path 2 can read.
        cls.curated_only = KnowledgeDocument.objects.create(
            collection=collection,
            title="Curated Only Document",
            curated_text="curatedonlymarker is reachable through document retrieval.",
            status=KnowledgeDocument.Status.ACTIVE,
        )

    def test_chunk_only_document_is_invisible_to_path2(self):
        path1 = search_knowledge(self.agent, query="chunkonlymarker", limit=SEARCH_LIMIT)
        path2 = run_path2_search(self.session, "chunkonlymarker")
        self.assertEqual(len(path1["results"]), 1)
        self.assertEqual(path2["knowledge_context"], [])

    def test_curated_text_only_document_is_invisible_to_path1(self):
        """No chunk exists, so canonical retrieval cannot see it.

        `ensure_initial_knowledge_chunk()` would create one, but nothing calls it
        automatically on document creation.
        """
        path1 = search_knowledge(self.agent, query="curatedonlymarker", limit=SEARCH_LIMIT)
        path2 = run_path2_search(self.session, "curatedonlymarker")
        self.assertEqual(path1["results"], [])
        self.assertEqual(len(path2["knowledge_context"]), 1)

    def test_path2_can_match_a_title_and_return_an_empty_snippet(self):
        """A title/tag match with no curated_text yields evidence-free output.

        The document is reported as matched, but the snippet is empty, so the
        model receives a citation-less claim with no supporting text.
        """
        result = run_path2_search(self.session, "Chunked Only Document")
        self.assertEqual(result["matched_documents"], 1)
        self.assertEqual(result["knowledge_context"][0]["snippet"], "")


@override_settings(
    AI_HUB_GAME_GOALS_ENABLED=True,
    AI_HUB_GAME_ACTION_DISPATCH_ENABLED=True,
)
class GameRetrievalAuditTests(TestCase):
    """Characterizes Path 2 through the real dispatcher: audit and identity."""

    @classmethod
    def setUpTestData(cls):
        provider = ProviderConfig.objects.create(
            name="game-audit-provider",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        cls.agent = AgentProfile.objects.create(
            name="game-audit-agent",
            role="GAME retrieval audit characterization",
            model_config=model,
        )
        build_dual_path_corpus(cls.agent)
        cls.workspace = create_workspace(name="GAME retrieval audit workspace")
        cls.goal = create_goal(
            workspace=cls.workspace,
            title="Characterize GAME retrieval audit",
            description="Slice 2.",
        )
        cls.session = ExecutionSession.objects.create(
            entry_agent=cls.agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            status=ExecutionSession.Status.RUNNING,
            goal=cls.goal,
            goal_text="Slice 2.",
        )
        GameActionDefinition.objects.create(
            name="search_knowledge",
            label="Search knowledge",
            action_type=GameActionDefinition.ActionType.CONTEXT_TOOL,
        )

    def test_path2_is_audited_by_gameactionrun_not_toolexecutionrun(self):
        """Both paths are audited, by different models with different fields."""
        action_run = execute_game_action(
            session=self.session,
            action_name="search_knowledge",
            action_input={"query": "refund"},
        )
        self.assertEqual(action_run.status, GameActionRun.Status.SUCCESS)
        self.assertEqual(action_run.session, self.session)
        self.assertEqual(action_run.input_payload, {"query": "refund"})
        self.assertIn("knowledge_context", action_run.output_payload)
        self.assertIsNotNone(action_run.latency_ms)
        self.assertTrue(action_run.idempotency_key)

        # No ToolExecutionRun is created, because no ToolDefinition is involved.
        self.assertFalse(ToolExecutionRun.objects.exists())

    def test_path2_audit_stores_full_retrieved_document_text(self):
        """Same data-governance consideration as Path 1 (C-11), different model."""
        action_run = execute_game_action(
            session=self.session,
            action_name="search_knowledge",
            action_input={"query": "identifier"},
        )
        stored = action_run.output_payload["knowledge_context"]
        self.assertTrue(stored)
        document = KnowledgeDocument.objects.get(title="Release Register")
        self.assertEqual(stored[0]["snippet"], document.curated_text)

    def test_path2_audit_records_no_retrieval_diagnostics(self):
        """No scores, no candidate counters, no authorization scope recorded."""
        action_run = execute_game_action(
            session=self.session,
            action_name="search_knowledge",
            action_input={"query": "contracts"},
        )
        self.assertEqual(
            set(action_run.output_payload),
            {"action_name", "query", "knowledge_context", "matched_documents"},
        )

    def test_path2_identity_comes_from_the_session_not_the_payload(self):
        """A model-supplied agent identifier is ignored: there is no such field.

        Path 1 defends this actively (bind_tool_runtime_context strips
        agent_name/agent_id). Path 2 is safe for a different reason - it never
        reads an identity from the payload at all, taking it from the session.
        Convergence must not lose this property.
        """
        other_agent = AgentProfile.objects.create(
            name="game-audit-other-agent",
            role="Should never be used",
            model_config=self.agent.model_config,
        )
        action_run = execute_game_action(
            session=self.session,
            action_name="search_knowledge",
            action_input={
                "query": "refund",
                "agent_id": other_agent.pk,
                "agent_name": other_agent.name,
            },
        )
        # The extra keys are audited verbatim but have no effect on scope.
        self.assertEqual(action_run.input_payload["agent_id"], other_agent.pk)
        titles = {row["title"] for row in action_run.output_payload["knowledge_context"]}
        self.assertIn("Refund Policy", titles)
        self.assertNotIn("Restricted Escalation Matrix", titles)

    def test_identical_repeated_action_is_deduplicated_by_idempotency_key(self):
        """A GAME-specific behavior Path 1 has no equivalent for."""
        first = execute_game_action(
            session=self.session,
            action_name="search_knowledge",
            action_input={"query": "days"},
        )
        second = execute_game_action(
            session=self.session,
            action_name="search_knowledge",
            action_input={"query": "days"},
        )
        self.assertEqual(first.pk, second.pk)
