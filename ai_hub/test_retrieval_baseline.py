"""Deterministic lexical retrieval baseline for reusable Knowledge (RAG Slice 1).

This module is a MEASURING INSTRUMENT. It changes no production behavior.

Why it exists
-------------
Several Knowledge acceptance goals are comparative ("adding semantic retrieval
must not destroy exact-match precision"), but no measurement of CURRENT lexical
retrieval existed. This module records that measurement so later retrieval work
is argued from numbers instead of impressions.

What it deliberately does NOT do
--------------------------------
It does not fix, improve or work around any current behavior. Substring
matching, the absent stopword list, the score floor of 1, the absent relevance
threshold and alphabetical candidate truncation are all CURRENT behavior and
are measured here exactly as they are. Changing any of them is a separate,
evidence-gated decision.

Properties this harness must keep
---------------------------------
* Deterministic — a fixed fixture corpus, no randomness, no ordering ambiguity.
* LLM-free — retrieval quality is evaluated on its own, with no model call.
* Backend-independent — ASCII-only corpus, no backend-specific query features,
  so SQLite and PostgreSQL agree.
* Core-owned — no dependency on any host application.
"""
from dataclasses import dataclass

from unittest.mock import patch

from django.test import TestCase

from ai_hub.models import (
    AgentProfile,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    ProviderConfig,
)
from ai_hub.services.knowledge_retrieval import search_knowledge


# Matches the retrieval service default and the seeded tool adapter default.
SEARCH_LIMIT = 5

CITATION_KEYS = {
    "collection",
    "document_id",
    "document_title",
    "chunk_id",
    "section_title",
    "chunk_index",
    "language",
    "tags",
}


# ---------------------------------------------------------------------------
# Fixture corpus
# ---------------------------------------------------------------------------
# ASCII only, so `icontains` behaves identically on SQLite and PostgreSQL.
# Collection names are chosen so alphabetical candidate ordering is explicit
# rather than accidental.

CORPUS = (
    {
        "key": "policies",
        "name": "Baseline Platform Policies",
        "description": "Approved platform policies used by the retrieval baseline.",
        "is_active": True,
        "attached": True,
        "documents": (
            {
                "key": "refund",
                "title": "Refund Policy",
                "tags": ["refunds", "billing"],
                "language": "en",
                "status": KnowledgeDocument.Status.ACTIVE,
                "chunks": (
                    (
                        1,
                        "Eligibility Window",
                        "Refunds are available within thirty days when the order is unused.",
                    ),
                    (
                        2,
                        "Exceptions",
                        "Digital gift cards are not refundable under any circumstance.",
                    ),
                ),
            },
            {
                "key": "release",
                "title": "Release Register",
                "tags": ["release", "identifiers"],
                "language": "en",
                "status": KnowledgeDocument.Status.ACTIVE,
                "chunks": (
                    (
                        1,
                        "Component Identifiers",
                        "The billing component ships under identifier PROD-10252-01 "
                        "for every supported region.",
                    ),
                    (
                        2,
                        "Superseded Identifiers",
                        "Identifier PROD-10251-99 was retired before general availability.",
                    ),
                ),
            },
            {
                "key": "contract",
                "title": "Contract Administration",
                "tags": ["contracts"],
                "language": "en",
                "status": KnowledgeDocument.Status.ACTIVE,
                "chunks": (
                    (
                        1,
                        "Contract Actions",
                        "Every contract action and transaction is recorded by the "
                        "administration team.",
                    ),
                ),
            },
            {
                # DRAFT: must never be retrievable. Contains terms that overlap
                # several golden queries so a status-filter regression is loud.
                "key": "draft",
                "title": "Draft Onboarding Guide",
                "tags": ["onboarding", "refunds"],
                "language": "en",
                "status": KnowledgeDocument.Status.DRAFT,
                "chunks": (
                    (
                        1,
                        "Draft Section",
                        "draft-status-marker Refunds and credential rotation for new tenants.",
                    ),
                ),
            },
        ),
    },
    {
        "key": "security",
        "name": "Baseline Security Standards",
        "description": "Security standards used by the retrieval baseline.",
        "is_active": True,
        "attached": True,
        "documents": (
            {
                "key": "credentials",
                "title": "Credential Management Standard",
                "tags": ["credentials", "secrets"],
                "language": "en",
                "status": KnowledgeDocument.Status.ACTIVE,
                "chunks": (
                    (
                        1,
                        "Secret Storage",
                        "Secret storage requires an approved vault and rotation "
                        "every ninety days.",
                    ),
                    (
                        2,
                        "Authentication Data",
                        "Authentication data must never be written to application logs.",
                    ),
                ),
            },
            {
                "key": "incident",
                "title": "Incident Response Runbook",
                "tags": ["incidents"],
                "language": "en",
                "status": KnowledgeDocument.Status.ACTIVE,
                "chunks": (
                    (
                        1,
                        "Escalation Path",
                        "Escalate a suspected breach to the security duty officer "
                        "within one hour.",
                    ),
                    (
                        2,
                        "Evidence Handling",
                        "Preserve evidence and record the incident timeline before "
                        "remediation.",
                    ),
                ),
            },
        ),
    },
    {
        # Active collection that is NOT attached to the agent. Deliberately
        # shares vocabulary with authorized documents so an authorization
        # regression shows up as a measured unauthorized hit, not as silence.
        "key": "restricted",
        "name": "Baseline Restricted Corpus",
        "description": "Restricted corpus the baseline agent must never reach.",
        "is_active": True,
        "attached": False,
        "documents": (
            {
                "key": "restricted_matrix",
                "title": "Restricted Escalation Matrix",
                "tags": ["restricted", "refunds", "credentials"],
                "language": "en",
                "status": KnowledgeDocument.Status.ACTIVE,
                "chunks": (
                    (
                        1,
                        "Restricted Contacts",
                        "zephyr-restricted-marker Refund approvals and credential "
                        "resets require director sign-off.",
                    ),
                ),
            },
        ),
    },
    {
        # Inactive collection that IS attached to the agent. Exercises the
        # collection-level active filter independently of attachment.
        "key": "archived",
        "name": "Baseline Archived Corpus",
        "description": "Archived corpus attached to the agent but deactivated.",
        "is_active": False,
        "attached": True,
        "documents": (
            {
                "key": "archived_refund",
                "title": "Archived Refund Policy",
                "tags": ["refunds"],
                "language": "en",
                "status": KnowledgeDocument.Status.ACTIVE,
                "chunks": (
                    (
                        1,
                        "Archived Terms",
                        "archived-collection-marker Refunds were once available "
                        "within ninety days.",
                    ),
                ),
            },
        ),
    },
)


# Chunks that must never appear in any result, for any query, ever.
FORBIDDEN_CHUNK_KEYS = frozenset(
    {
        "restricted_matrix/1",  # unattached collection
        "archived_refund/1",  # inactive collection
        "draft/1",  # non-ACTIVE document status
    }
)


# ---------------------------------------------------------------------------
# Golden query set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenQuery:
    """One evaluation case.

    ``expected`` is a human judgement of what a *correct* retriever should
    surface. It is authored from intent, never reverse-engineered from what the
    current implementation happens to return. An empty ``expected`` means a
    correct retriever should return nothing useful.

    ``outcome`` is the *observed classification* of CURRENT behavior, recorded
    so that a regression changes a named constant rather than passing silently:

    * ``EMPTY``  - returns zero results
    * ``HIT``    - returns at least one expected chunk inside top-K
    * ``NOISE``  - returns results, none of which are expected
    """

    query_id: str
    category: str
    query: str
    expected: tuple
    outcome: str
    note: str = ""


GOLDEN_QUERIES = (
    # -- exact match ------------------------------------------------------
    GoldenQuery(
        "G01", "exact_match", "refund policy",
        ("refund/1", "refund/2"), "HIT",
        "Title and tag both carry the query terms.",
    ),
    GoldenQuery(
        "G02", "exact_match", "gift cards",
        ("refund/2",), "HIT",
        "Content-only exact phrase, terms adjacent in one chunk.",
    ),
    GoldenQuery(
        "G03", "exact_match", "secret storage",
        ("credentials/1",), "HIT",
        "Section title match should dominate.",
    ),
    GoldenQuery(
        "G04", "exact_match", "escalation path",
        ("incident/1",), "HIT",
        "Section title match with no content match.",
    ),
    # -- identifier / code lookup ----------------------------------------
    GoldenQuery(
        "G05", "id_lookup", "PROD-10252-01",
        ("release/1",), "HIT",
        "Unique identifier, content only. Watch the SCORE, not just the rank.",
    ),
    GoldenQuery(
        "G06", "id_lookup", "PROD-10251-99",
        ("release/2",), "HIT",
        "Superseded identifier must still be findable.",
    ),
    GoldenQuery(
        "G07", "id_lookup_competing", "PROD-10252-01 refund",
        ("release/1",), "HIT",
        "Exact identifier competing with a high-weight title/tag keyword.",
    ),
    # -- tags -------------------------------------------------------------
    GoldenQuery(
        "G08", "tags", "contracts",
        ("contract/1",), "HIT",
        "Tag-only match: the term appears in no title, section or content.",
    ),
    GoldenQuery(
        "G09", "tags", "incidents",
        ("incident/1", "incident/2"), "HIT",
        "Tag-only match spanning both chunks of one document.",
    ),
    GoldenQuery(
        "G10", "tags_non_active", "onboarding",
        (), "EMPTY",
        "Tag exists only on a DRAFT document. Must return nothing.",
    ),
    # -- synonyms: lexical is expected to fail ----------------------------
    GoldenQuery(
        "G11", "synonym_expected_miss", "safe password handling",
        ("credentials/1", "credentials/2"), "NOISE",
        "No shared surface form with the credential document.",
    ),
    GoldenQuery(
        "G12", "synonym_expected_miss", "password rotation policy",
        ("credentials/1",), "HIT",
        "Correct chunk is recovered only at rank 3, behind two refund chunks "
        "matched on the word 'policy'. Recall hides a ranking failure.",
    ),
    # -- conceptual: lexical is expected to fail --------------------------
    GoldenQuery(
        "G13", "conceptual_expected_miss",
        "protecting sign-in material from disclosure",
        ("credentials/1", "credentials/2"), "EMPTY",
        "Purely conceptual phrasing with zero surface overlap.",
    ),
    GoldenQuery(
        "G14", "conceptual_expected_miss",
        "who do we tell when there is a security problem",
        ("incident/1",), "HIT",
        "Correct chunk lands at rank 5 of 5, behind four common-word matches. "
        "It survives only because top-K is generous for this corpus size.",
    ),
    # -- stopword noise ---------------------------------------------------
    GoldenQuery(
        "G15", "stopword_noise", "what is the refund process",
        ("refund/1", "refund/2"), "HIT",
        "Correct answers found, but measure how much of top-K is noise.",
    ),
    GoldenQuery(
        "G16", "stopword_noise", "the",
        (), "NOISE",
        "A single stopword. A correct retriever should return nothing.",
    ),
    # -- ambiguity --------------------------------------------------------
    GoldenQuery(
        "G17", "ambiguous", "identifier",
        ("release/1", "release/2"), "HIT",
        "Current vs superseded identifier: which wins, and why?",
    ),
    GoldenQuery(
        "G18", "ambiguous", "days",
        ("refund/1", "credentials/1"), "HIT",
        "Same term, two unrelated documents in two collections.",
    ),
    # -- no genuine match -------------------------------------------------
    GoldenQuery(
        "G19", "no_genuine_match", "quantum photolithography calibration",
        (), "EMPTY",
        "Well-formed query with no corpus overlap.",
    ),
    GoldenQuery(
        "G20", "no_genuine_match", "zzzz nonexistent token",
        (), "EMPTY",
        "Nonsense query with no corpus overlap.",
    ),
    GoldenQuery(
        "G21", "floor_noise", "ion",
        (), "NOISE",
        "Meaningless fragment inside many words. Tests the absent threshold.",
    ),
    # -- substring overmatching ------------------------------------------
    GoldenQuery(
        "G22", "substring_overmatch", "act",
        (), "NOISE",
        "Fragment of contract/action/transaction. Compare its score to G05.",
    ),
    GoldenQuery(
        "G23", "substring_overmatch", "cred",
        ("credentials/1", "credentials/2"), "HIT",
        "Prefix overmatching that happens to help.",
    ),
    # -- restricted knowledge ---------------------------------------------
    GoldenQuery(
        "G24", "restricted", "zephyr-restricted-marker",
        (), "EMPTY",
        "Term unique to the unattached collection.",
    ),
    GoldenQuery(
        "G25", "restricted", "director sign-off",
        (), "EMPTY",
        "Phrase unique to the unattached collection.",
    ),
    GoldenQuery(
        "G26", "restricted_overlap", "credential resets",
        (), "NOISE",
        "True answer is restricted; agent gets plausible substitutes instead.",
    ),
    # -- inactive knowledge -----------------------------------------------
    GoldenQuery(
        "G27", "inactive_document", "draft-status-marker",
        (), "EMPTY",
        "Term unique to a DRAFT document.",
    ),
    GoldenQuery(
        "G28", "inactive_collection", "archived-collection-marker",
        (), "EMPTY",
        "Term unique to an inactive collection that IS attached.",
    ),
    GoldenQuery(
        "G29", "inactive_overlap", "refunds within ninety days",
        ("refund/1",), "HIT",
        "Archived chunk is the closer textual match and must still be excluded.",
    ),
    # -- multi-document / cross-collection --------------------------------
    GoldenQuery(
        "G30", "multi_document", "record",
        ("contract/1", "incident/2"), "HIT",
        "One term, two documents, via substring of 'recorded'/'record'.",
    ),
    GoldenQuery(
        "G31", "cross_collection", "within",
        ("refund/1", "incident/1"), "HIT",
        "One term spanning two different collections.",
    ),
)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _round(value):
    return round(float(value), 4)


def evaluate_query(agent, chunk_key_by_id, golden, *, limit=SEARCH_LIMIT):
    """Run one golden query and return its measurement record."""
    result = search_knowledge(agent, query=golden.query, limit=limit)
    returned_keys = [chunk_key_by_id[row["chunk_id"]] for row in result["results"]]
    expected = set(golden.expected)

    hits = [key for key in returned_keys if key in expected]
    unauthorized = [key for key in returned_keys if key in FORBIDDEN_CHUNK_KEYS]

    first_expected_rank = None
    first_expected_score = None
    for position, row in enumerate(result["results"], start=1):
        if chunk_key_by_id[row["chunk_id"]] in expected:
            first_expected_rank = position
            first_expected_score = row["score"]
            break

    if not returned_keys:
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
        "returned": len(returned_keys),
        "returned_keys": tuple(returned_keys),
        "expected_count": len(expected),
        "hit_count": len(hits),
        # Recall@K over the chunks a correct retriever should have surfaced.
        "recall_at_k": _round(len(hits) / len(expected)) if expected else None,
        # Precision@K over what was actually returned.
        "precision_at_k": _round(len(hits) / len(returned_keys)) if returned_keys else None,
        "reciprocal_rank": _round(1 / first_expected_rank) if first_expected_rank else 0.0,
        "first_expected_rank": first_expected_rank,
        "first_expected_score": first_expected_score,
        "top_score": result["results"][0]["score"] if result["results"] else None,
        "unauthorized_hits": len(unauthorized),
        "candidates_scanned": result["candidates_scanned"],
        "candidates_truncated": bool(result["candidates_truncated"]),
        "citations_valid": _citations_valid(agent, result),
    }


def _citations_valid(agent, result) -> bool:
    """Every returned result must carry a complete, self-consistent citation."""
    authorized_names = set(
        agent.knowledge_collections.filter(is_active=True).values_list("name", flat=True)
    )
    for row in result["results"]:
        citation = row.get("citation")
        if not isinstance(citation, dict) or set(citation) != CITATION_KEYS:
            return False
        if citation["chunk_id"] != row["chunk_id"]:
            return False
        if citation["document_id"] != row["document_id"]:
            return False
        if citation["document_title"] != row["title"]:
            return False
        if citation["collection"] != row["collection"]:
            return False
        if citation["collection"] not in authorized_names:
            return False
    return True


def evaluate_golden_set(agent, chunk_key_by_id, *, limit=SEARCH_LIMIT):
    """Run the whole golden set and return per-query records plus aggregates."""
    records = [
        evaluate_query(agent, chunk_key_by_id, golden, limit=limit)
        for golden in GOLDEN_QUERIES
    ]

    answerable = [r for r in records if r["expected_count"]]
    non_empty = [r for r in records if r["returned"]]
    unanswerable = [r for r in records if not r["expected_count"]]
    total_results = sum(r["returned"] for r in records)

    aggregate = {
        "queries": len(records),
        "answerable_queries": len(answerable),
        "unanswerable_queries": len(unanswerable),
        "total_results_returned": total_results,
        "mean_recall_at_k": _round(
            sum(r["recall_at_k"] for r in answerable) / len(answerable)
        ),
        "mean_precision_at_k": _round(
            sum(r["precision_at_k"] for r in non_empty) / len(non_empty)
        ),
        "mrr": _round(sum(r["reciprocal_rank"] for r in answerable) / len(answerable)),
        "queries_returning_nothing": sum(1 for r in records if not r["returned"]),
        # A correct retriever answers "nothing useful here" for these.
        "unanswerable_queries_returning_results": sum(
            1 for r in unanswerable if r["returned"]
        ),
        "answerable_queries_fully_missed": sum(
            1 for r in answerable if not r["hit_count"]
        ),
        "unauthorized_hits": sum(r["unauthorized_hits"] for r in records),
        "unauthorized_hit_rate": _round(
            sum(r["unauthorized_hits"] for r in records) / total_results
        )
        if total_results
        else 0.0,
        "citation_validity_rate": _round(
            sum(1 for r in non_empty if r["citations_valid"]) / len(non_empty)
        )
        if non_empty
        else 1.0,
        "queries_with_candidate_truncation": sum(
            1 for r in records if r["candidates_truncated"]
        ),
    }
    return records, aggregate


def _cell(value, spec=""):
    """Render an optional metric, using '-' when it does not apply."""
    if value is None:
        return "-"
    return format(value, spec) if spec else str(value)


def format_report(records, aggregate) -> str:
    """Human-readable baseline report. Used for the Roadmap record."""
    header = (
        f"{'ID':<5}{'CATEGORY':<26}{'OUTCOME':<8}{'RET':>4}{'EXP':>4}"
        f"{'R@K':>7}{'P@K':>7}{'RR':>7}{'RANK':>6}{'ESCORE':>8}{'TOP':>6}  QUERY"
    )
    lines = [
        "",
        "=" * 112,
        "AI HUB - CURRENT LEXICAL RETRIEVAL BASELINE (RAG Slice 1)",
        "=" * 112,
        header,
        "-" * 112,
    ]
    for record in records:
        lines.append(
            f"{record['query_id']:<5}"
            f"{record['category']:<26}"
            f"{record['outcome']:<8}"
            f"{record['returned']:>4}"
            f"{record['expected_count']:>4}"
            f"{_cell(record['recall_at_k'], '.2f'):>7}"
            f"{_cell(record['precision_at_k'], '.2f'):>7}"
            f"{record['reciprocal_rank']:>7.2f}"
            f"{_cell(record['first_expected_rank']):>6}"
            f"{_cell(record['first_expected_score']):>8}"
            f"{_cell(record['top_score']):>6}"
            f"  {record['query']}"
        )
    lines.append("-" * 112)
    for key, value in aggregate.items():
        lines.append(f"  {key:<42} {value}")
    lines.append("=" * 112)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recorded baseline
# ---------------------------------------------------------------------------
# Measured against main @ 67f139140f846693af015a4f97643df819373d2a on SQLite.
# These are NOT quality targets. They are the CURRENT behavior of the code, and
# they exist so that any future change to retrieval is visible as a diff.

RECORDED_BASELINE = {
    "queries": 31,
    "answerable_queries": 20,
    "unanswerable_queries": 11,
    "total_results_returned": 55,
    # Recall looks healthy because top-K=5 is generous against a 12-chunk
    # corpus. Read it together with precision and MRR, not on its own.
    "mean_recall_at_k": 0.9,
    "mean_precision_at_k": 0.6094,
    "mrr": 0.7933,
    "queries_returning_nothing": 8,
    # Four queries that a correct retriever should have declined still
    # returned confidently-ranked results. This is the absent threshold.
    "unanswerable_queries_returning_results": 4,
    "answerable_queries_fully_missed": 2,
    # Security invariants: these two must stay at zero forever.
    "unauthorized_hits": 0,
    "unauthorized_hit_rate": 0.0,
    "citation_validity_rate": 1.0,
    "queries_with_candidate_truncation": 0,
}


class LexicalRetrievalBaselineTests(TestCase):
    """Measures CURRENT lexical retrieval. Asserts nothing about desirability."""

    @classmethod
    def setUpTestData(cls):
        provider = ProviderConfig.objects.create(
            name="baseline-retrieval-provider",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        cls.agent = AgentProfile.objects.create(
            name="baseline-retrieval-agent",
            role="Lexical retrieval baseline",
            model_config=model,
        )
        cls.chunk_key_by_id = {}
        cls.chunk_id_by_key = {}
        cls.collections = {}

        for collection_spec in CORPUS:
            collection = KnowledgeCollection.objects.create(
                name=collection_spec["name"],
                description=collection_spec["description"],
                is_active=collection_spec["is_active"],
            )
            cls.collections[collection_spec["key"]] = collection
            if collection_spec["attached"]:
                cls.agent.knowledge_collections.add(collection)

            for document_spec in collection_spec["documents"]:
                document = KnowledgeDocument.objects.create(
                    collection=collection,
                    title=document_spec["title"],
                    tags=list(document_spec["tags"]),
                    language=document_spec["language"],
                    status=document_spec["status"],
                )
                for index, section_title, content in document_spec["chunks"]:
                    chunk = KnowledgeDocumentChunk.objects.create(
                        document=document,
                        chunk_index=index,
                        section_title=section_title,
                        content=content,
                        token_estimate=len(content.split()),
                    )
                    key = f"{document_spec['key']}/{index}"
                    cls.chunk_key_by_id[chunk.pk] = key
                    cls.chunk_id_by_key[key] = chunk.pk

    # -- corpus sanity ----------------------------------------------------

    def test_fixture_corpus_is_built_as_specified(self):
        self.assertEqual(len(self.chunk_key_by_id), 12)
        self.assertEqual(self.agent.knowledge_collections.count(), 3)
        self.assertEqual(
            self.agent.knowledge_collections.filter(is_active=True).count(), 2
        )
        for key in FORBIDDEN_CHUNK_KEYS:
            self.assertIn(key, self.chunk_id_by_key)

    def test_golden_query_ids_are_unique(self):
        ids = [golden.query_id for golden in GOLDEN_QUERIES]
        self.assertEqual(len(ids), len(set(ids)))

    # -- the recorded baseline -------------------------------------------

    def test_measured_baseline_matches_recorded_baseline(self):
        """Regression floor. A diff here means retrieval behavior changed."""
        _records, aggregate = evaluate_golden_set(self.agent, self.chunk_key_by_id)
        self.assertEqual(aggregate, RECORDED_BASELINE)

    def test_per_query_outcomes_match_recorded_outcomes(self):
        """Each golden query's observed classification is pinned by name."""
        records, _aggregate = evaluate_golden_set(self.agent, self.chunk_key_by_id)
        observed = {record["query_id"]: record["outcome"] for record in records}
        recorded = {golden.query_id: golden.outcome for golden in GOLDEN_QUERIES}
        self.assertEqual(observed, recorded)

    # -- security invariants ---------------------------------------------

    def test_no_unauthorized_chunk_is_ever_returned(self):
        """Restricted, inactive-collection and non-ACTIVE chunks never surface."""
        records, aggregate = evaluate_golden_set(self.agent, self.chunk_key_by_id)
        self.assertEqual(aggregate["unauthorized_hits"], 0)
        self.assertEqual(aggregate["unauthorized_hit_rate"], 0.0)
        for record in records:
            for key in record["returned_keys"]:
                self.assertNotIn(key, FORBIDDEN_CHUNK_KEYS, msg=record["query_id"])

    def test_terms_unique_to_out_of_scope_knowledge_return_nothing(self):
        for query in (
            "zephyr-restricted-marker",  # unattached collection
            "archived-collection-marker",  # inactive collection
            "draft-status-marker",  # DRAFT document
        ):
            with self.subTest(query=query):
                result = search_knowledge(self.agent, query=query, limit=SEARCH_LIMIT)
                self.assertEqual(result["results"], [])
                self.assertEqual(result["total"], 0)

    def test_every_returned_result_carries_a_valid_citation(self):
        records, aggregate = evaluate_golden_set(self.agent, self.chunk_key_by_id)
        self.assertEqual(aggregate["citation_validity_rate"], 1.0)
        for record in records:
            self.assertTrue(record["citations_valid"], msg=record["query_id"])

    # -- measured properties of CURRENT lexical behavior ------------------
    # Each test below documents a behavior. None of them asserts the behavior
    # is desirable; they exist so a future change to it cannot pass silently.

    def test_exact_identifier_scores_the_same_as_the_floor_value(self):
        """An exact unique identifier match scores 1 - the floor value itself.

        Score therefore carries no information about match quality for
        identifier lookups. Evidence for ADR-N7 and ADR-04.
        """
        result = search_knowledge(self.agent, query="PROD-10252-01", limit=SEARCH_LIMIT)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(
            self.chunk_key_by_id[result["results"][0]["chunk_id"]], "release/1"
        )
        self.assertEqual(result["results"][0]["score"], 1)

    def test_exact_identifier_is_outranked_by_a_title_keyword(self):
        """`<exact-id> refund` ranks the exact identifier below refund matches.

        This is the doc 05 concern, measured. Evidence for ADR-04: exact-match
        precision is already fragile before any semantic branch exists.
        """
        result = search_knowledge(
            self.agent, query="PROD-10252-01 refund", limit=SEARCH_LIMIT
        )
        keys = [self.chunk_key_by_id[row["chunk_id"]] for row in result["results"]]
        self.assertIn("release/1", keys)
        self.assertGreater(keys.index("release/1"), 0)
        identifier_row = result["results"][keys.index("release/1")]
        self.assertGreater(result["results"][0]["score"], identifier_row["score"])

    def test_a_meaningless_fragment_outscores_an_exact_identifier(self):
        """`act` (a fragment of contract/action) scores far above an exact id.

        The single clearest demonstration that `score` is uncalibrated.
        """
        fragment = search_knowledge(self.agent, query="act", limit=SEARCH_LIMIT)
        identifier = search_knowledge(
            self.agent, query="PROD-10252-01", limit=SEARCH_LIMIT
        )
        self.assertTrue(fragment["results"])
        self.assertGreater(
            fragment["results"][0]["score"], identifier["results"][0]["score"]
        )

    def test_a_single_stopword_returns_a_full_page_of_results(self):
        """`_query_words` drops only length-1 words, so `the` is a scoring term."""
        result = search_knowledge(self.agent, query="the", limit=SEARCH_LIMIT)
        self.assertEqual(len(result["results"]), SEARCH_LIMIT)

    def test_there_is_no_relevance_threshold(self):
        """Meaningless queries still return confidently-ranked results."""
        for query in ("ion", "act", "the"):
            with self.subTest(query=query):
                result = search_knowledge(self.agent, query=query, limit=SEARCH_LIMIT)
                self.assertTrue(result["results"])
                self.assertGreaterEqual(result["results"][0]["score"], 1)

    def test_ambiguity_is_resolved_by_chunk_index_not_relevance(self):
        """Tied scores fall back to (document title, chunk_index) ordering."""
        result = search_knowledge(self.agent, query="identifier", limit=SEARCH_LIMIT)
        keys = [self.chunk_key_by_id[row["chunk_id"]] for row in result["results"]]
        self.assertEqual(keys[:2], ["release/1", "release/2"])
        scores = [row["score"] for row in result["results"][:2]]
        self.assertEqual(scores[0], scores[1])

    def test_conceptual_query_with_no_surface_overlap_returns_nothing(self):
        result = search_knowledge(
            self.agent,
            query="protecting sign-in material from disclosure",
            limit=SEARCH_LIMIT,
        )
        self.assertEqual(result["results"], [])

    def test_synonym_query_returns_an_irrelevant_result_rather_than_nothing(self):
        """`safe password handling` finds `Evidence Handling`, not credentials.

        A silent wrong answer is worse than an empty one; this is the concrete
        case behind ADR-08.
        """
        result = search_knowledge(
            self.agent, query="safe password handling", limit=SEARCH_LIMIT
        )
        keys = [self.chunk_key_by_id[row["chunk_id"]] for row in result["results"]]
        self.assertTrue(keys)
        self.assertNotIn("credentials/1", keys)
        self.assertNotIn("credentials/2", keys)

    def test_correct_answer_is_buried_behind_common_word_matches(self):
        """A natural-language question puts the right chunk last.

        Recall@5 scores this query as a success. It is not one: four of the
        five results are common-word noise and the answer is at rank 5. This is
        why the baseline records precision and MRR alongside recall.
        """
        result = search_knowledge(
            self.agent,
            query="who do we tell when there is a security problem",
            limit=SEARCH_LIMIT,
        )
        keys = [self.chunk_key_by_id[row["chunk_id"]] for row in result["results"]]
        self.assertEqual(len(keys), SEARCH_LIMIT)
        self.assertEqual(keys[-1], "incident/1")

    def test_substring_matching_reaches_inside_unrelated_words(self):
        """`the` scores against `Authentication`, in both section and content.

        Concrete evidence that matching has no word boundary.
        """
        result = search_knowledge(self.agent, query="the", limit=SEARCH_LIMIT)
        top = result["results"][0]
        self.assertEqual(self.chunk_key_by_id[top["chunk_id"]], "credentials/2")
        self.assertEqual(top["section_title"], "Authentication Data")
        # section (x3) + content (x1); the literal word "the" appears in neither.
        self.assertEqual(top["score"], 4)

    def test_baseline_report_renders(self):
        """The report is the artifact recorded in the Roadmap."""
        records, aggregate = evaluate_golden_set(self.agent, self.chunk_key_by_id)
        report = format_report(records, aggregate)
        self.assertIn("CURRENT LEXICAL RETRIEVAL BASELINE", report)
        for golden in GOLDEN_QUERIES:
            self.assertIn(golden.query_id, report)


class LexicalCandidateTruncationBaselineTests(TestCase):
    """Measures truncation-before-scoring and its alphabetical bias.

    Uses a patched candidate window so the property is provable in a small,
    fast, deterministic corpus. The production constant is unchanged.
    """

    PATCHED_CANDIDATE_LIMIT = 3

    @classmethod
    def setUpTestData(cls):
        provider = ProviderConfig.objects.create(
            name="truncation-baseline-provider",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        cls.agent = AgentProfile.objects.create(
            name="truncation-baseline-agent",
            role="Candidate truncation baseline",
            model_config=model,
        )
        collection = KnowledgeCollection.objects.create(
            name="Baseline Truncation Corpus",
            description="Single collection so document title decides candidate order.",
        )
        cls.agent.knowledge_collections.add(collection)

        # Five decoys whose titles sort before the target. Each matches the
        # needle in content only, so each scores exactly 1.
        for index in range(1, 6):
            document = KnowledgeDocument.objects.create(
                collection=collection,
                title=f"Alpha Decoy {index}",
                tags=[],
                status=KnowledgeDocument.Status.ACTIVE,
            )
            KnowledgeDocumentChunk.objects.create(
                document=document,
                chunk_index=1,
                section_title=f"Decoy Section {index}",
                content="This chunk mentions truncation-needle and nothing else useful.",
            )

        # The genuinely relevant document. Its title sorts LAST, but the needle
        # appears in its tags, section title and content, so it would score
        # highest if scoring happened before truncation.
        target_document = KnowledgeDocument.objects.create(
            collection=collection,
            title="Zulu Truncation Target",
            tags=["truncation-needle"],
            status=KnowledgeDocument.Status.ACTIVE,
        )
        cls.target_chunk = KnowledgeDocumentChunk.objects.create(
            document=target_document,
            chunk_index=1,
            section_title="truncation-needle summary",
            content="The authoritative answer about truncation-needle lives here.",
        )

    def test_without_truncation_the_relevant_chunk_ranks_first(self):
        """Control: the target outscores every decoy when it is in the window."""
        result = search_knowledge(
            self.agent, query="truncation-needle", limit=SEARCH_LIMIT
        )
        self.assertFalse(result["candidates_truncated"])
        self.assertEqual(result["results"][0]["chunk_id"], self.target_chunk.pk)
        self.assertGreater(
            result["results"][0]["score"], result["results"][1]["score"]
        )

    def test_truncation_happens_before_scoring_and_is_alphabetically_biased(self):
        """With a narrow window, the best chunk is dropped for sorting late.

        The candidate window is filled in (collection, document title,
        chunk_index) order and sliced, then scoring runs on the survivors. The
        highest-scoring chunk is therefore unreachable, and the caller is told
        only that truncation occurred - not that it lost the best answer.
        """
        with patch(
            "ai_hub.services.knowledge_retrieval.MAX_SEARCH_CANDIDATES",
            self.PATCHED_CANDIDATE_LIMIT,
        ):
            result = search_knowledge(
                self.agent, query="truncation-needle", limit=SEARCH_LIMIT
            )

        self.assertTrue(result["candidates_truncated"])
        self.assertEqual(result["candidates_scanned"], self.PATCHED_CANDIDATE_LIMIT)

        returned_ids = [row["chunk_id"] for row in result["results"]]
        self.assertNotIn(self.target_chunk.pk, returned_ids)
        # Every survivor is a content-only decoy, so the whole page scores 1.
        self.assertEqual({row["score"] for row in result["results"]}, {1})
