"""Tests for the read-only Knowledge preflight (Slice 5).

Covers the classification model, issue codes, scoping, bounds, the operator
security boundary, and — most importantly — the READ-ONLY invariant.
"""
import json
from io import StringIO

from django.core.management import call_command
from django.test import TestCase

from ai_hub.models import (
    AgentProfile,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    ProviderConfig,
    ToolDefinition,
)
from ai_hub.services.knowledge_preflight import (
    CLASSIFICATIONS,
    EMPTY_ACTIVE,
    ISSUE_CODES,
    KP001_SOURCE_WITHOUT_CHUNKS,
    KP002_EMPTY_ACTIVE,
    KP003_UNUSABLE_CHUNK_CONTENT,
    KP004_UNKNOWN_AUTHORITY,
    KP005_SOURCE_FILE_WITHOUT_CHUNKS,
    KP006_CHUNK_INDEX_ANOMALY,
    NON_ACTIVE,
    READY_CANONICAL,
    SEVERITY_INFO,
    SOURCE_WITHOUT_CHUNKS,
    UNKNOWN_LEGACY,
    UNUSABLE_CHUNKS,
    run_knowledge_preflight,
    summarize_preflight,
)


def make_document(collection, title, *, curated_text="", status=None, chunks=(), source_file=""):
    document = KnowledgeDocument.objects.create(
        collection=collection,
        title=title,
        curated_text=curated_text,
        source_file=source_file,
        status=status or KnowledgeDocument.Status.ACTIVE,
    )
    for index, content in chunks:
        KnowledgeDocumentChunk.objects.create(
            document=document,
            chunk_index=index,
            section_title=f"Section {index}",
            content=content,
        )
    return document


def document_row(report, document_id):
    for row in report["documents"]:
        if row["document_id"] == document_id:
            return row
    raise AssertionError(f"document {document_id} missing from report")


class PreflightClassificationTests(TestCase):
    """One document per shape; classification asserted for each."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(name="Preflight Primary")
        cls.ready = make_document(
            cls.collection, "Ready Canonical",
            chunks=((1, "canonical chunk body"),),
        )
        cls.unknown = make_document(
            cls.collection, "Unknown Legacy",
            curated_text="source body",
            chunks=((1, "chunk body that may or may not be derived"),),
        )
        cls.source_only = make_document(
            cls.collection, "Source Without Chunks", curated_text="orphan source body",
        )
        cls.empty_active = make_document(cls.collection, "Empty Active")
        cls.unusable = make_document(
            cls.collection, "Unusable Chunks", chunks=((1, ""), (2, "   \n\t  ")),
        )
        cls.draft = make_document(
            cls.collection, "Draft Source", curated_text="draft body",
            status=KnowledgeDocument.Status.DRAFT,
        )
        cls.archived = make_document(
            cls.collection, "Archived Source", curated_text="archived body",
            status=KnowledgeDocument.Status.ARCHIVED,
        )

    def test_each_shape_gets_its_expected_classification(self):
        report = run_knowledge_preflight()
        expected = {
            self.ready.pk: READY_CANONICAL,
            self.unknown.pk: UNKNOWN_LEGACY,
            self.source_only.pk: SOURCE_WITHOUT_CHUNKS,
            self.empty_active.pk: EMPTY_ACTIVE,
            self.unusable.pk: UNUSABLE_CHUNKS,
            self.draft.pk: NON_ACTIVE,
            self.archived.pk: NON_ACTIVE,
        }
        for document_id, classification in expected.items():
            with self.subTest(document=document_id):
                self.assertEqual(
                    document_row(report, document_id)["classification"], classification
                )

    def test_classification_is_exactly_one_per_document(self):
        report = run_knowledge_preflight()
        self.assertEqual(
            sum(report["summary"]["by_classification"].values()),
            report["summary"]["documents_in_scope"],
        )
        for row in report["documents"]:
            self.assertIn(row["classification"], CLASSIFICATIONS)

    def test_non_active_documents_without_chunks_are_not_a_repair_problem(self):
        """DRAFT/ARCHIVED are excluded from retrieval by design."""
        report = run_knowledge_preflight()
        for document in (self.draft, self.archived):
            with self.subTest(document=document.title):
                row = document_row(report, document.pk)
                self.assertEqual(row["classification"], NON_ACTIVE)
                self.assertEqual(row["issues"], [])
                self.assertFalse(row["canonically_retrievable"])

    def test_retrievability_is_reported_per_document(self):
        report = run_knowledge_preflight()
        self.assertTrue(document_row(report, self.ready.pk)["canonically_retrievable"])
        self.assertTrue(document_row(report, self.unknown.pk)["canonically_retrievable"])
        for document in (self.source_only, self.empty_active, self.unusable):
            with self.subTest(document=document.title):
                self.assertFalse(
                    document_row(report, document.pk)["canonically_retrievable"]
                )

    def test_summary_counts_are_consistent(self):
        report = run_knowledge_preflight()
        summary = report["summary"]
        self.assertEqual(summary["documents_in_scope"], 7)
        self.assertEqual(summary["documents_total"], 7)
        self.assertEqual(summary["by_status"]["active"], 5)
        self.assertEqual(summary["by_status"]["draft"], 1)
        self.assertEqual(summary["by_status"]["archived"], 1)
        self.assertEqual(summary["active_documents"], 5)
        self.assertEqual(summary["active_canonically_retrievable"], 2)
        self.assertEqual(summary["active_not_retrievable"], 3)
        self.assertEqual(summary["chunks_total"], 4)
        self.assertEqual(summary["chunks_usable"], 2)
        self.assertEqual(summary["chunks_unusable"], 2)

    def test_convergence_facts_are_reported_without_a_policy_verdict(self):
        """Facts only. No `canonical_transition_safe` boolean is emitted."""
        report = run_knowledge_preflight()
        convergence = report["convergence"]
        self.assertEqual(convergence["active_source_without_chunks"], 1)
        self.assertEqual(convergence["active_unusable_chunks"], 1)
        self.assertEqual(convergence["active_empty"], 1)
        self.assertEqual(convergence["active_unknown_authority"], 1)
        self.assertNotIn("canonical_transition_safe", convergence)
        self.assertNotIn("safe", json.dumps(convergence))


class PreflightIssueCodeTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(name="Preflight Issues")

    def test_source_without_chunks_raises_kp001(self):
        document = make_document(self.collection, "Src", curated_text="body")
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertIn(KP001_SOURCE_WITHOUT_CHUNKS, row["issues"])

    def test_empty_active_raises_kp002(self):
        document = make_document(self.collection, "Empty")
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertIn(KP002_EMPTY_ACTIVE, row["issues"])

    def test_unusable_chunk_content_raises_kp003(self):
        document = make_document(
            self.collection, "Partly Unusable",
            chunks=((1, "good body"), (2, "   ")),
        )
        report = run_knowledge_preflight()
        row = document_row(report, document.pk)
        self.assertIn(KP003_UNUSABLE_CHUNK_CONTENT, row["issues"])
        # Still retrievable: one usable chunk remains.
        self.assertEqual(row["classification"], READY_CANONICAL)
        self.assertTrue(row["canonically_retrievable"])
        detail = next(
            issue["detail"] for issue in report["issues"]
            if issue["document_id"] == document.pk
            and issue["code"] == KP003_UNUSABLE_CHUNK_CONTENT
        )
        self.assertEqual(detail["unusable_chunk_count"], 1)

    def test_unknown_authority_raises_kp004_and_is_informational_only(self):
        """KP004 must never imply a defect, drift or staleness."""
        document = make_document(
            self.collection, "Both", curated_text="source", chunks=((1, "chunk"),),
        )
        report = run_knowledge_preflight()
        row = document_row(report, document.pk)
        self.assertIn(KP004_UNKNOWN_AUTHORITY, row["issues"])
        self.assertEqual(ISSUE_CODES[KP004_UNKNOWN_AUTHORITY]["severity"], SEVERITY_INFO)
        meaning = ISSUE_CODES[KP004_UNKNOWN_AUTHORITY]["meaning"].lower()
        self.assertIn("not a defect", meaning)
        for forbidden in ("stale", "drift", "divergen"):
            self.assertNotIn(forbidden, row["classification"].lower())

    def test_source_file_without_chunks_raises_kp005(self):
        document = make_document(
            self.collection, "Uploaded", source_file="ai_hub/knowledge/manual.txt",
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertIn(KP005_SOURCE_FILE_WITHOUT_CHUNKS, row["issues"])
        self.assertEqual(row["classification"], SOURCE_WITHOUT_CHUNKS)
        self.assertTrue(row["has_source_file"])

    def test_chunk_index_anomalies_raise_kp006(self):
        gapped = make_document(
            self.collection, "Gapped", chunks=((1, "a"), (3, "b")),
        )
        late_start = make_document(
            self.collection, "Late Start", chunks=((5, "a"), (6, "b")),
        )
        zero_index = make_document(
            self.collection, "Zero Index", chunks=((0, "a"), (1, "b")),
        )
        report = run_knowledge_preflight()

        self.assertIn(KP006_CHUNK_INDEX_ANOMALY, document_row(report, gapped.pk)["issues"])
        self.assertIn(
            KP006_CHUNK_INDEX_ANOMALY, document_row(report, late_start.pk)["issues"]
        )
        self.assertIn(
            KP006_CHUNK_INDEX_ANOMALY, document_row(report, zero_index.pk)["issues"]
        )

        anomalies = {
            issue["document_id"]: issue["detail"]["anomalies"]
            for issue in report["issues"]
            if issue["code"] == KP006_CHUNK_INDEX_ANOMALY
        }
        self.assertIn("has_gaps", anomalies[gapped.pk])
        self.assertIn("does_not_start_at_one", anomalies[late_start.pk])
        self.assertIn("contains_index_zero", anomalies[zero_index.pk])

    def test_well_formed_multi_chunk_document_raises_no_anomaly(self):
        document = make_document(
            self.collection, "Well Formed",
            chunks=((1, "a"), (2, "b"), (3, "c")),
        )
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertNotIn(KP006_CHUNK_INDEX_ANOMALY, row["issues"])
        self.assertEqual(row["classification"], READY_CANONICAL)
        self.assertEqual(row["usable_chunk_count"], 3)

    def test_every_emitted_code_is_documented(self):
        make_document(self.collection, "Src", curated_text="body")
        make_document(self.collection, "Empty")
        report = run_knowledge_preflight()
        for issue in report["issues"]:
            with self.subTest(code=issue["code"]):
                self.assertIn(issue["code"], ISSUE_CODES)
                self.assertEqual(
                    issue["severity"], ISSUE_CODES[issue["code"]]["severity"]
                )


class PreflightProvenanceTests(TestCase):
    """Historical markers are reported as ORIGIN, never as current authority."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(name="Preflight Provenance")

    def _document_with_marker(self, title, marker):
        document = make_document(self.collection, title, curated_text="source")
        KnowledgeDocumentChunk.objects.create(
            document=document,
            chunk_index=1,
            section_title=title,
            content="body",
            metadata={"ingestion": marker} if marker else {},
        )
        return document

    def test_known_ingestion_markers_are_reported(self):
        runtime = self._document_with_marker("Runtime", "initial_curated_text")
        backfill = self._document_with_marker("Backfill", "initial_curated_text_backfill")
        report = run_knowledge_preflight()

        runtime_row = document_row(report, runtime.pk)
        self.assertTrue(runtime_row["provenance"]["known_derivation_origin"])
        self.assertEqual(
            runtime_row["provenance"]["ingestion_markers"], {"initial_curated_text": 1}
        )

        backfill_row = document_row(report, backfill.pk)
        self.assertEqual(
            backfill_row["provenance"]["ingestion_markers"],
            {"initial_curated_text_backfill": 1},
        )

    def test_a_known_marker_does_not_change_the_classification(self):
        """Origin evidence must not be promoted to current authority.

        Slice 4 measured that the marker is creation-time only, so a marked
        document is still UNKNOWN_LEGACY, not "derived".
        """
        marked = self._document_with_marker("Marked", "initial_curated_text")
        unmarked = self._document_with_marker("Unmarked", None)
        report = run_knowledge_preflight()
        self.assertEqual(
            document_row(report, marked.pk)["classification"], UNKNOWN_LEGACY
        )
        self.assertEqual(
            document_row(report, unmarked.pk)["classification"], UNKNOWN_LEGACY
        )

    def test_chunks_without_metadata_are_counted(self):
        document = self._document_with_marker("Unmarked", None)
        row = document_row(run_knowledge_preflight(), document.pk)
        self.assertFalse(row["provenance"]["known_derivation_origin"])
        self.assertEqual(row["provenance"]["chunks_without_metadata"], 1)


class PreflightScopeAndBoundsTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.first = KnowledgeCollection.objects.create(name="Alpha Collection")
        cls.second = KnowledgeCollection.objects.create(name="Beta Collection")
        cls.inactive = KnowledgeCollection.objects.create(
            name="Gamma Collection", is_active=False
        )
        make_document(cls.first, "Alpha Doc", curated_text="a")
        make_document(cls.second, "Beta Doc", curated_text="b")
        make_document(
            cls.inactive, "Gamma Doc", chunks=((1, "gamma body"),),
        )

    def test_collection_scoping(self):
        report = run_knowledge_preflight(collection_ids=[self.first.pk])
        self.assertEqual(report["summary"]["documents_in_scope"], 1)
        self.assertEqual(report["summary"]["documents_total"], 3)
        self.assertEqual(report["documents"][0]["collection"], "Alpha Collection")
        self.assertEqual(report["scope"]["collection_ids"], [self.first.pk])

    def test_status_scoping(self):
        make_document(
            self.first, "Alpha Draft", curated_text="d",
            status=KnowledgeDocument.Status.DRAFT,
        )
        report = run_knowledge_preflight(statuses=[KnowledgeDocument.Status.DRAFT])
        self.assertEqual(report["summary"]["documents_in_scope"], 1)
        self.assertEqual(report["documents"][0]["classification"], NON_ACTIVE)

    def test_collection_grouping(self):
        report = run_knowledge_preflight()
        names = [row["name"] for row in report["collections"]]
        self.assertEqual(len(names), 3)
        self.assertEqual(set(names), {"Alpha Collection", "Beta Collection", "Gamma Collection"})
        for row in report["collections"]:
            with self.subTest(collection=row["name"]):
                self.assertEqual(row["documents"], 1)

    def test_a_document_in_an_inactive_collection_is_not_retrievable(self):
        """Collection state is part of retrievability, not just document status."""
        report = run_knowledge_preflight(collection_ids=[self.inactive.pk])
        row = report["documents"][0]
        self.assertEqual(row["classification"], READY_CANONICAL)
        self.assertFalse(row["collection_is_active"])
        self.assertFalse(row["canonically_retrievable"])

    def test_detail_list_is_bounded_but_summary_is_not(self):
        report = run_knowledge_preflight(document_limit=1)
        self.assertEqual(len(report["documents"]), 1)
        self.assertTrue(report["scope"]["documents_truncated"])
        self.assertEqual(report["summary"]["documents_in_scope"], 3)
        self.assertEqual(sum(report["summary"]["by_classification"].values()), 3)

    def test_document_limit_is_clamped(self):
        self.assertEqual(
            run_knowledge_preflight(document_limit=0)["scope"]["document_limit"], 1
        )
        self.assertGreater(
            run_knowledge_preflight(document_limit=10**9)["scope"]["document_limit"], 0
        )
        self.assertEqual(
            run_knowledge_preflight(document_limit="nonsense")["scope"]["document_limit"],
            run_knowledge_preflight()["scope"]["document_limit"],
        )

    def test_report_contains_no_chunk_bodies(self):
        """Bounded output: full chunk text must never reach the report."""
        make_document(
            self.first, "Long Body",
            chunks=((1, "unmistakable-chunk-body-marker " * 200),),
        )
        report = run_knowledge_preflight()
        self.assertNotIn("unmistakable-chunk-body-marker", json.dumps(report))

    def test_empty_corpus_produces_a_valid_report(self):
        KnowledgeDocument.objects.all().delete()
        report = run_knowledge_preflight()
        self.assertEqual(report["summary"]["documents_in_scope"], 0)
        self.assertEqual(report["documents"], [])
        self.assertEqual(report["issues"], [])
        self.assertEqual(sum(report["summary"]["by_classification"].values()), 0)


class PreflightReadOnlyInvariantTests(TestCase):
    """THE core invariant: preflight changes nothing, ever."""

    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(name="Preflight ReadOnly")
        make_document(cls.collection, "Ready", chunks=((1, "body"),))
        make_document(cls.collection, "Source Only", curated_text="orphan source")
        make_document(cls.collection, "Empty")
        make_document(cls.collection, "Unusable", chunks=((1, "   "),))
        make_document(
            cls.collection, "Both", curated_text="source", chunks=((1, "chunk"),),
        )
        make_document(
            cls.collection, "Draft", curated_text="d",
            status=KnowledgeDocument.Status.DRAFT,
        )

    @staticmethod
    def _snapshot():
        return {
            "documents": list(
                KnowledgeDocument.objects.order_by("pk").values(
                    "pk", "collection_id", "title", "curated_text", "source_file",
                    "tags", "language", "status", "notes", "created_at", "updated_at",
                )
            ),
            "chunks": list(
                KnowledgeDocumentChunk.objects.order_by("pk").values(
                    "pk", "document_id", "chunk_index", "section_title", "content",
                    "token_estimate", "metadata", "created_at", "updated_at",
                )
            ),
            "collections": list(
                KnowledgeCollection.objects.order_by("pk").values(
                    "pk", "name", "description", "is_active", "created_at", "updated_at",
                )
            ),
        }

    def test_preflight_does_not_modify_any_row(self):
        before = self._snapshot()
        run_knowledge_preflight()
        run_knowledge_preflight(collection_ids=[self.collection.pk])
        run_knowledge_preflight(statuses=[KnowledgeDocument.Status.ACTIVE])
        after = self._snapshot()
        self.assertEqual(before, after)

    def test_row_counts_are_unchanged(self):
        counts = (
            KnowledgeCollection.objects.count(),
            KnowledgeDocument.objects.count(),
            KnowledgeDocumentChunk.objects.count(),
        )
        run_knowledge_preflight()
        self.assertEqual(
            counts,
            (
                KnowledgeCollection.objects.count(),
                KnowledgeDocument.objects.count(),
                KnowledgeDocumentChunk.objects.count(),
            ),
        )

    def test_timestamps_and_metadata_are_untouched(self):
        chunk = KnowledgeDocumentChunk.objects.first()
        document = chunk.document
        before = (
            chunk.updated_at, chunk.created_at, chunk.metadata,
            document.updated_at, document.created_at,
        )
        run_knowledge_preflight()
        chunk.refresh_from_db()
        document.refresh_from_db()
        self.assertEqual(
            before,
            (
                chunk.updated_at, chunk.created_at, chunk.metadata,
                document.updated_at, document.created_at,
            ),
        )

    def test_repeated_runs_return_an_identical_report(self):
        """No timestamp, no randomness, no ordering ambiguity."""
        first = run_knowledge_preflight()
        second = run_knowledge_preflight()
        self.assertEqual(first, second)

    def test_the_report_carries_no_timestamp(self):
        """A timestamp would break run-to-run comparability by construction."""
        serialized = json.dumps(run_knowledge_preflight(), default=str)
        for forbidden in ("generated_at", "timestamp", "run_at"):
            self.assertNotIn(forbidden, serialized)

    def test_the_service_exposes_no_write_capability(self):
        """No repair/fix/backfill entry point exists in the module."""
        from ai_hub.services import knowledge_preflight

        for forbidden in ("repair", "fix", "backfill", "regenerate", "apply", "write"):
            matches = [
                name for name in dir(knowledge_preflight)
                if forbidden in name.lower() and not name.startswith("__")
            ]
            with self.subTest(term=forbidden):
                self.assertEqual(matches, [])


class PreflightSecurityBoundaryTests(TestCase):
    """Operator diagnostic, not an Agent-facing Knowledge tool."""

    def test_preflight_is_not_registered_as_a_tool_definition(self):
        self.assertFalse(
            ToolDefinition.objects.filter(
                config__callable__icontains="knowledge_preflight"
            ).exists()
        )
        self.assertFalse(
            ToolDefinition.objects.filter(name__icontains="preflight").exists()
        )

    def test_preflight_is_not_one_of_the_canonical_knowledge_tools(self):
        from ai_hub.services.knowledge_tooling import KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES

        for callable_path in KNOWLEDGE_RETRIEVAL_TOOL_CALLABLES.values():
            self.assertNotIn("preflight", callable_path)

    def test_preflight_is_not_auto_resolved_into_an_agent_manifest(self):
        from ai_hub.services.tool_resolution import resolve_agent_tools

        provider = ProviderConfig.objects.create(
            name="preflight-provider",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        agent = AgentProfile.objects.create(
            name="preflight-agent", role="Boundary", model_config=model
        )
        collection = KnowledgeCollection.objects.create(name="Preflight Boundary")
        agent.knowledge_collections.add(collection)

        names = resolve_agent_tools(agent).tool_names()
        self.assertTrue(all("preflight" not in name for name in names))

    def test_preflight_takes_no_agent_argument(self):
        """Operator scope and retrieval authorization are separate concerns."""
        import inspect

        signature = inspect.signature(run_knowledge_preflight)
        self.assertNotIn("agent", signature.parameters)
        self.assertEqual(
            set(signature.parameters), {"collection_ids", "statuses", "document_limit"}
        )


class PreflightCommandTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.collection = KnowledgeCollection.objects.create(name="Preflight Command")
        make_document(cls.collection, "Ready", chunks=((1, "body"),))
        make_document(cls.collection, "Source Only", curated_text="orphan source")

    def _run(self, *args):
        out = StringIO()
        call_command("knowledge_preflight", *args, stdout=out)
        return out.getvalue()

    def test_command_reports_and_changes_nothing(self):
        before = PreflightReadOnlyInvariantTests._snapshot()
        output = self._run()
        self.assertIn("Documents in scope: 2", output)
        self.assertIn(READY_CANONICAL, output)
        self.assertIn("read-only", output)
        self.assertEqual(before, PreflightReadOnlyInvariantTests._snapshot())

    def test_command_emits_valid_json(self):
        payload = json.loads(self._run("--json"))
        self.assertEqual(payload["summary"]["documents_in_scope"], 2)
        self.assertIn("convergence", payload)

    def test_command_supports_scoping_flags(self):
        output = self._run("--status", "active", "--collection", str(self.collection.pk))
        self.assertIn("Documents in scope: 2", output)

    def test_command_reports_truncation(self):
        output = self._run("--limit", "1")
        self.assertIn("truncated", output)

    def test_command_has_no_write_flags(self):
        """Read-only by construction: no --fix, --write or --repair exists."""
        from django.core.management.base import CommandError

        for flag in ("--fix", "--write", "--repair", "--backfill", "--apply"):
            with self.subTest(flag=flag):
                with self.assertRaisesMessage(CommandError, "unrecognized arguments"):
                    call_command(
                        "knowledge_preflight", flag, stdout=StringIO(), stderr=StringIO()
                    )


class PreflightQueryEfficiencyTests(TestCase):
    """Query count must not grow with the corpus."""

    EXPECTED_QUERIES = 3  # documents + chunk facts + total document count

    def _build(self, document_count):
        collection = KnowledgeCollection.objects.create(
            name=f"Preflight Scale {document_count}"
        )
        KnowledgeDocument.objects.bulk_create(
            [
                KnowledgeDocument(
                    collection=collection,
                    title=f"Doc {index:04d}",
                    curated_text="source body",
                    status=KnowledgeDocument.Status.ACTIVE,
                )
                for index in range(document_count)
            ]
        )
        KnowledgeDocumentChunk.objects.bulk_create(
            [
                KnowledgeDocumentChunk(
                    document=document,
                    chunk_index=1,
                    section_title="Section",
                    content="chunk body",
                )
                for document in KnowledgeDocument.objects.filter(collection=collection)
            ]
        )

    def test_query_count_is_constant_for_a_small_corpus(self):
        self._build(5)
        with self.assertNumQueries(self.EXPECTED_QUERIES):
            run_knowledge_preflight()

    def test_query_count_is_unchanged_for_a_larger_corpus(self):
        """The same fixed number of queries — no N+1 per document or chunk."""
        self._build(120)
        with self.assertNumQueries(self.EXPECTED_QUERIES):
            report = run_knowledge_preflight()
        self.assertEqual(report["summary"]["documents_in_scope"], 120)
        self.assertEqual(report["summary"]["chunks_total"], 120)

    def test_truncating_the_detail_list_does_not_change_query_count(self):
        self._build(120)
        with self.assertNumQueries(self.EXPECTED_QUERIES):
            report = run_knowledge_preflight(document_limit=10)
        self.assertEqual(len(report["documents"]), 10)
        self.assertEqual(report["summary"]["documents_in_scope"], 120)


class PreflightSummaryRenderingTests(TestCase):
    def test_summary_lists_every_classification(self):
        collection = KnowledgeCollection.objects.create(name="Preflight Render")
        make_document(collection, "Ready", chunks=((1, "body"),))
        lines = summarize_preflight(run_knowledge_preflight())
        rendered = "\n".join(lines)
        for name in CLASSIFICATIONS:
            self.assertIn(name, rendered)

    def test_summary_reports_no_issues_when_the_corpus_is_clean(self):
        collection = KnowledgeCollection.objects.create(name="Preflight Clean")
        make_document(collection, "Ready", chunks=((1, "body"),))
        rendered = "\n".join(summarize_preflight(run_knowledge_preflight()))
        self.assertIn("none", rendered)
