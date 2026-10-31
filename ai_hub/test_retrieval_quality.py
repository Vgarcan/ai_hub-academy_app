"""S-29 controlled retrieval-quality harness.  No real provider or network."""

import json
import tempfile
from pathlib import Path
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from ai_hub.models import KnowledgeChunkEmbedding
from ai_hub.services.retrieval_quality import (
    CONTROLLED_CANDIDATE,
    QUALITY_CONTRACT_VERSION,
    RetrievalQualityError,
    execute_suite,
    load_suite,
    preflight_as_dict,
    preflight_suite,
    suite_from_dict,
)
from ai_hub.test_retrieval_audit import AuditFixtureMixin, TRANSPORT_PATH


class RetrievalQualityFixture(AuditFixtureMixin):
    def build_quality(self):
        self.build_corpus()
        self.scope_a.retrieval_embedding_model_config = self.config
        self.scope_a.save(update_fields=["retrieval_embedding_model_config"])
        for key, chunk in (("alpha", self.a1), ("release", self.a2)):
            chunk.metadata = {"evaluation_key": key}
            chunk.section_title = chunk.document.title
            chunk.save(update_fields=["metadata", "section_title"])
        # section_title participates in k1; refresh the two affected vectors.
        self.index(self.a1, (0.6, 0.8, 0.0, 0.0))
        self.index(self.a2, (1.0, 0.0, 0.0, 0.0))
        self.payload = {
            "schema_version": QUALITY_CONTRACT_VERSION,
            "suite_id": "unit-controlled-v1",
            "evidence_class": CONTROLLED_CANDIDATE,
            "key_metadata_field": "evaluation_key",
            "target_collection_name": self.coll_a1.name,
            "limit": 2,
            "cutoffs": [1, 2],
            "documents": [
                {
                    "key": "alpha",
                    "title": self.a1.document.title,
                    "content": self.a1.content,
                    "tags": [],
                },
                {
                    "key": "release",
                    "title": self.a2.document.title,
                    "content": self.a2.content,
                    "tags": [],
                },
            ],
            "cases": [
                {
                    "case_id": "answerable-alpha",
                    "query_class": "answerable",
                    "query": "alpha widget",
                    "judgments": [{"document_key": "alpha", "grade": 3}],
                },
                {
                    "case_id": "no-answer-zeta",
                    "query_class": "no_answer",
                    "query": "zeta unrelated",
                    "judgments": [],
                },
            ],
        }
        self.suite = suite_from_dict(self.payload)
        return self


class SuiteValidationTests(TestCase):
    def valid_payload(self):
        return {
            "schema_version": QUALITY_CONTRACT_VERSION,
            "suite_id": "small-v1",
            "evidence_class": CONTROLLED_CANDIDATE,
            "key_metadata_field": "evaluation_key",
            "target_collection_name": "Controlled corpus",
            "limit": 5,
            "cutoffs": [1, 3, 5],
            "documents": [{"key": "doc", "title": "T", "content": "C", "tags": []}],
            "cases": [{
                "case_id": "case-1", "query_class": "answerable", "query": "Q",
                "judgments": [{"document_key": "doc", "grade": 3}],
            }],
        }

    def test_accepts_the_versioned_controlled_contract(self):
        suite = suite_from_dict(self.valid_payload())
        self.assertEqual(suite.suite_id, "small-v1")
        self.assertEqual(suite.cutoffs, (1, 3, 5))

    def test_representative_claim_is_not_accepted(self):
        payload = self.valid_payload()
        payload["evidence_class"] = "representative"
        with self.assertRaisesMessage(RetrievalQualityError, "unsupported_evidence_class"):
            suite_from_dict(payload)

    def test_unknown_fields_are_rejected(self):
        payload = self.valid_payload()
        payload["implicit_seed"] = True
        with self.assertRaisesMessage(RetrievalQualityError, "invalid_suite_fields"):
            suite_from_dict(payload)

    def test_no_answer_case_cannot_hide_a_relevant_judgment(self):
        payload = self.valid_payload()
        payload["cases"][0]["query_class"] = "no_answer"
        with self.assertRaisesMessage(RetrievalQualityError, "no_answer_case_has_judgments"):
            suite_from_dict(payload)

    def test_loader_refuses_a_file_above_its_bounded_read_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large-suite.json"
            path.write_bytes(b"{}")
            with mock.patch("ai_hub.services.retrieval_quality.MAX_SUITE_BYTES", 1):
                with self.assertRaisesMessage(RetrievalQualityError, "suite_too_large"):
                    load_suite(path)


class PreflightAndExecutionTests(RetrievalQualityFixture, TestCase):
    def setUp(self):
        self.build_quality()

    def test_preflight_proves_exact_corpus_and_current_coverage(self):
        result = preflight_suite(self.suite, self.agent_a.pk)
        payload = preflight_as_dict(result)
        self.assertTrue(payload["preflight"]["ready"])
        self.assertTrue(payload["preflight"]["corpus_matches"])
        self.assertEqual(payload["preflight"]["coverage_counts"]["current"], 2)
        self.assertFalse(payload["executed"])

    def test_preflight_refuses_a_corpus_drift(self):
        self.a1.content = "changed"
        self.a1.save(update_fields=["content"])
        with self.assertRaisesMessage(RetrievalQualityError, "corpus_does_not_match_suite"):
            preflight_suite(self.suite, self.agent_a.pk)

    def test_preflight_refuses_an_unassigned_target(self):
        self.agent_a.knowledge_collections.remove(self.coll_a1)
        with self.assertRaisesMessage(RetrievalQualityError, "target_collection_not_authorized"):
            preflight_suite(self.suite, self.agent_a.pk)

    def test_execution_refuses_incomplete_vector_coverage_before_dispatch(self):
        KnowledgeChunkEmbedding.objects.filter(chunk=self.a1).delete()
        preflight = preflight_suite(self.suite, self.agent_a.pk)
        transport = self.transport_returning()
        with mock.patch(TRANSPORT_PATH, return_value=transport):
            with self.assertRaisesMessage(RetrievalQualityError, "preflight_not_ready"):
                execute_suite(preflight)
        self.assertEqual(transport.calls, [])

    def test_execution_is_content_free_and_uses_existing_metrics(self):
        preflight = preflight_suite(self.suite, self.agent_a.name)
        transport = self.transport_returning()
        with mock.patch(TRANSPORT_PATH, return_value=transport):
            payload = execute_suite(preflight)
        encoded = json.dumps(payload)
        self.assertTrue(payload["executed"])
        self.assertEqual(payload["metrics"]["lexical"]["top1_count"], 1)
        self.assertEqual(len(transport.calls), 2)
        self.assertNotIn("alpha widget", encoded)
        self.assertNotIn(self.a1.content, encoded)
        self.assertNotIn('"query":', encoded)

        no_answer = next(
            case for case in payload["cases"] if case["query_class"] == "no_answer"
        )
        for path in ("lexical", "hybrid"):
            result = no_answer[path]
            self.assertNotIn("mrr", result)
            self.assertNotIn("first_relevant_rank", result)
            self.assertNotIn("recall", result["no_answer"])
            self.assertEqual(
                result["returned_count"], result["no_answer"]["false_positive_count"]
            )
            self.assertIn("false_positive_count", result["no_answer"]["cutoffs"]["1"])
        self.assertIn(
            "no_answer_false_positive_at_k_count",
            payload["metrics"]["hybrid"],
        )

    def test_execution_refuses_drift_before_any_provider_dispatch(self):
        preflight = preflight_suite(self.suite, self.agent_a.pk)
        self.a1.content = "changed after preflight"
        self.a1.save(update_fields=["content"])
        transport = self.transport_returning()
        with mock.patch(TRANSPORT_PATH, return_value=transport):
            with self.assertRaisesMessage(RetrievalQualityError, "suite_snapshot_changed"):
                execute_suite(preflight)
        self.assertEqual(transport.calls, [])

    def test_execution_refuses_drift_after_cases_instead_of_emitting_an_aggregate(self):
        preflight = preflight_suite(self.suite, self.agent_a.pk)
        transport = self.transport_returning()
        original_hybrid = __import__(
            "ai_hub.services.retrieval_quality", fromlist=["search_hybrid_with_scope"]
        ).search_hybrid_with_scope

        def mutate_config_after_search(*args, **kwargs):
            result = original_hybrid(*args, **kwargs)
            self.config.request_timeout_seconds += 1
            self.config.save(update_fields=["request_timeout_seconds"])
            return result

        with mock.patch(TRANSPORT_PATH, return_value=transport):
            with mock.patch(
                "ai_hub.services.retrieval_quality.search_hybrid_with_scope",
                side_effect=mutate_config_after_search,
            ):
                with self.assertRaisesMessage(RetrievalQualityError, "suite_snapshot_changed"):
                    execute_suite(preflight)

    def test_command_is_report_only_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text(json.dumps(self.payload), encoding="utf-8")
            output = __import__("io").StringIO()
            with mock.patch(TRANSPORT_PATH) as transport:
                call_command(
                    "knowledge_retrieval_evaluate",
                    suite=str(path), agent=str(self.agent_a.pk), stdout=output,
                )
        self.assertIn("Report only: no provider was called", output.getvalue())
        transport.assert_not_called()

    def test_command_fails_when_report_is_not_ready(self):
        KnowledgeChunkEmbedding.objects.filter(chunk=self.a1).delete()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text(json.dumps(self.payload), encoding="utf-8")
            with self.assertRaisesMessage(CommandError, "preflight_not_ready"):
                call_command(
                    "knowledge_retrieval_evaluate",
                    suite=str(path), agent=str(self.agent_a.pk), stdout=__import__("io").StringIO(),
                )
