"""S-28 — Agent-facing hybrid Knowledge search.

Every provider call is mocked. **No real Ollama server, no network.**

What this file proves, in the order a reviewer should care:

    off by default, and gated on EVERY path a tool can reach an Agent by
    the runtime identity always wins over a model-supplied one
    ONE authorization answer for search and hydration
    a chunk changed after ranking is withheld, never served or dropped
    every call is audited; refusals too
    nothing about configuration reaches the model
    coverage tells the truth about a partly indexed corpus
    the six existing Knowledge tools are unchanged
"""

import ast
import inspect
import json
from unittest import mock

from django.core.exceptions import ValidationError
from django.db.models import ProtectedError
from django.test import TestCase, override_settings

from ai_hub.models import (
    AgentToolGrant,
    ExecutionSession,
    KnowledgeChunkEmbedding,
    RetrievalOutcome,
    RetrievalRun,
    ToolboxTool,
    ToolDefinition,
)
from ai_hub.services import hybrid_knowledge_search, knowledge_retrieval
from ai_hub.services.embedding_client import (
    EmbeddingProviderExecutionError,
    ErrorCategory,
)
from ai_hub.services.hybrid_knowledge_search import (
    CHANGED_SINCE_SEARCH,
    COLLECTION_NOT_ACCESSIBLE_MESSAGE,
    NOT_AVAILABLE_MESSAGE,
    TEMPORARILY_UNAVAILABLE_MESSAGE,
    hybrid_search_configuration,
    hybrid_search_offered,
    search_knowledge_hybrid,
)
from ai_hub.services.hybrid_retrieval import HybridMode, HybridRetrievalError
from ai_hub.services.knowledge_tooling import (
    HYBRID_KNOWLEDGE_SEARCH_TOOL_NAME,
    KNOWLEDGE_RETRIEVAL_TOOL_NAMES,
)
from ai_hub.services.tool_resolution import resolve_agent_tools
from ai_hub.services.tools_runtime import execute_tool
from ai_hub.test_retrieval_audit import (
    FOREIGN_SECRET,
    KNOWLEDGE_SECRET,
    PROVIDER_SECRET,
    TRANSPORT_PATH,
    AuditFixtureMixin,
)

HYBRID = HYBRID_KNOWLEDGE_SEARCH_TOOL_NAME
PATH1_TOOLS = tuple(name for name in KNOWLEDGE_RETRIEVAL_TOOL_NAMES if name != HYBRID)
ON = override_settings(AI_HUB_HYBRID_KNOWLEDGE_SEARCH_ENABLED=True)
OFF = override_settings(AI_HUB_HYBRID_KNOWLEDGE_SEARCH_ENABLED=False)

QUERY = "alpha widget"

#: The complete set of keys the model may see. Anything else is a leak.
TOP_LEVEL_KEYS = {"results", "total", "mode", "degraded", "coverage"}
HIT_KEYS = {
    "chunk_id", "document_id", "title", "collection", "section_title",
    "chunk_index", "excerpt", "excerpt_truncated", "matched_by", "citation",
}
WITHHELD_KEYS = {"chunk_id", "unavailable"}
COVERAGE_KEYS = {"indexed_chunks", "searchable_chunks"}


class HybridFixtureMixin(AuditFixtureMixin):
    def build(self, *, configure=True):
        self.build_corpus()
        if configure:
            self.scope_a.retrieval_embedding_model_config = self.config
            self.scope_a.save()
        self.agent_a.refresh_from_db()
        self.tool = ToolDefinition.objects.get(name=HYBRID)
        return self

    def search(self, transport=None, *, agent=None, **kwargs):
        self.transport = transport or self.transport_returning()
        kwargs.setdefault("query", QUERY)
        with mock.patch(TRANSPORT_PATH, return_value=self.transport):
            return search_knowledge_hybrid(agent or self.agent_a, **kwargs)

    def names(self, agent=None, **kwargs):
        return set(resolve_agent_tools(agent or self.agent_a, **kwargs).tool_names())


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

class GateTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build()

    @OFF
    def test_off_by_default_hides_the_tool_and_keeps_the_six(self):
        names = self.names()
        self.assertNotIn(HYBRID, names)
        self.assertTrue(set(PATH1_TOOLS) <= names)
        self.assertIsNone(hybrid_search_configuration(self.agent_a))

    @ON
    def test_on_with_a_scope_configuration_offers_it(self):
        resolution = resolve_agent_tools(self.agent_a)
        sources = {resolved.tool.name: resolved.source for resolved in resolution.tools}
        self.assertEqual(sources.get(HYBRID), "knowledge_retrieval")
        self.assertEqual(hybrid_search_configuration(self.agent_a), self.config)

    @ON
    def test_a_scope_without_configuration_is_not_offered(self):
        self.assertNotIn(HYBRID, self.names(self.agent_b))

    @ON
    def test_inactive_configuration_or_provider_is_not_offered(self):
        for target in (self.config, self.embed_provider):
            with self.subTest(target=type(target).__name__):
                target.is_active = False
                target.save()
                self.agent_a.refresh_from_db()
                self.assertNotIn(HYBRID, self.names())
                target.is_active = True
                target.save()
                self.agent_a.refresh_from_db()

    @ON
    def test_never_offered_in_a_game_context(self):
        game = mock.Mock(runtime_kind=ExecutionSession.RuntimeKind.GAME)
        orchestrator = mock.Mock(runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR)
        self.assertNotIn(HYBRID, self.names(execution_context={"session": game}))
        self.assertIn(HYBRID, self.names(execution_context={"session": orchestrator}))
        self.assertFalse(
            hybrid_search_offered(self.agent_a, workspace=mock.Mock(default_policy={}))
        )

    @OFF
    def test_no_other_assignment_path_can_reintroduce_it(self):
        AgentToolGrant.objects.create(agent=self.agent_a, tool=self.tool)
        self.agent_a.tools.add(self.tool)
        self.assertNotIn(HYBRID, self.names())

    @OFF
    def test_direct_execution_refuses_when_off_and_records_nothing(self):
        with self.assertRaisesMessage(ValidationError, NOT_AVAILABLE_MESSAGE):
            execute_tool(self.tool, {"query": QUERY}, agent=self.agent_a)
        self.assertEqual(RetrievalRun.objects.count(), 0)

    @ON
    def test_direct_execution_refuses_without_scope_configuration(self):
        with self.assertRaisesMessage(ValidationError, NOT_AVAILABLE_MESSAGE):
            execute_tool(self.tool, {"query": QUERY}, agent=self.agent_b)
        self.assertEqual(RetrievalRun.objects.count(), 0)


# ---------------------------------------------------------------------------
# Identity, authorization, audit
# ---------------------------------------------------------------------------

@ON
class AuthorizationAndAuditTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build()

    def test_a_model_supplied_identity_never_takes_effect(self):
        with mock.patch(TRANSPORT_PATH, return_value=self.transport_returning()):
            output = execute_tool(
                self.tool,
                {"query": QUERY, "agent_id": self.agent_b.pk, "agent_name": self.agent_b.name},
                agent=self.agent_a,
            )
        run = RetrievalRun.objects.get()
        self.assertEqual(run.agent_id_snapshot, self.agent_a.pk)
        self.assertNotIn(FOREIGN_SECRET, json.dumps(output))

    def test_execution_requires_a_runtime_agent(self):
        with self.assertRaises(ValidationError):
            execute_tool(self.tool, {"query": QUERY}, agent=None)

    def test_the_scope_is_resolved_exactly_once_per_call(self):
        real = hybrid_knowledge_search.resolve_effective_knowledge_scope
        with mock.patch.object(
            hybrid_knowledge_search, "resolve_effective_knowledge_scope", wraps=real
        ) as here, mock.patch(
            "ai_hub.services.retrieval_audit.resolve_effective_knowledge_scope"
        ) as inside_audit:
            self.search()
        self.assertEqual(here.call_count, 1)
        inside_audit.assert_not_called()

    def test_only_assigned_same_scope_collections_are_searched(self):
        # A deliberate cross-scope assignment row grants nothing (S-15).
        self.agent_a.knowledge_collections.add(self.coll_b1)
        output = self.search()
        ids = {hit["chunk_id"] for hit in output["results"]}
        self.assertTrue(ids)
        self.assertTrue(ids <= {self.a1.pk, self.a2.pk, self.a3.pk})
        self.assertNotIn(FOREIGN_SECRET, json.dumps(output))

    def test_every_call_leaves_one_run_and_one_outcome(self):
        self.search()
        self.search(query="widget")
        self.assertEqual(RetrievalRun.objects.count(), 2)
        self.assertEqual(RetrievalOutcome.objects.count(), 2)

    def test_a_refused_search_is_audited_and_told_uniformly(self):
        with mock.patch(
            "ai_hub.services.retrieval_audit.search_hybrid_with_scope",
            side_effect=HybridRetrievalError("no_complete_retrieval_branch"),
        ):
            with self.assertRaisesMessage(ValidationError, TEMPORARILY_UNAVAILABLE_MESSAGE):
                self.search()
        outcome = RetrievalOutcome.objects.get()
        self.assertEqual(outcome.outcome, RetrievalOutcome.Outcome.REFUSED)
        self.assertEqual(outcome.failure_category, "no_complete_retrieval_branch")

    def test_inaccessible_collections_refuse_exactly_like_path_1(self):
        for collection in (self.coll_a3, self.coll_b1):
            with self.subTest(collection=collection.name):
                with self.assertRaises(ValidationError) as path1:
                    knowledge_retrieval.search_knowledge(
                        self.agent_a, query=QUERY, collection_id=collection.pk
                    )
                with self.assertRaises(ValidationError) as hybrid:
                    self.search(collection_id=collection.pk)
                self.assertEqual(hybrid.exception.messages, path1.exception.messages)
                self.assertEqual(hybrid.exception.messages, [COLLECTION_NOT_ACCESSIBLE_MESSAGE])
        self.assertEqual(RetrievalRun.objects.count(), 0)

    def test_an_accessible_collection_narrows(self):
        output = self.search(collection_id=self.coll_a2.pk)
        self.assertEqual({hit["chunk_id"] for hit in output["results"]}, {self.a3.pk})

    def test_invalid_input_is_refused_before_any_record(self):
        for kwargs in (
            {"query": "   "},
            {"limit": 0},
            {"limit": 11},
            {"collection_id": "x"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValidationError):
                    self.search(**kwargs)
        self.assertEqual(RetrievalRun.objects.count(), 0)


# ---------------------------------------------------------------------------
# What the model reads
# ---------------------------------------------------------------------------

@ON
class ResultShapeTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build()

    def test_keys_are_an_exact_allow_list(self):
        output = self.search()
        self.assertEqual(set(output), TOP_LEVEL_KEYS)
        self.assertEqual(set(output["coverage"]), COVERAGE_KEYS)
        self.assertTrue(output["results"])
        for hit in output["results"]:
            self.assertEqual(set(hit), HIT_KEYS)

    def test_no_score_and_no_configuration_reach_the_model(self):
        output = self.search()
        serialized = json.dumps(output)
        for forbidden in (
            "score", "fusion", "e1:", "k1:", "retrieval_run", "provider",
            self.config.model_name, self.embed_provider.name, PROVIDER_SECRET,
            "semantic_failure", "reason_code",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized)

    def test_hits_carry_path_1_citations_and_matched_by(self):
        output = self.search()
        for hit in output["results"]:
            path1 = knowledge_retrieval.cite_knowledge_source(
                self.agent_a, chunk_id=hit["chunk_id"]
            )["citation"]
            self.assertEqual(hit["citation"], path1)
            self.assertTrue(set(hit["matched_by"]) <= {"lexical", "semantic"})
            self.assertTrue(hit["matched_by"])
        self.assertEqual(output["mode"], HybridMode.HYBRID)
        self.assertFalse(output["degraded"])

    def test_excerpts_are_bounded_and_say_so(self):
        output = self.search(max_excerpt_chars=10)
        for hit in output["results"]:
            self.assertLessEqual(len(hit["excerpt"]), 10)
            self.assertTrue(hit["excerpt_truncated"])

    def test_results_are_the_audited_ranking_in_order(self):
        output = self.search()
        run = RetrievalRun.objects.get()
        ranked = list(
            run.hits.order_by("final_rank").values_list("chunk_id_snapshot", flat=True)
        )
        self.assertEqual([hit["chunk_id"] for hit in output["results"]], ranked)

    def test_a_chunk_changed_after_ranking_is_withheld_not_dropped(self):
        real = hybrid_knowledge_search._audited_hybrid_search_with_scope

        def edit_after_ranking(*args, **kwargs):
            audited = real(*args, **kwargs)
            self.a1.content = "edited after ranking"
            self.a1.save()
            return audited

        with mock.patch.object(
            hybrid_knowledge_search, "_audited_hybrid_search_with_scope",
            side_effect=edit_after_ranking,
        ):
            output = self.search()
        by_id = {hit["chunk_id"]: hit for hit in output["results"]}
        self.assertEqual(by_id[self.a1.pk], {"chunk_id": self.a1.pk, "unavailable": CHANGED_SINCE_SEARCH})
        self.assertEqual(output["total"], len(output["results"]))
        self.assertNotIn("edited after ranking", json.dumps(output))
        for hit in output["results"]:
            if hit["chunk_id"] != self.a1.pk:
                self.assertEqual(set(hit), HIT_KEYS)

    def test_semantic_unavailable_degrades_to_lexical_and_says_so(self):
        failing = mock.Mock(
            side_effect=EmbeddingProviderExecutionError(ErrorCategory.PROVIDER_UNREACHABLE)
        )
        output = self.search(failing)
        self.assertEqual(output["mode"], HybridMode.LEXICAL_ONLY)
        self.assertTrue(output["degraded"])
        self.assertNotIn(ErrorCategory.PROVIDER_UNREACHABLE, json.dumps(output))
        self.assertTrue(output["results"])

    def test_output_is_json_serializable_through_the_tool_runtime(self):
        with mock.patch(TRANSPORT_PATH, return_value=self.transport_returning()):
            output = execute_tool(self.tool, {"query": QUERY}, agent=self.agent_a)
        json.dumps(output)
        self.assertEqual(set(output), TOP_LEVEL_KEYS)


# ---------------------------------------------------------------------------
# Coverage (S-27 Q6)
# ---------------------------------------------------------------------------

@ON
class CoverageTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build()

    def test_fully_indexed(self):
        coverage = self.search()["coverage"]
        self.assertEqual(coverage, {"indexed_chunks": 3, "searchable_chunks": 3})

    def test_partly_indexed_is_visible(self):
        KnowledgeChunkEmbedding.objects.filter(chunk=self.a3).delete()
        coverage = self.search()["coverage"]
        self.assertEqual(coverage, {"indexed_chunks": 2, "searchable_chunks": 3})

    def test_unindexed_is_visible_even_though_it_is_not_degraded(self):
        """S-22: a semantic branch with zero candidates is an answer, not a
        degradation - which is exactly why coverage has to be reported."""
        KnowledgeChunkEmbedding.objects.all().delete()
        output = self.search()
        self.assertEqual(output["coverage"], {"indexed_chunks": 0, "searchable_chunks": 3})
        self.assertFalse(output["degraded"])
        self.assertEqual(output["mode"], HybridMode.LEXICAL_ONLY)

    def test_coverage_follows_collection_narrowing(self):
        coverage = self.search(collection_id=self.coll_a2.pk)["coverage"]
        self.assertEqual(coverage, {"indexed_chunks": 1, "searchable_chunks": 1})


# ---------------------------------------------------------------------------
# Nothing that existed changes
# ---------------------------------------------------------------------------

class ExistingBehaviourTests(HybridFixtureMixin, TestCase):
    def setUp(self):
        self.build()

    def test_search_knowledge_is_identical_with_the_flag_on_or_off(self):
        with OFF:
            off = knowledge_retrieval.search_knowledge(self.agent_a, query=QUERY)
        with ON:
            on = knowledge_retrieval.search_knowledge(self.agent_a, query=QUERY)
        self.assertEqual(off, on)
        self.assertIn("score", off["results"][0])

    def test_the_six_path_1_tools_resolve_identically(self):
        with OFF:
            off = self.names() & set(PATH1_TOOLS)
        with ON:
            on = self.names() & set(PATH1_TOOLS)
        self.assertEqual(off, on)
        self.assertEqual(off, set(PATH1_TOOLS))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class PersistenceTests(HybridFixtureMixin, TestCase):
    def test_the_system_tool_is_bound_gated_and_outside_any_toolbox(self):
        tool = ToolDefinition.objects.get(name=HYBRID)
        self.assertTrue(tool.is_system_tool)
        self.assertEqual(tool.operation_mode, "read")
        self.assertEqual(tool.config["callable"], "ai_hub.tools.knowledge.search_knowledge_hybrid")
        self.assertIs(tool.config["bind_agent_context"], True)
        self.assertNotIn("game_tool_category", tool.config)
        self.assertFalse(ToolboxTool.objects.filter(tool=tool).exists())

    def test_the_scope_configuration_is_protected(self):
        self.build()
        with self.assertRaises(ProtectedError):
            self.config.delete()

    def test_scopes_have_no_configuration_by_default(self):
        self.build(configure=False)
        self.assertIsNone(self.scope_a.retrieval_embedding_model_config)


class BoundaryTests(TestCase):
    def test_the_service_catches_no_broad_exception(self):
        tree = ast.parse(inspect.getsource(hybrid_knowledge_search))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                self.assertIsNotNone(node.type)
                names = (
                    {element.id for element in node.type.elts if isinstance(element, ast.Name)}
                    if isinstance(node.type, ast.Tuple)
                    else {getattr(node.type, "id", "")}
                )
                self.assertFalse(names & {"Exception", "BaseException"})

    def test_search_knowledge_adapter_still_calls_the_lexical_service(self):
        from ai_hub.tools import knowledge as knowledge_tools

        source = inspect.getsource(knowledge_tools.search_knowledge)
        self.assertIn("knowledge_retrieval.search_knowledge", source)
        self.assertNotIn("hybrid", source)

    def test_the_service_never_writes_audit_rows_itself(self):
        source = inspect.getsource(hybrid_knowledge_search)
        tree = ast.parse(source)
        called = {
            getattr(node.func, "attr", None)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        for forbidden in ("create", "update", "delete", "save", "bulk_create", "update_or_create"):
            self.assertNotIn(forbidden, called)


class MigrationOwnershipTests(TestCase):
    """S-28 owns 0031 (schema) and 0032 (data), split per the CI #43 lesson."""

    def test_s28_owns_0031_and_0032_in_order(self):
        from django.db.migrations.loader import MigrationLoader

        loader = MigrationLoader(None, ignore_no_migrations=True)
        schema = loader.disk_migrations[("ai_hub", "0031_scope_retrieval_embedding_config")]
        data = loader.disk_migrations[("ai_hub", "0032_hybrid_knowledge_search_tool")]
        self.assertEqual(schema.dependencies, [("ai_hub", "0030_pgvector_ann_foundation")])
        self.assertEqual(data.dependencies, [("ai_hub", "0031_scope_retrieval_embedding_config")])
        self.assertEqual(
            [type(operation).__name__ for operation in schema.operations], ["AddField"]
        )
        self.assertEqual(
            [type(operation).__name__ for operation in data.operations], ["RunPython"]
        )

    def _data_migration(self):
        import importlib

        return importlib.import_module(
            "ai_hub.migrations.0032_hybrid_knowledge_search_tool"
        )

    def test_the_data_migration_reverses_and_reapplies(self):
        from django.apps import apps

        module = self._data_migration()
        module.remove_tool(apps, None)
        self.assertFalse(ToolDefinition.objects.filter(name=HYBRID).exists())
        module.create_tool(apps, None)
        module.create_tool(apps, None)  # idempotent
        self.assertEqual(ToolDefinition.objects.filter(name=HYBRID).count(), 1)

    def test_reversal_keeps_a_tool_that_has_execution_history(self):
        from django.apps import apps

        from ai_hub.models import ToolExecutionRun

        tool = ToolDefinition.objects.get(name=HYBRID)
        ToolExecutionRun.objects.create(tool=tool)
        self._data_migration().remove_tool(apps, None)
        tool.refresh_from_db()
        self.assertFalse(tool.is_active)
