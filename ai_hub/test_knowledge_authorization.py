"""S-15 — the effective ApplicationScope authorization boundary.

The fixture every test in this module shares is the BAD CONFIGURATION S-15
exists to neutralise:

    Scope A: Agent A, Collection A  (secret: A_ONLY_SECRET)
    Scope B: Agent B, Collection B  (secret: B_ONLY_SECRET)
    plus a deliberate cross-scope row:  Agent A -> Collection B

That row is allowed to exist. It must grant ZERO authorization. Every surface
below is checked against it, because fixing `search_knowledge` alone would still
have leaked Collection B's name, description, titles, tags, counts - and, under
the legacy eager prompt path, its full body text.

What is deliberately NOT here: no repair of the bad row, no lifecycle change, no
semantic or vector code, no provider egress, no workspace Knowledge allow-list.
"""

import json
from decimal import Decimal
from io import StringIO
from unittest import mock

from django.core.exceptions import ValidationError
from django.test import TestCase, override_settings

from ai_hub.models import (
    AgentProfile,
    ApplicationScope,
    ExecutionSession,
    GameWorkspace,
    GameWorkspaceAgent,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    ProviderConfig,
    ToolDefinition,
)
from ai_hub.services import knowledge_retrieval
from ai_hub.services.agent_runtime import build_agent_knowledge_context
from ai_hub.services.knowledge_authorization import (
    DENY_ALL,
    EffectiveKnowledgeScope,
    PipelineScopeError,
    require_coherent_pipeline_scope,
    resolve_effective_knowledge_scope,
)


A_SECRET = "A_ONLY_SECRET"
B_SECRET = "B_ONLY_SECRET"

#: Every string that would betray Scope B's existence, in one place, so the
#: leakage probes cannot silently check less than they claim.
B_MARKERS = (B_SECRET, "B Knowledge", "Beta Doc", "beta-tag", "B section")


def make_model_config(name="S15 Provider"):
    provider = ProviderConfig.objects.create(
        name=name, provider_type=ProviderConfig.ProviderType.TRAINING
    )
    return ModelConfig.objects.create(
        provider=provider, model_name="training",
        temperature_default=Decimal("0.10"),
    )


class ScopedCorpusMixin:
    """The two-application corpus, plus the deliberate cross-scope assignment."""

    def build_corpus(self, *, assign_cross_scope=True):
        self.model_config = make_model_config()

        self.scope_a = ApplicationScope.objects.create(name="App A", slug="app-a")
        self.scope_b = ApplicationScope.objects.create(name="App B", slug="app-b")

        self.collection_a = self._collection(
            self.scope_a, "A Knowledge", "Alpha Doc", "alpha-tag",
            "A section", A_SECRET,
        )
        self.collection_b = self._collection(
            self.scope_b, "B Knowledge", "Beta Doc", "beta-tag",
            "B section", B_SECRET,
        )

        self.agent_a = self._agent("A Agent", self.scope_a)
        self.agent_a.knowledge_collections.add(self.collection_a)
        self.agent_b = self._agent("B Agent", self.scope_b)
        self.agent_b.knowledge_collections.add(self.collection_b)

        if assign_cross_scope:
            # THE BAD ROW. Permitted to exist; must authorize nothing.
            self.agent_a.knowledge_collections.add(self.collection_b)

    def _collection(self, scope, name, title, tag, section, secret):
        collection = KnowledgeCollection.objects.create(
            name=name, description=f"{name} description", application_scope=scope
        )
        document = KnowledgeDocument.objects.create(
            collection=collection, title=title,
            curated_text=f"{secret} widgets and gadgets",
            tags=[tag], status=KnowledgeDocument.Status.ACTIVE,
        )
        KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1, section_title=section,
            content=f"{secret} widgets and gadgets in a chunk",
        )
        return collection

    def _agent(self, name, scope):
        return AgentProfile.objects.create(
            name=name, role="r", model_config=self.model_config,
            application_scope=scope, knowledge_max_chars=6000,
        )

    def assertNoScopeBLeakage(self, payload, *, label=""):
        """Serialize anything and prove not one Scope B fact survives."""
        blob = json.dumps(payload, default=str)
        for marker in B_MARKERS:
            self.assertNotIn(marker, blob, f"{label}: leaked {marker!r}")
        for identifier in (
            self.collection_b.pk,
            self.collection_b.documents.get().pk,
            self.collection_b.documents.get().chunks.get().pk,
        ):
            self.assertNotIn(
                f'"{identifier}"', blob, f"{label}: leaked id {identifier}"
            )


# ---------------------------------------------------------------------------
# The canonical resolver
# ---------------------------------------------------------------------------

class ResolveEffectiveKnowledgeScopeTests(ScopedCorpusMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_active_agent_in_active_scope_resolves_its_own_collections(self):
        scope = resolve_effective_knowledge_scope(self.agent_a)
        self.assertEqual(scope.application_scope_id, self.scope_a.pk)
        self.assertEqual(scope.agent_id, self.agent_a.pk)
        self.assertIsNone(scope.workspace_id)
        self.assertEqual(scope.collection_ids, frozenset({self.collection_a.pk}))

    def test_the_cross_scope_assignment_is_absent_from_the_result(self):
        scope = resolve_effective_knowledge_scope(self.agent_a)
        self.assertNotIn(self.collection_b.pk, scope.collection_ids)
        # ...even though the row is really there.
        self.assertIn(
            self.collection_b, self.agent_a.knowledge_collections.all()
        )

    def test_an_inactive_agent_authorizes_nothing(self):
        self.agent_a.is_active = False
        self.agent_a.save(update_fields=["is_active"])
        self.assertEqual(resolve_effective_knowledge_scope(self.agent_a), DENY_ALL)

    def test_an_inactive_application_scope_authorizes_nothing(self):
        self.scope_a.is_active = False
        self.scope_a.save(update_fields=["is_active"])
        self.agent_a.refresh_from_db()
        scope = resolve_effective_knowledge_scope(self.agent_a)
        self.assertEqual(scope, DENY_ALL)
        self.assertTrue(scope.is_empty)

    def test_an_inactive_scope_does_not_fall_back_to_another_scope(self):
        """A deactivated application is off, not relocated."""
        self.scope_a.is_active = False
        self.scope_a.save(update_fields=["is_active"])
        self.agent_a.refresh_from_db()
        scope = resolve_effective_knowledge_scope(self.agent_a)
        self.assertIsNone(scope.application_scope_id)
        self.assertNotEqual(scope.application_scope_id, self.scope_b.pk)

    def test_an_unsaved_agent_authorizes_nothing(self):
        self.assertEqual(
            resolve_effective_knowledge_scope(AgentProfile()), DENY_ALL
        )

    def test_an_unassigned_same_scope_collection_is_denied(self):
        other = KnowledgeCollection.objects.create(
            name="A Unassigned", application_scope=self.scope_a
        )
        scope = resolve_effective_knowledge_scope(self.agent_a)
        self.assertNotIn(other.pk, scope.collection_ids)

    def test_an_inactive_collection_is_denied(self):
        self.collection_a.is_active = False
        self.collection_a.save(update_fields=["is_active"])
        self.assertTrue(resolve_effective_knowledge_scope(self.agent_a).is_empty)

    # -- workspace ----------------------------------------------------------

    def test_a_same_scope_active_workspace_is_accepted(self):
        workspace = GameWorkspace.objects.create(
            name="A Workspace", application_scope=self.scope_a
        )
        scope = resolve_effective_knowledge_scope(self.agent_a, workspace=workspace)
        self.assertEqual(scope.workspace_id, workspace.pk)
        self.assertEqual(scope.collection_ids, frozenset({self.collection_a.pk}))

    def test_a_cross_scope_workspace_refuses_everything(self):
        workspace = GameWorkspace.objects.create(
            name="B Workspace", application_scope=self.scope_b
        )
        scope = resolve_effective_knowledge_scope(self.agent_a, workspace=workspace)
        self.assertEqual(scope, DENY_ALL)

    def test_an_inactive_workspace_refuses_everything(self):
        workspace = GameWorkspace.objects.create(
            name="A Idle", application_scope=self.scope_a, is_active=False
        )
        scope = resolve_effective_knowledge_scope(self.agent_a, workspace=workspace)
        self.assertEqual(scope, DENY_ALL)

    def test_a_workspace_never_widens_scope(self):
        """Workspace restricts execution; it never grants Knowledge."""
        workspace = GameWorkspace.objects.create(
            name="A Wide", application_scope=self.scope_a
        )
        with_workspace = resolve_effective_knowledge_scope(
            self.agent_a, workspace=workspace
        )
        without = resolve_effective_knowledge_scope(self.agent_a)
        self.assertTrue(with_workspace.collection_ids <= without.collection_ids)

    # -- the result object itself ------------------------------------------

    def test_the_result_is_immutable_and_not_a_model(self):
        scope = resolve_effective_knowledge_scope(self.agent_a)
        with self.assertRaises(Exception):
            scope.application_scope_id = 999
        self.assertIsInstance(scope.collection_ids, frozenset)
        self.assertFalse(hasattr(scope, "save"))
        self.assertFalse(hasattr(scope, "_meta"))

    def test_every_refusal_returns_the_same_denial(self):
        """One denial, carrying no reason - so it cannot become an oracle."""
        self.agent_a.is_active = False
        self.agent_a.save(update_fields=["is_active"])
        inactive_agent = resolve_effective_knowledge_scope(self.agent_a)

        foreign_workspace = resolve_effective_knowledge_scope(
            self.agent_b,
            workspace=GameWorkspace.objects.create(
                name="A WS2", application_scope=self.scope_a
            ),
        )
        self.assertEqual(inactive_agent, foreign_workspace)
        self.assertEqual(inactive_agent, DENY_ALL)

    def test_allows_is_narrow_only_and_survives_junk(self):
        scope = resolve_effective_knowledge_scope(self.agent_a)
        self.assertTrue(scope.allows(self.collection_a.pk))
        self.assertFalse(scope.allows(self.collection_b.pk))
        self.assertFalse(scope.allows(None))
        self.assertFalse(scope.allows("not-a-number"))

    def test_the_future_semantic_contract_is_consumable(self):
        """A future retriever needs scope id + collection ids, and only those."""
        scope = resolve_effective_knowledge_scope(self.agent_a)
        self.assertIsInstance(scope.application_scope_id, int)
        self.assertIsInstance(scope.collection_ids, frozenset)
        self.assertTrue(all(isinstance(i, int) for i in scope.collection_ids))


# ---------------------------------------------------------------------------
# Every Knowledge read surface, against the bad row
# ---------------------------------------------------------------------------

class CrossScopeRetrievalMatrixTests(ScopedCorpusMixin, TestCase):
    def setUp(self):
        self.build_corpus()
        self.document_b = self.collection_b.documents.get()
        self.chunk_b = self.document_b.chunks.get()
        self.chunk_a = self.collection_a.documents.get().chunks.get()

    def test_list_knowledge_libraries(self):
        result = knowledge_retrieval.list_knowledge_libraries(self.agent_a)
        self.assertNoScopeBLeakage(result, label="list")
        self.assertEqual(result["total"], 1)
        self.assertEqual([row["name"] for row in result["libraries"]], ["A Knowledge"])

    def test_browse_knowledge_index(self):
        result = knowledge_retrieval.browse_knowledge_index(self.agent_a)
        self.assertNoScopeBLeakage(result, label="browse")

    def test_browse_narrowed_explicitly_to_collection_b(self):
        result = knowledge_retrieval.browse_knowledge_index(
            self.agent_a, collection_id=self.collection_b.pk
        )
        self.assertNoScopeBLeakage(result, label="browse narrowed")
        self.assertEqual(result["collections"], [])
        self.assertEqual(result["total"], 0)

    def test_search_knowledge(self):
        result = knowledge_retrieval.search_knowledge(
            self.agent_a, query="widgets", limit=20
        )
        self.assertNoScopeBLeakage(result, label="search")
        self.assertEqual(result["total"], 1)
        # The unauthorized chunk was never even a candidate.
        self.assertEqual(result["candidates_scanned"], 1)

    def test_search_narrowed_explicitly_to_collection_b_is_refused(self):
        with self.assertRaises(ValidationError) as raised:
            knowledge_retrieval.search_knowledge(
                self.agent_a, query="widgets", collection_id=self.collection_b.pk
            )
        self.assertNoScopeBLeakage(
            {"message": raised.exception.messages[0]}, label="search narrowed"
        )

    def test_read_knowledge_chunk(self):
        with self.assertRaises(ValidationError) as raised:
            knowledge_retrieval.read_knowledge_chunk(
                self.agent_a, chunk_id=self.chunk_b.pk
            )
        self.assertNoScopeBLeakage(
            {"message": raised.exception.messages[0]}, label="read chunk"
        )

    def test_read_document_section(self):
        with self.assertRaises(ValidationError) as raised:
            knowledge_retrieval.read_document_section(
                self.agent_a, document_id=self.document_b.pk, chunk_index=1
            )
        self.assertNoScopeBLeakage(
            {"message": raised.exception.messages[0]}, label="read section"
        )

    def test_cite_knowledge_source(self):
        with self.assertRaises(ValidationError) as raised:
            knowledge_retrieval.cite_knowledge_source(
                self.agent_a, chunk_id=self.chunk_b.pk
            )
        self.assertNoScopeBLeakage(
            {"message": raised.exception.messages[0]}, label="cite"
        )

    def test_prompt_context_normal_mode(self):
        context = build_agent_knowledge_context(self.agent_a)
        self.assertNoScopeBLeakage(context, label="prompt normal")
        self.assertEqual(context["collections"], ["A Knowledge"])
        self.assertEqual(context["total_collections"], 1)

    @override_settings(AI_HUB_LEGACY_EAGER_KNOWLEDGE_CONTEXT_ENABLED=True)
    def test_prompt_context_legacy_eager_mode(self):
        """The path that carries full body text - the most dangerous one."""
        context = build_agent_knowledge_context(self.agent_a)
        self.assertNoScopeBLeakage(context, label="prompt eager")
        self.assertIn(A_SECRET, context["text"])
        self.assertEqual(context["collections"], ["A Knowledge"])

    def test_scope_b_agent_is_unaffected_and_still_sees_its_own(self):
        result = knowledge_retrieval.search_knowledge(
            self.agent_b, query="widgets", limit=20
        )
        self.assertEqual(
            {row["collection"] for row in result["results"]}, {"B Knowledge"}
        )


# ---------------------------------------------------------------------------
# ADR-N5: the three failure cases must be indistinguishable
# ---------------------------------------------------------------------------

class NonDisclosureTests(ScopedCorpusMixin, TestCase):
    def setUp(self):
        self.build_corpus()
        self.unassigned = KnowledgeCollection.objects.create(
            name="A Unassigned", application_scope=self.scope_a
        )
        unassigned_document = KnowledgeDocument.objects.create(
            collection=self.unassigned, title="Unassigned Doc",
            curated_text="widgets", status=KnowledgeDocument.Status.ACTIVE,
        )
        self.unassigned_chunk = KnowledgeDocumentChunk.objects.create(
            document=unassigned_document, chunk_index=1,
            section_title="U section", content="widgets",
        )
        self.chunk_b = self.collection_b.documents.get().chunks.get()

    def _message(self, call, **kwargs):
        try:
            call(self.agent_a, **kwargs)
        except ValidationError as exc:
            return exc.messages[0]
        return "<no error>"

    def test_collection_narrowing_is_indistinguishable(self):
        messages = {
            case: self._message(
                knowledge_retrieval.search_knowledge, query="widgets",
                collection_id=collection_id,
            )
            for case, collection_id in (
                ("nonexistent", 999_999),
                ("other scope", self.collection_b.pk),
                ("same scope, unassigned", self.unassigned.pk),
            )
        }
        self.assertEqual(len(set(messages.values())), 1, messages)

    def test_chunk_read_is_indistinguishable(self):
        messages = {
            case: self._message(
                knowledge_retrieval.read_knowledge_chunk, chunk_id=chunk_id
            )
            for case, chunk_id in (
                ("nonexistent", 999_999),
                ("other scope", self.chunk_b.pk),
                ("same scope, unassigned", self.unassigned_chunk.pk),
            )
        }
        self.assertEqual(len(set(messages.values())), 1, messages)

    def test_document_section_read_is_indistinguishable(self):
        messages = {
            case: self._message(
                knowledge_retrieval.read_document_section,
                document_id=document_id, chunk_index=1,
            )
            for case, document_id in (
                ("nonexistent", 999_999),
                ("other scope", self.collection_b.documents.get().pk),
                ("same scope, unassigned", self.unassigned.documents.get().pk),
            )
        }
        self.assertEqual(len(set(messages.values())), 1, messages)

    def test_no_error_message_names_a_scope(self):
        for message in (
            self._message(knowledge_retrieval.read_knowledge_chunk,
                          chunk_id=self.chunk_b.pk),
            self._message(knowledge_retrieval.search_knowledge, query="w",
                          collection_id=self.collection_b.pk),
        ):
            with self.subTest(message=message):
                for forbidden in ("App A", "App B", "app-a", "app-b", "scope"):
                    self.assertNotIn(forbidden, message.lower()
                                     if forbidden.islower() else message)

    def test_results_never_expose_application_scope_identifiers(self):
        """The new boundary must not become new metadata."""
        payloads = [
            knowledge_retrieval.list_knowledge_libraries(self.agent_a),
            knowledge_retrieval.browse_knowledge_index(self.agent_a),
            knowledge_retrieval.search_knowledge(self.agent_a, query="widgets"),
            build_agent_knowledge_context(self.agent_a),
        ]
        for index, payload in enumerate(payloads):
            with self.subTest(payload=index):
                blob = json.dumps(payload, default=str)
                self.assertNotIn("application_scope", blob)
                self.assertNotIn("App A", blob)
                self.assertNotIn("app-a", blob)


# ---------------------------------------------------------------------------
# Same-scope regression: nothing legitimate changed
# ---------------------------------------------------------------------------

class SameScopeRegressionTests(ScopedCorpusMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_every_surface_still_works_inside_one_scope(self):
        chunk = self.collection_a.documents.get().chunks.get()

        libraries = knowledge_retrieval.list_knowledge_libraries(self.agent_a)
        self.assertEqual(libraries["total"], 1)

        index = knowledge_retrieval.browse_knowledge_index(self.agent_a)
        self.assertEqual(len(index["collections"]), 1)

        search = knowledge_retrieval.search_knowledge(self.agent_a, query="widgets")
        self.assertEqual(search["total"], 1)

        read = knowledge_retrieval.read_knowledge_chunk(self.agent_a, chunk_id=chunk.pk)
        self.assertIn(A_SECRET, read["content"])

        section = knowledge_retrieval.read_document_section(
            self.agent_a, document_id=chunk.document_id, chunk_index=1
        )
        self.assertIn(A_SECRET, section["content"])

        citation = knowledge_retrieval.cite_knowledge_source(
            self.agent_a, chunk_id=chunk.pk
        )
        self.assertEqual(citation["citation"]["collection"], "A Knowledge")

        context = build_agent_knowledge_context(self.agent_a)
        self.assertEqual(context["collections"], ["A Knowledge"])

    def test_explicit_same_scope_narrowing_still_works(self):
        result = knowledge_retrieval.search_knowledge(
            self.agent_a, query="widgets", collection_id=self.collection_a.pk
        )
        self.assertEqual(result["total"], 1)

    def test_lexical_scoring_is_unchanged(self):
        """S-15 touched authorization, not ranking."""
        result = knowledge_retrieval.search_knowledge(self.agent_a, query="widgets")
        row = result["results"][0]
        self.assertIn("score", row)
        self.assertGreater(row["score"], 0)
        self.assertIn("citation", row)
        self.assertEqual(
            set(row["citation"]),
            {"collection", "document_id", "document_title", "chunk_id",
             "section_title", "chunk_index", "language", "tags"},
        )

    def test_knowledge_lifecycle_facts_are_untouched(self):
        document = self.collection_a.documents.get()
        self.assertEqual(
            document.chunk_authority_mode,
            KnowledgeDocument.ChunkAuthorityMode.UNKNOWN,
        )
        self.assertEqual(document.generation_input_fingerprint, "")
        self.assertEqual(document.generation_chunk_set_fingerprint, "")
        self.assertIsNone(document.generator_version)


# ---------------------------------------------------------------------------
# Server-bound tool identity, through the REAL runtime
# ---------------------------------------------------------------------------

class BoundToolIdentityTests(ScopedCorpusMixin, TestCase):
    """A model-supplied identity must never become the authorization principal."""

    def setUp(self):
        from ai_hub.services.starter_toolboxes import seed_starter_toolboxes

        # Seed while only ONE scope exists: `require_single_active_scope()`
        # correctly refuses to guess once a second application appears, which is
        # itself the S-14 behaviour under test elsewhere.
        self.model_config = make_model_config()
        self.scope_a = ApplicationScope.objects.create(name="App A", slug="app-a")
        seed_starter_toolboxes()
        self.tool = ToolDefinition.objects.get(name="search_knowledge")

        self.scope_b = ApplicationScope.objects.create(name="App B", slug="app-b")
        self.collection_a = self._collection(
            self.scope_a, "A Knowledge", "Alpha Doc", "alpha-tag",
            "A section", A_SECRET,
        )
        self.collection_b = self._collection(
            self.scope_b, "B Knowledge", "Beta Doc", "beta-tag",
            "B section", B_SECRET,
        )
        self.agent_a = self._agent("A Agent", self.scope_a)
        self.agent_a.knowledge_collections.add(self.collection_a)
        self.agent_b = self._agent("B Agent", self.scope_b)
        self.agent_b.knowledge_collections.add(self.collection_b)
        self.agent_a.knowledge_collections.add(self.collection_b)

    def _execute(self, payload, *, agent):
        from ai_hub.services.tools_runtime import execute_tool

        return execute_tool(self.tool, payload, agent=agent)

    def test_a_spoofed_agent_id_does_not_change_the_principal(self):
        result = self._execute(
            {"query": "widgets", "agent_id": self.agent_b.pk}, agent=self.agent_a
        )
        self.assertNoScopeBLeakage(result, label="spoofed agent_id")

    def test_a_spoofed_agent_name_does_not_change_the_principal(self):
        result = self._execute(
            {"query": "widgets", "agent_name": self.agent_b.name}, agent=self.agent_a
        )
        self.assertNoScopeBLeakage(result, label="spoofed agent_name")

    def test_the_runtime_overwrites_the_payload_identity(self):
        from ai_hub.services.tools_runtime import bind_tool_runtime_context

        bound = bind_tool_runtime_context(
            self.tool,
            {"query": "w", "agent_id": self.agent_b.pk, "agent_name": self.agent_b.name},
            agent=self.agent_a,
        )
        self.assertEqual(bound["agent_id"], self.agent_a.pk)
        self.assertNotIn("agent_name", bound)

    def test_a_bound_tool_without_a_runtime_agent_is_refused(self):
        from ai_hub.services.tools_runtime import execute_tool

        with self.assertRaises(ValidationError):
            execute_tool(self.tool, {"query": "widgets"}, agent=None)

    def test_the_legitimate_principal_still_gets_its_own_knowledge(self):
        result = self._execute({"query": "widgets"}, agent=self.agent_a)
        self.assertIn(A_SECRET, json.dumps(result, default=str))


# ---------------------------------------------------------------------------
# GAME: workspace / agent scope coherence
# ---------------------------------------------------------------------------

class GameWorkspaceScopeTests(ScopedCorpusMixin, TestCase):
    def setUp(self):
        self.build_corpus()
        from ai_hub.services.game_policy import (
            PolicyViolationError,
            validate_agent_for_workspace,
        )

        self.PolicyViolationError = PolicyViolationError
        self.validate = validate_agent_for_workspace
        self.workspace_a = GameWorkspace.objects.create(
            name="WS A", application_scope=self.scope_a
        )
        self.workspace_b = GameWorkspace.objects.create(
            name="WS B", application_scope=self.scope_b
        )

    def test_same_scope_with_an_empty_allow_list_is_permitted(self):
        """Legacy empty-means-open is preserved INSIDE one scope."""
        self.assertEqual(GameWorkspaceAgent.objects.filter(
            workspace=self.workspace_a).count(), 0)
        self.validate(self.workspace_a, self.agent_a)   # no raise

    def test_cross_scope_with_an_EMPTY_allow_list_is_still_refused(self):
        """The case the legacy rule would otherwise wave through."""
        self.assertEqual(GameWorkspaceAgent.objects.filter(
            workspace=self.workspace_b).count(), 0)
        with self.assertRaises(self.PolicyViolationError):
            self.validate(self.workspace_b, self.agent_a)

    def test_cross_scope_with_an_EXPLICITLY_ENABLED_row_is_still_refused(self):
        """Even an operator explicitly enabling it cannot bridge applications."""
        GameWorkspaceAgent.objects.create(
            workspace=self.workspace_b, agent=self.agent_a, is_enabled=True
        )
        with self.assertRaises(self.PolicyViolationError):
            self.validate(self.workspace_b, self.agent_a)

    def test_same_scope_but_disabled_is_still_refused(self):
        GameWorkspaceAgent.objects.create(
            workspace=self.workspace_a, agent=self.agent_a, is_enabled=False
        )
        with self.assertRaises(self.PolicyViolationError):
            self.validate(self.workspace_a, self.agent_a)

    def test_an_inactive_workspace_is_refused(self):
        self.workspace_a.is_active = False
        self.workspace_a.save(update_fields=["is_active"])
        with self.assertRaises(self.PolicyViolationError):
            self.validate(self.workspace_a, self.agent_a)

    def test_an_inactive_scope_is_refused(self):
        self.scope_a.is_active = False
        self.scope_a.save(update_fields=["is_active"])
        self.workspace_a.refresh_from_db()
        with self.assertRaises(self.PolicyViolationError):
            self.validate(self.workspace_a, self.agent_a)

    def test_the_refusal_does_not_name_the_other_scope(self):
        with self.assertRaises(self.PolicyViolationError) as raised:
            self.validate(self.workspace_b, self.agent_a)
        message = str(raised.exception)
        for forbidden in ("App A", "App B", "app-a", "app-b"):
            self.assertNotIn(forbidden, message)

    def test_goal_execution_policy_refuses_a_cross_scope_entry_agent(self):
        from ai_hub.models import GameGoal
        from ai_hub.services.game_policy import validate_goal_execution_policy

        goal = GameGoal.objects.create(workspace=self.workspace_b, title="g")
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal=goal, entry_agent=self.agent_a, goal_text="g",
        )
        with self.assertRaises(self.PolicyViolationError):
            validate_goal_execution_policy(self.workspace_b, goal, session)


# ---------------------------------------------------------------------------
# Orchestrator: pipeline scope coherence
# ---------------------------------------------------------------------------

class PipelineScopeCoherenceTests(ScopedCorpusMixin, TestCase):
    def setUp(self):
        self.build_corpus()
        for agent in (self.agent_a, self.agent_b):
            agent.input_contract = {"required": []}
            agent.output_contract = {"required": ["agent"]}
            agent.save(update_fields=["input_contract", "output_contract"])
        self.agent_a2 = self._agent("A Agent 2", self.scope_a)
        self.agent_a2.input_contract = {"required": []}
        self.agent_a2.output_contract = {"required": ["agent"]}
        self.agent_a2.save(update_fields=["input_contract", "output_contract"])

    def _pipeline(self, entry, steps, *, fallback=None):
        pipeline = PipelineDefinition.objects.create(
            name=f"P{PipelineDefinition.objects.count()}", entry_agent=entry
        )
        for order, agent in enumerate(steps, start=1):
            PipelineStep.objects.create(
                pipeline=pipeline, agent=agent, order=order,
                fallback_agent=fallback if order == 1 else None,
                on_error=(
                    PipelineStep.OnError.FALLBACK_AGENT if fallback and order == 1
                    else PipelineStep.OnError.STOP
                ),
            )
        return pipeline

    # -- the derived contract ----------------------------------------------

    def test_a_same_scope_pipeline_resolves_to_one_scope(self):
        pipeline = self._pipeline(self.agent_a, [self.agent_a, self.agent_a2])
        self.assertEqual(require_coherent_pipeline_scope(pipeline), self.scope_a.pk)

    def test_a_cross_scope_pipeline_is_refused(self):
        pipeline = self._pipeline(self.agent_a, [self.agent_a, self.agent_b])
        with self.assertRaises(PipelineScopeError):
            require_coherent_pipeline_scope(pipeline)

    def test_a_cross_scope_FALLBACK_agent_is_refused(self):
        """Fallback agents are part of the boundary, not an afterthought."""
        pipeline = self._pipeline(
            self.agent_a, [self.agent_a, self.agent_a2], fallback=self.agent_b
        )
        with self.assertRaises(PipelineScopeError):
            require_coherent_pipeline_scope(pipeline)

    def test_a_cross_scope_ENTRY_agent_is_refused(self):
        pipeline = self._pipeline(self.agent_b, [self.agent_a, self.agent_a2])
        with self.assertRaises(PipelineScopeError):
            require_coherent_pipeline_scope(pipeline)

    def test_the_error_does_not_name_the_scopes(self):
        pipeline = self._pipeline(self.agent_a, [self.agent_a, self.agent_b])
        with self.assertRaises(PipelineScopeError) as raised:
            require_coherent_pipeline_scope(pipeline)
        for forbidden in ("App A", "App B", "app-a", "app-b"):
            self.assertNotIn(forbidden, str(raised.exception))

    # -- configuration time -------------------------------------------------

    def test_activation_refuses_a_cross_scope_pipeline(self):
        pipeline = self._pipeline(self.agent_a, [self.agent_a, self.agent_b])
        pipeline.is_active = True
        with self.assertRaises(ValidationError):
            pipeline.full_clean()

    def test_step_validation_refuses_a_cross_scope_fallback(self):
        pipeline = self._pipeline(self.agent_a, [self.agent_a])
        step = PipelineStep(
            pipeline=pipeline, agent=self.agent_a2, order=2,
            on_error=PipelineStep.OnError.FALLBACK_AGENT,
            fallback_agent=self.agent_b,
        )
        with self.assertRaises(ValidationError):
            step.clean()

    # -- runtime is authoritative ------------------------------------------

    def test_runtime_refuses_a_pipeline_mutated_after_validation(self):
        """The reason configuration-time validation is not the boundary.

        Build and activate a legitimate same-scope pipeline, create a session
        for it, and only THEN swap a step's agent using raw ORM - which bypasses
        `full_clean()` entirely. Execution must still refuse, and must refuse
        before any provider or tool call.
        """
        from ai_hub.services.execution_runner import run_execution_session

        pipeline = self._pipeline(self.agent_a, [self.agent_a, self.agent_a2])
        pipeline.is_active = True
        pipeline.save(update_fields=["is_active"])
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR,
            pipeline=pipeline, entry_agent=self.agent_a,
        )

        # raw ORM: no validation runs at all
        PipelineStep.objects.filter(pipeline=pipeline, order=2).update(
            agent=self.agent_b
        )

        with mock.patch(
            "ai_hub.services.execution_runner._execute_session_agent"
        ) as execute_agent:
            run_execution_session(session.pk)

        session.refresh_from_db()
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        execute_agent.assert_not_called()   # refused BEFORE any downstream call

    def test_runtime_permits_a_coherent_pipeline(self):
        from ai_hub.services.execution_runner import run_execution_session

        pipeline = self._pipeline(self.agent_a, [self.agent_a])
        pipeline.is_active = True
        pipeline.save(update_fields=["is_active"])
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR,
            pipeline=pipeline, entry_agent=self.agent_a,
        )

        with mock.patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            return_value={"agent": self.agent_a.name},
        ) as execute_agent:
            run_execution_session(session.pk)

        execute_agent.assert_called()

    def test_a_cross_scope_fallback_refuses_at_runtime_too(self):
        from ai_hub.services.execution_runner import run_execution_session

        pipeline = self._pipeline(self.agent_a, [self.agent_a])
        pipeline.is_active = True
        pipeline.save(update_fields=["is_active"])
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR,
            pipeline=pipeline, entry_agent=self.agent_a,
        )
        PipelineStep.objects.filter(pipeline=pipeline, order=1).update(
            fallback_agent=self.agent_b,
            on_error=PipelineStep.OnError.FALLBACK_AGENT,
        )

        with mock.patch(
            "ai_hub.services.execution_runner._execute_session_agent"
        ) as execute_agent:
            run_execution_session(session.pk)

        session.refresh_from_db()
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        execute_agent.assert_not_called()


# ---------------------------------------------------------------------------
# Query shape
# ---------------------------------------------------------------------------

class AuthorizationQueryShapeTests(ScopedCorpusMixin, TestCase):
    def setUp(self):
        self.build_corpus()
        for index in range(5):
            document = KnowledgeDocument.objects.create(
                collection=self.collection_a, title=f"Extra {index}",
                curated_text="widgets", status=KnowledgeDocument.Status.ACTIVE,
            )
            KnowledgeDocumentChunk.objects.create(
                document=document, chunk_index=1,
                section_title=f"E{index}", content="widgets and gadgets",
            )

    def test_the_scope_resolves_once_per_search_not_per_chunk(self):
        """Authorization must constrain one queryset, not run per candidate."""
        with self.assertNumQueries(2):
            knowledge_retrieval.search_knowledge(self.agent_a, query="widgets", limit=20)

    def test_search_query_count_does_not_grow_with_the_corpus(self):
        before = len(self._capture(lambda: knowledge_retrieval.search_knowledge(
            self.agent_a, query="widgets", limit=20)))
        for index in range(20):
            document = KnowledgeDocument.objects.create(
                collection=self.collection_a, title=f"Bulk {index}",
                curated_text="widgets", status=KnowledgeDocument.Status.ACTIVE,
            )
            KnowledgeDocumentChunk.objects.create(
                document=document, chunk_index=1,
                section_title=f"B{index}", content="widgets and gadgets",
            )
        after = len(self._capture(lambda: knowledge_retrieval.search_knowledge(
            self.agent_a, query="widgets", limit=20)))
        self.assertEqual(before, after)

    def _capture(self, call):
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        with CaptureQueriesContext(connection) as captured:
            call()
        return list(captured)

    def test_the_scope_predicate_reaches_sql(self):
        """Not a post-filter: the predicate is in the query itself."""
        from ai_hub.services.knowledge_authorization import authorized_chunks

        scope = resolve_effective_knowledge_scope(self.agent_a)
        sql = str(authorized_chunks(scope).query).lower()
        self.assertIn("application_scope", sql)
        self.assertIn("collection_id", sql)
