from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.test import Client, TestCase
from django.urls import reverse

from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    ExecutionStepRun,
    KnowledgeCollection,
    KnowledgeDocument,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    ProviderConfig,
)
from ai_hub.services.litellm_client import completion_call
from ai_hub.services.admin_control_center import build_control_center_context
from ai_hub.services.execution_sessions import create_execution_session
from ai_hub.services.agent_runtime import build_agent_knowledge_context
from ai_hub.services.contracts import validate_payload
from ai_hub.services.execution_runner import run_execution_session
# DreamPost was the original host-app model; replaced with User for portability


class HubModelValidationTests(TestCase):
    def test_cannot_activate_agent_with_inactive_provider(self):
        provider = ProviderConfig.objects.create(name="p", provider_type="openai", is_active=False)
        model = ModelConfig.objects.create(provider=provider, model_name="gpt-x")
        agent = AgentProfile(
            name="agent-1",
            role="extractor",
            model_config=model,
            input_contract={"required": ["dream_id"]},
            output_contract={"required": ["agent"]},
            is_active=True,
        )
        with self.assertRaises(ValidationError):
            agent.full_clean()

    def test_cannot_activate_pipeline_with_gaps(self):
        provider = ProviderConfig.objects.create(name="p2", provider_type="openai")
        model = ModelConfig.objects.create(provider=provider, model_name="gpt-y")
        agent = AgentProfile.objects.create(
            name="agent-2",
            role="extractor",
            model_config=model,
            input_contract={"required": ["dream_id"]},
            output_contract={"required": ["agent"]},
        )
        pipeline = PipelineDefinition.objects.create(name="pipe-1", is_active=False)
        PipelineStep.objects.create(pipeline=pipeline, agent=agent, order=2)
        pipeline.is_active = True
        with self.assertRaises(ValidationError):
            pipeline.full_clean()

    def test_contract_validation_checks_basic_types(self):
        schema = {"required": ["dream_id"], "properties": {"dream_id": {"type": "integer"}}}
        with self.assertRaises(ValidationError):
            validate_payload({"dream_id": "1"}, schema, "Test")

    def test_agent_knowledge_context_uses_only_active_documents(self):
        provider = ProviderConfig.objects.create(name="p3", provider_type="openai")
        model = ModelConfig.objects.create(provider=provider, model_name="gpt-z")
        agent = AgentProfile.objects.create(
            name="agent-knowledge",
            role="reader",
            model_config=model,
            input_contract={"required": ["knowledge_context"]},
            output_contract={"required": ["agent"]},
            knowledge_max_chars=20,
        )
        active_collection = KnowledgeCollection.objects.create(name="Symbols")
        inactive_collection = KnowledgeCollection.objects.create(name="Hidden", is_active=False)
        KnowledgeDocument.objects.create(
            collection=active_collection,
            title="Active doc",
            curated_text="Active knowledge content that should be truncated.",
            status=KnowledgeDocument.Status.ACTIVE,
            language="en",
        )
        KnowledgeDocument.objects.create(
            collection=active_collection,
            title="Draft doc",
            curated_text="Draft knowledge should not appear.",
            status=KnowledgeDocument.Status.DRAFT,
        )
        KnowledgeDocument.objects.create(
            collection=inactive_collection,
            title="Inactive collection doc",
            curated_text="Inactive collection should not appear.",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        agent.knowledge_collections.add(active_collection, inactive_collection)

        context = build_agent_knowledge_context(agent)

        self.assertEqual(len(context["documents"]), 1)
        self.assertEqual(context["documents"][0]["title"], "Active doc")
        self.assertEqual(context["documents"][0]["content"], "Active knowledge con")
        self.assertTrue(context["truncated"])

    def test_ai_hub_section_changelists_show_guidance_cards(self):
        admin_user = get_user_model().objects.create_user(
            username="admin-ui",
            password="testpass123",
            is_staff=True,
            is_superuser=True,
        )
        client = Client()
        client.force_login(admin_user)

        provider_response = client.get(reverse("admin:ai_hub_providerconfig_changelist"))
        agent_response = client.get(reverse("admin:ai_hub_agentprofile_changelist"))

        self.assertEqual(provider_response.status_code, 200)
        self.assertEqual(agent_response.status_code, 200)
        self.assertContains(provider_response, "Connect the AI services your agents will call")
        self.assertContains(provider_response, "Open control center")
        self.assertContains(agent_response, "Define a specialist once")
        self.assertContains(agent_response, "Open GAME")


class HubExecutionSessionTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="sessionuser",
            password="testpass123",
        )
        # Use the User itself as the generic source_object (any Django model works)
        self.source = self.user
        self.provider = ProviderConfig.objects.create(name="session-provider", provider_type="openai")
        self.model = ModelConfig.objects.create(provider=self.provider, model_name="session-model")
        self.agent = AgentProfile.objects.create(
            name="session-agent",
            role="Generic entry agent",
            model_config=self.model,
            input_contract={"required": ["source"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        self.pipeline = PipelineDefinition.objects.create(name="session-pipeline", is_active=False)
        self.pipeline_step = PipelineStep.objects.create(
            pipeline=self.pipeline,
            agent=self.agent,
            order=1,
            input_mapping={"source": "source"},
            output_mapping={"first_agent": "agent"},
        )

    def test_create_execution_session_links_to_any_project_object(self):
        session = create_execution_session(
            source_object=self.source,
            pipeline=self.pipeline,
            triggered_by=self.user,
            goal_text="Interpret this source object.",
            initial_context={"source": "dream"},
        )

        self.assertEqual(session.source_object, self.source)
        self.assertEqual(session.source_label, str(self.source))
        self.assertEqual(session.pipeline, self.pipeline)
        self.assertEqual(session.triggered_by, self.user)
        self.assertEqual(session.runtime_kind, ExecutionSession.RuntimeKind.ORCHESTRATOR)
        self.assertEqual(session.runtime_mode, ExecutionSession.RuntimeMode.ASYNC)
        self.assertEqual(session.status, ExecutionSession.Status.PENDING)
        self.assertEqual(session.initial_context, {"source": "dream"})

    def test_orchestrator_execution_session_requires_pipeline(self):
        with self.assertRaises(ValidationError):
            create_execution_session(
                source_object=self.source,
                triggered_by=self.user,
                runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR,
            )

    def test_game_execution_session_can_start_from_entry_agent(self):
        session = create_execution_session(
            source_object=self.source,
            entry_agent=self.agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_config={"max_iterations": 3},
        )

        self.assertIsNone(session.pipeline)
        self.assertEqual(session.entry_agent, self.agent)
        self.assertEqual(session.runtime_kind, ExecutionSession.RuntimeKind.GAME)
        self.assertEqual(session.runtime_config, {"max_iterations": 3})

    def test_execution_step_order_is_unique_per_session(self):
        session = create_execution_session(source_object=self.source, pipeline=self.pipeline, triggered_by=self.user)
        ExecutionStepRun.objects.create(
            session=session,
            order=1,
            pipeline_step=self.pipeline_step,
            agent=self.agent,
            action_name="call_model",
        )

        with self.assertRaises(IntegrityError):
            ExecutionStepRun.objects.create(session=session, order=1, agent=self.agent)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_run_execution_session_executes_pipeline_without_dream_binding(self, mocked_call):
        second_agent = AgentProfile.objects.create(
            name="session-agent-2",
            role="Second generic agent",
            model_config=self.model,
            input_contract={"required": ["first_agent"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        PipelineStep.objects.create(
            pipeline=self.pipeline,
            agent=second_agent,
            order=2,
            input_mapping={"first_agent": "first_agent"},
            output_mapping={"second_agent": "agent"},
        )
        self.pipeline.is_active = True
        self.pipeline.global_input_contract = {"required": ["source"]}
        self.pipeline.global_output_contract = {"required": ["second_agent"]}
        self.pipeline.save(update_fields=["is_active", "global_input_contract", "global_output_contract"])
        session = create_execution_session(
            source_object=self.source,
            pipeline=self.pipeline,
            triggered_by=self.user,
            initial_context={"source": "plain reusable input"},
        )
        mocked_call.return_value = {"status": "ok", "content": "done"}

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(session.step_runs.count(), 2)
        self.assertEqual(session.final_context["source"], "plain reusable input")
        self.assertEqual(session.final_context["first_agent"], "session-agent")
        self.assertEqual(session.final_context["second_agent"], "session-agent-2")

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_run_execution_session_merges_json_final_output_before_contract_validation(self, mocked_call):
        self.pipeline.is_active = True
        self.pipeline.global_input_contract = {"required": ["source"]}
        self.pipeline.global_output_contract = {"required": ["result", "score"]}
        self.pipeline.steps.filter(order=1).update(output_mapping={"final_output": "llm.content"})
        self.pipeline.save(update_fields=["is_active", "global_input_contract", "global_output_contract"])
        session = create_execution_session(
            source_object=self.source,
            pipeline=self.pipeline,
            triggered_by=self.user,
            initial_context={"source": "plain reusable input"},
        )
        mocked_call.return_value = {"status": "ok", "content": '{"result": "done", "score": 0.91}'}

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(session.final_context["result"], "done")
        self.assertEqual(session.final_context["score"], 0.91)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_run_execution_session_marks_failed_on_agent_error(self, mocked_call):
        self.pipeline.is_active = True
        self.pipeline.save(update_fields=["is_active"])
        session = create_execution_session(
            source_object=self.source,
            pipeline=self.pipeline,
            triggered_by=self.user,
            initial_context={"source": "plain reusable input"},
        )
        mocked_call.side_effect = Exception("model failed")

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(session.error_detail, "model failed")
        self.assertEqual(session.step_runs.filter(status=ExecutionStepRun.Status.FAILED).count(), 1)

    def test_run_execution_session_rejects_inactive_pipeline(self):
        session = create_execution_session(
            source_object=self.source,
            pipeline=self.pipeline,
            triggered_by=self.user,
            initial_context={"source": "plain reusable input"},
        )

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertIn("Pipeline must be active", session.error_detail)
        self.assertIsNotNone(session.started_at)
        self.assertIsNotNone(session.finished_at)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_run_game_execution_session_stops_when_agent_finishes(self, mocked_call):
        game_agent = AgentProfile.objects.create(
            name="game-agent",
            role="Autonomous goal runner",
            model_config=self.model,
            input_contract={"required": ["goal", "iteration", "memory"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        session = create_execution_session(
            source_object=self.source,
            entry_agent=game_agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Produce a concise answer.",
            runtime_config={"max_iterations": 5},
            initial_context={"source": "game source"},
        )
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"action": "finish", "final_answer": "Goal complete."}',
        }

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(session.step_runs.count(), 1)
        self.assertEqual(session.final_context["finish_reason"], "agent_finished")
        self.assertEqual(session.final_context["final_answer"], "Goal complete.")
        step_run = session.step_runs.get()
        self.assertEqual(step_run.action_name, "game_iteration")
        self.assertTrue(step_run.observation_payload["complete"])
        self.assertEqual(step_run.request_payload["goal"], "Produce a concise answer.")
        self.assertIn("game_response_contract", step_run.request_payload)
        self.assertIn("finish", step_run.request_payload["game_response_contract"]["actions"])

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_run_game_execution_session_stops_at_max_iterations(self, mocked_call):
        game_agent = AgentProfile.objects.create(
            name="game-agent-max",
            role="Autonomous goal runner",
            model_config=self.model,
            input_contract={"required": ["goal", "iteration"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        session = create_execution_session(
            source_object=self.source,
            entry_agent=game_agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Explore until the cap.",
            runtime_config={"max_iterations": 2},
        )
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"action": "think", "message": "Still working."}',
        }

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(session.step_runs.count(), 2)
        self.assertEqual(session.final_context["finish_reason"], "max_iterations")
        self.assertEqual(len(session.final_context["memory"]), 2)

    def test_run_game_execution_session_requires_goal(self):
        game_agent = AgentProfile.objects.create(
            name="game-agent-no-goal",
            role="Autonomous goal runner",
            model_config=self.model,
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        session = create_execution_session(
            source_object=self.source,
            entry_agent=game_agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
        )

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertIn("goal_text", session.error_detail)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_run_game_execution_session_strict_contract_rejects_plain_text(self, mocked_call):
        game_agent = AgentProfile.objects.create(
            name="game-agent-strict-contract",
            role="Autonomous goal runner",
            model_config=self.model,
            input_contract={"required": ["goal", "game_response_contract"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        session = create_execution_session(
            source_object=self.source,
            entry_agent=game_agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Require valid JSON.",
            runtime_config={"max_iterations": 2, "strict_response_contract": True},
        )
        mocked_call.return_value = {"status": "ok", "content": "plain answer"}

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertIn("GAME response contract failed", session.error_detail)
        self.assertEqual(session.final_context["finish_reason"], "failed")
        self.assertEqual(session.final_context["failed_iteration"], 1)
        step_run = session.step_runs.get()
        self.assertEqual(step_run.status, ExecutionStepRun.Status.FAILED)
        self.assertIn("game_response_contract", step_run.request_payload)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_run_game_execution_session_strict_contract_requires_complete_true_to_finish(self, mocked_call):
        game_agent = AgentProfile.objects.create(
            name="game-agent-strict-finish",
            role="Autonomous goal runner",
            model_config=self.model,
            input_contract={"required": ["goal", "iteration", "memory", "game_response_contract"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        session = create_execution_session(
            source_object=self.source,
            entry_agent=game_agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Do not finish until complete is true.",
            runtime_config={"max_iterations": 2, "strict_response_contract": True},
        )
        mocked_call.side_effect = [
            {
                "status": "ok",
                "content": (
                    '{"action": "finish", "message": "Not actually done.", '
                    '"complete": false, "final_answer": ""}'
                ),
            },
            {
                "status": "ok",
                "content": (
                    '{"action": "finish", "message": "Done now.", '
                    '"complete": true, "final_answer": "Strict complete."}'
                ),
            },
        ]

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(session.step_runs.count(), 2)
        self.assertEqual(session.final_context["finish_reason"], "agent_finished")
        self.assertEqual(session.final_context["final_answer"], "Strict complete.")

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_run_game_execution_session_preserves_reserved_payload_keys_after_input_mapping(self, mocked_call):
        game_agent = AgentProfile.objects.create(
            name="game-agent-mapped-payload",
            role="Autonomous goal runner",
            model_config=self.model,
            input_contract={"required": ["source", "goal", "iteration", "memory", "game_response_contract"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        session = create_execution_session(
            source_object=self.source,
            entry_agent=game_agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Preserve internal runtime keys.",
            runtime_config={
                "max_iterations": 1,
                "strict_response_contract": True,
                "input_mapping": {"source": "source"},
            },
            initial_context={"source": "mapped source"},
        )
        mocked_call.return_value = {
            "status": "ok",
            "content": (
                '{"action": "finish", "message": "Done.", '
                '"complete": true, "final_answer": "Mapped complete."}'
            ),
        }

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        step_run = session.step_runs.get()
        self.assertEqual(step_run.request_payload["source"], "mapped source")
        self.assertEqual(step_run.request_payload["goal"], "Preserve internal runtime keys.")
        self.assertEqual(step_run.request_payload["iteration"], 1)
        self.assertEqual(step_run.request_payload["memory"], [])
        self.assertIn("game_response_contract", step_run.request_payload)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_run_game_execution_session_preserves_partial_context_on_failure(self, mocked_call):
        game_agent = AgentProfile.objects.create(
            name="game-agent-partial-failure",
            role="Autonomous goal runner",
            model_config=self.model,
            input_contract={"required": ["goal", "iteration", "memory"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        session = create_execution_session(
            source_object=self.source,
            entry_agent=game_agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Fail after useful work.",
            runtime_config={"max_iterations": 3},
        )
        mocked_call.side_effect = [
            {"status": "ok", "content": '{"action": "think", "message": "First useful step."}'},
            Exception("second iteration failed"),
        ]

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(session.final_context["finish_reason"], "failed")
        self.assertEqual(session.final_context["failed_iteration"], 2)
        self.assertEqual(session.final_context["last_error"], "second iteration failed")
        self.assertEqual(len(session.final_context["memory"]), 1)
        self.assertEqual(session.step_runs.count(), 2)
        self.assertEqual(session.step_runs.filter(status=ExecutionStepRun.Status.FAILED).count(), 1)

    def test_run_execution_session_rejects_session_that_is_already_running(self):
        session = create_execution_session(
            source_object=self.source,
            pipeline=self.pipeline,
            triggered_by=self.user,
            initial_context={"source": "plain reusable input"},
        )
        session.status = ExecutionSession.Status.RUNNING
        session.save(update_fields=["status"])

        with self.assertRaisesMessage(ValidationError, "already running"):
            run_execution_session(session.id)


class HubOllamaClientTests(TestCase):
    @patch("ai_hub.services.litellm_client.requests.post")
    def test_ollama_models_use_native_chat_api(self, mocked_post):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"message": {"content": "{\"ok\": true}"}}

        mocked_post.return_value = Response()

        result = completion_call(
            model="ollama/qwen3:8b",
            messages=[{"role": "user", "content": "hello"}],
            base_url="http://localhost:11434",
            timeout=30,
            temperature=0.2,
            max_tokens=128,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["content"], "{\"ok\": true}")
        mocked_post.assert_called_once()
        url = mocked_post.call_args.args[0]
        payload = mocked_post.call_args.kwargs["json"]
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["model"], "qwen3:8b")


class HubAdminControlCenterTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = get_user_model().objects.create_superuser(
            username="adminuser",
            password="testpass123",
        )
        self.provider = ProviderConfig.objects.create(
            name="Ollama LAN",
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://localhost:11434",
        )
        self.model = ModelConfig.objects.create(provider=self.provider, model_name="ollama/qwen3:8b")
        self.agent = AgentProfile.objects.create(
            name="visual-agent",
            role="Visual test agent",
            model_config=self.model,
            input_contract={"required": ["dream_id"]},
            output_contract={"required": ["agent"]},
        )
        self.pipeline = PipelineDefinition.objects.create(name="visual-pipeline", is_active=True)
        PipelineStep.objects.create(pipeline=self.pipeline, agent=self.agent, order=1)

    @patch("ai_hub.services.admin_control_center.requests.get")
    def test_control_center_graph_reflects_configured_pipeline(self, mocked_get):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"models": [{"name": "qwen3:8b"}]}

        mocked_get.return_value = Response()

        context = build_control_center_context()
        node_ids = {node["id"] for node in context["graph"]["nodes"]}
        edge_labels = {edge["label"] for edge in context["graph"]["edges"]}
        pipeline_scope = context["graph"]["pipelineScopes"][0]

        self.assertIn(f"provider:{self.provider.id}", node_ids)
        self.assertIn(f"model:{self.model.id}", node_ids)
        self.assertIn(f"agent:{self.agent.id}", node_ids)
        self.assertIn(f"pipeline:{self.pipeline.id}", node_ids)
        self.assertIn("calls", edge_labels)
        self.assertIn(f"pipeline:{self.pipeline.id}", pipeline_scope["node_ids"])
        self.assertIn(f"agent:{self.agent.id}", pipeline_scope["node_ids"])
        self.assertEqual(context["warnings"], [])

    @patch("ai_hub.services.admin_control_center.requests.get")
    def test_control_center_warns_about_missing_ollama_model(self, mocked_get):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"models": [{"name": "bge-m3:latest"}]}

        mocked_get.return_value = Response()

        context = build_control_center_context()

        self.assertIn("Model 'ollama/qwen3:8b' is configured but was not reported by Ollama.", context["warnings"])

    @patch("ai_hub.services.admin_control_center.requests.get")
    def test_control_center_handles_invalid_provider_json(self, mocked_get):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                raise ValueError("invalid json")

        mocked_get.return_value = Response()

        context = build_control_center_context()

        self.assertIn("Provider 'Ollama LAN': invalid json", context["warnings"])

    @patch("ai_hub.services.admin_control_center.requests.get")
    def test_control_center_caches_provider_health(self, mocked_get):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"models": [{"name": "qwen3:8b"}]}

        mocked_get.return_value = Response()

        build_control_center_context()
        build_control_center_context()

        mocked_get.assert_called_once()

    @patch("ai_hub.services.admin_control_center.requests.get")
    def test_control_center_admin_view_renders_for_staff(self, mocked_get):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"models": [{"name": "qwen3:8b"}]}

        mocked_get.return_value = Response()
        client = Client()
        client.force_login(self.user)

        response = client.get(reverse("admin:ai_hub_control_center"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AI Hub Control Center")

    @patch("ai_hub.services.admin_control_center.requests.get")
    def test_control_center_requires_pipeline_view_permission(self, mocked_get):
        staff_user = get_user_model().objects.create_user(
            username="staff-user",
            password="testpass123",
            is_staff=True,
        )
        client = Client()
        client.force_login(staff_user)

        response = client.get(reverse("admin:ai_hub_control_center"))

        self.assertEqual(response.status_code, 403)
        mocked_get.assert_not_called()

    def test_ai_hub_app_index_shows_two_workspaces(self):
        client = Client()
        client.force_login(self.user)

        response = client.get(reverse("admin:app_list", kwargs={"app_label": "ai_hub"}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Build AI workflows without touching code")
        self.assertContains(response, "Recommended next action")
        self.assertContains(response, "Setup checklist")
        self.assertContains(response, "Example blueprints")
        self.assertContains(response, "Open Orchestrator")
        self.assertContains(response, "Open GAME")
        self.assertContains(response, "Shared resources")

    def test_clean_workspace_urls_render(self):
        client = Client()
        client.force_login(self.user)

        orchestrator_response = client.get(reverse("admin:ai_hub_workspace_orchestrator"))
        game_response = client.get(reverse("admin:ai_hub_workspace_game"))

        self.assertEqual(orchestrator_response.status_code, 200)
        self.assertEqual(game_response.status_code, 200)
        self.assertContains(orchestrator_response, "Orchestrator workspace")
        self.assertContains(orchestrator_response, "How Orchestrator works")
        self.assertContains(game_response, "GAME workspace")
        self.assertContains(game_response, "How GAME works")
        self.assertContains(game_response, "GAME decision graph")

    def test_orchestrator_workspace_shows_pipelines(self):
        client = Client()
        client.force_login(self.user)

        response = client.get(reverse("admin:ai_hub_workspace_orchestrator"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Orchestrator workspace")
        self.assertContains(response, self.pipeline.name)
        self.assertContains(response, "Active pipelines")
        self.assertContains(response, "View orchestrator sessions")

    def test_game_workspace_shows_game_sessions(self):
        game_ready_agent = AgentProfile.objects.create(
            name="goal-runner",
            role="Autonomous GAME goal runner",
            model_config=self.model,
            input_contract={"required": ["goal", "iteration", "memory", "game_response_contract"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        create_execution_session(
            source_label="Visible GAME session",
            entry_agent=self.agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Show this in the GAME workspace.",
        )
        client = Client()
        client.force_login(self.user)

        response = client.get(reverse("admin:ai_hub_workspace_game"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "GAME workspace")
        self.assertContains(response, "Running / waiting")
        self.assertContains(response, "Visible GAME session")
        self.assertContains(response, self.agent.name)
        self.assertContains(response, game_ready_agent.name)
        self.assertContains(response, "GAME-ready")

    def test_agent_changelist_shows_workspace_usage(self):
        create_execution_session(
            source_label="Agent workspace marker",
            entry_agent=self.agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Mark this agent as GAME capable.",
        )
        client = Client()
        client.force_login(self.user)

        response = client.get(reverse("admin:ai_hub_agentprofile_changelist"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Workspace")
        self.assertContains(response, "Both")


class HubExecutionSessionEndpointTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="staffuser",
            password="testpass123",
            is_staff=True,
            is_superuser=True,
        )
        self.provider = ProviderConfig.objects.create(name="provider", provider_type="openai")
        self.model = ModelConfig.objects.create(provider=self.provider, model_name="model-a")
        self.agent = AgentProfile.objects.create(
            name="a1",
            role="extract",
            model_config=self.model,
            input_contract={"required": ["source"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        self.pipeline = PipelineDefinition.objects.create(
            name="pipe-ok",
            global_input_contract={"required": ["source"]},
            global_output_contract={"required": ["agent"]},
            is_active=True,
        )
        PipelineStep.objects.create(
            pipeline=self.pipeline,
            agent=self.agent,
            order=1,
            input_mapping={"source": "source"},
            output_mapping={"agent": "agent"},
        )
        self.session = create_execution_session(
            source_label="Generic source",
            pipeline=self.pipeline,
            entry_agent=self.agent,
            triggered_by=self.user,
            initial_context={"source": "hello"},
        )
        self.client = Client()
        self.client.force_login(self.user)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_staff_can_run_execution_session(self, mocked_call):
        mocked_call.return_value = {"status": "ok", "content": "done"}

        response = self.client.post(reverse("ai_hub:run_execution_session"), {"session_id": self.session.id})

        self.assertEqual(response.status_code, 200)
        self.session.refresh_from_db()
        self.assertEqual(response.json()["session_id"], self.session.id)
        self.assertEqual(response.json()["status"], ExecutionSession.Status.SUCCESS)
        self.assertEqual(self.session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(self.session.step_runs.count(), 1)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_staff_run_execution_session_reports_failed_result(self, mocked_call):
        failed_session = create_execution_session(
            source_label="Bad generic source",
            pipeline=self.pipeline,
            entry_agent=self.agent,
            triggered_by=self.user,
            initial_context={},
        )
        mocked_call.return_value = {"status": "ok", "content": "done"}

        response = self.client.post(reverse("ai_hub:run_execution_session"), {"session_id": failed_session.id})

        self.assertEqual(response.status_code, 400)
        failed_session.refresh_from_db()
        self.assertEqual(response.json()["session_id"], failed_session.id)
        self.assertEqual(response.json()["status"], ExecutionSession.Status.FAILED)
        self.assertIn("source", response.json()["error"])
        self.assertEqual(failed_session.status, ExecutionSession.Status.FAILED)

    def test_non_staff_cannot_run_execution_session(self):
        non_staff = get_user_model().objects.create_user(
            username="nonstaffuser",
            password="testpass123",
        )
        self.client.force_login(non_staff)

        response = self.client.post(reverse("ai_hub:run_execution_session"), {"session_id": self.session.id})

        self.assertEqual(response.status_code, 302)

    @patch("ai_hub.admin.run_execution_session")
    def test_admin_action_can_launch_selected_sessions(self, mocked_run):
        response = self.client.post(
            reverse("admin:ai_hub_executionsession_changelist"),
            {
                "action": "run_selected_sessions",
                "_selected_action": [self.session.id],
                "index": 0,
            },
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        mocked_run.assert_called_once_with(self.session.id)

    def test_admin_game_session_create_view_renders_for_staff(self):
        response = self.client.get(reverse("admin:ai_hub_executionsession_game_new"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Create GAME session")
        self.assertContains(response, "Entry agent")

    def test_admin_game_session_create_view_creates_game_session(self):
        response = self.client.post(
            reverse("admin:ai_hub_executionsession_game_new"),
            {
                "entry_agent": self.agent.id,
                "goal_text": "Plan a reusable AI workflow.",
                "max_iterations": 4,
                "runtime_mode": ExecutionSession.RuntimeMode.ASYNC,
                "strict_response_contract": "on",
                "source_label": "GAME smoke test",
                "initial_context": '{"topic": "workflow"}',
            },
        )

        session = ExecutionSession.objects.get(source_label="GAME smoke test")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(session.runtime_kind, ExecutionSession.RuntimeKind.GAME)
        self.assertEqual(session.entry_agent, self.agent)
        self.assertEqual(session.triggered_by, self.user)
        self.assertEqual(session.goal_text, "Plan a reusable AI workflow.")
        self.assertEqual(session.runtime_config, {"max_iterations": 4, "strict_response_contract": True})
        self.assertEqual(session.initial_context, {"topic": "workflow"})

    def test_admin_game_session_create_view_validates_initial_context_json(self):
        response = self.client.post(
            reverse("admin:ai_hub_executionsession_game_new"),
            {
                "entry_agent": self.agent.id,
                "goal_text": "Plan a reusable AI workflow.",
                "max_iterations": 4,
                "runtime_mode": ExecutionSession.RuntimeMode.ASYNC,
                "strict_response_contract": "on",
                "source_label": "Invalid GAME",
                "initial_context": "[1, 2, 3]",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Initial context must be a JSON object.")
        self.assertFalse(ExecutionSession.objects.filter(source_label="Invalid GAME").exists())

    def test_admin_execution_session_change_view_renders_timeline(self):
        game_session = create_execution_session(
            source_label="Timeline GAME",
            entry_agent=self.agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Summarize the timeline.",
            runtime_config={"max_iterations": 1},
            initial_context={"source": "hello"},
        )
        game_session.status = ExecutionSession.Status.SUCCESS
        game_session.final_context = {
            "finish_reason": "agent_finished",
            "final_answer": "Timeline complete.",
        }
        game_session.save(update_fields=["status", "final_context"])
        ExecutionStepRun.objects.create(
            session=game_session,
            order=1,
            agent=self.agent,
            action_name="game_iteration",
            status=ExecutionStepRun.Status.SUCCESS,
            latency_ms=25,
            response_payload={"llm": {"content": '{"action": "finish"}'}},
            observation_payload={
                "action": "finish",
                "complete": True,
                "decision": {"message": "Timeline complete."},
            },
        )

        response = self.client.get(reverse("admin:ai_hub_executionsession_change", args=[game_session.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Execution timeline")
        self.assertContains(response, "Timeline complete.")
        self.assertContains(response, "game_iteration")
        self.assertContains(response, "25 ms")


