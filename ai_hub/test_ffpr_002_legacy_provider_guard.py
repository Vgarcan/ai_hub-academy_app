from unittest.mock import patch

from django.test import TestCase, override_settings

from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    ExecutionStepRun,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    ProviderConfig,
    ToolDefinition,
    ToolExecutionRun,
)
from ai_hub.services.execution_runner import run_execution_session


CALLABLE_PATH = "ai_hub.test_ffpr_002_legacy_provider_guard.record_side_effect"
SIDE_EFFECTS = []


def record_side_effect(payload, config):
    SIDE_EFFECTS.append(
        {
            "payload": dict(payload or {}),
            "marker": (config or {}).get("marker"),
        }
    )
    return {"effect": "recorded"}


@override_settings(AI_HUB_ALLOWED_TOOL_CALLABLES=(CALLABLE_PATH,))
class FFPR002LegacyProviderGuardTests(TestCase):
    def setUp(self):
        SIDE_EFFECTS.clear()
        self.provider = ProviderConfig.objects.create(
            name="ffpr-002-provider",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        self.model = ModelConfig.objects.create(
            provider=self.provider,
            model_name="training/ffpr-002",
        )
        self.agent = AgentProfile.objects.create(
            name="ffpr-002-agent",
            role="Legacy provider guard regression",
            model_config=self.model,
            input_contract={"required": []},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        self.tool = ToolDefinition.objects.create(
            name="ffpr-002-side-effect",
            tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
            operation_mode=ToolDefinition.OperationMode.EXTERNAL_WRITE,
            config={
                "callable": CALLABLE_PATH,
                "marker": "synthetic-external-write",
            },
        )
        self.agent.tools.add(self.tool)
        self.pipeline = PipelineDefinition.objects.create(
            name="ffpr-002-pipeline",
            entry_agent=self.agent,
            is_active=True,
        )
        PipelineStep.objects.create(
            pipeline=self.pipeline,
            agent=self.agent,
            order=1,
        )

    def _session(self):
        return ExecutionSession.objects.create(
            pipeline=self.pipeline,
            runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            runtime_config={"agent_tool_runtime": "legacy_preexecute"},
            initial_context={"probe": "ffpr-002"},
        )

    def _assert_inactive_chain_fails_before_side_effect(self, session, mocked_call):
        run_execution_session(session.pk)

        session.refresh_from_db()
        step = session.step_runs.get()
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(step.status, ExecutionStepRun.Status.FAILED)
        self.assertIn("Model/provider is inactive", session.error_detail)
        self.assertIn("Model/provider is inactive", step.error_detail)
        self.assertEqual(SIDE_EFFECTS, [])
        mocked_call.assert_not_called()
        self.assertEqual(ToolExecutionRun.objects.filter(session=session).count(), 0)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_ffpr_002_a_inactive_provider_blocks_tool_side_effect(self, mocked_call):
        session = self._session()
        ProviderConfig.objects.filter(pk=self.provider.pk).update(is_active=False)

        self._assert_inactive_chain_fails_before_side_effect(session, mocked_call)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_ffpr_002_b_inactive_model_blocks_tool_side_effect(self, mocked_call):
        session = self._session()
        ModelConfig.objects.filter(pk=self.model.pk).update(is_active=False)

        self._assert_inactive_chain_fails_before_side_effect(session, mocked_call)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_ffpr_002_c_active_legacy_behavior_is_preserved(self, mocked_call):
        session = self._session()
        mocked_call.return_value = {"status": "ok", "content": "active"}

        run_execution_session(session.pk)

        session.refresh_from_db()
        step = session.step_runs.get()
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(step.status, ExecutionStepRun.Status.SUCCESS)
        self.assertEqual(len(SIDE_EFFECTS), 1)
        self.assertEqual(SIDE_EFFECTS[0]["marker"], "synthetic-external-write")
        mocked_call.assert_called_once()
        self.assertEqual(
            step.response_payload["tools"],
            {"ffpr-002-side-effect": {"effect": "recorded"}},
        )
