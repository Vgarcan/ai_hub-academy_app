from unittest.mock import patch

from django.core.exceptions import ValidationError
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
from ai_hub.services.litellm_client import ProviderExecutionError
from ai_hub.test_application_scope_helpers import test_scope


CALLABLE_PATH = "ai_hub.test_ffpr_003_legacy_tool_audit.record_side_effect"
SIDE_EFFECTS = []


def record_side_effect(payload, config):
    invocation = {
        "payload": dict(payload or {}),
        "marker": (config or {}).get("marker"),
    }
    SIDE_EFFECTS.append(invocation)
    if (config or {}).get("fail"):
        raise ValidationError("synthetic legacy Tool failure")
    return {"effect": "completed", "marker": invocation["marker"]}


@override_settings(AI_HUB_ALLOWED_TOOL_CALLABLES=(CALLABLE_PATH,))
class FFPR003LegacyToolAuditTests(TestCase):
    def setUp(self):
        SIDE_EFFECTS.clear()

    def _runtime(
        self,
        suffix,
        *,
        tool_fails=False,
        requires_approval=False,
    ):
        provider = ProviderConfig.objects.create(
            name=f"ffpr-003-provider-{suffix}",
            provider_type=ProviderConfig.ProviderType.OPENAI,
        )
        model = ModelConfig.objects.create(
            provider=provider,
            model_name=f"ffpr-003-model-{suffix}",
        )
        agent = AgentProfile.objects.create(
            application_scope=test_scope(),
            name=f"ffpr-003-agent-{suffix}",
            role="Legacy Tool audit regression",
            model_config=model,
            input_contract={"required": []},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        tool = ToolDefinition.objects.create(
            name=f"ffpr-003-tool-{suffix}",
            tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
            operation_mode=ToolDefinition.OperationMode.EXTERNAL_WRITE,
            requires_approval=requires_approval,
            config={
                "callable": CALLABLE_PATH,
                "marker": suffix,
                "fail": tool_fails,
            },
        )
        agent.tools.add(tool)
        pipeline = PipelineDefinition.objects.create(
            name=f"ffpr-003-pipeline-{suffix}",
            entry_agent=agent,
            is_active=True,
        )
        PipelineStep.objects.create(
            pipeline=pipeline,
            agent=agent,
            order=1,
        )
        session = ExecutionSession.objects.create(
            pipeline=pipeline,
            runtime_kind=ExecutionSession.RuntimeKind.ORCHESTRATOR,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            runtime_config={"agent_tool_runtime": "legacy_preexecute"},
            initial_context={"probe": suffix},
        )
        return provider, model, agent, tool, session

    @patch(
        "ai_hub.services.agent_runtime.completion_call",
        side_effect=ProviderExecutionError(
            "provider_unreachable",
            "synthetic provider failure after Tool success",
        ),
    )
    def test_ffpr_003_a_tool_success_survives_provider_failure_in_audit(
        self,
        mocked_call,
    ):
        _provider, _model, agent, tool, session = self._runtime("provider-failure")

        run_execution_session(session.pk)

        session.refresh_from_db()
        step = session.step_runs.get()
        tool_run = ToolExecutionRun.objects.get(session=session)
        self.assertEqual(len(SIDE_EFFECTS), 1)
        mocked_call.assert_called_once()
        self.assertEqual(tool_run.status, ToolExecutionRun.Status.SUCCESS)
        self.assertEqual(tool_run.session, session)
        self.assertEqual(tool_run.step_run, step)
        self.assertEqual(tool_run.agent, agent)
        self.assertEqual(tool_run.tool, tool)
        self.assertEqual(tool_run.input_payload, SIDE_EFFECTS[0]["payload"])
        self.assertEqual(
            tool_run.output_payload,
            {"effect": "completed", "marker": "provider-failure"},
        )
        self.assertIsNotNone(tool_run.started_at)
        self.assertIsNotNone(tool_run.finished_at)
        self.assertIsNotNone(tool_run.latency_ms)
        self.assertEqual(
            tool_run.approval_state,
            ToolExecutionRun.ApprovalState.NOT_REQUIRED,
        )
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(step.status, ExecutionStepRun.Status.FAILED)
        self.assertIn("provider_unreachable", session.error_detail)
        self.assertIn("provider_unreachable", step.error_detail)
        self.assertEqual(step.response_payload, {})

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_ffpr_003_b_tool_failure_is_persisted_before_provider(
        self,
        mocked_call,
    ):
        _provider, _model, agent, tool, session = self._runtime(
            "tool-failure",
            tool_fails=True,
        )

        run_execution_session(session.pk)

        session.refresh_from_db()
        step = session.step_runs.get()
        tool_run = ToolExecutionRun.objects.get(session=session)
        self.assertEqual(len(SIDE_EFFECTS), 1)
        mocked_call.assert_not_called()
        self.assertEqual(tool_run.status, ToolExecutionRun.Status.FAILED)
        self.assertEqual(tool_run.session, session)
        self.assertEqual(tool_run.step_run, step)
        self.assertEqual(tool_run.agent, agent)
        self.assertEqual(tool_run.tool, tool)
        self.assertEqual(tool_run.input_payload, SIDE_EFFECTS[0]["payload"])
        self.assertIn("synthetic legacy Tool failure", tool_run.error_detail)
        self.assertIsNotNone(tool_run.started_at)
        self.assertIsNotNone(tool_run.finished_at)
        self.assertIsNotNone(tool_run.latency_ms)
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(step.status, ExecutionStepRun.Status.FAILED)
        self.assertIn("synthetic legacy Tool failure", step.error_detail)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_ffpr_003_c_successful_legacy_execution_is_preserved(
        self,
        mocked_call,
    ):
        _provider, _model, agent, tool, session = self._runtime("success")
        mocked_call.return_value = {"status": "ok", "content": "legacy complete"}

        run_execution_session(session.pk)

        session.refresh_from_db()
        step = session.step_runs.get()
        tool_run = ToolExecutionRun.objects.get(session=session)
        self.assertEqual(len(SIDE_EFFECTS), 1)
        mocked_call.assert_called_once()
        self.assertEqual(tool_run.status, ToolExecutionRun.Status.SUCCESS)
        self.assertEqual(tool_run.agent, agent)
        self.assertEqual(tool_run.tool, tool)
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(step.status, ExecutionStepRun.Status.SUCCESS)
        self.assertEqual(
            step.response_payload["tools"],
            {
                "ffpr-003-tool-success": {
                    "effect": "completed",
                    "marker": "success",
                }
            },
        )

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_ffpr_003_d_inactive_provider_or_model_still_blocks_tool_and_audit(
        self,
        mocked_call,
    ):
        for inactive_target in ("provider", "model"):
            with self.subTest(inactive_target=inactive_target):
                provider, model, _agent, _tool, session = self._runtime(
                    f"inactive-{inactive_target}"
                )
                if inactive_target == "provider":
                    ProviderConfig.objects.filter(pk=provider.pk).update(
                        is_active=False
                    )
                else:
                    ModelConfig.objects.filter(pk=model.pk).update(
                        is_active=False
                    )

                run_execution_session(session.pk)

                session.refresh_from_db()
                step = session.step_runs.get()
                self.assertEqual(session.status, ExecutionSession.Status.FAILED)
                self.assertEqual(step.status, ExecutionStepRun.Status.FAILED)
                self.assertIn("Model/provider is inactive", step.error_detail)
                self.assertEqual(
                    ToolExecutionRun.objects.filter(session=session).count(),
                    0,
                )
        self.assertEqual(SIDE_EFFECTS, [])
        mocked_call.assert_not_called()

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_ffpr_003_e_approval_required_tool_remains_blocked(
        self,
        mocked_call,
    ):
        _provider, _model, _agent, _tool, session = self._runtime(
            "approval-required",
            requires_approval=True,
        )
        mocked_call.return_value = {"status": "ok", "content": "no Tool executed"}

        run_execution_session(session.pk)

        session.refresh_from_db()
        step = session.step_runs.get()
        self.assertEqual(SIDE_EFFECTS, [])
        mocked_call.assert_called_once()
        self.assertEqual(
            ToolExecutionRun.objects.filter(session=session).count(),
            0,
        )
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(step.status, ExecutionStepRun.Status.SUCCESS)
        self.assertEqual(step.response_payload["tools"], {})
