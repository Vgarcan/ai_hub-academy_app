import json
import os
from unittest import skipUnless
from unittest.mock import patch

import requests
from django.test import TestCase

from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    ExecutionStepRun,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    ProviderConfig,
)
from ai_hub.services.agent_runtime import execute_agent
from ai_hub.services.execution_runner import run_execution_session
from ai_hub.services.execution_sessions import create_execution_session
from ai_hub.services import litellm_client


LIVE_BASE_URL = os.getenv("AI_HUB_LIVE_OLLAMA_BASE_URL", "").strip()
LIVE_MODEL = os.getenv("AI_HUB_LIVE_OLLAMA_MODEL", "").strip()
LIVE_OLLAMA_CONFIGURED = bool(LIVE_BASE_URL and LIVE_MODEL)
LIVE_SKIP_REASON = (
    "Set AI_HUB_LIVE_OLLAMA_BASE_URL and AI_HUB_LIVE_OLLAMA_MODEL "
    "to run the opt-in live Ollama integration tests."
)


@skipUnless(LIVE_OLLAMA_CONFIGURED, LIVE_SKIP_REASON)
class LiveOllamaIntegrationTests(TestCase):
    def setUp(self):
        self.provider = ProviderConfig.objects.create(
            name="Live Ollama integration",
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url=LIVE_BASE_URL,
            default_timeout=120,
        )
        self.model = ModelConfig.objects.create(
            provider=self.provider,
            model_name=LIVE_MODEL,
            temperature_default=0,
            max_tokens_default=64,
        )
        self.agent = AgentProfile.objects.create(
            name="Live Ollama foundation agent",
            role="Provider integration probe",
            model_config=self.model,
            system_prompt=(
                "Answer briefly. When a tool response contract is present, "
                "return a final decision and do not call a tool."
            ),
            input_contract={"required": ["prompt"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )

    def test_live_direct_agent_uses_resolved_ollama_provider(self):
        real_post = requests.post
        with patch(
            "ai_hub.services.litellm_client.requests.post",
            wraps=real_post,
        ) as observed_post:
            result = execute_agent(
                self.agent,
                {"prompt": "Reply with the short phrase AI Hub Ollama connected."},
            )

        self.assertEqual(result["agent"], self.agent.name)
        self.assertEqual(result["tools"], {})
        self.assertEqual(result["llm"]["status"], "ok")
        self.assertEqual(
            result["llm"]["provider"],
            ProviderConfig.ProviderType.OLLAMA,
        )
        self.assertEqual(
            result["llm"]["provider_model"],
            LIVE_MODEL.removeprefix("ollama/"),
        )
        self.assertTrue(result["llm"]["content"].strip())
        self.assertEqual(
            observed_post.call_args.args[0],
            f"{LIVE_BASE_URL.rstrip('/')}/api/chat",
        )

    def test_live_one_step_orchestrator_persists_successful_ollama_output(self):
        pipeline = PipelineDefinition.objects.create(
            name="Live Ollama one-step pipeline",
            global_input_contract={"required": ["prompt"]},
            global_output_contract={"required": ["live_output"]},
        )
        PipelineStep.objects.create(
            pipeline=pipeline,
            agent=self.agent,
            order=1,
            input_mapping={"prompt": "prompt"},
            output_mapping={"live_output": "llm.content"},
        )
        pipeline.is_active = True
        pipeline.save(update_fields=["is_active"])
        session = create_execution_session(
            pipeline=pipeline,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            initial_context={
                "prompt": "Finish with one short confirmation that Ollama is connected."
            },
        )

        real_post = requests.post
        with patch(
            "ai_hub.services.litellm_client.requests.post",
            wraps=real_post,
        ) as observed_post:
            run_execution_session(session.pk)

        session.refresh_from_db()
        step_run = session.step_runs.get()
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(step_run.status, ExecutionStepRun.Status.SUCCESS)
        self.assertTrue(session.final_context["live_output"].strip())
        self.assertEqual(
            step_run.response_payload["llm"]["provider"],
            ProviderConfig.ProviderType.OLLAMA,
        )
        self.assertEqual(
            step_run.response_payload["llm"]["provider_model"],
            LIVE_MODEL.removeprefix("ollama/"),
        )
        self.assertEqual(
            observed_post.call_args.args[0],
            f"{LIVE_BASE_URL.rstrip('/')}/api/chat",
        )
        self.assertFalse(self.provider.api_key_env_var)
        self.assertNotIn("api_key", json.dumps(step_run.response_payload).lower())

    def test_controlled_unreachable_ollama_does_not_fall_back_to_litellm(self):
        unreachable_provider = ProviderConfig.objects.create(
            name="Controlled unavailable Ollama",
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://127.0.0.1:1",
            default_timeout=2,
        )
        unreachable_model = ModelConfig.objects.create(
            provider=unreachable_provider,
            model_name=LIVE_MODEL,
            temperature_default=0,
            max_tokens_default=16,
        )
        unreachable_agent = AgentProfile.objects.create(
            name="Controlled failure agent",
            role="Provider failure probe",
            model_config=unreachable_model,
        )

        with patch.object(litellm_client.litellm, "completion") as litellm_completion:
            with self.assertRaises(litellm_client.ProviderExecutionError) as raised:
                execute_agent(unreachable_agent, {"prompt": "This request must fail."})

        self.assertEqual(raised.exception.category, "provider_unreachable")
        litellm_completion.assert_not_called()
