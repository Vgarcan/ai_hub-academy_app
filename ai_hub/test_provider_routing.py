from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from django.core.exceptions import ValidationError
from django.test import SimpleTestCase, TestCase

from ai_hub.models import AgentProfile, ModelConfig, ProviderConfig
from ai_hub.services import litellm_client
from ai_hub.services.agent_runtime import execute_agent
from ai_hub.services.health import STATUS_OK, evaluate_provider
from ai_hub.services.provider_registry import resolve_model_config


class ProviderTypeRoutingTests(SimpleTestCase):
    def _completion(self, *, provider_type, model, base_url="https://provider.example/v1"):
        return litellm_client.completion_call(
            provider_type=provider_type,
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            base_url=base_url,
            timeout=30,
            temperature=0.2,
            max_tokens=128,
        )

    @patch("ai_hub.services.litellm_client._ollama_chat_call")
    def test_ollama_provider_type_selects_native_adapter_without_name_or_port_hint(
        self, mocked_ollama
    ):
        mocked_ollama.return_value = {"status": "ok", "content": "ollama"}

        result = self._completion(
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            model="qwen3:8b",
            base_url="http://ollama.example:9999",
        )

        self.assertEqual(result["content"], "ollama")
        mocked_ollama.assert_called_once()

    @patch("ai_hub.services.litellm_client._training_completion_call")
    @patch("ai_hub.services.litellm_client._ollama_chat_call")
    def test_ollama_port_does_not_override_non_ollama_provider_type(
        self, mocked_ollama, mocked_training
    ):
        result, completion = self._cloud_completion(
            provider_type=ProviderConfig.ProviderType.OPENAI,
            model="gpt-test",
            base_url="https://compatible.example:11434/v1",
        )

        self.assertEqual(result["content"], "cloud")
        completion.assert_called_once()
        mocked_ollama.assert_not_called()
        mocked_training.assert_not_called()

    @patch("ai_hub.services.litellm_client._training_completion_call")
    @patch("ai_hub.services.litellm_client._ollama_chat_call")
    def test_ollama_model_prefix_does_not_override_non_ollama_provider_type(
        self, mocked_ollama, mocked_training
    ):
        result, completion = self._cloud_completion(
            provider_type=ProviderConfig.ProviderType.OPENAI,
            model="ollama/not-an-ollama-route",
        )

        self.assertEqual(result["content"], "cloud")
        completion.assert_called_once()
        mocked_ollama.assert_not_called()
        mocked_training.assert_not_called()

    @patch("ai_hub.services.litellm_client._training_completion_call")
    @patch("ai_hub.services.litellm_client._ollama_chat_call")
    def test_training_model_name_does_not_override_non_training_provider_type(
        self, mocked_ollama, mocked_training
    ):
        result, completion = self._cloud_completion(
            provider_type=ProviderConfig.ProviderType.OPENAI,
            model="training",
        )

        self.assertEqual(result["content"], "cloud")
        completion.assert_called_once()
        mocked_ollama.assert_not_called()
        mocked_training.assert_not_called()

    @patch("ai_hub.services.litellm_client._training_completion_call")
    @patch("ai_hub.services.litellm_client._ollama_chat_call")
    def test_training_provider_type_selects_training_adapter(
        self, mocked_ollama, mocked_training
    ):
        mocked_training.return_value = {
            "status": "ok",
            "content": "training",
            "stubbed": True,
        }

        result = self._completion(
            provider_type=ProviderConfig.ProviderType.TRAINING,
            model="training/assistant",
            base_url="",
        )

        self.assertEqual(result["content"], "training")
        mocked_training.assert_called_once()
        mocked_ollama.assert_not_called()

    @patch("ai_hub.services.litellm_client._training_completion_call")
    @patch("ai_hub.services.litellm_client._ollama_chat_call")
    def test_representative_cloud_provider_stays_on_litellm(
        self, mocked_ollama, mocked_training
    ):
        result, completion = self._cloud_completion(
            provider_type=ProviderConfig.ProviderType.ANTHROPIC,
            model="anthropic/test-model",
        )

        self.assertEqual(result["content"], "cloud")
        completion.assert_called_once()
        mocked_ollama.assert_not_called()
        mocked_training.assert_not_called()

    def _cloud_completion(self, **kwargs):
        completion = Mock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="cloud"))],
                model_dump=lambda: {"provider": "cloud"},
            )
        )
        fake_litellm = SimpleNamespace(completion=completion)
        with patch.object(litellm_client, "litellm", fake_litellm):
            result = self._completion(**kwargs)
        return result, completion


class ProviderTypePropagationTests(TestCase):
    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_agent_runtime_passes_resolved_provider_type_to_completion_boundary(
        self, mocked_completion
    ):
        provider = ProviderConfig.objects.create(
            name="Routing Ollama",
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://ollama.example:9999",
        )
        model = ModelConfig.objects.create(provider=provider, model_name="qwen3:8b")
        agent = AgentProfile.objects.create(
            name="Routing agent",
            role="Routing regression",
            model_config=model,
        )
        mocked_completion.return_value = {"status": "ok", "content": "done"}

        execute_agent(agent, {"prompt": "hello"})

        self.assertEqual(
            mocked_completion.call_args.kwargs["provider_type"],
            ProviderConfig.ProviderType.OLLAMA,
        )

    def test_training_provider_health_does_not_require_external_credentials(self):
        provider = ProviderConfig.objects.create(
            name="Healthy Training",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        ModelConfig.objects.create(provider=provider, model_name="training")

        result = evaluate_provider(provider)

        self.assertEqual(result.status, STATUS_OK)
        self.assertTrue(all(check.ok for check in result.checks))

    def test_resolver_rejects_provider_type_outside_model_choices(self):
        provider = ProviderConfig.objects.create(
            name="Invalid provider type",
            provider_type="surprise",
        )
        model = ModelConfig.objects.create(provider=provider, model_name="model")

        with self.assertRaisesMessage(ValidationError, "Unsupported provider type"):
            resolve_model_config(model)


class ProviderFailureSemanticsTests(SimpleTestCase):
    def _ollama_completion(self, *, base_url="http://ollama.example:9999", model="qwen3:8b"):
        return litellm_client.completion_call(
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            base_url=base_url,
            timeout=1,
        )

    @patch("ai_hub.services.litellm_client.requests.post")
    def test_ollama_requires_an_explicit_base_url(self, mocked_post):
        with self.assertRaises(litellm_client.ProviderExecutionError) as raised:
            self._ollama_completion(base_url="")

        self.assertEqual(raised.exception.category, "invalid_provider_configuration")
        mocked_post.assert_not_called()

    @patch(
        "ai_hub.services.litellm_client.requests.post",
        side_effect=requests.ConnectionError("connection refused"),
    )
    def test_ollama_connection_failure_is_classified_as_unreachable(self, mocked_post):
        with self.assertRaises(litellm_client.ProviderExecutionError) as raised:
            self._ollama_completion()

        self.assertEqual(raised.exception.category, "provider_unreachable")
        mocked_post.assert_called_once()

    @patch("ai_hub.services.litellm_client.requests.post")
    def test_ollama_missing_model_error_is_classified(self, mocked_post):
        response = Mock(status_code=404)
        response.json.return_value = {"error": "model 'missing:latest' not found"}
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
        mocked_post.return_value = response

        with self.assertRaises(litellm_client.ProviderExecutionError) as raised:
            self._ollama_completion(model="missing:latest")

        self.assertEqual(raised.exception.category, "model_not_found")

    @patch("ai_hub.services.litellm_client.requests.post")
    def test_ollama_http_error_is_classified_as_provider_error(self, mocked_post):
        response = Mock(status_code=500)
        response.json.return_value = {"error": "internal server error"}
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
        mocked_post.return_value = response

        with self.assertRaises(litellm_client.ProviderExecutionError) as raised:
            self._ollama_completion()

        self.assertEqual(raised.exception.category, "provider_returned_error")

    @patch("ai_hub.services.litellm_client.requests.post")
    def test_ollama_invalid_json_is_classified_as_invalid_response(self, mocked_post):
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("invalid json")
        mocked_post.return_value = response

        with self.assertRaises(litellm_client.ProviderExecutionError) as raised:
            self._ollama_completion()

        self.assertEqual(raised.exception.category, "invalid_provider_response")

    def test_missing_litellm_dependency_is_an_error_not_a_fake_success(self):
        with patch.object(litellm_client, "litellm", None):
            with self.assertRaises(litellm_client.ProviderExecutionError) as raised:
                litellm_client.completion_call(
                    provider_type=ProviderConfig.ProviderType.OPENAI,
                    model="gpt-test",
                    messages=[{"role": "user", "content": "hello"}],
                )

        self.assertEqual(raised.exception.category, "invalid_provider_configuration")

    def test_completion_boundary_rejects_unknown_provider_type(self):
        with self.assertRaises(litellm_client.ProviderExecutionError) as raised:
            litellm_client.completion_call(
                provider_type="surprise",
                model="model",
                messages=[{"role": "user", "content": "hello"}],
            )

        self.assertEqual(raised.exception.category, "invalid_provider_configuration")
