"""
Tests for the dashboard app.

All views are read-only, so we test:
- HTTP 200 responses
- Correct templates
- Status filter validation
"""
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse

from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    ExecutionStepRun,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    ProviderConfig,
    ToolDefinition,
)
from ai_hub.test_application_scope_helpers import test_scope

User = get_user_model()


def _make_provider():
    return ProviderConfig.objects.create(
        name="Test Provider",
        provider_type=ProviderConfig.ProviderType.TRAINING,
        is_active=True,
    )


def _make_model(provider):
    return ModelConfig.objects.create(
        provider=provider,
        model_name="training/test",
        temperature_default="0.5",
        max_tokens_default=100,
        is_active=True,
    )


def _make_agent(model):
    return AgentProfile.objects.create(
        application_scope=test_scope(),
        name="Test Agent",
        role="tester",
        system_prompt="Respond with valid GAME JSON.",
        model_config=model,
        is_active=True,
    )


def _make_session(agent):
    return ExecutionSession.objects.create(
        runtime_kind=ExecutionSession.RuntimeKind.GAME,
        entry_agent=agent,
        goal_text="Test goal",
        status=ExecutionSession.Status.SUCCESS,
        source_label="test",
    )


class DashboardViewTests(TestCase):
    def setUp(self):
        self.staff_user = User.objects.create_user(
            username="dashboard-staff",
            password="testpass123",
            is_staff=True,
            is_superuser=True,
        )
        self.client.force_login(self.staff_user)
        self.provider = _make_provider()
        self.model = _make_model(self.provider)
        self.agent = _make_agent(self.model)
        self.session = _make_session(self.agent)

    def test_anonymous_user_cannot_view_session_payloads(self):
        self.client.logout()

        response = self.client.get(reverse("dashboard_session_detail", args=[self.session.pk]))

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login/", response.url)

    def test_staff_without_view_permission_cannot_view_session_payloads(self):
        restricted_staff = User.objects.create_user(
            username="restricted-dashboard-staff",
            password="testpass123",
            is_staff=True,
        )
        self.client.force_login(restricted_staff)

        response = self.client.get(reverse("dashboard_session_detail", args=[self.session.pk]))

        self.assertEqual(response.status_code, 403)

    # ── List views ────────────────────────────────────────────────────────────

    def test_overview_200(self):
        r = self.client.get(reverse("dashboard_overview"))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Control Room")

    def test_providers_200(self):
        r = self.client.get(reverse("dashboard_providers"))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Test Provider")

    def test_agents_200(self):
        r = self.client.get(reverse("dashboard_agents"))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Test Agent")

    def test_pipelines_200(self):
        r = self.client.get(reverse("dashboard_pipelines"))
        self.assertEqual(r.status_code, 200)

    def test_sessions_200(self):
        r = self.client.get(reverse("dashboard_sessions"))
        self.assertEqual(r.status_code, 200)
        # Sessions list shows agent name and source_label, not goal_text
        self.assertContains(r, "Test Agent")
        self.assertContains(r, "test")

    # ── Detail views ──────────────────────────────────────────────────────────

    def test_provider_detail_200(self):
        r = self.client.get(reverse("dashboard_provider_detail", args=[self.provider.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Test Provider")

    def test_agent_detail_200(self):
        r = self.client.get(reverse("dashboard_agent_detail", args=[self.agent.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Test Agent")

    def test_session_detail_200(self):
        r = self.client.get(reverse("dashboard_session_detail", args=[self.session.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Test goal")

    def test_session_detail_redacts_sensitive_runtime_payloads(self):
        self.session.initial_context = {
            "api_key": "initial-secret",
            "safe": "visible",
        }
        self.session.error_detail = "password=error-secret"
        self.session.save(update_fields=["initial_context", "error_detail"])
        ExecutionStepRun.objects.create(
            session=self.session,
            order=1,
            agent=self.agent,
            status=ExecutionStepRun.Status.SUCCESS,
            request_payload={"access_token": "request-secret"},
            response_payload={
                "tools": {"lookup": {"private_key": "tool-secret"}},
                "llm": {
                    "content": (
                        '{"answer": "ok", "authorization": "Bearer llm-secret"}'
                    )
                },
            },
        )

        response = self.client.get(
            reverse("dashboard_session_detail", args=[self.session.pk])
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "***REDACTED***")
        self.assertContains(response, "visible")
        for secret in (
            "initial-secret",
            "error-secret",
            "request-secret",
            "tool-secret",
            "llm-secret",
        ):
            self.assertNotContains(response, secret)

    def test_pipeline_detail_200(self):
        pipeline = PipelineDefinition.objects.create(name="Test Pipeline")
        r = self.client.get(reverse("dashboard_pipeline_detail", args=[pipeline.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Test Pipeline")

    def test_detail_404_on_missing(self):
        r = self.client.get(reverse("dashboard_session_detail", args=[99999]))
        self.assertEqual(r.status_code, 404)

    # ── Status filter validation ───────────────────────────────────────────────

    def test_sessions_valid_status_filter(self):
        r = self.client.get(reverse("dashboard_sessions") + "?status=success")
        self.assertEqual(r.status_code, 200)
        # source_label is shown in the sessions table
        self.assertContains(r, "test")

    def test_sessions_failed_filter_hides_success_session(self):
        r = self.client.get(reverse("dashboard_sessions") + "?status=failed")
        self.assertEqual(r.status_code, 200)
        # Our session is 'success', so it should not appear in 'failed' filter
        self.assertNotContains(r, f'href="/dashboard/sessions/{self.session.pk}/"')

    def test_sessions_invalid_status_filter_ignored(self):
        raw = "<script>alert(1)</script>"
        r = self.client.get(reverse("dashboard_sessions") + f"?status={raw}")
        self.assertEqual(r.status_code, 200)
        # Filter is silently ignored — no 500, no XSS reflection
        self.assertNotContains(r, "<script>alert(1)</script>")

    def test_sessions_unknown_status_returns_all(self):
        r = self.client.get(reverse("dashboard_sessions") + "?status=unknown_value")
        self.assertEqual(r.status_code, 200)
        # Invalid status is ignored, session still visible
        self.assertContains(r, f'href="/dashboard/sessions/{self.session.pk}/"')

    @override_settings(AI_HUB_PROVIDER_HEALTH_ALLOWED_HOSTS=("localhost",))
    @patch("dashboard.views.urllib.request.urlopen")
    def test_api_status_refuses_disallowed_ollama_host(self, mocked_urlopen):
        ProviderConfig.objects.create(
            name="Untrusted Ollama",
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://metadata.internal:11434",
        )
        cache.clear()

        response = self.client.get(reverse("dashboard_api_status"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "is not in AI_HUB_PROVIDER_HEALTH_ALLOWED_HOSTS",
        )
        mocked_urlopen.assert_not_called()


class GetItemFilterTest(TestCase):
    """Test the get_item template filter added to academy templatetags."""

    def test_get_item_returns_value(self):
        from academy.templatetags.markdownify import get_item
        d = {"a": 1, "b": 2}
        self.assertEqual(get_item(d, "a"), 1)
        self.assertEqual(get_item(d, "b"), 2)

    def test_get_item_missing_key_returns_none(self):
        from academy.templatetags.markdownify import get_item
        self.assertIsNone(get_item({"a": 1}, "z"))

    def test_get_item_non_dict_returns_none(self):
        from academy.templatetags.markdownify import get_item
        self.assertIsNone(get_item(None, "a"))
        self.assertIsNone(get_item("not a dict", "a"))
        self.assertIsNone(get_item([], "a"))
