from types import SimpleNamespace

from django import forms
from django.contrib import admin
from django.contrib.admin.utils import flatten_fieldsets
from django.contrib.auth import get_user_model
from django.db import models
from django.test import Client, RequestFactory, SimpleTestCase, TestCase
from django.urls import reverse

from ai_hub.admin_json import (
    SafeAdminJSONField,
    SafeAdminJSONWidget,
    expected_json_container,
)
from ai_hub.models import (
    AgentProfile,
    ExecutionStepRun,
    GameActionApprovalRequest,
    GameActionRun,
    GameContinuationRequest,
    GameGoal,
    GameGoalPlan,
    GameMemoryEntry,
    GameWorkspace,
    KnowledgeDocument,
    ModelConfig,
    ProviderConfig,
    ToolDefinition,
    ToolExecutionRun,
)
from ai_hub.test_application_scope_helpers import test_scope


class SafeAdminJSONFieldTests(SimpleTestCase):
    def test_optional_object_field_normalizes_blank_to_object(self):
        field = SafeAdminJSONField(required=False, expected_type=dict)

        self.assertEqual(field.clean(""), {})

    def test_optional_array_field_normalizes_blank_to_array(self):
        field = SafeAdminJSONField(required=False, expected_type=list)

        self.assertEqual(field.clean(""), [])

    def test_field_rejects_wrong_root_type(self):
        field = SafeAdminJSONField(required=False, expected_type=dict)

        with self.assertRaisesMessage(forms.ValidationError, "JSON object"):
            field.clean('["not", "an", "object"]')

    def test_widget_formats_valid_json_and_preserves_invalid_input(self):
        widget = SafeAdminJSONWidget(expected_type=dict)

        self.assertEqual(
            widget.format_value('{"nested":{"enabled":true}}'),
            '{\n  "nested": {\n    "enabled": true\n  }\n}',
        )
        self.assertEqual(widget.format_value('{"broken":'), '{"broken":')

    def test_container_type_is_inferred_only_from_explicit_container_default(self):
        self.assertIs(expected_json_container(ToolDefinition._meta.get_field("config")), dict)
        self.assertIs(expected_json_container(KnowledgeDocument._meta.get_field("tags")), list)
        self.assertIs(expected_json_container(SimpleNamespace(default={})), dict)
        self.assertIs(expected_json_container(SimpleNamespace(default=[])), list)
        self.assertIsNone(expected_json_container(SimpleNamespace(default=lambda: {})))

    def test_widget_preserves_additional_css_classes(self):
        widget = SafeAdminJSONWidget(attrs={"class": "host-editor"})

        self.assertIn("ai-hub-json-editor", widget.attrs["class"])
        self.assertIn("host-editor", widget.attrs["class"])


class SafeAdminJSONIntegrationTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_superuser(
            username="json-admin",
            password="testpass123",
        )
        self.client = Client()
        self.client.force_login(self.user)
        self.provider = ProviderConfig.objects.create(
            name="JSON provider",
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://localhost:11434",
        )
        self.model = ModelConfig.objects.create(
            provider=self.provider,
            model_name="ollama/json-test",
        )
        self.agent = AgentProfile.objects.create(
            application_scope=test_scope(),
            name="json-agent",
            role="JSON tester",
            model_config=self.model,
        )

    def test_agent_change_form_uses_safe_editor_and_local_assets(self):
        response = self.client.get(
            reverse("admin:ai_hub_agentprofile_change", args=[self.agent.pk])
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-ai-json-editor="true"', count=2)
        self.assertContains(response, 'data-json-root="object"', count=2)
        self.assertContains(response, "ai_hub/CSS/json-editor.css")
        self.assertContains(response, "ai_hub/JS/json-editor.js")

    def test_every_editable_admin_json_field_uses_the_safe_field_and_widget(self):
        request = RequestFactory().get("/admin/")
        request.user = self.user
        checked_fields = set()

        def assert_safe_fields(model, form_class):
            for model_field in model._meta.fields:
                if not isinstance(model_field, models.JSONField):
                    continue
                form_field = form_class.base_fields.get(model_field.name)
                if form_field is None:
                    continue
                checked_fields.add((model.__name__, model_field.name))
                self.assertIsInstance(form_field, SafeAdminJSONField)
                self.assertIsInstance(form_field.widget, SafeAdminJSONWidget)

        for model, model_admin in admin.site._registry.items():
            if model._meta.app_label != "ai_hub":
                continue
            with self.subTest(admin=model.__name__):
                assert_safe_fields(model, model_admin.get_form(request))
                for inline in model_admin.get_inline_instances(request):
                    assert_safe_fields(
                        inline.model,
                        inline.get_formset(request).form,
                    )

        self.assertTrue(
            {
                ("ToolDefinition", "config"),
                ("AgentProfile", "input_contract"),
                ("GameWorkspace", "default_policy"),
                ("PipelineStep", "input_mapping"),
                ("KnowledgeDocument", "tags"),
            }.issubset(checked_fields)
        )

    def test_invalid_json_is_rejected_without_losing_submitted_value(self):
        response = self.client.post(
            reverse("admin:ai_hub_agentprofile_change", args=[self.agent.pk]),
            {
                "name": self.agent.name,
                "role": self.agent.role,
                "model_config": self.model.pk,
                "execution_mode": AgentProfile.ExecutionMode.INHERIT,
                "knowledge_max_chars": "6000",
                "system_prompt": "",
                "input_contract": '{"broken":',
                "output_contract": "{}",
                "is_active": "on",
                "toolbox_assignments-TOTAL_FORMS": "0",
                "toolbox_assignments-INITIAL_FORMS": "0",
                "toolbox_assignments-MIN_NUM_FORMS": "0",
                "toolbox_assignments-MAX_NUM_FORMS": "1000",
                "tool_grants-TOTAL_FORMS": "0",
                "tool_grants-INITIAL_FORMS": "0",
                "tool_grants-MIN_NUM_FORMS": "0",
                "tool_grants-MAX_NUM_FORMS": "1000",
                "_save": "Save",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Enter a valid JSON")
        self.assertContains(response, "{&quot;broken&quot;:")

    def test_wrong_root_type_is_rejected_by_admin(self):
        response = self.client.post(
            reverse("admin:ai_hub_agentprofile_change", args=[self.agent.pk]),
            {
                "name": self.agent.name,
                "role": self.agent.role,
                "model_config": self.model.pk,
                "execution_mode": AgentProfile.ExecutionMode.INHERIT,
                "knowledge_max_chars": "6000",
                "system_prompt": "",
                "input_contract": "[]",
                "output_contract": "{}",
                "is_active": "on",
                "toolbox_assignments-TOTAL_FORMS": "0",
                "toolbox_assignments-INITIAL_FORMS": "0",
                "toolbox_assignments-MIN_NUM_FORMS": "0",
                "toolbox_assignments-MAX_NUM_FORMS": "1000",
                "tool_grants-TOTAL_FORMS": "0",
                "tool_grants-INITIAL_FORMS": "0",
                "tool_grants-MIN_NUM_FORMS": "0",
                "tool_grants-MAX_NUM_FORMS": "1000",
                "_save": "Save",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Enter a JSON object at the top level.")

    def test_game_memory_change_form_builds_with_active_status_display(self):
        workspace = GameWorkspace.objects.create(application_scope=test_scope(), name="JSON memory workspace")
        memory = GameMemoryEntry.objects.create(
            workspace=workspace,
            scope_type=GameMemoryEntry.ScopeType.WORKSPACE,
            content="Remember this",
        )

        response = self.client.get(
            reverse("admin:ai_hub_gamememoryentry_change", args=[memory.pk])
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Active")

    def test_goal_and_plan_audit_json_is_redacted(self):
        workspace = GameWorkspace.objects.create(application_scope=test_scope(), name="JSON audit workspace")
        goal = GameGoal.objects.create(
            workspace=workspace,
            title="JSON audit goal",
            description="Check redaction",
            result={"api_key": "goal-secret"},
            transition_metadata={"access_token": "transition-secret"},
        )
        plan = GameGoalPlan.objects.create(
            goal=goal,
            revision_history=[{"password": "plan-secret"}],
        )

        goal_response = self.client.get(
            reverse("admin:ai_hub_gamegoal_change", args=[goal.pk])
        )
        plan_response = self.client.get(
            reverse("admin:ai_hub_gamegoalplan_change", args=[plan.pk])
        )

        self.assertEqual(goal_response.status_code, 200)
        self.assertEqual(plan_response.status_code, 200)
        self.assertContains(goal_response, "***REDACTED***")
        self.assertContains(plan_response, "***REDACTED***")
        self.assertNotContains(goal_response, "goal-secret")
        self.assertNotContains(goal_response, "transition-secret")
        self.assertNotContains(plan_response, "plan-secret")

    def test_audit_payloads_are_excluded_and_replaced_by_redacted_views(self):
        audit_models = {
            ExecutionStepRun: {
                "raw": {"request_payload", "response_payload", "observation_payload"},
                "redacted": {
                    "request_payload_redacted",
                    "response_payload_redacted",
                    "observation_payload_redacted",
                },
            },
            ToolExecutionRun: {
                "raw": {"input_payload", "output_payload"},
                "redacted": {"input_payload_redacted", "output_payload_redacted"},
            },
            GameActionRun: {
                "raw": {"input_payload", "output_payload", "observation_payload"},
                "redacted": {
                    "input_payload_redacted",
                    "output_payload_redacted",
                    "observation_payload_redacted",
                },
            },
            GameContinuationRequest: {
                "raw": {"payload"},
                "redacted": {"payload_redacted"},
            },
            GameActionApprovalRequest: {
                "raw": {"requested_payload", "execution_intent_snapshot"},
                "redacted": {
                    "requested_payload_redacted",
                    "execution_intent_snapshot_redacted",
                },
            },
            GameGoal: {
                "raw": {"result", "transition_metadata"},
                "redacted": {"result_redacted", "transition_metadata_redacted"},
                "excluded": True,
            },
            GameGoalPlan: {
                "raw": {"revision_history"},
                "redacted": {"revision_history_redacted"},
                "excluded": True,
            },
        }

        for model, field_names in audit_models.items():
            with self.subTest(model=model.__name__):
                model_admin = admin.site._registry[model]
                configured_fields = set(model_admin.readonly_fields)
                if model_admin.fieldsets:
                    configured_fields.update(flatten_fieldsets(model_admin.fieldsets))
                elif model_admin.fields:
                    configured_fields.update(model_admin.fields)

                if field_names.get("excluded"):
                    self.assertTrue(
                        field_names["raw"].issubset(set(model_admin.exclude or ()))
                    )
                else:
                    self.assertTrue(field_names["raw"].isdisjoint(configured_fields))
                self.assertTrue(field_names["redacted"].issubset(configured_fields))
                self.assertTrue(
                    field_names["redacted"].issubset(set(model_admin.readonly_fields))
                )
