import json
import importlib
from io import BytesIO
from unittest.mock import patch
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.core.cache import cache
from django.core.management import call_command
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import IntegrityError, close_old_connections, connection, transaction
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from ai_hub.models import (
    AgentProfile,
    AgentToolGrant,
    AgentToolboxAssignment,
    ExecutionSession,
    ExecutionStepRun,
    GameActionDefinition,
    GameActionRun,
    GameGoal,
    GameGoalDependency,
    GameMemoryEntry,
    GameWorkspace,
    GameWorkspaceAction,
    GameWorkspaceAgent,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    ProviderConfig,
    ToolDefinition,
    Toolbox,
    ToolboxTool,
    ToolExecutionRun,
)
from ai_hub.services.litellm_client import ProviderExecutionError, completion_call
from ai_hub.services.admin_control_center import build_control_center_context
from ai_hub.services.execution_sessions import create_execution_session
from ai_hub.services.agent_runtime import (
    apply_mapping,
    build_agent_knowledge_context,
    execute_agent,
    execute_agent_deliberate,
    prepare_agent_payload,
)
from ai_hub.services.contracts import validate_payload
from ai_hub.services.execution_runner import run_execution_session
from ai_hub.services.tool_resolution import resolve_agent_tools
from ai_hub.services.knowledge_retrieval import (
    browse_knowledge_index,
    cite_knowledge_source,
    list_knowledge_libraries,
    read_document_section,
    read_knowledge_chunk,
    search_knowledge,
)
from ai_hub.services.starter_demo import seed_starter_demo
from ai_hub.services.starter_toolboxes import ROLE_TOOLBOXES, seed_starter_toolboxes
from ai_hub.services.game_dependencies import add_goal_dependency, get_goal_blockers
from ai_hub.services.game_goals import create_goal, reopen_goal, transition_goal_status, update_goal_priority
from ai_hub.services.game_workspaces import create_workspace
from ai_hub.services.game_priority import calculate_goal_priority
from ai_hub.services.game_scheduler import (
    claim_next_goal,
    get_next_eligible_goal,
    refresh_workspace_goal_priorities,
)
from ai_hub.services.game_goal_execution import create_goal_execution_session
from ai_hub.services.game_goal_outcomes import apply_session_outcome_to_goal, reconcile_goal_outcomes
from ai_hub.services.tools_runtime import (
    GAME_ACTION_TOOL,
    GAME_CONTEXT_TOOL,
    execute_tool,
    execute_tools,
    get_game_tool_category,
)
from _core.database_config import build_database_config
# DreamPost was the original host-app model; replaced with User for portability


class StreamingHttpResponse:
    """Requests-like response double that fails if an unbounded text read occurs."""

    class RawBody:
        def __init__(self, body):
            self.buffer = BytesIO(body)
            self.read_sizes = []

        def read(self, amount=-1, decode_content=False):
            self.read_sizes.append(amount)
            return self.buffer.read(amount)

    def __init__(self, *, status_code=200, headers=None, body=b"", encoding="utf-8"):
        self.status_code = status_code
        self.headers = headers or {}
        self.raw = self.RawBody(body)
        self.encoding = encoding
        self.closed = False

    @property
    def text(self):
        raise AssertionError("HTTP Tool runtime must not materialize response.text")

    def close(self):
        self.closed = True


class DatabaseConfigurationTests(TestCase):
    def test_sqlite_is_the_zero_configuration_default(self):
        base_dir = (Path.cwd() / "database-config-tests").resolve()
        config = build_database_config({}, base_dir=base_dir)
        relative = build_database_config(
            {"SQLITE_NAME": "var/test.sqlite3"},
            base_dir=base_dir,
        )

        self.assertEqual(config["ENGINE"], "django.db.backends.sqlite3")
        self.assertEqual(config["NAME"], base_dir / "db.sqlite3")
        self.assertEqual(relative["NAME"], base_dir / "var/test.sqlite3")

    def test_discrete_postgresql_environment_is_supported(self):
        config = build_database_config(
            {
                "DATABASE_ENGINE": "postgresql",
                "POSTGRES_DB": "ai_hub_test",
                "POSTGRES_USER": "ai_hub",
                "POSTGRES_PASSWORD": "secret",
                "POSTGRES_HOST": "db",
                "POSTGRES_PORT": "5433",
                "DB_CONN_MAX_AGE": "60",
                "DB_CONN_HEALTH_CHECKS": "false",
            },
            base_dir=Path("C:/workspace"),
        )

        self.assertEqual(config["ENGINE"], "django.db.backends.postgresql")
        self.assertEqual(config["NAME"], "ai_hub_test")
        self.assertEqual(config["HOST"], "db")
        self.assertEqual(config["PORT"], "5433")
        self.assertEqual(config["CONN_MAX_AGE"], 60)
        self.assertFalse(config["CONN_HEALTH_CHECKS"])

    def test_database_url_takes_precedence_and_decodes_credentials(self):
        config = build_database_config(
            {
                "DATABASE_ENGINE": "sqlite",
                "DATABASE_URL": (
                    "postgresql://user%40hub:p%40ss@localhost:5432/ai_hub"
                    "?sslmode=require&connect_timeout=99"
                ),
            },
            base_dir=Path("C:/workspace"),
        )

        self.assertEqual(config["USER"], "user@hub")
        self.assertEqual(config["PASSWORD"], "p" + chr(64) + "ss")
        self.assertEqual(config["NAME"], "ai_hub")
        self.assertEqual(config["OPTIONS"], {"sslmode": "require"})

    def test_invalid_database_configuration_fails_early(self):
        with self.assertRaisesMessage(ImproperlyConfigured, "DATABASE_ENGINE"):
            build_database_config(
                {"DATABASE_ENGINE": "mysql"},
                base_dir=Path("C:/workspace"),
            )
        with self.assertRaisesMessage(ImproperlyConfigured, "POSTGRES_DB"):
            build_database_config(
                {"DATABASE_ENGINE": "postgresql"},
                base_dir=Path("C:/workspace"),
            )
        with self.assertRaisesMessage(ImproperlyConfigured, "invalid PostgreSQL port"):
            build_database_config(
                {"DATABASE_URL": "postgresql://postgres@localhost:not-a-port/db"},
                base_dir=Path("C:/workspace"),
            )
        with self.assertRaisesMessage(ImproperlyConfigured, "POSTGRES_PORT"):
            build_database_config(
                {
                    "DATABASE_ENGINE": "postgresql",
                    "POSTGRES_DB": "ai_hub_test",
                    "POSTGRES_PORT": "70000",
                },
                base_dir=Path("C:/workspace"),
            )


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

    def test_training_model_name_must_follow_routing_convention(self):
        provider = ProviderConfig.objects.create(name="p-train", provider_type="training")

        bad = ModelConfig(provider=provider, model_name="training-game-v1")
        with self.assertRaises(ValidationError):
            bad.full_clean()

        for good_name in ("training", "training/assistant"):
            ok = ModelConfig(provider=provider, model_name=good_name)
            try:
                ok.full_clean()
            except ValidationError:
                self.fail(f"model_name {good_name!r} should pass the training convention")

    def test_non_training_model_name_is_not_constrained(self):
        provider = ProviderConfig.objects.create(name="p-openai", provider_type="openai")
        model = ModelConfig(provider=provider, model_name="gpt-4.1-mini")
        try:
            model.full_clean()
        except ValidationError:
            self.fail("non-training providers must not be subject to the training convention")

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

    @override_settings(AI_HUB_LEGACY_EAGER_KNOWLEDGE_CONTEXT_ENABLED=True)
    def test_legacy_agent_knowledge_context_uses_only_active_documents(self):
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

    def test_toolbox_can_group_tools_and_be_assigned_to_agent(self):
        provider = ProviderConfig.objects.create(name="toolbox-provider", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        agent = AgentProfile.objects.create(name="toolbox-agent", role="Toolbox tester", model_config=model)
        tool = ToolDefinition.objects.create(
            name="grouped_read_tool",
            label="Grouped read tool",
            description="A neutral read tool used to verify toolbox grouping.",
            tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
            operation_mode=ToolDefinition.OperationMode.READ,
            risk_level=ToolDefinition.RiskLevel.LOW,
        )
        toolbox = Toolbox.objects.create(
            name="Knowledge Retrieval",
            slug="knowledge-retrieval",
            label="Knowledge Retrieval",
        )

        membership = ToolboxTool.objects.create(toolbox=toolbox, tool=tool, display_order=10)
        assignment = AgentToolboxAssignment.objects.create(agent=agent, toolbox=toolbox)

        self.assertEqual(toolbox.tool_entries.get(), membership)
        self.assertEqual(agent.toolbox_assignments.get(), assignment)
        self.assertTrue(membership.is_enabled)
        self.assertTrue(assignment.is_enabled)

    def test_agent_admin_shows_safe_resolved_tool_manifest(self):
        admin_user = get_user_model().objects.create_user(
            username="tool-admin",
            password="testpass123",
            is_staff=True,
            is_superuser=True,
        )
        provider = ProviderConfig.objects.create(name="admin-tool-provider", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        agent = AgentProfile.objects.create(name="admin-tool-agent", role="Tool manifest tester", model_config=model)
        tool = ToolDefinition.objects.create(
            name="admin_safe_tool",
            label="Admin safe tool",
            description="Visible admin manifest description.",
            tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
            operation_mode=ToolDefinition.OperationMode.READ,
            risk_level=ToolDefinition.RiskLevel.LOW,
            config={"callable": "ai_hub.tools.knowledge.search_knowledge"},
        )
        toolbox = Toolbox.objects.create(name="Admin Toolbox", slug="admin-toolbox", label="Admin Toolbox")
        ToolboxTool.objects.create(toolbox=toolbox, tool=tool)
        AgentToolboxAssignment.objects.create(agent=agent, toolbox=toolbox)
        client = Client()
        client.force_login(admin_user)

        response = client.get(reverse("admin:ai_hub_agentprofile_change", args=[agent.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Resolved tool access")
        self.assertContains(response, "admin_safe_tool")
        self.assertContains(response, "Visible admin manifest description.")
        self.assertNotContains(response, "ai_hub.tools.knowledge.search_knowledge")

    def test_agent_toolbox_assignment_can_be_disabled(self):
        provider = ProviderConfig.objects.create(name="toolbox-provider-disabled", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        agent = AgentProfile.objects.create(name="toolbox-agent-disabled", role="Toolbox tester", model_config=model)
        toolbox = Toolbox.objects.create(name="Core Foundation", slug="core-foundation", label="Core Foundation")

        assignment = AgentToolboxAssignment.objects.create(agent=agent, toolbox=toolbox, is_enabled=False)

        self.assertFalse(assignment.is_enabled)

    def test_agent_tool_grant_can_allow_or_deny_specific_tool(self):
        provider = ProviderConfig.objects.create(name="grant-provider", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        agent = AgentProfile.objects.create(name="grant-agent", role="Grant tester", model_config=model)
        allowed_tool = ToolDefinition.objects.create(
            name="create_draft",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            operation_mode=ToolDefinition.OperationMode.DRAFT_WRITE,
        )
        denied_tool = ToolDefinition.objects.create(
            name="send_email",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.EXTERNAL_WRITE,
            risk_level=ToolDefinition.RiskLevel.HIGH,
            requires_approval=True,
        )

        allow_grant = AgentToolGrant.objects.create(
            agent=agent,
            tool=allowed_tool,
            permission_level=AgentToolGrant.PermissionLevel.DRAFT_WRITE,
        )
        deny_grant = AgentToolGrant.objects.create(agent=agent, tool=denied_tool, is_enabled=False)

        self.assertTrue(allow_grant.is_enabled)
        self.assertEqual(allow_grant.permission_level, AgentToolGrant.PermissionLevel.DRAFT_WRITE)
        self.assertFalse(deny_grant.is_enabled)

    def test_duplicate_toolbox_membership_is_rejected(self):
        tool = ToolDefinition.objects.create(name="duplicate-tool", tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO)
        toolbox = Toolbox.objects.create(name="Duplicate Toolbox", slug="duplicate-toolbox", label="Duplicate Toolbox")
        ToolboxTool.objects.create(toolbox=toolbox, tool=tool)

        with self.assertRaises(IntegrityError), transaction.atomic():
            ToolboxTool.objects.create(toolbox=toolbox, tool=tool)

    def test_duplicate_agent_toolbox_assignment_is_rejected(self):
        provider = ProviderConfig.objects.create(name="duplicate-assignment-provider", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        agent = AgentProfile.objects.create(name="duplicate-assignment-agent", role="Tester", model_config=model)
        toolbox = Toolbox.objects.create(
            name="Duplicate Assignment",
            slug="duplicate-assignment",
            label="Duplicate Assignment",
        )
        AgentToolboxAssignment.objects.create(agent=agent, toolbox=toolbox)

        with self.assertRaises(IntegrityError), transaction.atomic():
            AgentToolboxAssignment.objects.create(agent=agent, toolbox=toolbox)

    def test_duplicate_agent_tool_grant_is_rejected(self):
        provider = ProviderConfig.objects.create(name="duplicate-grant-provider", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        agent = AgentProfile.objects.create(name="duplicate-grant-agent", role="Tester", model_config=model)
        tool = ToolDefinition.objects.create(name="duplicate-grant-tool", tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO)
        AgentToolGrant.objects.create(agent=agent, tool=tool)

        with self.assertRaises(IntegrityError), transaction.atomic():
            AgentToolGrant.objects.create(agent=agent, tool=tool)

    def test_tool_definition_rejects_invalid_risk_level_or_operation_mode(self):
        bad_risk = ToolDefinition(
            name="bad-risk-tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            risk_level="unknown",
        )
        with self.assertRaises(ValidationError):
            bad_risk.full_clean()

        bad_mode = ToolDefinition(
            name="bad-mode-tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            operation_mode="silent_write",
        )
        with self.assertRaises(ValidationError):
            bad_mode.full_clean()

    def test_tool_execution_run_accepts_generic_audit_payload(self):
        tool = ToolDefinition.objects.create(
            name="audited-tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            risk_level=ToolDefinition.RiskLevel.MEDIUM,
        )

        run = ToolExecutionRun.objects.create(
            tool=tool,
            status=ToolExecutionRun.Status.SUCCESS,
            input_payload={"query": "baseline"},
            output_payload={"result": "ok"},
            risk_level=tool.risk_level,
            approval_state=ToolExecutionRun.ApprovalState.NOT_REQUIRED,
            latency_ms=12,
        )

        self.assertEqual(run.tool, tool)
        self.assertEqual(run.output_payload["result"], "ok")
        self.assertEqual(run.risk_level, ToolDefinition.RiskLevel.MEDIUM)


class ToolResolutionTests(TestCase):
    def setUp(self):
        self.provider = ProviderConfig.objects.create(name="resolver-provider", provider_type="training")
        self.model = ModelConfig.objects.create(provider=self.provider, model_name="training")
        self.agent = AgentProfile.objects.create(
            name="resolver-agent",
            role="Resolver tester",
            model_config=self.model,
        )

    def make_tool(self, name, **kwargs):
        defaults = {
            "tool_kind": ToolDefinition.ToolKind.PROMPT_MACRO,
            "input_schema": {"required": ["query"]},
            "output_schema": {"required": ["result"]},
            "config": {"template": "secret implementation detail"},
        }
        defaults.update(kwargs)
        return ToolDefinition.objects.create(name=name, **defaults)

    def test_resolves_tools_from_active_toolbox_assignment(self):
        tool = self.make_tool("toolbox-search", label="Toolbox Search")
        toolbox = Toolbox.objects.create(name="Resolver Toolbox", slug="resolver-toolbox", label="Resolver Toolbox")
        ToolboxTool.objects.create(toolbox=toolbox, tool=tool)
        AgentToolboxAssignment.objects.create(agent=self.agent, toolbox=toolbox)

        resolution = resolve_agent_tools(self.agent)

        self.assertEqual(resolution.tool_names(), ["toolbox-search"])
        self.assertEqual(resolution.tools[0].source, "toolbox")

    def test_agent_deny_override_removes_toolbox_tool(self):
        tool = self.make_tool("denied-tool")
        toolbox = Toolbox.objects.create(name="Deny Toolbox", slug="deny-toolbox", label="Deny Toolbox")
        ToolboxTool.objects.create(toolbox=toolbox, tool=tool)
        AgentToolboxAssignment.objects.create(agent=self.agent, toolbox=toolbox)
        AgentToolGrant.objects.create(agent=self.agent, tool=tool, is_enabled=False)

        resolution = resolve_agent_tools(self.agent)

        self.assertEqual(resolution.tool_names(), [])

    def test_agent_allow_override_adds_extra_tool(self):
        tool = self.make_tool(
            "grant-draft",
            operation_mode=ToolDefinition.OperationMode.DRAFT_WRITE,
        )
        AgentToolGrant.objects.create(
            agent=self.agent,
            tool=tool,
            permission_level=AgentToolGrant.PermissionLevel.DRAFT_WRITE,
            requires_approval_override=True,
        )

        resolution = resolve_agent_tools(self.agent)

        self.assertEqual(resolution.tool_names(), ["grant-draft"])
        self.assertEqual(resolution.tools[0].source, "agent_grant")
        self.assertEqual(resolution.tools[0].permission_level, AgentToolGrant.PermissionLevel.DRAFT_WRITE)
        self.assertTrue(resolution.tools[0].requires_approval)

    def test_agent_grant_permission_level_filters_operation_mode(self):
        read_tool = self.make_tool("grant-read-tool", operation_mode=ToolDefinition.OperationMode.READ)
        draft_tool = self.make_tool("grant-draft-tool", operation_mode=ToolDefinition.OperationMode.DRAFT_WRITE)
        AgentToolGrant.objects.create(
            agent=self.agent,
            tool=read_tool,
            permission_level=AgentToolGrant.PermissionLevel.READ_ONLY,
        )
        AgentToolGrant.objects.create(
            agent=self.agent,
            tool=draft_tool,
            permission_level=AgentToolGrant.PermissionLevel.READ_ONLY,
        )

        resolution = resolve_agent_tools(self.agent)

        self.assertEqual(resolution.tool_names(), ["grant-read-tool"])

    def test_invalid_http_read_configuration_fails_during_resolution(self):
        tool = self.make_tool(
            "misclassified-http-write",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/resource",
                "method": "POST",
                "allowed_hosts": ["allowed.example"],
            },
        )
        self.agent.tools.add(tool)

        with self.assertRaisesMessage(
            ValidationError,
            "HTTP tools with operation_mode READ must use GET or HEAD",
        ):
            resolve_agent_tools(self.agent)

    def test_writable_http_tool_resolves_with_matching_grant(self):
        tool = self.make_tool(
            "governed-http-write",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.STATE_WRITE,
            config={
                "url": "https://allowed.example/resource",
                "method": "POST",
                "allowed_hosts": ["allowed.example"],
            },
        )
        tool.full_clean()
        AgentToolGrant.objects.create(
            agent=self.agent,
            tool=tool,
            permission_level=AgentToolGrant.PermissionLevel.STATE_WRITE,
        )

        resolution = resolve_agent_tools(self.agent)

        self.assertEqual(resolution.tool_names(), ["governed-http-write"])
        self.assertEqual(
            resolution.tools[0].permission_level,
            AgentToolGrant.PermissionLevel.STATE_WRITE,
        )

    def test_restrictive_agent_grant_overrides_toolbox_access(self):
        tool = self.make_tool(
            "toolbox-draft-restricted",
            operation_mode=ToolDefinition.OperationMode.DRAFT_WRITE,
        )
        toolbox = Toolbox.objects.create(
            name="Restricted Toolbox",
            slug="restricted-toolbox",
            label="Restricted Toolbox",
        )
        ToolboxTool.objects.create(toolbox=toolbox, tool=tool)
        AgentToolboxAssignment.objects.create(agent=self.agent, toolbox=toolbox)
        AgentToolGrant.objects.create(
            agent=self.agent,
            tool=tool,
            permission_level=AgentToolGrant.PermissionLevel.READ_ONLY,
        )

        resolution = resolve_agent_tools(self.agent)

        self.assertEqual(resolution.tool_names(), [])

    def test_legacy_direct_tools_remain_compatible(self):
        tool = self.make_tool("legacy-direct")
        self.agent.tools.add(tool)

        resolution = resolve_agent_tools(self.agent)

        self.assertEqual(resolution.tool_names(), ["legacy-direct"])
        self.assertEqual(resolution.tools[0].source, "legacy_direct")

    def test_inactive_tools_and_toolboxes_are_filtered(self):
        active_tool = self.make_tool("active-tool")
        inactive_tool = self.make_tool("inactive-tool", is_active=False)
        inactive_toolbox_tool = self.make_tool("inactive-toolbox-tool")
        disabled_membership_tool = self.make_tool("disabled-membership-tool")

        active_toolbox = Toolbox.objects.create(name="Active Toolbox", slug="active-toolbox", label="Active Toolbox")
        inactive_toolbox = Toolbox.objects.create(
            name="Inactive Toolbox",
            slug="inactive-toolbox",
            label="Inactive Toolbox",
            is_active=False,
        )
        ToolboxTool.objects.create(toolbox=active_toolbox, tool=active_tool)
        ToolboxTool.objects.create(toolbox=active_toolbox, tool=inactive_tool)
        ToolboxTool.objects.create(toolbox=inactive_toolbox, tool=inactive_toolbox_tool)
        ToolboxTool.objects.create(toolbox=active_toolbox, tool=disabled_membership_tool, is_enabled=False)
        AgentToolboxAssignment.objects.create(agent=self.agent, toolbox=active_toolbox)
        AgentToolboxAssignment.objects.create(agent=self.agent, toolbox=inactive_toolbox)

        resolution = resolve_agent_tools(self.agent)

        self.assertEqual(resolution.tool_names(), ["active-tool"])

    def test_inactive_assignment_is_filtered(self):
        tool = self.make_tool("assignment-disabled-tool")
        toolbox = Toolbox.objects.create(
            name="Assignment Disabled Toolbox",
            slug="assignment-disabled-toolbox",
            label="Assignment Disabled Toolbox",
        )
        ToolboxTool.objects.create(toolbox=toolbox, tool=tool)
        AgentToolboxAssignment.objects.create(agent=self.agent, toolbox=toolbox, is_enabled=False)

        resolution = resolve_agent_tools(self.agent)

        self.assertEqual(resolution.tool_names(), [])

    def test_workspace_policy_filters_allowed_and_blocked_tools(self):
        allowed = self.make_tool("allowed-tool")
        blocked = self.make_tool("blocked-tool")
        absent = self.make_tool("absent-tool")
        for tool in (allowed, blocked, absent):
            self.agent.tools.add(tool)
        workspace = GameWorkspace.objects.create(
            name="resolver-workspace",
            default_policy={
                "allowed_tools": ["allowed-tool", "blocked-tool"],
                "blocked_tools": ["blocked-tool"],
            },
        )

        resolution = resolve_agent_tools(self.agent, workspace=workspace)

        self.assertEqual(resolution.tool_names(), ["allowed-tool"])

    def test_workspace_policy_blocks_external_writes_by_default(self):
        read_tool = self.make_tool("read-tool")
        external_tool = self.make_tool(
            "external-tool",
            operation_mode=ToolDefinition.OperationMode.EXTERNAL_WRITE,
        )
        self.agent.tools.add(read_tool, external_tool)
        workspace = GameWorkspace.objects.create(name="resolver-external-workspace")

        resolution = resolve_agent_tools(self.agent, workspace=workspace)

        self.assertEqual(resolution.tool_names(), ["read-tool"])

    def test_workspace_policy_can_mark_medium_or_high_risk_as_approval_required(self):
        medium_tool = self.make_tool("medium-tool", risk_level=ToolDefinition.RiskLevel.MEDIUM)
        high_tool = self.make_tool("high-tool", risk_level=ToolDefinition.RiskLevel.HIGH)
        self.agent.tools.add(medium_tool, high_tool)
        workspace = GameWorkspace.objects.create(
            name="resolver-approval-workspace",
            default_policy={
                "safety": {
                    "require_approval_for_medium_risk": True,
                    "require_approval_for_high_risk": True,
                }
            },
        )

        resolution = resolve_agent_tools(self.agent, workspace=workspace)

        self.assertEqual(resolution.tool_names(), ["high-tool", "medium-tool"])
        self.assertTrue(all(resolved_tool.requires_approval for resolved_tool in resolution.tools))

    def test_manifest_is_safe_and_omits_runtime_config(self):
        tool = self.make_tool(
            "safe-manifest-tool",
            label="Safe Manifest Tool",
            description="Visible model description.",
            config={
                "callable": "private.module.fn",
                "headers": {"Authorization": "Bearer secret"},
            },
        )
        self.agent.tools.add(tool)

        manifest = resolve_agent_tools(self.agent).manifest()

        self.assertEqual(
            manifest,
            [
                {
                    "name": "safe-manifest-tool",
                    "label": "Safe Manifest Tool",
                    "description": "Visible model description.",
                    "operation_mode": ToolDefinition.OperationMode.READ,
                    "risk_level": ToolDefinition.RiskLevel.LOW,
                    "requires_approval": False,
                    "input_schema": {"required": ["query"]},
                    "output_schema": {"required": ["result"]},
                }
            ],
        )
        self.assertNotIn("config", manifest[0])
        self.assertNotIn("callable", str(manifest[0]))
        self.assertNotIn("Authorization", str(manifest[0]))


class RetrievalFoundationMigrationTests(TestCase):
    def test_migration_renames_and_secures_the_legacy_list_tool(self):
        ToolDefinition.objects.filter(name="list_knowledge_libraries").delete()
        old_tool = ToolDefinition.objects.create(
            name="list_available_knowledge_libraries",
            tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
            input_schema={"required": ["agent_id"]},
            config={
                "callable": "ai_hub.tools.knowledge.list_knowledge_libraries",
                "read_only": True,
            },
        )
        core_toolbox = Toolbox.objects.create(
            name="Legacy Core",
            slug="core-foundation",
        )
        ToolboxTool.objects.create(toolbox=core_toolbox, tool=old_tool)
        migration = importlib.import_module(
            "ai_hub.migrations.0019_retrieval_first_foundation"
        )

        from django.apps import apps as django_apps

        migration.establish_retrieval_foundation(django_apps, None)
        old_tool.refresh_from_db()

        self.assertEqual(old_tool.name, "list_knowledge_libraries")
        self.assertTrue(old_tool.is_system_tool)
        self.assertTrue(old_tool.config["bind_agent_context"])
        self.assertNotIn("agent_id", str(old_tool.input_schema))
        self.assertFalse(
            ToolboxTool.objects.filter(
                toolbox__slug="core-foundation",
                tool=old_tool,
            ).exists()
        )
        self.assertTrue(
            ToolboxTool.objects.filter(
                toolbox__slug="knowledge-discovery-retrieval",
                tool=old_tool,
            ).exists()
        )

    def test_migration_refuses_to_overwrite_a_custom_canonical_tool(self):
        tool = ToolDefinition.objects.get(name="list_knowledge_libraries")
        tool.config = {"callable": "custom.tools.list_knowledge"}
        tool.save(update_fields=["config"])
        migration = importlib.import_module(
            "ai_hub.migrations.0019_retrieval_first_foundation"
        )

        from django.apps import apps as django_apps

        with self.assertRaisesMessage(RuntimeError, "already used"):
            migration.establish_retrieval_foundation(django_apps, None)


class StarterToolboxSeedTests(TestCase):
    def test_seed_creates_starter_toolboxes_roles_and_assignments(self):
        stats = seed_starter_toolboxes()

        self.assertEqual(Toolbox.objects.count(), 5)
        self.assertEqual(AgentProfile.objects.filter(name__in=ROLE_TOOLBOXES.keys()).count(), 5)
        self.assertGreaterEqual(ToolDefinition.objects.count(), 20)
        self.assertEqual(
            AgentToolboxAssignment.objects.filter(agent__name="Developer Assistant").count(),
            4,
        )
        self.assertEqual(
            set(
                AgentToolboxAssignment.objects.filter(agent__name="Developer Assistant")
                .values_list("toolbox__slug", flat=True)
            ),
            {
                "core-foundation",
                "knowledge-discovery-retrieval",
                "development-code-assistance",
                "workspace-draft-artifacts",
            },
        )
        self.assertGreater(stats["tools_created"], 0)
        self.assertGreater(stats["assignments_created"], 0)

    def test_seed_is_idempotent(self):
        first = seed_starter_toolboxes()
        second = seed_starter_toolboxes()

        self.assertGreater(first["tools_created"], 0)
        self.assertEqual(second["tools_created"], 0)
        self.assertEqual(second["toolboxes_created"], 0)
        self.assertEqual(second["agents_created"], 0)
        self.assertEqual(second["assignments_created"], 0)
        self.assertEqual(Toolbox.objects.count(), 5)
        self.assertEqual(AgentProfile.objects.filter(name__in=ROLE_TOOLBOXES.keys()).count(), 5)

    def test_knowledge_retrieval_toolbox_uses_real_read_only_callables(self):
        seed_starter_toolboxes()

        toolbox = Toolbox.objects.get(slug="knowledge-discovery-retrieval")
        tool_names = set(toolbox.tool_entries.values_list("tool__name", flat=True))
        self.assertEqual(
            tool_names,
            {
                "list_knowledge_libraries",
                "browse_knowledge_index",
                "search_knowledge",
                "read_knowledge_chunk",
                "read_document_section",
                "cite_knowledge_source",
            },
        )
        search_tool = ToolDefinition.objects.get(name="search_knowledge")
        section_tool = ToolDefinition.objects.get(name="read_document_section")
        self.assertEqual(search_tool.tool_kind, ToolDefinition.ToolKind.PYTHON_CALLABLE)
        self.assertTrue(search_tool.config["read_only"])
        self.assertEqual(search_tool.operation_mode, ToolDefinition.OperationMode.READ)
        self.assertEqual(
            set(search_tool.input_schema["properties"]),
            {"query", "collection_id", "limit"},
        )
        self.assertEqual(
            set(section_tool.input_schema["properties"]),
            {"document_id", "section_title", "chunk_index"},
        )

    def test_knowledge_access_resolves_system_retrieval_tools_without_manual_toolbox(self):
        seed_starter_toolboxes()
        provider = ProviderConfig.objects.create(
            name="automatic-knowledge-provider",
            provider_type="training",
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        agent = AgentProfile.objects.create(
            name="automatic-knowledge-agent",
            role="Knowledge reader",
            model_config=model,
        )
        collection = KnowledgeCollection.objects.create(name="Automatic knowledge")
        agent.knowledge_collections.add(collection)

        resolution = resolve_agent_tools(agent)

        self.assertEqual(
            set(resolution.tool_names()),
            {
                "list_knowledge_libraries",
                "browse_knowledge_index",
                "search_knowledge",
                "read_knowledge_chunk",
                "read_document_section",
                "cite_knowledge_source",
            },
        )
        self.assertTrue(all(item.source == "knowledge_retrieval" for item in resolution.tools))

    def test_automatic_knowledge_tools_obey_grants_and_workspace_policy(self):
        seed_starter_toolboxes()
        provider = ProviderConfig.objects.create(
            name="governed-knowledge-provider",
            provider_type="training",
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        agent = AgentProfile.objects.create(
            name="governed-knowledge-agent",
            role="Governed knowledge reader",
            model_config=model,
        )
        collection = KnowledgeCollection.objects.create(name="Governed knowledge")
        agent.knowledge_collections.add(collection)
        search_tool = ToolDefinition.objects.get(name="search_knowledge")
        AgentToolGrant.objects.create(agent=agent, tool=search_tool, is_enabled=False)
        workspace = GameWorkspace.objects.create(
            name="governed-knowledge-workspace",
            default_policy={"blocked_tools": ["read_knowledge_chunk"]},
        )

        resolution = resolve_agent_tools(agent, workspace=workspace)

        self.assertNotIn("search_knowledge", resolution.tool_names())
        self.assertNotIn("read_knowledge_chunk", resolution.tool_names())
        self.assertIn("browse_knowledge_index", resolution.tool_names())

    def test_seed_command_runs(self):
        call_command("seed_ai_hub_starter_toolboxes", verbosity=0)

        self.assertTrue(Toolbox.objects.filter(slug="core-foundation").exists())
        self.assertTrue(AgentProfile.objects.filter(name="General Assistant").exists())


class StarterDemoSeedTests(TestCase):
    def test_demo_seed_creates_safe_workspace_knowledge_and_approval_action(self):
        stats = seed_starter_demo()

        workspace = GameWorkspace.objects.get(name="AI Hub Starter GAME Workspace")
        collection = KnowledgeCollection.objects.get(name="AI Hub Starter Knowledge")
        action = GameActionDefinition.objects.get(name="submit_for_approval")

        self.assertEqual(stats["workspaces_created"], 1)
        self.assertEqual(collection.documents.count(), 1)
        self.assertEqual(
            KnowledgeDocumentChunk.objects.filter(document__collection=collection).count(),
            1,
        )
        self.assertEqual(action.action_type, GameActionDefinition.ActionType.TOOL)
        self.assertEqual(action.tool.name, "submit_for_approval")
        self.assertTrue(action.requires_approval)
        self.assertTrue(
            GameWorkspaceAction.objects.filter(
                workspace=workspace,
                action=action,
                is_enabled=True,
                requires_approval_override=True,
            ).exists()
        )
        self.assertEqual(
            set(
                GameWorkspaceAgent.objects.filter(workspace=workspace)
                .values_list("agent__name", flat=True)
            ),
            {"Business Analyst Agent", "Developer Assistant"},
        )
        self.assertFalse(workspace.default_policy["safety"]["allow_external_writes"])

    def test_demo_seed_is_idempotent(self):
        first = seed_starter_demo()
        second = seed_starter_demo()

        self.assertEqual(first["workspaces_created"], 1)
        self.assertEqual(second["collections_created"], 0)
        self.assertEqual(second["documents_created"], 0)
        self.assertEqual(second["chunks_created"], 0)
        self.assertEqual(second["workspaces_created"], 0)
        self.assertEqual(second["actions_created"], 0)
        self.assertEqual(second["workspace_actions_created"], 0)
        self.assertEqual(second["workspace_agents_created"], 0)

    def test_demo_seed_command_runs(self):
        call_command("seed_ai_hub_starter_demo", verbosity=0)

        self.assertTrue(GameWorkspace.objects.filter(name="AI Hub Starter GAME Workspace").exists())
        self.assertTrue(KnowledgeCollection.objects.filter(name="AI Hub Starter Knowledge").exists())


class KnowledgeRetrievalTests(TestCase):
    def setUp(self):
        self.provider = ProviderConfig.objects.create(name="knowledge-provider", provider_type="training")
        self.model = ModelConfig.objects.create(provider=self.provider, model_name="training")
        self.agent = AgentProfile.objects.create(
            name="knowledge-agent",
            role="Knowledge tester",
            model_config=self.model,
            knowledge_max_chars=50,
        )
        self.collection = KnowledgeCollection.objects.create(
            name="Policies",
            description="Approved support policies.",
        )
        self.other_collection = KnowledgeCollection.objects.create(name="Private policies")
        self.agent.knowledge_collections.add(self.collection)
        self.document = KnowledgeDocument.objects.create(
            collection=self.collection,
            title="Refund policy",
            curated_text="Full refund policy text that should not be injected when retrieval-only mode is active.",
            tags=["refunds", "support"],
            language="en",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        self.chunk = KnowledgeDocumentChunk.objects.create(
            document=self.document,
            chunk_index=1,
            section_title="Eligibility",
            content="Refunds are available within 30 days when the order is unused.",
            token_estimate=14,
            metadata={"source_path": "refunds.md"},
        )
        KnowledgeDocumentChunk.objects.create(
            document=self.document,
            chunk_index=2,
            section_title="Exceptions",
            content="Digital gift cards are not refundable.",
            token_estimate=8,
        )
        self.private_document = KnowledgeDocument.objects.create(
            collection=self.other_collection,
            title="Private refund policy",
            curated_text="Private refund instructions.",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        self.private_chunk = KnowledgeDocumentChunk.objects.create(
            document=self.private_document,
            chunk_index=1,
            section_title="Private",
            content="Private refund escalation instructions.",
        )

    def test_agent_can_list_allowed_knowledge_libraries(self):
        result = list_knowledge_libraries(self.agent)

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["libraries"][0]["name"], "Policies")
        self.assertEqual(result["libraries"][0]["active_documents"], 1)
        self.assertEqual(result["libraries"][0]["chunk_count"], 2)

    def test_agent_can_browse_knowledge_index(self):
        result = browse_knowledge_index(self.agent)

        self.assertEqual(result["total"], 1)
        document = result["collections"][0]["documents"][0]
        self.assertEqual(document["title"], "Refund policy")
        self.assertEqual([chunk["section_title"] for chunk in document["chunks"]], ["Eligibility", "Exceptions"])

    def test_search_returns_only_accessible_active_chunk_results(self):
        result = search_knowledge(self.agent, query="unused", limit=5)

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["results"][0]["chunk_id"], self.chunk.pk)
        self.assertEqual(result["results"][0]["section_title"], "Eligibility")
        self.assertIn("30 days", result["results"][0]["snippet"])
        self.assertEqual(result["results"][0]["citation"]["document_title"], "Refund policy")

    def test_search_returns_chunk_when_query_exists_only_in_document_tags(self):
        self.document.title = "Operations Manual"
        self.document.tags = ["invoice", "finance", "approval"]
        self.document.save(update_fields=["title", "tags"])

        result = search_knowledge(self.agent, query="finance", limit=5)

        self.assertEqual(result["total"], 2)
        self.assertEqual(result["candidates_scanned"], 2)
        self.assertEqual(
            {item["chunk_id"] for item in result["results"]},
            set(self.document.chunks.values_list("pk", flat=True)),
        )
        self.assertTrue(all(item["score"] == 3 for item in result["results"]))

    def test_tag_search_does_not_cross_agent_collection_boundary(self):
        self.private_document.tags = ["confidential-finance"]
        self.private_document.save(update_fields=["tags"])

        result = search_knowledge(self.agent, query="confidential-finance", limit=5)

        self.assertEqual(result["total"], 0)
        self.assertEqual(result["candidates_scanned"], 0)

    def test_tag_search_excludes_inactive_documents_and_collections(self):
        archived = KnowledgeDocument.objects.create(
            collection=self.collection,
            title="Archived operations",
            tags=["dormant-finance"],
            status=KnowledgeDocument.Status.ARCHIVED,
        )
        KnowledgeDocumentChunk.objects.create(
            document=archived,
            chunk_index=1,
            section_title="Archived",
            content="No query term in this chunk.",
        )
        self.private_document.tags = ["dormant-finance"]
        self.private_document.save(update_fields=["tags"])
        self.other_collection.is_active = False
        self.other_collection.save(update_fields=["is_active"])
        self.agent.knowledge_collections.add(self.other_collection)

        result = search_knowledge(self.agent, query="dormant-finance", limit=5)

        self.assertEqual(result["total"], 0)
        self.assertEqual(result["candidates_scanned"], 0)

    def test_search_rejects_unauthorised_collection(self):
        with self.assertRaisesMessage(ValidationError, "not accessible"):
            search_knowledge(self.agent, query="refund", collection_id=self.other_collection.pk)

    @patch("ai_hub.services.knowledge_retrieval.MAX_SEARCH_CANDIDATES", 2)
    def test_search_bounds_the_candidate_window(self):
        KnowledgeDocumentChunk.objects.bulk_create(
            [
                KnowledgeDocumentChunk(
                    document=self.document,
                    chunk_index=index,
                    section_title=f"Candidate {index}",
                    content="candidate-window-needle",
                )
                for index in range(3, 6)
            ]
        )

        result = search_knowledge(
            self.agent,
            query="candidate-window-needle",
            limit=20,
        )

        self.assertEqual(result["candidates_scanned"], 2)
        self.assertEqual(result["candidate_limit"], 2)
        self.assertTrue(result["candidates_truncated"])
        self.assertEqual(result["total"], 2)

    @patch("ai_hub.services.knowledge_retrieval.MAX_SEARCH_RESULTS", 1)
    @patch("ai_hub.services.knowledge_retrieval.MAX_SEARCH_CANDIDATES", 2)
    def test_tag_search_preserves_candidate_and_result_bounds(self):
        for index in range(4):
            document = KnowledgeDocument.objects.create(
                collection=self.collection,
                title=f"Tag-only reference {index}",
                tags=["bounded-finance"],
                status=KnowledgeDocument.Status.ACTIVE,
            )
            KnowledgeDocumentChunk.objects.create(
                document=document,
                chunk_index=1,
                section_title=f"Reference {index}",
                content="No query term is present in this bounded candidate.",
            )

        result = search_knowledge(
            self.agent,
            query="bounded-finance",
            limit=1000,
        )

        self.assertEqual(result["candidates_scanned"], 2)
        self.assertEqual(result["candidate_limit"], 2)
        self.assertTrue(result["candidates_truncated"])
        self.assertEqual(result["total"], 1)
        self.assertLessEqual(len(result["results"][0]["snippet"]), 500)

    @patch(
        "ai_hub.services.knowledge_retrieval.MAX_SEARCH_CANDIDATE_CONTENT_CHARS",
        20,
    )
    def test_search_does_not_materialise_unbounded_chunk_bodies(self):
        self.chunk.content = ("x" * 100) + " distant-needle"
        self.chunk.save(update_fields=["content"])

        result = search_knowledge(
            self.agent,
            query="distant-needle",
            limit=1,
        )

        self.assertEqual(result["total"], 1)
        self.assertTrue(result["results"][0]["content_window_truncated"])
        self.assertLessEqual(len(result["results"][0]["snippet"]), 20)

    def test_read_returns_only_selected_chunk_content(self):
        result = read_knowledge_chunk(self.agent, chunk_id=self.chunk.pk)

        self.assertEqual(result["content"], self.chunk.content)
        self.assertNotIn("Digital gift cards", result["content"])
        self.assertEqual(result["metadata"], {"source_path": "refunds.md"})

    def test_read_blocks_unauthorised_chunk(self):
        with self.assertRaisesMessage(ValidationError, "not accessible"):
            read_knowledge_chunk(self.agent, chunk_id=self.private_chunk.pk)

    def test_read_document_section_by_title_or_index(self):
        by_title = read_document_section(
            self.agent,
            document_id=self.document.pk,
            section_title="Exceptions",
        )
        by_index = read_document_section(
            self.agent,
            document_id=self.document.pk,
            chunk_index=2,
        )

        self.assertEqual(by_title["section_title"], "Exceptions")
        self.assertEqual(by_index["section_title"], "Exceptions")

    def test_citations_include_useful_source_metadata(self):
        result = cite_knowledge_source(self.agent, chunk_id=self.chunk.pk)

        self.assertEqual(result["citation"]["collection"], "Policies")
        self.assertEqual(result["citation"]["document_title"], "Refund policy")
        self.assertEqual(result["citation"]["section_title"], "Eligibility")
        self.assertEqual(result["citation"]["tags"], ["refunds", "support"])

    def test_retrieval_first_context_is_the_default_and_does_not_inject_documents(self):
        context = build_agent_knowledge_context(self.agent)

        self.assertTrue(context["retrieval_required"])
        self.assertEqual(context["documents"], [])
        self.assertEqual(context["text"], "")
        self.assertEqual(context["collections"], ["Policies"])
        self.assertEqual(context["collection_indexes"][0]["documents"][0]["chunk_count"], 2)
        self.assertIn("search_knowledge", context["available_retrieval_tools"])
        self.assertNotIn("Full refund policy text", str(context))

    def test_retrieval_is_not_required_without_attached_collections(self):
        agent = AgentProfile.objects.create(
            name="knowledge-free-agent",
            role="No knowledge",
            model_config=self.model,
        )

        context = build_agent_knowledge_context(agent)

        self.assertFalse(context["retrieval_required"])
        self.assertFalse(context["retrieval_available"])
        self.assertEqual(context["collection_indexes"], [])

    def test_retrieval_prompt_index_is_bounded(self):
        KnowledgeDocument.objects.bulk_create(
            [
                KnowledgeDocument(
                    collection=self.collection,
                    title=f"Reference {index:02d}",
                    status=KnowledgeDocument.Status.ACTIVE,
                )
                for index in range(55)
            ]
        )

        context = build_agent_knowledge_context(self.agent)
        collection_index = context["collection_indexes"][0]

        self.assertEqual(collection_index["total_documents"], 56)
        self.assertEqual(collection_index["returned_documents"], 50)
        self.assertTrue(collection_index["has_more_documents"])
        self.assertTrue(context["index_truncated"])

    @override_settings(AI_HUB_LEGACY_EAGER_KNOWLEDGE_CONTEXT_ENABLED=True)
    def test_legacy_eager_context_remains_available(self):
        context = build_agent_knowledge_context(self.agent)

        self.assertEqual(len(context["documents"]), 1)
        self.assertIn("Full refund policy text", context["text"])

    def test_search_knowledge_python_callable_tool_uses_agent_access(self):
        tool = ToolDefinition.objects.create(
            name="search_knowledge_tool",
            tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
            input_schema={"required": ["agent_id", "query"]},
            output_schema={"required": ["query", "results", "total"]},
            config={"callable": "ai_hub.tools.knowledge.search_knowledge"},
        )

        result = execute_tool(tool, {"agent_id": self.agent.pk, "query": "unused"})

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["results"][0]["chunk_id"], self.chunk.pk)

    def test_bound_knowledge_tools_limit_read_search_and_browse_outputs(self):
        seed_starter_toolboxes()
        self.chunk.content = "x" * 9000
        self.chunk.metadata = {"oversized": "m" * 3000}
        self.chunk.save(update_fields=["content", "metadata"])
        self.document.tags = [f"{index}-" + ("t" * 120) for index in range(30)]
        self.document.save(update_fields=["tags"])
        KnowledgeDocumentChunk.objects.bulk_create(
            [
                KnowledgeDocumentChunk(
                    document=self.document,
                    chunk_index=index,
                    section_title=f"Bounded {index}",
                    content="bounded needle result",
                )
                for index in range(3, 33)
            ]
        )
        read_tool = ToolDefinition.objects.get(name="read_knowledge_chunk")
        search_tool = ToolDefinition.objects.get(name="search_knowledge")
        browse_tool = ToolDefinition.objects.get(name="browse_knowledge_index")

        read_result = execute_tool(
            read_tool,
            {"agent_id": -1, "chunk_id": self.chunk.pk},
            agent=self.agent,
        )
        search_result = execute_tool(
            search_tool,
            {"query": "bounded needle", "limit": 1000},
            agent=self.agent,
        )
        browse_result = execute_tool(
            browse_tool,
            {"collection_id": self.collection.pk},
            agent=self.agent,
        )

        self.assertEqual(len(read_result["content"]), 8000)
        self.assertEqual(read_result["content_chars"], 9000)
        self.assertTrue(read_result["content_truncated"])
        self.assertTrue(read_result["metadata_truncated"])
        self.assertTrue(read_result["metadata"]["truncated"])
        self.assertEqual(search_result["total"], 20)
        self.assertEqual(len(search_result["results"][0]["citation"]["tags"]), 20)
        indexed_document = browse_result["collections"][0]["documents"][0]
        self.assertEqual(indexed_document["returned_chunks"], 25)
        self.assertTrue(indexed_document["chunks_have_more"])

        bounded_query = search_knowledge(
            self.agent,
            query="unused " * 300,
            limit=1,
        )
        self.assertLessEqual(len(bounded_query["query"]), 1000)

    @override_settings(
        AI_HUB_DEFAULT_AGENT_TOOL_RUNTIME="resolved",
        AI_HUB_LEGACY_EAGER_KNOWLEDGE_CONTEXT_ENABLED=False,
        AI_HUB_MAX_TOOL_ROUNDS_PER_AGENT_CALL=6,
    )
    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_resolved_runner_retrieves_all_stages_with_server_bound_agent(self, mocked_call):
        seed_starter_toolboxes()
        private_agent = AgentProfile.objects.create(
            name="private-knowledge-agent",
            role="Must not be impersonated",
            model_config=self.model,
        )
        private_agent.knowledge_collections.add(self.other_collection)
        self.agent.input_contract = {"required": ["question"]}
        self.agent.output_contract = {"required": ["agent", "llm", "tools"]}
        self.agent.save(update_fields=["input_contract", "output_contract", "updated_at"])
        pipeline = PipelineDefinition.objects.create(
            name="knowledge-retrieval-pipeline",
            is_active=False,
            global_input_contract={"required": ["question"]},
            global_output_contract={"required": ["answer"]},
        )
        PipelineStep.objects.create(
            pipeline=pipeline,
            agent=self.agent,
            order=1,
            input_mapping={"question": "question"},
            output_mapping={"answer": "llm.content"},
        )
        pipeline.is_active = True
        pipeline.save(update_fields=["is_active"])
        session = create_execution_session(
            pipeline=pipeline,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            initial_context={"question": "When are unused orders refundable?"},
        )
        mocked_call.side_effect = [
            {
                "status": "ok",
                "content": json.dumps(
                    {
                        "type": "tool_call",
                        "tool_name": "list_knowledge_libraries",
                        "arguments": {"agent_id": private_agent.pk},
                    }
                ),
            },
            {
                "status": "ok",
                "content": json.dumps(
                    {
                        "type": "tool_call",
                        "tool_name": "browse_knowledge_index",
                        "arguments": {},
                    }
                ),
            },
            {
                "status": "ok",
                "content": json.dumps(
                    {
                        "type": "tool_call",
                        "tool_name": "search_knowledge",
                        "arguments": {"query": "unused order"},
                    }
                ),
            },
            {
                "status": "ok",
                "content": json.dumps(
                    {
                        "type": "tool_call",
                        "tool_name": "read_knowledge_chunk",
                        "arguments": {"chunk_id": self.chunk.pk},
                    }
                ),
            },
            {
                "status": "ok",
                "content": json.dumps(
                    {
                        "type": "tool_call",
                        "tool_name": "read_document_section",
                        "arguments": {
                            "document_id": self.document.pk,
                            "section_title": "Eligibility",
                        },
                    }
                ),
            },
            {
                "status": "ok",
                "content": json.dumps(
                    {
                        "type": "tool_call",
                        "tool_name": "cite_knowledge_source",
                        "arguments": {"chunk_id": self.chunk.pk},
                    }
                ),
            },
            {
                "status": "ok",
                "content": '{"type":"final","answer":"Unused orders are refundable within 30 days."}',
            },
        ]

        run_execution_session(session.pk)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(
            session.final_context["answer"],
            "Unused orders are refundable within 30 days.",
        )
        step_run = session.step_runs.get()
        knowledge_context = step_run.request_payload["knowledge_context"]
        self.assertEqual(knowledge_context["text"], "")
        self.assertNotIn("Full refund policy text", str(step_run.request_payload))
        self.assertEqual(
            set(item["name"] for item in step_run.response_payload["tool_manifest"]),
            {
                "list_knowledge_libraries",
                "browse_knowledge_index",
                "search_knowledge",
                "read_knowledge_chunk",
                "read_document_section",
                "cite_knowledge_source",
            },
        )
        first_manifest = json.loads(
            mocked_call.call_args_list[0].kwargs["messages"][1]["content"]
        )["available_tools"]
        self.assertNotIn("agent_id", str(first_manifest))
        first_tool_result = step_run.response_payload["tools"]["list_knowledge_libraries"]
        self.assertEqual(first_tool_result["total"], 1)
        self.assertEqual(first_tool_result["libraries"][0]["name"], self.collection.name)
        self.assertNotIn(self.other_collection.name, str(first_tool_result))
        self.assertEqual(ToolExecutionRun.objects.filter(session=session).count(), 6)
        self.assertTrue(
            all(
                run.input_payload["agent_id"] == self.agent.pk
                for run in ToolExecutionRun.objects.filter(session=session)
            )
        )

    def test_build_console_created_knowledge_gets_an_initial_retrievable_chunk(self):
        from ai_hub.services.build_console import attach_knowledge

        errors = {}
        attach_knowledge(
            self.agent,
            {
                "knowledge_mode": "create",
                "knowledge_collection_name": "Build knowledge",
                "knowledge_doc_title": "Build document",
                "knowledge_doc_content": "A document created through the Build Console.",
            },
            errors,
        )

        self.assertEqual(errors, {})
        document = KnowledgeDocument.objects.get(title="Build document")
        chunk = document.chunks.get()
        self.assertEqual(chunk.chunk_index, 1)
        self.assertEqual(chunk.content, document.curated_text)


class DeliberateToolRuntimeTests(TestCase):
    def setUp(self):
        self.provider = ProviderConfig.objects.create(name="deliberate-provider", provider_type="training")
        self.model = ModelConfig.objects.create(provider=self.provider, model_name="training")
        self.agent = AgentProfile.objects.create(
            name="deliberate-agent",
            role="Deliberate runtime tester",
            model_config=self.model,
            input_contract={"required": ["query"]},
            output_contract={"required": ["agent", "tools", "llm"]},
        )

    def make_prompt_tool(self, name="deliberate-tool", **kwargs):
        defaults = {
            "tool_kind": ToolDefinition.ToolKind.PROMPT_MACRO,
            "input_schema": {"required": ["query"]},
            "output_schema": {"required": ["macro"]},
            "config": {"template": "macro result"},
        }
        defaults.update(kwargs)
        tool = ToolDefinition.objects.create(name=name, **defaults)
        self.agent.tools.add(tool)
        return tool

    @patch("ai_hub.services.agent_runtime.execute_tools")
    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_inactive_agent_cannot_execute_legacy_runtime_tools_or_provider(
        self,
        mocked_call,
        mocked_execute_tools,
    ):
        self.make_prompt_tool(name="inactive-legacy-boundary-tool")
        self.agent.is_active = False
        self.agent.save(update_fields=["is_active"])

        with self.assertRaisesMessage(
            ValidationError,
            "Agent 'deliberate-agent' is inactive",
        ):
            execute_agent(self.agent, {"query": "must not execute"})

        mocked_execute_tools.assert_not_called()
        mocked_call.assert_not_called()

    @patch("ai_hub.services.agent_runtime.execute_tool")
    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_inactive_agent_cannot_execute_deliberate_tools_or_provider(
        self,
        mocked_call,
        mocked_execute_tool,
    ):
        self.make_prompt_tool(name="inactive-deliberate-boundary-tool")
        self.agent.is_active = False
        self.agent.save(update_fields=["is_active"])

        with self.assertRaisesMessage(
            ValidationError,
            "Agent 'deliberate-agent' is inactive",
        ):
            execute_agent_deliberate(self.agent, {"query": "must not execute"})

        mocked_execute_tool.assert_not_called()
        mocked_call.assert_not_called()

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_final_response_does_not_execute_available_tools(self, mocked_call):
        self.make_prompt_tool()
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"type": "final", "answer": "done"}',
        }

        output = execute_agent_deliberate(self.agent, {"query": "hello"})

        self.assertEqual(output["status"], "final")
        self.assertEqual(output["final_answer"], "done")
        self.assertEqual(output["tools"], {})
        self.assertEqual(ToolExecutionRun.objects.count(), 0)
        sent_user_payload = json.loads(mocked_call.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(sent_user_payload["available_tools"][0]["name"], "deliberate-tool")

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_agent_requests_one_valid_tool_then_finalises(self, mocked_call):
        self.make_prompt_tool()
        mocked_call.side_effect = [
            {
                "status": "ok",
                "content": '{"type": "tool_call", "tool_name": "deliberate-tool", "arguments": {"query": "hello"}}',
            },
            {
                "status": "ok",
                "content": '{"type": "final", "answer": "used tool"}',
            },
        ]

        output = execute_agent_deliberate(self.agent, {"query": "hello"})

        self.assertEqual(output["status"], "final")
        self.assertEqual(output["tools"], {"deliberate-tool": {"macro": "macro result"}})
        run = ToolExecutionRun.objects.get()
        self.assertEqual(run.status, ToolExecutionRun.Status.SUCCESS)
        self.assertEqual(run.input_payload, {"query": "hello"})
        self.assertEqual(run.output_payload, {"macro": "macro result"})

    @override_settings(AI_HUB_MAX_TOOL_OBSERVATION_CHARS=512)
    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_large_tool_results_are_bounded_in_prompt_but_raw_in_audit(self, mocked_call):
        self.make_prompt_tool(config={"template": "x" * 2000})
        mocked_call.side_effect = [
            {
                "status": "ok",
                "content": (
                    '{"type":"tool_call","tool_name":"deliberate-tool",'
                    '"arguments":{"query":"hello"}}'
                ),
            },
            {
                "status": "ok",
                "content": '{"type":"final","answer":"used bounded tool result"}',
            },
        ]

        output = execute_agent_deliberate(self.agent, {"query": "hello"})

        prompt_result = output["tools"]["deliberate-tool"]
        self.assertTrue(prompt_result["truncated"])
        self.assertEqual(prompt_result["tool_name"], "deliberate-tool")
        self.assertLessEqual(len(json.dumps(prompt_result, sort_keys=True)), 512)
        raw_result = ToolExecutionRun.objects.get().output_payload
        self.assertEqual(len(raw_result["macro"]), 2000)
        second_messages = mocked_call.call_args_list[1].kwargs["messages"]
        tool_observation = json.loads(second_messages[-1]["content"])
        self.assertEqual(
            tool_observation["tool_observation"]["result"],
            prompt_result,
        )

    @override_settings(AI_HUB_MAX_TOOL_OBSERVATION_CHARS=512)
    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_legacy_tool_prompt_is_bounded_without_losing_step_output(self, mocked_call):
        self.make_prompt_tool(config={"template": "x" * 2000})
        mocked_call.return_value = {"status": "ok", "content": "done"}

        output = execute_agent(self.agent, {"query": "hello"})

        self.assertEqual(len(output["tools"]["deliberate-tool"]["macro"]), 2000)
        sent_payload = json.loads(
            mocked_call.call_args.kwargs["messages"][1]["content"]
        )
        prompt_result = sent_payload["tool_results"]["deliberate-tool"]
        self.assertTrue(prompt_result["truncated"])
        self.assertLessEqual(len(json.dumps(prompt_result, sort_keys=True)), 512)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_legacy_runtime_does_not_preexecute_approval_required_tool(self, mocked_call):
        self.make_prompt_tool(requires_approval=True)
        mocked_call.return_value = {"status": "ok", "content": "done"}

        output = execute_agent(self.agent, {"query": "hello"})

        self.assertEqual(output["tools"], {})
        sent_payload = json.loads(
            mocked_call.call_args.kwargs["messages"][1]["content"]
        )
        self.assertNotIn("tool_results", sent_payload)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_legacy_runtime_honours_workspace_effective_approval(self, mocked_call):
        self.make_prompt_tool(risk_level=ToolDefinition.RiskLevel.MEDIUM)
        workspace = GameWorkspace.objects.create(
            name="legacy-approval-workspace",
            default_policy={
                "safety": {"require_approval_for_medium_risk": True},
            },
        )
        mocked_call.return_value = {"status": "ok", "content": "done"}

        output = execute_agent(
            self.agent,
            {"query": "hello"},
            workspace=workspace,
        )

        self.assertEqual(output["tools"], {})

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_unknown_or_unauthorised_tool_is_blocked(self, mocked_call):
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"type": "tool_call", "tool_name": "missing-tool", "arguments": {}}',
        }

        output = execute_agent_deliberate(self.agent, {"query": "hello"})

        self.assertEqual(output["status"], "unauthorised_tool")
        self.assertIn("missing-tool", output["error"])
        self.assertEqual(ToolExecutionRun.objects.count(), 0)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_invalid_tool_arguments_are_audited_as_failed(self, mocked_call):
        self.make_prompt_tool()
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"type": "tool_call", "tool_name": "deliberate-tool", "arguments": {}}',
        }

        output = execute_agent_deliberate(self.agent, {"query": "hello"})

        self.assertEqual(output["status"], "tool_error")
        run = ToolExecutionRun.objects.get()
        self.assertEqual(run.status, ToolExecutionRun.Status.FAILED)
        self.assertIn("missing required keys", run.error_detail)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_tool_output_contract_violation_is_audited_as_failed(self, mocked_call):
        self.make_prompt_tool(output_schema={"required": ["result"]})
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"type": "tool_call", "tool_name": "deliberate-tool", "arguments": {"query": "hello"}}',
        }

        output = execute_agent_deliberate(self.agent, {"query": "hello"})

        self.assertEqual(output["status"], "tool_error")
        run = ToolExecutionRun.objects.get()
        self.assertEqual(run.status, ToolExecutionRun.Status.FAILED)
        self.assertIn("missing required keys", run.error_detail)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_approval_required_tool_pauses_without_execution(self, mocked_call):
        self.make_prompt_tool(requires_approval=True)
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"type": "tool_call", "tool_name": "deliberate-tool", "arguments": {"query": "hello"}}',
        }

        output = execute_agent_deliberate(self.agent, {"query": "hello"})

        self.assertEqual(output["status"], "waiting_approval")
        self.assertEqual(output["tools"], {})
        run = ToolExecutionRun.objects.get()
        self.assertEqual(run.status, ToolExecutionRun.Status.WAITING_APPROVAL)
        self.assertEqual(run.approval_state, ToolExecutionRun.ApprovalState.REQUIRED)
        self.assertEqual(run.output_payload, {})

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_max_tool_rounds_stops_uncontrolled_chains(self, mocked_call):
        self.make_prompt_tool()
        mocked_call.side_effect = [
            {
                "status": "ok",
                "content": '{"type": "tool_call", "tool_name": "deliberate-tool", "arguments": {"query": "one"}}',
            },
            {
                "status": "ok",
                "content": '{"type": "tool_call", "tool_name": "deliberate-tool", "arguments": {"query": "two"}}',
            },
        ]

        output = execute_agent_deliberate(self.agent, {"query": "hello"}, max_tool_rounds=1)

        self.assertEqual(output["status"], "max_tool_rounds")
        self.assertEqual(ToolExecutionRun.objects.count(), 1)
        self.assertEqual(ToolExecutionRun.objects.get().status, ToolExecutionRun.Status.SUCCESS)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_invalid_model_tool_response_is_blocked(self, mocked_call):
        mocked_call.return_value = {"status": "ok", "content": "not json"}

        output = execute_agent_deliberate(self.agent, {"query": "hello"})

        self.assertEqual(output["status"], "invalid_model_response")
        self.assertEqual(ToolExecutionRun.objects.count(), 0)


class GameWorkspaceGoalTests(TestCase):
    def setUp(self):
        self.workspace = create_workspace(name="Academy GAME", description="Test workspace")

    def make_goal(self, title, **kwargs):
        return create_goal(
            workspace=kwargs.pop("workspace", self.workspace),
            title=title,
            description=kwargs.pop("description", f"Complete {title}"),
            **kwargs,
        )

    def test_workspace_can_have_multiple_goals(self):
        self.make_goal("First")
        self.make_goal("Second")

        self.assertEqual(self.workspace.goals.count(), 2)
        self.assertEqual(ExecutionSession.objects.count(), 0)

    def test_goal_defaults_to_queued(self):
        goal = self.make_goal("Queued by default")

        self.assertEqual(goal.status, GameGoal.Status.QUEUED)
        self.assertEqual(goal.base_priority, 50)
        self.assertEqual(goal.success_criteria, {})

    def test_new_goal_cannot_start_in_terminal_or_running_state(self):
        for status in (GameGoal.Status.RUNNING, GameGoal.Status.COMPLETED, GameGoal.Status.CANCELLED):
            with self.subTest(status=status), self.assertRaisesMessage(ValidationError, "draft or queued"):
                self.make_goal(f"Invalid {status}", status=status)

    def test_priority_bounds_are_validated_before_database_write(self):
        with self.assertRaises(ValidationError):
            self.make_goal("Too large", base_priority=999901)

    def test_goal_cannot_depend_on_itself(self):
        goal = self.make_goal("Self")

        with self.assertRaisesMessage(ValidationError, "cannot depend on itself"):
            add_goal_dependency(goal, goal)

    def test_goal_cannot_depend_on_goal_in_another_workspace(self):
        other_workspace = create_workspace(name="Other GAME")
        goal = self.make_goal("Local")
        other_goal = self.make_goal("Remote", workspace=other_workspace)

        with self.assertRaisesMessage(ValidationError, "same workspace"):
            add_goal_dependency(goal, other_goal)

    def test_duplicate_dependency_is_rejected(self):
        goal = self.make_goal("Dependent")
        prerequisite = self.make_goal("Prerequisite")
        add_goal_dependency(goal, prerequisite)

        with self.assertRaises(ValidationError):
            add_goal_dependency(goal, prerequisite)

    def test_circular_dependency_is_rejected(self):
        first = self.make_goal("First")
        second = self.make_goal("Second")
        third = self.make_goal("Third")
        add_goal_dependency(first, second)
        add_goal_dependency(second, third)

        with self.assertRaisesMessage(ValidationError, "Circular"):
            add_goal_dependency(third, first)

    def test_goal_transition_rejects_invalid_state_change(self):
        goal = self.make_goal("Invalid transition")

        with self.assertRaisesMessage(ValidationError, "Cannot transition"):
            transition_goal_status(goal, GameGoal.Status.COMPLETED)

    def test_completed_goal_requires_explicit_reopen(self):
        goal = self.make_goal("Complete me")
        goal = transition_goal_status(goal, GameGoal.Status.RUNNING, reason="claimed")
        goal = transition_goal_status(goal, GameGoal.Status.COMPLETED, reason="done", result={"ok": True})

        with self.assertRaises(ValidationError):
            transition_goal_status(goal, GameGoal.Status.QUEUED)

        reopened = reopen_goal(goal, reason="review")
        self.assertEqual(reopened.status, GameGoal.Status.QUEUED)
        self.assertTrue(reopened.transition_metadata["reopened"])
        self.assertEqual(reopened.result, {})

    def test_cancelled_goal_requires_explicit_reopen(self):
        goal = self.make_goal("Cancel me")
        goal = transition_goal_status(goal, GameGoal.Status.CANCELLED)

        with self.assertRaises(ValidationError):
            transition_goal_status(goal, GameGoal.Status.QUEUED)

        self.assertEqual(reopen_goal(goal).status, GameGoal.Status.QUEUED)

    def test_required_dependency_blocks_goal(self):
        goal = self.make_goal("Blocked goal")
        prerequisite = self.make_goal("Required goal")
        add_goal_dependency(goal, prerequisite, is_required=True)

        self.assertEqual(
            get_goal_blockers(goal),
            [{"goal_id": prerequisite.id, "title": prerequisite.title, "status": GameGoal.Status.QUEUED}],
        )

        prerequisite = transition_goal_status(prerequisite, GameGoal.Status.RUNNING)
        transition_goal_status(prerequisite, GameGoal.Status.COMPLETED)
        self.assertEqual(get_goal_blockers(goal), [])

    def test_optional_dependency_does_not_block_goal(self):
        goal = self.make_goal("Unblocked goal")
        optional = self.make_goal("Optional goal")
        add_goal_dependency(goal, optional, is_required=False)

        self.assertEqual(get_goal_blockers(goal), [])

    def test_workspace_deletion_cascades_to_goals_and_dependencies(self):
        goal = self.make_goal("Dependent")
        prerequisite = self.make_goal("Prerequisite")
        add_goal_dependency(goal, prerequisite)

        self.workspace.delete()

        self.assertEqual(GameGoal.objects.count(), 0)
        self.assertEqual(GameGoalDependency.objects.count(), 0)

    def test_priority_updates_through_service(self):
        goal = self.make_goal("Prioritise")

        updated = update_goal_priority(goal, "87.25")

        self.assertEqual(updated.calculated_priority, Decimal("87.25"))

    def test_game_domain_models_are_registered_in_admin(self):
        self.assertTrue(admin.site.is_registered(GameWorkspace))
        self.assertTrue(admin.site.is_registered(GameGoal))
        self.assertTrue(admin.site.is_registered(GameGoalDependency))


class GameSchedulerTests(TestCase):
    def setUp(self):
        self.workspace = create_workspace(name="Scheduler GAME")
        self.now = timezone.now().replace(microsecond=0)

    def make_goal(self, title, **kwargs):
        return create_goal(
            workspace=self.workspace,
            title=title,
            description=f"Complete {title}",
            **kwargs,
        )

    def test_high_priority_goal_is_selected_before_low_priority_goal(self):
        low = self.make_goal("Low", base_priority=10)
        high = self.make_goal("High", base_priority=90)

        selected = get_next_eligible_goal(self.workspace.id, now=self.now)

        self.assertEqual(selected, high)
        self.assertEqual(selected.calculated_priority, Decimal("90.00"))
        self.assertEqual(GameGoal.objects.get(pk=high.pk).calculated_priority, Decimal("0"))
        self.assertNotEqual(selected, low)

    def test_overdue_goal_outranks_non_urgent_goal(self):
        overdue = self.make_goal("Overdue", base_priority=20, due_at=self.now - timedelta(days=1))
        self.make_goal("Normal", base_priority=50)

        self.assertEqual(get_next_eligible_goal(self.workspace.id, now=self.now), overdue)
        self.assertEqual(calculate_goal_priority(overdue, now=self.now), Decimal("60.00"))

    def test_due_today_goal_outranks_same_priority_goal(self):
        due_today = self.make_goal("Due today", due_at=self.now)
        self.make_goal("No deadline")

        self.assertEqual(get_next_eligible_goal(self.workspace.id, now=self.now), due_today)

    def test_blocked_goal_is_not_selected(self):
        blocked = self.make_goal("Blocked", base_priority=1000)
        transition_goal_status(blocked, GameGoal.Status.BLOCKED)
        queued = self.make_goal("Queued", base_priority=1)

        self.assertEqual(get_next_eligible_goal(self.workspace.id, now=self.now), queued)

    def test_waiting_approval_goal_is_not_selected(self):
        waiting = self.make_goal("Waiting", base_priority=1000)
        waiting = transition_goal_status(waiting, GameGoal.Status.RUNNING)
        transition_goal_status(waiting, GameGoal.Status.WAITING_APPROVAL)

        self.assertIsNone(get_next_eligible_goal(self.workspace.id, now=self.now))

    def test_goal_with_unresolved_required_dependency_is_not_selected(self):
        blocked = self.make_goal("Dependent", base_priority=100)
        prerequisite = self.make_goal("Prerequisite", base_priority=10)
        add_goal_dependency(blocked, prerequisite, is_required=True)

        self.assertEqual(get_next_eligible_goal(self.workspace.id, now=self.now), prerequisite)

    def test_optional_dependency_does_not_prevent_selection(self):
        goal = self.make_goal("Optional dependent", base_priority=100)
        optional = self.make_goal("Optional prerequisite", base_priority=10)
        add_goal_dependency(goal, optional, is_required=False)

        self.assertEqual(get_next_eligible_goal(self.workspace.id, now=self.now), goal)

    def test_goal_unlocking_dependents_receives_bonus(self):
        unlocker = self.make_goal("Unlocker", base_priority=50)
        dependent = self.make_goal("Dependent", base_priority=100)
        self.make_goal("Rival", base_priority=55)
        add_goal_dependency(dependent, unlocker, is_required=True)

        self.assertEqual(calculate_goal_priority(unlocker, now=self.now), Decimal("60.00"))
        self.assertEqual(get_next_eligible_goal(self.workspace.id, now=self.now), unlocker)

    def test_queued_for_more_than_seven_days_receives_bonus(self):
        aged = self.make_goal("Aged", base_priority=50)
        GameGoal.objects.filter(pk=aged.pk).update(queued_at=self.now - timedelta(days=8))
        aged.refresh_from_db()

        self.assertEqual(calculate_goal_priority(aged, now=self.now), Decimal("55.00"))

    def test_reopened_old_goal_does_not_immediately_receive_queue_age_bonus(self):
        goal = self.make_goal("Old completed goal")
        GameGoal.objects.filter(pk=goal.pk).update(
            created_at=self.now - timedelta(days=30),
            queued_at=self.now - timedelta(days=30),
        )
        goal.refresh_from_db()
        goal = transition_goal_status(goal, GameGoal.Status.RUNNING)
        goal = transition_goal_status(goal, GameGoal.Status.COMPLETED)
        goal = reopen_goal(goal)

        self.assertEqual(calculate_goal_priority(goal, now=self.now), Decimal("50.00"))

    def test_no_eligible_goal_returns_none(self):
        complete = self.make_goal("Complete")
        complete = transition_goal_status(complete, GameGoal.Status.RUNNING)
        transition_goal_status(complete, GameGoal.Status.COMPLETED)

        self.assertIsNone(get_next_eligible_goal(self.workspace.id, now=self.now))

    def test_priority_refresh_is_deterministic_for_fixed_time(self):
        first = self.make_goal("First", base_priority=25, due_at=self.now)
        second = self.make_goal("Second", base_priority=70)

        refresh_workspace_goal_priorities(self.workspace.id, now=self.now)
        first_values = dict(GameGoal.objects.values_list("pk", "calculated_priority"))
        refresh_workspace_goal_priorities(self.workspace.id, now=self.now)
        second_values = dict(GameGoal.objects.values_list("pk", "calculated_priority"))

        self.assertEqual(first_values, second_values)
        self.assertEqual(first_values[first.pk], Decimal("55.00"))
        self.assertEqual(first_values[second.pk], Decimal("70.00"))

    def test_claim_marks_goal_running(self):
        goal = self.make_goal("Claim me")

        claimed = claim_next_goal(self.workspace.id, actor="test worker", now=self.now)

        self.assertEqual(claimed.pk, goal.pk)
        self.assertEqual(claimed.status, GameGoal.Status.RUNNING)
        self.assertEqual(claimed.calculated_priority, Decimal("50.00"))
        self.assertEqual(claimed.transition_metadata["reason"], "claimed by test worker")
        self.assertIsNone(claim_next_goal(self.workspace.id, now=self.now))

    def test_inactive_workspace_cannot_claim_goal(self):
        self.make_goal("Inactive")
        self.workspace.is_active = False
        self.workspace.save(update_fields=["is_active", "updated_at"])

        self.assertIsNone(get_next_eligible_goal(self.workspace.id, now=self.now))
        self.assertIsNone(claim_next_goal(self.workspace.id, now=self.now))

    def test_equal_scores_have_stable_creation_order(self):
        first = self.make_goal("First tie")
        self.make_goal("Second tie")

        self.assertEqual(get_next_eligible_goal(self.workspace.id, now=self.now), first)


class GameSchedulerConcurrencyTests(TransactionTestCase):
    reset_sequences = True

    def test_concurrent_claims_do_not_claim_same_goal(self):
        if not connection.features.has_select_for_update:
            self.skipTest("SQLite cannot validate select_for_update locking semantics; run this test on PostgreSQL CI.")

        workspace = create_workspace(name="Concurrent GAME")
        goal = create_goal(workspace=workspace, title="Only goal", description="Claim once")

        def claim():
            close_old_connections()
            try:
                claimed = claim_next_goal(workspace.id)
                return claimed.pk if claimed else None
            finally:
                close_old_connections()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: claim(), range(2)))

        self.assertEqual(results.count(goal.pk), 1)
        self.assertEqual(results.count(None), 1)


class GameGoalExecutionTests(TestCase):
    def setUp(self):
        self.workspace = create_workspace(
            name="Goal execution GAME",
            default_runtime_config={"max_iterations": 5, "shared": "workspace"},
        )
        self.provider = ProviderConfig.objects.create(name="goal-provider", provider_type="training")
        self.model = ModelConfig.objects.create(provider=self.provider, model_name="goal-model")
        self.agent = AgentProfile.objects.create(
            name="goal-agent",
            role="Goal runner",
            model_config=self.model,
            input_contract={"required": ["goal"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )

    def make_goal(self, title="Goal", **kwargs):
        return create_goal(
            workspace=self.workspace,
            title=title,
            description=kwargs.pop("description", "Complete the linked goal."),
            **kwargs,
        )

    def test_goal_execution_session_links_goal_and_marks_it_running(self):
        goal = self.make_goal()

        session = create_goal_execution_session(goal=goal, entry_agent=self.agent)
        goal.refresh_from_db()

        self.assertEqual(session.goal, goal)
        self.assertEqual(session.runtime_kind, ExecutionSession.RuntimeKind.GAME)
        self.assertEqual(session.status, ExecutionSession.Status.PENDING)
        self.assertEqual(goal.status, GameGoal.Status.RUNNING)

    def test_scheduler_claimed_goal_can_create_session(self):
        goal = self.make_goal("Claimed goal")
        claimed = claim_next_goal(self.workspace.id)

        session = create_goal_execution_session(goal=claimed, entry_agent=self.agent)

        self.assertEqual(session.goal_id, goal.id)
        self.assertEqual(session.goal.status, GameGoal.Status.RUNNING)

    def test_inactive_workspace_goal_cannot_create_session(self):
        goal = self.make_goal("Inactive workspace goal")
        self.workspace.is_active = False
        self.workspace.save(update_fields=["is_active", "updated_at"])

        with self.assertRaisesMessage(ValidationError, "inactive GAME workspace"):
            create_goal_execution_session(goal=goal, entry_agent=self.agent)

    def test_goal_execution_session_builds_goal_text(self):
        goal = self.make_goal(
            title="Review docs",
            description="Find architecture gaps.",
            success_criteria={"required": ["summary", "gaps"]},
        )

        session = create_goal_execution_session(goal=goal, entry_agent=self.agent)

        self.assertEqual(
            session.goal_text,
            'Title: Review docs\n\nObjective:\nFind architecture gaps.\n\nSuccess criteria:\n{\n  "required": [\n    "summary",\n    "gaps"\n  ]\n}',
        )

    def test_goal_execution_session_merges_context_in_documented_order(self):
        goal = self.make_goal(
            context={
                "topic": "GAME",
                "runtime_config": {"max_iterations": 3, "shared": "goal", "goal_only": True},
            }
        )

        session = create_goal_execution_session(
            goal=goal,
            entry_agent=self.agent,
            runtime_config={"max_iterations": 1, "shared": "call", "call_only": True},
        )

        self.assertEqual(
            session.runtime_config,
            {
                "max_iterations": 1,
                "shared": "call",
                "goal_only": True,
                "call_only": True,
            },
        )
        self.assertEqual(session.initial_context, goal.context)

    def test_active_goal_session_is_not_duplicated(self):
        goal = self.make_goal()
        create_goal_execution_session(goal=goal, entry_agent=self.agent)

        with self.assertRaisesMessage(ValidationError, "already has an active"):
            create_goal_execution_session(goal=goal, entry_agent=self.agent)

    def test_database_rejects_duplicate_active_goal_sessions(self):
        goal = self.make_goal("Database uniqueness")
        create_goal_execution_session(goal=goal, entry_agent=self.agent)

        with self.assertRaises(IntegrityError), transaction.atomic():
            ExecutionSession.objects.create(
                goal=goal,
                entry_agent=self.agent,
                runtime_kind=ExecutionSession.RuntimeKind.GAME,
                status=ExecutionSession.Status.PENDING,
                goal_text="Bypass service",
            )

    def test_completed_goal_cannot_start_new_session_without_reopen(self):
        goal = self.make_goal()
        goal = transition_goal_status(goal, GameGoal.Status.RUNNING)
        goal = transition_goal_status(goal, GameGoal.Status.COMPLETED)

        with self.assertRaisesMessage(ValidationError, "status 'completed'"):
            create_goal_execution_session(goal=goal, entry_agent=self.agent)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_goal_complete_maps_to_completed(self, mocked_call):
        goal = self.make_goal("Complete goal")
        session = create_goal_execution_session(
            goal=goal,
            entry_agent=self.agent,
            runtime_config={"max_iterations": 2},
        )
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"action": "finish", "final_answer": "Goal achieved."}',
        }

        run_execution_session(session.id)
        goal.refresh_from_db()

        self.assertEqual(goal.status, GameGoal.Status.COMPLETED)
        self.assertEqual(goal.result, {"session_id": session.id, "final_answer": "Goal achieved."})

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_iteration_cap_maps_to_partial(self, mocked_call):
        goal = self.make_goal("Partial goal")
        session = create_goal_execution_session(
            goal=goal,
            entry_agent=self.agent,
            runtime_config={"max_iterations": 1},
        )
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"action": "think", "message": "Still working."}',
        }

        run_execution_session(session.id)
        goal.refresh_from_db()

        self.assertEqual(goal.status, GameGoal.Status.PARTIAL)
        self.assertEqual(goal.result, {"session_id": session.id})

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_failure_maps_to_failed(self, mocked_call):
        goal = self.make_goal("Failed goal")
        session = create_goal_execution_session(goal=goal, entry_agent=self.agent)
        mocked_call.side_effect = RuntimeError("provider failed")

        run_execution_session(session.id)
        goal.refresh_from_db()

        self.assertEqual(goal.status, GameGoal.Status.FAILED)

    def test_waiting_information_maps_to_waiting_info(self):
        goal = self.make_goal("Waiting goal")
        session = create_goal_execution_session(goal=goal, entry_agent=self.agent)
        session.status = ExecutionSession.Status.WAITING_ASYNC
        session.final_context = {
            "execution_outcome": "waiting",
            "goal_outcome": "unknown",
            "waiting_reason": "needs_information",
        }
        session.save(update_fields=["status", "final_context", "updated_at"])

        apply_session_outcome_to_goal(session)
        goal.refresh_from_db()

        self.assertEqual(goal.status, GameGoal.Status.WAITING_INFO)

    def test_waiting_approval_maps_to_waiting_approval(self):
        goal = self.make_goal("Approval goal")
        session = create_goal_execution_session(goal=goal, entry_agent=self.agent)
        session.status = ExecutionSession.Status.WAITING_ASYNC
        session.final_context = {
            "execution_outcome": "waiting",
            "goal_outcome": "unknown",
            "waiting_reason": "needs_approval",
        }
        session.save(update_fields=["status", "final_context", "updated_at"])

        apply_session_outcome_to_goal(session)
        goal.refresh_from_db()

        self.assertEqual(goal.status, GameGoal.Status.WAITING_APPROVAL)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_legacy_session_does_not_create_or_update_goal(self, mocked_call):
        unrelated_goal = self.make_goal("Unrelated")
        legacy_session = create_execution_session(
            entry_agent=self.agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Legacy goal text.",
            runtime_config={"max_iterations": 1},
        )
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"action": "finish", "final_answer": "Legacy done."}',
        }

        run_execution_session(legacy_session.id)
        unrelated_goal.refresh_from_db()

        self.assertIsNone(legacy_session.goal)
        self.assertEqual(unrelated_goal.status, GameGoal.Status.QUEUED)
        self.assertEqual(GameGoal.objects.count(), 1)

    def test_multiple_historical_sessions_can_belong_to_same_goal(self):
        goal = self.make_goal("Retry goal")
        first = create_goal_execution_session(goal=goal, entry_agent=self.agent)
        first.status = ExecutionSession.Status.FAILED
        first.final_context = {
            "execution_outcome": "failed",
            "goal_outcome": "unknown",
            "finish_reason": "failed",
        }
        first.save(update_fields=["status", "final_context", "updated_at"])
        apply_session_outcome_to_goal(first)
        goal.refresh_from_db()
        goal = transition_goal_status(goal, GameGoal.Status.QUEUED, reason="retry")

        second = create_goal_execution_session(goal=goal, entry_agent=self.agent)

        self.assertNotEqual(first.pk, second.pk)
        self.assertEqual(goal.execution_sessions.count(), 2)

    def test_old_session_outcome_cannot_be_replayed_after_retry(self):
        goal = self.make_goal("Replay-safe goal")
        first = create_goal_execution_session(goal=goal, entry_agent=self.agent)
        first.status = ExecutionSession.Status.SUCCESS
        first.final_context = {
            "execution_outcome": "completed",
            "goal_outcome": "achieved",
            "finish_reason": "agent_finished",
            "final_answer": "First result",
        }
        first.save(update_fields=["status", "final_context", "updated_at"])
        apply_session_outcome_to_goal(first)
        first.refresh_from_db()
        goal.refresh_from_db()
        goal = reopen_goal(goal)
        goal = transition_goal_status(goal, GameGoal.Status.RUNNING)

        apply_session_outcome_to_goal(first)
        goal.refresh_from_db()

        self.assertTrue(first.goal_outcome_fingerprint)
        self.assertEqual(goal.status, GameGoal.Status.RUNNING)

    def test_reconciliation_applies_unmapped_terminal_outcome(self):
        goal = self.make_goal("Reconcile goal")
        session = create_goal_execution_session(goal=goal, entry_agent=self.agent)
        session.status = ExecutionSession.Status.FAILED
        session.final_context = {
            "execution_outcome": "failed",
            "goal_outcome": "unknown",
            "finish_reason": "failed",
        }
        session.save(update_fields=["status", "final_context", "updated_at"])

        result = reconcile_goal_outcomes()
        goal.refresh_from_db()

        self.assertEqual(result["applied"], 1)
        self.assertEqual(result["errors"], [])
        self.assertEqual(goal.status, GameGoal.Status.FAILED)


class OrchestratorFallbackTests(TestCase):
    def setUp(self):
        self.provider = ProviderConfig.objects.create(
            name="fallback-provider",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        self.model = ModelConfig.objects.create(
            provider=self.provider,
            model_name="training",
        )
        self.primary = AgentProfile.objects.create(
            name="fallback-primary",
            role="Primary fallback test agent",
            model_config=self.model,
            input_contract={"required": ["source"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        self.fallback = AgentProfile.objects.create(
            name="fallback-secondary",
            role="Secondary fallback test agent",
            model_config=self.model,
            input_contract={"required": ["source"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )

    def _create_pipeline(
        self,
        *,
        on_error=PipelineStep.OnError.FALLBACK_AGENT,
        fallback_agent=None,
        input_mapping=None,
        output_mapping=None,
        global_output_contract=None,
    ):
        pipeline = PipelineDefinition.objects.create(
            name=f"fallback-pipeline-{PipelineDefinition.objects.count() + 1}",
            global_input_contract={"required": ["source"]},
            global_output_contract=(
                {"required": ["result"]}
                if global_output_contract is None
                else global_output_contract
            ),
        )
        step = PipelineStep.objects.create(
            pipeline=pipeline,
            agent=self.primary,
            fallback_agent=self.fallback if fallback_agent is None else fallback_agent,
            order=1,
            input_mapping={"source": "source"} if input_mapping is None else input_mapping,
            output_mapping={"result": "llm.content"} if output_mapping is None else output_mapping,
            on_error=on_error,
        )
        pipeline.is_active = True
        pipeline.save(update_fields=["is_active"])
        return pipeline, step

    def _create_session(self, pipeline, initial_context=None):
        return create_execution_session(
            pipeline=pipeline,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            initial_context=initial_context or {"source": "logical input"},
        )

    @staticmethod
    def _success_output(agent, content="fallback result"):
        return {
            "agent": agent.name,
            "llm": {"status": "ok", "content": content},
            "tools": {},
        }

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_primary_deactivated_after_pipeline_activation_fails_before_provider(
        self,
        mocked_call,
    ):
        pipeline, _step = self._create_pipeline(
            on_error=PipelineStep.OnError.STOP,
            global_output_contract={},
        )
        self.primary.is_active = False
        self.primary.save(update_fields=["is_active"])
        session = self._create_session(pipeline)

        run_execution_session(session.pk)

        session.refresh_from_db()
        step_run = session.step_runs.get()
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(step_run.status, ExecutionStepRun.Status.FAILED)
        self.assertIn("Agent 'fallback-primary' is inactive", step_run.error_detail)
        mocked_call.assert_not_called()

    @patch(
        "ai_hub.services.agent_runtime.completion_call",
        side_effect=Exception("primary provider failed"),
    )
    def test_fallback_deactivated_after_activation_fails_before_second_provider_call(
        self,
        mocked_call,
    ):
        pipeline, _step = self._create_pipeline(global_output_contract={})
        self.fallback.is_active = False
        self.fallback.save(update_fields=["is_active"])
        session = self._create_session(pipeline)

        run_execution_session(session.pk)

        session.refresh_from_db()
        step_run = session.step_runs.get()
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(step_run.status, ExecutionStepRun.Status.FAILED)
        self.assertIn("Agent 'fallback-secondary' is inactive", step_run.error_detail)
        self.assertEqual(mocked_call.call_count, 1)

    def test_fallback_A_prepares_fresh_payload_from_logical_step_input(self):
        pipeline, _step = self._create_pipeline()
        session = self._create_session(pipeline)
        observed = []

        def fake_prepare(agent, context, mapping, *, workspace=None):
            payload = apply_mapping(context, mapping or {})
            payload["prepared_for"] = agent.name
            payload["knowledge_context"] = {"collections": [agent.name]}
            return payload

        def fake_execute(*, agent, payload, **kwargs):
            observed.append((agent, dict(payload)))
            if agent == self.primary:
                raise RuntimeError("primary failed")
            return self._success_output(agent)

        with patch(
            "ai_hub.services.execution_runner.prepare_agent_payload",
            side_effect=fake_prepare,
        ), patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=fake_execute,
        ):
            run_execution_session(session.pk)

        session.refresh_from_db()
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual([agent for agent, _payload in observed], [self.primary, self.fallback])
        self.assertEqual(observed[0][1]["prepared_for"], self.primary.name)
        self.assertEqual(observed[1][1]["prepared_for"], self.fallback.name)
        self.assertEqual(observed[1][1]["source"], "logical input")

    def test_fallback_A_isolates_nested_logical_input_from_primary_mutation(self):
        pipeline, _step = self._create_pipeline(
            input_mapping={
                "source": "source",
                "details": "details",
            },
        )
        session = self._create_session(
            pipeline,
            initial_context={
                "source": "logical input",
                "details": {"owner": "session", "items": ["original"]},
            },
        )
        fallback_payload = {}

        def fake_execute(*, agent, payload, **kwargs):
            if agent == self.primary:
                payload["details"]["owner"] = "primary mutation"
                payload["details"]["items"].append("primary")
                raise RuntimeError("primary failed after mutation")
            fallback_payload.update(payload)
            return self._success_output(agent)

        with patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=fake_execute,
        ):
            run_execution_session(session.pk)

        session.refresh_from_db()
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(
            fallback_payload["details"],
            {"owner": "session", "items": ["original"]},
        )
        self.assertEqual(
            session.final_context["details"],
            {"owner": "session", "items": ["original"]},
        )

    def test_fallback_B_rejects_obvious_input_incompatibility_at_activation(self):
        self.fallback.input_contract = {"required": ["source", "language"]}
        self.fallback.save(update_fields=["input_contract"])
        pipeline, _step = self._create_pipeline()

        with self.assertRaisesMessage(
            ValidationError,
            "Fallback agent 'fallback-secondary' cannot receive required input keys: language",
        ):
            pipeline.full_clean()

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_fallback_B_validates_fallback_input_before_provider_execution(self, mocked_call):
        self.fallback.input_contract = {"required": ["source", "language"]}
        self.fallback.save(update_fields=["input_contract"])
        pipeline, _step = self._create_pipeline()
        session = self._create_session(pipeline)
        mocked_call.side_effect = RuntimeError("primary provider failed")

        run_execution_session(session.pk)

        session.refresh_from_db()
        step_run = session.step_runs.get()
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(step_run.status, ExecutionStepRun.Status.FAILED)
        self.assertIn("fallback-secondary", session.error_detail)
        self.assertIn("language", session.error_detail)
        self.assertEqual(mocked_call.call_count, 1)

    def test_fallback_preparation_failure_does_not_retain_primary_request(self):
        pipeline, _step = self._create_pipeline()
        session = self._create_session(pipeline)

        def fake_prepare(agent, context, mapping, *, workspace=None):
            if agent == self.fallback:
                raise ValidationError("fallback preparation failed")
            return prepare_agent_payload(
                agent,
                context,
                mapping,
                workspace=workspace,
            )

        with patch(
            "ai_hub.services.execution_runner.prepare_agent_payload",
            side_effect=fake_prepare,
        ), patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=RuntimeError("primary execution failed"),
        ) as mocked_execute:
            run_execution_session(session.pk)

        session.refresh_from_db()
        step_run = session.step_runs.get()
        recovery = step_run.response_payload["fallback_recovery"]
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(step_run.status, ExecutionStepRun.Status.FAILED)
        self.assertEqual(step_run.agent, self.fallback)
        self.assertEqual(step_run.request_payload, {})
        self.assertEqual(mocked_execute.call_count, 1)
        self.assertEqual(
            recovery["primary"]["error"]["detail"],
            "primary execution failed",
        )
        self.assertEqual(recovery["fallback"]["status"], "failed")
        self.assertIn(
            "fallback preparation failed",
            recovery["fallback"]["error"]["detail"],
        )

    def test_fallback_contracts_are_required_when_activating_pipeline(self):
        self.fallback.input_contract = {}
        self.fallback.output_contract = {}
        self.fallback.save(update_fields=["input_contract", "output_contract"])
        pipeline, _step = self._create_pipeline()

        with self.assertRaisesMessage(
            ValidationError,
            "Fallback agent 'fallback-secondary' must define input/output contracts",
        ):
            pipeline.full_clean()

    def test_fallback_C_does_not_inherit_primary_knowledge(self):
        primary_collection = KnowledgeCollection.objects.create(name="Primary-only knowledge")
        fallback_collection = KnowledgeCollection.objects.create(name="Fallback-only knowledge")
        self.primary.knowledge_collections.add(primary_collection)
        self.fallback.knowledge_collections.add(fallback_collection)
        pipeline, _step = self._create_pipeline()
        session = self._create_session(pipeline)
        observed = {}

        def fake_execute(*, agent, payload, **kwargs):
            observed[agent.name] = payload
            if agent == self.primary:
                raise RuntimeError("primary failed")
            return self._success_output(agent)

        with patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=fake_execute,
        ):
            run_execution_session(session.pk)

        fallback_knowledge = observed[self.fallback.name]["knowledge_context"]
        self.assertIn(fallback_collection.name, fallback_knowledge["collections"])
        self.assertNotIn(primary_collection.name, fallback_knowledge["collections"])

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_fallback_C_resolves_its_own_tools_and_model(self, mocked_call):
        fallback_model = ModelConfig.objects.create(
            provider=self.provider,
            model_name="training/assistant",
        )
        self.fallback.model_config = fallback_model
        self.fallback.save(update_fields=["model_config"])
        primary_tool = ToolDefinition.objects.create(
            name="primary-only-tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "primary"},
        )
        fallback_tool = ToolDefinition.objects.create(
            name="fallback-only-tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "fallback"},
        )
        self.primary.tools.add(primary_tool)
        self.fallback.tools.add(fallback_tool)
        pipeline, _step = self._create_pipeline()
        session = self._create_session(pipeline)
        mocked_call.side_effect = [
            RuntimeError("primary provider failed"),
            {"status": "ok", "content": "fallback model result"},
        ]

        run_execution_session(session.pk)

        session.refresh_from_db()
        step_run = session.step_runs.get()
        manifest_names = {
            item["name"] for item in step_run.response_payload["tool_manifest"]
        }
        called_models = [
            call.kwargs["model"] for call in mocked_call.call_args_list
        ]
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(called_models, ["training", "training/assistant"])
        self.assertEqual(manifest_names, {fallback_tool.name})
        self.assertNotIn(primary_tool.name, manifest_names)

    def test_fallback_D_executes_and_records_fallback_identity(self):
        pipeline, _step = self._create_pipeline()
        session = self._create_session(pipeline)
        executed_agents = []

        def fake_execute(*, agent, **kwargs):
            executed_agents.append(agent)
            if agent == self.primary:
                raise RuntimeError("primary failed")
            return self._success_output(agent)

        with patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=fake_execute,
        ):
            run_execution_session(session.pk)

        step_run = ExecutionStepRun.objects.get(session=session)
        recovery = step_run.response_payload["fallback_recovery"]
        self.assertEqual(executed_agents, [self.primary, self.fallback])
        self.assertEqual(step_run.response_payload["agent"], self.fallback.name)
        self.assertEqual(recovery["primary"]["agent_id"], self.primary.pk)
        self.assertEqual(recovery["fallback"]["agent_id"], self.fallback.pk)
        self.assertEqual(recovery["fallback"]["agent"], self.fallback.name)

    def test_fallback_E_applies_valid_output_mapping_to_final_context(self):
        pipeline, _step = self._create_pipeline()
        session = self._create_session(pipeline)

        def fake_execute(*, agent, **kwargs):
            if agent == self.primary:
                raise RuntimeError("primary failed")
            return self._success_output(agent, content="mapped fallback result")

        with patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=fake_execute,
        ):
            run_execution_session(session.pk)

        session.refresh_from_db()
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(session.final_context["result"], "mapped fallback result")

    def test_fallback_F_missing_mapped_output_path_is_terminal_failure(self):
        pipeline, _step = self._create_pipeline(
            output_mapping={"summary": "result.summary"},
            global_output_contract={"required": ["summary"]},
        )
        session = self._create_session(pipeline)

        def fake_execute(*, agent, **kwargs):
            if agent == self.primary:
                raise RuntimeError("primary failed")
            return self._success_output(agent)

        with patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=fake_execute,
        ):
            run_execution_session(session.pk)

        session.refresh_from_db()
        step_run = session.step_runs.get()
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(step_run.status, ExecutionStepRun.Status.FAILED)
        self.assertIn("result.summary", session.error_detail)

    def test_fallback_G_persists_unambiguous_recovered_audit(self):
        pipeline, _step = self._create_pipeline()
        session = self._create_session(pipeline)

        def fake_execute(*, agent, **kwargs):
            if agent == self.primary:
                raise RuntimeError("primary audit failure")
            return self._success_output(agent)

        with patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=fake_execute,
        ):
            run_execution_session(session.pk)

        step_run = ExecutionStepRun.objects.get(session=session)
        recovery = step_run.response_payload["fallback_recovery"]
        self.assertEqual(step_run.status, ExecutionStepRun.Status.SUCCESS)
        self.assertEqual(step_run.error_detail, "")
        self.assertTrue(recovery["attempted"])
        self.assertEqual(recovery["primary"]["status"], "failed")
        self.assertEqual(recovery["primary"]["error"]["detail"], "primary audit failure")
        self.assertEqual(recovery["fallback"]["status"], "success")
        self.assertEqual(recovery["final_outcome"], "recovered")

    def test_provider_failure_category_enters_normal_fallback_audit(self):
        pipeline, _step = self._create_pipeline()
        session = self._create_session(pipeline)

        def fake_execute(*, agent, **kwargs):
            if agent == self.primary:
                raise ProviderExecutionError(
                    "provider_unreachable",
                    "Primary provider is unavailable.",
                )
            return self._success_output(agent)

        with patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=fake_execute,
        ):
            run_execution_session(session.pk)

        step_run = ExecutionStepRun.objects.get(session=session)
        recovery = step_run.response_payload["fallback_recovery"]
        self.assertEqual(step_run.status, ExecutionStepRun.Status.SUCCESS)
        self.assertEqual(
            recovery["primary"]["error"]["category"],
            "provider_unreachable",
        )
        self.assertEqual(recovery["fallback"]["status"], "success")

    def test_fallback_H_preserves_both_failures_and_terminal_error(self):
        pipeline, _step = self._create_pipeline()
        session = self._create_session(pipeline)

        def fake_execute(*, agent, **kwargs):
            if agent == self.primary:
                raise RuntimeError("primary failed first")
            raise ValidationError("fallback failed second")

        with patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=fake_execute,
        ) as mocked_execute:
            run_execution_session(session.pk)

        session.refresh_from_db()
        step_run = session.step_runs.get()
        recovery = step_run.response_payload["fallback_recovery"]
        self.assertEqual(mocked_execute.call_count, 2)
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(step_run.status, ExecutionStepRun.Status.FAILED)
        self.assertIn("fallback failed second", step_run.error_detail)
        self.assertEqual(recovery["primary"]["error"]["detail"], "primary failed first")
        self.assertEqual(recovery["fallback"]["error"]["detail"], "['fallback failed second']")
        self.assertEqual(recovery["fallback"]["status"], "failed")
        self.assertEqual(recovery["final_outcome"], "failed")

    def test_fallback_I_stop_policy_never_invokes_configured_fallback(self):
        pipeline, _step = self._create_pipeline(on_error=PipelineStep.OnError.STOP)
        session = self._create_session(pipeline)

        with patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=RuntimeError("primary stop failure"),
        ) as mocked_execute:
            run_execution_session(session.pk)

        session.refresh_from_db()
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(mocked_execute.call_count, 1)
        self.assertEqual(mocked_execute.call_args.kwargs["agent"], self.primary)

    def test_fallback_I_continue_policy_never_invokes_configured_fallback(self):
        pipeline, _step = self._create_pipeline(
            on_error=PipelineStep.OnError.CONTINUE,
            global_output_contract={},
        )
        session = self._create_session(pipeline)

        with patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            side_effect=RuntimeError("optional enrichment failed"),
        ) as mocked_execute:
            run_execution_session(session.pk)

        session.refresh_from_db()
        step_run = session.step_runs.get()
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(step_run.status, ExecutionStepRun.Status.FAILED)
        self.assertIn("optional enrichment failed", step_run.error_detail)
        self.assertEqual(mocked_execute.call_count, 1)
        self.assertEqual(mocked_execute.call_args.kwargs["agent"], self.primary)

    def test_fallback_J_primary_success_never_prepares_or_executes_fallback(self):
        pipeline, _step = self._create_pipeline()
        session = self._create_session(pipeline)

        with patch(
            "ai_hub.services.execution_runner.prepare_agent_payload",
            wraps=prepare_agent_payload,
        ) as mocked_prepare, patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            return_value=self._success_output(self.primary, content="primary result"),
        ) as mocked_execute:
            run_execution_session(session.pk)

        session.refresh_from_db()
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(mocked_prepare.call_count, 1)
        self.assertEqual(mocked_prepare.call_args.args[0], self.primary)
        self.assertEqual(mocked_execute.call_count, 1)
        self.assertEqual(mocked_execute.call_args.kwargs["agent"], self.primary)

    def test_primary_missing_mapped_output_path_is_terminal_failure(self):
        pipeline, _step = self._create_pipeline(
            on_error=PipelineStep.OnError.STOP,
            output_mapping={"summary": "result.summary"},
            global_output_contract={"required": ["summary"]},
        )
        session = self._create_session(pipeline)

        with patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            return_value=self._success_output(self.primary),
        ):
            run_execution_session(session.pk)

        session.refresh_from_db()
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertIn("result.summary", session.error_detail)

    def test_step_preparation_failure_does_not_reuse_previous_step_payload(self):
        second_agent = AgentProfile.objects.create(
            name="preparation-failure-agent",
            role="Preparation failure probe",
            model_config=self.model,
            input_contract={"required": ["source"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        pipeline, _step = self._create_pipeline(
            on_error=PipelineStep.OnError.STOP,
            global_output_contract={},
        )
        PipelineStep.objects.create(
            pipeline=pipeline,
            agent=second_agent,
            order=2,
            input_mapping={"source": "source"},
            on_error=PipelineStep.OnError.STOP,
        )
        session = self._create_session(pipeline)

        def fake_prepare(agent, context, mapping, *, workspace=None):
            if agent == second_agent:
                raise ValidationError("second preparation failed")
            return prepare_agent_payload(
                agent,
                context,
                mapping,
                workspace=workspace,
            )

        with patch(
            "ai_hub.services.execution_runner.prepare_agent_payload",
            side_effect=fake_prepare,
        ), patch(
            "ai_hub.services.execution_runner._execute_session_agent",
            return_value=self._success_output(self.primary),
        ):
            run_execution_session(session.pk)

        session.refresh_from_db()
        second_run = session.step_runs.get(order=2)
        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(second_run.status, ExecutionStepRun.Status.FAILED)
        self.assertEqual(second_run.request_payload, {})
        self.assertIn("second preparation failed", second_run.error_detail)


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

    def test_game_hybrid_session_creation_is_rejected(self):
        with self.assertRaisesMessage(ValidationError, "GAME Hybrid continuation is not enabled"):
            create_execution_session(
                source_object=self.source,
                entry_agent=self.agent,
                runtime_kind=ExecutionSession.RuntimeKind.GAME,
                runtime_mode=ExecutionSession.RuntimeMode.HYBRID,
                goal_text="Unsupported continuation.",
            )

    def test_persisted_game_hybrid_session_is_rejected_by_runner(self):
        session = ExecutionSession.objects.create(
            entry_agent=self.agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.HYBRID,
            goal_text="Unsupported continuation.",
        )

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertIn("GAME Hybrid continuation is not enabled", session.error_detail)
        self.assertEqual(session.final_context["execution_outcome"], "failed")
        self.assertEqual(session.final_context["goal_outcome"], "unknown")
        self.assertEqual(session.final_context["finish_reason"], "failed")
        self.assertEqual(session.step_runs.count(), 0)

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
        orchestrator_action = ToolDefinition.objects.create(
            name="orchestrator-action",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "legacy action", "game_tool_category": "action_tool"},
        )
        self.agent.tools.add(orchestrator_action)
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
            runtime_config={"agent_tool_runtime": "legacy_preexecute"},
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
        self.assertEqual(
            set(session.step_runs.get(order=1).response_payload["tools"]),
            {"orchestrator-action"},
        )

    @override_settings(AI_HUB_DEFAULT_AGENT_TOOL_RUNTIME="resolved")
    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_resolved_orchestrator_executes_toolbox_tool_with_linked_audit(self, mocked_call):
        tool = ToolDefinition.objects.create(
            name="toolbox-only-context",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            input_schema={"required": ["source"]},
            output_schema={"required": ["macro"]},
            config={"template": "resolved context"},
        )
        toolbox = Toolbox.objects.create(
            name="Session Resolver Toolbox",
            slug="session-resolver-toolbox",
            label="Session Resolver Toolbox",
        )
        ToolboxTool.objects.create(toolbox=toolbox, tool=tool)
        AgentToolboxAssignment.objects.create(agent=self.agent, toolbox=toolbox)
        self.pipeline.is_active = True
        self.pipeline.save(update_fields=["is_active"])
        session = create_execution_session(
            source_object=self.source,
            pipeline=self.pipeline,
            triggered_by=self.user,
            initial_context={"source": "resolved input"},
        )
        mocked_call.side_effect = [
            {
                "status": "ok",
                "content": (
                    '{"type":"tool_call","tool_name":"toolbox-only-context",'
                    '"arguments":{"source":"resolved input"}}'
                ),
            },
            {"status": "ok", "content": '{"type":"final","answer":"resolved result"}'},
        ]

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        step_run = session.step_runs.get()
        self.assertEqual(step_run.response_payload["agent_tool_runtime"], "resolved")
        self.assertEqual(step_run.response_payload["llm"]["content"], "resolved result")
        self.assertEqual(step_run.response_payload["tools"], {tool.name: {"macro": "resolved context"}})
        tool_run = ToolExecutionRun.objects.get()
        self.assertEqual(tool_run.session, session)
        self.assertEqual(tool_run.step_run, step_run)
        self.assertEqual(tool_run.agent, self.agent)
        self.assertEqual(tool_run.status, ToolExecutionRun.Status.SUCCESS)
        first_user_payload = json.loads(mocked_call.call_args_list[0].kwargs["messages"][1]["content"])
        self.assertEqual(
            [item["name"] for item in first_user_payload["available_tools"]],
            ["toolbox-only-context"],
        )

    @override_settings(AI_HUB_DEFAULT_AGENT_TOOL_RUNTIME="resolved")
    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_resolved_game_exposes_only_context_tools_and_unwraps_final_decision(self, mocked_call):
        context_tool = ToolDefinition.objects.create(
            name="resolved-game-context",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "safe context", "game_tool_category": "context_tool"},
        )
        action_tool = ToolDefinition.objects.create(
            name="resolved-game-action",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "unsafe action", "game_tool_category": "action_tool"},
        )
        toolbox = Toolbox.objects.create(
            name="Resolved GAME Toolbox",
            slug="resolved-game-toolbox",
            label="Resolved GAME Toolbox",
        )
        ToolboxTool.objects.create(toolbox=toolbox, tool=context_tool, display_order=1)
        ToolboxTool.objects.create(toolbox=toolbox, tool=action_tool, display_order=2)
        game_agent = AgentProfile.objects.create(
            name="resolved-game-agent",
            role="Resolved GAME runner",
            model_config=self.model,
            input_contract={"required": ["goal"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        AgentToolboxAssignment.objects.create(agent=game_agent, toolbox=toolbox)
        session = create_execution_session(
            entry_agent=game_agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            goal_text="Use governed context.",
            runtime_config={"max_iterations": 1, "strict_response_contract": True},
        )
        game_decision = json.dumps(
            {
                "action": "finish",
                "message": "Context checked.",
                "complete": True,
                "final_answer": "Resolved GAME result.",
            }
        )
        mocked_call.side_effect = [
            {
                "status": "ok",
                "content": (
                    '{"type":"tool_call","tool_name":"resolved-game-context",'
                    '"arguments":{}}'
                ),
            },
            {"status": "ok", "content": json.dumps({"type": "final", "answer": game_decision})},
        ]

        run_execution_session(session.id, allow_legacy_game_action_tools=True)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(session.final_context["final_answer"], "Resolved GAME result.")
        step_run = session.step_runs.get()
        self.assertEqual(step_run.response_payload["agent_tool_runtime"], "resolved")
        self.assertEqual(set(step_run.response_payload["tools"]), {"resolved-game-context"})
        self.assertEqual(ToolExecutionRun.objects.get().tool, context_tool)
        first_user_payload = json.loads(mocked_call.call_args_list[0].kwargs["messages"][1]["content"])
        self.assertEqual(
            [item["name"] for item in first_user_payload["available_tools"]],
            ["resolved-game-context"],
        )

    @override_settings(AI_HUB_DEFAULT_AGENT_TOOL_RUNTIME="resolved")
    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_resolved_runner_excludes_unresumable_approval_tools(self, mocked_call):
        tool = ToolDefinition.objects.create(
            name="generic-approval-tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            requires_approval=True,
            config={"template": "must not run"},
        )
        toolbox = Toolbox.objects.create(
            name="Approval Toolbox",
            slug="approval-toolbox",
            label="Approval Toolbox",
        )
        ToolboxTool.objects.create(toolbox=toolbox, tool=tool)
        AgentToolboxAssignment.objects.create(agent=self.agent, toolbox=toolbox)
        self.pipeline.is_active = True
        self.pipeline.save(update_fields=["is_active"])
        session = create_execution_session(
            source_object=self.source,
            pipeline=self.pipeline,
            triggered_by=self.user,
            initial_context={"source": "approval input"},
        )
        mocked_call.return_value = {
            "status": "ok",
            "content": (
                '{"type":"tool_call","tool_name":"generic-approval-tool",'
                '"arguments":{}}'
            ),
        }

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertIn("not available", session.error_detail)
        self.assertEqual(ToolExecutionRun.objects.count(), 0)
        user_payload = json.loads(mocked_call.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(user_payload["available_tools"], [])

    @override_settings(AI_HUB_DEFAULT_AGENT_TOOL_RUNTIME="resolved")
    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_invalid_agent_tool_runtime_fails_before_creating_steps(self, mocked_call):
        self.pipeline.is_active = True
        self.pipeline.save(update_fields=["is_active"])
        session = create_execution_session(
            source_object=self.source,
            pipeline=self.pipeline,
            triggered_by=self.user,
            runtime_config={"agent_tool_runtime": "surprise"},
            initial_context={"source": "invalid runtime"},
        )

        run_execution_session(session.id)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertIn("Unknown agent tool runtime 'surprise'", session.error_detail)
        self.assertEqual(session.step_runs.count(), 0)
        mocked_call.assert_not_called()

    @override_settings(AI_HUB_DEFAULT_AGENT_TOOL_RUNTIME="resolved")
    def test_training_provider_smoke_keeps_existing_orchestrator_and_game_outputs(self):
        provider = ProviderConfig.objects.create(
            name="resolved-training-smoke-provider",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        model = ModelConfig.objects.create(provider=provider, model_name="training")
        tool = ToolDefinition.objects.create(
            name="resolved-training-smoke-tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "smoke context", "game_tool_category": "context_tool"},
        )
        toolbox = Toolbox.objects.create(
            name="Resolved Training Smoke Toolbox",
            slug="resolved-training-smoke-toolbox",
            label="Resolved Training Smoke Toolbox",
        )
        ToolboxTool.objects.create(toolbox=toolbox, tool=tool)

        orchestrator_agent = AgentProfile.objects.create(
            name="resolved-training-orchestrator",
            role="Resolved training smoke",
            model_config=model,
            system_prompt="Return a concise generic result.",
            input_contract={"required": ["source"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        AgentToolboxAssignment.objects.create(agent=orchestrator_agent, toolbox=toolbox)
        pipeline = PipelineDefinition.objects.create(
            name="resolved-training-smoke-pipeline",
            is_active=False,
        )
        PipelineStep.objects.create(
            pipeline=pipeline,
            agent=orchestrator_agent,
            order=1,
            input_mapping={"source": "source"},
            output_mapping={"result": "llm.content"},
        )
        pipeline.is_active = True
        pipeline.save(update_fields=["is_active"])
        orchestrator_session = create_execution_session(
            pipeline=pipeline,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            initial_context={"source": "smoke"},
        )

        game_agent = AgentProfile.objects.create(
            name="resolved-training-game",
            role="Resolved GAME smoke",
            model_config=model,
            system_prompt=(
                "You are a GAME agent. Return action, message, complete and final_answer."
            ),
            input_contract={"required": ["goal"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        AgentToolboxAssignment.objects.create(agent=game_agent, toolbox=toolbox)
        game_session = create_execution_session(
            entry_agent=game_agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            goal_text="Complete the resolved runtime smoke.",
            runtime_config={"max_iterations": 1, "strict_response_contract": True},
        )

        run_execution_session(orchestrator_session.pk)
        run_execution_session(game_session.pk)
        orchestrator_session.refresh_from_db()
        game_session.refresh_from_db()

        self.assertEqual(orchestrator_session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(game_session.status, ExecutionSession.Status.SUCCESS)
        orchestrator_response = orchestrator_session.step_runs.get().response_payload
        game_response = game_session.step_runs.get().response_payload
        self.assertEqual(orchestrator_response["agent_tool_runtime"], "resolved")
        self.assertEqual(game_response["agent_tool_runtime"], "resolved")
        self.assertEqual(orchestrator_response["model_response_mode"], "plain_final_compatibility")
        self.assertEqual(game_response["model_response_mode"], "plain_final_compatibility")
        self.assertEqual(
            [item["name"] for item in orchestrator_response["tool_manifest"]],
            [tool.name],
        )
        self.assertEqual(
            [item["name"] for item in game_response["tool_manifest"]],
            [tool.name],
        )
        self.assertEqual(game_session.final_context["finish_reason"], "agent_finished")

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
        self.assertEqual(session.final_context["execution_outcome"], "completed")
        self.assertEqual(session.final_context["goal_outcome"], "achieved")
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
        self.assertEqual(session.final_context["execution_outcome"], "completed")
        self.assertEqual(session.final_context["goal_outcome"], "incomplete")
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
        self.assertEqual(session.final_context["execution_outcome"], "failed")
        self.assertEqual(session.final_context["goal_outcome"], "unknown")
        self.assertEqual(session.final_context["finish_reason"], "failed")

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
        self.assertEqual(session.final_context["execution_outcome"], "failed")
        self.assertEqual(session.final_context["goal_outcome"], "unknown")
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

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_game_auto_executes_only_context_tools(self, mocked_call):
        context_tool = ToolDefinition.objects.create(
            name="safe-context",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "safe", "game_tool_category": "context_tool"},
        )
        action_tool = ToolDefinition.objects.create(
            name="write-action",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "write", "game_tool_category": "action_tool"},
        )
        unknown_tool = ToolDefinition.objects.create(
            name="unknown-category",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "unknown"},
        )
        game_agent = AgentProfile.objects.create(
            name="game-agent-tools-safe",
            role="Autonomous goal runner",
            model_config=self.model,
            input_contract={"required": ["goal"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        game_agent.tools.add(context_tool, action_tool, unknown_tool)
        session = create_execution_session(
            entry_agent=game_agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Use safe context only.",
            runtime_config={
                "max_iterations": 1,
                "agent_tool_runtime": "legacy_preexecute",
            },
        )
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"action": "finish", "final_answer": "Done."}',
        }

        run_execution_session(session.id)
        session.refresh_from_db()

        tools = session.step_runs.get().response_payload["tools"]
        self.assertEqual(set(tools), {"safe-context"})

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_game_legacy_action_tool_opt_in_is_explicit(self, mocked_call):
        action_tool = ToolDefinition.objects.create(
            name="legacy-action",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "legacy", "game_tool_category": "action_tool"},
        )
        game_agent = AgentProfile.objects.create(
            name="game-agent-tools-opt-in",
            role="Autonomous goal runner",
            model_config=self.model,
            input_contract={"required": ["goal"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        game_agent.tools.add(action_tool)
        session = create_execution_session(
            entry_agent=game_agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Run trusted legacy action.",
            runtime_config={
                "max_iterations": 1,
                "agent_tool_runtime": "legacy_preexecute",
            },
        )
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"action": "finish", "final_answer": "Done."}',
        }

        run_execution_session(session.id, allow_legacy_game_action_tools=True)
        session.refresh_from_db()

        self.assertEqual(set(session.step_runs.get().response_payload["tools"]), {"legacy-action"})

    def test_run_execution_session_rejects_session_that_is_already_running(self):
        session = create_execution_session(
            source_object=self.source,
            pipeline=self.pipeline,
            triggered_by=self.user,
            initial_context={"source": "plain reusable input"},
        )
        session.status = ExecutionSession.Status.RUNNING
        session.save(update_fields=["status"])

        with self.assertRaisesMessage(ValidationError, "must be pending"):
            run_execution_session(session.id)

    def test_run_execution_session_rejects_every_non_pending_status(self):
        for index, status in enumerate(
            (
                ExecutionSession.Status.FAILED,
                ExecutionSession.Status.CANCELLED,
                ExecutionSession.Status.SUCCESS,
                ExecutionSession.Status.WAITING_ASYNC,
            ),
            start=1,
        ):
            with self.subTest(status=status):
                session = create_execution_session(
                    source_object=self.source,
                    pipeline=self.pipeline,
                    triggered_by=self.user,
                    source_label=f"non-pending-{index}",
                )
                session.status = status
                session.save(update_fields=["status"])
                with self.assertRaisesMessage(ValidationError, "must be pending"):
                    run_execution_session(session.id)


class HubToolSafetyTests(TestCase):
    def test_python_context_tool_requires_explicit_read_only_declaration(self):
        undeclared = ToolDefinition(
            name="undeclared-reader",
            tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
            config={
                "callable": "academy.tools.doc_search.search_docs",
                "game_tool_category": "context_tool",
            },
        )
        declared = ToolDefinition(
            name="declared-reader",
            tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
            config={
                "callable": "academy.tools.doc_search.search_docs",
                "game_tool_category": "context_tool",
                "read_only": True,
            },
        )

        self.assertEqual(get_game_tool_category(undeclared), GAME_ACTION_TOOL)
        self.assertEqual(get_game_tool_category(declared), GAME_CONTEXT_TOOL)

    @override_settings(AI_HUB_ALLOWED_TOOL_CALLABLES=())
    def test_python_callable_outside_allow_list_is_rejected(self):
        tool = ToolDefinition.objects.create(
            name="untrusted-callable",
            tool_kind=ToolDefinition.ToolKind.PYTHON_CALLABLE,
            config={"callable": "academy.tools.doc_search.search_docs"},
        )

        with self.assertRaisesMessage(ValidationError, "AI_HUB_ALLOWED_TOOL_CALLABLES"):
            execute_tools([tool], {})

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_tool_rejects_host_outside_its_allow_list(self, mocked_request):
        tool = ToolDefinition.objects.create(
            name="blocked-http",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            config={
                "url": "http://127.0.0.1:9000/private",
                "method": "GET",
                "allowed_hosts": ["example.com"],
            },
        )

        with self.assertRaisesMessage(ValidationError, "HTTP host is not explicitly allowed"):
            execute_tools([tool], {})
        mocked_request.assert_not_called()

    def test_http_read_operation_accepts_only_get_and_head(self):
        for method in ("GET", "HEAD"):
            with self.subTest(method=method):
                tool = ToolDefinition(
                    name=f"read-{method.lower()}",
                    tool_kind=ToolDefinition.ToolKind.HTTP,
                    operation_mode=ToolDefinition.OperationMode.READ,
                    config={
                        "url": "https://allowed.example/resource",
                        "method": method,
                        "allowed_hosts": ["allowed.example"],
                    },
                )
                tool.full_clean()

    def test_http_read_operation_rejects_write_capable_methods(self):
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            with self.subTest(method=method):
                tool = ToolDefinition(
                    name=f"invalid-read-{method.lower()}",
                    tool_kind=ToolDefinition.ToolKind.HTTP,
                    operation_mode=ToolDefinition.OperationMode.READ,
                    config={
                        "url": "https://allowed.example/resource",
                        "method": method,
                        "allowed_hosts": ["allowed.example"],
                    },
                )
                with self.assertRaisesMessage(
                    ValidationError,
                    "HTTP tools with operation_mode READ must use GET or HEAD",
                ):
                    tool.full_clean()

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_runtime_rejects_invalid_read_configuration_before_request(self, mocked_request):
        tool = ToolDefinition.objects.create(
            name="runtime-invalid-read-post",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/resource",
                "method": "POST",
                "allowed_hosts": ["allowed.example"],
            },
        )

        with self.assertRaisesMessage(
            ValidationError,
            "HTTP tools with operation_mode READ must use GET or HEAD",
        ):
            execute_tool(tool, {"value": "must not leave"})
        mocked_request.assert_not_called()

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_writable_configuration_still_executes(self, mocked_request):
        response = StreamingHttpResponse(
            status_code=200,
            headers={},
            body=b"written",
        )
        mocked_request.return_value = response
        tool = ToolDefinition(
            name="valid-state-write-post",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.STATE_WRITE,
            config={
                "url": "https://allowed.example/resource",
                "method": "POST",
                "allowed_hosts": ["allowed.example"],
            },
        )
        tool.full_clean()
        tool.save()

        result = execute_tool(tool, {"value": "allowed"})

        self.assertEqual(result, {"status_code": 200, "body": "written"})
        mocked_request.assert_called_once()
        self.assertTrue(mocked_request.call_args.kwargs["stream"])
        self.assertEqual(mocked_request.call_args.kwargs["timeout"], 30)
        self.assertTrue(response.closed)

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_content_length_over_byte_limit_fails_before_body_read(
        self,
        mocked_request,
    ):
        response = StreamingHttpResponse(
            headers={"Content-Length": "1025"},
            body=b"x" * 1025,
        )
        mocked_request.return_value = response
        tool = ToolDefinition.objects.create(
            name="content-length-byte-limit",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/large",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
                "max_response_bytes": 1024,
            },
        )

        with self.assertRaisesMessage(ValidationError, "maximum response size"):
            execute_tool(tool, {})

        self.assertEqual(response.raw.read_sizes, [])
        self.assertTrue(response.closed)

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_unknown_length_stream_stops_at_byte_limit(self, mocked_request):
        response = StreamingHttpResponse(body=b"x" * 4096)
        mocked_request.return_value = response
        tool = ToolDefinition.objects.create(
            name="unknown-length-byte-limit",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/large",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
                "max_response_bytes": 1024,
            },
        )

        with self.assertRaisesMessage(ValidationError, "maximum response size"):
            execute_tool(tool, {})

        self.assertTrue(response.raw.read_sizes)
        self.assertLessEqual(max(response.raw.read_sizes), 1025)
        self.assertTrue(response.closed)

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_exact_byte_limit_succeeds(self, mocked_request):
        response = StreamingHttpResponse(body=b"x" * 1024)
        mocked_request.return_value = response
        tool = ToolDefinition.objects.create(
            name="exact-response-byte-limit",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/exact",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
                "max_response_bytes": 1024,
            },
        )

        result = execute_tool(tool, {})

        self.assertEqual(len(result["body"]), 1024)
        self.assertTrue(response.closed)

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_max_bytes_plus_one_fails(self, mocked_request):
        response = StreamingHttpResponse(body=b"x" * 1025)
        mocked_request.return_value = response
        tool = ToolDefinition.objects.create(
            name="plus-one-response-byte-limit",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/plus-one",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
                "max_response_bytes": 1024,
            },
        )

        with self.assertRaisesMessage(ValidationError, "maximum response size"):
            execute_tool(tool, {})

        self.assertTrue(response.closed)

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_redirect_response_closes_before_next_request(self, mocked_request):
        redirect_response = StreamingHttpResponse(
            status_code=302,
            headers={"Location": "/final"},
            body=b"redirect body must not be read",
        )
        final_response = StreamingHttpResponse(body=b"ok")
        responses = [redirect_response, final_response]

        def next_response(*args, **kwargs):
            if len(responses) == 1:
                self.assertTrue(redirect_response.closed)
            return responses.pop(0)

        mocked_request.side_effect = next_response
        tool = ToolDefinition.objects.create(
            name="close-before-redirect",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/start",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
            },
        )

        result = execute_tool(tool, {})

        self.assertEqual(result["body"], "ok")
        self.assertEqual(redirect_response.raw.read_sizes, [])
        self.assertTrue(redirect_response.closed)
        self.assertTrue(final_response.closed)

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_large_error_response_is_bounded(self, mocked_request):
        response = StreamingHttpResponse(status_code=500, body=b"e" * 4096)
        mocked_request.return_value = response
        tool = ToolDefinition.objects.create(
            name="bounded-http-error",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/error",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
                "max_response_bytes": 1024,
            },
        )

        with self.assertRaisesMessage(ValidationError, "maximum response size"):
            execute_tool(tool, {})

        self.assertLessEqual(max(response.raw.read_sizes), 1025)
        self.assertTrue(response.closed)

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_unknown_encoding_uses_safe_fallback(self, mocked_request):
        mocked_request.return_value = StreamingHttpResponse(
            body=b"\xffsafe",
            encoding="not-a-real-codec",
        )
        tool = ToolDefinition.objects.create(
            name="safe-http-decoding",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/encoded",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
            },
        )

        result = execute_tool(tool, {})

        self.assertIn("safe", result["body"])

    def test_http_response_byte_limit_is_clamped_and_malformed_values_fail(self):
        from ai_hub.services.http_tool_policy import (
            MAX_HTTP_RESPONSE_BYTES,
            MIN_HTTP_RESPONSE_BYTES,
            build_http_tool_configuration,
        )

        base_config = {
            "url": "https://allowed.example/resource",
            "method": "GET",
            "allowed_hosts": ["allowed.example"],
        }
        too_small = ToolDefinition(
            name="http-min-byte-clamp",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={**base_config, "max_response_bytes": 0},
        )
        too_large = ToolDefinition(
            name="http-max-byte-clamp",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={**base_config, "max_response_bytes": MAX_HTTP_RESPONSE_BYTES + 1},
        )
        malformed = ToolDefinition(
            name="http-malformed-byte-limit",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={**base_config, "max_response_bytes": "unlimited"},
        )

        self.assertEqual(
            build_http_tool_configuration(too_small).max_response_bytes,
            MIN_HTTP_RESPONSE_BYTES,
        )
        self.assertEqual(
            build_http_tool_configuration(too_large).max_response_bytes,
            MAX_HTTP_RESPONSE_BYTES,
        )
        with self.assertRaisesMessage(ValidationError, "max_response_bytes"):
            build_http_tool_configuration(malformed)

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_redirect_within_allowed_host_succeeds(self, mocked_request):
        mocked_request.side_effect = [
            StreamingHttpResponse(
                status_code=302,
                headers={"Location": "/final"},
                body=b"redirect",
            ),
            StreamingHttpResponse(status_code=200, headers={}, body=b"ok"),
        ]
        tool = ToolDefinition.objects.create(
            name="same-host-redirect",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/start",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
            },
        )

        result = execute_tool(tool, {})

        self.assertEqual(result["status_code"], 200)
        self.assertEqual(mocked_request.call_count, 2)
        self.assertEqual(
            [call.args[1] for call in mocked_request.call_args_list],
            ["https://allowed.example/start", "https://allowed.example/final"],
        )

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_redirect_to_second_allowed_host_succeeds(self, mocked_request):
        mocked_request.side_effect = [
            StreamingHttpResponse(
                status_code=307,
                headers={"Location": "https://second.example/final"},
                body=b"redirect",
            ),
            StreamingHttpResponse(status_code=200, headers={}, body=b"ok"),
        ]
        tool = ToolDefinition.objects.create(
            name="second-host-redirect",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/start",
                "method": "GET",
                "allowed_hosts": ["allowed.example", "second.example"],
            },
        )

        result = execute_tool(tool, {})

        self.assertEqual(result["status_code"], 200)
        self.assertEqual(mocked_request.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["allow_redirects"] is False
                for call in mocked_request.call_args_list
            )
        )

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_cross_origin_redirect_drops_sensitive_headers(self, mocked_request):
        mocked_request.side_effect = [
            StreamingHttpResponse(
                status_code=302,
                headers={"Location": "https://second.example/final"},
                body=b"redirect",
            ),
            StreamingHttpResponse(status_code=200, headers={}, body=b"ok"),
        ]
        tool = ToolDefinition.objects.create(
            name="credential-safe-redirect",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/start",
                "method": "GET",
                "allowed_hosts": ["allowed.example", "second.example"],
                "headers": {
                    "Authorization": "Bearer secret",
                    "Cookie": "session=secret",
                    "X-Public": "kept",
                },
            },
        )

        execute_tool(tool, {})

        second_headers = mocked_request.call_args_list[1].kwargs["headers"]
        self.assertEqual(second_headers, {"X-Public": "kept"})

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_redirect_to_forbidden_host_stops_before_contact(self, mocked_request):
        mocked_request.return_value = StreamingHttpResponse(
            status_code=302,
            headers={"Location": "https://forbidden.example/private"},
            body=b"redirect",
        )
        tool = ToolDefinition.objects.create(
            name="forbidden-host-redirect",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/start",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
            },
        )

        with self.assertRaisesMessage(ValidationError, "HTTP host is not explicitly allowed"):
            execute_tool(tool, {})
        self.assertEqual(mocked_request.call_count, 1)

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_redirect_to_loopback_stops_before_contact(self, mocked_request):
        mocked_request.return_value = StreamingHttpResponse(
            status_code=302,
            headers={"Location": "http://127.0.0.1:9000/private"},
            body=b"redirect",
        )
        tool = ToolDefinition.objects.create(
            name="loopback-redirect",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/start",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
            },
        )

        with self.assertRaisesMessage(ValidationError, "HTTP host is not explicitly allowed"):
            execute_tool(tool, {})
        self.assertEqual(mocked_request.call_count, 1)

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_redirect_limit_terminates_loop(self, mocked_request):
        mocked_request.side_effect = [
            StreamingHttpResponse(status_code=302, headers={"Location": "/two"}),
            StreamingHttpResponse(status_code=302, headers={"Location": "/three"}),
            StreamingHttpResponse(status_code=302, headers={"Location": "/four"}),
        ]
        tool = ToolDefinition.objects.create(
            name="redirect-loop",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/one",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
                "max_redirects": 2,
            },
        )

        with self.assertRaisesMessage(ValidationError, "exceeded its redirect limit"):
            execute_tool(tool, {})
        self.assertEqual(mocked_request.call_count, 3)

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_unsupported_initial_scheme_fails_before_request(self, mocked_request):
        tool = ToolDefinition.objects.create(
            name="unsupported-http-scheme",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "file://allowed.example/private",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
            },
        )

        with self.assertRaisesMessage(ValidationError, "HTTP URL scheme must be http or https"):
            execute_tool(tool, {})
        mocked_request.assert_not_called()

    @patch("ai_hub.services.tools_runtime.requests.request")
    def test_http_redirect_to_unsupported_scheme_stops_before_contact(self, mocked_request):
        mocked_request.return_value = StreamingHttpResponse(
            status_code=302,
            headers={"Location": "file://allowed.example/private"},
            body=b"redirect",
        )
        tool = ToolDefinition.objects.create(
            name="unsupported-redirect-scheme",
            tool_kind=ToolDefinition.ToolKind.HTTP,
            operation_mode=ToolDefinition.OperationMode.READ,
            config={
                "url": "https://allowed.example/start",
                "method": "GET",
                "allowed_hosts": ["allowed.example"],
            },
        )

        with self.assertRaisesMessage(ValidationError, "HTTP URL scheme must be http or https"):
            execute_tool(tool, {})
        self.assertEqual(mocked_request.call_count, 1)


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
            provider_type=ProviderConfig.ProviderType.OLLAMA,
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
    def test_control_center_graph_uses_resolved_tool_access(self, mocked_get):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"models": [{"name": "qwen3:8b"}]}

        mocked_get.return_value = Response()
        tool = ToolDefinition.objects.create(
            name="graph-toolbox-only",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
        )
        toolbox = Toolbox.objects.create(
            name="Graph Resolver Toolbox",
            slug="graph-resolver-toolbox",
            label="Graph Resolver Toolbox",
        )
        ToolboxTool.objects.create(toolbox=toolbox, tool=tool)
        AgentToolboxAssignment.objects.create(agent=self.agent, toolbox=toolbox)

        context = build_control_center_context()

        edges = {
            (edge["source"], edge["target"], edge["label"])
            for edge in context["graph"]["edges"]
        }
        pipeline_scope = context["graph"]["pipelineScopes"][0]
        self.assertIn(
            (f"tool:{tool.id}", f"agent:{self.agent.id}", "enables"),
            edges,
        )
        self.assertIn(f"tool:{tool.id}", pipeline_scope["node_ids"])

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

    def test_workspace_requires_execution_session_view_permission(self):
        staff_user = get_user_model().objects.create_user(
            username="pipeline-only-staff",
            password="testpass123",
            is_staff=True,
        )
        staff_user.user_permissions.add(Permission.objects.get(codename="view_pipelinedefinition"))
        client = Client()
        client.force_login(staff_user)

        response = client.get(reverse("admin:ai_hub_workspace_game"))

        self.assertEqual(response.status_code, 403)

    def test_ai_hub_app_index_shows_two_workspaces(self):
        client = Client()
        client.force_login(self.user)

        response = client.get(reverse("admin:app_list", kwargs={"app_label": "ai_hub"}))

        self.assertEqual(response.status_code, 200)
        # five-area navigation map (IA Step 3)
        self.assertContains(response, "Foundation")
        self.assertContains(response, "Operations")
        self.assertContains(response, "Recommended next")
        self.assertContains(response, "Setup progress")
        self.assertContains(response, "Start from a pattern")
        self.assertContains(response, "Open Orchestrator")
        self.assertContains(response, "Open GAME")
        self.assertContains(response, "All records")
        # root entities stay listed in the "All records" fallback
        self.assertContains(response, "Agent profiles")
        self.assertContains(response, "Pipeline definitions")
        # supporting models are accessible via the hidden-models toggle (not in the main Django table)
        self.assertContains(response, "Show supporting tables")
        self.assertContains(response, "hidden-models-section")
        self.assertContains(response, "Pipeline steps")
        self.assertContains(response, "Execution step runs")

    def test_demoted_model_changelist_still_reachable(self):
        """Hidden-from-index models keep working URLs (registered, perms intact)."""
        client = Client()
        client.force_login(self.user)
        # PipelineStep is demoted from the index but its changelist must still resolve
        response = client.get(reverse("admin:ai_hub_pipelinestep_changelist"))
        self.assertEqual(response.status_code, 200)

    def test_agent_composer_change_page_is_composed(self):
        """IA Step 4: AgentProfile change page renders the composed workspace shell."""
        client = Client()
        client.force_login(self.user)
        response = client.get(reverse("admin:ai_hub_agentprofile_change", args=[self.agent.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Agent Composer")
        self.assertContains(response, "data-ws-tabs")          # tab nav
        self.assertContains(response, 'data-tab="overview"')   # overview panel
        self.assertContains(response, 'data-tab="config"')     # editable form wrapped in a tab
        self.assertContains(response, "Used in pipelines")
        self.assertContains(response, self.pipeline.name)      # overview lists the pipeline that uses the agent
        self.assertContains(response, 'name="name"')           # editable field still present inside the form

    def test_agent_composer_save_still_works(self):
        """IA Step 4: wrapping fieldsets/inlines in tab panels must not break submission."""
        client = Client()
        client.force_login(self.user)
        data = {
            "name": "visual-agent",
            "role": "Updated role",
            "model_config": self.model.id,
            "execution_mode": "inherit",
            "knowledge_max_chars": "6000",
            "system_prompt": "Updated prompt",
            "input_contract": '{"required": ["dream_id"]}',
            "output_contract": '{"required": ["agent"]}',
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
        }
        response = client.post(reverse("admin:ai_hub_agentprofile_change", args=[self.agent.id]), data)
        self.assertEqual(response.status_code, 302)  # redirect = saved successfully
        self.agent.refresh_from_db()
        self.assertEqual(self.agent.role, "Updated role")

    def test_orchestrator_designer_change_page_is_composed(self):
        """IA Step 4: PipelineDefinition change page renders the composed designer shell."""
        client = Client()
        client.force_login(self.user)
        response = client.get(reverse("admin:ai_hub_pipelinedefinition_change", args=[self.pipeline.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Orchestrator")
        self.assertContains(response, "data-ws-tabs")
        self.assertContains(response, 'data-tab="overview"')
        self.assertContains(response, 'data-tab="steps"')
        self.assertContains(response, self.agent.name)  # step's agent listed in overview

    def test_game_workspace_change_page_is_composed(self):
        """IA Step 4: GameWorkspace change page renders the composed workspace shell."""
        from ai_hub.models import GameWorkspace
        workspace = GameWorkspace.objects.create(
            name="Composed WS",
            default_policy={"allowed_actions": ["submit_for_approval"], "safety": {"allow_external_writes": False}},
        )
        client = Client()
        client.force_login(self.user)
        response = client.get(reverse("admin:ai_hub_gameworkspace_change", args=[workspace.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-ws-tabs")
        self.assertContains(response, "Open dashboard")
        self.assertContains(response, "Goals by status")

    def test_goal_detail_change_page_is_composed(self):
        """IA Step 4: GameGoal change page renders the composed goal-detail shell."""
        from ai_hub.models import GameWorkspace, GameGoal
        workspace = GameWorkspace.objects.create(name="Goal WS")
        goal = GameGoal.objects.create(workspace=workspace, title="Composed goal", description="desc")
        client = Client()
        client.force_login(self.user)
        response = client.get(reverse("admin:ai_hub_gamegoal_change", args=[goal.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Goal Detail")
        self.assertContains(response, "data-ws-tabs")
        self.assertContains(response, 'data-tab="activity"')
        self.assertContains(response, 'data-tab="planmem"')

    def test_connectivity_provider_change_page_is_composed(self):
        """IA Step 6: ProviderConfig change page renders the composed Connectivity shell."""
        client = Client()
        client.force_login(self.user)
        response = client.get(reverse("admin:ai_hub_providerconfig_change", args=[self.provider.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Connectivity")
        self.assertContains(response, "data-ws-tabs")
        self.assertContains(response, 'data-tab="overview"')
        self.assertContains(response, 'data-tab="config"')
        self.assertContains(response, self.model.model_name)   # model listed in overview
        self.assertContains(response, self.agent.name)         # agent reachable via this provider
        self.assertContains(response, 'name="name"')           # editable field still present

    def test_connectivity_provider_save_still_works(self):
        """IA Step 6: wrapping the provider form in a tab panel must not break submission."""
        client = Client()
        client.force_login(self.user)
        data = {
            "name": "Ollama LAN renamed",
            "provider_type": ProviderConfig.ProviderType.OLLAMA,
            "is_active": "on",
            "base_url": "http://localhost:11434",
            "default_timeout": "60",
            "api_key_env_var": "",
            "_save": "Save",
        }
        response = client.post(reverse("admin:ai_hub_providerconfig_change", args=[self.provider.id]), data)
        self.assertEqual(response.status_code, 302)
        self.provider.refresh_from_db()
        self.assertEqual(self.provider.name, "Ollama LAN renamed")

    def test_knowledge_collection_change_page_is_composed(self):
        """IA Step 6: KnowledgeCollection change page renders the composed Library shell."""
        from ai_hub.models import KnowledgeCollection, KnowledgeDocument
        collection = KnowledgeCollection.objects.create(name="Support rules", is_active=True)
        KnowledgeDocument.objects.create(
            collection=collection, title="Refund policy", status=KnowledgeDocument.Status.ACTIVE
        )
        self.agent.knowledge_collections.add(collection)
        client = Client()
        client.force_login(self.user)
        response = client.get(reverse("admin:ai_hub_knowledgecollection_change", args=[collection.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Knowledge Library")
        self.assertContains(response, "data-ws-tabs")
        self.assertContains(response, 'data-tab="overview"')
        self.assertContains(response, "Refund policy")        # document listed in overview
        self.assertContains(response, self.agent.name)        # agent using the collection

    def test_knowledge_collection_save_still_works(self):
        """IA Step 6: inlines wrapped in a tab panel must still submit (document formset)."""
        from ai_hub.models import KnowledgeCollection
        collection = KnowledgeCollection.objects.create(name="Editable collection", is_active=True)
        client = Client()
        client.force_login(self.user)
        data = {
            "name": "Editable collection renamed",
            "description": "updated",
            "is_active": "on",
            "documents-TOTAL_FORMS": "0",
            "documents-INITIAL_FORMS": "0",
            "documents-MIN_NUM_FORMS": "0",
            "documents-MAX_NUM_FORMS": "1000",
            "_save": "Save",
        }
        response = client.post(reverse("admin:ai_hub_knowledgecollection_change", args=[collection.id]), data)
        self.assertEqual(response.status_code, 302)
        collection.refresh_from_db()
        self.assertEqual(collection.name, "Editable collection renamed")

    def test_tool_registry_toolbox_change_page_is_composed(self):
        """IA Step 6: Toolbox change page renders the composed Tool Registry shell."""
        from ai_hub.models import Toolbox, ToolboxTool, ToolDefinition
        tool = ToolDefinition.objects.create(name="fetch_profile", label="Fetch profile", is_active=True)
        toolbox = Toolbox.objects.create(name="support_box", slug="support-box", label="Support box", is_active=True)
        ToolboxTool.objects.create(toolbox=toolbox, tool=tool, is_enabled=True, display_order=1)
        client = Client()
        client.force_login(self.user)
        response = client.get(reverse("admin:ai_hub_toolbox_change", args=[toolbox.id]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Tool Registry")
        self.assertContains(response, "data-ws-tabs")
        self.assertContains(response, 'data-tab="overview"')
        self.assertContains(response, "Fetch profile")        # tool listed in overview

    def test_provider_health_rejects_non_http_scheme(self):
        """P0: the live health check refuses non-http(s) base URLs (e.g. file://)."""
        from ai_hub.services.admin_control_center import (
            validate_provider_health_endpoint,
        )
        ok, reason = validate_provider_health_endpoint("file:///etc/passwd")
        self.assertFalse(ok)
        self.assertIn("http", reason)

    @override_settings(AI_HUB_PROVIDER_HEALTH_ALLOWED_HOSTS=("ollama.internal",))
    def test_provider_health_allowlist_blocks_other_hosts(self):
        """P0: with an allow-list set, hosts outside it are rejected before any request."""
        from ai_hub.services.admin_control_center import (
            validate_provider_health_endpoint,
        )
        ok, reason = validate_provider_health_endpoint("http://localhost:11434")
        self.assertFalse(ok)
        self.assertIn("not in", reason)

    @override_settings(AI_HUB_PROVIDER_HEALTH_ALLOWED_HOSTS=("ollama.internal",))
    @patch("ai_hub.services.admin_control_center.requests.get")
    def test_provider_health_disallowed_host_makes_no_request(self, mocked_get):
        """P0 (SSRF): a disallowed host must NOT trigger an outbound HTTP request."""
        from ai_hub.services.admin_control_center import _fetch_provider_health
        cache.clear()
        health = _fetch_provider_health(self.provider)  # base_url is http://localhost:11434
        self.assertEqual(health.status, "warning")
        mocked_get.assert_not_called()

    def test_operations_inbox_renders_empty(self):
        """IA Step 5: inbox renders the queue shell and an all-clear state when idle."""
        client = Client()
        client.force_login(self.user)
        response = client.get(reverse("admin:ai_hub_operations_inbox"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Needs a human")
        self.assertContains(response, "inbox-filter")
        self.assertContains(response, "All clear")

    def test_operations_inbox_lists_failed_session(self):
        """IA Step 5: a failed session surfaces as an inbox item under Failures."""
        session = create_execution_session(
            source_label="Boom session",
            entry_agent=self.agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="trigger a failure",
        )
        session.status = ExecutionSession.Status.FAILED
        session.error_detail = "kaboom"
        session.save(update_fields=["status", "error_detail"])
        client = Client()
        client.force_login(self.user)
        response = client.get(reverse("admin:ai_hub_operations_inbox"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Boom session")
        self.assertContains(response, 'data-cat="failure"')
        self.assertNotContains(response, "All clear")

    def test_operations_inbox_requires_execution_permission(self):
        """IA Step 5: inbox is gated behind view_executionsession like the workspaces."""
        weak_staff = get_user_model().objects.create_user(
            username="weakstaff", password="pw", is_staff=True
        )
        client = Client()
        client.force_login(weak_staff)
        response = client.get(reverse("admin:ai_hub_operations_inbox"))
        self.assertEqual(response.status_code, 403)

    def test_operations_inbox_hides_approve_controls_without_approval_permission(self):
        """P0 authz: a viewer (view perms, no approve_game_action) can open the inbox
        but `can_approve` is False, so no inline Approve/Reject controls are offered."""
        from django.contrib.auth.models import Permission
        viewer = get_user_model().objects.create_user(
            username="inbox_viewer", password="pw", is_staff=True
        )
        viewer.user_permissions.add(
            Permission.objects.get(codename="view_pipelinedefinition"),
            Permission.objects.get(codename="view_executionsession"),
        )
        client = Client()
        client.force_login(viewer)
        response = client.get(reverse("admin:ai_hub_operations_inbox"))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["can_approve"])

    def test_operations_inbox_shows_approve_controls_for_approver(self):
        """P0 authz: a user holding approve_game_action gets `can_approve` True."""
        client = Client()
        client.force_login(self.user)  # superuser → holds approve_game_action
        response = client.get(reverse("admin:ai_hub_operations_inbox"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["can_approve"])

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
        self.assertContains(response, "active")
        self.assertContains(response, "View sessions")

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

    @patch("ai_hub.services.admin_control_center.requests.get")
    def test_control_center_renders_mission_deck(self, mocked_get):
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
        # new console shell + graph engine hooks (built only from existing context)
        self.assertContains(response, 'class="ai-console"')
        self.assertContains(response, "data-mc-graph")
        self.assertContains(response, "data-mc-stage-inner")
        self.assertContains(response, "data-mc-edges")
        self.assertContains(response, 'id="ai-control-graph-data"')
        # progressive-disclosure tabs + always-on attention band
        self.assertContains(response, "data-mc-tabs")
        self.assertContains(response, "Connection graph")
        self.assertContains(response, "Needs attention")
        # themed stylesheet + graph engine wired (cache-bust version is not asserted
        # so it can change freely; the JS is loaded as an ES module barrel).
        self.assertContains(response, "admin_control_center.css?v=")
        self.assertContains(response, 'type="module"')
        self.assertContains(response, "admin_control_center.js?v=")

    def test_game_workspace_uses_shared_graph_engine(self):
        client = Client()
        client.force_login(self.user)

        response = client.get(reverse("admin:ai_hub_workspace_game"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "data-mc-graph")
        self.assertContains(response, "GAME decision graph")
        self.assertContains(response, 'id="ai-control-graph-data"')

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

    def test_build_wizard_advanced_game_links_goal_to_session(self):
        """Regression: advanced wizard must set session.goal — not leave it NULL."""
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse("admin:ai_hub_workspace_build"),
            {
                "wizard_kind": "game",
                "game_flavor": "advanced",
                "engine_mode": "reuse",
                "engine_reuse_model_id": self.model.pk,
                "agent_mode": "reuse",
                "agent_reuse_id": self.agent.pk,
                "goal_text": "Regression test: advanced goal",
                "workspace_name": "RegressionWS",
                "max_iterations": "3",
            },
        )

        self.assertIn(response.status_code, (200, 302))
        session = ExecutionSession.objects.filter(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
        ).order_by("-created_at").first()
        self.assertIsNotNone(session, "Wizard should have created a session")
        self.assertIsNotNone(session.goal, "Advanced wizard session must have goal FK set")
        goal = session.goal
        self.assertEqual(goal.title[:50], "Regression test: advanced goal"[:50])
        self.assertEqual(goal.workspace.name, "RegressionWS")
        # Service should have transitioned goal queued → running
        self.assertEqual(goal.status, GameGoal.Status.RUNNING)

    def test_build_wizard_orchestrator_creates_pipeline(self):
        """Regression: the Orchestrator wizard must create a pipeline, not 500.

        It previously passed input_contract/output_contract kwargs to
        PipelineDefinition.objects.create(), but the model fields are
        global_input_contract/global_output_contract — a TypeError on submit.
        """
        from ai_hub.models import PipelineDefinition, PipelineStep
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse("admin:ai_hub_workspace_build") + "?kind=orchestrator",
            {
                "wizard_kind": "orchestrator",
                "engine_mode": "reuse",
                "engine_reuse_model_id": self.model.pk,
                "agent_mode": "reuse",
                "agent_reuse_id": self.agent.pk,
                "pipeline_name": "Wizard pipeline",
                "pipeline_description": "Built by the wizard",
                "pipeline_input_contract": '{"required": ["dream_id"]}',
                "pipeline_output_contract": '{"required": ["result"]}',
                "step_agent_id": [str(self.agent.pk)],
                "step_on_error": ["stop"],
            },
        )

        self.assertEqual(response.status_code, 302)  # redirect = created successfully
        pipeline = PipelineDefinition.objects.filter(name="Wizard pipeline").first()
        self.assertIsNotNone(pipeline, "Orchestrator wizard should have created a pipeline")
        self.assertEqual(pipeline.global_input_contract, {"required": ["dream_id"]})
        self.assertEqual(pipeline.global_output_contract, {"required": ["result"]})
        # Steps must be created with a VALID on_error choice (regression: the wizard
        # select previously offered fail/skip/retry, none of which are valid).
        step = pipeline.steps.first()
        self.assertIsNotNone(step, "Wizard should have created the step row")
        valid_on_error = {c[0] for c in PipelineStep.OnError.choices}
        self.assertIn(step.on_error, valid_on_error)
        step.full_clean()  # would raise if on_error were an invalid choice

    def test_build_wizard_invalid_toolbox_surfaces_error_and_rolls_back(self):
        """P0: a bad toolbox id must become a field-level error and roll back — not pass silently."""
        from ai_hub.models import PipelineDefinition
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse("admin:ai_hub_workspace_build") + "?kind=orchestrator",
            {
                "wizard_kind": "orchestrator",
                "engine_mode": "reuse",
                "engine_reuse_model_id": self.model.pk,
                "agent_mode": "reuse",
                "agent_reuse_id": self.agent.pk,
                "agent_toolbox_ids": ["999999"],  # nonexistent toolbox
                "pipeline_name": "Should not exist",
            },
        )

        self.assertEqual(response.status_code, 200)  # re-render, not a redirect
        self.assertContains(response, "could not be found")
        # Whole transaction rolled back: no pipeline created.
        self.assertFalse(PipelineDefinition.objects.filter(name="Should not exist").exists())

    def test_build_wizard_rejects_invalid_contract_json(self):
        from ai_hub.models import PipelineDefinition

        client = Client()
        client.force_login(self.user)
        response = client.post(
            reverse("admin:ai_hub_workspace_build") + "?kind=orchestrator",
            {
                "wizard_kind": "orchestrator",
                "engine_mode": "reuse",
                "engine_reuse_model_id": self.model.pk,
                "agent_mode": "reuse",
                "agent_reuse_id": self.agent.pk,
                "pipeline_name": "Invalid contract pipeline",
                "pipeline_input_contract": '{"required": ',
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Pipeline input contract must be valid JSON.")
        self.assertFalse(
            PipelineDefinition.objects.filter(
                name="Invalid contract pipeline"
            ).exists()
        )

    def test_build_wizard_rejects_tampered_provider_type(self):
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse("admin:ai_hub_workspace_build"),
            {
                "wizard_kind": "game",
                "engine_mode": "create",
                "engine_provider_name": "Tampered provider",
                "engine_provider_type": "not-a-provider",
                "engine_model_name": "anything",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Select a valid provider type.")
        self.assertFalse(
            ProviderConfig.objects.filter(name="Tampered provider").exists()
        )

    def test_build_wizard_validates_new_model_configuration(self):
        client = Client()
        client.force_login(self.user)

        invalid_name_response = client.post(
            reverse("admin:ai_hub_workspace_build"),
            {
                "wizard_kind": "game",
                "engine_mode": "create",
                "engine_provider_name": "Training validation",
                "engine_provider_type": ProviderConfig.ProviderType.TRAINING,
                "engine_model_name": "real-model-name",
                "engine_temperature": "0.3",
            },
        )
        invalid_temperature_response = client.post(
            reverse("admin:ai_hub_workspace_build"),
            {
                "wizard_kind": "game",
                "engine_mode": "create",
                "engine_provider_name": "Temperature validation",
                "engine_provider_type": ProviderConfig.ProviderType.OTHER,
                "engine_model_name": "custom-model",
                "engine_temperature": "3",
            },
        )

        self.assertContains(
            invalid_name_response,
            "Training-provider models must be named",
        )
        self.assertContains(
            invalid_temperature_response,
            "Temperature must be between 0 and 2.",
        )
        self.assertFalse(
            ProviderConfig.objects.filter(
                name__in=["Training validation", "Temperature validation"]
            ).exists()
        )

    def test_build_wizard_rejects_inactive_reused_model(self):
        self.model.is_active = False
        self.model.save(update_fields=["is_active"])
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse("admin:ai_hub_workspace_build"),
            {
                "wizard_kind": "game",
                "engine_mode": "reuse",
                "engine_reuse_model_id": self.model.pk,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Select an active model with an active provider.",
        )

    def test_build_wizard_choices_hide_inactive_engine_chains(self):
        inactive_provider = ProviderConfig.objects.create(
            name="Inactive engine provider",
            provider_type=ProviderConfig.ProviderType.OTHER,
            is_active=False,
        )
        hidden_model = ModelConfig.objects.create(
            provider=inactive_provider,
            model_name="hidden-model",
            is_active=True,
        )
        hidden_agent = AgentProfile.objects.create(
            name="Hidden engine agent",
            role="hidden",
            model_config=hidden_model,
            is_active=True,
        )
        client = Client()
        client.force_login(self.user)

        response = client.get(reverse("admin:ai_hub_workspace_build"))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(hidden_model, list(response.context_data["model_configs"]))
        self.assertNotIn(hidden_agent, list(response.context_data["agents"]))

    def test_build_wizard_rejects_invalid_game_flavor(self):
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse("admin:ai_hub_workspace_build"),
            {
                "wizard_kind": "game",
                "game_flavor": "tampered",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Select a valid GAME session flavor.")

    def test_build_wizard_requires_content_for_new_knowledge(self):
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse("admin:ai_hub_workspace_build"),
            {
                "wizard_kind": "game",
                "engine_mode": "reuse",
                "engine_reuse_model_id": self.model.pk,
                "agent_mode": "reuse",
                "agent_reuse_id": self.agent.pk,
                "knowledge_mode": "create",
                "knowledge_collection_name": "Empty knowledge",
                "knowledge_doc_title": "Empty document",
                "knowledge_doc_content": "",
                "goal_text": "Should not be created",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Document content is required to create a retrievable chunk.",
        )
        self.assertFalse(
            KnowledgeCollection.objects.filter(name="Empty knowledge").exists()
        )

    def test_build_wizard_rejects_inactive_pipeline_step_agent(self):
        inactive_agent = AgentProfile.objects.create(
            name="Inactive step agent",
            role="inactive",
            model_config=self.model,
            is_active=False,
            input_contract={"required": ["input"]},
            output_contract={"required": ["output"]},
        )
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse("admin:ai_hub_workspace_build") + "?kind=orchestrator",
            {
                "wizard_kind": "orchestrator",
                "engine_mode": "reuse",
                "engine_reuse_model_id": self.model.pk,
                "agent_mode": "reuse",
                "agent_reuse_id": self.agent.pk,
                "pipeline_name": "Inactive step pipeline",
                "step_agent_id": [str(inactive_agent.pk)],
                "step_on_error": ["stop"],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "must use an active agent with an active engine",
        )
        self.assertFalse(
            PipelineDefinition.objects.filter(name="Inactive step pipeline").exists()
        )

    def test_build_wizard_rejects_invalid_step_mapping_json(self):
        from ai_hub.models import PipelineDefinition

        client = Client()
        client.force_login(self.user)
        response = client.post(
            reverse("admin:ai_hub_workspace_build") + "?kind=orchestrator",
            {
                "wizard_kind": "orchestrator",
                "engine_mode": "reuse",
                "engine_reuse_model_id": self.model.pk,
                "agent_mode": "reuse",
                "agent_reuse_id": self.agent.pk,
                "pipeline_name": "Invalid mapping pipeline",
                "step_agent_id": [str(self.agent.pk)],
                "step_on_error": ["stop"],
                "step_input_mapping": ["[]"],
                "step_output_mapping": ["{}"],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Step 1 input mapping must be a JSON object.")
        self.assertFalse(
            PipelineDefinition.objects.filter(
                name="Invalid mapping pipeline"
            ).exists()
        )

    def test_build_wizard_rejects_invalid_advanced_budget(self):
        client = Client()
        client.force_login(self.user)
        response = client.post(
            reverse("admin:ai_hub_workspace_build"),
            {
                "wizard_kind": "game",
                "game_flavor": "advanced",
                "engine_mode": "reuse",
                "engine_reuse_model_id": self.model.pk,
                "agent_mode": "reuse",
                "agent_reuse_id": self.agent.pk,
                "goal_text": "Invalid budget",
                "workspace_name": "Invalid budget workspace",
                "max_iterations": "3",
                "budget_max_actions": "many",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Maximum action runs must be a whole number.",
        )
        self.assertFalse(
            GameWorkspace.objects.filter(name="Invalid budget workspace").exists()
        )

    def test_build_wizard_full_chain_rollback_on_late_failure(self):
        """P1.3: a late failure rolls back the ENTIRE chain — newly created provider,
        agent and knowledge collection — not just the failing object."""
        from ai_hub.models import (
            PipelineDefinition, ProviderConfig, AgentProfile, KnowledgeCollection,
        )
        PipelineDefinition.objects.create(name="DupPipeline")  # forces a duplicate-name failure
        client = Client()
        client.force_login(self.user)

        response = client.post(
            reverse("admin:ai_hub_workspace_build") + "?kind=orchestrator",
            {
                "wizard_kind": "orchestrator",
                "engine_mode": "create",
                "engine_provider_name": "RollbackProv",
                "engine_provider_type": ProviderConfig.ProviderType.TRAINING,
                "engine_model_name": "training/rollback",
                "agent_mode": "create",
                "agent_name": "RollbackAgent",
                "agent_prompt": "do x",
                "knowledge_mode": "create",
                "knowledge_collection_name": "RollbackColl",
                "knowledge_doc_title": "RollbackDoc",
                "knowledge_doc_content": "content",
                "pipeline_name": "DupPipeline",  # duplicate → fails AFTER engine/agent/knowledge
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "already exists")
        # Everything created before the failure must be gone.
        self.assertFalse(ProviderConfig.objects.filter(name="RollbackProv").exists())
        self.assertFalse(AgentProfile.objects.filter(name="RollbackAgent").exists())
        self.assertFalse(KnowledgeCollection.objects.filter(name="RollbackColl").exists())

    def test_home_vitals_running_and_waiting_counts_are_separate(self):
        """Regression: vitals must carry separate 'running' and 'waiting' keys, not a hardcoded zero."""
        from ai_hub.services.admin_control_center import build_ai_hub_home_context
        running_session = create_execution_session(
            source_label="Running",
            entry_agent=self.agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Running goal",
        )
        running_session.status = ExecutionSession.Status.RUNNING
        running_session.save(update_fields=["status"])
        waiting_session = create_execution_session(
            source_label="Waiting",
            entry_agent=self.agent,
            triggered_by=self.user,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            goal_text="Waiting goal",
        )
        waiting_session.status = ExecutionSession.Status.WAITING_ASYNC
        waiting_session.save(update_fields=["status"])

        ctx = build_ai_hub_home_context()
        vitals = ctx["ai_hub_home"]["vitals"]

        self.assertIn("running", vitals, "vitals must have a 'running' key")
        self.assertIn("waiting", vitals, "vitals must have a 'waiting' key")
        self.assertEqual(vitals["running"], 1)
        self.assertEqual(vitals["waiting"], 1)
        self.assertEqual(vitals["live"], 2)


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

    def test_staff_without_change_permission_cannot_run_execution_session(self):
        restricted_staff = get_user_model().objects.create_user(
            username="restrictedstaff",
            password="testpass123",
            is_staff=True,
        )
        self.client.force_login(restricted_staff)

        response = self.client.post(reverse("ai_hub:run_execution_session"), {"session_id": self.session.id})

        self.assertEqual(response.status_code, 403)

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

    def test_admin_game_session_create_view_rejects_hybrid_mode(self):
        response = self.client.post(
            reverse("admin:ai_hub_executionsession_game_new"),
            {
                "entry_agent": self.agent.id,
                "goal_text": "Unsupported hybrid GAME.",
                "max_iterations": 2,
                "runtime_mode": ExecutionSession.RuntimeMode.HYBRID,
                "strict_response_contract": "on",
                "source_label": "Hybrid GAME",
                "initial_context": "{}",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Select a valid choice")
        self.assertFalse(ExecutionSession.objects.filter(source_label="Hybrid GAME").exists())

    @override_settings(AI_HUB_GAME_GOALS_ENABLED=False)
    def test_goal_flag_does_not_block_orchestrator_session_admin_add(self):
        response = self.client.get(reverse("admin:ai_hub_executionsession_add"))

        self.assertEqual(response.status_code, 200)

    @override_settings(AI_HUB_GAME_GOALS_ENABLED=False)
    def test_goal_flag_hides_direct_goal_add_and_lifecycle_actions(self):
        add_response = self.client.get(reverse("admin:ai_hub_gamegoal_add"))
        list_response = self.client.get(reverse("admin:ai_hub_gamegoal_changelist"))

        self.assertEqual(add_response.status_code, 403)
        self.assertEqual(list_response.status_code, 200)
        self.assertNotContains(list_response, "Queue selected goals")
        self.assertNotContains(list_response, "Cancel selected goals")

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
        # IA Step 4: composed Session Explorer shell
        self.assertContains(response, "Session Explorer")
        self.assertContains(response, "data-ws-tabs")
        self.assertContains(response, 'data-tab="overview"')
        self.assertContains(response, 'data-tab="timeline"')
        self.assertContains(response, "At a glance")


# ============================================================
# Phase 06 — GAME action dispatcher
# ============================================================

from ai_hub.services.game_action_dispatcher import execute_game_action  # noqa: E402


class GameActionDispatcherTests(TestCase):
    def setUp(self):
        self.workspace = create_workspace(name="Dispatcher workspace")
        self.provider = ProviderConfig.objects.create(name="dispatcher-provider", provider_type="training")
        self.model = ModelConfig.objects.create(provider=self.provider, model_name="dispatcher-model")
        self.agent = AgentProfile.objects.create(
            name="dispatcher-agent",
            role="Dispatcher goal runner",
            model_config=self.model,
            input_contract={"required": ["goal"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )
        self.goal = create_goal(
            workspace=self.workspace,
            title="Dispatcher test goal",
            description="Test dispatcher.",
        )
        # Session linked to goal (goal is QUEUED; unit tests call execute_game_action directly)
        self.session = ExecutionSession.objects.create(
            entry_agent=self.agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            status=ExecutionSession.Status.RUNNING,
            goal=self.goal,
            goal_text="Test dispatcher.",
        )
        # Standard action definitions
        self.finish_def = GameActionDefinition.objects.create(
            name="finish_goal",
            label="Finish goal",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )
        self.search_def = GameActionDefinition.objects.create(
            name="search_knowledge",
            label="Search knowledge",
            action_type=GameActionDefinition.ActionType.CONTEXT_TOOL,
        )
        self.read_def = GameActionDefinition.objects.create(
            name="read_document",
            label="Read document",
            action_type=GameActionDefinition.ActionType.CONTEXT_TOOL,
        )
        # Knowledge for search/read tests
        self.collection = KnowledgeCollection.objects.create(name="Dispatcher collection")
        self.agent.knowledge_collections.add(self.collection)
        self.document = KnowledgeDocument.objects.create(
            collection=self.collection,
            title="Django security guide",
            curated_text="This guide covers CSRF, XSS, and SQL injection.",
            status=KnowledgeDocument.Status.ACTIVE,
        )

    def _dispatch(self, action_name, action_input, **kwargs):
        return execute_game_action(
            session=self.session,
            action_name=action_name,
            action_input=action_input,
            **kwargs,
        )

    # ---- Phase 06 spec test names ------------------------------------------

    def test_action_definition_input_contract_validation(self):
        self.finish_def.input_contract = {"required": ["mandatory_field"]}
        self.finish_def.save(update_fields=["input_contract"])
        with self.assertRaisesMessage(ValidationError, "mandatory_field"):
            self._dispatch("finish_goal", {"final_answer": "done"})

    def test_action_definition_output_contract_validation(self):
        self.finish_def.output_contract = {"required": ["unexpected_key"]}
        self.finish_def.save(update_fields=["output_contract"])
        with self.assertRaisesMessage(ValidationError, "unexpected_key"):
            self._dispatch("finish_goal", {"final_answer": "done"})

    def test_disabled_action_cannot_run(self):
        self.finish_def.is_active = False
        self.finish_def.save(update_fields=["is_active"])
        with self.assertRaisesMessage(ValidationError, "Unknown or inactive GAME action"):
            self._dispatch("finish_goal", {"final_answer": "done"})

    def test_unknown_action_is_rejected(self):
        with self.assertRaisesMessage(ValidationError, "Unknown or inactive GAME action"):
            self._dispatch("nonexistent_action", {})

    def test_action_not_allowed_in_workspace_is_rejected(self):
        self.workspace.default_policy = {"allowed_actions": ["search_knowledge"]}
        self.workspace.save(update_fields=["default_policy"])
        self.goal.workspace.refresh_from_db()
        with self.assertRaisesMessage(ValidationError, "allowed_actions policy"):
            self._dispatch("finish_goal", {"final_answer": "done"})

    def test_action_run_is_created_for_selected_action(self):
        action_run = self._dispatch(
            "finish_goal", {"final_answer": "The answer is 42.", "message": "Done."}
        )
        self.assertEqual(action_run.status, GameActionRun.Status.SUCCESS)
        self.assertEqual(action_run.action_name, "finish_goal")
        self.assertEqual(action_run.action, self.finish_def)
        self.assertEqual(action_run.output_payload["final_answer"], "The answer is 42.")
        self.assertTrue(action_run.output_payload["complete"])
        self.assertIsNotNone(action_run.finished_at)
        self.assertIsNotNone(action_run.latency_ms)

    def test_record_memory_dispatcher_maps_each_scope_to_valid_links(self):
        GameActionDefinition.objects.create(
            name="record_memory",
            label="Record memory",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )

        expected_links = {
            GameMemoryEntry.ScopeType.WORKSPACE: (None, None),
            GameMemoryEntry.ScopeType.GOAL: (self.goal.pk, None),
            GameMemoryEntry.ScopeType.SESSION: (None, self.session.pk),
            GameMemoryEntry.ScopeType.ACTION_RESULT: (self.goal.pk, self.session.pk),
        }
        for scope_type, expected in expected_links.items():
            with self.subTest(scope_type=scope_type):
                action_run = self._dispatch(
                    "record_memory",
                    {
                        "scope_type": scope_type,
                        "content": f"{scope_type} dispatcher fact",
                    },
                )
                self.assertEqual(action_run.status, GameActionRun.Status.SUCCESS)
                entry = GameMemoryEntry.objects.get(
                    pk=action_run.output_payload["memory_entry_id"]
                )
                self.assertEqual((entry.goal_id, entry.session_id), expected)

    def test_action_output_becomes_next_iteration_observation(self):
        """search_knowledge output is stored in action_run.observation_payload."""
        action_run = self._dispatch("search_knowledge", {"query": "CSRF"})
        self.assertEqual(action_run.status, GameActionRun.Status.SUCCESS)
        self.assertIn("action_name", action_run.observation_payload)
        self.assertEqual(action_run.observation_payload["action_name"], "search_knowledge")
        self.assertEqual(action_run.output_payload["matched_documents"], 1)

    def test_action_failure_is_recorded(self):
        with self.assertRaisesMessage(ValidationError, "requires a non-empty 'query'"):
            self._dispatch("search_knowledge", {"query": ""})
        failed_run = GameActionRun.objects.filter(
            session=self.session, action_name="search_knowledge", status=GameActionRun.Status.FAILED
        ).first()
        self.assertIsNotNone(failed_run)
        self.assertIn("requires a non-empty", failed_run.error_detail)
        self.assertIsNotNone(failed_run.finished_at)

    def test_completed_equivalent_action_uses_idempotent_result(self):
        run1 = self._dispatch("finish_goal", {"final_answer": "first"})
        run2 = self._dispatch("finish_goal", {"final_answer": "first"})
        self.assertEqual(run1.pk, run2.pk)
        self.assertEqual(GameActionRun.objects.filter(session=self.session).count(), 1)

    def test_internal_finish_goal_action_completes_goal(self):
        action_run = self._dispatch(
            "finish_goal", {"final_answer": "Goal complete.", "message": "All done."}
        )
        self.assertEqual(action_run.status, GameActionRun.Status.SUCCESS)
        self.assertTrue(action_run.output_payload["complete"])
        self.assertEqual(action_run.output_payload["final_answer"], "Goal complete.")

    def test_context_tool_is_safe_and_read_only(self):
        """read_document returns content without modifying any records."""
        action_run = self._dispatch("read_document", {"document_id": self.document.pk})
        self.assertEqual(action_run.status, GameActionRun.Status.SUCCESS)
        self.assertEqual(action_run.output_payload["document_id"], self.document.pk)
        self.assertIn("CSRF", action_run.output_payload["content"])
        # The document itself is unchanged
        self.document.refresh_from_db()
        self.assertEqual(self.document.status, KnowledgeDocument.Status.ACTIVE)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_legacy_game_session_runs_without_dispatcher_flag(self, mocked_call):
        """Sessions without game_action_dispatch_enabled still complete normally."""
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"action": "finish", "complete": true, "final_answer": "Legacy done.", "message": "ok"}',
        }
        goal = create_goal(
            workspace=self.workspace,
            title="Legacy dispatcher goal",
            description="No dispatcher flag.",
        )
        session = create_goal_execution_session(
            goal=goal,
            entry_agent=self.agent,
            runtime_config={"max_iterations": 2},
        )

        run_execution_session(session.id)  # no use_action_dispatcher flag
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(GameActionRun.objects.filter(session=session).count(), 0)

    # ---- extra coverage (not in spec test list but valuable) ---------------

    def test_dispatch_action_input_must_be_dict(self):
        with self.assertRaisesMessage(ValidationError, "must be a JSON object"):
            self._dispatch("finish_goal", "not a dict")

    def test_dispatch_search_knowledge_returns_matching_documents(self):
        action_run = self._dispatch("search_knowledge", {"query": "CSRF"})
        self.assertEqual(action_run.output_payload["query"], "CSRF")
        self.assertEqual(action_run.output_payload["matched_documents"], 1)
        self.assertEqual(
            action_run.output_payload["knowledge_context"][0]["title"],
            "Django security guide",
        )

    def test_dispatch_read_document_blocked_outside_agent_collections(self):
        other_collection = KnowledgeCollection.objects.create(name="Dispatcher other collection")
        other_doc = KnowledgeDocument.objects.create(
            collection=other_collection,
            title="Hidden doc",
            curated_text="Secret content.",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        with self.assertRaisesMessage(ValidationError, "not found or not accessible"):
            self._dispatch("read_document", {"document_id": other_doc.pk})

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=False)
    def test_unified_game_tool_action_is_blocked_when_flag_disabled(self):
        tool = ToolDefinition.objects.create(
            name="game_common_context",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            input_schema={"required": ["query"]},
            output_schema={"required": ["macro"]},
            config={"template": "common context"},
        )
        self.agent.tools.add(tool)
        GameActionDefinition.objects.create(
            name="common_context",
            label="Common context",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
            input_contract={"required": ["query"]},
        )

        with self.assertRaisesMessage(ValidationError, "Unified tool runtime is disabled"):
            self._dispatch("common_context", {"query": "policy"})

        failed_run = GameActionRun.objects.get(action_name="common_context")
        self.assertEqual(failed_run.status, GameActionRun.Status.FAILED)
        self.assertEqual(ToolExecutionRun.objects.count(), 0)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_unified_game_tool_action_executes_common_tool_and_audits_both_layers(self):
        tool = ToolDefinition.objects.create(
            name="game_common_context_enabled",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            input_schema={"required": ["query"]},
            output_schema={"required": ["macro"]},
            config={"template": "common context enabled"},
            operation_mode=ToolDefinition.OperationMode.READ,
            risk_level=ToolDefinition.RiskLevel.LOW,
        )
        self.agent.tools.add(tool)
        GameActionDefinition.objects.create(
            name="common_context_enabled",
            label="Common context enabled",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
            input_contract={"required": ["query"]},
        )

        action_run = self._dispatch("common_context_enabled", {"query": "policy"})

        self.assertEqual(action_run.status, GameActionRun.Status.SUCCESS)
        self.assertEqual(action_run.output_payload["tool_name"], "game_common_context_enabled")
        self.assertEqual(action_run.output_payload["tool_result"], {"macro": "common context enabled"})
        tool_run = ToolExecutionRun.objects.get()
        self.assertEqual(tool_run.status, ToolExecutionRun.Status.SUCCESS)
        self.assertEqual(tool_run.session, self.session)
        self.assertEqual(tool_run.agent, self.agent)
        self.assertEqual(tool_run.tool, tool)
        self.assertEqual(tool_run.input_payload, {"query": "policy"})
        self.assertEqual(tool_run.output_payload, {"macro": "common context enabled"})

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_unified_game_tool_action_requires_agent_tool_permission(self):
        tool = ToolDefinition.objects.create(
            name="game_common_context_unassigned",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            input_schema={"required": ["query"]},
            output_schema={"required": ["macro"]},
            config={"template": "not assigned"},
        )
        GameActionDefinition.objects.create(
            name="common_context_unassigned",
            label="Common context unassigned",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
            input_contract={"required": ["query"]},
        )

        with self.assertRaisesMessage(ValidationError, "not available to agent"):
            self._dispatch("common_context_unassigned", {"query": "policy"})

        failed_run = GameActionRun.objects.get(action_name="common_context_unassigned")
        self.assertEqual(failed_run.status, GameActionRun.Status.FAILED)
        self.assertEqual(ToolExecutionRun.objects.count(), 0)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_approval_required_unified_tool_action_pauses_before_tool_execution(self):
        transition_goal_status(self.goal, GameGoal.Status.RUNNING, reason="start approval test")
        tool = ToolDefinition.objects.create(
            name="game_common_approval_tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            input_schema={"required": ["query"]},
            output_schema={"required": ["macro"]},
            config={"template": "approval gated"},
        )
        self.agent.tools.add(tool)
        GameActionDefinition.objects.create(
            name="common_context_approval",
            label="Common context approval",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
            input_contract={"required": ["query"]},
            requires_approval=True,
        )

        action_run = self._dispatch("common_context_approval", {"query": "policy"})

        self.assertEqual(action_run.status, GameActionRun.Status.WAITING_APPROVAL)
        self.assertTrue(GameActionApprovalRequest.objects.filter(action_run=action_run).exists())
        self.assertEqual(ToolExecutionRun.objects.count(), 0)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_unified_tool_requirement_cannot_be_disabled_by_action(self):
        transition_goal_status(self.goal, GameGoal.Status.RUNNING, reason="tool approval")
        tool = ToolDefinition.objects.create(
            name="tool_requires_approval",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "must wait"},
            requires_approval=True,
        )
        self.agent.tools.add(tool)
        GameActionDefinition.objects.create(
            name="action_without_approval",
            label="Action without approval",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
            requires_approval=False,
        )

        action_run = self._dispatch("action_without_approval", {})

        self.assertEqual(action_run.status, GameActionRun.Status.WAITING_APPROVAL)
        self.assertTrue(GameActionApprovalRequest.objects.filter(action_run=action_run).exists())
        self.assertEqual(ToolExecutionRun.objects.count(), 0)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_unified_action_requirement_still_gates_tool_without_requirement(self):
        transition_goal_status(self.goal, GameGoal.Status.RUNNING, reason="action approval")
        tool = ToolDefinition.objects.create(
            name="action_requires_approval_tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "must wait"},
            requires_approval=False,
        )
        self.agent.tools.add(tool)
        GameActionDefinition.objects.create(
            name="action_requires_approval",
            label="Action requires approval",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
            requires_approval=True,
        )

        action_run = self._dispatch("action_requires_approval", {})

        self.assertEqual(action_run.status, GameActionRun.Status.WAITING_APPROVAL)
        self.assertEqual(ToolExecutionRun.objects.count(), 0)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_unified_tool_workspace_risk_policy_requires_approval(self):
        transition_goal_status(self.goal, GameGoal.Status.RUNNING, reason="workspace approval")
        self.workspace.default_policy = {
            "safety": {"require_approval_for_medium_risk": True},
        }
        self.workspace.save(update_fields=["default_policy"])
        tool = ToolDefinition.objects.create(
            name="workspace_requires_approval_tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "must wait"},
            risk_level=ToolDefinition.RiskLevel.MEDIUM,
            requires_approval=False,
        )
        self.agent.tools.add(tool)
        GameActionDefinition.objects.create(
            name="low_risk_action_wrapper",
            label="Low-risk action wrapper",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
            risk_level=ToolDefinition.RiskLevel.LOW,
            requires_approval=False,
        )

        action_run = self._dispatch("low_risk_action_wrapper", {})

        self.assertEqual(action_run.status, GameActionRun.Status.WAITING_APPROVAL)
        self.assertEqual(ToolExecutionRun.objects.count(), 0)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_workspace_approval_cannot_be_disabled_by_agent_grant_override(self):
        transition_goal_status(self.goal, GameGoal.Status.RUNNING, reason="workspace approval")
        self.workspace.default_policy = {
            "safety": {"require_approval_for_medium_risk": True},
        }
        self.workspace.save(update_fields=["default_policy"])
        tool = ToolDefinition.objects.create(
            name="grant_cannot_lower_workspace_approval",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "must wait"},
            risk_level=ToolDefinition.RiskLevel.MEDIUM,
            requires_approval=True,
        )
        AgentToolGrant.objects.create(
            agent=self.agent,
            tool=tool,
            requires_approval_override=False,
        )
        GameActionDefinition.objects.create(
            name="grant_override_action",
            label="Grant override action",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
            requires_approval=False,
        )

        action_run = self._dispatch("grant_override_action", {})

        self.assertEqual(action_run.status, GameActionRun.Status.WAITING_APPROVAL)
        self.assertEqual(ToolExecutionRun.objects.count(), 0)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_low_level_dispatch_cannot_bypass_effective_tool_approval(self):
        from ai_hub.services.game_action_dispatcher import dispatch_game_action

        tool = ToolDefinition.objects.create(
            name="low_level_approval_tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "must not run"},
            requires_approval=True,
        )
        self.agent.tools.add(tool)
        action = GameActionDefinition.objects.create(
            name="low_level_approval_action",
            label="Low-level approval action",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
            requires_approval=False,
        )
        action_run = GameActionRun.objects.create(
            session=self.session,
            action=action,
            idempotency_key="low-level-approval-bypass",
            action_name=action.name,
            iteration=1,
            status=GameActionRun.Status.RUNNING,
        )

        with self.assertRaisesMessage(ValidationError, "requires approval"):
            dispatch_game_action(
                action_run=action_run,
                workspace=self.workspace,
                goal=self.goal,
                payload={},
            )

        self.assertEqual(ToolExecutionRun.objects.count(), 0)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_pipeline_game_uses_first_step_agent_for_tool_resolution_and_audit(self):
        pipeline_agent = AgentProfile.objects.create(
            name="pipeline-dispatcher-agent",
            role="Pipeline dispatcher",
            model_config=self.model,
        )
        pipeline = PipelineDefinition.objects.create(name="GAME effective agent pipeline")
        PipelineStep.objects.create(pipeline=pipeline, agent=pipeline_agent, order=1)
        self.session.entry_agent = None
        self.session.pipeline = pipeline
        self.session.save(update_fields=["entry_agent", "pipeline"])
        tool = ToolDefinition.objects.create(
            name="pipeline_agent_tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "pipeline agent only"},
        )
        pipeline_agent.tools.add(tool)
        GameActionDefinition.objects.create(
            name="pipeline_agent_action",
            label="Pipeline agent action",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
        )

        action_run = self._dispatch("pipeline_agent_action", {})

        self.assertEqual(action_run.status, GameActionRun.Status.SUCCESS)
        self.assertEqual(ToolExecutionRun.objects.get().agent, pipeline_agent)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_pipeline_game_rejects_tool_not_granted_to_effective_agent(self):
        pipeline_agent = AgentProfile.objects.create(
            name="pipeline-restricted-agent",
            role="Restricted pipeline dispatcher",
            model_config=self.model,
        )
        pipeline = PipelineDefinition.objects.create(name="GAME restricted agent pipeline")
        PipelineStep.objects.create(pipeline=pipeline, agent=pipeline_agent, order=1)
        self.session.entry_agent = None
        self.session.pipeline = pipeline
        self.session.save(update_fields=["entry_agent", "pipeline"])
        tool = ToolDefinition.objects.create(
            name="wrong_agent_tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "wrong agent"},
        )
        self.agent.tools.add(tool)
        GameActionDefinition.objects.create(
            name="wrong_agent_action",
            label="Wrong agent action",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
        )

        with self.assertRaisesMessage(ValidationError, "not available to agent"):
            self._dispatch("wrong_agent_action", {})

        self.assertEqual(ToolExecutionRun.objects.count(), 0)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_unified_tool_rejects_inactive_effective_agent(self):
        self.agent.is_active = False
        self.agent.save(update_fields=["is_active"])
        tool = ToolDefinition.objects.create(
            name="inactive_agent_tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "must not run"},
        )
        self.agent.tools.add(tool)
        GameActionDefinition.objects.create(
            name="inactive_agent_action",
            label="Inactive agent action",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
        )

        with self.assertRaisesMessage(ValidationError, "must be active"):
            self._dispatch("inactive_agent_action", {})

        self.assertEqual(ToolExecutionRun.objects.count(), 0)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_pipeline_game_bound_tool_ignores_model_supplied_agent_identity(self):
        seed_starter_toolboxes()
        pipeline_agent = AgentProfile.objects.create(
            name="pipeline-bound-agent",
            role="Bound pipeline dispatcher",
            model_config=self.model,
        )
        pipeline_collection = KnowledgeCollection.objects.create(name="Pipeline agent library")
        pipeline_agent.knowledge_collections.add(pipeline_collection)
        pipeline = PipelineDefinition.objects.create(name="GAME bound agent pipeline")
        PipelineStep.objects.create(pipeline=pipeline, agent=pipeline_agent, order=1)
        self.session.entry_agent = None
        self.session.pipeline = pipeline
        self.session.save(update_fields=["entry_agent", "pipeline"])
        tool = ToolDefinition.objects.get(name="list_knowledge_libraries")
        GameActionDefinition.objects.create(
            name="pipeline_bound_knowledge_action",
            label="Pipeline bound knowledge action",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
        )

        action_run = self._dispatch(
            "pipeline_bound_knowledge_action",
            {"agent_id": self.agent.pk, "agent_name": self.agent.name},
        )

        tool_run = ToolExecutionRun.objects.get()
        self.assertEqual(action_run.status, GameActionRun.Status.SUCCESS)
        self.assertEqual(tool_run.agent, pipeline_agent)
        self.assertEqual(tool_run.input_payload["agent_id"], pipeline_agent.pk)
        self.assertNotIn("agent_name", tool_run.input_payload)
        self.assertEqual(
            action_run.output_payload["tool_result"]["libraries"][0]["name"],
            pipeline_collection.name,
        )

    def test_pipeline_game_context_actions_use_effective_agent_knowledge(self):
        pipeline_agent = AgentProfile.objects.create(
            name="pipeline-knowledge-agent",
            role="Pipeline knowledge reader",
            model_config=self.model,
        )
        pipeline_collection = KnowledgeCollection.objects.create(name="Pipeline private library")
        pipeline_agent.knowledge_collections.add(pipeline_collection)
        pipeline_document = KnowledgeDocument.objects.create(
            collection=pipeline_collection,
            title="Pipeline-only runbook",
            curated_text="The pipeline-only marker is delta-echo.",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        pipeline = PipelineDefinition.objects.create(name="GAME knowledge pipeline")
        PipelineStep.objects.create(pipeline=pipeline, agent=pipeline_agent, order=1)
        self.session.entry_agent = None
        self.session.pipeline = pipeline
        self.session.save(update_fields=["entry_agent", "pipeline"])

        search_run = self._dispatch("search_knowledge", {"query": "delta-echo"})
        read_run = self._dispatch(
            "read_document",
            {"document_id": pipeline_document.pk},
        )

        self.assertEqual(search_run.output_payload["matched_documents"], 1)
        self.assertEqual(
            search_run.output_payload["knowledge_context"][0]["title"],
            pipeline_document.title,
        )
        self.assertIn("delta-echo", read_run.output_payload["content"])

    def test_idempotency_key_unique_db_constraint(self):
        import hashlib, json
        payload = {"session_id": self.session.pk, "step_run_id": None, "action_id": self.finish_def.pk, "input": {}}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        GameActionRun.objects.create(
            session=self.session,
            action=self.finish_def,
            idempotency_key=key,
            action_name="finish_goal",
            iteration=99,
        )
        with self.assertRaises(IntegrityError):
            GameActionRun.objects.create(
                session=self.session,
                action=self.finish_def,
                idempotency_key=key,
                action_name="finish_goal",
                iteration=99,
            )

    def test_action_run_str(self):
        run = GameActionRun.objects.create(
            session=self.session,
            action=self.finish_def,
            idempotency_key="test-str-key-unique",
            action_name="finish_goal",
            iteration=1,
        )
        self.assertIn("finish_goal", str(run))
        self.assertIn(str(self.session.pk), str(run))

    # ---- integration (with mocked LLM) -------------------------------------

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_action_output_becomes_next_iteration_observation_integration(self, mocked_call):
        mocked_call.side_effect = [
            {
                "status": "ok",
                "content": (
                    '{"action": "search_knowledge", "message": "Searching.", '
                    '"complete": false, "final_answer": "", '
                    '"action_input": {"query": "CSRF"}}'
                ),
            },
            {
                "status": "ok",
                "content": (
                    '{"action": "finish", "message": "Done.", '
                    '"complete": true, "final_answer": "Security summary."}'
                ),
            },
        ]
        goal = create_goal(
            workspace=self.workspace,
            title="Dispatcher search integration",
            description="Search for CSRF info.",
        )
        session = create_goal_execution_session(
            goal=goal,
            entry_agent=self.agent,
            runtime_config={"max_iterations": 3},
        )

        run_execution_session(session.id, use_action_dispatcher=True)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        action_runs = GameActionRun.objects.filter(session=session)
        self.assertEqual(action_runs.count(), 1)
        run = action_runs.get()
        self.assertEqual(run.action_name, "search_knowledge")
        self.assertEqual(run.status, GameActionRun.Status.SUCCESS)
        first_step = session.step_runs.get(order=1)
        self.assertIn("action_run_id", first_step.observation_payload)
        self.assertIn("action_output", first_step.observation_payload)
        memory = session.final_context["memory"]
        self.assertTrue(any("action_output_summary" in m for m in memory))

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_recorded_memory_is_refreshed_before_next_iteration(self, mocked_call):
        GameActionDefinition.objects.create(
            name="record_memory",
            label="Record memory",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )
        mocked_call.side_effect = [
            {
                "status": "ok",
                "content": (
                    '{"action":"record_memory","action_input":{"scope_type":"goal",'
                    '"content":"New fact from this run."},"message":"remember",'
                    '"complete":false,"final_answer":""}'
                ),
            },
            {
                "status": "ok",
                "content": (
                    '{"action":"finish","message":"used memory",'
                    '"complete":true,"final_answer":"done"}'
                ),
            },
        ]
        goal = create_goal(
            workspace=self.workspace,
            title="Refresh scoped memory",
            description="Use memory recorded during the same run.",
        )
        session = create_goal_execution_session(
            goal=goal,
            entry_agent=self.agent,
            runtime_config={"max_iterations": 2},
        )

        run_execution_session(session.pk, use_action_dispatcher=True)

        second_request = session.step_runs.get(order=2).request_payload
        self.assertIn(
            "New fact from this run.",
            [
                entry["content"]
                for entry in second_request["scoped_memory"]["entries"]
            ],
        )

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_game_rolling_context_is_bounded_without_truncating_audit(self, mocked_call):
        large_message = "x" * 2000
        mocked_call.return_value = {
            "status": "ok",
            "content": json.dumps(
                {
                    "action": "think",
                    "message": large_message,
                    "complete": False,
                    "final_answer": "",
                }
            ),
        }
        goal = create_goal(
            workspace=self.workspace,
            title="Bound rolling context",
            description="Keep prompt context bounded.",
        )
        session = create_goal_execution_session(
            goal=goal,
            entry_agent=self.agent,
            runtime_config={
                "max_iterations": 4,
                "game_memory_max_entries": 2,
                "game_observations_max_entries": 2,
                "game_observation_max_chars": 300,
                "game_previous_response_max_chars": 300,
                "game_memory_entry_max_chars": 80,
            },
        )

        run_execution_session(session.pk)
        session.refresh_from_db()

        self.assertEqual(len(session.final_context["memory"]), 2)
        self.assertEqual(len(session.final_context["observations"]), 2)
        self.assertLessEqual(
            len(json.dumps(session.final_context["observations"][0])),
            300,
        )
        self.assertLessEqual(
            len(session.final_context["memory"][0]["summary"]),
            80,
        )
        second_request = session.step_runs.get(order=2).request_payload
        self.assertTrue(second_request["previous_response"]["truncated"])
        self.assertLessEqual(
            len(json.dumps(second_request["previous_response"])),
            300,
        )
        # Raw audit remains complete even when the prompt receives a preview.
        self.assertEqual(
            session.step_runs.get(order=1).observation_payload["decision"]["message"],
            large_message,
        )

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_internal_finish_goal_action_completes_goal_integration(self, mocked_call):
        mocked_call.return_value = {
            "status": "ok",
            "content": (
                '{"action": "finish_goal", "message": "All done.", '
                '"complete": false, "final_answer": "", '
                '"action_input": {"final_answer": "Dispatcher answer.", "message": "All done."}}'
            ),
        }
        goal = create_goal(
            workspace=self.workspace,
            title="Dispatcher finish integration",
            description="Complete via dispatcher.",
        )
        session = create_goal_execution_session(
            goal=goal,
            entry_agent=self.agent,
            runtime_config={"max_iterations": 5},
        )

        run_execution_session(session.id, use_action_dispatcher=True)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(session.step_runs.count(), 1)
        self.assertEqual(session.final_context["finish_reason"], "agent_finished")
        self.assertEqual(session.final_context["final_answer"], "Dispatcher answer.")
        runs = GameActionRun.objects.filter(session=session, action_name="finish_goal")
        self.assertEqual(runs.count(), 1)
        self.assertEqual(runs.get().status, GameActionRun.Status.SUCCESS)


# ============================================================
# Phase 07 — GAME scoped memory
# ============================================================

from ai_hub.services.game_memory import build_goal_memory_context, record_memory  # noqa: E402
from ai_hub.services.game_memory_compaction import compact_goal_memory  # noqa: E402


class GameMemoryTests(TestCase):
    def setUp(self):
        self.workspace = create_workspace(name="Memory workspace")
        self.other_workspace = create_workspace(name="Other memory workspace")
        self.provider = ProviderConfig.objects.create(name="memory-provider", provider_type="training")
        self.model = ModelConfig.objects.create(provider=self.provider, model_name="memory-model")
        self.agent = AgentProfile.objects.create(
            name="memory-agent",
            role="Memory goal runner",
            model_config=self.model,
        )
        self.goal = create_goal(
            workspace=self.workspace,
            title="Memory test goal",
            description="Test scoped memory.",
        )
        self.other_goal = create_goal(
            workspace=self.workspace,
            title="Other memory goal",
            description="Should not see first goal memory.",
        )
        self.session = ExecutionSession.objects.create(
            entry_agent=self.agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            status=ExecutionSession.Status.RUNNING,
            goal=self.goal,
            goal_text="Test memory.",
        )

    def _record(self, content, scope_type=GameMemoryEntry.ScopeType.GOAL, **kwargs):
        return record_memory(
            scope_type=scope_type,
            workspace=self.workspace,
            content=content,
            goal=kwargs.pop("goal", self.goal),
            **kwargs,
        )

    def test_record_memory_coerces_float_importance_score(self):
        from decimal import Decimal

        entry = self._record("Float importance is coerced", importance_score=0.9)
        entry.refresh_from_db()
        self.assertEqual(entry.importance_score, Decimal("0.90"))

    # ---- Phase 07 spec test names ------------------------------------------

    def test_workspace_memory_visible_to_its_goals(self):
        record_memory(
            scope_type=GameMemoryEntry.ScopeType.WORKSPACE,
            workspace=self.workspace,
            content="Shared workspace fact.",
        )
        ctx = build_goal_memory_context(
            workspace=self.workspace, goal=self.goal, session=self.session, max_chars=4000
        )
        contents = [e["content"] for e in ctx["entries"]]
        self.assertIn("Shared workspace fact.", contents)

    def test_goal_memory_not_visible_to_other_goal_by_default(self):
        self._record("Private goal fact.")
        ctx = build_goal_memory_context(
            workspace=self.workspace, goal=self.other_goal, session=None, max_chars=4000
        )
        contents = [e["content"] for e in ctx["entries"]]
        self.assertNotIn("Private goal fact.", contents)

    def test_session_memory_not_visible_to_other_session(self):
        other_session = ExecutionSession.objects.create(
            entry_agent=self.agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            status=ExecutionSession.Status.RUNNING,
            goal=self.other_goal,
            goal_text="Other session.",
        )
        record_memory(
            scope_type=GameMemoryEntry.ScopeType.SESSION,
            workspace=self.workspace,
            content="Session-only fact.",
            session=self.session,
        )
        ctx = build_goal_memory_context(
            workspace=self.workspace, goal=self.other_goal, session=other_session, max_chars=4000
        )
        contents = [e["content"] for e in ctx["entries"]]
        self.assertNotIn("Session-only fact.", contents)

    def test_memory_scope_integrity_is_validated(self):
        # Workspace-scoped entry must have no goal or session
        with self.assertRaises(ValidationError):
            record_memory(
                scope_type=GameMemoryEntry.ScopeType.WORKSPACE,
                workspace=self.workspace,
                content="Bad workspace entry.",
                goal=self.goal,  # must be null for workspace scope
            )
        # Goal-scoped entry must belong to the same workspace
        with self.assertRaises(ValidationError):
            record_memory(
                scope_type=GameMemoryEntry.ScopeType.GOAL,
                workspace=self.other_workspace,
                content="Cross-workspace entry.",
                goal=self.goal,  # goal belongs to self.workspace, not other_workspace
            )

    def test_expired_memory_is_excluded(self):
        past = timezone.now() - timedelta(hours=1)
        record_memory(
            scope_type=GameMemoryEntry.ScopeType.GOAL,
            workspace=self.workspace,
            content="Expired fact.",
            goal=self.goal,
            expires_at=past,
        )
        ctx = build_goal_memory_context(
            workspace=self.workspace, goal=self.goal, session=self.session, max_chars=4000
        )
        contents = [e["content"] for e in ctx["entries"]]
        self.assertNotIn("Expired fact.", contents)

    def test_memory_context_respects_max_char_budget(self):
        for i in range(5):
            self._record(f"{'x' * 300} entry {i}")
        ctx = build_goal_memory_context(
            workspace=self.workspace, goal=self.goal, session=self.session, max_chars=500
        )
        self.assertLessEqual(ctx["chars_used"], 500)

    def test_memory_context_reports_truncation(self):
        for i in range(5):
            self._record(f"{'x' * 300} entry {i}")
        ctx = build_goal_memory_context(
            workspace=self.workspace, goal=self.goal, session=self.session, max_chars=400
        )
        self.assertTrue(ctx["truncated"])
        self.assertGreater(ctx["truncated_count"], 0)

    def test_high_importance_memory_is_selected_first(self):
        self._record("Low importance fact.", importance_score=Decimal("0.20"))
        self._record("High importance fact.", importance_score=Decimal("0.95"))
        ctx = build_goal_memory_context(
            workspace=self.workspace, goal=self.goal, session=self.session, max_chars=4000
        )
        contents = [e["content"] for e in ctx["entries"]]
        high_idx = contents.index("High importance fact.")
        low_idx = contents.index("Low importance fact.")
        self.assertLess(high_idx, low_idx)

    def test_action_result_can_be_recorded_as_goal_memory(self):
        entry = record_memory(
            scope_type=GameMemoryEntry.ScopeType.ACTION_RESULT,
            workspace=self.workspace,
            content="Search result: CSRF protection.",
            goal=self.goal,
            metadata={"source": "action_run", "source_id": 1},
        )
        self.assertEqual(entry.scope_type, GameMemoryEntry.ScopeType.ACTION_RESULT)
        ctx = build_goal_memory_context(
            workspace=self.workspace, goal=self.goal, session=self.session, max_chars=4000
        )
        contents = [e["content"] for e in ctx["entries"]]
        self.assertIn("Search result: CSRF protection.", contents)

    def test_compaction_preserves_reference_to_raw_audit_logs(self):
        """Compaction expires stale entries but never modifies GameActionRun records."""
        finish_def = GameActionDefinition.objects.create(
            name="finish_goal_compact",
            label="Finish goal",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )
        for i in range(30):
            self._record(f"Old fact {i}", importance_score=Decimal("0.30"))
        result = compact_goal_memory(goal=self.goal, workspace=self.workspace, keep_n=10)
        self.assertGreater(result["compacted"], 0)
        self.assertLessEqual(result["retained"], result["total_before"])
        # Audit records untouched
        self.assertEqual(
            GameActionRun.objects.filter(session=self.session).count(), 0
        )

    def test_model_derived_memory_is_marked_in_metadata(self):
        entry = record_memory(
            scope_type=GameMemoryEntry.ScopeType.GOAL,
            workspace=self.workspace,
            content="The model says X is true.",
            goal=self.goal,
            metadata={"source": "model_output", "model_derived": True},
        )
        self.assertTrue(entry.metadata.get("model_derived"))
        self.assertEqual(entry.metadata.get("source"), "model_output")


# ============================================================
# Phase 08 — pause, approval, and resume
# ============================================================

from django.contrib.auth.models import Permission  # noqa: E402
from django.contrib.contenttypes.models import ContentType  # noqa: E402

from ai_hub.models import (  # noqa: E402
    GameActionApprovalRequest,
    GameContinuationRequest,
)
from ai_hub.services.game_resume import (  # noqa: E402
    approve_action_run,
    pause_session,
    reject_action_run,
    resume_goal_execution,
)


class GamePauseApprovalResumeTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.workspace = create_workspace(name="Pause workspace")
        self.provider = ProviderConfig.objects.create(name="pause-provider", provider_type="training")
        self.model = ModelConfig.objects.create(provider=self.provider, model_name="pause-model")
        self.agent = AgentProfile.objects.create(
            name="pause-agent",
            role="Pause goal runner",
            model_config=self.model,
        )
        self.goal = create_goal(
            workspace=self.workspace,
            title="Pause test goal",
            description="Test pause/approval/resume.",
        )
        # Goal must be RUNNING so pause_session can transition it to a waiting state
        transition_goal_status(self.goal, GameGoal.Status.RUNNING, reason="test setup")

        self.session = ExecutionSession.objects.create(
            entry_agent=self.agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            status=ExecutionSession.Status.RUNNING,
            goal=self.goal,
            goal_text="Test pause and resume.",
            started_at=timezone.now(),
        )

        # Action definitions with registered handler names
        self.finish_def = GameActionDefinition.objects.create(
            name="finish_goal",
            label="Finish goal",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            requires_approval=False,
        )
        self.approval_def = GameActionDefinition.objects.create(
            name="finish_goal_gated",
            label="Finish goal (approval required)",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            requires_approval=True,
        )

        # User with approval permission
        ct = ContentType.objects.get_for_model(GameActionApprovalRequest)
        perm = Permission.objects.get(content_type=ct, codename="approve_game_action")
        self.approver = User.objects.create_user(username="pa-approver", password="test")
        self.approver.user_permissions.add(perm)
        # Reload to clear perm cache
        self.approver = User.objects.get(pk=self.approver.pk)

        # Regular user (no approval perm)
        self.regular_user = User.objects.create_user(username="pa-regular", password="test")

    def _make_waiting_approval_run(self, action_name="finish_goal", action_def=None):
        """Helper: create action_run (WAITING_APPROVAL) + approval_req (PENDING)."""
        if action_def is None:
            action_def = self.finish_def
        import hashlib, json
        payload = {
            "session_id": self.session.pk,
            "step_run_id": None,
            "action_id": action_def.pk,
            "input": {"final_answer": "pending test"},
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        action_run = GameActionRun.objects.create(
            session=self.session,
            action=action_def,
            idempotency_key=key,
            action_name=action_name,
            iteration=1,
            status=GameActionRun.Status.WAITING_APPROVAL,
            input_payload={"final_answer": "pending test"},
            started_at=timezone.now(),
        )
        from ai_hub.services.game_action_dispatcher import (
            build_game_action_approval_intent,
        )

        intent_snapshot, intent_fingerprint = build_game_action_approval_intent(
            session=self.session,
            workspace=self.workspace,
            goal=self.goal,
            action_definition=action_def,
            payload=dict(action_run.input_payload),
        )
        approval_req = GameActionApprovalRequest.objects.create(
            action_run=action_run,
            goal=self.goal,
            requested_payload={"final_answer": "pending test"},
            execution_intent_snapshot=intent_snapshot,
            execution_intent_fingerprint=intent_fingerprint,
        )
        return action_run, approval_req

    def _pause_session(self, reason_code="needs_information"):
        """Helper: pause self.session and return the continuation request."""
        self.session.status = ExecutionSession.Status.WAITING_ASYNC
        self.session.save(update_fields=["status", "updated_at"])
        self.goal.refresh_from_db()
        if self.goal.status not in {
            GameGoal.Status.WAITING_INFO,
            GameGoal.Status.WAITING_APPROVAL,
            GameGoal.Status.BLOCKED,
        }:
            transition_goal_status(self.goal, GameGoal.Status.WAITING_INFO, reason="test pause")
        return GameContinuationRequest.objects.create(
            session=self.session,
            goal=self.goal,
            reason_code=reason_code,
        )

    def _request_finish_approval(self, *, final_answer="approved answer"):
        self.finish_def.requires_approval = True
        self.finish_def.save(update_fields=["requires_approval", "updated_at"])
        return execute_game_action(
            session=self.session,
            action_name=self.finish_def.name,
            action_input={"final_answer": final_answer},
        )

    def _request_tool_approval(
        self,
        *,
        name,
        tool_config=None,
        action_config=None,
        action_input=None,
    ):
        tool = ToolDefinition.objects.create(
            name=f"{name}_tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config=tool_config or {"template": "approved result"},
            requires_approval=True,
        )
        self.agent.tools.add(tool)
        action = GameActionDefinition.objects.create(
            name=name,
            label=name,
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
            config=action_config or {},
        )
        action_run = execute_game_action(
            session=self.session,
            action_name=action.name,
            action_input=action_input or {"query": "approved input"},
        )
        return action_run, action, tool

    # ---- Phase 08 spec test names ------------------------------------------

    def test_waiting_information_goal_is_not_selected_by_scheduler(self):
        waiting_goal = create_goal(
            workspace=self.workspace,
            title="Waiting info goal",
            description="This goal is stuck.",
        )
        transition_goal_status(waiting_goal, GameGoal.Status.RUNNING, reason="test")
        transition_goal_status(waiting_goal, GameGoal.Status.WAITING_INFO, reason="test")

        selected = get_next_eligible_goal(self.workspace.pk)
        # scheduler only selects QUEUED goals; waiting_goal and self.goal are not QUEUED
        if selected is not None:
            self.assertNotEqual(selected.pk, waiting_goal.pk)

        # More direct: create a fresh QUEUED goal and verify WAITING_INFO goal is not returned
        queued_goal = create_goal(
            workspace=self.workspace,
            title="Queued goal",
            description="Should be picked up.",
        )
        selected = get_next_eligible_goal(self.workspace.pk)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.pk, queued_goal.pk)

    def test_approval_required_action_creates_approval_request(self):
        # Patch _INTERNAL_HANDLERS to include finish_goal_gated mapped to the finish_goal handler
        from ai_hub.services import game_action_dispatcher as dispatcher
        original = dict(dispatcher._INTERNAL_HANDLERS)
        dispatcher._INTERNAL_HANDLERS["finish_goal_gated"] = dispatcher._handle_finish_goal
        try:
            action_run = execute_game_action(
                session=self.session,
                action_name="finish_goal_gated",
                action_input={"final_answer": "done"},
            )
        finally:
            dispatcher._INTERNAL_HANDLERS.clear()
            dispatcher._INTERNAL_HANDLERS.update(original)

        self.assertEqual(action_run.status, GameActionRun.Status.WAITING_APPROVAL)
        approval_req = GameActionApprovalRequest.objects.filter(action_run=action_run).first()
        self.assertIsNotNone(approval_req)
        self.assertEqual(approval_req.status, GameActionApprovalRequest.Status.PENDING)
        self.assertTrue(approval_req.execution_intent_snapshot)
        self.assertEqual(len(approval_req.execution_intent_fingerprint), 64)

    def test_goal_moves_to_waiting_approval(self):
        from ai_hub.services import game_action_dispatcher as dispatcher
        original = dict(dispatcher._INTERNAL_HANDLERS)
        dispatcher._INTERNAL_HANDLERS["finish_goal_gated"] = dispatcher._handle_finish_goal
        try:
            execute_game_action(
                session=self.session,
                action_name="finish_goal_gated",
                action_input={"final_answer": "done"},
            )
        finally:
            dispatcher._INTERNAL_HANDLERS.clear()
            dispatcher._INTERNAL_HANDLERS.update(original)

        self.goal.refresh_from_db()
        self.assertEqual(self.goal.status, GameGoal.Status.WAITING_APPROVAL)

    def test_rejecting_action_creates_agent_observation(self):
        action_run, _ = self._make_waiting_approval_run()
        self.goal.status = GameGoal.Status.WAITING_APPROVAL
        self.goal.save(update_fields=["status"])

        action_run_back, obs = reject_action_run(
            action_run_id=action_run.pk,
            reviewed_by=self.approver,
            review_note="Too risky.",
        )

        action_run.refresh_from_db()
        self.assertEqual(action_run.status, GameActionRun.Status.REJECTED)
        self.assertIn("rejected", obs["status"])
        self.assertIn("finish_goal", obs["action_name"])
        self.assertIn("rejected", obs["message"].lower())

    def test_approving_action_allows_resume(self):
        action_run, _ = self._make_waiting_approval_run()
        # Pause the session properly
        self.session.status = ExecutionSession.Status.WAITING_ASYNC
        self.session.final_context = {
            "memory": [],
            "observations": [],
            "goal": self.session.goal_text,
        }
        self.session.save(update_fields=["status", "final_context", "updated_at"])
        GameContinuationRequest.objects.create(
            session=self.session,
            goal=self.goal,
            reason_code="needs_approval",
        )
        self.goal.status = GameGoal.Status.WAITING_APPROVAL
        self.goal.save(update_fields=["status"])

        result_run = approve_action_run(
            action_run_id=action_run.pk,
            reviewed_by=self.approver,
        )
        result_run.refresh_from_db()
        self.assertEqual(result_run.status, GameActionRun.Status.SUCCESS)
        approval_req = GameActionApprovalRequest.objects.get(action_run=action_run)
        self.assertEqual(approval_req.status, GameActionApprovalRequest.Status.APPROVED)
        # Continuation request is still PENDING so resume_goal_execution can find it
        cont_req = GameContinuationRequest.objects.get(session=self.session)
        self.assertEqual(cont_req.status, GameContinuationRequest.Status.PENDING)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_approved_unified_tool_executes_with_approved_audit_state(self):
        tool = ToolDefinition.objects.create(
            name="approved_unified_tool",
            tool_kind=ToolDefinition.ToolKind.PROMPT_MACRO,
            config={"template": "approved result"},
            requires_approval=True,
        )
        self.agent.tools.add(tool)
        GameActionDefinition.objects.create(
            name="approved_unified_action",
            label="Approved unified action",
            action_type=GameActionDefinition.ActionType.TOOL,
            tool=tool,
            requires_approval=False,
        )
        action_run = execute_game_action(
            session=self.session,
            action_name="approved_unified_action",
            action_input={},
        )

        self.assertEqual(action_run.status, GameActionRun.Status.WAITING_APPROVAL)
        self.assertEqual(ToolExecutionRun.objects.count(), 0)

        approved_run = approve_action_run(
            action_run_id=action_run.pk,
            reviewed_by=self.approver,
        )

        tool_run = ToolExecutionRun.objects.get()
        self.assertEqual(approved_run.status, GameActionRun.Status.SUCCESS)
        self.assertEqual(tool_run.status, ToolExecutionRun.Status.SUCCESS)
        self.assertEqual(tool_run.approval_state, ToolExecutionRun.ApprovalState.APPROVED)
        self.assertEqual(tool_run.agent, self.agent)

    def test_approval_rejects_action_configuration_drift_before_dispatch(self):
        action_run = self._request_finish_approval()
        self.finish_def.config = {"mode": "changed-after-request"}
        self.finish_def.save(update_fields=["config", "updated_at"])

        with patch(
            "ai_hub.services.game_action_dispatcher.dispatch_game_action",
            return_value={"complete": True},
        ) as mocked_dispatch:
            with self.assertRaises(ValidationError):
                approve_action_run(
                    action_run_id=action_run.pk,
                    reviewed_by=self.approver,
                )

        mocked_dispatch.assert_not_called()
        action_run.refresh_from_db()
        approval = action_run.approval_request
        approval.refresh_from_db()
        self.assertEqual(action_run.status, GameActionRun.Status.REJECTED)
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.REJECTED)

    def test_approval_rejects_action_contract_drift_before_dispatch(self):
        action_run = self._request_finish_approval()
        self.finish_def.output_contract = {"required": ["new_field"]}
        self.finish_def.save(update_fields=["output_contract", "updated_at"])

        with patch(
            "ai_hub.services.game_action_dispatcher.dispatch_game_action",
            return_value={"complete": True},
        ) as mocked_dispatch:
            with self.assertRaises(ValidationError):
                approve_action_run(
                    action_run_id=action_run.pk,
                    reviewed_by=self.approver,
                )

        mocked_dispatch.assert_not_called()
        action_run.refresh_from_db()
        self.assertEqual(action_run.status, GameActionRun.Status.REJECTED)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_approval_rejects_tool_configuration_drift_before_execution(self):
        action_run, _, tool = self._request_tool_approval(
            name="approval_tool_config_drift"
        )
        tool.config = {"template": "changed after request"}
        tool.save(update_fields=["config", "updated_at"])

        with patch(
            "ai_hub.services.game_action_dispatcher.execute_tool",
            return_value={"macro": "executed"},
        ) as mocked_execute:
            with self.assertRaises(ValidationError):
                approve_action_run(
                    action_run_id=action_run.pk,
                    reviewed_by=self.approver,
                )

        mocked_execute.assert_not_called()
        approval = action_run.approval_request
        approval.refresh_from_db()
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.REJECTED)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_approval_rejects_revoked_tool_permission_before_execution(self):
        action_run, _, tool = self._request_tool_approval(
            name="approval_tool_permission_drift"
        )
        AgentToolGrant.objects.create(
            agent=self.agent,
            tool=tool,
            is_enabled=False,
        )

        with patch(
            "ai_hub.services.game_action_dispatcher.execute_tool",
            return_value={"macro": "executed"},
        ) as mocked_execute:
            with self.assertRaises(ValidationError):
                approve_action_run(
                    action_run_id=action_run.pk,
                    reviewed_by=self.approver,
                )

        mocked_execute.assert_not_called()
        approval = action_run.approval_request
        approval.refresh_from_db()
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.REJECTED)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_approval_rejects_inactive_agent_before_tool_execution(self):
        action_run, _, _ = self._request_tool_approval(
            name="approval_inactive_agent"
        )
        self.agent.is_active = False
        self.agent.save(update_fields=["is_active", "updated_at"])

        with patch(
            "ai_hub.services.game_action_dispatcher.execute_tool",
            return_value={"macro": "executed"},
        ) as mocked_execute:
            with self.assertRaises(ValidationError):
                approve_action_run(
                    action_run_id=action_run.pk,
                    reviewed_by=self.approver,
                )

        mocked_execute.assert_not_called()
        approval = action_run.approval_request
        approval.refresh_from_db()
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.REJECTED)

    def test_approval_rejects_stricter_workspace_policy_before_dispatch(self):
        action_run = self._request_finish_approval()
        self.workspace.default_policy = {"allowed_actions": []}
        self.workspace.save(update_fields=["default_policy", "updated_at"])

        with patch(
            "ai_hub.services.game_action_dispatcher.dispatch_game_action",
            return_value={"complete": True},
        ) as mocked_dispatch:
            with self.assertRaises(ValidationError):
                approve_action_run(
                    action_run_id=action_run.pk,
                    reviewed_by=self.approver,
                )

        mocked_dispatch.assert_not_called()
        approval = action_run.approval_request
        approval.refresh_from_db()
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.REJECTED)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_dispatch_reloads_action_changed_after_approval_review(self):
        from ai_hub.services import game_action_dispatcher as dispatcher

        action_run, action, _ = self._request_tool_approval(
            name="approval_post_review_action_drift"
        )
        original_dispatch = dispatcher.dispatch_game_action

        def mutate_then_dispatch(**kwargs):
            GameActionDefinition.objects.filter(pk=action.pk).update(
                config={"mode": "changed-after-review"}
            )
            return original_dispatch(**kwargs)

        with (
            patch(
                "ai_hub.services.game_action_dispatcher.dispatch_game_action",
                side_effect=mutate_then_dispatch,
            ),
            patch(
                "ai_hub.services.game_action_dispatcher.execute_tool",
                return_value={"result": "must not execute"},
            ) as mocked_execute,
            self.assertRaisesMessage(
                ValidationError,
                "APPROVAL_REAPPROVAL_REQUIRED",
            ),
        ):
            approve_action_run(
                action_run_id=action_run.pk,
                reviewed_by=self.approver,
            )

        mocked_execute.assert_not_called()
        action_run.refresh_from_db()
        approval = GameActionApprovalRequest.objects.get(action_run=action_run)
        self.assertEqual(action_run.status, GameActionRun.Status.REJECTED)
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.REJECTED)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_dispatch_reloads_workspace_changed_after_approval_review(self):
        from ai_hub.services import game_action_dispatcher as dispatcher

        action_run, _, _ = self._request_tool_approval(
            name="approval_post_review_workspace_drift"
        )
        original_dispatch = dispatcher.dispatch_game_action

        def mutate_then_dispatch(**kwargs):
            GameWorkspace.objects.filter(pk=self.workspace.pk).update(
                default_policy={"allowed_tools": []}
            )
            return original_dispatch(**kwargs)

        with (
            patch(
                "ai_hub.services.game_action_dispatcher.dispatch_game_action",
                side_effect=mutate_then_dispatch,
            ),
            patch(
                "ai_hub.services.game_action_dispatcher.execute_tool",
                return_value={"result": "must not execute"},
            ) as mocked_execute,
            self.assertRaisesMessage(
                ValidationError,
                "APPROVAL_REAPPROVAL_REQUIRED",
            ),
        ):
            approve_action_run(
                action_run_id=action_run.pk,
                reviewed_by=self.approver,
            )

        mocked_execute.assert_not_called()
        action_run.refresh_from_db()
        approval = GameActionApprovalRequest.objects.get(action_run=action_run)
        self.assertEqual(action_run.status, GameActionRun.Status.REJECTED)
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.REJECTED)

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_dispatch_executes_the_tool_snapshot_used_for_intent_verification(self):
        from ai_hub.services import game_action_dispatcher as dispatcher

        reviewed_config = {"template": "reviewed tool configuration"}
        action_run, _, tool = self._request_tool_approval(
            name="approval_single_tool_resolution",
            tool_config=reviewed_config,
        )
        original_resolver = dispatcher._resolve_unified_tool_authorization
        dispatch_resolution_count = 0
        mutated = False
        executed_configs = []
        external_execution_atomic_depths = []
        baseline_atomic_depth = len(connection.atomic_blocks)

        def resolve_then_mutate(**kwargs):
            nonlocal dispatch_resolution_count, mutated
            agent, resolved_tool = original_resolver(**kwargs)
            approval_status = GameActionApprovalRequest.objects.get(
                action_run_id=action_run.pk
            ).status
            if approval_status == GameActionApprovalRequest.Status.APPROVED:
                dispatch_resolution_count += 1
                if not mutated:
                    ToolDefinition.objects.filter(pk=tool.pk).update(
                        config={"template": "changed-after-linearization"}
                    )
                    mutated = True
            return agent, resolved_tool

        def execute_resolved_tool(executed_tool, payload, *, agent=None):
            executed_configs.append(dict(executed_tool.config or {}))
            external_execution_atomic_depths.append(len(connection.atomic_blocks))
            return {"result": "executed reviewed capability"}

        with (
            patch(
                "ai_hub.services.game_action_dispatcher._resolve_unified_tool_authorization",
                side_effect=resolve_then_mutate,
            ),
            patch(
                "ai_hub.services.game_action_dispatcher.execute_tool",
                side_effect=execute_resolved_tool,
            ),
        ):
            approved = approve_action_run(
                action_run_id=action_run.pk,
                reviewed_by=self.approver,
            )

        self.assertEqual(approved.status, GameActionRun.Status.SUCCESS)
        self.assertEqual(dispatch_resolution_count, 1)
        self.assertEqual(executed_configs, [reviewed_config])
        self.assertEqual(external_execution_atomic_depths, [baseline_atomic_depth])

    def test_approval_rejects_input_payload_tampering_before_dispatch(self):
        action_run = self._request_finish_approval(final_answer="reviewed value")
        action_run.input_payload = {"final_answer": "tampered value"}
        action_run.save(update_fields=["input_payload"])

        with patch(
            "ai_hub.services.game_action_dispatcher.dispatch_game_action",
            return_value={"complete": True},
        ) as mocked_dispatch:
            with self.assertRaises(ValidationError):
                approve_action_run(
                    action_run_id=action_run.pk,
                    reviewed_by=self.approver,
                )

        mocked_dispatch.assert_not_called()
        approval = action_run.approval_request
        approval.refresh_from_db()
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.REJECTED)

    def test_approval_request_metadata_change_does_not_create_false_drift(self):
        action_run = self._request_finish_approval()
        approval = action_run.approval_request
        approval.expires_at = timezone.now() + timedelta(hours=1)
        approval.review_note = "Reviewer-only draft note"
        approval.save(update_fields=["expires_at", "review_note"])

        approved = approve_action_run(
            action_run_id=action_run.pk,
            reviewed_by=self.approver,
            review_note="Approved unchanged intent.",
        )

        approved.refresh_from_db()
        self.assertEqual(approved.status, GameActionRun.Status.SUCCESS)

    def test_historical_approval_without_fingerprint_requires_fresh_request(self):
        action_run, approval = self._make_waiting_approval_run()
        approval.execution_intent_snapshot = {}
        approval.execution_intent_fingerprint = ""
        approval.save(
            update_fields=[
                "execution_intent_snapshot",
                "execution_intent_fingerprint",
            ]
        )

        with patch(
            "ai_hub.services.game_action_dispatcher.dispatch_game_action",
            return_value={"complete": True},
        ) as mocked_dispatch:
            with self.assertRaisesMessage(
                ValidationError,
                "APPROVAL_REAPPROVAL_REQUIRED",
            ):
                approve_action_run(
                    action_run_id=action_run.pk,
                    reviewed_by=self.approver,
                )

        mocked_dispatch.assert_not_called()
        approval.refresh_from_db()
        action_run.refresh_from_db()
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.REJECTED)
        self.assertEqual(action_run.status, GameActionRun.Status.REJECTED)

    def test_low_level_dispatch_rechecks_approved_intent_fingerprint(self):
        from ai_hub.services.game_action_dispatcher import dispatch_game_action

        action_run = self._request_finish_approval()
        approval = action_run.approval_request
        approval.status = GameActionApprovalRequest.Status.APPROVED
        approval.save(update_fields=["status"])
        action_run.status = GameActionRun.Status.RUNNING
        action_run.save(update_fields=["status"])
        self.finish_def.config = {"mode": "changed-before-low-level-dispatch"}
        self.finish_def.save(update_fields=["config", "updated_at"])
        action_run.refresh_from_db()

        with self.assertRaisesMessage(
            ValidationError,
            "APPROVAL_REAPPROVAL_REQUIRED",
        ):
            dispatch_game_action(
                action_run=action_run,
                workspace=self.workspace,
                goal=self.goal,
                payload=dict(action_run.input_payload),
            )

    @override_settings(AI_HUB_UNIFIED_TOOL_RUNTIME_ENABLED=True)
    def test_approval_audit_snapshot_never_persists_plaintext_credentials(self):
        action_run, _, _ = self._request_tool_approval(
            name="approval_secret_redaction",
            tool_config={
                "template": "safe result",
                "api_key": "tool-plaintext-secret",
                "headers": {"Authorization": "Bearer header-plaintext-secret"},
            },
            action_config={"password": "action-plaintext-secret"},
            action_input={
                "query": "approved input",
                "access_token": "payload-plaintext-secret",
            },
        )

        approval = action_run.approval_request
        persisted_approval_audit = json.dumps(
            {
                "requested_payload": approval.requested_payload,
                "execution_intent_snapshot": approval.execution_intent_snapshot,
                "execution_intent_fingerprint": approval.execution_intent_fingerprint,
            },
            sort_keys=True,
        )
        for plaintext in (
            "tool-plaintext-secret",
            "header-plaintext-secret",
            "action-plaintext-secret",
            "payload-plaintext-secret",
        ):
            self.assertNotIn(plaintext, persisted_approval_audit)
        self.assertIn("***REDACTED***", persisted_approval_audit)
        self.assertEqual(len(approval.execution_intent_fingerprint), 64)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_resume_preserves_historical_step_runs(self, mocked_call):
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"action": "finish", "complete": true, "final_answer": "Resume done.", "message": "ok"}',
        }
        # Manually create historical step runs (simulating a previous partial run)
        ExecutionStepRun.objects.create(
            session=self.session, order=1, agent=self.agent,
            action_name="game_iteration", status=ExecutionStepRun.Status.SUCCESS,
        )
        ExecutionStepRun.objects.create(
            session=self.session, order=2, agent=self.agent,
            action_name="game_iteration", status=ExecutionStepRun.Status.SUCCESS,
        )
        self.session.status = ExecutionSession.Status.WAITING_ASYNC
        self.session.final_context = {
            "memory": [{"iteration": 1, "summary": "step 1"}, {"iteration": 2, "summary": "step 2"}],
            "observations": [],
            "goal": self.session.goal_text,
        }
        self.session.save(update_fields=["status", "final_context", "updated_at"])
        GameContinuationRequest.objects.create(
            session=self.session, goal=self.goal, reason_code="needs_information"
        )
        transition_goal_status(self.goal, GameGoal.Status.WAITING_INFO, reason="paused")

        resume_goal_execution(session_id=self.session.pk)
        self.session.refresh_from_db()

        self.assertEqual(self.session.status, ExecutionSession.Status.SUCCESS)
        orders = list(self.session.step_runs.order_by("order").values_list("order", flat=True))
        self.assertIn(1, orders)
        self.assertIn(2, orders)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_resume_uses_next_step_run_order(self, mocked_call):
        mocked_call.return_value = {
            "status": "ok",
            "content": '{"action": "finish", "complete": true, "final_answer": "done.", "message": "ok"}',
        }
        ExecutionStepRun.objects.create(
            session=self.session, order=1, agent=self.agent,
            action_name="game_iteration", status=ExecutionStepRun.Status.SUCCESS,
        )
        ExecutionStepRun.objects.create(
            session=self.session, order=2, agent=self.agent,
            action_name="game_iteration", status=ExecutionStepRun.Status.SUCCESS,
        )
        self.session.status = ExecutionSession.Status.WAITING_ASYNC
        self.session.final_context = {
            "memory": [], "observations": [], "goal": self.session.goal_text,
        }
        self.session.save(update_fields=["status", "final_context", "updated_at"])
        GameContinuationRequest.objects.create(
            session=self.session, goal=self.goal, reason_code="needs_information"
        )
        transition_goal_status(self.goal, GameGoal.Status.WAITING_INFO, reason="paused")

        resume_goal_execution(session_id=self.session.pk)

        new_step_orders = list(
            self.session.step_runs.filter(order__gte=3).values_list("order", flat=True)
        )
        self.assertTrue(len(new_step_orders) >= 1, "Expected at least one new step run at order >= 3")
        self.assertEqual(min(new_step_orders), 3)

    def test_resume_does_not_repeat_completed_idempotent_action(self):
        run1 = execute_game_action(
            session=self.session,
            action_name="finish_goal",
            action_input={"final_answer": "idempotent answer"},
        )
        self.assertEqual(run1.status, GameActionRun.Status.SUCCESS)

        # Call again with identical parameters — must return same run
        run2 = execute_game_action(
            session=self.session,
            action_name="finish_goal",
            action_input={"final_answer": "idempotent answer"},
        )
        self.assertEqual(run1.pk, run2.pk)
        self.assertEqual(
            GameActionRun.objects.filter(session=self.session, action_name="finish_goal").count(),
            1,
        )

    def test_cancelled_goal_cannot_resume(self):
        self._pause_session()
        transition_goal_status(self.goal, GameGoal.Status.CANCELLED, reason="manual cancel")

        with self.assertRaisesMessage(ValidationError, "cancelled goal"):
            resume_goal_execution(session_id=self.session.pk)

    def test_expired_approval_cannot_resume_without_new_request(self):
        action_run, approval_req = self._make_waiting_approval_run()
        # Set expiry to the past
        past = timezone.now() - timedelta(hours=2)
        approval_req.expires_at = past
        approval_req.save(update_fields=["expires_at"])

        with self.assertRaisesMessage(ValidationError, "expired"):
            approve_action_run(action_run_id=action_run.pk, reviewed_by=self.approver)

        approval_req.refresh_from_db()
        self.assertEqual(approval_req.status, GameActionApprovalRequest.Status.EXPIRED)

    def test_only_authorised_user_can_approve_action(self):
        action_run, _ = self._make_waiting_approval_run()

        with self.assertRaisesMessage(ValidationError, "permission to approve"):
            approve_action_run(action_run_id=action_run.pk, reviewed_by=self.regular_user)

        # The regular user rejection should NOT have changed the status
        action_run.refresh_from_db()
        self.assertEqual(action_run.status, GameActionRun.Status.WAITING_APPROVAL)

        # Now approve with authorized user — should succeed
        result = approve_action_run(action_run_id=action_run.pk, reviewed_by=self.approver)
        result.refresh_from_db()
        self.assertEqual(result.status, GameActionRun.Status.SUCCESS)

    def test_double_approval_is_refused(self):
        """P1.3 (approval race): a second approve on an already-approved action must
        fail via the status guard, never double-execute. The service takes
        select_for_update locks and re-checks status inside the transaction."""
        action_run, _ = self._make_waiting_approval_run()
        first = approve_action_run(action_run_id=action_run.pk, reviewed_by=self.approver)
        first.refresh_from_db()
        self.assertEqual(first.status, GameActionRun.Status.SUCCESS)

        with self.assertRaises(ValidationError):
            approve_action_run(action_run_id=action_run.pk, reviewed_by=self.approver)
        # Still exactly one run; status unchanged.
        action_run.refresh_from_db()
        self.assertEqual(action_run.status, GameActionRun.Status.SUCCESS)

    def test_reject_after_approve_is_refused(self):
        """P1.3 (approval race): once approved, the same request cannot be rejected."""
        action_run, _ = self._make_waiting_approval_run()
        approve_action_run(action_run_id=action_run.pk, reviewed_by=self.approver)
        with self.assertRaises(ValidationError):
            reject_action_run(action_run_id=action_run.pk, reviewed_by=self.approver)


# ============================================================
# Phase 09 — policies, budgets, and permissions
# ============================================================

import json as _json  # noqa: E402

from ai_hub.models import (  # noqa: E402
    GameWorkspaceAction,
    GameWorkspaceAgent,
)
from ai_hub.services.game_policy import (  # noqa: E402
    ApprovalRequiredByPolicyError,
    BudgetExhaustedError,
    PolicyViolationError,
    check_budget_before_action,
    check_budget_before_iteration,
    validate_action_policy,
    validate_goal_execution_policy,
    validate_workspace_policy,
)


class GamePoliciesTests(TestCase):
    def setUp(self):
        self.provider = ProviderConfig.objects.create(name="p-policy", provider_type="openai")
        self.model_cfg = ModelConfig.objects.create(provider=self.provider, model_name="gpt-policy")
        self.agent = AgentProfile.objects.create(
            name="policy-agent",
            role="policy-runner",
            model_config=self.model_cfg,
        )
        self.workspace = GameWorkspace.objects.create(name="policy-ws")
        self.goal = GameGoal.objects.create(
            workspace=self.workspace,
            title="Policy test goal",
            description="Testing policies",
        )
        self.low_action = GameActionDefinition.objects.create(
            name="low_risk_test",
            label="Low Risk Test",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            risk_level="low",
            requires_approval=False,
        )
        self.medium_action = GameActionDefinition.objects.create(
            name="medium_risk_test",
            label="Medium Risk Test",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            risk_level="medium",
            requires_approval=False,
        )
        self.high_action = GameActionDefinition.objects.create(
            name="high_risk_test",
            label="High Risk Test",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            risk_level="high",
            requires_approval=False,
        )

    def _make_session(self, goal=None):
        g = goal or self.goal
        return ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=g,
            goal_text=g.description,
            runtime_config={"max_iterations": 3},
            status=ExecutionSession.Status.RUNNING,
            started_at=timezone.now(),
        )

    def test_workspace_rejects_disabled_action(self):
        GameWorkspaceAction.objects.create(
            workspace=self.workspace, action=self.low_action, is_enabled=False
        )
        with self.assertRaises(PolicyViolationError):
            validate_action_policy(self.workspace, self.goal, self.low_action, {})

    def test_workspace_rejects_disabled_agent(self):
        GameWorkspaceAgent.objects.create(
            workspace=self.workspace, agent=self.agent, is_enabled=False
        )
        session = self._make_session()
        with self.assertRaises(PolicyViolationError):
            validate_goal_execution_policy(self.workspace, self.goal, session)

    def test_workspace_policy_uses_pipeline_effective_agent(self):
        pipeline_agent = AgentProfile.objects.create(
            name="policy-pipeline-agent",
            role="Pipeline policy agent",
            model_config=self.model_cfg,
        )
        pipeline = PipelineDefinition.objects.create(name="Policy GAME pipeline")
        PipelineStep.objects.create(pipeline=pipeline, agent=pipeline_agent, order=1)
        GameWorkspaceAgent.objects.create(
            workspace=self.workspace,
            agent=self.agent,
            is_enabled=True,
        )
        session = self._make_session()
        session.entry_agent = None
        session.pipeline = pipeline
        session.save(update_fields=["entry_agent", "pipeline"])

        with self.assertRaisesMessage(
            PolicyViolationError,
            pipeline_agent.name,
        ):
            validate_goal_execution_policy(self.workspace, self.goal, session)

    def test_low_risk_action_runs_without_approval_when_allowed(self):
        GameWorkspaceAction.objects.create(
            workspace=self.workspace, action=self.low_action, is_enabled=True
        )
        # Should not raise any exception
        validate_action_policy(self.workspace, self.goal, self.low_action, {})

    def test_medium_risk_action_requires_approval_when_policy_requires(self):
        self.workspace.default_policy = {"safety": {"require_approval_for_medium_risk": True}}
        self.workspace.save()
        with self.assertRaises(ApprovalRequiredByPolicyError):
            validate_action_policy(self.workspace, self.goal, self.medium_action, {})

    def test_high_risk_action_is_rejected_when_external_writes_disabled(self):
        self.workspace.default_policy = {"safety": {"allow_external_writes": False}}
        self.workspace.save()
        with self.assertRaises(PolicyViolationError):
            validate_action_policy(self.workspace, self.goal, self.high_action, {})

    def test_iteration_budget_prevents_new_iteration(self):
        self.workspace.default_policy = {"budget": {"max_iterations_per_session": 2}}
        self.workspace.save()
        session = self._make_session()
        for i in range(2):
            ExecutionStepRun.objects.create(
                session=session,
                order=i + 1,
                action_name="game_iteration",
                status=ExecutionStepRun.Status.SUCCESS,
            )
        with self.assertRaises(BudgetExhaustedError):
            check_budget_before_iteration(session)

    def test_action_budget_prevents_new_action(self):
        self.workspace.default_policy = {"budget": {"max_action_runs_per_session": 1}}
        self.workspace.save()
        session = self._make_session()
        GameActionRun.objects.create(
            session=session,
            action=self.low_action,
            idempotency_key="test-key-budget-1",
            action_name="low_risk_test",
            iteration=1,
            status=GameActionRun.Status.SUCCESS,
        )
        with self.assertRaises(BudgetExhaustedError):
            check_budget_before_action(session, self.low_action)

    def test_budget_exhaustion_marks_goal_partial_or_blocked(self):
        self.workspace.default_policy = {"budget": {"max_iterations_per_session": 1}}
        self.workspace.save()
        from ai_hub.services.game_goals import transition_goal_status
        transition_goal_status(self.goal, GameGoal.Status.RUNNING, reason="test")
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=self.goal,
            goal_text=self.goal.description,
            runtime_config={"max_iterations": 3},
            status=ExecutionSession.Status.PENDING,
            initial_context={},
        )
        mock_llm_response = {
            "status": "ok",
            "content": _json.dumps({
                "action": "think",
                "message": "Still working...",
                "complete": False,
                "final_answer": "",
            }),
        }
        with patch("ai_hub.services.agent_runtime.completion_call", return_value=mock_llm_response):
            run_execution_session(session.pk)
        session.refresh_from_db()
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(session.final_context.get("finish_reason"), "budget_exhausted")
        self.assertEqual(session.final_context.get("goal_outcome"), "incomplete")
        self.goal.refresh_from_db()
        self.assertIn(self.goal.status, {GameGoal.Status.PARTIAL, GameGoal.Status.BLOCKED})

    def test_policy_validation_rejects_invalid_limits(self):
        with self.assertRaises(ValidationError):
            validate_workspace_policy({"budget": {"max_iterations_per_session": -1}})

    def test_unknown_policy_keys_are_handled_intentionally(self):
        # Unknown top-level keys must not raise — reserved for future extensions
        validate_workspace_policy({
            "budget": {"max_iterations_per_session": 5},
            "unknown_future_key": {"some": "value"},
        })

    def test_lack_of_token_metrics_does_not_bypass_other_budgets(self):
        # Token and cost metrics are recorded but not yet enforced;
        # other limits (action runs) must still apply.
        self.workspace.default_policy = {
            "budget": {
                "max_total_tokens": 30000,
                "max_total_cost_usd": 0.25,
                "max_action_runs_per_session": 1,
            }
        }
        self.workspace.save()
        session = self._make_session()
        GameActionRun.objects.create(
            session=session,
            action=self.low_action,
            idempotency_key="test-key-tokens-1",
            action_name="low_risk_test",
            iteration=1,
            status=GameActionRun.Status.SUCCESS,
        )
        with self.assertRaises(BudgetExhaustedError):
            check_budget_before_action(session, self.low_action)


class GamePrePhase10StabilizationTests(TestCase):
    """End-to-end regressions for the Phase 06-09 readiness audit."""

    def setUp(self):
        self.provider = ProviderConfig.objects.create(
            name="stabilization-provider",
            provider_type="training",
        )
        self.model_cfg = ModelConfig.objects.create(
            provider=self.provider,
            model_name="stabilization-model",
        )
        self.agent = AgentProfile.objects.create(
            name="stabilization-agent",
            role="stabilization",
            model_config=self.model_cfg,
        )
        self.workspace = GameWorkspace.objects.create(name="stabilization-workspace")

    def make_goal(self, title="Stabilization goal"):
        return create_goal(
            workspace=self.workspace,
            title=title,
            description="Exercise the integrated GAME runtime.",
        )

    def test_disabled_workspace_agent_cannot_create_goal_session(self):
        goal = self.make_goal()
        GameWorkspaceAgent.objects.create(
            workspace=self.workspace,
            agent=self.agent,
            is_enabled=False,
        )

        with self.assertRaisesMessage(PolicyViolationError, "not enabled"):
            create_goal_execution_session(goal=goal, entry_agent=self.agent)

    def test_workspace_agent_allow_list_is_closed_when_configured(self):
        other_agent = AgentProfile.objects.create(
            name="other-stabilization-agent",
            role="other",
            model_config=self.model_cfg,
        )
        GameWorkspaceAgent.objects.create(
            workspace=self.workspace,
            agent=other_agent,
            is_enabled=True,
        )

        with self.assertRaisesMessage(PolicyViolationError, "not enabled"):
            create_goal_execution_session(goal=self.make_goal(), entry_agent=self.agent)

    def test_runner_rechecks_agent_policy_defensively(self):
        session = create_goal_execution_session(
            goal=self.make_goal(),
            entry_agent=self.agent,
        )
        GameWorkspaceAgent.objects.create(
            workspace=self.workspace,
            agent=self.agent,
            is_enabled=False,
        )

        run_execution_session(session.pk)
        session.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.FAILED)
        self.assertEqual(session.step_runs.count(), 0)
        self.assertIn("not enabled", session.error_detail)

    def test_high_risk_action_is_closed_by_default(self):
        action = GameActionDefinition.objects.create(
            name="closed_by_default",
            label="Closed by default",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            risk_level="high",
        )

        with self.assertRaisesMessage(PolicyViolationError, "allow_external_writes"):
            validate_action_policy(self.workspace, self.make_goal(), action, {})

    def test_unknown_action_risk_level_is_blocked(self):
        action = GameActionDefinition.objects.create(
            name="unknown_risk",
            label="Unknown risk",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            risk_level="critical",
        )

        with self.assertRaisesMessage(PolicyViolationError, "unknown risk level"):
            validate_action_policy(self.workspace, self.make_goal(), action, {})

    def test_workspace_model_validation_rejects_invalid_policy(self):
        self.workspace.default_policy = {
            "safety": {"allow_external_writes": "yes"},
        }

        with self.assertRaises(ValidationError):
            self.workspace.full_clean()

    def test_workspace_action_allow_list_is_closed_when_configured(self):
        allowed = GameActionDefinition.objects.create(
            name="explicitly_allowed",
            label="Allowed",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )
        absent = GameActionDefinition.objects.create(
            name="absent_from_allow_list",
            label="Absent",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )
        GameWorkspaceAction.objects.create(
            workspace=self.workspace,
            action=allowed,
            is_enabled=True,
        )

        with self.assertRaisesMessage(PolicyViolationError, "not enabled"):
            validate_action_policy(self.workspace, self.make_goal(), absent, {})

    def test_session_memory_rejects_cross_workspace_session(self):
        goal = self.make_goal()
        session = create_goal_execution_session(goal=goal, entry_agent=self.agent)
        other_workspace = GameWorkspace.objects.create(name="other-memory-workspace")
        entry = GameMemoryEntry(
            workspace=other_workspace,
            session=session,
            scope_type=GameMemoryEntry.ScopeType.SESSION,
            content="Must remain isolated.",
        )

        with self.assertRaisesMessage(ValidationError, "session goal's workspace"):
            entry.full_clean()

    def test_failed_equivalent_action_returns_controlled_error(self):
        from ai_hub.services.game_action_dispatcher import _build_idempotency_key

        action = GameActionDefinition.objects.create(
            name="finish_goal",
            label="Finish",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )
        session = create_goal_execution_session(
            goal=self.make_goal(),
            entry_agent=self.agent,
        )
        key = _build_idempotency_key(session.pk, None, action.pk, {})
        GameActionRun.objects.create(
            session=session,
            action=action,
            idempotency_key=key,
            action_name=action.name,
            iteration=1,
            status=GameActionRun.Status.FAILED,
        )

        with self.assertRaisesMessage(ValidationError, "already ended with status 'failed'"):
            execute_game_action(
                session=session,
                action_name=action.name,
                action_input={},
            )
        self.assertEqual(GameActionRun.objects.filter(session=session).count(), 1)

    def test_contract_rejection_creates_failed_action_audit(self):
        action = GameActionDefinition.objects.create(
            name="finish_goal",
            label="Finish",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            input_contract={"required": ["result"]},
        )
        session = create_goal_execution_session(
            goal=self.make_goal(),
            entry_agent=self.agent,
        )

        with self.assertRaises(ValidationError):
            execute_game_action(
                session=session,
                action_name=action.name,
                action_input={},
            )

        audit = GameActionRun.objects.get(session=session)
        self.assertEqual(audit.status, GameActionRun.Status.FAILED)
        self.assertIn("result", audit.error_detail)

    def test_policy_rejection_creates_failed_action_audit(self):
        blocked = GameActionDefinition.objects.create(
            name="finish_goal",
            label="Finish",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )
        GameWorkspaceAction.objects.create(
            workspace=self.workspace,
            action=blocked,
            is_enabled=False,
        )
        session = create_goal_execution_session(
            goal=self.make_goal(),
            entry_agent=self.agent,
        )

        with self.assertRaises(PolicyViolationError):
            execute_game_action(
                session=session,
                action_name=blocked.name,
                action_input={},
            )

        self.assertEqual(
            session.game_action_runs.get().status,
            GameActionRun.Status.FAILED,
        )

    def test_pause_rejects_unknown_reason_and_duplicate_pending_request(self):
        from ai_hub.services.game_resume import pause_session

        goal = self.make_goal()
        session = create_goal_execution_session(goal=goal, entry_agent=self.agent)
        with self.assertRaisesMessage(ValidationError, "Unknown GAME continuation reason"):
            pause_session(
                session=session,
                goal=goal,
                reason_code="invented_reason",
            )

        pause_session(
            session=session,
            goal=goal,
            reason_code=GameContinuationRequest.ReasonCode.NEEDS_INFORMATION,
        )
        with self.assertRaisesMessage(ValidationError, "already has pending continuation"):
            pause_session(
                session=session,
                goal=goal,
                reason_code=GameContinuationRequest.ReasonCode.NEEDS_INFORMATION,
            )
        self.assertEqual(session.continuation_requests.filter(status="pending").count(), 1)

    def test_model_selected_memory_write_is_marked_model_derived(self):
        GameActionDefinition.objects.create(
            name="record_memory",
            label="Record memory",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )
        session = create_goal_execution_session(
            goal=self.make_goal(),
            entry_agent=self.agent,
        )

        execute_game_action(
            session=session,
            action_name="record_memory",
            action_input={"content": "Model-proposed fact."},
        )

        entry = GameMemoryEntry.objects.get(content="Model-proposed fact.")
        self.assertTrue(entry.metadata["model_derived"])

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_approval_pause_stops_runner_without_duplicates(self, mocked_call):
        mocked_call.return_value = {
            "status": "ok",
            "content": (
                '{"action":"finish_goal","action_input":{"final_answer":"approved"},'
                '"message":"needs approval","complete":false,"final_answer":""}'
            ),
        }
        GameActionDefinition.objects.create(
            name="finish_goal",
            label="Finish",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            requires_approval=True,
        )
        goal = self.make_goal()
        session = create_goal_execution_session(
            goal=goal,
            entry_agent=self.agent,
            runtime_config={"max_iterations": 3},
        )

        run_execution_session(session.pk, use_action_dispatcher=True)
        session.refresh_from_db()
        goal.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.WAITING_ASYNC)
        self.assertEqual(goal.status, GameGoal.Status.WAITING_APPROVAL)
        self.assertEqual(session.final_context["finish_reason"], "needs_approval")
        self.assertEqual(session.step_runs.count(), 1)
        self.assertEqual(session.game_action_runs.filter(status="waiting_approval").count(), 1)
        self.assertEqual(session.continuation_requests.filter(status="pending").count(), 1)

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_approved_output_is_persisted_and_consumed_on_resume(self, mocked_call):
        mocked_call.side_effect = [
            {
                "status": "ok",
                "content": (
                    '{"action":"finish_goal","action_input":{"final_answer":"approved"},'
                    '"message":"needs approval","complete":false,"final_answer":""}'
                ),
            },
            {
                "status": "ok",
                "content": (
                    '{"action":"finish","message":"used approved result",'
                    '"complete":true,"final_answer":"done"}'
                ),
            },
        ]
        GameActionDefinition.objects.create(
            name="finish_goal",
            label="Finish",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            requires_approval=True,
        )
        goal = self.make_goal()
        session = create_goal_execution_session(
            goal=goal,
            entry_agent=self.agent,
            runtime_config={"max_iterations": 3},
        )
        run_execution_session(session.pk, use_action_dispatcher=True)
        action_run = session.game_action_runs.get()
        reviewer = get_user_model().objects.create_superuser(
            username="stabilization-reviewer",
            email="reviewer@example.com",
            password="test",
        )

        with self.assertRaisesMessage(ValidationError, "approved or rejected"):
            resume_goal_execution(session_id=session.pk)

        approve_action_run(action_run_id=action_run.pk, reviewed_by=reviewer)
        session.refresh_from_db()
        self.assertTrue(
            any(
                item.get("action_run_id") == action_run.pk
                and item.get("resolution_status") == "approved"
                for item in session.final_context["observations"]
            )
        )

        resumed = resume_goal_execution(session_id=session.pk)
        self.assertEqual(resumed.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(resumed.step_runs.count(), 2)
        self.assertTrue(resumed.final_context["game_action_dispatch_enabled"])
        resumed_request = resumed.step_runs.get(order=2).request_payload
        self.assertTrue(
            any(
                item.get("action_run_id") == action_run.pk
                and item.get("resolution_status") == "approved"
                for item in resumed_request["observations"]
            )
        )

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_scoped_memory_is_injected_into_goal_runner_payload(self, mocked_call):
        mocked_call.return_value = {
            "status": "ok",
            "content": (
                '{"action":"finish","message":"done",'
                '"complete":true,"final_answer":"done"}'
            ),
        }
        goal = self.make_goal()
        record_memory(
            scope_type=GameMemoryEntry.ScopeType.WORKSPACE,
            workspace=self.workspace,
            content="Shared scoped fact.",
        )
        session = create_goal_execution_session(goal=goal, entry_agent=self.agent)

        run_execution_session(session.pk)

        step = session.step_runs.get()
        self.assertEqual(
            step.request_payload["scoped_memory"]["entries"][0]["content"],
            "Shared scoped fact.",
        )


class GameApprovalConcurrencyTests(TransactionTestCase):
    reset_sequences = True

    def test_only_one_concurrent_reviewer_executes_approved_action(self):
        if connection.vendor != "postgresql":
            self.skipTest("Approval row-locking requires PostgreSQL semantics.")

        from threading import Barrier, Lock

        provider = ProviderConfig.objects.create(name="approval-lock-provider", provider_type="training")
        model_cfg = ModelConfig.objects.create(provider=provider, model_name="approval-lock-model")
        agent = AgentProfile.objects.create(
            name="approval-lock-agent",
            role="approval-lock",
            model_config=model_cfg,
        )
        workspace = GameWorkspace.objects.create(name="approval-lock-workspace")
        goal = create_goal(
            workspace=workspace,
            title="Approval locking",
            description="Only one reviewer may execute.",
        )
        session = create_goal_execution_session(goal=goal, entry_agent=agent)
        GameActionDefinition.objects.create(
            name="finish_goal",
            label="Finish",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            requires_approval=True,
        )
        action_run = execute_game_action(
            session=session,
            action_name="finish_goal",
            action_input={"final_answer": "approved once"},
        )
        reviewer = get_user_model().objects.create_superuser(
            username="approval-lock-reviewer",
            email="lock@example.com",
            password="test",
        )
        barrier = Barrier(2)
        result_lock = Lock()
        results = []

        def review():
            close_old_connections()
            try:
                local_reviewer = get_user_model().objects.get(pk=reviewer.pk)
                barrier.wait(timeout=10)
                approve_action_run(
                    action_run_id=action_run.pk,
                    reviewed_by=local_reviewer,
                )
                result = "success"
            except Exception as exc:
                result = type(exc).__name__
            finally:
                close_old_connections()
            with result_lock:
                results.append(result)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(review) for _ in range(2)]
            for future in futures:
                future.result(timeout=20)

        action_run.refresh_from_db()
        self.assertEqual(action_run.status, GameActionRun.Status.SUCCESS)
        self.assertEqual(results.count("success"), 1)
        self.assertEqual(len(results), 2)


# ============================================================
# Phase 10 — plans and multi-agent delegation
# ============================================================

from ai_hub.models import (  # noqa: E402
    GameDelegationRun,
    GameGoalPlan,
    GameGoalPlanStep,
)
from ai_hub.services.game_delegation import run_delegated_agent  # noqa: E402
from ai_hub.services.game_plans import add_plan_step, create_plan  # noqa: E402
from ai_hub.services.game_policy import (  # noqa: E402
    check_delegation_depth,
    get_session_workspace,
)


class GamePlansAndDelegationTests(TestCase):
    def setUp(self):
        self.provider = ProviderConfig.objects.create(name="p-plans", provider_type="openai")
        self.model_cfg = ModelConfig.objects.create(
            provider=self.provider, model_name="gpt-plans"
        )
        self.agent = AgentProfile.objects.create(
            name="plans-parent-agent",
            role="coordinator",
            model_config=self.model_cfg,
        )
        self.target_agent = AgentProfile.objects.create(
            name="plans-target-agent",
            role="specialist",
            model_config=self.model_cfg,
        )
        self.workspace = GameWorkspace.objects.create(name="plans-ws")
        self.goal = GameGoal.objects.create(
            workspace=self.workspace,
            title="Plans test goal",
            description="Testing plans and delegation.",
        )
        self.delegate_action = GameActionDefinition.objects.create(
            name="delegate_to_agent",
            label="Delegate to Agent",
            action_type=GameActionDefinition.ActionType.SUB_AGENT,
            risk_level="medium",
            requires_approval=False,
        )

    def _make_running_session(self, goal=None):
        g = goal or self.goal
        return ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=g,
            goal_text=g.description,
            runtime_config={"max_iterations": 3},
            status=ExecutionSession.Status.RUNNING,
            started_at=timezone.now(),
        )

    def _make_action_run(self, session, key_suffix="default"):
        return GameActionRun.objects.create(
            session=session,
            action=self.delegate_action,
            idempotency_key=f"delegation-test-{key_suffix}",
            action_name="delegate_to_agent",
            iteration=1,
            status=GameActionRun.Status.RUNNING,
            input_payload={"agent_name": self.target_agent.name, "task": "Test task."},
            started_at=timezone.now(),
        )

    # ---- Phase 10 spec test names ------------------------------------------

    def test_agent_can_create_valid_plan(self):
        plan = create_plan(goal=self.goal, summary="Initial execution plan")
        step = add_plan_step(plan=plan, title="Gather context", order=1)
        self.assertEqual(plan.goal, self.goal)
        self.assertEqual(plan.summary, "Initial execution plan")
        self.assertEqual(plan.status, GameGoalPlan.Status.ACTIVE)
        self.assertEqual(step.plan, plan)
        self.assertEqual(step.order, 1)

    def test_plan_step_order_is_unique_per_plan(self):
        plan = create_plan(goal=self.goal)
        add_plan_step(plan=plan, title="Step 1", order=1)
        with self.assertRaises(ValidationError):
            add_plan_step(plan=plan, title="Step 1 duplicate", order=1)

    def test_plan_step_dependency_is_validated(self):
        plan1 = create_plan(goal=self.goal)
        other_goal = GameGoal.objects.create(
            workspace=self.workspace, title="Other goal", description="Other."
        )
        plan2 = create_plan(goal=other_goal)
        step_from_plan1 = add_plan_step(plan=plan1, title="Step A", order=1)
        with self.assertRaises(ValidationError):
            add_plan_step(
                plan=plan2,
                title="Step B cross-plan",
                order=1,
                depends_on_step=step_from_plan1,
            )

    def test_delegation_requires_allowed_target_agent(self):
        # Workspace has an entry for a *different* agent — allow-list is now closed.
        other_agent = AgentProfile.objects.create(
            name="plans-other-agent", role="other", model_config=self.model_cfg
        )
        GameWorkspaceAgent.objects.create(
            workspace=self.workspace, agent=other_agent, is_enabled=True
        )
        session = self._make_running_session()
        action_run = self._make_action_run(session, key_suffix="requires-allowed")
        with self.assertRaises(PolicyViolationError):
            run_delegated_agent(
                session=session,
                action_run=action_run,
                workspace=self.workspace,
                goal=self.goal,
                target_agent_name=self.target_agent.name,
                task="Do something.",
            )

    def test_delegated_session_recovers_parent_workspace_policy_context(self):
        parent_session = self._make_running_session()
        parent_action_run = self._make_action_run(
            parent_session,
            key_suffix="workspace-context",
        )
        delegated_session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.target_agent,
            goal_text="Delegated task",
            status=ExecutionSession.Status.PENDING,
        )
        GameDelegationRun.objects.create(
            parent_action_run=parent_action_run,
            parent_goal=self.goal,
            delegated_session=delegated_session,
            target_agent=self.target_agent,
            status=GameDelegationRun.Status.RUNNING,
            task="Delegated task",
        )

        self.assertEqual(get_session_workspace(delegated_session), self.workspace)

    def test_delegation_counts_against_budget(self):
        self.workspace.default_policy = {"budget": {"max_sub_agent_runs_per_goal": 1}}
        self.workspace.save()
        session = self._make_running_session()
        action_run1 = self._make_action_run(session, key_suffix="budget-1")
        action_run2 = self._make_action_run(session, key_suffix="budget-2")
        mock_response = {
            "status": "ok",
            "content": _json.dumps({
                "action": "finish",
                "message": "Done.",
                "complete": True,
                "final_answer": "Result.",
            }),
        }
        with patch("ai_hub.services.agent_runtime.completion_call", return_value=mock_response):
            run_delegated_agent(
                session=session,
                action_run=action_run1,
                workspace=self.workspace,
                goal=self.goal,
                target_agent_name=self.target_agent.name,
                task="First task.",
            )
        with self.assertRaises(BudgetExhaustedError):
            run_delegated_agent(
                session=session,
                action_run=action_run2,
                workspace=self.workspace,
                goal=self.goal,
                target_agent_name=self.target_agent.name,
                task="Second task.",
            )

    def test_delegated_result_becomes_parent_observation(self):
        session = self._make_running_session()
        action_run = self._make_action_run(session, key_suffix="result-obs")
        mock_response = {
            "status": "ok",
            "content": _json.dumps({
                "action": "finish",
                "message": "Found the docs.",
                "complete": True,
                "final_answer": "Documentation found at section 4.2.",
            }),
        }
        with patch("ai_hub.services.agent_runtime.completion_call", return_value=mock_response):
            result = run_delegated_agent(
                session=session,
                action_run=action_run,
                workspace=self.workspace,
                goal=self.goal,
                target_agent_name=self.target_agent.name,
                task="Find documentation.",
            )
        self.assertEqual(result["status"], GameDelegationRun.Status.SUCCESS)
        self.assertIn("Documentation found", result["result_summary"])
        delegation_run = GameDelegationRun.objects.get(parent_action_run=action_run)
        self.assertEqual(delegation_run.status, GameDelegationRun.Status.SUCCESS)
        self.assertIn("Documentation found", delegation_run.result_summary)

    def test_delegation_depth_limit_is_enforced(self):
        # Create a parent session and mark a separate session as its delegation output.
        parent_session = self._make_running_session()
        parent_action_run = self._make_action_run(parent_session, key_suffix="depth-parent")
        # Delegated session has no goal link (matches what run_delegated_agent produces).
        delegated_session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.target_agent,
            goal_text="Delegated task.",
            runtime_config={"max_iterations": 3},
            status=ExecutionSession.Status.RUNNING,
            started_at=timezone.now(),
        )
        GameDelegationRun.objects.create(
            parent_action_run=parent_action_run,
            parent_goal=self.goal,
            delegated_session=delegated_session,
            target_agent=self.target_agent,
            task="Delegated task.",
            status=GameDelegationRun.Status.RUNNING,
        )
        # The delegated session itself is now a depth-1 session;
        # any further delegation attempt from it must be rejected.
        with self.assertRaises(PolicyViolationError):
            check_delegation_depth(delegated_session)

    def test_agent_cannot_delegate_to_disallowed_workspace_agent(self):
        GameWorkspaceAgent.objects.create(
            workspace=self.workspace, agent=self.target_agent, is_enabled=False
        )
        session = self._make_running_session()
        action_run = self._make_action_run(session, key_suffix="disallowed")
        with self.assertRaises(PolicyViolationError):
            run_delegated_agent(
                session=session,
                action_run=action_run,
                workspace=self.workspace,
                goal=self.goal,
                target_agent_name=self.target_agent.name,
                task="Do something.",
            )

    def test_delegated_agent_does_not_receive_unrelated_goal_memory(self):
        GameMemoryEntry.objects.create(
            workspace=self.workspace,
            scope_type=GameMemoryEntry.ScopeType.WORKSPACE,
            content="Sensitive workspace information not relevant to delegation task.",
        )
        session = self._make_running_session()
        action_run = self._make_action_run(session, key_suffix="memory-isolation")
        mock_response = {
            "status": "ok",
            "content": _json.dumps({
                "action": "finish",
                "message": "Done.",
                "complete": True,
                "final_answer": "Done.",
            }),
        }
        with patch("ai_hub.services.agent_runtime.completion_call", return_value=mock_response):
            run_delegated_agent(
                session=session,
                action_run=action_run,
                workspace=self.workspace,
                goal=self.goal,
                target_agent_name=self.target_agent.name,
                task="Specific delegated task.",
            )
        delegation_run = GameDelegationRun.objects.get(parent_action_run=action_run)
        context = delegation_run.delegated_session.initial_context or {}
        # Narrowed context must not include pre-loaded memory entries.
        self.assertNotIn("memory", context)
        self.assertNotIn("Sensitive workspace information", str(context))

    def test_failed_delegation_follows_configured_error_policy(self):
        session = self._make_running_session()
        action_run = self._make_action_run(session, key_suffix="failed-delegation")
        with patch(
            "ai_hub.services.agent_runtime.completion_call",
            side_effect=Exception("LLM unavailable"),
        ):
            with self.assertRaises(ValidationError):
                run_delegated_agent(
                    session=session,
                    action_run=action_run,
                    workspace=self.workspace,
                    goal=self.goal,
                    target_agent_name=self.target_agent.name,
                    task="Task that will fail.",
                )
        delegation_run = GameDelegationRun.objects.get(parent_action_run=action_run)
        self.assertEqual(delegation_run.status, GameDelegationRun.Status.FAILED)


# ============================================================
# Phase 11 — Admin and operational UX
# ============================================================

from ai_hub.services.game_operational_ux import (  # noqa: E402
    build_goal_detail_context,
    build_scheduler_explanation,
    build_workspace_dashboard_context,
    redact_payload,
)


class GameAdminOperationalUXTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.staff = User.objects.create_superuser(
            username="ux-staff", password="test", email="ux@test.com"
        )
        self.provider = ProviderConfig.objects.create(name="ux-prov", provider_type="openai")
        self.model_cfg = ModelConfig.objects.create(provider=self.provider, model_name="gpt-ux")
        self.agent = AgentProfile.objects.create(
            name="ux-agent", role="r", model_config=self.model_cfg
        )
        self.workspace = create_workspace(name="ux-ws", description="test")
        self.goal = create_goal(
            workspace=self.workspace,
            title="UX goal",
            description="test",
        )

    def test_workspace_dashboard_scopes_data_to_workspace(self):
        other_ws = create_workspace(name="ux-other-ws", description="other")
        create_goal(workspace=other_ws, title="Other goal", description="d")

        ctx = build_workspace_dashboard_context(self.workspace)

        self.assertEqual(ctx["workspace"], self.workspace)
        total = sum(ctx["status_counts"].values())
        self.assertEqual(total, 1)

    def test_goal_detail_shows_session_history(self):
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=self.goal,
            goal_text="test",
            status=ExecutionSession.Status.SUCCESS,
        )

        ctx = build_goal_detail_context(self.goal)

        self.assertIn("session_history", ctx)
        self.assertIn(session.pk, [s.pk for s in ctx["session_history"]])

    def test_goal_detail_shows_action_runs_in_order(self):
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=self.goal,
            goal_text="test",
            status=ExecutionSession.Status.RUNNING,
        )
        action_def = GameActionDefinition.objects.create(
            name="ux11-action",
            label="UX11",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )
        run1 = GameActionRun.objects.create(
            session=session,
            action=action_def,
            action_name="ux11-action",
            idempotency_key="ux11-k1",
            iteration=1,
            status=GameActionRun.Status.SUCCESS,
        )
        run2 = GameActionRun.objects.create(
            session=session,
            action=action_def,
            action_name="ux11-action",
            idempotency_key="ux11-k2",
            iteration=2,
            status=GameActionRun.Status.SUCCESS,
        )

        ctx = build_goal_detail_context(self.goal)

        action_run_ids = [r.pk for r in ctx["action_runs"]]
        self.assertIn(run1.pk, action_run_ids)
        self.assertIn(run2.pk, action_run_ids)
        self.assertEqual(action_run_ids, [run1.pk, run2.pk])

    def test_scheduler_explanation_is_visible(self):
        explanation = build_scheduler_explanation(self.goal)

        self.assertIn("base_priority", explanation)
        self.assertIn("bonuses", explanation)
        self.assertIn("total", explanation)
        self.assertEqual(explanation["total"], explanation["base_priority"])

    def test_pending_approval_is_visible_only_to_authorised_staff(self):
        User = get_user_model()
        non_approver = User.objects.create_user(
            username="ux-nonapprover", password="test", is_staff=True
        )
        ct = ContentType.objects.get_for_model(GameActionApprovalRequest)
        view_perm = Permission.objects.get(
            content_type=ct, codename="view_gameactionapprovalrequest"
        )
        non_approver.user_permissions.add(view_perm)

        action_def = GameActionDefinition.objects.create(
            name="ux11-appr",
            label="UX11 Approval",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            requires_approval=True,
        )
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=self.goal,
            goal_text="test",
            status=ExecutionSession.Status.WAITING_ASYNC,
        )
        action_run = GameActionRun.objects.create(
            session=session,
            action=action_def,
            action_name="ux11-appr",
            idempotency_key="ux11-appr-k",
            iteration=1,
            status=GameActionRun.Status.WAITING_APPROVAL,
        )
        approval = GameActionApprovalRequest.objects.create(
            action_run=action_run,
            goal=self.goal,
            status=GameActionApprovalRequest.Status.PENDING,
        )

        client = Client()
        client.force_login(non_approver)
        client.post(
            reverse("admin:ai_hub_gameactionapprovalrequest_changelist"),
            {
                "action": "approve_selected_actions",
                "_selected_action": [str(approval.pk)],
            },
        )

        approval.refresh_from_db()
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.PENDING)

    def test_approve_control_uses_service_layer(self):
        action_def = GameActionDefinition.objects.create(
            name="ux11-sl",
            label="UX11 SL",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            requires_approval=True,
        )
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=self.goal,
            goal_text="test",
            status=ExecutionSession.Status.WAITING_ASYNC,
        )
        action_run = GameActionRun.objects.create(
            session=session,
            action=action_def,
            action_name="ux11-sl",
            idempotency_key="ux11-sl-k",
            iteration=1,
            status=GameActionRun.Status.WAITING_APPROVAL,
        )
        approval = GameActionApprovalRequest.objects.create(
            action_run=action_run,
            goal=self.goal,
            status=GameActionApprovalRequest.Status.PENDING,
        )

        client = Client()
        client.force_login(self.staff)
        with patch("ai_hub.admin.approve_action_run") as mock_approve:
            client.post(
                reverse("admin:ai_hub_gameactionapprovalrequest_changelist"),
                {
                    "action": "approve_selected_actions",
                    "_selected_action": [str(approval.pk)],
                },
            )
            mock_approve.assert_called_once_with(
                action_run_id=action_run.pk,
                reviewed_by=self.staff,
            )

    def test_resume_control_only_shows_for_resumable_goal(self):
        ctx = build_goal_detail_context(self.goal)
        self.assertFalse(ctx["is_resumable"])

        self.goal = transition_goal_status(self.goal, GameGoal.Status.RUNNING, reason="test")
        self.goal = transition_goal_status(self.goal, GameGoal.Status.WAITING_APPROVAL, reason="test")
        waiting_session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=self.goal,
            goal_text="test",
            status=ExecutionSession.Status.WAITING_ASYNC,
        )
        GameContinuationRequest.objects.create(
            session=waiting_session,
            goal=self.goal,
            reason_code=GameContinuationRequest.ReasonCode.NEEDS_INFORMATION,
        )

        ctx = build_goal_detail_context(self.goal)
        self.assertTrue(ctx["is_resumable"])

    def test_sensitive_payload_values_are_redacted(self):
        payload = {
            "username": "alice",
            "api_key": "secret-key-123",
            "data": {
                "result": "ok",
                "password": "mypassword",
            },
            "token": "bearer-abc",
        }

        redacted = redact_payload(payload)

        self.assertEqual(redacted["username"], "alice")
        self.assertEqual(redacted["api_key"], "***REDACTED***")
        self.assertEqual(redacted["token"], "***REDACTED***")
        self.assertEqual(redacted["data"]["result"], "ok")
        self.assertEqual(redacted["data"]["password"], "***REDACTED***")

    def test_action_run_admin_masks_sensitive_payload(self):
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=self.goal,
            goal_text="test",
            status=ExecutionSession.Status.RUNNING,
        )
        action_def = GameActionDefinition.objects.create(
            name="ux11-redact",
            label="UX11 Redact",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )
        action_run = GameActionRun.objects.create(
            session=session,
            action=action_def,
            action_name="ux11-redact",
            idempotency_key="ux11-redact-k",
            iteration=1,
            status=GameActionRun.Status.SUCCESS,
            input_payload={"api_key": "super-secret-xyz", "prompt": "hello world"},
        )

        client = Client()
        client.force_login(self.staff)
        response = client.get(
            reverse("admin:ai_hub_gameactionrun_change", args=[action_run.pk])
        )

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn("***REDACTED***", body)
        self.assertNotIn("super-secret-xyz", body)
        self.assertIn("hello world", body)

    def test_goal_change_form_renders_session_card(self):
        ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=self.goal,
            goal_text="test",
            source_label="Support ticket #4832",
            status=ExecutionSession.Status.SUCCESS,
        )

        client = Client()
        client.force_login(self.staff)
        response = client.get(
            reverse("admin:ai_hub_gamegoal_change", args=[self.goal.pk])
        )

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn("Support ticket #4832", body)

    def test_goal_change_form_session_card_falls_back_to_pk(self):
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=self.goal,
            goal_text="test",
            source_label="",
            status=ExecutionSession.Status.SUCCESS,
        )

        client = Client()
        client.force_login(self.staff)
        response = client.get(
            reverse("admin:ai_hub_gamegoal_change", args=[self.goal.pk])
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(f"Session #{session.pk}", response.content.decode())

    def test_workspace_dashboard_view_renders_eligible_and_action_runs(self):
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=self.goal,
            goal_text="test",
            status=ExecutionSession.Status.RUNNING,
        )
        action_def = GameActionDefinition.objects.create(
            name="ux11-dash",
            label="UX11 Dash",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )
        GameActionRun.objects.create(
            session=session,
            action=action_def,
            action_name="ux11-dash-action",
            idempotency_key="ux11-dash-k",
            iteration=1,
            status=GameActionRun.Status.SUCCESS,
        )

        client = Client()
        client.force_login(self.staff)
        response = client.get(
            reverse("admin:ai_hub_gameworkspace_dashboard", args=[self.workspace.pk])
        )

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn(self.goal.title, body)  # top_eligible panel
        self.assertIn("ux11-dash-action", body)  # recent action runs panel


# ============================================================
# Phase 12 — Feature flags
# ============================================================

from ai_hub.services.game_feature_flags import require_game_feature  # noqa: E402


class GameFeatureFlagTests(TestCase):
    def setUp(self):
        self.provider = ProviderConfig.objects.create(name="ff-prov", provider_type="openai")
        self.model_cfg = ModelConfig.objects.create(provider=self.provider, model_name="gpt-ff")
        self.agent = AgentProfile.objects.create(
            name="ff-agent", role="r", model_config=self.model_cfg
        )
        self.workspace = create_workspace(name="ff-ws", description="test")

    def test_create_goal_blocked_when_flag_disabled(self):
        with override_settings(AI_HUB_GAME_GOALS_ENABLED=False):
            with self.assertRaises(ValidationError):
                create_goal(workspace=self.workspace, title="blocked", description="d")

    def test_claim_next_goal_blocked_when_flag_disabled(self):
        goal = create_goal(workspace=self.workspace, title="ff-goal", description="d")
        with override_settings(AI_HUB_GAME_SCHEDULER_ENABLED=False):
            with self.assertRaises(ValidationError):
                claim_next_goal(self.workspace.pk)
        goal.delete()

    def test_execute_game_action_blocked_when_flag_disabled(self):
        goal = create_goal(workspace=self.workspace, title="ff-goal2", description="d")
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=goal,
            goal_text="test",
            status=ExecutionSession.Status.RUNNING,
        )
        with override_settings(AI_HUB_GAME_ACTION_DISPATCH_ENABLED=False):
            with self.assertRaises(ValidationError):
                execute_game_action(
                    session=session,
                    action_name="finish",
                    action_input={},
                )

    def test_record_memory_blocked_when_flag_disabled(self):
        with override_settings(AI_HUB_GAME_MEMORY_ENABLED=False):
            with self.assertRaises(ValidationError):
                record_memory(
                    scope_type=GameMemoryEntry.ScopeType.WORKSPACE,
                    workspace=self.workspace,
                    content="should be blocked",
                )

    def test_resume_goal_execution_blocked_when_flag_disabled(self):
        with override_settings(AI_HUB_GAME_RESUME_ENABLED=False):
            with self.assertRaises(ValidationError):
                resume_goal_execution(session_id=0)

    def test_run_delegated_agent_blocked_when_flag_disabled(self):
        goal = create_goal(workspace=self.workspace, title="ff-goal3", description="d")
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=goal,
            goal_text="test",
            status=ExecutionSession.Status.RUNNING,
        )
        action_def = GameActionDefinition.objects.create(
            name="ff-delegate",
            label="FF Delegate",
            action_type=GameActionDefinition.ActionType.SUB_AGENT,
        )
        action_run = GameActionRun.objects.create(
            session=session,
            action=action_def,
            action_name="ff-delegate",
            idempotency_key="ff-del-k",
            iteration=1,
            status=GameActionRun.Status.RUNNING,
        )
        with override_settings(AI_HUB_GAME_DELEGATION_ENABLED=False):
            with self.assertRaises(ValidationError):
                run_delegated_agent(
                    session=session,
                    action_run=action_run,
                    workspace=self.workspace,
                    goal=goal,
                    target_agent_name=self.agent.name,
                    task="blocked task",
                )

    def test_require_game_feature_passes_when_flag_enabled(self):
        with override_settings(AI_HUB_GAME_GOALS_ENABLED=True):
            try:
                require_game_feature("AI_HUB_GAME_GOALS_ENABLED")
            except ValidationError:
                self.fail("require_game_feature raised unexpectedly when flag is True")

    def test_require_game_feature_rejects_unknown_flag(self):
        with self.assertRaises(ValueError):
            require_game_feature("AI_HUB_GAME_GOALS_ENABLE")  # typo: missing 'D'

    def test_approve_action_run_blocked_when_dispatch_flag_disabled(self):
        User = get_user_model()
        approver = User.objects.create_user(username="ff-approver", password="test", is_staff=True)
        approver.user_permissions.add(
            Permission.objects.get(codename="approve_game_action")
        )
        approver = User.objects.get(pk=approver.pk)  # refresh cached perms

        goal = create_goal(workspace=self.workspace, title="ff-appr", description="d")
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=goal,
            goal_text="test",
            status=ExecutionSession.Status.WAITING_ASYNC,
        )
        action_def = GameActionDefinition.objects.create(
            name="ff-appr-action",
            label="FF Appr",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            requires_approval=True,
        )
        action_run = GameActionRun.objects.create(
            session=session,
            action=action_def,
            action_name="ff-appr-action",
            idempotency_key="ff-appr-k",
            iteration=1,
            status=GameActionRun.Status.WAITING_APPROVAL,
        )
        approval = GameActionApprovalRequest.objects.create(
            action_run=action_run,
            goal=goal,
            status=GameActionApprovalRequest.Status.PENDING,
        )

        with override_settings(AI_HUB_GAME_ACTION_DISPATCH_ENABLED=False):
            with self.assertRaises(ValidationError):
                approve_action_run(action_run_id=action_run.pk, reviewed_by=approver)

        approval.refresh_from_db()
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.PENDING)


class GamePostPhase12StabilizationTests(TestCase):
    def setUp(self):
        self.provider = ProviderConfig.objects.create(
            name="post12-provider", provider_type="training"
        )
        self.model_cfg = ModelConfig.objects.create(
            provider=self.provider, model_name="post12-model"
        )
        self.parent_agent = AgentProfile.objects.create(
            name="post12-parent", role="parent", model_config=self.model_cfg
        )
        self.target_agent = AgentProfile.objects.create(
            name="post12-target", role="target", model_config=self.model_cfg
        )
        self.workspace = create_workspace(name="post12-workspace")
        self.goal = create_goal(
            workspace=self.workspace, title="Post-12 goal", description="stabilize"
        )
        self.parent_session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.parent_agent,
            goal=self.goal,
            goal_text="stabilize",
            status=ExecutionSession.Status.RUNNING,
        )
        self.delegate_action = GameActionDefinition.objects.create(
            name="delegate_to_agent",
            label="Delegate",
            action_type=GameActionDefinition.ActionType.SUB_AGENT,
            risk_level="medium",
        )

    def make_parent_action(self, suffix="one"):
        return GameActionRun.objects.create(
            session=self.parent_session,
            action=self.delegate_action,
            idempotency_key=f"post12-{suffix}",
            action_name=self.delegate_action.name,
            iteration=1,
            status=GameActionRun.Status.RUNNING,
        )

    def test_self_delegation_is_denied_by_default(self):
        with self.assertRaisesMessage(ValidationError, "Self-delegation"):
            run_delegated_agent(
                session=self.parent_session,
                action_run=self.make_parent_action("self"),
                workspace=self.workspace,
                goal=self.goal,
                target_agent_name=self.parent_agent.name,
                task="Do not recurse into yourself.",
            )

    def test_pipeline_effective_agent_cannot_self_delegate_by_default(self):
        pipeline = PipelineDefinition.objects.create(name="Post-12 parent pipeline")
        PipelineStep.objects.create(
            pipeline=pipeline,
            agent=self.parent_agent,
            order=1,
        )
        self.parent_session.entry_agent = None
        self.parent_session.pipeline = pipeline
        self.parent_session.save(update_fields=["entry_agent", "pipeline"])

        with self.assertRaisesMessage(ValidationError, "Self-delegation"):
            run_delegated_agent(
                session=self.parent_session,
                action_run=self.make_parent_action("pipeline-self"),
                workspace=self.workspace,
                goal=self.goal,
                target_agent_name=self.parent_agent.name,
                task="Do not recurse through the pipeline identity.",
            )

    def test_self_delegation_requires_explicit_policy(self):
        self.workspace.default_policy = {"safety": {"allow_self_delegation": True}}
        self.workspace.full_clean()
        self.workspace.save(update_fields=["default_policy"])
        response = {
            "status": "ok",
            "content": _json.dumps(
                {"action": "finish", "message": "done", "complete": True, "final_answer": "ok"}
            ),
        }
        with patch("ai_hub.services.agent_runtime.completion_call", return_value=response):
            result = run_delegated_agent(
                session=self.parent_session,
                action_run=self.make_parent_action("self-allowed"),
                workspace=self.workspace,
                goal=self.goal,
                target_agent_name=self.parent_agent.name,
                task="Explicit self task.",
            )
        self.assertEqual(result["status"], GameDelegationRun.Status.SUCCESS)

    def test_delegated_session_reapplies_workspace_policy(self):
        parent_action = self.make_parent_action("policy")
        child = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.target_agent,
            goal_text="child",
            status=ExecutionSession.Status.RUNNING,
        )
        GameDelegationRun.objects.create(
            parent_action_run=parent_action,
            parent_goal=self.goal,
            delegated_session=child,
            target_agent=self.target_agent,
            task="child",
            status=GameDelegationRun.Status.RUNNING,
        )
        GameActionDefinition.objects.create(
            name="finish_goal",
            label="High-risk finish",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            risk_level="high",
        )

        with self.assertRaises(PolicyViolationError):
            execute_game_action(
                session=child,
                action_name="finish_goal",
                action_input={"final_answer": "must be blocked"},
            )
        self.assertEqual(child.game_action_runs.get().status, GameActionRun.Status.FAILED)

    def test_plan_model_rejects_reverse_dependency_and_cycle(self):
        plan = create_plan(goal=self.goal)
        first = add_plan_step(plan=plan, title="first", order=1)
        second = add_plan_step(plan=plan, title="second", order=2)
        first.depends_on_step = second
        with self.assertRaisesMessage(ValidationError, "earlier step"):
            first.full_clean()

    def test_plan_revision_preserves_snapshot(self):
        from ai_hub.services.game_plans import revise_plan

        plan = create_plan(goal=self.goal, summary="v1")
        add_plan_step(plan=plan, title="first", order=1)
        revised = revise_plan(plan=plan, summary="v2")
        self.assertEqual(revised.version, 2)
        self.assertEqual(revised.revision_history[0]["summary"], "v1")
        self.assertEqual(revised.revision_history[0]["steps"][0]["title"], "first")

    def _make_pending_approval(self):
        action = GameActionDefinition.objects.create(
            name="post12-approval",
            label="Approval",
            action_type=GameActionDefinition.ActionType.INTERNAL,
            requires_approval=True,
        )
        self.parent_session.status = ExecutionSession.Status.WAITING_ASYNC
        self.parent_session.save(update_fields=["status"])
        run = GameActionRun.objects.create(
            session=self.parent_session,
            action=action,
            idempotency_key="post12-approval-run",
            action_name=action.name,
            iteration=2,
            status=GameActionRun.Status.WAITING_APPROVAL,
        )
        approval = GameActionApprovalRequest.objects.create(
            action_run=run,
            goal=self.goal,
            requested_payload={"password": "approval-secret"},
        )
        continuation = GameContinuationRequest.objects.create(
            session=self.parent_session,
            goal=self.goal,
            reason_code=GameContinuationRequest.ReasonCode.NEEDS_APPROVAL,
            payload={"action_run_id": run.pk, "token": "continuation-secret"},
        )
        return run, approval, continuation

    def test_nonapprover_cannot_edit_or_see_pending_approval_dashboard(self):
        _, approval, _ = self._make_pending_approval()
        User = get_user_model()
        user = User.objects.create_user(username="post12-limited", password="test", is_staff=True)
        for codename in (
            "view_gameworkspace", "view_executionsession",
            "view_gameactionapprovalrequest", "change_gameactionapprovalrequest",
        ):
            user.user_permissions.add(Permission.objects.get(codename=codename))
        client = Client()
        client.force_login(User.objects.get(pk=user.pk))

        dashboard = client.get(
            reverse("admin:ai_hub_gameworkspace_dashboard", args=[self.workspace.pk])
        )
        self.assertNotContains(dashboard, "post12-approval")
        response = client.post(
            reverse("admin:ai_hub_gameactionapprovalrequest_change", args=[approval.pk]),
            {"status": "approved", "requested_payload": '{"changed": true}', "_save": "Save"},
        )
        self.assertEqual(response.status_code, 302)
        approval.refresh_from_db()
        self.assertEqual(approval.status, GameActionApprovalRequest.Status.PENDING)
        self.assertEqual(approval.requested_payload["password"], "approval-secret")

    def test_review_endpoint_passes_note_to_service(self):
        _, approval, _ = self._make_pending_approval()
        reviewer = get_user_model().objects.create_superuser(
            username="post12-reviewer", password="test", email="review@example.com"
        )
        client = Client()
        client.force_login(reviewer)
        with patch("ai_hub.admin.approve_action_run") as approve_mock:
            response = client.post(
                reverse(
                    "admin:ai_hub_gameactionapprovalrequest_approve", args=[approval.pk]
                ),
                {"review_note": "Reviewed safely."},
            )
        self.assertEqual(response.status_code, 302)
        approve_mock.assert_called_once_with(
            action_run_id=approval.action_run_id,
            reviewed_by=reviewer,
            review_note="Reviewed safely.",
        )

    def test_step_and_session_admin_redact_sensitive_values(self):
        _, approval, continuation = self._make_pending_approval()
        step = ExecutionStepRun.objects.create(
            session=self.parent_session,
            order=1,
            agent=self.parent_agent,
            action_name="audit",
            status=ExecutionStepRun.Status.SUCCESS,
            request_payload={"password": "step-secret"},
        )
        self.parent_session.final_context = {"api_key": "session-secret"}
        self.parent_session.save(update_fields=["final_context"])
        user = get_user_model().objects.create_superuser(
            username="post12-admin", password="test", email="admin@example.com"
        )
        client = Client()
        client.force_login(user)
        step_response = client.get(
            reverse("admin:ai_hub_executionsteprun_change", args=[step.pk])
        )
        session_response = client.get(
            reverse("admin:ai_hub_executionsession_change", args=[self.parent_session.pk])
        )
        approval_response = client.get(
            reverse("admin:ai_hub_gameactionapprovalrequest_change", args=[approval.pk])
        )
        continuation_response = client.get(
            reverse("admin:ai_hub_gamecontinuationrequest_change", args=[continuation.pk])
        )
        self.assertNotContains(step_response, "step-secret")
        self.assertNotContains(session_response, "session-secret")
        self.assertNotContains(approval_response, "approval-secret")
        self.assertNotContains(continuation_response, "continuation-secret")
        self.assertContains(step_response, "***REDACTED***")
        self.assertContains(session_response, "***REDACTED***")
        self.assertContains(approval_response, "***REDACTED***")
        self.assertContains(continuation_response, "***REDACTED***")

    def test_unified_timeline_contains_all_event_types(self):
        from ai_hub.services.game_operational_ux import build_session_timeline

        run, _, _ = self._make_pending_approval()
        ExecutionStepRun.objects.create(
            session=self.parent_session,
            order=1,
            agent=self.parent_agent,
            action_name="step",
            status=ExecutionStepRun.Status.SUCCESS,
        )
        events = build_session_timeline(self.parent_session)
        self.assertEqual(
            {event["kind"] for event in events},
            {"step", "action", "continuation", "approval"},
        )
        self.assertIn(run.pk, [event["pk"] for event in events if event["kind"] == "action"])

    def test_pipeline_effective_agent_is_used_in_game_audit_views(self):
        from ai_hub.services.admin_control_center import build_game_graph_context
        from ai_hub.services.game_operational_ux import build_session_timeline

        pipeline = PipelineDefinition.objects.create(name="Post-12 audit pipeline")
        PipelineStep.objects.create(
            pipeline=pipeline,
            agent=self.parent_agent,
            order=1,
        )
        self.parent_session.entry_agent = None
        self.parent_session.pipeline = pipeline
        self.parent_session.save(update_fields=["entry_agent", "pipeline"])
        run, _, _ = self._make_pending_approval()

        timeline = build_session_timeline(self.parent_session)
        action_event = next(
            event for event in timeline
            if event["kind"] == "action" and event["pk"] == run.pk
        )
        graph = build_game_graph_context()
        agent_node = next(
            node for node in graph["game_graph"]["graph"]["nodes"]
            if node["id"] == f"agent:{self.parent_session.pk}"
        )

        self.assertEqual(action_event["agent"], self.parent_agent.name)
        self.assertEqual(agent_node["label"], self.parent_agent.name)

    def test_goal_and_memory_flags_close_execution_and_read_boundaries(self):
        with override_settings(AI_HUB_GAME_GOALS_ENABLED=False):
            with self.assertRaises(ValidationError):
                create_goal_execution_session(goal=self.goal, entry_agent=self.target_agent)
        record_memory(
            scope_type=GameMemoryEntry.ScopeType.WORKSPACE,
            workspace=self.workspace,
            content="flagged",
        )
        with override_settings(AI_HUB_GAME_MEMORY_ENABLED=False):
            with self.assertRaises(ValidationError):
                build_goal_memory_context(
                    workspace=self.workspace,
                    goal=self.goal,
                    session=self.parent_session,
                    max_chars=100,
                )

    def test_reusable_feature_flag_defaults_are_closed(self):
        from ai_hub.services.game_feature_flags import _FLAG_DEFAULTS

        self.assertTrue(_FLAG_DEFAULTS)
        self.assertTrue(all(value is False for value in _FLAG_DEFAULTS.values()))


class GameDelegationBudgetConcurrencyTests(TransactionTestCase):
    reset_sequences = True

    def test_delegation_budget_reservation_is_serialized(self):
        if connection.vendor != "postgresql":
            self.skipTest("Delegation budget locking requires PostgreSQL semantics.")

        from threading import Barrier, Lock

        provider = ProviderConfig.objects.create(name="delegation-lock-provider", provider_type="training")
        model_cfg = ModelConfig.objects.create(provider=provider, model_name="delegation-lock-model")
        parent = AgentProfile.objects.create(name="delegation-lock-parent", role="p", model_config=model_cfg)
        target = AgentProfile.objects.create(name="delegation-lock-target", role="t", model_config=model_cfg)
        workspace = GameWorkspace.objects.create(
            name="delegation-lock-workspace",
            default_policy={"budget": {"max_sub_agent_runs_per_goal": 1}},
        )
        goal = create_goal(workspace=workspace, title="lock goal", description="d")
        session = ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=parent,
            goal=goal,
            goal_text="d",
            status=ExecutionSession.Status.RUNNING,
        )
        definition = GameActionDefinition.objects.create(
            name="delegate_to_agent",
            label="Delegate",
            action_type=GameActionDefinition.ActionType.SUB_AGENT,
        )
        action_ids = []
        for index in range(2):
            action_ids.append(
                GameActionRun.objects.create(
                    session=session,
                    action=definition,
                    idempotency_key=f"delegation-lock-{index}",
                    action_name=definition.name,
                    iteration=index + 1,
                    status=GameActionRun.Status.RUNNING,
                ).pk
            )
        barrier = Barrier(2)
        result_lock = Lock()
        results = []
        response = {
            "status": "ok",
            "content": '{"action":"finish","message":"done","complete":true,"final_answer":"ok"}',
        }

        def delegate(action_id):
            close_old_connections()
            try:
                barrier.wait(timeout=10)
                run_delegated_agent(
                    session=ExecutionSession.objects.get(pk=session.pk),
                    action_run=GameActionRun.objects.get(pk=action_id),
                    workspace=GameWorkspace.objects.get(pk=workspace.pk),
                    goal=GameGoal.objects.get(pk=goal.pk),
                    target_agent_name=target.name,
                    task="one slot only",
                )
                result = "success"
            except Exception as exc:
                result = type(exc).__name__
            finally:
                close_old_connections()
            with result_lock:
                results.append(result)

        with patch("ai_hub.services.agent_runtime.completion_call", return_value=response):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(delegate, action_id) for action_id in action_ids]
                for future in futures:
                    future.result(timeout=30)

        self.assertEqual(GameDelegationRun.objects.count(), 1)
        self.assertEqual(results.count("success"), 1)


class GameVerticalSliceRegressionTests(TestCase):
    """Proves the complete safe, read-only GAME vertical slice."""

    @patch("ai_hub.services.agent_runtime.completion_call")
    def test_vertical_slice_complete_chain_and_no_external_write(self, mocked_call):
        workspace = create_workspace(
            name="Vertical slice workspace",
            default_runtime_config={"max_iterations": 4},
            default_policy={
                "budget": {
                    "max_iterations_per_session": 4,
                    "max_action_runs_per_session": 6,
                },
                "safety": {"allow_external_writes": False},
            },
        )
        provider = ProviderConfig.objects.create(
            name="vertical-slice-provider", provider_type="training"
        )
        model = ModelConfig.objects.create(
            provider=provider, model_name="vertical-slice-model"
        )
        agent = AgentProfile.objects.create(
            name="vertical-slice-agent",
            role="GAME documentation analyst",
            model_config=model,
        )
        collection = KnowledgeCollection.objects.create(name="Vertical slice docs")
        document = KnowledgeDocument.objects.create(
            collection=collection,
            title="GAME architecture",
            curated_text="The GAME architecture uses goals, scoped memory, and safe actions.",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        agent.knowledge_collections.add(collection)

        for name, action_type in (
            ("search_knowledge", GameActionDefinition.ActionType.CONTEXT_TOOL),
            ("read_document", GameActionDefinition.ActionType.CONTEXT_TOOL),
            ("record_memory", GameActionDefinition.ActionType.INTERNAL),
            ("finish_goal", GameActionDefinition.ActionType.INTERNAL),
        ):
            GameActionDefinition.objects.create(
                name=name,
                label=name.replace("_", " ").title(),
                action_type=action_type,
                risk_level="low",
            )

        selected_goal = create_goal(
            workspace=workspace,
            title="Review GAME docs",
            description="Identify missing architecture components.",
            base_priority=100,
        )
        next_goal = create_goal(
            workspace=workspace,
            title="Update README",
            description="Update the public guide after the review.",
            base_priority=50,
        )
        create_goal(
            workspace=workspace,
            title="Clean admin copy",
            description="Polish minor labels.",
            base_priority=10,
        )

        refresh_workspace_goal_priorities(workspace.pk)
        claimed = claim_next_goal(workspace.pk)
        self.assertEqual(claimed.pk, selected_goal.pk)
        session = create_goal_execution_session(goal=claimed, entry_agent=agent)

        mocked_call.side_effect = [
            {
                "status": "ok",
                "content": (
                    '{"action":"search_knowledge","action_input":{"query":"architecture"},'
                    '"message":"search","complete":false,"final_answer":""}'
                ),
            },
            {
                "status": "ok",
                "content": (
                    f'{{"action":"read_document","action_input":{{"document_id":{document.pk}}},'
                    '"message":"read","complete":false,"final_answer":""}'
                ),
            },
            {
                "status": "ok",
                "content": (
                    '{"action":"record_memory","action_input":{"scope_type":"goal",'
                    '"content":"GAME uses scoped memory and safe actions."},'
                    '"message":"remember","complete":false,"final_answer":""}'
                ),
            },
            {
                "status": "ok",
                "content": (
                    '{"action":"finish_goal","action_input":{"final_answer":"Review complete."},'
                    '"message":"finish","complete":false,"final_answer":""}'
                ),
            },
        ]

        run_execution_session(session.pk, use_action_dispatcher=True)
        session.refresh_from_db()
        selected_goal.refresh_from_db()

        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)
        self.assertEqual(selected_goal.status, GameGoal.Status.COMPLETED)
        self.assertEqual(session.step_runs.count(), 4)
        self.assertEqual(
            list(
                session.game_action_runs.order_by("iteration").values_list(
                    "action_name", flat=True
                )
            ),
            ["search_knowledge", "read_document", "record_memory", "finish_goal"],
        )
        self.assertIn("action_output", session.step_runs.get(order=1).observation_payload)
        self.assertTrue(
            GameMemoryEntry.objects.filter(
                workspace=workspace,
                goal=selected_goal,
                scope_type=GameMemoryEntry.ScopeType.GOAL,
                content="GAME uses scoped memory and safe actions.",
            ).exists()
        )
        self.assertFalse(
            session.game_action_runs.filter(
                action__action_type__in=[
                    GameActionDefinition.ActionType.TOOL,
                    GameActionDefinition.ActionType.HTTP,
                    GameActionDefinition.ActionType.PYTHON_CALLABLE,
                ]
            ).exists()
        )
        self.assertEqual(get_next_eligible_goal(workspace.pk).pk, next_goal.pk)


# ============================================================
# Orphaned RUNNING goal cleanup
# ============================================================

from datetime import timedelta as _timedelta  # noqa: E402

from ai_hub.services.game_goals import (  # noqa: E402
    cancel_orphaned_running_goals,
    find_orphaned_running_goals,
)


class GameOrphanedGoalCleanupTests(TestCase):
    def setUp(self):
        self.provider = ProviderConfig.objects.create(name="orphan-prov", provider_type="training")
        self.model_cfg = ModelConfig.objects.create(provider=self.provider, model_name="training")
        self.agent = AgentProfile.objects.create(
            name="orphan-agent", role="r", model_config=self.model_cfg
        )
        self.workspace = create_workspace(name="orphan-ws", description="test")

    def _running_goal(self, title):
        goal = create_goal(workspace=self.workspace, title=title, description="d")
        return transition_goal_status(goal, GameGoal.Status.RUNNING, reason="test")

    def test_orphaned_running_goal_is_cancelled(self):
        goal = self._running_goal("orphaned")

        cancelled = cancel_orphaned_running_goals()

        self.assertIn(goal.pk, [g.pk for g in cancelled])
        goal.refresh_from_db()
        self.assertEqual(goal.status, GameGoal.Status.CANCELLED)

    def test_running_goal_with_active_session_is_left_untouched(self):
        goal = self._running_goal("has-active-session")
        ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=goal,
            goal_text="test",
            status=ExecutionSession.Status.WAITING_ASYNC,
        )

        cancelled = cancel_orphaned_running_goals()

        self.assertNotIn(goal.pk, [g.pk for g in cancelled])
        goal.refresh_from_db()
        self.assertEqual(goal.status, GameGoal.Status.RUNNING)

    def test_cleanup_revalidates_candidate_that_gains_active_session(self):
        goal = self._running_goal("candidate-gains-session")
        real_find = find_orphaned_running_goals

        def find_then_create_session(*args, **kwargs):
            candidates = real_find(*args, **kwargs)
            create_goal_execution_session(goal=goal, entry_agent=self.agent)
            return candidates

        with patch(
            "ai_hub.services.game_goals.find_orphaned_running_goals",
            side_effect=find_then_create_session,
        ):
            cancelled = cancel_orphaned_running_goals()

        goal.refresh_from_db()
        self.assertEqual(cancelled, [])
        self.assertEqual(goal.status, GameGoal.Status.RUNNING)
        self.assertTrue(
            goal.execution_sessions.filter(
                status__in=(
                    ExecutionSession.Status.PENDING,
                    ExecutionSession.Status.RUNNING,
                    ExecutionSession.Status.WAITING_ASYNC,
                )
            ).exists()
        )

    def test_running_goal_with_terminal_session_is_an_orphan(self):
        goal = self._running_goal("terminal-session")
        ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=goal,
            goal_text="test",
            status=ExecutionSession.Status.FAILED,
        )

        cancel_orphaned_running_goals()

        goal.refresh_from_db()
        self.assertEqual(goal.status, GameGoal.Status.CANCELLED)

    def test_older_than_filters_recent_goals(self):
        goal = self._running_goal("recent")

        # Recent goal is not swept when older_than is set.
        self.assertEqual(find_orphaned_running_goals(older_than=_timedelta(hours=1)), [])

        # Force updated_at into the past (update() bypasses auto_now).
        old = timezone.now() - _timedelta(hours=2)
        GameGoal.objects.filter(pk=goal.pk).update(updated_at=old)
        candidates = find_orphaned_running_goals(older_than=_timedelta(hours=1))
        self.assertIn(goal.pk, [g.pk for g in candidates])

    def test_cleanup_scoped_to_workspace(self):
        goal = self._running_goal("in-scope")
        other_ws = create_workspace(name="orphan-other-ws", description="d")
        other_goal = create_goal(workspace=other_ws, title="out-of-scope", description="d")
        other_goal = transition_goal_status(other_goal, GameGoal.Status.RUNNING, reason="test")

        cancel_orphaned_running_goals(workspace=self.workspace)

        goal.refresh_from_db()
        other_goal.refresh_from_db()
        self.assertEqual(goal.status, GameGoal.Status.CANCELLED)
        self.assertEqual(other_goal.status, GameGoal.Status.RUNNING)

    def test_management_command_dry_run_does_not_mutate(self):
        from io import StringIO
        from django.core.management import call_command

        goal = self._running_goal("dry-run-goal")

        out = StringIO()
        call_command("cleanup_orphaned_goals", "--dry-run", stdout=out)

        self.assertIn("dry-run", out.getvalue())
        goal.refresh_from_db()
        self.assertEqual(goal.status, GameGoal.Status.RUNNING)

    def test_management_command_cancels_orphans(self):
        from io import StringIO
        from django.core.management import call_command

        goal = self._running_goal("cmd-goal")

        out = StringIO()
        call_command("cleanup_orphaned_goals", stdout=out)

        self.assertIn("Cancelled 1", out.getvalue())
        goal.refresh_from_db()
        self.assertEqual(goal.status, GameGoal.Status.CANCELLED)


class GameOrphanedGoalCleanupConcurrencyTests(TransactionTestCase):
    reset_sequences = True

    def test_session_created_after_candidate_discovery_prevents_cancellation(self):
        if connection.vendor != "postgresql":
            self.skipTest(
                "Orphan cleanup serialization requires PostgreSQL row-lock semantics."
            )

        from threading import Event

        provider = ProviderConfig.objects.create(
            name="orphan-race-provider",
            provider_type="training",
        )
        model_cfg = ModelConfig.objects.create(
            provider=provider,
            model_name="training",
        )
        agent = AgentProfile.objects.create(
            name="orphan-race-agent",
            role="orphan race",
            model_config=model_cfg,
        )
        workspace = create_workspace(name="orphan-race-workspace")
        goal = create_goal(
            workspace=workspace,
            title="Orphan serialization",
            description="Never cancel with an active session.",
        )
        goal = transition_goal_status(goal, GameGoal.Status.RUNNING, reason="race setup")
        candidate_selected = Event()
        allow_cleanup = Event()
        real_find = find_orphaned_running_goals

        def delayed_find(*args, **kwargs):
            candidates = real_find(*args, **kwargs)
            candidate_selected.set()
            if not allow_cleanup.wait(timeout=10):
                raise AssertionError("Timed out waiting for session creation.")
            return candidates

        def cleanup():
            close_old_connections()
            try:
                with patch(
                    "ai_hub.services.game_goals.find_orphaned_running_goals",
                    side_effect=delayed_find,
                ):
                    return cancel_orphaned_running_goals()
            finally:
                close_old_connections()

        with ThreadPoolExecutor(max_workers=1) as executor:
            cleanup_future = executor.submit(cleanup)
            self.assertTrue(
                candidate_selected.wait(timeout=10),
                "Cleanup did not discover the candidate.",
            )
            try:
                session = create_goal_execution_session(
                    goal=goal,
                    entry_agent=agent,
                )
            finally:
                allow_cleanup.set()
            cancelled = cleanup_future.result(timeout=20)

        goal.refresh_from_db()
        session.refresh_from_db()
        self.assertEqual(cancelled, [])
        self.assertEqual(goal.status, GameGoal.Status.RUNNING)
        self.assertIn(
            session.status,
            (
                ExecutionSession.Status.PENDING,
                ExecutionSession.Status.RUNNING,
                ExecutionSession.Status.WAITING_ASYNC,
            ),
        )
        self.assertFalse(
            goal.status == GameGoal.Status.CANCELLED
            and goal.execution_sessions.filter(
                status__in=(
                    ExecutionSession.Status.PENDING,
                    ExecutionSession.Status.RUNNING,
                    ExecutionSession.Status.WAITING_ASYNC,
                )
            ).exists()
        )


class HubHealthEvaluatorTests(TestCase):
    """P1.4: the reusable runtime-health evaluator (services/health.py)."""

    def setUp(self):
        self.provider = ProviderConfig.objects.create(
            name="Health Ollama",
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://localhost:11434",
        )
        self.model = ModelConfig.objects.create(provider=self.provider, model_name="ollama/qwen3:8b")
        self.agent = AgentProfile.objects.create(
            name="health-agent",
            role="r",
            model_config=self.model,
            system_prompt="do the thing",
            input_contract={"required": ["x"]},
            output_contract={"required": ["y"]},
        )

    def test_provider_healthy(self):
        from ai_hub.services.health import evaluate_provider, STATUS_OK
        result = evaluate_provider(self.provider)
        self.assertEqual(result.status, STATUS_OK)
        self.assertTrue(result.ok)
        self.assertTrue(all(c.ok for c in result.checks))

    def test_provider_inactive(self):
        from ai_hub.services.health import evaluate_provider, STATUS_INACTIVE
        self.provider.is_active = False
        self.provider.save(update_fields=["is_active"])
        self.assertEqual(evaluate_provider(self.provider).status, STATUS_INACTIVE)

    def test_provider_ollama_without_base_url_warns(self):
        from ai_hub.services.health import evaluate_provider, STATUS_WARNING
        self.provider.base_url = ""
        self.provider.save(update_fields=["base_url"])
        result = evaluate_provider(self.provider)
        self.assertEqual(result.status, STATUS_WARNING)
        self.assertIn("Base URL set", [c.label for c in result.failing])

    def test_model_active_provider_off_is_warning(self):
        from ai_hub.services.health import evaluate_model, STATUS_WARNING
        self.provider.is_active = False
        self.provider.save(update_fields=["is_active"])
        result = evaluate_model(self.model)
        self.assertEqual(result.status, STATUS_WARNING)

    def test_agent_healthy_then_missing_contracts_warns(self):
        from ai_hub.services.health import evaluate_agent, STATUS_OK, STATUS_WARNING
        self.assertEqual(evaluate_agent(self.agent).status, STATUS_OK)
        self.agent.input_contract = {}
        self.agent.output_contract = {}
        self.agent.save(update_fields=["input_contract", "output_contract"])
        result = evaluate_agent(self.agent)
        self.assertEqual(result.status, STATUS_WARNING)
        self.assertIn("Contracts set", [c.label for c in result.failing])
