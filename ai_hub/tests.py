from unittest.mock import patch
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import IntegrityError, close_old_connections, connection, transaction
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    ExecutionStepRun,
    GameActionDefinition,
    GameActionRun,
    GameGoal,
    GameGoalDependency,
    GameMemoryEntry,
    GameWorkspace,
    KnowledgeCollection,
    KnowledgeDocument,
    ModelConfig,
    PipelineDefinition,
    PipelineStep,
    ProviderConfig,
    ToolDefinition,
)
from ai_hub.services.litellm_client import completion_call
from ai_hub.services.admin_control_center import build_control_center_context
from ai_hub.services.execution_sessions import create_execution_session
from ai_hub.services.agent_runtime import build_agent_knowledge_context
from ai_hub.services.contracts import validate_payload
from ai_hub.services.execution_runner import run_execution_session
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
    execute_tools,
    get_game_tool_category,
)
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
            runtime_config={"max_iterations": 1},
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
            runtime_config={"max_iterations": 1},
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
        approval_req = GameActionApprovalRequest.objects.create(
            action_run=action_run,
            goal=self.goal,
            requested_payload={"final_answer": "pending test"},
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
        self.assertLessEqual(action_run_ids.index(run1.pk), action_run_ids.index(run2.pk))

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
        ExecutionSession.objects.create(
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            entry_agent=self.agent,
            goal=self.goal,
            goal_text="test",
            status=ExecutionSession.Status.WAITING_ASYNC,
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
