"""S-35m: GAME `record_memory` provenance metadata is runtime-owned (issue #32, parent #31).

`_handle_record_memory()` used `setdefault()` for `source`, `source_id` and
`model_derived`, so a model that supplied those keys overwrote the runtime's
provenance, and `build_goal_memory_context()` then showed the forged values back
to the model - including in other goals' memory for workspace-scoped entries.

The contract pinned here:

  * `source`, `source_id`, `model_derived` are written by the runtime and cannot
    be influenced by model input;
  * the complete model-supplied metadata is kept, verbatim, under the single
    `model_supplied` key; a model key literally named `model_supplied` is not a
    collision, it simply stays inside that namespace;
  * missing or empty metadata keeps the previous stored shape (no namespace key);
  * non-object metadata is a `ValidationError` before any memory is written;
  * scope, content, importance, persistence and retrieval are unchanged, and the
    service-level `record_memory()` (operators, Admin) is untouched.

No migration, no flag, and nothing here classifies or labels content (S-35a).
"""
from django.core.exceptions import ValidationError
from django.test import TestCase, override_settings

from ai_hub.models import (
    AgentProfile,
    ExecutionSession,
    GameActionDefinition,
    GameActionRun,
    GameMemoryEntry,
    ModelConfig,
    ProviderConfig,
)
from ai_hub.services.game_action_dispatcher import execute_game_action
from ai_hub.services.game_goals import create_goal
from ai_hub.services.game_memory import build_goal_memory_context, record_memory
from ai_hub.services.game_workspaces import create_workspace
from ai_hub.test_application_scope_helpers import test_scope

RUNTIME_KEYS = {"source", "source_id", "model_derived"}


@override_settings(
    AI_HUB_GAME_ACTION_DISPATCH_ENABLED=True,
    AI_HUB_GAME_MEMORY_ENABLED=True,
    AI_HUB_GAME_GOALS_ENABLED=True,
)
class RecordMemoryMetadataIntegrityTests(TestCase):
    def setUp(self):
        self.workspace = create_workspace(application_scope=test_scope(), name="S35m workspace")
        provider = ProviderConfig.objects.create(name="s35m-provider", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="s35m-model")
        self.agent = AgentProfile.objects.create(
            application_scope=test_scope(), name="s35m-agent", role="r", model_config=model
        )
        self.goal = create_goal(workspace=self.workspace, title="S35m goal", description="d")
        self.other_goal = create_goal(workspace=self.workspace, title="Other goal", description="d")
        self.session = ExecutionSession.objects.create(
            entry_agent=self.agent,
            runtime_kind=ExecutionSession.RuntimeKind.GAME,
            runtime_mode=ExecutionSession.RuntimeMode.SYNC,
            status=ExecutionSession.Status.RUNNING,
            goal=self.goal,
            goal_text="S35m",
        )
        GameActionDefinition.objects.create(
            name="record_memory",
            label="Record memory",
            action_type=GameActionDefinition.ActionType.INTERNAL,
        )

    def write(self, **action_input):
        action_input.setdefault("content", "A model-proposed fact.")
        return execute_game_action(
            session=self.session, action_name="record_memory", action_input=action_input
        )

    def stored(self, action_run):
        return GameMemoryEntry.objects.get(pk=action_run.output_payload["memory_entry_id"])

    def snapshot(self):
        return (
            GameMemoryEntry.objects.count(),
            GameActionRun.objects.filter(status=GameActionRun.Status.SUCCESS).count(),
        )


class SpoofedRuntimeFieldsTests(RecordMemoryMetadataIntegrityTests):
    def test_all_three_spoofed_fields_are_overwritten_by_the_runtime(self):
        run = self.write(
            metadata={"source": "operator", "source_id": 424242, "model_derived": False}
        )

        metadata = self.stored(run).metadata
        self.assertEqual(metadata["source"], "action_run")
        self.assertEqual(metadata["source_id"], run.pk)
        self.assertIs(metadata["model_derived"], True)

    def test_each_spoofed_field_is_overwritten_independently(self):
        for forged in (
            {"source": "operator"},
            {"source_id": 1},
            {"model_derived": False},
            {"model_derived": "false"},
            {"model_derived": None},
        ):
            with self.subTest(forged=forged):
                run = self.write(content=f"fact {forged}", metadata=forged)
                metadata = self.stored(run).metadata
                self.assertEqual(metadata["source"], "action_run")
                self.assertEqual(metadata["source_id"], run.pk)
                self.assertIs(metadata["model_derived"], True)

    def test_the_forged_values_are_preserved_for_audit_under_the_namespace(self):
        forged = {"source": "operator", "source_id": 424242, "model_derived": False}

        run = self.write(metadata=forged)

        self.assertEqual(self.stored(run).metadata["model_supplied"], forged)

    def test_source_id_is_the_action_run_that_wrote_the_entry(self):
        first = self.write(content="first", metadata={"source_id": 999})
        second = self.write(content="second", metadata={"source_id": 999})

        self.assertNotEqual(first.pk, second.pk)
        self.assertEqual(self.stored(first).metadata["source_id"], first.pk)
        self.assertEqual(self.stored(second).metadata["source_id"], second.pk)

    def test_the_forged_value_does_not_survive_read_back_to_the_model(self):
        self.write(
            scope_type="workspace",
            content="workspace fact",
            importance_score="1.0",
            metadata={"model_derived": False, "source": "operator"},
        )

        context = build_goal_memory_context(workspace=self.workspace, goal=self.other_goal)

        entry = context["entries"][0]
        self.assertEqual(entry["content"], "workspace fact")
        self.assertIs(entry["metadata"]["model_derived"], True)
        self.assertEqual(entry["metadata"]["source"], "action_run")
        self.assertEqual(
            entry["metadata"]["model_supplied"],
            {"model_derived": False, "source": "operator"},
        )


class CustomMetadataTests(RecordMemoryMetadataIntegrityTests):
    def test_valid_custom_metadata_is_kept_verbatim_under_the_namespace(self):
        custom = {"topic": "refunds", "tags": ["a", "b"], "nested": {"k": [1, 2, {"x": None}]}}

        run = self.write(metadata=custom)

        metadata = self.stored(run).metadata
        self.assertEqual(metadata["model_supplied"], custom)
        self.assertEqual(
            set(metadata), RUNTIME_KEYS | {"model_supplied"}
        )  # custom keys are not mixed into the runtime level

    def test_custom_metadata_is_not_accepted_at_the_top_level(self):
        run = self.write(metadata={"topic": "refunds"})

        self.assertNotIn("topic", self.stored(run).metadata)

    def test_missing_metadata_keeps_the_previous_stored_shape(self):
        run = self.write()

        metadata = self.stored(run).metadata
        self.assertEqual(set(metadata), RUNTIME_KEYS)
        self.assertEqual(metadata["source"], "action_run")
        self.assertEqual(metadata["source_id"], run.pk)
        self.assertIs(metadata["model_derived"], True)

    def test_null_and_empty_object_metadata_are_treated_as_missing(self):
        for value in (None, {}):
            with self.subTest(value=value):
                run = self.write(content=f"fact {value}", metadata=value)
                self.assertEqual(set(self.stored(run).metadata), RUNTIME_KEYS)


class NamespaceCollisionTests(RecordMemoryMetadataIntegrityTests):
    def test_a_model_key_named_model_supplied_is_nested_not_dropped(self):
        supplied = {"model_supplied": {"model_derived": False}, "other": 1}

        run = self.write(metadata=supplied)

        metadata = self.stored(run).metadata
        self.assertEqual(metadata["model_supplied"], supplied)
        self.assertEqual(
            metadata["model_supplied"]["model_supplied"], {"model_derived": False}
        )
        self.assertIs(metadata["model_derived"], True)

    def test_a_scalar_model_supplied_key_is_preserved_too(self):
        supplied = {"model_supplied": "x", "model_derived": False}

        run = self.write(metadata=supplied)

        self.assertEqual(self.stored(run).metadata["model_supplied"], supplied)

    def test_the_runtime_level_cannot_be_reached_through_the_namespace(self):
        run = self.write(metadata={"model_supplied": {"source": "operator"}})

        metadata = self.stored(run).metadata
        self.assertEqual(metadata["source"], "action_run")


class MalformedMetadataTests(RecordMemoryMetadataIntegrityTests):
    def test_non_object_metadata_is_rejected_without_writing_anything(self):
        for bad in ("text", "", 0, 7, 1.5, True, False, ["a"], []):
            with self.subTest(bad=bad):
                before = self.snapshot()

                with self.assertRaisesMessage(ValidationError, "'metadata' must be a JSON object"):
                    self.write(content="never stored", metadata=bad)

                self.assertEqual(self.snapshot(), before)
                self.assertFalse(GameMemoryEntry.objects.filter(content="never stored").exists())

    def test_a_rejected_write_is_recorded_as_a_failed_action_run(self):
        with self.assertRaises(ValidationError):
            self.write(content="never stored", metadata=["x"])

        failed = GameActionRun.objects.get(status=GameActionRun.Status.FAILED)
        self.assertIn("metadata", failed.error_detail)

    def test_content_validation_is_unchanged(self):
        before = self.snapshot()

        with self.assertRaisesMessage(ValidationError, "requires non-empty 'content'"):
            self.write(content="   ", metadata={"model_derived": False})

        self.assertEqual(self.snapshot(), before)

    def test_unknown_scope_is_still_rejected_without_writing(self):
        before = self.snapshot()

        with self.assertRaisesMessage(ValidationError, "Unknown GAME memory scope"):
            self.write(scope_type="galaxy", metadata={"a": 1})

        self.assertEqual(self.snapshot(), before)


class UnchangedBehaviourTests(RecordMemoryMetadataIntegrityTests):
    def test_scope_content_and_importance_are_unchanged(self):
        run = self.write(
            scope_type="workspace",
            content="  workspace fact  ",
            importance_score="0.90",
            metadata={"topic": "t"},
        )

        entry = self.stored(run)
        self.assertEqual(entry.scope_type, GameMemoryEntry.ScopeType.WORKSPACE)
        self.assertEqual(entry.content, "workspace fact")
        self.assertEqual(str(entry.importance_score), "0.90")
        self.assertIsNone(entry.goal_id)
        self.assertIsNone(entry.session_id)
        self.assertEqual(run.output_payload["content_preview"], "workspace fact")

    def test_each_scope_still_maps_to_its_links(self):
        expected = {
            GameMemoryEntry.ScopeType.WORKSPACE: (None, None),
            GameMemoryEntry.ScopeType.GOAL: (self.goal.pk, None),
            GameMemoryEntry.ScopeType.SESSION: (None, self.session.pk),
            GameMemoryEntry.ScopeType.ACTION_RESULT: (self.goal.pk, self.session.pk),
        }
        for scope_type, links in expected.items():
            with self.subTest(scope_type=scope_type):
                run = self.write(scope_type=scope_type, content=f"{scope_type} fact")
                entry = self.stored(run)
                self.assertEqual((entry.goal_id, entry.session_id), links)

    def test_default_scope_importance_and_goal_read_back_are_unchanged(self):
        self.write(content="goal fact")

        context = build_goal_memory_context(workspace=self.workspace, goal=self.goal)

        self.assertEqual([e["content"] for e in context["entries"]], ["goal fact"])
        self.assertEqual(context["entries"][0]["importance"], "0.50")

    def test_service_level_record_memory_is_untouched(self):
        entry = record_memory(
            scope_type=GameMemoryEntry.ScopeType.GOAL,
            workspace=self.workspace,
            goal=self.goal,
            content="operator fact",
            metadata={"source": "operator", "model_derived": False, "custom": 1},
        )

        self.assertEqual(
            entry.metadata, {"source": "operator", "model_derived": False, "custom": 1}
        )

    def test_manually_created_admin_style_entries_are_untouched_and_readable(self):
        legacy = GameMemoryEntry.objects.create(
            workspace=self.workspace,
            goal=self.goal,
            scope_type=GameMemoryEntry.ScopeType.GOAL,
            content="manual entry",
            metadata={"category": "manual"},
        )
        legacy_forged = GameMemoryEntry.objects.create(
            workspace=self.workspace,
            goal=self.goal,
            scope_type=GameMemoryEntry.ScopeType.GOAL,
            content="historical entry",
            metadata={"source": "operator", "model_derived": False},
        )

        context = build_goal_memory_context(workspace=self.workspace, goal=self.goal)

        by_content = {e["content"]: e["metadata"] for e in context["entries"]}
        self.assertEqual(by_content["manual entry"], {"category": "manual"})
        # Historical rows are not repaired: their provenance cannot be established.
        self.assertEqual(
            by_content["historical entry"], {"source": "operator", "model_derived": False}
        )
        legacy.refresh_from_db()
        legacy_forged.refresh_from_db()
        self.assertEqual(legacy.metadata, {"category": "manual"})
        self.assertEqual(legacy_forged.metadata, {"source": "operator", "model_derived": False})
