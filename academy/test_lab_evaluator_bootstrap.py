"""Issue #26: `seed_lab_exercises --create-evaluator` must be reliable and scope-safe.

The command used to create the 'Lab Exercise Evaluator' AgentProfile without an
`application_scope` (a required, non-null FK since S-14), swallow the resulting
IntegrityError in a bare `except Exception`, print an error and exit 0, after
having already committed the lab exercises. Operators saw "Done" plus a
failure line, and AI scoring silently stayed off.

The contract pinned here:

  * Ownership is resolved with Academy's `resolve_seed_scope()` - never inferred
    from how many scopes are active - and BEFORE any write, so a refusal leaves
    the database exactly as it was.
  * The evaluator is created in Academy's explicitly configured scope.
  * An evaluator that belongs to another scope is never reused, modified or
    reassigned: the command refuses.
  * Every evaluator failure is a `CommandError` (non-zero exit), and the lab
    exercises are not left behind as a misleading partial success.
  * Without `--create-evaluator`, nothing about lab seeding changes.

No external AI provider is contacted: model configs are plain database rows and
the one test that runs the evaluation service stubs the execution runner.
"""
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings

from academy.management.commands.seed_lab_exercises import (
    EVALUATOR_SYSTEM_PROMPT,
    EXERCISES,
)
from academy.models import LabExercise, TutorialModule
from academy.services.lab_evaluator import evaluate_lab_answer
from ai_hub.models import AgentProfile, ApplicationScope, ModelConfig, ProviderConfig

EVALUATOR = "Lab Exercise Evaluator"
FIRST_MODULE_ORDER = min(EXERCISES)


def seed(*args):
    out = StringIO()
    call_command("seed_lab_exercises", *args, stdout=out, stderr=StringIO())
    return out.getvalue()


def make_scope(slug="academy", *, active=True):
    return ApplicationScope.objects.create(
        slug=slug, name=f"Scope {slug}", is_active=active
    )


def make_model(
    name="ollama/test-model",
    *,
    provider_type=ProviderConfig.ProviderType.OLLAMA,
    model_active=True,
    provider_active=True,
):
    provider, _ = ProviderConfig.objects.get_or_create(
        name=f"provider-{provider_type}-{provider_active}",
        defaults={"provider_type": provider_type, "is_active": provider_active},
    )
    return ModelConfig.objects.create(
        provider=provider, model_name=name, is_active=model_active
    )


def snapshot():
    """Everything the command could write, comparable before and after a refusal."""
    return {
        "exercises": sorted(LabExercise.objects.values_list("slug", "title")),
        "agents": sorted(
            AgentProfile.objects.values_list(
                "name", "application_scope_id", "model_config_id",
                "role", "system_prompt", "is_active",
            )
        ),
        "scopes": sorted(ApplicationScope.objects.values_list("slug", "is_active")),
        "models": ModelConfig.objects.count(),
    }


class EvaluatorTestCase(TestCase):
    def setUp(self):
        self.module = TutorialModule.objects.create(
            title="Orientation", slug="orientation", order=FIRST_MODULE_ORDER
        )

    def assertRefused(self, *args, contains):
        before = snapshot()
        with self.assertRaises(CommandError) as ctx:
            seed(*args)
        self.assertIn(contains, str(ctx.exception))
        self.assertEqual(snapshot(), before, "a refusal must not write anything")
        return str(ctx.exception)

    def evaluator(self):
        return AgentProfile.objects.get(name=EVALUATOR)


class FreshSetupTests(EvaluatorTestCase):
    def test_creates_the_evaluator_in_academys_scope(self):
        scope = make_scope()
        model = make_model()

        out = seed("--create-evaluator")

        agent = self.evaluator()
        self.assertEqual(agent.application_scope, scope)
        self.assertEqual(agent.model_config, model)
        self.assertTrue(agent.is_active)
        self.assertEqual(agent.system_prompt, EVALUATOR_SYSTEM_PROMPT)
        self.assertIn("Created", out)
        self.assertTrue(LabExercise.objects.filter(module=self.module).exists())

    def test_the_created_evaluator_passes_model_validation(self):
        make_scope()
        make_model()
        seed("--create-evaluator")

        self.evaluator().full_clean()

    def test_the_lab_evaluation_service_finds_and_runs_the_evaluator(self):
        make_scope()
        make_model()
        seed("--create-evaluator")
        exercise = LabExercise.objects.filter(module=self.module).first()
        answer = '{"score": "pass", "feedback": "Good.", "follow_up_question": ""}'

        def fake_runner(session_pk, **kwargs):
            from ai_hub.models import ExecutionSession

            session = ExecutionSession.objects.get(pk=session_pk)
            session.final_context = {"final_answer": answer}
            session.save()

        with patch(
            "ai_hub.services.execution_runner.run_execution_session",
            side_effect=fake_runner,
        ) as runner:
            result = evaluate_lab_answer(exercise=exercise, user_input="my answer")

        runner.assert_called_once()
        self.assertEqual(result["score"], "pass")
        self.assertIsNotNone(result["ai_session_pk"])

    def test_picks_an_active_non_training_model(self):
        make_scope()
        make_model("training/assistant", provider_type=ProviderConfig.ProviderType.TRAINING)
        make_model("inactive-model", model_active=False)
        usable = make_model("usable-model")

        seed("--create-evaluator")

        self.assertEqual(self.evaluator().model_config, usable)


class RepeatedExecutionTests(EvaluatorTestCase):
    def setUp(self):
        super().setUp()
        self.scope = make_scope()
        self.model = make_model()

    def test_running_twice_is_idempotent(self):
        seed("--create-evaluator")
        first = snapshot()

        out = seed("--create-evaluator")

        self.assertEqual(snapshot(), first)
        self.assertEqual(AgentProfile.objects.filter(name=EVALUATOR).count(), 1)
        self.assertIn("already up to date", out)

    def test_a_compatible_existing_evaluator_keeps_its_chosen_model(self):
        chosen = make_model("operator-chosen")
        agent = AgentProfile.objects.create(
            application_scope=self.scope, name=EVALUATOR, role="old",
            model_config=chosen, system_prompt="old prompt", is_active=True,
        )
        make_model("another-candidate")

        out = seed("--create-evaluator")

        agent.refresh_from_db()
        self.assertEqual(agent.model_config, chosen)
        self.assertEqual(agent.system_prompt, EVALUATOR_SYSTEM_PROMPT)
        self.assertEqual(agent.application_scope, self.scope)
        self.assertIn("Updated", out)

    def test_a_compatible_evaluator_on_an_unusable_model_is_moved_to_a_usable_one(self):
        dead = make_model("dead-model", model_active=False)
        agent = AgentProfile.objects.create(
            application_scope=self.scope, name=EVALUATOR, role="old",
            model_config=dead, is_active=False,
        )

        seed("--create-evaluator")

        agent.refresh_from_db()
        self.assertEqual(agent.model_config, self.model)
        self.assertTrue(agent.is_active)
        agent.full_clean()


class NoSuitableModelTests(EvaluatorTestCase):
    def setUp(self):
        super().setUp()
        make_scope()

    def test_no_model_at_all_is_refused_and_exercises_are_not_seeded(self):
        self.assertRefused("--create-evaluator", contains="No active non-training model")
        self.assertFalse(LabExercise.objects.exists())

    def test_only_a_training_model_is_refused(self):
        make_model("training/assistant", provider_type=ProviderConfig.ProviderType.TRAINING)

        self.assertRefused("--create-evaluator", contains="No active non-training model")

    def test_an_inactive_model_is_refused_and_named(self):
        make_model("inactive-model", model_active=False)

        message = self.assertRefused("--create-evaluator", contains="inactive")
        self.assertIn("inactive-model", message)

    def test_a_model_on_an_inactive_provider_is_refused(self):
        make_model("orphaned", provider_active=False)

        message = self.assertRefused("--create-evaluator", contains="inactive")
        self.assertIn("orphaned", message)


class ScopeOwnershipTests(EvaluatorTestCase):
    def setUp(self):
        super().setUp()
        make_model()

    def test_no_academy_scope_is_refused_and_none_is_created(self):
        self.assertRefused("--create-evaluator", contains="does not exist yet")
        self.assertFalse(ApplicationScope.objects.exists())

    def test_a_single_foreign_active_scope_is_not_adopted(self):
        make_scope("someone-elses-app")

        self.assertRefused("--create-evaluator", contains="will not adopt")
        self.assertFalse(AgentProfile.objects.exists())

    def test_two_active_scopes_do_not_confuse_ownership(self):
        academy = make_scope()
        make_scope("other-app")

        seed("--create-evaluator")

        self.assertEqual(self.evaluator().application_scope, academy)

    def test_an_inactive_academy_scope_is_refused_and_not_reactivated(self):
        make_scope(active=False)

        self.assertRefused("--create-evaluator", contains="inactive")
        self.assertFalse(ApplicationScope.objects.get(slug="academy").is_active)

    @override_settings(ACADEMY_APPLICATION_SCOPE_SLUG="declared-scope")
    def test_an_explicitly_declared_scope_is_used(self):
        declared = make_scope("declared-scope")
        make_scope()  # a decoy named like the default

        seed("--create-evaluator")

        self.assertEqual(self.evaluator().application_scope, declared)

    @override_settings(ACADEMY_APPLICATION_SCOPE_SLUG="missing-scope")
    def test_a_declared_scope_that_does_not_exist_is_never_created(self):
        self.assertRefused("--create-evaluator", contains="never created for you")
        self.assertFalse(ApplicationScope.objects.exists())


class ConflictingEvaluatorTests(EvaluatorTestCase):
    def setUp(self):
        super().setUp()
        self.academy = make_scope()
        self.foreign = make_scope("other-app")
        self.model = make_model()
        self.foreign_agent = AgentProfile.objects.create(
            application_scope=self.foreign, name=EVALUATOR, role="not ours",
            model_config=self.model, system_prompt="foreign prompt", is_active=True,
        )

    def test_an_evaluator_owned_by_another_scope_is_never_reused_or_changed(self):
        message = self.assertRefused("--create-evaluator", contains="other-app")

        self.assertIn("will not", message)
        self.foreign_agent.refresh_from_db()
        self.assertEqual(self.foreign_agent.application_scope, self.foreign)
        self.assertEqual(self.foreign_agent.system_prompt, "foreign prompt")
        self.assertEqual(self.foreign_agent.role, "not ours")

    def test_an_inactive_foreign_evaluator_is_not_reactivated(self):
        self.foreign_agent.is_active = False
        self.foreign_agent.save()

        self.assertRefused("--create-evaluator", contains="other-app")
        self.foreign_agent.refresh_from_db()
        self.assertFalse(self.foreign_agent.is_active)

    def test_exercises_are_not_left_behind_by_the_refusal(self):
        self.assertRefused("--create-evaluator", contains="other-app")

        self.assertFalse(LabExercise.objects.exists())


class ErrorPropagationTests(EvaluatorTestCase):
    def setUp(self):
        super().setUp()
        make_scope()
        make_model()

    def test_an_unexpected_failure_during_creation_exits_non_zero_and_rolls_back(self):
        before = snapshot()

        with patch(
            "academy.management.commands.seed_lab_exercises.AgentProfile.objects.create",
            side_effect=RuntimeError("database exploded"),
        ):
            with self.assertRaises(CommandError) as ctx:
                seed("--create-evaluator")

        self.assertIn("database exploded", str(ctx.exception))
        self.assertEqual(snapshot(), before)

    def test_a_failure_is_not_reported_as_success(self):
        out = StringIO()
        with self.assertRaises(CommandError):
            with patch(
                "academy.management.commands.seed_lab_exercises.AgentProfile.objects.create",
                side_effect=RuntimeError("boom"),
            ):
                call_command("seed_lab_exercises", "--create-evaluator", stdout=out)

        self.assertNotIn("Done.", out.getvalue())

    def test_recovery_after_fixing_the_cause_succeeds(self):
        ApplicationScope.objects.all().update(is_active=False)
        with self.assertRaises(CommandError):
            seed("--create-evaluator")
        ApplicationScope.objects.all().update(is_active=True)

        seed("--create-evaluator")

        self.assertEqual(self.evaluator().application_scope.slug, "academy")
        self.assertTrue(LabExercise.objects.exists())


class WithoutEvaluatorFlagTests(EvaluatorTestCase):
    def test_lab_seeding_needs_no_scope_and_no_model(self):
        out = seed()

        self.assertTrue(LabExercise.objects.filter(module=self.module).exists())
        self.assertFalse(AgentProfile.objects.exists())
        self.assertFalse(ApplicationScope.objects.exists())
        self.assertIn("--create-evaluator", out)

    def test_lab_seeding_ignores_a_conflicting_evaluator(self):
        foreign = make_scope("other-app")
        AgentProfile.objects.create(
            application_scope=foreign, name=EVALUATOR, role="not ours",
            model_config=make_model(),
        )

        seed()

        self.assertEqual(self.evaluator().application_scope, foreign)
        self.assertTrue(LabExercise.objects.exists())

    def test_force_still_overwrites_exercises(self):
        seed()
        LabExercise.objects.update(title="edited")

        seed("--force")

        self.assertFalse(LabExercise.objects.filter(title="edited").exists())
