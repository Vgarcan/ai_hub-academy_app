"""Review of PR #29: the runtime evaluator lookup must respect Academy's scope.

`evaluate_lab_answer()` used to select the evaluator by its globally unique
name only. If another application's active agent carried that name, Academy
would have sent student answers to it. The creation command already refuses a
foreign evaluator; these tests pin the same boundary at evaluation time.

The contract:

  * Academy's scope is resolved from its explicit configuration
    (`ACADEMY_APPLICATION_SCOPE_SLUG`, default `academy`) and must exist and be
    active. Nothing is created, and the number of active scopes decides nothing.
  * Only an ACTIVE evaluator OWNED by that scope is used.
  * Missing, inactive or foreign ownership fails closed to the existing
    `pending` result: no ExecutionSession is created and no runner is invoked.
  * The response contract (`score`, `feedback`, `follow_up_question`,
    `ai_session_pk`) is unchanged.

The execution runner is always mocked: no AI provider is contacted.
"""
from unittest.mock import patch

from django.test import TestCase, override_settings

from academy.models import LabExercise, TutorialModule
from academy.services.lab_evaluator import evaluate_lab_answer
from academy.services.scope_bootstrap import LAB_EVALUATOR_AGENT_NAME
from ai_hub.models import (
    AgentProfile,
    ApplicationScope,
    ExecutionSession,
    ModelConfig,
    ProviderConfig,
)

RUNNER = "ai_hub.services.execution_runner.run_execution_session"
ANSWER = '{"score": "pass", "feedback": "Well done.", "follow_up_question": ""}'
CONTRACT_KEYS = {"score", "feedback", "follow_up_question", "ai_session_pk"}


def fake_runner(session_pk, **kwargs):
    session = ExecutionSession.objects.get(pk=session_pk)
    session.final_context = {"final_answer": ANSWER}
    session.save()


class RuntimeScopeTestCase(TestCase):
    def setUp(self):
        module = TutorialModule.objects.create(title="M", slug="m", order=0)
        self.exercise = LabExercise.objects.create(
            module=module, title="E", slug="e", prompt="p", evaluation_rubric="r"
        )
        provider = ProviderConfig.objects.create(
            name="p", provider_type=ProviderConfig.ProviderType.OLLAMA
        )
        self.model = ModelConfig.objects.create(provider=provider, model_name="m")

    def scope(self, slug, *, active=True):
        return ApplicationScope.objects.create(
            slug=slug, name=f"Scope {slug}", is_active=active
        )

    def evaluator(self, scope, *, active=True):
        return AgentProfile.objects.create(
            application_scope=scope, name=LAB_EVALUATOR_AGENT_NAME, role="r",
            model_config=self.model, is_active=active,
        )

    def evaluate(self):
        with patch(RUNNER, side_effect=fake_runner) as runner:
            result = evaluate_lab_answer(exercise=self.exercise, user_input="student answer")
        return result, runner

    def assertPending(self):
        result, runner = self.evaluate()
        self.assertEqual(result["score"], "pending")
        self.assertIsNone(result["ai_session_pk"])
        self.assertEqual(set(result), CONTRACT_KEYS | set())  # contract unchanged
        runner.assert_not_called()
        self.assertFalse(ExecutionSession.objects.exists())

    def assertEvaluated(self, agent):
        result, runner = self.evaluate()
        self.assertEqual(result["score"], "pass")
        self.assertEqual(set(result), CONTRACT_KEYS)
        runner.assert_called_once()
        session = ExecutionSession.objects.get(pk=result["ai_session_pk"])
        self.assertEqual(session.entry_agent, agent)


class ForeignEvaluatorTests(RuntimeScopeTestCase):
    def test_an_active_evaluator_of_another_scope_is_never_executed(self):
        self.scope("academy")
        foreign = self.scope("other-app")
        self.evaluator(foreign)

        self.assertPending()

    def test_student_content_does_not_reach_a_foreign_evaluator(self):
        self.scope("academy")
        self.evaluator(self.scope("other-app"))

        with patch(RUNNER) as runner:
            result = evaluate_lab_answer(exercise=self.exercise, user_input="secret answer")

        runner.assert_not_called()
        self.assertEqual(result["score"], "pending")
        self.assertFalse(ExecutionSession.objects.exists())

    def test_the_pending_result_still_carries_the_rubric(self):
        self.scope("academy")
        self.evaluator(self.scope("other-app"))

        result, _ = self.evaluate()

        self.assertIn(self.exercise.evaluation_rubric, result["feedback"])
        self.assertEqual(result["follow_up_question"], "")


class MissingOrInactiveOwnershipTests(RuntimeScopeTestCase):
    def test_no_academy_scope_fails_closed_and_creates_nothing(self):
        self.evaluator(self.scope("someone-elses-app"))

        self.assertPending()
        self.assertEqual(ApplicationScope.objects.count(), 1)

    def test_a_single_active_foreign_scope_is_not_adopted(self):
        self.scope("only-active-scope")

        self.assertPending()

    def test_an_inactive_academy_scope_fails_closed_even_with_its_evaluator(self):
        academy = self.scope("academy", active=False)
        self.evaluator(academy)

        self.assertPending()
        academy.refresh_from_db()
        self.assertFalse(academy.is_active)  # never reactivated

    def test_an_inactive_evaluator_in_academys_scope_is_not_used(self):
        self.evaluator(self.scope("academy"), active=False)

        self.assertPending()

    def test_no_evaluator_at_all_is_pending(self):
        self.scope("academy")

        self.assertPending()

    @override_settings(ACADEMY_APPLICATION_SCOPE_SLUG="not a valid slug!")
    def test_an_invalid_scope_setting_fails_closed(self):
        self.evaluator(self.scope("academy"))

        self.assertPending()


class LegitimateEvaluationTests(RuntimeScopeTestCase):
    def test_the_evaluator_in_the_default_academy_scope_is_used(self):
        agent = self.evaluator(self.scope("academy"))

        self.assertEvaluated(agent)

    @override_settings(ACADEMY_APPLICATION_SCOPE_SLUG="legacy-default")
    def test_an_explicitly_declared_legacy_scope_is_used(self):
        agent = self.evaluator(self.scope("legacy-default"))

        self.assertEvaluated(agent)

    @override_settings(ACADEMY_APPLICATION_SCOPE_SLUG="legacy-default")
    def test_a_declared_scope_does_not_fall_back_to_the_default_one(self):
        self.scope("legacy-default")
        self.evaluator(self.scope("academy"))

        self.assertPending()

    def test_several_active_scopes_do_not_change_ownership(self):
        agent = self.evaluator(self.scope("academy"))
        self.scope("other-app")
        self.scope("third-app")

        self.assertEvaluated(agent)

    def test_several_active_scopes_never_select_the_foreign_evaluator(self):
        self.scope("academy")
        self.scope("third-app")
        self.evaluator(self.scope("other-app"))

        self.assertPending()

    def test_an_unparseable_answer_keeps_its_existing_fallback(self):
        self.evaluator(self.scope("academy"))

        def prose_runner(session_pk, **kwargs):
            session = ExecutionSession.objects.get(pk=session_pk)
            session.final_context = {"final_answer": "Clearly a pass."}
            session.save()

        with patch(RUNNER, side_effect=prose_runner):
            result = evaluate_lab_answer(exercise=self.exercise, user_input="a")

        self.assertEqual(result["score"], "pass")
        self.assertEqual(result["feedback"], "Clearly a pass.")
