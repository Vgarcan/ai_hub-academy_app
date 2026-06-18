from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from academy.models import (
    DocumentationChunk,
    DocumentationChatSession,
    DocumentationPage,
    DocumentationSource,
    TutorialMission,
    TutorialModule,
    UserMissionProgress,
)
from academy.services.documentation_search import search_documentation

User = get_user_model()


class DocumentationImportTest(TestCase):
    def setUp(self):
        source = DocumentationSource.objects.create(name="Test Docs", slug="test-docs")
        self.page = DocumentationPage.objects.create(
            source=source,
            title="Core Concepts",
            slug="core-concepts",
            source_path="04_CORE_CONCEPTS.md",
            body_markdown="# Core\n\n## Provider\n\nA provider is...\n\n## Model\n\nA model is...",
            order=1,
        )
        DocumentationChunk.objects.create(
            page=self.page,
            heading="Provider",
            anchor="provider",
            body_markdown="A provider is an AI service account.",
            order=1,
            search_text="a provider is an ai service account",
        )
        DocumentationChunk.objects.create(
            page=self.page,
            heading="Model",
            anchor="model",
            body_markdown="A model is a specific AI model configuration.",
            order=2,
            search_text="a model is a specific ai model configuration",
        )

    def test_import_creates_chunks(self):
        self.assertEqual(DocumentationChunk.objects.filter(page=self.page).count(), 2)

    def test_search_returns_relevant_chunks(self):
        results = search_documentation("provider")
        self.assertTrue(any(c.heading == "Provider" for c in results))

    def test_search_empty_query(self):
        results = search_documentation("")
        self.assertEqual(results, [])

    def test_docs_list_view(self):
        response = self.client.get(reverse("academy:docs_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Core Concepts")

    def test_docs_detail_view(self):
        response = self.client.get(reverse("academy:docs_detail", args=["core-concepts"]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Core Concepts")

    def test_docs_search_view(self):
        response = self.client.get(reverse("academy:docs_search") + "?q=provider")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Provider")


class TutorialTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("testuser", password="testpass")
        self.module = TutorialModule.objects.create(
            title="Test Module",
            slug="test-module",
            order=1,
        )
        self.mission = TutorialMission.objects.create(
            module=self.module,
            title="Test Mission",
            slug="test-mission",
            order=1,
            goal="Do something",
            instructions_markdown="## Instructions\n\nDo the thing.",
            validation_key="visited_control_room",
        )

    def test_tutorial_list_view(self):
        response = self.client.get(reverse("academy:tutorial_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Module")

    def test_tutorial_module_view(self):
        response = self.client.get(
            reverse("academy:tutorial_module", args=["test-module"])
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Mission")

    def test_tutorial_mission_view(self):
        response = self.client.get(
            reverse("academy:tutorial_mission", args=["test-module", "test-mission"])
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Mission")

    def test_check_mission_creates_progress(self):
        self.client.login(username="testuser", password="testpass")
        response = self.client.post(
            reverse("academy:check_mission", args=["test-module", "test-mission"]),
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("passed", data)
        self.assertIn("feedback", data)

        progress = UserMissionProgress.objects.filter(
            user=self.user, mission=self.mission
        ).first()
        self.assertIsNotNone(progress)
        self.assertEqual(progress.attempts, 1)

    def test_check_mission_validator_passes(self):
        self.client.login(username="testuser", password="testpass")
        response = self.client.post(
            reverse("academy:check_mission", args=["test-module", "test-mission"]),
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        data = response.json()
        # visited_control_room always passes
        self.assertTrue(data["passed"])


class TrainingProviderTest(TestCase):
    def test_training_completion_call(self):
        from ai_hub.services.litellm_client import completion_call

        result = completion_call(
            model="training/assistant",
            messages=[
                {"role": "system", "content": "You are a documentation assistant."},
                {"role": "user", "content": "What is a provider?"},
            ],
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result.get("stubbed"))
        self.assertIn("content", result)

    def test_training_provider_no_api_key_required(self):
        from ai_hub.models import ModelConfig, ProviderConfig
        from ai_hub.services.provider_registry import resolve_model_config

        provider = ProviderConfig.objects.create(
            name="Test Training Provider",
            provider_type="training",
            is_active=True,
        )
        model = ModelConfig.objects.create(
            provider=provider,
            model_name="training/assistant",
            is_active=True,
        )
        config = resolve_model_config(model)
        self.assertEqual(config["provider_type"], "training")
        self.assertEqual(config["api_key"], "")

    def test_training_execution_session_succeeds(self):
        from ai_hub.models import (
            AgentProfile,
            ExecutionSession,
            ModelConfig,
            PipelineDefinition,
            PipelineStep,
            ProviderConfig,
        )
        from ai_hub.services.execution_runner import run_execution_session

        provider = ProviderConfig.objects.create(
            name="Training Provider Test",
            provider_type="training",
            is_active=True,
        )
        model = ModelConfig.objects.create(
            provider=provider,
            model_name="training/assistant",
            is_active=True,
        )
        agent = AgentProfile.objects.create(
            name="Test Classifier",
            role="Classify",
            system_prompt="Classify the ticket.",
            model_config=model,
            input_contract={"required": ["ticket_title", "ticket_text"]},
            output_contract={},
            is_active=True,
        )
        pipeline = PipelineDefinition.objects.create(
            name="Test Pipeline",
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
            initial_context={"ticket_title": "Bug", "ticket_text": "It is broken."},
        )
        run_execution_session(session.pk)
        session.refresh_from_db()
        self.assertEqual(session.status, ExecutionSession.Status.SUCCESS)


class LandingViewTest(TestCase):
    def test_landing_200(self):
        response = self.client.get(reverse("academy:landing"))
        self.assertEqual(response.status_code, 200)

    def test_assistant_200(self):
        response = self.client.get(reverse("academy:assistant"))
        self.assertEqual(response.status_code, 200)
