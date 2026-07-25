from pathlib import Path
from io import StringIO
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import unquote

from django.contrib.auth import get_user_model
from django.contrib.staticfiles import finders
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from academy.models import (
    DocumentationChunk,
    DocumentationChatSession,
    DocumentationPage,
    DocumentationSource,
    LabExercise,
    TutorialMission,
    TutorialModule,
    UserMissionProgress,
)
from academy.services.documentation_search import search_documentation
from academy.tools.doc_sync import sync_all_docs
from ai_hub.models import AgentProfile, ExecutionSession, ModelConfig, ProviderConfig

User = get_user_model()


class ProjectDocumentationIntegrityTests(SimpleTestCase):
    def test_relative_markdown_links_resolve(self):
        project_root = Path(__file__).resolve().parent.parent
        doc_paths = [
            project_root / "README.md",
            project_root / "DEMO_SCRIPT.md",
            project_root / "ai_hub" / "README.md",
            project_root / "ai_hub" / "OPERATING_MODEL.md",
            *sorted((project_root / "ai_hub" / "_docs").glob("*.md")),
            *sorted((project_root / "docs_source").glob("*.md")),
        ]
        missing = []
        link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

        for doc_path in doc_paths:
            body = doc_path.read_text(encoding="utf-8")
            for raw_target in link_pattern.findall(body):
                target = raw_target.strip().strip("<>")
                if (
                    not target
                    or target.startswith(("#", "/", "http://", "https://", "mailto:"))
                ):
                    continue
                target = unquote(target.split("#", 1)[0])
                resolved = (doc_path.parent / target).resolve()
                if not resolved.exists():
                    missing.append(f"{doc_path.relative_to(project_root)} -> {target}")

        self.assertEqual(missing, [], "Broken local Markdown links:\n" + "\n".join(missing))

    def test_tutorial_static_assets_resolve(self):
        project_root = Path(__file__).resolve().parent.parent
        seed_path = (
            project_root
            / "academy"
            / "management"
            / "commands"
            / "seed_academy_training_data.py"
        )
        body = seed_path.read_text(encoding="utf-8")
        static_paths = sorted(
            set(re.findall(r"/static/([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)", body))
        )

        self.assertTrue(static_paths, "No tutorial static references were found.")
        missing = [path for path in static_paths if finders.find(path) is None]
        self.assertEqual(
            missing,
            [],
            "Missing tutorial static assets:\n" + "\n".join(missing),
        )


class SetupScriptTests(SimpleTestCase):
    @patch("setup_dev._manage")
    @patch("setup_dev._ok")
    def test_existing_admin_password_is_not_reported_as_new(
        self,
        _mocked_ok,
        mocked_manage,
    ):
        import setup_dev

        mocked_manage.return_value = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="That username is already taken. A user with that username already exists.",
        )

        self.assertIsNone(setup_dev.create_admin(Path("python")))

    @patch("builtins.print")
    def test_summary_says_existing_password_was_not_changed(self, mocked_print):
        import setup_dev

        setup_dev.print_summary(None)

        output = "\n".join(
            str(arg)
            for call in mocked_print.call_args_list
            for arg in call.args
        )
        self.assertIn("existing password was not changed", output)
        self.assertNotIn(setup_dev.ADMIN_PASSWORD, output)


class DocumentationSyncBoundaryTests(TestCase):
    def _sync_from(self, source_root: Path):
        with override_settings(
            AIHUB_DOCS_SOURCE=None,
            ACADEMY_DOCS_SOURCE=source_root,
        ):
            return sync_all_docs({}, {"source_name": "Boundary Test Docs"})

    def test_document_sync_uses_configured_docs_root(self):
        with TemporaryDirectory() as temp_dir:
            docs_source = Path(temp_dir) / "docs_source"
            docs_source.mkdir()
            (docs_source / "official.md").write_text("# Official\n", encoding="utf-8")

            result = self._sync_from(docs_source)

        self.assertEqual(result["checked"], 1)
        self.assertTrue(DocumentationPage.objects.filter(slug="official").exists())

    def test_document_sync_reads_platform_and_academy_roots(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            platform_source = root / "ai_hub_docs"
            academy_source = root / "academy_docs"
            platform_source.mkdir()
            academy_source.mkdir()
            (platform_source / "01_PLATFORM.md").write_text(
                "# Platform\n",
                encoding="utf-8",
            )
            (platform_source / "README.md").write_text(
                "# Developer index\n",
                encoding="utf-8",
            )
            (academy_source / "15_ACADEMY.md").write_text(
                "# Academy\n",
                encoding="utf-8",
            )

            with override_settings(
                AIHUB_DOCS_SOURCE=platform_source,
                ACADEMY_DOCS_SOURCE=academy_source,
            ):
                result = sync_all_docs({}, {"source_name": "Combined Docs"})

        self.assertEqual(result["checked"], 2)
        self.assertTrue(DocumentationPage.objects.filter(slug="01_platform").exists())
        self.assertTrue(DocumentationPage.objects.filter(slug="15_academy").exists())
        self.assertFalse(DocumentationPage.objects.filter(slug="readme").exists())

    def test_unchanged_sync_preserves_embeddings(self):
        with TemporaryDirectory() as temp_dir:
            docs_source = Path(temp_dir) / "docs_source"
            docs_source.mkdir()
            (docs_source / "official.md").write_text(
                "# Official\n\nStable text.\n",
                encoding="utf-8",
            )

            first = self._sync_from(docs_source)
            chunk = DocumentationChunk.objects.get()
            chunk.embedding = [0.1, 0.2]
            chunk.save(update_fields=["embedding"])
            second = self._sync_from(docs_source)

        chunk.refresh_from_db()
        self.assertEqual(first["synced"], 1)
        self.assertEqual(second["synced"], 0)
        self.assertEqual(second["unchanged"], 1)
        self.assertEqual(chunk.embedding, [0.1, 0.2])

    def test_complete_sync_deactivates_removed_pages(self):
        with TemporaryDirectory() as temp_dir:
            docs_source = Path(temp_dir) / "docs_source"
            docs_source.mkdir()
            stale_file = docs_source / "stale.md"
            stale_file.write_text("# Stale\n", encoding="utf-8")
            self._sync_from(docs_source)
            stale_file.unlink()
            (docs_source / "current.md").write_text("# Current\n", encoding="utf-8")

            result = self._sync_from(docs_source)

        self.assertEqual(result["deactivated"], 1)
        self.assertFalse(DocumentationPage.objects.get(slug="stale").is_active)
        self.assertTrue(DocumentationPage.objects.get(slug="current").is_active)

    def test_document_sync_does_not_read_myideas(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_source = root / "docs_source"
            private_source = root / ".myideas"
            docs_source.mkdir()
            private_source.mkdir()
            (docs_source / "public.md").write_text("# Public\n", encoding="utf-8")
            (private_source / "private.md").write_text("# Private\n", encoding="utf-8")

            result = self._sync_from(docs_source)

        self.assertEqual(result["checked"], 1)
        self.assertTrue(DocumentationPage.objects.filter(slug="public").exists())
        self.assertFalse(DocumentationPage.objects.filter(slug="private").exists())

    def test_document_sync_does_not_import_gitignored_private_markdown(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_source = root / "docs_source"
            private_source = root / ".myideas" / "GAME_IMPLEMENTATION"
            docs_source.mkdir()
            private_source.mkdir(parents=True)
            (private_source / "CURRENT_STATUS.md").write_text("# Private status\n", encoding="utf-8")

            result = self._sync_from(docs_source)

        self.assertEqual(result["checked"], 0)
        self.assertFalse(DocumentationPage.objects.filter(slug="current-status").exists())


class DocumentationSyncCommandTests(TestCase):
    @patch("academy.management.commands.run_doc_sync.run_execution_session")
    def test_run_doc_sync_explicitly_opts_in_to_legacy_action_tool(self, mocked_run):
        provider = ProviderConfig.objects.create(name="docs-provider", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="docs-model")
        AgentProfile.objects.create(
            name="Documentation Sync Agent",
            role="Documentation sync",
            model_config=model,
            input_contract={"required": ["goal"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )

        call_command("run_doc_sync", stdout=StringIO())

        session = ExecutionSession.objects.get(source_label="run_doc_sync management command")
        self.assertEqual(
            session.runtime_config["agent_tool_runtime"],
            "legacy_preexecute",
        )
        mocked_run.assert_called_once_with(session.pk, allow_legacy_game_action_tools=True)

    def test_startup_doc_sync_declares_the_same_legacy_action_compatibility(self):
        provider = ProviderConfig.objects.create(name="startup-docs-provider", provider_type="training")
        model = ModelConfig.objects.create(provider=provider, model_name="startup-docs-model")
        AgentProfile.objects.create(
            name="Documentation Sync Agent",
            role="Documentation sync",
            model_config=model,
            input_contract={"required": ["goal"]},
            output_contract={"required": ["agent", "llm", "tools"]},
        )

        from academy.apps import _trigger_doc_sync

        with patch("ai_hub.services.execution_runner.run_execution_session") as runner:
            _trigger_doc_sync()

        session = ExecutionSession.objects.get(source_label="startup auto-sync")
        self.assertEqual(
            session.runtime_config["agent_tool_runtime"],
            "legacy_preexecute",
        )
        runner.assert_called_once_with(
            session.pk,
            allow_legacy_game_action_tools=True,
        )


class OllamaSeedCommandTests(TestCase):
    def test_seed_output_points_to_the_real_assistant_route(self):
        stdout = StringIO()

        call_command(
            "seed_ollama_agents",
            "--base-url",
            "http://localhost:11434",
            stdout=stdout,
        )

        self.assertIn("/assistant/", stdout.getvalue())
        self.assertNotIn("/chat/", stdout.getvalue())

    def test_seed_upgrades_the_seeded_training_assistant_to_ollama(self):
        training_provider = ProviderConfig.objects.create(
            name="Training",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        training_model = ModelConfig.objects.create(
            provider=training_provider,
            model_name="training/assistant",
        )
        assistant = AgentProfile.objects.create(
            name="AI Hub Documentation Assistant",
            role="Documentation assistant",
            model_config=training_model,
        )

        call_command(
            "seed_ollama_agents",
            "--base-url",
            "http://localhost:11434",
            stdout=StringIO(),
        )

        assistant.refresh_from_db()
        self.assertEqual(
            assistant.model_config.provider.provider_type,
            ProviderConfig.ProviderType.OLLAMA,
        )
        self.assertEqual(assistant.model_config.model_name, "ollama/qwen3:8b")


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

    def test_embed_command_reports_when_active_chunks_are_complete(self):
        DocumentationChunk.objects.filter(page=self.page).update(
            embedding=[1.0, 0.0]
        )
        stdout = StringIO()

        call_command("embed_docs", stdout=stdout)

        self.assertIn(
            "All active documentation chunks already embedded.",
            stdout.getvalue(),
        )

    def test_search_excludes_inactive_pages_and_sources(self):
        provider_chunk = self.page.chunks.get(heading="Provider")
        provider_chunk.embedding = [1.0, 0.0]
        provider_chunk.save(update_fields=["embedding"])

        self.page.is_active = False
        self.page.save(update_fields=["is_active"])
        with patch(
            "academy.services.embeddings.get_embedding",
            return_value=[1.0, 0.0],
        ):
            self.assertEqual(search_documentation("provider"), [])

        self.page.is_active = True
        self.page.save(update_fields=["is_active"])
        self.page.source.is_active = False
        self.page.source.save(update_fields=["is_active"])
        with patch(
            "academy.services.embeddings.get_embedding",
            return_value=[1.0, 0.0],
        ):
            self.assertEqual(search_documentation("provider"), [])

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

    def test_inactive_source_hides_pages_from_browser(self):
        self.page.source.is_active = False
        self.page.source.save(update_fields=["is_active"])

        list_response = self.client.get(reverse("academy:docs_list"))
        detail_response = self.client.get(
            reverse("academy:docs_detail", args=["core-concepts"])
        )

        self.assertNotContains(list_response, "Core Concepts")
        self.assertEqual(detail_response.status_code, 404)


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


class AcademySeedCommandTests(TestCase):
    def test_seed_force_update_refreshes_existing_mission_text(self):
        with patch("builtins.print"):
            call_command("seed_academy_training_data", stdout=StringIO())

        mission = TutorialMission.objects.get(slug="enter-the-control-room")
        mission.instructions_markdown = "old interface copy"
        mission.save(update_fields=["instructions_markdown"])

        with patch("builtins.print"):
            call_command("seed_academy_training_data", "--force-update", stdout=StringIO())

        mission.refresh_from_db()
        # marker from the refreshed real content (enter-the-control-room now
        # describes the cockpit + Control Center rather than "Mission Deck")
        self.assertIn("Control Center", mission.instructions_markdown)
        self.assertTrue(
            TutorialMission.objects.filter(slug="inspect-the-control-center").exists()
        )
        # The legacy slug must not linger after the rename migration runs.
        self.assertFalse(
            TutorialMission.objects.filter(slug="inspect-the-mission-deck").exists()
        )

    def test_seed_lab_exercises_adds_control_center_interface_lab(self):
        TutorialModule.objects.create(
            title="Orientation",
            slug="orientation",
            order=0,
        )

        call_command("seed_lab_exercises", "--force", stdout=StringIO())

        exercise = LabExercise.objects.get(slug="control-center-interface-audit")
        self.assertEqual(exercise.module.slug, "orientation")
        self.assertFalse(exercise.requires_api)
        self.assertIn("Open record in admin", exercise.context)


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

    def test_anonymous_assistant_execution_is_disabled_by_default(self):
        response = self.client.post(
            reverse("academy:assistant_ask"),
            {"question": "Explain GAME"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 403)

    def test_assistant_rejects_oversized_question(self):
        user = User.objects.create_user("assistant-user", password="testpass")
        self.client.force_login(user)

        response = self.client.post(
            reverse("academy:assistant_ask"),
            {"question": "x" * 4001},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )

        self.assertEqual(response.status_code, 400)
