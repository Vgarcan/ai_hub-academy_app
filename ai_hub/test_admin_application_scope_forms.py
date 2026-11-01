"""Admin add/change forms must let the operator choose the ApplicationScope.

GitHub issue #25. S-14 made `application_scope` a required, non-null FK on
`KnowledgeCollection` and `AgentProfile`, but both Admin fieldsets omitted it,
so the documented Admin path could not create either object: the POST re-rendered
the form with a hidden validation error and saved nothing.

The contract these tests pin:

  * the field is rendered and REQUIRED on the add forms;
  * Core never guesses - even with exactly one active scope nothing is
    preselected, and a missing or unknown scope creates nothing;
  * the chosen scope is persisted exactly as submitted;
  * ownership is not editable afterwards (it is displayed read-only), because a
    collection's scope is denormalized onto its chunks and embeddings and an
    agent's scope decides which collections it may reach;
  * only active scopes are offered for new resources;
  * the standard Django model permissions still gate everything.

These are real Admin GET/POST requests through the test client against the
disposable test database. Nothing here touches the Core authorization model.
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.test import Client, TestCase
from django.urls import reverse

from ai_hub.models import (
    AgentProfile,
    ApplicationScope,
    KnowledgeCollection,
    ModelConfig,
    ProviderConfig,
)


COLLECTION_ADD = "admin:ai_hub_knowledgecollection_add"
COLLECTION_CHANGE = "admin:ai_hub_knowledgecollection_change"
AGENT_ADD = "admin:ai_hub_agentprofile_add"
AGENT_CHANGE = "admin:ai_hub_agentprofile_change"


def _scope(name, *, active=True):
    return ApplicationScope.objects.create(
        name=name, slug=name.lower().replace(" ", "-"), is_active=active
    )


class AdminScopeFormTestCase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.superuser = get_user_model().objects.create_superuser(
            "scope-admin", "scope-admin@example.test", "not-a-real-password"
        )
        provider = ProviderConfig.objects.create(
            name="Admin scope provider",
            provider_type=ProviderConfig.ProviderType.TRAINING,
        )
        cls.model_config = ModelConfig.objects.create(
            provider=provider, model_name="training"
        )

    def setUp(self):
        self.client = Client()
        self.client.force_login(self.superuser)

    def collection_payload(self, **overrides):
        data = {
            "name": "Admin collection",
            "description": "Created through the Admin.",
            "is_active": "on",
            # Inline formset management data: no documents are submitted.
            "documents-TOTAL_FORMS": "0",
            "documents-INITIAL_FORMS": "0",
            "documents-MIN_NUM_FORMS": "0",
            "documents-MAX_NUM_FORMS": "1000",
        }
        data.update(overrides)
        return data

    def agent_payload(self, **overrides):
        data = {
            "name": "Admin agent",
            "role": "Created through the Admin.",
            "is_active": "on",
            "execution_mode": AgentProfile.ExecutionMode.INHERIT,
            "model_config": str(self.model_config.pk),
            "knowledge_max_chars": "6000",
            "system_prompt": "",
            "input_contract": "{}",
            "output_contract": "{}",
        }
        for prefix in ("toolbox_assignments", "tool_grants"):
            data.update(
                {
                    f"{prefix}-TOTAL_FORMS": "0",
                    f"{prefix}-INITIAL_FORMS": "0",
                    f"{prefix}-MIN_NUM_FORMS": "0",
                    f"{prefix}-MAX_NUM_FORMS": "1000",
                }
            )
        data.update(overrides)
        return data


class CollectionAddFormTests(AdminScopeFormTestCase):
    def test_get_renders_a_required_scope_selector(self):
        _scope("Alpha")

        response = self.client.get(reverse(COLLECTION_ADD))

        self.assertEqual(response.status_code, 200)
        form = response.context["adminform"].form
        self.assertIn("application_scope", form.fields)
        self.assertTrue(form.fields["application_scope"].required)
        self.assertContains(response, 'name="application_scope"')

    def test_post_with_selected_scope_creates_collection_in_that_scope(self):
        _scope("Alpha")
        beta = _scope("Beta")

        response = self.client.post(
            reverse(COLLECTION_ADD),
            self.collection_payload(application_scope=str(beta.pk)),
        )

        self.assertEqual(response.status_code, 302, self._errors(response))
        collection = KnowledgeCollection.objects.get(name="Admin collection")
        self.assertEqual(collection.application_scope, beta)

    def test_post_without_scope_creates_nothing_and_reports_the_field(self):
        _scope("Alpha")

        response = self.client.post(reverse(COLLECTION_ADD), self.collection_payload())

        self.assertEqual(response.status_code, 200)
        self.assertFalse(KnowledgeCollection.objects.exists())
        self.assertIn(
            "application_scope", response.context["adminform"].form.errors
        )

    def test_post_with_blank_scope_creates_nothing(self):
        _scope("Alpha")

        response = self.client.post(
            reverse(COLLECTION_ADD), self.collection_payload(application_scope="")
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(KnowledgeCollection.objects.exists())

    def test_a_single_active_scope_is_not_guessed(self):
        only = _scope("Only")

        response = self.client.get(reverse(COLLECTION_ADD))
        form = response.context["adminform"].form
        self.assertIsNone(form["application_scope"].value())

        response = self.client.post(reverse(COLLECTION_ADD), self.collection_payload())
        self.assertEqual(response.status_code, 200)
        self.assertFalse(KnowledgeCollection.objects.exists())
        self.assertTrue(ApplicationScope.objects.filter(pk=only.pk).exists())

    def test_unknown_scope_id_creates_nothing(self):
        _scope("Alpha")

        response = self.client.post(
            reverse(COLLECTION_ADD), self.collection_payload(application_scope="999999")
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(KnowledgeCollection.objects.exists())

    def test_inactive_scope_is_not_offered_and_not_accepted(self):
        active = _scope("Active")
        inactive = _scope("Retired", active=False)

        response = self.client.get(reverse(COLLECTION_ADD))
        offered = response.context["adminform"].form.fields["application_scope"].queryset
        self.assertEqual(list(offered), [active])

        response = self.client.post(
            reverse(COLLECTION_ADD),
            self.collection_payload(application_scope=str(inactive.pk)),
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(KnowledgeCollection.objects.exists())

    def test_zero_scopes_cannot_create_and_nothing_is_created_for_the_operator(self):
        response = self.client.post(
            reverse(COLLECTION_ADD), self.collection_payload(application_scope="1")
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(KnowledgeCollection.objects.exists())
        self.assertFalse(ApplicationScope.objects.exists())

    def test_other_invalid_fields_still_fail_with_a_valid_scope(self):
        alpha = _scope("Alpha")

        response = self.client.post(
            reverse(COLLECTION_ADD),
            self.collection_payload(application_scope=str(alpha.pk), name=""),
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(KnowledgeCollection.objects.exists())
        self.assertIn("name", response.context["adminform"].form.errors)
        self.assertNotIn("application_scope", response.context["adminform"].form.errors)

    def _errors(self, response):
        context = getattr(response, "context", None)
        if context is None:
            return ""
        return str(context["adminform"].form.errors)


class CollectionChangeFormTests(AdminScopeFormTestCase):
    def setUp(self):
        super().setUp()
        self.alpha = _scope("Alpha")
        self.beta = _scope("Beta")
        self.collection = KnowledgeCollection.objects.create(
            application_scope=self.alpha, name="Existing collection"
        )

    def test_change_form_shows_the_scope_read_only(self):
        response = self.client.get(reverse(COLLECTION_CHANGE, args=[self.collection.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("application_scope", response.context["adminform"].form.fields)
        self.assertIn(
            "application_scope", response.context["adminform"].readonly_fields
        )
        self.assertContains(response, "Alpha")

    def test_editing_other_fields_preserves_ownership(self):
        response = self.client.post(
            reverse(COLLECTION_CHANGE, args=[self.collection.pk]),
            self.collection_payload(name="Renamed collection"),
        )

        self.assertEqual(response.status_code, 302)
        self.collection.refresh_from_db()
        self.assertEqual(self.collection.name, "Renamed collection")
        self.assertEqual(self.collection.application_scope, self.alpha)

    def test_a_forged_scope_in_a_change_post_is_ignored(self):
        response = self.client.post(
            reverse(COLLECTION_CHANGE, args=[self.collection.pk]),
            self.collection_payload(
                name="Forged", application_scope=str(self.beta.pk)
            ),
        )

        self.assertEqual(response.status_code, 302)
        self.collection.refresh_from_db()
        self.assertEqual(self.collection.application_scope, self.alpha)

    def test_collection_in_an_inactive_scope_remains_editable(self):
        self.alpha.is_active = False
        self.alpha.save()

        response = self.client.post(
            reverse(COLLECTION_CHANGE, args=[self.collection.pk]),
            self.collection_payload(name="Still editable"),
        )

        self.assertEqual(response.status_code, 302)
        self.collection.refresh_from_db()
        self.assertEqual(self.collection.name, "Still editable")
        self.assertEqual(self.collection.application_scope, self.alpha)


class AgentAddFormTests(AdminScopeFormTestCase):
    def test_get_renders_a_required_scope_selector(self):
        _scope("Alpha")

        response = self.client.get(reverse(AGENT_ADD))

        self.assertEqual(response.status_code, 200)
        form = response.context["adminform"].form
        self.assertIn("application_scope", form.fields)
        self.assertTrue(form.fields["application_scope"].required)
        self.assertContains(response, 'name="application_scope"')

    def test_post_with_selected_scope_creates_agent_in_that_scope(self):
        _scope("Alpha")
        beta = _scope("Beta")

        response = self.client.post(
            reverse(AGENT_ADD), self.agent_payload(application_scope=str(beta.pk))
        )

        self.assertEqual(response.status_code, 302, self._errors(response))
        agent = AgentProfile.objects.get(name="Admin agent")
        self.assertEqual(agent.application_scope, beta)

    def test_post_without_scope_creates_nothing_and_reports_the_field(self):
        _scope("Alpha")

        response = self.client.post(reverse(AGENT_ADD), self.agent_payload())

        self.assertEqual(response.status_code, 200)
        self.assertFalse(AgentProfile.objects.exists())
        self.assertIn(
            "application_scope", response.context["adminform"].form.errors
        )

    def test_a_single_active_scope_is_not_guessed(self):
        _scope("Only")

        response = self.client.get(reverse(AGENT_ADD))
        self.assertIsNone(response.context["adminform"].form["application_scope"].value())

        response = self.client.post(reverse(AGENT_ADD), self.agent_payload())
        self.assertEqual(response.status_code, 200)
        self.assertFalse(AgentProfile.objects.exists())

    def test_inactive_scope_is_not_offered_and_not_accepted(self):
        active = _scope("Active")
        inactive = _scope("Retired", active=False)

        response = self.client.get(reverse(AGENT_ADD))
        offered = response.context["adminform"].form.fields["application_scope"].queryset
        self.assertEqual(list(offered), [active])

        response = self.client.post(
            reverse(AGENT_ADD), self.agent_payload(application_scope=str(inactive.pk))
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(AgentProfile.objects.exists())

    def test_unknown_scope_id_creates_nothing(self):
        _scope("Alpha")

        response = self.client.post(
            reverse(AGENT_ADD), self.agent_payload(application_scope="999999")
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(AgentProfile.objects.exists())

    def test_other_invalid_fields_still_fail_with_a_valid_scope(self):
        alpha = _scope("Alpha")

        response = self.client.post(
            reverse(AGENT_ADD),
            self.agent_payload(application_scope=str(alpha.pk), role=""),
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(AgentProfile.objects.exists())
        self.assertIn("role", response.context["adminform"].form.errors)

    def _errors(self, response):
        context = getattr(response, "context", None)
        if context is None:
            return ""
        return str(context["adminform"].form.errors)


class AgentChangeFormTests(AdminScopeFormTestCase):
    def setUp(self):
        super().setUp()
        self.alpha = _scope("Alpha")
        self.beta = _scope("Beta")
        self.agent = AgentProfile.objects.create(
            application_scope=self.alpha,
            name="Existing agent",
            role="Existing role",
            model_config=self.model_config,
        )

    def test_change_form_shows_the_scope_read_only(self):
        response = self.client.get(reverse(AGENT_CHANGE, args=[self.agent.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("application_scope", response.context["adminform"].form.fields)
        self.assertIn(
            "application_scope", response.context["adminform"].readonly_fields
        )

    def test_editing_other_fields_preserves_ownership(self):
        response = self.client.post(
            reverse(AGENT_CHANGE, args=[self.agent.pk]),
            self.agent_payload(name="Existing agent", role="Edited role"),
        )

        self.assertEqual(response.status_code, 302)
        self.agent.refresh_from_db()
        self.assertEqual(self.agent.role, "Edited role")
        self.assertEqual(self.agent.application_scope, self.alpha)

    def test_a_forged_scope_in_a_change_post_is_ignored(self):
        response = self.client.post(
            reverse(AGENT_CHANGE, args=[self.agent.pk]),
            self.agent_payload(
                name="Existing agent", application_scope=str(self.beta.pk)
            ),
        )

        self.assertEqual(response.status_code, 302)
        self.agent.refresh_from_db()
        self.assertEqual(self.agent.application_scope, self.alpha)


class AdminScopePermissionTests(AdminScopeFormTestCase):
    def _staff_with(self, *codenames):
        user = get_user_model().objects.create_user(
            "limited", "limited@example.test", "not-a-real-password", is_staff=True
        )
        user.user_permissions.set(Permission.objects.filter(codename__in=codenames))
        client = Client()
        client.force_login(user)
        return client

    def test_staff_without_add_permission_cannot_create_a_collection(self):
        alpha = _scope("Alpha")
        client = self._staff_with("view_knowledgecollection")

        self.assertEqual(client.get(reverse(COLLECTION_ADD)).status_code, 403)
        response = client.post(
            reverse(COLLECTION_ADD),
            self.collection_payload(application_scope=str(alpha.pk)),
        )

        self.assertEqual(response.status_code, 403)
        self.assertFalse(KnowledgeCollection.objects.exists())

    def test_staff_without_add_permission_cannot_create_an_agent(self):
        alpha = _scope("Alpha")
        client = self._staff_with("view_agentprofile")

        self.assertEqual(client.get(reverse(AGENT_ADD)).status_code, 403)
        response = client.post(
            reverse(AGENT_ADD), self.agent_payload(application_scope=str(alpha.pk))
        )

        self.assertEqual(response.status_code, 403)
        self.assertFalse(AgentProfile.objects.exists())

    def test_staff_with_add_permission_can_create_with_an_explicit_scope(self):
        alpha = _scope("Alpha")
        client = self._staff_with("add_knowledgecollection", "view_knowledgecollection")

        response = client.post(
            reverse(COLLECTION_ADD),
            self.collection_payload(application_scope=str(alpha.pk)),
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            KnowledgeCollection.objects.get().application_scope, alpha
        )

    def test_staff_with_add_permission_still_must_choose_a_scope(self):
        _scope("Alpha")
        client = self._staff_with("add_agentprofile", "view_agentprofile")

        response = client.post(reverse(AGENT_ADD), self.agent_payload())

        self.assertEqual(response.status_code, 200)
        self.assertFalse(AgentProfile.objects.exists())

    def test_staff_with_change_but_not_add_cannot_reassign_an_agent_scope(self):
        alpha = _scope("Alpha")
        beta = _scope("Beta")
        agent = AgentProfile.objects.create(
            application_scope=alpha,
            name="Guarded agent",
            role="Guarded",
            model_config=self.model_config,
        )
        client = self._staff_with("change_agentprofile", "view_agentprofile")

        response = client.post(
            reverse(AGENT_CHANGE, args=[agent.pk]),
            self.agent_payload(name="Guarded agent", application_scope=str(beta.pk)),
        )

        self.assertEqual(response.status_code, 302)
        agent.refresh_from_db()
        self.assertEqual(agent.application_scope, alpha)


class MultipleScopeTests(AdminScopeFormTestCase):
    def test_each_resource_lands_in_the_scope_chosen_for_it(self):
        alpha = _scope("Alpha")
        beta = _scope("Beta")

        self.client.post(
            reverse(COLLECTION_ADD),
            self.collection_payload(name="In alpha", application_scope=str(alpha.pk)),
        )
        self.client.post(
            reverse(COLLECTION_ADD),
            self.collection_payload(name="In beta", application_scope=str(beta.pk)),
        )
        self.client.post(
            reverse(AGENT_ADD),
            self.agent_payload(name="Agent in beta", application_scope=str(beta.pk)),
        )

        self.assertEqual(
            KnowledgeCollection.objects.get(name="In alpha").application_scope, alpha
        )
        self.assertEqual(
            KnowledgeCollection.objects.get(name="In beta").application_scope, beta
        )
        self.assertEqual(
            AgentProfile.objects.get(name="Agent in beta").application_scope, beta
        )

    def test_the_selector_offers_every_active_scope_and_no_default(self):
        alpha = _scope("Alpha")
        beta = _scope("Beta")
        _scope("Retired", active=False)

        for url_name in (COLLECTION_ADD, AGENT_ADD):
            with self.subTest(url_name=url_name):
                response = self.client.get(reverse(url_name))
                form = response.context["adminform"].form
                offered = form.fields["application_scope"].queryset
                self.assertEqual(set(offered), {alpha, beta})
                self.assertIsNone(form["application_scope"].value())
