"""Admin must not accept an Agent -> Knowledge collection assignment across scopes.

GitHub issue #28. `AgentProfile.knowledge_collections` is a plain many-to-many,
so the Admin saved a collection owned by ANOTHER ApplicationScope onto an agent.
S-15 already makes such a row grant zero authorization, so nothing leaked - but
the Admin displayed an assignment that can never work, which misleads operators.

Validation strategy: a form-layer check on the agent Admin form. The rule is
"a collection's scope must equal the agent's scope":

  * on ADD the scope is the one selected in the same submission, so creating an
    agent and attaching collections in one POST still works;
  * on CHANGE the scope is the (read-only) stored one, never a posted value, so
    a forged `application_scope` cannot move the goalposts;
  * a foreign collection is rejected with a field-level error on
    `knowledge_collections`, never silently dropped;
  * the check reads the cleaned form data, so a hand-crafted POST goes through
    exactly the same validation as the rendered form.

Existing cross-scope rows (written before this fix, by raw ORM, fixtures or a
pre-S-14 database) are NOT deleted or rewritten. They may stay while the
operator edits other fields, they may be removed, and the Admin warns about them
by name. Only NEWLY ADDING a foreign collection is refused - otherwise one bad
legacy row would lock an agent against even being deactivated.

S-15's runtime authorization is asserted unchanged at the end.
"""
from django.contrib.messages import get_messages
from django.urls import reverse

from ai_hub.models import AgentProfile, KnowledgeCollection
from ai_hub.services.knowledge_authorization import resolve_effective_knowledge_scope
from ai_hub.test_admin_application_scope_forms import (
    AGENT_ADD,
    AGENT_CHANGE,
    AdminScopeFormTestCase,
    _scope,
)

THROUGH = AgentProfile.knowledge_collections.through


def collection(scope, name):
    return KnowledgeCollection.objects.create(application_scope=scope, name=name)


class AgentCollectionScopeTestCase(AdminScopeFormTestCase):
    def setUp(self):
        super().setUp()
        self.alpha = _scope("Alpha")
        self.beta = _scope("Beta")
        self.in_alpha = collection(self.alpha, "alpha-docs")
        self.in_alpha_2 = collection(self.alpha, "alpha-faq")
        self.in_beta = collection(self.beta, "beta-docs")

    def create_payload(self, scope, *collections, **overrides):
        return self.agent_payload(
            application_scope=str(scope.pk),
            knowledge_collections=[str(c.pk) for c in collections],
            **overrides,
        )

    def edit_payload(self, agent, *collections, **overrides):
        return self.agent_payload(
            name=agent.name,
            knowledge_collections=[str(c.pk) for c in collections],
            **overrides,
        )

    def assigned(self, agent):
        return set(agent.knowledge_collections.values_list("name", flat=True))

    def form_errors(self, response):
        return response.context["adminform"].form.errors

    def messages(self, response):
        return [str(m) for m in get_messages(response.wsgi_request)]


class CreateAgentTests(AgentCollectionScopeTestCase):
    def test_same_scope_collection_is_accepted_in_the_creating_submission(self):
        response = self.client.post(
            reverse(AGENT_ADD), self.create_payload(self.alpha, self.in_alpha, self.in_alpha_2)
        )

        self.assertEqual(response.status_code, 302)
        agent = AgentProfile.objects.get(name="Admin agent")
        self.assertEqual(agent.application_scope, self.alpha)
        self.assertEqual(self.assigned(agent), {"alpha-docs", "alpha-faq"})

    def test_a_collection_from_another_scope_is_rejected_with_a_field_error(self):
        response = self.client.post(
            reverse(AGENT_ADD), self.create_payload(self.alpha, self.in_beta)
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("knowledge_collections", self.form_errors(response))
        message = " ".join(self.form_errors(response)["knowledge_collections"])
        self.assertIn("beta-docs", message)
        self.assertIn(self.alpha.slug, message)
        self.assertFalse(AgentProfile.objects.exists())
        self.assertFalse(THROUGH.objects.exists())

    def test_one_foreign_collection_among_valid_ones_rejects_the_whole_submission(self):
        response = self.client.post(
            reverse(AGENT_ADD),
            self.create_payload(self.alpha, self.in_alpha, self.in_beta),
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("knowledge_collections", self.form_errors(response))
        self.assertFalse(AgentProfile.objects.exists())
        self.assertFalse(THROUGH.objects.exists())

    def test_the_selected_scope_decides_not_the_collections(self):
        """Same collection, different chosen scope: the verdict follows the scope."""
        response = self.client.post(
            reverse(AGENT_ADD), self.create_payload(self.beta, self.in_beta)
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            self.assigned(AgentProfile.objects.get(name="Admin agent")), {"beta-docs"}
        )

    def test_no_collections_is_still_valid(self):
        response = self.client.post(reverse(AGENT_ADD), self.create_payload(self.alpha))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.assigned(AgentProfile.objects.get()), set())

    def test_missing_scope_reports_the_scope_and_creates_nothing(self):
        response = self.client.post(
            reverse(AGENT_ADD),
            self.agent_payload(knowledge_collections=[str(self.in_alpha.pk)]),
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("application_scope", self.form_errors(response))
        self.assertFalse(AgentProfile.objects.exists())
        self.assertFalse(THROUGH.objects.exists())

    def test_unknown_scope_creates_nothing(self):
        response = self.client.post(
            reverse(AGENT_ADD),
            self.agent_payload(
                application_scope="999999",
                knowledge_collections=[str(self.in_alpha.pk)],
            ),
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(AgentProfile.objects.exists())

    def test_unknown_collection_id_is_a_field_error(self):
        response = self.client.post(
            reverse(AGENT_ADD),
            self.agent_payload(
                application_scope=str(self.alpha.pk), knowledge_collections=["999999"]
            ),
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("knowledge_collections", self.form_errors(response))
        self.assertFalse(AgentProfile.objects.exists())

    def test_other_invalid_fields_leave_nothing_behind_with_valid_collections(self):
        response = self.client.post(
            reverse(AGENT_ADD),
            self.create_payload(self.alpha, self.in_alpha, role=""),
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(AgentProfile.objects.exists())
        self.assertFalse(THROUGH.objects.exists())

    def test_three_scopes_each_accept_only_their_own_collections(self):
        gamma = _scope("Gamma")
        in_gamma = collection(gamma, "gamma-docs")

        ok = self.client.post(
            reverse(AGENT_ADD), self.create_payload(gamma, in_gamma, name="gamma-agent")
        )
        bad = self.client.post(
            reverse(AGENT_ADD),
            self.create_payload(gamma, self.in_alpha, self.in_beta, name="mixed-agent"),
        )

        self.assertEqual(ok.status_code, 302)
        self.assertEqual(bad.status_code, 200)
        self.assertTrue(AgentProfile.objects.filter(name="gamma-agent").exists())
        self.assertFalse(AgentProfile.objects.filter(name="mixed-agent").exists())
        message = " ".join(self.form_errors(bad)["knowledge_collections"])
        self.assertIn("alpha-docs", message)
        self.assertIn("beta-docs", message)


class EditAgentTests(AgentCollectionScopeTestCase):
    def setUp(self):
        super().setUp()
        self.agent = AgentProfile.objects.create(
            application_scope=self.alpha,
            name="Existing agent",
            role="Existing role",
            model_config=self.model_config,
        )
        self.agent.knowledge_collections.set([self.in_alpha])

    def post_edit(self, *collections, **overrides):
        return self.client.post(
            reverse(AGENT_CHANGE, args=[self.agent.pk]),
            self.edit_payload(self.agent, *collections, **overrides),
        )

    def test_editing_other_fields_keeps_ownership_and_assignments(self):
        response = self.post_edit(self.in_alpha, role="Edited role")

        self.assertEqual(response.status_code, 302)
        self.agent.refresh_from_db()
        self.assertEqual(self.agent.role, "Edited role")
        self.assertEqual(self.agent.application_scope, self.alpha)
        self.assertEqual(self.assigned(self.agent), {"alpha-docs"})

    def test_adding_a_same_scope_collection(self):
        response = self.post_edit(self.in_alpha, self.in_alpha_2)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.assigned(self.agent), {"alpha-docs", "alpha-faq"})

    def test_removing_a_collection(self):
        response = self.post_edit()

        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.assigned(self.agent), set())

    def test_adding_a_foreign_collection_is_rejected_and_changes_nothing(self):
        response = self.post_edit(self.in_alpha, self.in_beta, role="Should not save")

        self.assertEqual(response.status_code, 200)
        self.assertIn("knowledge_collections", self.form_errors(response))
        self.agent.refresh_from_db()
        self.assertEqual(self.agent.role, "Existing role")
        self.assertEqual(self.assigned(self.agent), {"alpha-docs"})

    def test_a_forged_scope_cannot_legitimize_a_foreign_collection(self):
        """The change form ignores a posted scope; ownership stays Alpha's."""
        response = self.post_edit(
            self.in_alpha, self.in_beta, application_scope=str(self.beta.pk)
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("knowledge_collections", self.form_errors(response))
        self.agent.refresh_from_db()
        self.assertEqual(self.agent.application_scope, self.alpha)
        self.assertEqual(self.assigned(self.agent), {"alpha-docs"})

    def test_a_forged_scope_still_cannot_reassign_ownership(self):
        response = self.post_edit(self.in_alpha, application_scope=str(self.beta.pk))

        self.assertEqual(response.status_code, 302)
        self.agent.refresh_from_db()
        self.assertEqual(self.agent.application_scope, self.alpha)

    def test_unknown_collection_id_on_edit_is_a_field_error(self):
        response = self.client.post(
            reverse(AGENT_CHANGE, args=[self.agent.pk]),
            self.agent_payload(name=self.agent.name, knowledge_collections=["999999"]),
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("knowledge_collections", self.form_errors(response))
        self.assertEqual(self.assigned(self.agent), {"alpha-docs"})


class LegacyCrossScopeAssignmentTests(AgentCollectionScopeTestCase):
    """Rows that predate the validation: never lost silently, always disclosed."""

    def setUp(self):
        super().setUp()
        self.agent = AgentProfile.objects.create(
            application_scope=self.alpha,
            name="Legacy agent",
            role="Has a stale assignment",
            model_config=self.model_config,
        )
        # Raw ORM, exactly how such a row can exist without the Admin.
        self.agent.knowledge_collections.set([self.in_alpha, self.in_beta])

    def post_edit(self, *collections, **overrides):
        return self.client.post(
            reverse(AGENT_CHANGE, args=[self.agent.pk]),
            self.edit_payload(self.agent, *collections, **overrides),
            follow=False,
        )

    def test_the_change_form_warns_and_names_the_foreign_collection(self):
        response = self.client.get(reverse(AGENT_CHANGE, args=[self.agent.pk]))

        self.assertEqual(response.status_code, 200)
        text = " ".join(self.messages(response))
        self.assertIn("beta-docs", text)
        self.assertIn("Remove", text)
        # The row is shown as assigned, not hidden.
        self.assertIn(
            str(self.in_beta.pk),
            [str(pk) for pk in response.context["adminform"].form["knowledge_collections"].value()],
        )

    def test_saving_without_touching_the_legacy_row_keeps_it(self):
        response = self.post_edit(self.in_alpha, self.in_beta, role="Edited anyway")

        self.assertEqual(response.status_code, 302)
        self.agent.refresh_from_db()
        self.assertEqual(self.agent.role, "Edited anyway")
        self.assertEqual(self.assigned(self.agent), {"alpha-docs", "beta-docs"})

    def test_save_and_continue_editing_keeps_it_and_still_warns(self):
        response = self.client.post(
            reverse(AGENT_CHANGE, args=[self.agent.pk]),
            {**self.edit_payload(self.agent, self.in_alpha, self.in_beta), "_continue": "1"},
            follow=True,
        )
        self.assertEqual(self.assigned(self.agent), {"alpha-docs", "beta-docs"})

        text = " ".join(str(m) for m in response.context["messages"])
        self.assertIn("beta-docs", text)

    def test_an_agent_with_a_legacy_row_can_still_be_deactivated(self):
        payload = self.edit_payload(self.agent, self.in_alpha, self.in_beta)
        payload.pop("is_active")

        response = self.client.post(reverse(AGENT_CHANGE, args=[self.agent.pk]), payload)

        self.assertEqual(response.status_code, 302)
        self.agent.refresh_from_db()
        self.assertFalse(self.agent.is_active)

    def test_the_operator_can_remove_the_legacy_row(self):
        response = self.post_edit(self.in_alpha)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.assigned(self.agent), {"alpha-docs"})

    def test_keeping_a_legacy_row_does_not_allow_adding_another_foreign_one(self):
        other = collection(self.beta, "beta-faq")

        response = self.post_edit(self.in_alpha, self.in_beta, other)

        self.assertEqual(response.status_code, 200)
        message = " ".join(self.form_errors(response)["knowledge_collections"])
        self.assertIn("beta-faq", message)
        self.assertNotIn("beta-docs", message)  # the legacy row is not re-blamed
        self.assertEqual(self.assigned(self.agent), {"alpha-docs", "beta-docs"})

    def test_keeping_a_legacy_row_while_adding_a_valid_collection(self):
        response = self.post_edit(self.in_alpha, self.in_alpha_2, self.in_beta)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            self.assigned(self.agent), {"alpha-docs", "alpha-faq", "beta-docs"}
        )

    def test_the_collections_list_is_not_modified_by_viewing_the_form(self):
        self.client.get(reverse(AGENT_CHANGE, args=[self.agent.pk]))

        self.assertEqual(self.assigned(self.agent), {"alpha-docs", "beta-docs"})

    def test_an_agent_without_legacy_rows_gets_no_warning(self):
        clean = AgentProfile.objects.create(
            application_scope=self.alpha, name="Clean agent", role="r",
            model_config=self.model_config,
        )
        clean.knowledge_collections.set([self.in_alpha])

        response = self.client.get(reverse(AGENT_CHANGE, args=[clean.pk]))

        self.assertEqual(self.messages(response), [])


class PermissionTests(AgentCollectionScopeTestCase):
    def test_add_permission_is_still_required(self):
        from django.contrib.auth import get_user_model
        from django.contrib.auth.models import Permission
        from django.test import Client

        user = get_user_model().objects.create_user(
            "viewer", "viewer@example.test", "not-a-real-password", is_staff=True
        )
        user.user_permissions.set(Permission.objects.filter(codename="view_agentprofile"))
        client = Client()
        client.force_login(user)

        response = client.post(
            reverse(AGENT_ADD), self.create_payload(self.alpha, self.in_alpha)
        )

        self.assertEqual(response.status_code, 403)
        self.assertFalse(AgentProfile.objects.exists())

    def _staff(self, *codenames):
        from django.contrib.auth import get_user_model
        from django.contrib.auth.models import Permission
        from django.test import Client

        user = get_user_model().objects.create_user(
            "limited", "limited@example.test", "not-a-real-password", is_staff=True
        )
        user.user_permissions.set(Permission.objects.filter(codename__in=codenames))
        client = Client()
        client.force_login(user)
        return client

    def test_staff_with_add_permission_gets_the_same_validation(self):
        client = self._staff("add_agentprofile", "view_agentprofile")

        bad = client.post(
            reverse(AGENT_ADD), self.create_payload(self.alpha, self.in_beta)
        )
        good = client.post(
            reverse(AGENT_ADD), self.create_payload(self.alpha, self.in_alpha)
        )

        self.assertEqual(bad.status_code, 200)
        self.assertIn("knowledge_collections", self.form_errors(bad))
        self.assertEqual(good.status_code, 302)
        self.assertEqual(
            self.assigned(AgentProfile.objects.get()), {"alpha-docs"}
        )

    def test_staff_with_change_permission_cannot_add_a_foreign_collection(self):
        agent = AgentProfile.objects.create(
            application_scope=self.alpha, name="Guarded", role="r",
            model_config=self.model_config,
        )
        client = self._staff("change_agentprofile", "view_agentprofile")

        response = client.post(
            reverse(AGENT_CHANGE, args=[agent.pk]),
            self.edit_payload(agent, self.in_beta),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.assigned(agent), set())


class RuntimeAuthorizationUnchangedTests(AgentCollectionScopeTestCase):
    def test_a_legacy_cross_scope_row_still_grants_nothing_at_runtime(self):
        agent = AgentProfile.objects.create(
            application_scope=self.alpha, name="Runtime agent", role="r",
            model_config=self.model_config,
        )
        agent.knowledge_collections.set([self.in_alpha, self.in_beta])

        effective = resolve_effective_knowledge_scope(agent)

        self.assertEqual(effective.collection_ids, frozenset({self.in_alpha.pk}))

    def test_an_admin_created_agent_is_authorized_for_exactly_its_own_collections(self):
        self.client.post(
            reverse(AGENT_ADD), self.create_payload(self.alpha, self.in_alpha, self.in_alpha_2)
        )
        agent = AgentProfile.objects.get(name="Admin agent")

        effective = resolve_effective_knowledge_scope(agent)

        self.assertEqual(
            effective.collection_ids,
            frozenset({self.in_alpha.pk, self.in_alpha_2.pk}),
        )

    def test_keeping_a_legacy_row_through_the_admin_does_not_widen_authorization(self):
        agent = AgentProfile.objects.create(
            application_scope=self.alpha, name="Runtime agent", role="r",
            model_config=self.model_config,
        )
        agent.knowledge_collections.set([self.in_alpha, self.in_beta])

        self.client.post(
            reverse(AGENT_CHANGE, args=[agent.pk]),
            self.edit_payload(agent, self.in_alpha, self.in_beta),
        )

        self.assertEqual(
            resolve_effective_knowledge_scope(agent).collection_ids,
            frozenset({self.in_alpha.pk}),
        )
