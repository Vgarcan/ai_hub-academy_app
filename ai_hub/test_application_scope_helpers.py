"""Shared ApplicationScope helper for tests. Contains no tests of its own.

S-14 makes scope ownership mandatory on `KnowledgeCollection`, `AgentProfile`
and `GameWorkspace`. Rather than repeat scope construction in ~220 places, tests
call `test_scope()` explicitly at each creation site.

This is deliberately NOT a model default, a signal, or a manager override. Every
call site still names `application_scope=` and can be read, grepped and changed
individually - which matters, because S-15's cross-scope tests need call sites
that pass a DIFFERENT scope. A hidden fallback would have made those tests
impossible to write without first undoing it.
"""

from ai_hub.models import ApplicationScope


DEFAULT_TEST_SCOPE_SLUG = "test-scope"


def test_scope(name: str = "Test Scope", *, slug: str | None = None):
    """Return (creating if needed) a scope for test fixtures.

    Idempotent per slug so repeated calls inside one test share one scope,
    which is what almost every existing test wants: a single application whose
    behaviour must be unchanged by S-14.
    """
    resolved_slug = slug or (
        DEFAULT_TEST_SCOPE_SLUG
        if name == "Test Scope"
        else name.strip().lower().replace(" ", "-")[:140]
    )
    scope, _created = ApplicationScope.objects.get_or_create(
        slug=resolved_slug,
        defaults={"name": name, "is_active": True},
    )
    return scope
