"""Academy's explicit ApplicationScope.

AI Hub Core never creates a scope: a resource without a decided security
boundary is refused. A fresh Academy database therefore has no scope, and the
seed commands cannot run until the host has decided which scope it owns.

Ownership here is **explicit, never inferred**. Academy owns exactly one scope,
named by the setting `ACADEMY_APPLICATION_SCOPE_SLUG` (default `academy`). The
existence of some other active scope - a different application's, or a legacy
one - is not evidence that it belongs to Academy, and Academy never adopts it.
Core's `require_single_active_scope()` is a creation-time compatibility helper
for Core's own surfaces; Academy does not use it to decide ownership.

One decision path serves the bootstrap command and every Academy seed, so they
cannot disagree:

* The configured scope exists and is active: use it, unchanged.
* It exists but is inactive: refuse. Nothing reactivates a deactivated scope.
* It does not exist, and the identity is the default `academy`:
  - another active scope exists: refuse, naming it. That scope is not Academy's
    and creating a second active scope would break Core's single-scope
    surfaces for whichever application owns it.
  - the name is taken by an inactive scope with another slug: refuse.
  - otherwise (nothing active): create the `academy` scope. It grants no
    embedding egress and has no retrieval configuration.
* It does not exist and the identity was configured explicitly: refuse. A
  scope the operator named is never auto-created.

A legacy installation (rows Core adopted into `legacy-default`, or any scope an
operator knows is Academy's) is supported by one explicit declaration:
`ACADEMY_APPLICATION_SCOPE_SLUG=<that slug>` in `.env`.

Whatever the scope, Academy-seeded resources that already exist in a different
scope are reported as a conflict instead of being reused or reassigned. Nothing
here renames, deletes, reactivates or reassigns existing data.
"""
import re
from dataclasses import dataclass, field

from django.conf import settings
from django.core.management.base import CommandError
from django.db import IntegrityError, transaction

from ai_hub.models import AgentProfile, ApplicationScope, KnowledgeCollection

DEFAULT_SCOPE_SLUG = "academy"
ACADEMY_SCOPE_SLUG = DEFAULT_SCOPE_SLUG  # the identity used when none is configured
ACADEMY_SCOPE_NAME = "AI Hub Academy"
ACADEMY_SCOPE_DESCRIPTION = (
    "Development scope for the AI Hub Academy host application. Created by "
    "`python manage.py bootstrap_academy_scope`."
)
SETTING_NAME = "ACADEMY_APPLICATION_SCOPE_SLUG"

# Resources Academy's seed commands create by name. If one already exists in a
# scope other than Academy's, the seeds would silently reuse it across the
# boundary, so it is a conflict.
# The one name shared by `seed_lab_exercises --create-evaluator` (which creates
# it) and the runtime lookup in `lab_evaluator` (which uses it). It lives here, a
# module with no Academy imports, so both can import it without a cycle.
LAB_EVALUATOR_AGENT_NAME = "Lab Exercise Evaluator"

ACADEMY_AGENT_NAMES = (
    "AI Hub Documentation Assistant",
    "Input Normalizer",
    "Ticket Classifier",
    "Documentation Sync Agent",
    LAB_EVALUATOR_AGENT_NAME,
)
ACADEMY_COLLECTION_NAMES = ("AI Hub Official Docs",)

_SLUG = re.compile(r"[-a-zA-Z0-9_]{1,140}")


class ScopeBootstrapError(Exception):
    """The scopes are in a state Academy setup refuses to resolve."""


@dataclass(frozen=True)
class ScopeBootstrapResult:
    scope: ApplicationScope
    created: bool
    # Other active scopes that exist alongside Academy's. Reported, never touched.
    other_active_slugs: tuple = field(default_factory=tuple)


def configured_scope_slug() -> str:
    """The slug of the one scope Academy owns, and whether it was configured."""
    slug = str(getattr(settings, SETTING_NAME, "") or "").strip() or DEFAULT_SCOPE_SLUG
    if not _SLUG.fullmatch(slug):
        raise ScopeBootstrapError(
            f"{SETTING_NAME} is '{slug}', which is not a valid scope slug "
            "(letters, digits, '-' and '_'). Fix it in .env or unset it."
        )
    return slug


def active_academy_scope() -> ApplicationScope | None:
    """Academy's explicitly configured scope if it exists and is active, else None.

    For RUNTIME lookups, where a refusal must degrade rather than raise. It
    decides nothing the seeds do not: it reads the same configured slug, never
    creates or reactivates a scope, and never falls back to another one - the
    number of active scopes is irrelevant. An invalid configuration is "no scope".
    """
    try:
        slug = configured_scope_slug()
    except ScopeBootstrapError:
        return None
    return ApplicationScope.objects.filter(slug=slug, is_active=True).first()


def _quoted(slugs) -> str:
    return ", ".join(f"'{slug}'" for slug in slugs)


def _foreign_resources(scope: ApplicationScope | None) -> list:
    """Academy-named agents/collections that live in a scope other than `scope`.

    With `scope=None` (Academy's scope does not exist yet) every such resource is
    foreign by definition.
    """
    agents = AgentProfile.objects.filter(name__in=ACADEMY_AGENT_NAMES)
    collections = KnowledgeCollection.objects.filter(name__in=ACADEMY_COLLECTION_NAMES)
    if scope is not None:
        agents = agents.exclude(application_scope=scope)
        collections = collections.exclude(application_scope=scope)
    found = [
        f"agent '{row.name}' (scope '{row.application_scope.slug}')"
        for row in agents.select_related("application_scope")
    ]
    found += [
        f"collection '{row.name}' (scope '{row.application_scope.slug}')"
        for row in collections.select_related("application_scope")
    ]
    return found


def _declare_hint(slug: str) -> str:
    return (
        "If that scope really is Academy's (for example an installation upgraded "
        "from before scopes existed, which Core placed in 'legacy-default'), "
        f"declare it explicitly by adding {SETTING_NAME}={slug} to .env and run "
        "setup again."
    )


def _foreign_scope_error(active) -> ScopeBootstrapError:
    slugs = [scope.slug for scope in active]
    evidence = [
        f"{row.name} in '{row.application_scope.slug}'"
        for row in list(
            AgentProfile.objects.filter(name__in=ACADEMY_AGENT_NAMES).select_related(
                "application_scope"
            )
        )
        + list(
            KnowledgeCollection.objects.filter(
                name__in=ACADEMY_COLLECTION_NAMES
            ).select_related("application_scope")
        )
    ]
    message = (
        f"Academy's own application scope '{DEFAULT_SCOPE_SLUG}' does not exist, but "
        f"another application scope is active ({_quoted(slugs)}). Academy will not "
        "adopt a scope it was not told it owns, and it will not create a second "
        "active scope next to it (that would break single-scope surfaces of the "
        "application that owns it). Nothing was changed.\n\n"
        "Choose one:\n"
        "  * Install Academy in a dedicated development database. The default "
        "SQLite file is one; or point SQLITE_NAME at a new file in .env.\n"
        f"  * {_declare_hint(slugs[0] if len(slugs) == 1 else '<slug>')}"
    )
    if evidence:
        message += (
            "\n\nAcademy-seeded resources already exist here: "
            + "; ".join(evidence[:4])
            + ". That suggests the scope holding them is Academy's, but Academy "
            "does not assume it."
        )
    return ScopeBootstrapError(message)


def ensure_academy_scope() -> ScopeBootstrapResult:
    """Resolve Academy's scope, creating the default one only when it is safe."""
    try:
        return _decide(create=True)
    except IntegrityError:
        # A concurrent setup created it between our read and our write. Decide
        # again from what is now committed; the second pass cannot create.
        return _decide(create=True)


def resolve_seed_scope() -> ApplicationScope:
    """The scope an Academy seed writes into, or an actionable CommandError.

    Same decision as the bootstrap, but it never creates anything. Seeds call it
    before their first write, so a refusal leaves the database as it was.
    """
    try:
        return _decide(create=False).scope
    except ScopeBootstrapError as exc:
        raise CommandError(f"{exc}\n\nNothing was written by this seed.") from exc


def _decide(*, create: bool) -> ScopeBootstrapResult:
    slug = configured_scope_slug()
    with transaction.atomic():
        active = list(ApplicationScope.objects.filter(is_active=True).order_by("slug"))
        scope = ApplicationScope.objects.filter(slug=slug).first()

        if scope is not None:
            if not scope.is_active:
                raise ScopeBootstrapError(
                    f"The '{slug}' application scope exists but is inactive. Setup "
                    "will not reactivate a scope that someone deactivated. If it "
                    "should be used, tick 'Is active' on it in Admin > Application "
                    "scopes, then run setup again."
                )
            conflicts = _foreign_resources(scope)
            if conflicts:
                raise ScopeBootstrapError(
                    f"Academy's scope is '{slug}', but Academy-seeded resources "
                    "already exist in a different scope: "
                    + "; ".join(conflicts)
                    + ". Setup will not reuse or reassign them across the scope "
                    "boundary. Move them deliberately in the Admin, or declare the "
                    f"scope that really holds them with {SETTING_NAME}=<slug> in "
                    ".env, then run setup again."
                )
            return ScopeBootstrapResult(
                scope=scope,
                created=False,
                other_active_slugs=tuple(s.slug for s in active if s.pk != scope.pk),
            )

        # The configured scope does not exist.
        if slug != DEFAULT_SCOPE_SLUG:
            raise ScopeBootstrapError(
                f"{SETTING_NAME} names the scope '{slug}', but no such application "
                "scope exists. A scope you name is never created for you. Fix the "
                "slug in .env, or create it deliberately in Admin > Application "
                "scopes."
            )

        if active:
            raise _foreign_scope_error(active)

        # Nothing is active, but Academy-named resources may sit in an inactive
        # scope. Creating `academy` beside them would only make the seeds refuse
        # afterwards and leave a scope behind, so refuse first and create nothing.
        conflicts = _foreign_resources(None)
        if conflicts:
            raise ScopeBootstrapError(
                f"Academy's own application scope '{DEFAULT_SCOPE_SLUG}' does not "
                "exist, but Academy-seeded resources already exist in another "
                "(inactive) scope: "
                + "; ".join(conflicts)
                + ". Setup will not create a new scope beside them, and will not "
                "reactivate, rename or reassign anything. If that scope really is "
                "Academy's, reactivate it in Admin > Application scopes and "
                f"declare it with {SETTING_NAME}=<slug> in .env, then run setup "
                "again."
            )

        taken_name = ApplicationScope.objects.filter(name=ACADEMY_SCOPE_NAME).first()
        if taken_name is not None:
            raise ScopeBootstrapError(
                f"An application scope named '{ACADEMY_SCOPE_NAME}' already exists "
                f"with the slug '{taken_name.slug}' and is inactive. Setup will not "
                "rename or reactivate it. Reactivate it in Admin > Application "
                "scopes and declare it with "
                f"{SETTING_NAME}={taken_name.slug}, or rename it, then run setup "
                "again."
            )

        if not create:
            raise ScopeBootstrapError(
                f"The '{slug}' application scope does not exist yet. Create it "
                "first:\n    python manage.py bootstrap_academy_scope"
            )

        scope = ApplicationScope.objects.create(
            slug=slug,
            name=ACADEMY_SCOPE_NAME,
            description=ACADEMY_SCOPE_DESCRIPTION,
            is_active=True,
        )
        return ScopeBootstrapResult(scope=scope, created=True)
