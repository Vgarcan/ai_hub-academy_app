"""Academy's explicit ApplicationScope for a development installation.

AI Hub Core never creates a scope: a resource without a decided security
boundary is refused, and `require_single_active_scope()` refuses to guess when
there is not exactly one. A fresh Academy database therefore has no scope, and
the seed commands cannot run until a person has decided which application owns
the data they create.

This module is that decision for the Academy *host*. It is deliberately narrow
and lives in the host, not in Core. `setup_dev.py` and the README call it
explicitly through `bootstrap_academy_scope`; nothing calls it implicitly.

Rules, in order:

* Two or more active scopes: refuse. Choosing one would be a guess.
* Exactly one active scope: use it unchanged. It already satisfies the Core
  helper, whoever created it, and a second active scope would break every
  single-scope surface.
* No active scope: create the stable Academy scope, unless that identity is
  already taken by an inactive scope or by a different scope's name. Those are
  refused rather than reactivated or renamed.

It never reactivates, renames, edits or reassigns anything that exists, and the
scope it creates has no embedding-egress permission and no retrieval
configuration (the model defaults).
"""
from dataclasses import dataclass

from django.core.exceptions import ValidationError
from django.core.management.base import CommandError
from django.db import IntegrityError, transaction

from ai_hub.models import ApplicationScope
from ai_hub.services.application_scope import require_single_active_scope

ACADEMY_SCOPE_SLUG = "academy"
ACADEMY_SCOPE_NAME = "AI Hub Academy"
ACADEMY_SCOPE_DESCRIPTION = (
    "Development scope for the AI Hub Academy host application. Created by "
    "`python manage.py bootstrap_academy_scope`."
)


class ScopeBootstrapError(Exception):
    """The installation's scopes are in a state setup refuses to resolve."""


@dataclass(frozen=True)
class ScopeBootstrapResult:
    scope: ApplicationScope
    created: bool


def _describe(scopes) -> str:
    return ", ".join(f"'{scope.slug}'" for scope in scopes)


def ensure_academy_scope() -> ScopeBootstrapResult:
    """Return the single active scope, creating Academy's only when none exists."""
    try:
        return _ensure()
    except IntegrityError:
        # A concurrent setup created it between our read and our write. Decide
        # again from what is now committed; the second pass cannot create.
        return _ensure()


def _ensure() -> ScopeBootstrapResult:
    with transaction.atomic():
        active = list(ApplicationScope.objects.filter(is_active=True).order_by("slug"))

        if len(active) > 1:
            raise ScopeBootstrapError(
                f"This database has {len(active)} active application scopes "
                f"({_describe(active)}), so setup cannot tell which one Academy "
                "belongs to and will not choose. In Admin > Application scopes, "
                "clear 'Is active' on the scopes that do not belong to this "
                "installation, leaving exactly one, then run setup again."
            )

        if len(active) == 1:
            return ScopeBootstrapResult(scope=active[0], created=False)

        taken_slug = ApplicationScope.objects.filter(slug=ACADEMY_SCOPE_SLUG).first()
        if taken_slug is not None:
            raise ScopeBootstrapError(
                f"The '{ACADEMY_SCOPE_SLUG}' application scope exists but is "
                "inactive. Setup will not reactivate a scope that someone "
                "deactivated. If it should be used, tick 'Is active' on it in "
                "Admin > Application scopes, then run setup again."
            )

        taken_name = ApplicationScope.objects.filter(name=ACADEMY_SCOPE_NAME).first()
        if taken_name is not None:
            raise ScopeBootstrapError(
                f"An application scope named '{ACADEMY_SCOPE_NAME}' already "
                f"exists with the slug '{taken_name.slug}' and is inactive. "
                "Setup will not rename or reactivate it. Reactivate it in "
                "Admin > Application scopes, or rename it, then run setup again."
            )

        scope = ApplicationScope.objects.create(
            slug=ACADEMY_SCOPE_SLUG,
            name=ACADEMY_SCOPE_NAME,
            description=ACADEMY_SCOPE_DESCRIPTION,
            is_active=True,
        )
        return ScopeBootstrapResult(scope=scope, created=True)


def resolve_seed_scope() -> ApplicationScope:
    """The scope an Academy seed writes into, or an actionable CommandError.

    This only wraps Core's `require_single_active_scope()`; it adds a pointer to
    the host command and never relaxes the Core rule. Seeds call it before they
    write anything, so a refusal leaves the database as it was.
    """
    try:
        return require_single_active_scope()
    except ValidationError as exc:
        if ApplicationScope.objects.filter(is_active=True).exists():
            # Several are active. Bootstrap would refuse too, so do not send the
            # operator to it; the decision is theirs, in the Admin.
            hint = (
                "In Admin > Application scopes, clear 'Is active' on the scopes "
                "that do not belong to this installation, leaving exactly one."
            )
        else:
            hint = (
                "On a new Academy development database, create the scope first:\n"
                "    python manage.py bootstrap_academy_scope\n"
                "It creates the scope only when none is active and refuses to "
                "guess when the scopes are ambiguous."
            )
        raise CommandError(
            " ".join(exc.messages) + "\n\n" + hint + "\nNothing was written by this seed."
        ) from exc
