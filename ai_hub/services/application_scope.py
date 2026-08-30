"""Explicit ApplicationScope resolution for surfaces that cannot yet select one.

S-14 makes scope ownership mandatory. Several operator surfaces - the Build
Console, the Admin GAME quick-start, the starter demo and the host seed
commands - create root-owned resources but have no scope selector yet, because
building those selectors is UI work outside this slice.

`require_single_active_scope()` is the bridge, and it is deliberately narrow:

  * it never guesses. Zero active scopes raises; two or more raises.
  * it never resolves the legacy migration scope by name. It has no opinion
    about WHICH scope it returns - only that there is exactly one, so no
    isolation question exists to get wrong.
  * it is not a default. Nothing calls it implicitly; every caller names it,
    and every caller is listed in the S-14 record.

The moment an installation has two applications, every surface using this helper
stops working and says why. That is the intended behaviour: refusing is correct,
and silently picking one - or falling back to `legacy-default` - would be exactly
the fail-open mechanism S-14 exists to prevent.

Remove this helper when those surfaces become scope-aware.
"""

from django.core.exceptions import ValidationError

from ai_hub.models import ApplicationScope


SCOPE_SELECTION_REQUIRED = (
    "This installation has {count} active application scopes, so the scope for "
    "a new resource cannot be inferred. This surface does not support scope "
    "selection yet - create the resource through a scope-aware path, or "
    "deactivate the scopes that do not apply."
)

NO_ACTIVE_SCOPE = (
    "No active application scope exists. Every Knowledge collection, agent and "
    "workspace must belong to one, and Core will not create a scope on your "
    "behalf. Create an application scope first."
)


def require_single_active_scope() -> ApplicationScope:
    """Return THE active scope, or raise. Never guesses, never creates one."""
    scopes = list(ApplicationScope.objects.filter(is_active=True)[:2])
    if not scopes:
        raise ValidationError(NO_ACTIVE_SCOPE)
    if len(scopes) > 1:
        total = ApplicationScope.objects.filter(is_active=True).count()
        raise ValidationError(SCOPE_SELECTION_REQUIRED.format(count=total))
    return scopes[0]
