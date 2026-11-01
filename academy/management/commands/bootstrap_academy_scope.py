"""
Provide the ApplicationScope that Academy's seed commands need.

AI Hub Core refuses to create or guess a scope, so a fresh database has none.
Run this once after `migrate` and before the seeds:

    python manage.py bootstrap_academy_scope

Academy owns one explicitly named scope (ACADEMY_APPLICATION_SCOPE_SLUG,
default 'academy'). The command creates the default one only when no other scope
is active, reuses it when it already exists, and never adopts another
application's scope. It stops with an explanation when the scope is inactive,
when a different scope is active, or when Academy-seeded resources sit in
another scope. It never reactivates, renames or reassigns anything. Safe to repeat.
"""
from django.core.management.base import BaseCommand, CommandError

from academy.services.scope_bootstrap import ScopeBootstrapError, ensure_academy_scope


class Command(BaseCommand):
    help = (
        "Create Academy's own ApplicationScope when it is safe to, or confirm the "
        "existing one. Never adopts another application's scope. Idempotent."
    )

    def handle(self, *args, **options):
        try:
            result = ensure_academy_scope()
        except ScopeBootstrapError as exc:
            raise CommandError(str(exc)) from exc

        scope = result.scope
        if result.created:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Created application scope '{scope.slug}' ({scope.name}). "
                    "External embedding egress is denied."
                )
            )
        else:
            self.stdout.write(
                f"Using Academy's application scope '{scope.slug}' "
                f"({scope.name}); nothing was changed."
            )
        if result.other_active_slugs:
            others = ", ".join(f"'{slug}'" for slug in result.other_active_slugs)
            self.stdout.write(
                self.style.WARNING(
                    f"Other application scopes are also active ({others}). They "
                    "were not touched, but Core surfaces that resolve a single "
                    "active scope (Build Console, GAME quick-start) will refuse "
                    "while more than one is active."
                )
            )
