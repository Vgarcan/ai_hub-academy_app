"""
Provide the ApplicationScope that Academy's seed commands need.

AI Hub Core refuses to create or guess a scope, so a fresh database has none.
Run this once after `migrate` and before the seeds:

    python manage.py bootstrap_academy_scope

It is safe to repeat. It creates the 'academy' scope only when the database has
no active scope at all; with exactly one active scope it uses that one
unchanged, and with several, or with an inactive 'academy' scope, it stops and
explains what to fix. It never reactivates, renames or reassigns anything.
"""
from django.core.management.base import BaseCommand, CommandError

from academy.services.scope_bootstrap import ScopeBootstrapError, ensure_academy_scope


class Command(BaseCommand):
    help = (
        "Create Academy's development ApplicationScope when the database has no "
        "active scope; otherwise report the single active one. Idempotent."
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
                f"Using the existing active application scope '{scope.slug}' "
                f"({scope.name}); nothing was changed."
            )
