"""Run the public-document pilot only in an explicitly marked SQLite database."""
import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from ai_hub.services.exploratory_retrieval import PilotError, load_package, provision, report

DATABASE_NAME = 'ai-hub-exploratory-pilot.sqlite3'
MARKER_NAME = 'ai-hub-exploratory-pilot.marker'
MARKER_TEXT = 'isolated-public-exploratory-pilot-v1'


def require_isolated_database():
    name = str(connection.settings_dict['NAME'])
    path = Path(name)
    if connection.vendor != 'sqlite' or not path.is_absolute() or path.name != DATABASE_NAME or path.is_symlink() or not path.is_file():
        raise PilotError('isolated_database_required')
    marker = path.parent / MARKER_NAME
    if (marker.is_symlink() or not marker.is_file()
        or marker.stat().st_size != len(MARKER_TEXT)
        or marker.read_bytes() != MARKER_TEXT.encode('ascii')):
        raise PilotError('isolation_marker_required')


class Command(BaseCommand):
    help = 'Report or execute an isolated unjudged public-document exploratory retrieval pilot.'

    def add_arguments(self, parser):
        parser.add_argument('--package', required=True)
        parser.add_argument('--agent', required=True, type=int)
        parser.add_argument('--collection', required=True, type=int)
        action = parser.add_mutually_exclusive_group()
        action.add_argument('--provision', action='store_true', help='Create the fixture in the selected empty collection.')
        action.add_argument('--execute', action='store_true', help='Run lexical, exact semantic and hybrid queries.')

    def handle(self, *args, **options):
        try:
            require_isolated_database()
            package = load_package(options['package'])
            if options['provision']:
                provision(package, options['agent'], options['collection'])
            payload = report(package, options['agent'], options['collection'], execute=options['execute'], reload_package=lambda: load_package(options['package']))
        except PilotError as exc:
            raise CommandError(str(exc)) from exc
        except Exception as exc:
            # Provider bodies, paths, database errors and configuration are not
            # safe diagnostics. Fail without emitting a partial result.
            raise CommandError('pilot_failed') from exc
        self.stdout.write(json.dumps(payload, indent=2, sort_keys=True))
