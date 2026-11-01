"""Security and evidence boundaries for the isolated exploratory pilot."""
import json
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, TestCase

from ai_hub.models import KnowledgeDocumentChunk
from ai_hub.services.exploratory_retrieval import Package, PilotError, load_package, preflight, provision, report
from ai_hub.test_retrieval_audit import AuditFixtureMixin, TRANSPORT_PATH


class PackageAndIsolationTests(SimpleTestCase):
    def test_marked_database_guard_requires_exact_identity(self):
        from ai_hub.management.commands.knowledge_exploratory_pilot import require_isolated_database
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / 'ai-hub-exploratory-pilot.sqlite3'
            database.touch()
            fake = SimpleNamespace(vendor='sqlite', settings_dict={'NAME': str(database)})
            with mock.patch('ai_hub.management.commands.knowledge_exploratory_pilot.connection', fake):
                with self.assertRaisesMessage(PilotError, 'isolation_marker_required'):
                    require_isolated_database()
                (root / 'ai-hub-exploratory-pilot.marker').write_bytes(b'isolated-public-exploratory-pilot-v1')
                require_isolated_database()
                fake.vendor = 'postgresql'
                with self.assertRaisesMessage(PilotError, 'isolated_database_required'):
                    require_isolated_database()

    def test_package_detects_source_tampering(self):
        digest = lambda value: hashlib.sha256(value).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = root / 'candidate-v3'
            candidate_dir.mkdir()
            names = ['docs/ref/settings.txt', 'docs/topics/auth/default.txt', 'docs/topics/db/queries.txt']
            data = b'public source\n'
            files = []
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                files.append({'path': name, 'bytes': len(data), 'sha256': digest(data)})
            (root / 'LICENSE').write_bytes(data)
            files.append({'path': 'LICENSE', 'bytes': len(data), 'sha256': digest(data)})
            source = json.dumps({'commit': 'pinned', 'files': files}).encode()
            (root / 'source-manifest.json').write_bytes(source)
            rows = [{'source': names[i % 3], 'key': f'key{i}', 'line_start': 1, 'line_end': 1, 'bytes': len(data), 'sha256': digest(data)} for i in range(111)]
            chunks = json.dumps({'source_commit': 'pinned', 'chunks': rows}).encode()
            (root / 'draft-chunks.json').write_bytes(chunks)
            candidate = json.dumps({'schema': 'public-django-exploratory-v3', 'source_commit': 'pinned', 'source_manifest_sha256': digest(source), 'chunk_manifest_sha256': digest(chunks), 'cases': [{'case_id': f'P{i:03}', 'query': 'query'} for i in range(1, 25)]}).encode()
            (candidate_dir / 'candidate-v3.json').write_bytes(candidate)
            (candidate_dir / 'manifest-v3.json').write_text(json.dumps({'candidate_sha256': digest(candidate), 'source_manifest_sha256': digest(source), 'source_chunk_manifest_sha256': digest(chunks)}))
            self.assertEqual(len(load_package(candidate_dir).windows), 111)
            (root / names[0]).write_bytes(b'tampered\n')
            with self.assertRaisesMessage(PilotError, 'source_digest_mismatch'):
                load_package(candidate_dir)


class ExploratoryPilotTests(AuditFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()
        self.scope_a.slug = 'exploratory-django-v3'
        self.scope_a.retrieval_embedding_model_config = self.config
        self.scope_a.save(update_fields=['slug', 'retrieval_embedding_model_config'])
        self.collection = self._collection(self.scope_a, 'Exploratory Django 5.2 v3')
        self.agent_a.knowledge_collections.add(self.collection)
        self.package = Package('candidate-digest', (('w001', 'alpha widget', 'source-hash'), ('w002', 'other content', 'source-hash')), (('P001', 'alpha widget'),))
        provision(self.package, self.agent_a.pk, self.collection.pk)
        self.chunks = list(KnowledgeDocumentChunk.objects.filter(document__collection=self.collection).order_by('chunk_index'))

    def index_fixture(self):
        for chunk in self.chunks:
            self.index(chunk, (1.0, 0.0, 0.0, 0.0))

    def test_report_is_read_only_without_provider_and_never_contains_content(self):
        with mock.patch(TRANSPORT_PATH) as transport:
            result = report(self.package, self.agent_a.pk, self.collection.pk)
        self.assertFalse(result['executed'])
        self.assertFalse(result['coverage_complete'])
        self.assertEqual(result['coverage']['missing'], 2)
        self.assertNotIn('alpha widget', json.dumps(result))
        transport.assert_not_called()

    def test_provision_refuses_nonempty_and_foreign_collection(self):
        with self.assertRaisesMessage(PilotError, 'collection_not_empty'):
            provision(self.package, self.agent_a.pk, self.collection.pk)
        with self.assertRaisesMessage(PilotError, 'target_not_authorized'):
            provision(self.package, self.agent_a.pk, self.coll_b1.pk)

    def test_revoked_collection_access_is_refused(self):
        self.agent_a.knowledge_collections.remove(self.collection)
        with self.assertRaisesMessage(PilotError, 'target_not_authorized'):
            report(self.package, self.agent_a.pk, self.collection.pk)

    def test_cross_scope_assignment_cannot_contribute_to_pilot(self):
        self.index_fixture()
        self.agent_a.knowledge_collections.add(self.coll_b1)
        with mock.patch(TRANSPORT_PATH, return_value=self.transport_returning()):
            result = report(self.package, self.agent_a.pk, self.collection.pk, execute=True)
        self.assertNotIn(self.foreign.pk, {row['chunk_id'] for row in result['windows']})
        self.assertTrue(set(result['cases'][0]['hybrid']) <= {'w001', 'w002'})

    def test_execution_uses_existing_paths_without_relevance_claims(self):
        self.index_fixture()
        transport = self.transport_returning()
        with mock.patch(TRANSPORT_PATH, return_value=transport):
            result = report(self.package, self.agent_a.pk, self.collection.pk, execute=True)
        self.assertTrue(result['executed'])
        self.assertEqual(result['cases'][0]['case_id'], 'P001')
        self.assertTrue(result['cases'][0]['exact_semantic'])
        self.assertTrue(result['cases'][0]['hybrid'])
        self.assertEqual(len(transport.calls), 2)
        self.assertNotIn('alpha widget', json.dumps(result))
        self.assertNotIn('relevance', json.dumps(result))

    def test_coverage_and_corpus_drift_refuse_before_queries(self):
        with mock.patch(TRANSPORT_PATH) as transport:
            with self.assertRaisesMessage(PilotError, 'preflight_not_ready'):
                report(self.package, self.agent_a.pk, self.collection.pk, execute=True)
        transport.assert_not_called()
        self.chunks[0].section_title = 'changed'
        self.chunks[0].save(update_fields=['section_title'])
        with self.assertRaisesMessage(PilotError, 'fixture_mismatch'):
            preflight(self.package, self.agent_a.pk, self.collection.pk)

    def test_mid_run_change_never_returns_results(self):
        self.index_fixture()
        def change():
            KnowledgeDocumentChunk.objects.filter(pk=self.chunks[0].pk).update(content='changed')
        transport = self.transport_returning(hook=change)
        with mock.patch(TRANSPORT_PATH, return_value=transport):
            with self.assertRaisesMessage(PilotError, 'snapshot_changed'):
                report(self.package, self.agent_a.pk, self.collection.pk, execute=True)

    def test_changed_input_package_never_returns_results(self):
        self.index_fixture()
        altered = Package('different-digest', self.package.windows, self.package.cases)
        with mock.patch(TRANSPORT_PATH, return_value=self.transport_returning()):
            with self.assertRaisesMessage(PilotError, 'snapshot_changed'):
                report(self.package, self.agent_a.pk, self.collection.pk, execute=True, reload_package=lambda: altered)

    def test_degraded_hybrid_rejected_without_aggregate(self):
        self.index_fixture()
        from ai_hub.services.embedding_client import EmbeddingProviderExecutionError, ErrorCategory
        good_transport = self.transport_returning()
        calls = []
        def second_call_fails(**kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise EmbeddingProviderExecutionError(ErrorCategory.PROVIDER_UNREACHABLE)
            return good_transport(**kwargs)
        with mock.patch(TRANSPORT_PATH, return_value=second_call_fails):
            with self.assertRaisesMessage(PilotError, 'hybrid_degraded'):
                report(self.package, self.agent_a.pk, self.collection.pk, execute=True)

    def test_command_refuses_unmarked_database(self):
        with self.assertRaisesMessage(CommandError, 'isolated_database_required'):
            call_command('knowledge_exploratory_pilot', package='not-read', agent=self.agent_a.pk, collection=self.collection.pk)


class ExploratoryPilotScaleTests(AuditFixtureMixin, TestCase):
    def test_111_windows_and_24_queries_complete_with_stubbed_vectors(self):
        self.build_corpus()
        self.scope_a.slug = 'exploratory-django-v3'
        self.scope_a.retrieval_embedding_model_config = self.config
        self.scope_a.save(update_fields=['slug', 'retrieval_embedding_model_config'])
        collection = self._collection(self.scope_a, 'Exploratory Django 5.2 v3')
        self.agent_a.knowledge_collections.add(collection)
        package = Package('test-digest', tuple((f'w{i:03}', f'document {i}', f'hash{i}') for i in range(1, 112)), tuple((f'P{i:03}', f'document {i}') for i in range(1, 25)))
        provision(package, self.agent_a.pk, collection.pk)
        for chunk in KnowledgeDocumentChunk.objects.filter(document__collection=collection):
            self.index(chunk, (1.0, 0.0, 0.0, 0.0))
        transport = self.transport_returning()
        with mock.patch(TRANSPORT_PATH, return_value=transport):
            result = report(package, self.agent_a.pk, collection.pk, execute=True)
        self.assertEqual(len(result['cases']), 24)
        self.assertEqual(len(result['windows']), 111)
        self.assertEqual(len(transport.calls), 48)
        self.assertTrue(all(len(item['hybrid']) <= 5 for item in result['cases']))
