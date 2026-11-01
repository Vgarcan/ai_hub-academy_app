"""Isolated public-document retrieval pilot; never representative evidence.

This service only composes the existing governed retrieval and coverage paths.
The management command enforces the separate SQLite pilot database boundary.
"""
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from django.db import transaction

from ai_hub.models import AgentProfile, KnowledgeCollection, KnowledgeDocument, KnowledgeDocumentChunk, KnowledgeChunkEmbedding
from ai_hub.services.chunk_embedding_identity import canonical_chunk_embedding_text, chunk_embedding_fingerprint
from ai_hub.services.embedding_coverage import build_coverage_report, resolve_index_target
from ai_hub.services.hybrid_retrieval import search_hybrid_with_scope
from ai_hub.services.knowledge_authorization import resolve_effective_knowledge_scope
from ai_hub.services.knowledge_retrieval import rank_knowledge_chunks_with_scope
from ai_hub.services.semantic_retrieval import search_semantic_with_scope

FIXTURE = "django-5.2-public-v3"
DOC_TITLE = "Exploratory public Django 5.2 v3"
SCOPE_SLUG = 'exploratory-django-v3'
COLLECTION_NAME = 'Exploratory Django 5.2 v3'
MAX_FILE_BYTES = 4 * 1024 * 1024


class PilotError(ValueError):
    pass


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    with path.open('rb') as handle:
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise PilotError('artifact_too_large')
    return data


@dataclass(frozen=True)
class Package:
    identity: str
    windows: tuple
    cases: tuple


def load_package(directory):
    """Load the local v3 artifacts and reconstruct all windows from pinned bytes."""
    try:
        directory = Path(directory).resolve(strict=True)
        root = directory.parent
        manifest_data = _read(directory / 'manifest-v3.json')
        manifest = json.loads(manifest_data)
        candidate_data = _read(directory / 'candidate-v3.json')
        source_data = _read(root / 'source-manifest.json')
        chunk_data = _read(root / 'draft-chunks.json')
        if (_sha(candidate_data) != manifest['candidate_sha256']
            or _sha(source_data) != manifest['source_manifest_sha256']
            or _sha(chunk_data) != manifest['source_chunk_manifest_sha256']):
            raise PilotError('package_digest_mismatch')
        candidate = json.loads(candidate_data)
        source = json.loads(source_data.decode('utf-8-sig'))
        chunk_manifest = json.loads(chunk_data)
        if (candidate['schema'] != 'public-django-exploratory-v3'
            or candidate['source_manifest_sha256'] != _sha(source_data)
            or candidate['chunk_manifest_sha256'] != _sha(chunk_data)
            or candidate['source_commit'] != source['commit']
            or candidate['source_commit'] != chunk_manifest['source_commit']):
            raise PilotError('package_invalid')
        allowed = {'docs/ref/settings.txt', 'docs/topics/auth/default.txt', 'docs/topics/db/queries.txt'}
        source_rows = {row['path']: row for row in source['files']}
        if not (allowed | {'LICENSE'}).issubset(source_rows):
            raise PilotError('package_invalid')
        lines = {}
        for name in allowed | {'LICENSE'}:
            path = (root / name).resolve(strict=True)
            if not path.is_relative_to(root):
                raise PilotError('package_invalid')
            data = _read(path)
            if len(data) != source_rows[name]['bytes'] or _sha(data) != source_rows[name]['sha256']:
                raise PilotError('source_digest_mismatch')
            if name in allowed:
                lines[name] = data.splitlines(keepends=True)
        windows = []
        keys = set()
        for index, row in enumerate(chunk_manifest['chunks'], 1):
            name, first, last = row['source'], row['line_start'], row['line_end']
            if name not in allowed or not isinstance(first, int) or not isinstance(last, int) or not 1 <= first <= last <= len(lines[name]) or row['key'] in keys:
                raise PilotError('package_invalid')
            raw = b''.join(lines[name][first - 1:last])
            if len(raw) != row['bytes'] or _sha(raw) != row['sha256']:
                raise PilotError('window_digest_mismatch')
            keys.add(row['key'])
            windows.append((f'w{index:03}', raw.decode('utf-8'), row['sha256']))
        cases = tuple((row['case_id'], row['query']) for row in candidate['cases'])
        if len(windows) != 111 or len(cases) != 24 or len({i for i, _ in cases}) != 24 or any(not isinstance(i, str) or not re.fullmatch(r'P[0-9]{3}', i) or not isinstance(q, str) or not q.strip() or len(q) > 1000 for i, q in cases):
            raise PilotError('package_invalid')
        return Package(_sha(candidate_data), tuple(windows), cases)
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PilotError):
            raise
        raise PilotError('package_invalid') from exc


def _target(agent_id, collection_id):
    try:
        agent = AgentProfile.objects.select_related('application_scope__retrieval_embedding_model_config__provider').get(pk=agent_id)
        scope = resolve_effective_knowledge_scope(agent)
        collection = KnowledgeCollection.objects.get(pk=collection_id, application_scope_id=scope.application_scope_id, is_active=True)
        config = agent.application_scope.retrieval_embedding_model_config
        if (scope.is_empty or agent.application_scope.slug != SCOPE_SLUG
            or collection.name != COLLECTION_NAME or not scope.allows(collection.pk)
            or config is None or not config.is_active or not config.provider.is_active):
            raise PilotError('target_not_authorized')
        return agent, scope, collection, config
    except (AgentProfile.DoesNotExist, KnowledgeCollection.DoesNotExist, AttributeError) as exc:
        raise PilotError('target_not_authorized') from exc


def provision(package, agent_id, collection_id):
    """Create the fixture once in an empty explicitly authorized collection."""
    with transaction.atomic():
        _, _, collection, _ = _target(agent_id, collection_id)
        collection = KnowledgeCollection.objects.select_for_update().get(pk=collection.pk)
        if collection.documents.exists():
            raise PilotError('collection_not_empty')
        document = KnowledgeDocument.objects.create(collection=collection, title=DOC_TITLE, status=KnowledgeDocument.Status.ACTIVE, chunk_authority_mode=KnowledgeDocument.ChunkAuthorityMode.EXPLICIT)
        for index, (key, content, _) in enumerate(package.windows, 1):
            KnowledgeDocumentChunk.objects.create(document=document, chunk_index=index, section_title='', content=content, metadata={'exploratory_fixture': FIXTURE, 'window': key})
        # Refuse and roll back even fixture creation if configuration or the
        # resulting mapping cannot be validated.
        preflight(package, agent_id, collection_id)


def preflight(package, agent_id, collection_id):
    agent, scope, collection, config = _target(agent_id, collection_id)
    documents = list(collection.documents.all())
    if (len(documents) != 1 or documents[0].title != DOC_TITLE
        or documents[0].status != KnowledgeDocument.Status.ACTIVE
        or documents[0].curated_text != '' or documents[0].tags != []
        or documents[0].chunk_authority_mode != KnowledgeDocument.ChunkAuthorityMode.EXPLICIT):
        raise PilotError('fixture_mismatch')
    chunks = list(KnowledgeDocumentChunk.objects.filter(document=documents[0]).order_by('chunk_index'))
    if len(chunks) != len(package.windows):
        raise PilotError('fixture_mismatch')
    mapping = {}
    for index, (chunk, (key, content, _)) in enumerate(zip(chunks, package.windows), 1):
        if (chunk.chunk_index != index or chunk.section_title != '' or chunk.content != content
            or chunk.metadata != {'exploratory_fixture': FIXTURE, 'window': key}
            or canonical_chunk_embedding_text(chunk) != unicodedata.normalize('NFC', content.replace('\r\n', '\n').replace('\r', '\n'))):
            raise PilotError('fixture_mismatch')
        mapping[chunk.pk] = (key, chunk_embedding_fingerprint(chunk))
    target = resolve_index_target(scope_ref=scope.application_scope_id, embedding_config_ref=config.pk, collection_ids=(collection.pk,))
    coverage = build_coverage_report(target)
    if coverage.preconditions.contract is None:
        raise PilotError('contract_unavailable')
    e1 = coverage.preconditions.contract.e1
    vector_rows = KnowledgeChunkEmbedding.objects.filter(chunk_id__in=mapping, e1=e1).order_by('chunk_id')
    vector_identity = tuple((row.chunk_id, row.k1, row.e1, row.embedding_model_config_id, row.application_scope_id, row.collection_id, row.vector_dimension, row.vector_format, _sha(bytes(row.vector_bytes))) for row in vector_rows)
    # Endpoint and grants may change without changing e1; keep operational facts
    # in the internal snapshot, never print them in the report.
    provider = config.provider
    config_facts = (config.pk, config.is_active, config.max_input_chars, config.request_timeout_seconds, provider.pk, provider.is_active, provider.provider_type, provider.declared_locality, provider.base_url, provider.api_key_env_var, provider.default_timeout)
    snapshot = (package.identity, scope.application_scope_id, tuple(sorted(scope.collection_ids)), agent.pk, collection.pk, tuple(sorted(mapping.items())), config_facts, e1, coverage.preconditions, coverage.chunks, vector_identity)
    return scope, config, mapping, coverage, snapshot


def report(package, agent_id, collection_id, *, execute=False, reload_package=None):
    scope, config, mapping, coverage, before = preflight(package, agent_id, collection_id)
    window_hashes = {key: digest for key, _, digest in package.windows}
    payload = {'evidence_class': 'public_exploratory_unjudged', 'fixture': FIXTURE, 'candidate_sha256': package.identity, 'e1': coverage.preconditions.contract.e1, 'coverage': coverage.counts(), 'preconditions_ok': coverage.preconditions.ok, 'coverage_complete': coverage.complete, 'executed': False, 'snapshot_status': 'preflight_only', 'windows': [{'window_id': key, 'source_sha256': window_hashes[key], 'chunk_id': pk, 'k1': k1} for pk, (key, k1) in sorted(mapping.items())]}
    if not execute:
        return payload
    if not coverage.preconditions.ok or not coverage.complete:
        raise PilotError('preflight_not_ready')
    cases = []
    for case_id, query in package.cases:
        lexical = rank_knowledge_chunks_with_scope(scope, query=query, collection_ids=(collection_id,), limit=5)
        if lexical.candidates_truncated:
            raise PilotError('lexical_truncated')
        semantic = search_semantic_with_scope(scope, query=query, embedding_model_config=config, collection_id=collection_id, limit=5)
        hybrid = search_hybrid_with_scope(scope, query=query, embedding_model_config=config, collection_id=collection_id, target_ids=(collection_id,), limit=5).result
        if hybrid.degraded:
            raise PilotError('hybrid_degraded')
        def ids(entries):
            result = []
            for entry in entries:
                if entry.chunk_id not in mapping:
                    raise PilotError('foreign_result')
                result.append(mapping[entry.chunk_id][0])
            return result
        cases.append({'case_id': case_id, 'lexical': ids(lexical.candidates), 'exact_semantic': ids(semantic.matches), 'hybrid': ids(hybrid.matches), 'hybrid_mode': hybrid.mode, 'hybrid_degraded': hybrid.degraded})
    try:
        fresh = reload_package() if reload_package else package
        after = preflight(fresh, agent_id, collection_id)[-1]
        if before != after:
            raise PilotError('snapshot_changed')
    except Exception as exc:
        raise PilotError('snapshot_changed') from exc
    payload.update(executed=True, cases=cases, cutoffs=[1, 3, 5], snapshot_status='unchanged_before_after', query_embeddings='separate_semantic_and_hybrid_calls')
    return payload
