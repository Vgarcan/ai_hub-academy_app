"""Versioned, content-free retrieval quality evaluation (S-29).

This is an OPERATOR surface, never an Agent tool.  A suite contains controlled
query text and an exact synthetic corpus.  Report mode validates the suite,
authorization, corpus and embedding coverage without calling a provider.
Execution runs the existing lexical and governed hybrid implementations and
returns metrics and bounded diagnostics only; it never persists a query,
ranking or vector.

The only evidence class accepted by this first contract is
``controlled_candidate``.  Calling a small synthetic suite representative is a
governance decision, not something code may infer from a good score.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ai_hub.models import AgentProfile, KnowledgeCollection
from ai_hub.services.embedding_coverage import (
    build_coverage_report,
    resolve_index_target,
)
from ai_hub.services.hybrid_retrieval import search_hybrid_with_scope
from ai_hub.services.knowledge_authorization import (
    authorized_chunks,
    resolve_effective_knowledge_scope,
)
from ai_hub.services.knowledge_retrieval import rank_knowledge_chunks_with_scope
from ai_hub.services.retrieval_evaluation import (
    EvaluationError,
    is_relevant,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    validate_relevance_grade,
)


QUALITY_CONTRACT_VERSION = "retrieval-quality1"
CONTROLLED_CANDIDATE = "controlled_candidate"
SUPPORTED_EVIDENCE_CLASSES = frozenset({CONTROLLED_CANDIDATE})
MAX_DOCUMENTS = 200
MAX_CASES = 500
MAX_QUERY_CHARS = 1000
MAX_CONTENT_CHARS = 50_000
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


class RetrievalQualityError(ValueError):
    """A bounded refusal.  Messages never contain query or document text."""

    def __init__(self, category: str, message: str = ""):
        self.category = category
        super().__init__(message or category)


@dataclass(frozen=True)
class QualityDocument:
    key: str
    title: str
    content: str
    tags: tuple


@dataclass(frozen=True)
class QualityJudgment:
    document_key: str
    grade: int


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    query_class: str
    query: str
    judgments: tuple


@dataclass(frozen=True)
class RetrievalQualitySuite:
    schema_version: str
    suite_id: str
    evidence_class: str
    key_metadata_field: str
    target_collection_name: str
    limit: int
    cutoffs: tuple
    documents: tuple
    cases: tuple


@dataclass(frozen=True)
class RetrievalQualityPreflight:
    suite: RetrievalQualitySuite
    agent: AgentProfile
    scope: object
    collection: KnowledgeCollection
    embedding_model_config: object
    key_to_chunk_id: dict
    coverage: object

    @property
    def ready(self) -> bool:
        return self.coverage.preconditions.ok and self.coverage.complete


def _object(value, *, category):
    if not isinstance(value, dict):
        raise RetrievalQualityError(category)
    return value


def _exact_keys(value, expected, *, category):
    if set(value) != set(expected):
        raise RetrievalQualityError(category)


def _bounded_text(value, *, category, maximum, identifier=False):
    if not isinstance(value, str):
        raise RetrievalQualityError(category)
    text = value.strip()
    if not text or len(text) > maximum:
        raise RetrievalQualityError(category)
    if identifier and not _IDENTIFIER.fullmatch(text):
        raise RetrievalQualityError(category)
    return text


def suite_from_dict(payload) -> RetrievalQualitySuite:
    """Strictly validate one suite.  Unknown keys are rejected."""
    payload = _object(payload, category="suite_not_an_object")
    _exact_keys(
        payload,
        {
            "schema_version", "suite_id", "evidence_class",
            "key_metadata_field", "target_collection_name", "limit",
            "cutoffs", "documents", "cases",
        },
        category="invalid_suite_fields",
    )
    if payload["schema_version"] != QUALITY_CONTRACT_VERSION:
        raise RetrievalQualityError("unsupported_schema_version")
    evidence_class = payload["evidence_class"]
    if evidence_class not in SUPPORTED_EVIDENCE_CLASSES:
        raise RetrievalQualityError("unsupported_evidence_class")
    suite_id = _bounded_text(
        payload["suite_id"], category="invalid_suite_id", maximum=80,
        identifier=True,
    )
    metadata_field = _bounded_text(
        payload["key_metadata_field"], category="invalid_metadata_field",
        maximum=64,
    )
    if metadata_field != "evaluation_key":
        raise RetrievalQualityError("unsupported_metadata_field")
    collection_name = _bounded_text(
        payload["target_collection_name"],
        category="invalid_collection_name", maximum=140,
    )
    limit = payload["limit"]
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise RetrievalQualityError("invalid_limit")
    cutoffs_raw = payload["cutoffs"]
    if (
        not isinstance(cutoffs_raw, list)
        or not cutoffs_raw
        or any(isinstance(k, bool) or not isinstance(k, int) for k in cutoffs_raw)
    ):
        raise RetrievalQualityError("invalid_cutoffs")
    cutoffs = tuple(cutoffs_raw)
    if tuple(sorted(set(cutoffs))) != cutoffs or cutoffs[-1] > limit or cutoffs[0] < 1:
        raise RetrievalQualityError("invalid_cutoffs")

    documents_raw = payload["documents"]
    if not isinstance(documents_raw, list) or not 1 <= len(documents_raw) <= MAX_DOCUMENTS:
        raise RetrievalQualityError("invalid_documents")
    documents = []
    document_keys = set()
    for raw in documents_raw:
        raw = _object(raw, category="invalid_document")
        _exact_keys(raw, {"key", "title", "content", "tags"}, category="invalid_document_fields")
        key = _bounded_text(raw["key"], category="invalid_document_key", maximum=80, identifier=True)
        if key in document_keys:
            raise RetrievalQualityError("duplicate_document_key")
        document_keys.add(key)
        tags = raw["tags"]
        if (
            not isinstance(tags, list)
            or len(tags) > 20
            or any(not isinstance(tag, str) or not tag.strip() or len(tag) > 80 for tag in tags)
        ):
            raise RetrievalQualityError("invalid_document_tags")
        documents.append(QualityDocument(
            key=key,
            title=_bounded_text(raw["title"], category="invalid_document_title", maximum=180),
            content=_bounded_text(raw["content"], category="invalid_document_content", maximum=MAX_CONTENT_CHARS),
            tags=tuple(tag.strip() for tag in tags),
        ))

    cases_raw = payload["cases"]
    if not isinstance(cases_raw, list) or not 1 <= len(cases_raw) <= MAX_CASES:
        raise RetrievalQualityError("invalid_cases")
    cases = []
    case_ids = set()
    for raw in cases_raw:
        raw = _object(raw, category="invalid_case")
        _exact_keys(raw, {"case_id", "query_class", "query", "judgments"}, category="invalid_case_fields")
        case_id = _bounded_text(raw["case_id"], category="invalid_case_id", maximum=80, identifier=True)
        if case_id in case_ids:
            raise RetrievalQualityError("duplicate_case_id")
        case_ids.add(case_id)
        query_class = raw["query_class"]
        if query_class not in {"answerable", "no_answer"}:
            raise RetrievalQualityError("invalid_query_class")
        query = _bounded_text(raw["query"], category="invalid_query", maximum=MAX_QUERY_CHARS)
        judgments_raw = raw["judgments"]
        if not isinstance(judgments_raw, list):
            raise RetrievalQualityError("invalid_judgments")
        judgments = []
        judged_keys = set()
        for judgment_raw in judgments_raw:
            judgment_raw = _object(judgment_raw, category="invalid_judgment")
            _exact_keys(judgment_raw, {"document_key", "grade"}, category="invalid_judgment_fields")
            document_key = judgment_raw["document_key"]
            if document_key not in document_keys:
                raise RetrievalQualityError("unknown_judgment_document")
            if document_key in judged_keys:
                raise RetrievalQualityError("duplicate_judgment_document")
            judged_keys.add(document_key)
            try:
                grade = validate_relevance_grade(judgment_raw["grade"])
            except EvaluationError as exc:
                raise RetrievalQualityError("invalid_judgment_grade") from exc
            judgments.append(QualityJudgment(document_key=document_key, grade=grade))
        has_relevant = any(is_relevant(judgment.grade) for judgment in judgments)
        if query_class == "answerable" and not has_relevant:
            raise RetrievalQualityError("answerable_case_without_relevant_judgment")
        if query_class == "no_answer" and judgments:
            raise RetrievalQualityError("no_answer_case_has_judgments")
        cases.append(QualityCase(
            case_id=case_id, query_class=query_class, query=query,
            judgments=tuple(judgments),
        ))

    return RetrievalQualitySuite(
        schema_version=QUALITY_CONTRACT_VERSION,
        suite_id=suite_id,
        evidence_class=evidence_class,
        key_metadata_field=metadata_field,
        target_collection_name=collection_name,
        limit=limit,
        cutoffs=cutoffs,
        documents=tuple(documents),
        cases=tuple(cases),
    )


def load_suite(path) -> RetrievalQualitySuite:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetrievalQualityError("suite_unreadable") from exc
    return suite_from_dict(payload)


def resolve_agent(agent_ref) -> AgentProfile:
    text = str(agent_ref if agent_ref is not None else "").strip()
    if not text:
        raise RetrievalQualityError("agent_required")
    lookup = {"pk": int(text)} if text.isdigit() else {"name": text}
    try:
        return AgentProfile.objects.select_related(
            "application_scope", "application_scope__retrieval_embedding_model_config",
            "application_scope__retrieval_embedding_model_config__provider",
        ).get(**lookup)
    except AgentProfile.DoesNotExist as exc:
        raise RetrievalQualityError("agent_not_found") from exc


def preflight_suite(suite: RetrievalQualitySuite, agent_ref) -> RetrievalQualityPreflight:
    """Resolve and validate everything required before any query dispatch."""
    agent = resolve_agent(agent_ref)
    scope = resolve_effective_knowledge_scope(agent)
    if scope.is_empty:
        raise RetrievalQualityError("agent_scope_unavailable")
    config = agent.application_scope.retrieval_embedding_model_config
    if config is None or not config.is_active or not config.provider.is_active:
        raise RetrievalQualityError("retrieval_configuration_unavailable")
    collections = KnowledgeCollection.objects.filter(
        pk__in=scope.collection_ids,
        application_scope_id=scope.application_scope_id,
        is_active=True,
        name=suite.target_collection_name,
    )
    try:
        collection = collections.get()
    except KnowledgeCollection.DoesNotExist as exc:
        raise RetrievalQualityError("target_collection_not_authorized") from exc

    chunks = list(
        authorized_chunks(scope)
        .filter(document__collection_id=collection.pk)
        .select_related("document")
        .order_by("pk")
    )
    expected = {document.key: document for document in suite.documents}
    key_to_chunk_id = {}
    corpus_valid = len(chunks) == len(expected)
    for chunk in chunks:
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        key = metadata.get(suite.key_metadata_field)
        spec = expected.get(key)
        if spec is None or key in key_to_chunk_id:
            corpus_valid = False
            continue
        document = chunk.document
        key_to_chunk_id[key] = chunk.pk
        if (
            document.title != spec.title
            or document.curated_text != spec.content
            or tuple(document.tags) != spec.tags
            or chunk.section_title != spec.title
            or chunk.content != spec.content
        ):
            corpus_valid = False
    if not corpus_valid or set(key_to_chunk_id) != set(expected):
        raise RetrievalQualityError("corpus_does_not_match_suite")

    target = resolve_index_target(
        scope_ref=scope.application_scope_id,
        embedding_config_ref=config.pk,
        collection_ids=(collection.pk,),
    )
    coverage = build_coverage_report(target)
    return RetrievalQualityPreflight(
        suite=suite,
        agent=agent,
        scope=scope,
        collection=collection,
        embedding_model_config=config,
        key_to_chunk_id=key_to_chunk_id,
        coverage=coverage,
    )


def preflight_as_dict(preflight: RetrievalQualityPreflight) -> dict:
    coverage = preflight.coverage
    counts = coverage.counts()
    return {
        "contract_version": QUALITY_CONTRACT_VERSION,
        "suite_id": preflight.suite.suite_id,
        "evidence_class": preflight.suite.evidence_class,
        "executed": False,
        "target": {
            "agent_id": preflight.agent.pk,
            "application_scope_id": preflight.scope.application_scope_id,
            "collection_id": preflight.collection.pk,
            "embedding_model_config_id": preflight.embedding_model_config.pk,
        },
        "preflight": {
            "ready": preflight.ready,
            "corpus_matches": True,
            "document_count": len(preflight.suite.documents),
            "case_count": len(preflight.suite.cases),
            "answerable_case_count": sum(case.query_class == "answerable" for case in preflight.suite.cases),
            "no_answer_case_count": sum(case.query_class == "no_answer" for case in preflight.suite.cases),
            "coverage_complete": coverage.complete,
            "coverage_available": coverage.coverage_available,
            "coverage_counts": counts,
            "preconditions_ok": coverage.preconditions.ok,
            "refusal_codes": list(coverage.preconditions.refusal_codes),
        },
    }


def _first_relevant_rank(ranked_ids, judgments):
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if is_relevant(judgments.get(chunk_id, 0)):
            return rank
    return None


def _path_case(ranked_ids, judgments, cutoffs):
    return {
        "returned_count": len(ranked_ids),
        "first_relevant_rank": _first_relevant_rank(ranked_ids, judgments),
        "mrr": mean_reciprocal_rank(ranked_ids, judgments),
        "cutoffs": {
            str(k): {
                "hit": any(is_relevant(judgments.get(cid, 0)) for cid in ranked_ids[:k]),
                "precision": precision_at_k(ranked_ids, judgments, k),
                "recall": recall_at_k(ranked_ids, judgments, k),
                "ndcg": ndcg_at_k(ranked_ids, judgments, k),
            }
            for k in cutoffs
        },
    }


def _aggregate(case_results, path, cutoffs):
    answerable = [case for case in case_results if case["query_class"] == "answerable"]
    no_answer = [case for case in case_results if case["query_class"] == "no_answer"]
    count = len(answerable)
    return {
        "answerable_cases": count,
        "top1_count": sum(case[path]["first_relevant_rank"] == 1 for case in answerable),
        "hit_at_k_count": {
            str(k): sum(case[path]["cutoffs"][str(k)]["hit"] for case in answerable)
            for k in cutoffs
        },
        "mean_reciprocal_rank": (
            sum(case[path]["mrr"] for case in answerable) / count if count else 0.0
        ),
        "mean_precision_at_k": {
            str(k): (
                sum(case[path]["cutoffs"][str(k)]["precision"] for case in answerable) / count
                if count else 0.0
            )
            for k in cutoffs
        },
        "mean_recall_at_k": {
            str(k): (
                sum(case[path]["cutoffs"][str(k)]["recall"] for case in answerable) / count
                if count else 0.0
            )
            for k in cutoffs
        },
        "mean_ndcg_at_k": {
            str(k): (
                sum(case[path]["cutoffs"][str(k)]["ndcg"] for case in answerable) / count
                if count else 0.0
            )
            for k in cutoffs
        },
        "no_answer_cases": len(no_answer),
        "no_answer_empty_count": sum(case[path]["returned_count"] == 0 for case in no_answer),
    }


def execute_suite(preflight: RetrievalQualityPreflight) -> dict:
    """Run both retrieval paths.  No database writes occur in this service."""
    if not preflight.ready:
        raise RetrievalQualityError("preflight_not_ready")
    suite = preflight.suite
    key_to_id = preflight.key_to_chunk_id
    case_results = []
    for case in suite.cases:
        judgments = {
            key_to_id[judgment.document_key]: judgment.grade
            for judgment in case.judgments
        }
        lexical = rank_knowledge_chunks_with_scope(
            preflight.scope,
            query=case.query,
            collection_ids=(preflight.collection.pk,),
            limit=suite.limit,
        )
        lexical_ids = [candidate.chunk_id for candidate in lexical.candidates]
        hybrid = search_hybrid_with_scope(
            preflight.scope,
            query=case.query,
            embedding_model_config=preflight.embedding_model_config,
            collection_id=preflight.collection.pk,
            limit=suite.limit,
            target_ids=(preflight.collection.pk,),
        ).result
        hybrid_ids = [match.chunk_id for match in hybrid.matches]
        item = {
            "case_id": case.case_id,
            "query_class": case.query_class,
            "lexical": _path_case(lexical_ids, judgments, suite.cutoffs),
            "hybrid": _path_case(hybrid_ids, judgments, suite.cutoffs),
            "hybrid_mode": hybrid.mode,
            "hybrid_degraded": hybrid.degraded,
        }
        case_results.append(item)

    payload = preflight_as_dict(preflight)
    payload["executed"] = True
    payload["metrics"] = {
        "lexical": _aggregate(case_results, "lexical", suite.cutoffs),
        "hybrid": _aggregate(case_results, "hybrid", suite.cutoffs),
    }
    payload["cases"] = case_results
    return payload
