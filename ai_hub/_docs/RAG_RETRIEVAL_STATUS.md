# RAG Retrieval Status

This is the concise, versioned status reference for reusable Core RAG
retrieval. Historical design notes are not an operational source of truth.

## Shipped

- Governed Knowledge lifecycle, explicit collection attachment and
  `ApplicationScope` isolation.
- Separate permission, embedding-contract (`e1`) and chunk-vector identity
  (`k1`) boundaries.
- Local corpus/query embedding execution and portable canonical vector storage.
- Report-first operator coverage and indexing through
  `knowledge_embedding_index`.
- Exact pre-filtered semantic retrieval, reciprocal-rank hybrid fusion and
  reference-first retrieval audit.
- A default-off `search_knowledge_hybrid` Agent tool with bounded excerpts,
  citations, coverage and changed-since-ranking protection.
- PostgreSQL pgvector/HNSW provisioning, rebuild and evaluation as a shadow
  backend.
- Golden-query evaluation and a fail-closed promotion decision contract.

## Activation Contract

Hybrid Agent search requires every condition below:

1. `AI_HUB_HYBRID_KNOWLEDGE_SEARCH_ENABLED=True`.
2. The Agent and its `ApplicationScope` are active.
3. The scope selects an active `retrieval_embedding_model_config` whose provider
   is active.
4. Provider grants and embedding egress policy permit each request.
5. The Agent has at least one active attached Knowledge collection.
6. The execution is not a GAME context.

The corpus is never indexed implicitly. Operators inspect and execute coverage
explicitly:

```text
python manage.py knowledge_embedding_index --scope <slug|id> --embedding-config <name|id>
python manage.py knowledge_embedding_index --scope <slug|id> --embedding-config <name|id> --execute
```

When the semantic branch is unavailable, governed hybrid retrieval may degrade
to lexical results and reports that mode to the Agent. It never exposes the
provider, model, vector-space fingerprint, failure category or retrieval run id
to the model.

## Deliberate Boundaries

- `search_knowledge` remains lexical and unchanged.
- `search_knowledge_hybrid` is not available to GAME; convergence needs its own
  approved migration.
- pgvector/HNSW is shadow-only. No runtime selector promotes it.
- External embedding execution is absent; the current execution path accepts
  providers declared local only.
- No relevance threshold, no-answer policy or reranker is claimed.
- Retrieved content has no dedicated prompt-injection trust boundary beyond
  the existing Knowledge authorization boundary.
- The reusable Core does not provide automatic file parsing/chunking.

## Evidence Still Required

Before broad production enablement:

1. Build a versioned, representative judged corpus including semantic
   paraphrases, exact identifiers, noisy natural language, no-answer cases and
   cross-scope isolation attempts.
2. Measure exact/hybrid/ANN behavior with the existing golden-query evaluator.
3. Decide lexical semantics and no-answer thresholds from those results.
4. Re-run the evaluation after any lexical change.
5. Promote an ANN backend only through a separate reviewed runtime decision.
6. Add a retrieved-content trust boundary and complete GAME/Path 1 convergence
   before making hybrid retrieval a universal default.

## Verification

The CI contract covers SQLite on Django 5.2 LTS and current Django, plus
PostgreSQL 16 with pgvector. A change is not closed until all three jobs pass.
