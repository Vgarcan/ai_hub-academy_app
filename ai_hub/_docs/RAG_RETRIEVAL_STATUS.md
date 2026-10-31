# RAG Retrieval Status

This is the concise, versioned status reference for reusable Core RAG
retrieval. Historical design notes are not an operational source of truth.

## Current Checkpoint — 2026-10-31

- S-28 Agent-facing hybrid Knowledge search is merged to `main` at `b91aa7c`
  through PR #15.
- CI #61 passed SQLite on Django 5.2 LTS, SQLite on current Django, and
  PostgreSQL 16 with pgvector 0.8.6.
- The default remains safe: hybrid Agent search is disabled until an operator
  enables the flag and configures a scope.
- A live operator-controlled Ollama endpoint was revalidated with
  `bge-m3:latest`: 1,024 dimensions and the declared L2-normalized contract.
  Endpoint-specific details remain outside versioned documentation.
- **Current work:** S-29 now has a versioned controlled-candidate corpus and a
  report-first evaluation harness. S-30 supplies the representative-evidence
  protocol; the private corpus and independent judgments are still required.
  No lexical behavior, no-answer threshold or ANN routing change is authorized
  by this checkpoint.

## S-29 Controlled Live Baseline — 2026-10-31

This is **controlled candidate evidence**, not representative evidence. It used
10 synthetic but policy-shaped documents, 15 judged answerable queries and 5
explicit no-answer queries against live `bge-m3:latest`. All 10 chunks were
current under one application scope and one 1,024-dimensional cosine/L2
contract.

| Metric | Lexical | Hybrid |
| --- | ---: | ---: |
| Top-1 correct, answerable cases | 7/15 | 11/15 |
| Hit@3, answerable cases | 12/15 | 13/15 |
| Hit@5, answerable cases | 13/15 | 14/15 |
| Mean reciprocal rank | 0.6244 | 0.8022 |
| Correctly empty, no-answer cases | 0/5 | 0/5 |

This initial ad-hoc result supports continuing hybrid evaluation: ranking improved materially,
especially for paraphrased travel expense and removable-media questions. It
also proves that broad production enablement would be premature: neither path
has an abstention policy, and both returned a full result page for every
no-answer query. The versioned checkpoint below supersedes this run as the
reproducible controlled baseline; independent adjudication is still pending.

## S-29 Reproducible Controlled Checkpoint — 2026-10-31

The repository now includes `s29-policy-candidate-v1`, an exact 10-document,
20-case suite whose evidence class is fixed to `controlled_candidate`. The
operator command validates suite schema, Agent authorization, exact corpus
identity and complete current-vector coverage before it can dispatch a query.
Report mode is the default and performs no provider call or write; execute mode
uses the existing governed lexical and hybrid services and persists no
evaluation data.

The first live execution against `bge-m3:latest` produced:

| Metric | Lexical | Hybrid |
| --- | ---: | ---: |
| Top-1 correct, answerable cases | 8/15 | 13/15 |
| Hit@3, answerable cases | 11/15 | 14/15 |
| Hit@5, answerable cases | 14/15 | 15/15 |
| Mean reciprocal rank | 0.6656 | 0.9022 |
| Correctly empty, no-answer cases | 0/5 | 0/5 |

All hybrid cases completed without degradation. The ranking evidence favors
hybrid retrieval on this controlled suite, but the no-answer result is a hard
warning: retrieval currently ranks candidates, it does not establish that the
corpus can answer. These 20 cases do not justify a relevance threshold or broad
production activation.

Reproduce the checkpoint after provisioning the exact suite corpus and its
current embeddings:

```text
python manage.py knowledge_retrieval_evaluate --suite ai_hub/evaluation_corpora/s29_policy_candidate_v1.json --agent <name|id>
python manage.py knowledge_retrieval_evaluate --suite ai_hub/evaluation_corpora/s29_policy_candidate_v1.json --agent <name|id> --execute --json
```

Local verification after adding the harness: 2,192 tests passed on the current
Django line, with 111 environment-specific tests skipped; system checks and the
missing-migration check also passed.

## S-31 Audit Findings — 2026-10-31

This is a read-only code audit of the S-29/S-30 evaluation path. It found no
write call in `retrieval_quality` or its management command, and no examined
authorization widening. The following findings remain open; none changes
runtime retrieval behavior today.

1. **High — suite-wide snapshot drift:** S-29 validates corpus identity and
   vector coverage once before execution, but it does not compare a frozen
   target/configuration snapshot before and after all cases. A mutation between
   two cases can therefore produce an aggregate over more than one corpus or
   retrieval configuration. Treat a run as valid only while the target is
   quiescent. The next implementation must capture and compare target `k1`,
   `e1`, scope configuration and authorization facts, then fail closed on any
   change.
2. **High — no-answer case metrics are misleading at case level:** the shared
   relevance conventions define Recall and NDCG as `1.0` where no relevant
   target exists. S-29 exposes those values per no-answer case even when the
   path returned candidates. Its aggregate correctly reports
   `no_answer_empty_count`, but no consumer may use the per-case Recall/NDCG
   values to assess abstention. The next change should omit undefined relevance
   metrics for no-answer cases and report an explicit false-positive shape.
3. **High — no-answer policy is still absent:** controlled live evidence was
   0/5 correctly empty for both lexical and hybrid. No threshold, reranker or
   evidence-sufficiency rule may be activated from the current evidence.
4. **Medium — unbounded raw suite file read:** schema limits apply after
   `load_suite()` has read and decoded the entire JSON file. Add a conservative
   byte ceiling before parsing so a malformed local operator file cannot cause
   avoidable memory pressure.
5. **Medium — representative adapter remains deliberately unimplemented:** the
   S-30 protocol specifies a private package, but S-29 accepts only controlled
   synthetic suites and cannot yield representative or ANN-promotion evidence.
   Build a separate report-first adapter after the data owner supplies a valid
   package; do not widen S-29 by relabelling its evidence class.

The remediation order is snapshot integrity, no-answer metric semantics,
input-size bound, then the representative adapter. Re-run the controlled suite
and the full matrix after each implementation change.

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
- Versioned, report-first controlled retrieval-quality evaluation through
  `knowledge_retrieval_evaluate`.

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

1. Have the data owner prepare an independently adjudicated representative
   package following
   [`RAG_REPRESENTATIVE_EVIDENCE_PROTOCOL.md`](RAG_REPRESENTATIVE_EVIDENCE_PROTOCOL.md),
   including noisy natural language and cross-scope isolation attempts.
2. Repair S-31 snapshot and no-answer-metric findings before treating a
   controlled run as a stable regression baseline.
3. Measure exact/hybrid/ANN behavior with the existing golden-query evaluator.
4. Decide lexical semantics and no-answer thresholds from those results.
5. Re-run the evaluation after any lexical change.
6. Promote an ANN backend only through a separate reviewed runtime decision.
7. Add a retrieved-content trust boundary and complete GAME/Path 1 convergence
   before making hybrid retrieval a universal default.

## Verification

The CI contract covers SQLite on Django 5.2 LTS and current Django, plus
PostgreSQL 16 with pgvector. A change is not closed until all three jobs pass.
