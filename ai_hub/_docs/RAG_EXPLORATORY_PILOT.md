# Isolated public-document retrieval pilot

This operator-only mechanism exercises existing lexical, exact semantic and
hybrid services on the locally supplied Django 5.2 candidate-v3 package. Its
evidence class is `public_exploratory_unjudged`: rankings are not relevance
judgments, no-answer outcomes, production-quality evidence or ANN-promotion
evidence. It does not accept private representative packages and does not
replace the S-30 representative-evidence protocol or the controlled synthetic
evaluator. No source documents or questions are committed by this mechanism.

## Isolation and setup

Use a **new, dedicated, quiescent SQLite database** named
`ai-hub-exploratory-pilot.sqlite3` in a separate directory. Configure the host's
database settings to point there; the bundled host uses the absolute
`SQLITE_NAME` environment variable. Before running migrations or any other
write, independently verify the selected database path. Initialize that new
database using the normal host migration process; never point these instructions
at an existing development or production database. Place a UTF-8/ASCII file
named `ai-hub-exploratory-pilot.marker` beside the database, containing exactly
`isolated-public-exploratory-pilot-v1` with no newline. The pilot command
refuses other database engines, filenames, missing databases or markers. This
marker and filename are operator safeguards, not proof that a database is safe:
the operator must ensure the directory and database are genuinely isolated.

In this database only, explicitly configure an active local embedding provider,
an `EmbeddingModelConfig`, the applicable provider grant, an active Agent and
an `ApplicationScope` with slug `exploratory-django-v3`. Set the scope's
retrieval embedding configuration and attach an active, empty collection named
`Exploratory Django 5.2 v3` to the Agent. The scope's normal egress and provider
grant rules still apply; no external execution is enabled. Confirm the model's
input length supports all canonical chunks. Keep this database and the local
evidence package quiescent throughout execution.

With the supplied local candidate package, for example:

```text
python manage.py knowledge_exploratory_pilot --package <candidate-v3-directory> --agent <id> --collection <id> --provision
python manage.py knowledge_exploratory_pilot --package <candidate-v3-directory> --agent <id> --collection <id>
python manage.py knowledge_embedding_index --scope exploratory-django-v3 --embedding-config <id> --collection <id>
python manage.py knowledge_embedding_index --scope exploratory-django-v3 --embedding-config <id> --collection <id> --execute
python manage.py knowledge_exploratory_pilot --package <candidate-v3-directory> --agent <id> --collection <id> --execute
```

`--provision` is explicit and refuses a collection containing any document;
it creates one active document with 111 explicitly authored chunks and never
deletes or updates existing Knowledge. It uses blank section titles, preserving
the source windows as content. `--provision` and `--execute` are mutually
exclusive. Without either flag the pilot command only reports coverage and
identity; it makes no provider call and persists nothing. The separate existing
index command makes provider calls and writes vectors only with its own
`--execute` flag. Ensure its host settings still name the isolated database:
that separate command does not enforce the pilot marker.

## Integrity and interpretation

The loader verifies the candidate and source manifests, source bytes and each
window hash. Window IDs `w001` through `w111` correspond to the ordered entries
of `draft-chunks.json`; output maps them to actual chunk IDs, source hashes and
canonical `k1`. File hashes are not `k1`: the latter is computed by the existing
canonical embedding-text contract from the persisted section title and content.
Coverage uses the existing `e1`, current-vector and local-egress checks. Missing,
stale or oversized chunks prevent execution. The report includes `e1` and IDs,
but no query text, content, provider configuration, credentials, scores or
provider response. Treat hashes and identifiers as operational metadata.

Execution requests up to five ranked IDs for each of the three paths; prefixes
of lengths 1, 3 and 5 are the requested cutoffs. Exact semantic and hybrid make
separate query-embedding calls, so they are not asserted to use the same query
vector. No precision, recall, MRR, NDCG or false-positive label is computed.
Incomplete lexical ranking, a foreign result, hybrid degradation, detected
target/configuration/vector drift or changed input invalidates the run and
returns no partial ranking report. The pre/post snapshot checks cannot detect a
transient change that is restored between checks; isolation and quiescence are
still required. Service-level execution is not the Agent-facing retrieval audit
boundary, so this command does not claim to create `RetrievalRun` evidence.

The package contains no independent relevance review. Two distinct qualified
reviewers, a distinct reconciler, an authorized private evidence package and all
other S-30 requirements remain necessary for representative evaluation.
