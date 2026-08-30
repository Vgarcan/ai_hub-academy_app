"""S-24: the PostgreSQL/pgvector ANN mirror. Backend-only; no Django model state.

Everything created here is PostgreSQL-specific and DERIVED. `KnowledgeChunkEmbedding`
remains the canonical, portable, source-of-truth vector store; these tables are a
rebuildable index representation of it, and `models.py` is deliberately unchanged
so no domain model becomes coupled to a PostgreSQL-only column type.

On SQLite this migration is a NO-OP, which is what keeps the reference retriever
portable and the SQLite CI leg meaningful.

**Extension ownership.** The forward direction may run `CREATE EXTENSION IF NOT
EXISTS vector`, but the reverse direction deliberately does NOT drop it. The
extension is database infrastructure that other applications in the same database
may also be using; AI Hub owns its own tables, functions and triggers and removes
exactly those. Dropping a shared extension to undo one application's migration
would be taking something that was never ours.

**Invalidation is done by database triggers, not Django signals.** A signal is
bypassed by `queryset.update()`, by `bulk_create`, by raw SQL and by any future
management command. A missed invalidation means a stale HNSW graph keeps
consuming finite ANN candidate slots while reporting itself ready - which is the
one failure mode an approximate index makes invisible.

**No referential dependency on the canonical store.** `source_embedding_id`
carries no FOREIGN KEY, and that is deliberate rather than an omission.

CI #53 proved why. Django's `flush` - which `TransactionTestCase` runs, and which
any operator may run - issues `TRUNCATE` over the tables Django MANAGES. These
ANN tables are outside Django's model state by design, so they are never in that
set. PostgreSQL then refuses the whole statement:

    cannot truncate a table referenced in a foreign key constraint
    Table "ai_hub_pgvector_ann_embedding" references
    "ai_hub_knowledgechunkembedding".

The truncate fails, rows survive, and every later test collides on duplicate
keys. A derived index mirror that Django does not manage must never become a
truncation dependency of a table Django does manage; that inverts which of the
two is subordinate.

Nothing is weakened by removing it. Source existence is checked where it
actually matters - at the retrieval boundary, where every ANN candidate id is
re-loaded from `KnowledgeChunkEmbedding` and an unresolvable one REFUSES the
whole search rather than being dropped. A schema constraint would have caught
the same corruption later and at the cost of breaking the ORM's lifecycle.
"""

from django.db import migrations

ANN_PARENT_TABLE = "ai_hub_pgvector_ann_embedding"
ANN_GENERATION_TABLE = "ai_hub_pgvector_ann_generation"
ANN_LEAF_STATE_TABLE = "ai_hub_pgvector_ann_leaf_state"


FORWARD_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

-- The ANN mirror. Partitioned by ApplicationScope, then Collection, then e1,
-- so every HNSW graph belongs to exactly one authorized namespace. No leaf and
-- no index is created here: leaves are provisioned explicitly per
-- (scope, collection, e1) by ai_hub/services/pgvector_ann.py.
--
-- Deliberately holds no Knowledge text, no title, no metadata, no query and no
-- JSON. An index mirror that accumulated content would become a second, unaudited
-- copy of the corpus.
--
-- `source_embedding_id` deliberately carries NO FOREIGN KEY. See the
-- "No referential dependency" section of the module docstring: a derived mirror
-- that Django does not manage must never become a truncation dependency of a
-- table Django DOES manage, or `flush` stops working on PostgreSQL.
--
-- Source existence is validated where it matters - at the retrieval boundary,
-- against `KnowledgeChunkEmbedding` - not by a schema constraint that outranks
-- the ORM's own lifecycle.
CREATE TABLE IF NOT EXISTS {ANN_PARENT_TABLE} (
    source_embedding_id  BIGINT       NOT NULL,
    application_scope_id BIGINT       NOT NULL,
    collection_id        BIGINT       NOT NULL,
    chunk_id             BIGINT       NOT NULL,
    k1                   VARCHAR(80)  NOT NULL,
    e1                   VARCHAR(80)  NOT NULL,
    embedding            vector       NOT NULL
) PARTITION BY LIST (application_scope_id);

-- Source freshness per (scope, collection). Monotonic: incremented only, never
-- reset and never decremented, so "indexed_generation == generation" is a
-- statement that nothing has happened since, not merely that a number matches.
CREATE TABLE IF NOT EXISTS {ANN_GENERATION_TABLE} (
    application_scope_id BIGINT NOT NULL,
    collection_id        BIGINT NOT NULL,
    generation           BIGINT NOT NULL DEFAULT 1,
    PRIMARY KEY (application_scope_id, collection_id),
    CONSTRAINT ai_hub_pgv_generation_positive CHECK (generation >= 1)
);

-- What each leaf currently claims. Bounded facts only.
CREATE TABLE IF NOT EXISTS {ANN_LEAF_STATE_TABLE} (
    application_scope_id BIGINT       NOT NULL,
    collection_id        BIGINT       NOT NULL,
    e1                   VARCHAR(80)  NOT NULL,
    vector_dimension     INTEGER      NOT NULL,
    distance_metric      VARCHAR(20)  NOT NULL,
    backend_version      VARCHAR(32)  NOT NULL,
    indexed_generation   BIGINT       NOT NULL,
    source_count         INTEGER      NOT NULL,
    rebuilt_at           TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (application_scope_id, collection_id, e1)
);

CREATE OR REPLACE FUNCTION ai_hub_pgv_bump(p_scope BIGINT, p_collection BIGINT)
RETURNS void AS $$
BEGIN
    IF p_scope IS NULL OR p_collection IS NULL THEN
        RETURN;
    END IF;
    INSERT INTO {ANN_GENERATION_TABLE}
        (application_scope_id, collection_id, generation)
    VALUES (p_scope, p_collection, 1)
    ON CONFLICT (application_scope_id, collection_id)
    DO UPDATE SET generation = {ANN_GENERATION_TABLE}.generation + 1;
END;
$$ LANGUAGE plpgsql;

-- Canonical vector state. An INSERT, UPDATE or DELETE changes what a rebuilt
-- leaf would contain. An UPDATE that moves the row between namespaces bumps
-- BOTH, because both graphs are now wrong.
CREATE OR REPLACE FUNCTION ai_hub_pgv_embedding_touch() RETURNS trigger AS $$
BEGIN
    IF (TG_OP = 'DELETE') THEN
        PERFORM ai_hub_pgv_bump(OLD.application_scope_id, OLD.collection_id);
        RETURN OLD;
    END IF;
    IF (TG_OP = 'UPDATE') THEN
        PERFORM ai_hub_pgv_bump(OLD.application_scope_id, OLD.collection_id);
    END IF;
    PERFORM ai_hub_pgv_bump(NEW.application_scope_id, NEW.collection_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ai_hub_pgv_embedding_trg
    ON ai_hub_knowledgechunkembedding;
CREATE TRIGGER ai_hub_pgv_embedding_trg
AFTER INSERT OR UPDATE OR DELETE ON ai_hub_knowledgechunkembedding
FOR EACH ROW EXECUTE FUNCTION ai_hub_pgv_embedding_touch();

-- Chunk identity. `content` and `section_title` are exactly the S-19 `k1`
-- inputs, and `document_id` can move a chunk between collections. Deliberately
-- NOT `metadata` or `token_estimate`: neither changes what a vector represents
-- or where it may be retrieved from, and invalidating on them would make
-- ordinary bookkeeping edits force index rebuilds.
--
-- The WHEN clause compares VALUES, so a `Model.save()` that rewrites every
-- column without changing these three does not bump.
CREATE OR REPLACE FUNCTION ai_hub_pgv_chunk_touch() RETURNS trigger AS $$
DECLARE
    v_old_collection BIGINT;
    v_old_scope      BIGINT;
    v_new_collection BIGINT;
    v_new_scope      BIGINT;
BEGIN
    SELECT c.id, c.application_scope_id INTO v_old_collection, v_old_scope
      FROM ai_hub_knowledgedocument d
      JOIN ai_hub_knowledgecollection c ON c.id = d.collection_id
     WHERE d.id = OLD.document_id;
    SELECT c.id, c.application_scope_id INTO v_new_collection, v_new_scope
      FROM ai_hub_knowledgedocument d
      JOIN ai_hub_knowledgecollection c ON c.id = d.collection_id
     WHERE d.id = NEW.document_id;
    PERFORM ai_hub_pgv_bump(v_old_scope, v_old_collection);
    IF v_new_collection IS DISTINCT FROM v_old_collection THEN
        PERFORM ai_hub_pgv_bump(v_new_scope, v_new_collection);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ai_hub_pgv_chunk_trg ON ai_hub_knowledgedocumentchunk;
CREATE TRIGGER ai_hub_pgv_chunk_trg
AFTER UPDATE OF content, section_title, document_id
ON ai_hub_knowledgedocumentchunk
FOR EACH ROW
WHEN (
    OLD.content IS DISTINCT FROM NEW.content
    OR OLD.section_title IS DISTINCT FROM NEW.section_title
    OR OLD.document_id IS DISTINCT FROM NEW.document_id
)
EXECUTE FUNCTION ai_hub_pgv_chunk_touch();

-- Retrievability. ACTIVE <-> ARCHIVED changes whether a document's vectors may
-- be mirrored at all, and `collection_id` moves every one of its chunks.
CREATE OR REPLACE FUNCTION ai_hub_pgv_document_touch() RETURNS trigger AS $$
DECLARE
    v_old_scope BIGINT;
    v_new_scope BIGINT;
BEGIN
    SELECT c.application_scope_id INTO v_old_scope
      FROM ai_hub_knowledgecollection c WHERE c.id = OLD.collection_id;
    SELECT c.application_scope_id INTO v_new_scope
      FROM ai_hub_knowledgecollection c WHERE c.id = NEW.collection_id;
    PERFORM ai_hub_pgv_bump(v_old_scope, OLD.collection_id);
    IF NEW.collection_id IS DISTINCT FROM OLD.collection_id THEN
        PERFORM ai_hub_pgv_bump(v_new_scope, NEW.collection_id);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ai_hub_pgv_document_trg ON ai_hub_knowledgedocument;
CREATE TRIGGER ai_hub_pgv_document_trg
AFTER UPDATE OF status, collection_id ON ai_hub_knowledgedocument
FOR EACH ROW
WHEN (
    OLD.status IS DISTINCT FROM NEW.status
    OR OLD.collection_id IS DISTINCT FROM NEW.collection_id
)
EXECUTE FUNCTION ai_hub_pgv_document_touch();

-- Collection-level retrievability and namespace ownership.
CREATE OR REPLACE FUNCTION ai_hub_pgv_collection_touch() RETURNS trigger AS $$
BEGIN
    PERFORM ai_hub_pgv_bump(OLD.application_scope_id, OLD.id);
    IF NEW.application_scope_id IS DISTINCT FROM OLD.application_scope_id THEN
        PERFORM ai_hub_pgv_bump(NEW.application_scope_id, NEW.id);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ai_hub_pgv_collection_trg ON ai_hub_knowledgecollection;
CREATE TRIGGER ai_hub_pgv_collection_trg
AFTER UPDATE OF is_active, application_scope_id ON ai_hub_knowledgecollection
FOR EACH ROW
WHEN (
    OLD.is_active IS DISTINCT FROM NEW.is_active
    OR OLD.application_scope_id IS DISTINCT FROM NEW.application_scope_id
)
EXECUTE FUNCTION ai_hub_pgv_collection_touch();
"""


# Removes exactly what AI Hub created. `DROP TABLE ... CASCADE` on the ANN parent
# takes its whole partition subtree with it, leaves and HNSW indexes included.
#
# `DROP EXTENSION vector` is deliberately ABSENT. See the module docstring.
REVERSE_SQL = f"""
DROP TRIGGER IF EXISTS ai_hub_pgv_collection_trg ON ai_hub_knowledgecollection;
DROP TRIGGER IF EXISTS ai_hub_pgv_document_trg ON ai_hub_knowledgedocument;
DROP TRIGGER IF EXISTS ai_hub_pgv_chunk_trg ON ai_hub_knowledgedocumentchunk;
DROP TRIGGER IF EXISTS ai_hub_pgv_embedding_trg ON ai_hub_knowledgechunkembedding;

DROP FUNCTION IF EXISTS ai_hub_pgv_collection_touch();
DROP FUNCTION IF EXISTS ai_hub_pgv_document_touch();
DROP FUNCTION IF EXISTS ai_hub_pgv_chunk_touch();
DROP FUNCTION IF EXISTS ai_hub_pgv_embedding_touch();
DROP FUNCTION IF EXISTS ai_hub_pgv_bump(BIGINT, BIGINT);

DROP TABLE IF EXISTS {ANN_LEAF_STATE_TABLE};
DROP TABLE IF EXISTS {ANN_GENERATION_TABLE};
DROP TABLE IF EXISTS {ANN_PARENT_TABLE} CASCADE;
"""


def forward(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        # SQLite keeps the portable reference retriever and gains nothing here.
        return
    schema_editor.execute(FORWARD_SQL)


def reverse(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    schema_editor.execute(REVERSE_SQL)


class Migration(migrations.Migration):

    dependencies = [
        ("ai_hub", "0029_retrieval_audit_foundation"),
    ]

    operations = [
        # No `state_operations`: these tables are deliberately outside Django's
        # model state, so `makemigrations` has nothing to detect and `models.py`
        # stays free of a PostgreSQL-only column type.
        migrations.RunPython(forward, reverse),
    ]
