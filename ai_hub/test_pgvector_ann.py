"""S-24: the PostgreSQL/pgvector HNSW backend.

This module has two halves, and the split is deliberate.

**Backend-agnostic tests run everywhere.** SQL construction, identifier safety,
`e1` validation, the dimension ceiling, the metric/opclass mapping, vendor
refusal and every absence check are pure or vendor-guarded, so they execute on
SQLite and on PostgreSQL alike. They are what makes the injection and naming
contracts real rather than asserted in prose.

**The structural-isolation campaign requires PostgreSQL and is skipped without
it.** Partition pruning, `EXPLAIN` plans, HNSW index catalogs, trigger firing and
operator parity cannot be simulated. On SQLite they SKIP; on PostgreSQL they must
all RUN - a suite that quietly skips them on PostgreSQL would report green while
proving nothing about the one property this slice exists for.

The load-bearing proofs are **`EXPLAIN` plans**, not result lists. "The foreign
document is absent from the results" is passed by a global index that scanned it
and filtered afterwards; only the plan shows whether its graph was opened at all.
"""

import ast
import inspect
import json
import re
from decimal import Decimal
import threading
from unittest import mock, skipUnless

from django.db import connection, connections
from django.test import TestCase, TransactionTestCase

from ai_hub.models import (
    AgentProfile,
    ApplicationScope,
    EmbeddingModelConfig,
    KnowledgeChunkEmbedding,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ModelConfig,
    ProviderConfig,
)
from ai_hub.services import pgvector_ann
from ai_hub.services.embedding_contract import resolve_embedding_contract
from ai_hub.services.knowledge_authorization import resolve_effective_knowledge_scope
from ai_hub.services.pgvector_ann import (
    ANN_CANDIDATE_POOL,
    ANN_GENERATION_TABLE,
    ANN_LEAF_STATE_TABLE,
    ANN_PARENT_TABLE,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    MAX_ANN_RESULTS,
    MAX_HNSW_VECTOR_DIMENSION,
    MIN_PGVECTOR_VERSION,
    PGVECTOR_BACKEND_VERSION,
    METRIC_BACKENDS,
    PgvectorAnnError,
    PgvectorFailureCategory,
    build_ann_candidate_sql,
    e1_digest,
    is_cosine_unscorable,
    leaf_identity,
    leaf_index_sql,
    leaf_partition_sql,
    leaf_readiness,
    provision_pgvector_ann_leaf,
    rebuild_pgvector_ann_leaf,
    render_sql,
    resolve_metric_backend,
    search_pgvector_ann_with_scope,
    validate_e1,
)
from ai_hub.services.semantic_retrieval import (
    cosine_similarity,
    dot_product_similarity,
    euclidean_distance,
)
from ai_hub.services.vector_store import store_chunk_vector

METRIC = EmbeddingModelConfig.DistanceMetric
NORMALIZATION = EmbeddingModelConfig.Normalization
LOCALITY = ProviderConfig.DeclaredLocality

POSTGRES = connection.vendor == "postgresql"
requires_postgres = skipUnless(
    POSTGRES, "pgvector/HNSW structural proofs require PostgreSQL"
)

E1_A = "e1:sha256:" + ("ab" * 32)
E1_B = "e1:sha256:" + ("cd" * 32)

KNOWLEDGE_SECRET = "KNOWLEDGE-SECRET-PGV-5150"


def _migration_module():
    """Import `0030_...` by name. A leading digit is illegal in `import`."""
    import importlib

    return importlib.import_module("ai_hub.migrations.0030_pgvector_ann_foundation")


def code_strings(module_source: str) -> set:
    """Every string LITERAL in the source, minus every docstring.

    These modules explain at length which pgvector features they deliberately do
    not use, so a raw text scan for `halfvec` or `ivfflat` matches their own
    prose. This is the seventh time that trap has appeared across these slices;
    any source assertion here is structural from the first draft.
    """
    tree = ast.parse(module_source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    # f-strings arrive as JoinedStr; their literal halves matter too.
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    literals.add(part.value)
    return {literal for literal in literals if literal not in docstrings}


# ---------------------------------------------------------------------------
# Backend-agnostic: SQL construction and the naming contract
# ---------------------------------------------------------------------------

class IdentifierContractTests(TestCase):
    def test_identifiers_are_built_from_numbers_and_a_digest_only(self):
        identity = leaf_identity(3, 7, E1_A)
        self.assertEqual(identity.scope_partition, "ah_pgv_s3")
        self.assertEqual(identity.collection_partition, "ah_pgv_s3_c7")
        self.assertEqual(identity.leaf_table, f"ah_pgv_s3_c7_e{e1_digest(E1_A)}")
        self.assertEqual(identity.leaf_index, f"{identity.leaf_table}_hnsw")
        for name in (
            identity.scope_partition, identity.collection_partition,
            identity.leaf_table, identity.leaf_index,
        ):
            with self.subTest(name=name):
                self.assertRegex(name, r"^[a-z0-9_]+$")
                self.assertLessEqual(len(name), 63)

    def test_no_operator_supplied_text_can_reach_an_identifier(self):
        """The whole injection surface, closed by construction.

        Identifiers are the one place SQL cannot be parameterized. A scope name,
        collection name, model name or document title is operator free text, so
        none of them is an input here at all - only validated positive integers
        and a hex digest.
        """
        source = inspect.getsource(pgvector_ann.leaf_identity)
        tree = ast.parse(source.lstrip())
        names = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for forbidden in ("name", "slug", "title", "model_name", "provider"):
            self.assertNotIn(forbidden, names)

    def test_a_non_integer_id_is_refused(self):
        for scope_id, collection_id in (
            ("1; DROP TABLE x", 2), (1, "2 OR 1=1"), (0, 2), (1, -3),
            (True, 2), (None, 2), (1, None),
        ):
            with self.subTest(scope=scope_id, collection=collection_id):
                with self.assertRaises(PgvectorAnnError):
                    leaf_identity(scope_id, collection_id, E1_A)

    def test_only_a_canonical_e1_is_accepted(self):
        self.assertEqual(validate_e1(E1_A), E1_A)
        for bad in (
            "", None, 12, "e1:sha256:short",
            "e1:sha256:" + ("AB" * 32),                 # uppercase
            "e1:sha256:" + ("ab" * 32) + "x",           # too long
            "k1:sha256:" + ("ab" * 32),                 # wrong contract
            "e1:sha256:" + ("ab" * 31) + "'; DROP--",
        ):
            with self.subTest(e1=bad):
                with self.assertRaises(PgvectorAnnError) as raised:
                    validate_e1(bad)
                self.assertEqual(
                    raised.exception.category, PgvectorFailureCategory.INVALID_E1
                )

    def test_the_e1_digest_is_bounded_hex(self):
        digest = e1_digest(E1_A)
        self.assertEqual(len(digest), 12)
        self.assertRegex(digest, r"^[0-9a-f]{12}$")

    def test_a_hostile_e1_cannot_terminate_a_sql_literal(self):
        """Defence in depth: validation refuses it, and quoting neuters it.

        The property is not that the text disappears - it stays, as DATA. It is
        that the embedded quote is doubled, so the literal cannot be closed and
        the payload can never become a statement.
        """
        from psycopg import sql as psycopg_sql

        hostile = "e1:sha256:'); DROP TABLE ai_hub_knowledgedocument; --"
        with self.assertRaises(PgvectorAnnError):
            validate_e1(hostile)

        rendered = psycopg_sql.Literal(hostile).as_string(None)
        self.assertTrue(rendered.startswith("'") and rendered.endswith("'"))
        self.assertIn("''", rendered, "the embedded quote is escaped")
        # Exactly two unescaped quotes: the opening and closing delimiters.
        self.assertEqual(rendered.replace("''", "").count("'"), 2)

    def test_two_different_e1_values_yield_different_leaves(self):
        self.assertNotEqual(
            leaf_identity(1, 1, E1_A).leaf_table,
            leaf_identity(1, 1, E1_B).leaf_table,
        )

    def test_the_partition_path_is_scope_then_collection_then_e1(self):
        identity = leaf_identity(4, 9, E1_A)
        scope_ddl = render_sql(pgvector_ann.scope_partition_sql(identity))
        collection_ddl = render_sql(pgvector_ann.collection_partition_sql(identity))
        leaf_ddl = render_sql(leaf_partition_sql(identity, dimension=4))

        self.assertIn(f'PARTITION OF "{ANN_PARENT_TABLE}"', scope_ddl)
        self.assertIn("FOR VALUES IN (4)", scope_ddl)
        self.assertIn("PARTITION BY LIST (collection_id)", scope_ddl)

        self.assertIn('PARTITION OF "ah_pgv_s4"', collection_ddl)
        self.assertIn("FOR VALUES IN (9)", collection_ddl)
        self.assertIn("PARTITION BY LIST (e1)", collection_ddl)

        self.assertIn('PARTITION OF "ah_pgv_s4_c9"', leaf_ddl)
        self.assertIn(f"'{E1_A}'", leaf_ddl)
        self.assertNotIn("PARTITION BY", leaf_ddl, "the e1 leaf is a real table")


class MetricMappingTests(TestCase):
    def test_the_three_metrics_map_to_the_documented_operator_classes(self):
        self.assertEqual(
            {metric: (backend.opclass, backend.operator)
             for metric, backend in METRIC_BACKENDS.items()},
            {
                "cosine": ("vector_cosine_ops", "<=>"),
                "dot_product": ("vector_ip_ops", "<#>"),
                "euclidean": ("vector_l2_ops", "<->"),
            },
        )

    def test_every_declared_s18_metric_has_a_backend(self):
        for metric, _label in METRIC.choices:
            with self.subTest(metric=metric):
                self.assertIsNotNone(resolve_metric_backend(metric))

    def test_an_unknown_metric_never_falls_back_to_cosine(self):
        for bad in ("manhattan", None, "", "vector_cosine_ops"):
            with self.subTest(metric=bad):
                with self.assertRaises(PgvectorAnnError) as raised:
                    resolve_metric_backend(bad)
                self.assertEqual(
                    raised.exception.category,
                    PgvectorFailureCategory.UNSUPPORTED_METRIC,
                )

    def test_the_index_ddl_uses_the_matching_operator_class(self):
        identity = leaf_identity(1, 1, E1_A)
        for metric, opclass in (
            (METRIC.COSINE, "vector_cosine_ops"),
            (METRIC.DOT_PRODUCT, "vector_ip_ops"),
            (METRIC.EUCLIDEAN, "vector_l2_ops"),
        ):
            with self.subTest(metric=metric):
                ddl = render_sql(
                    leaf_index_sql(identity, dimension=8, metric=metric)
                )
                self.assertIn(f"USING hnsw ((embedding::vector(8)) {opclass})", ddl)
                self.assertIn(
                    f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})",
                    ddl,
                )

    def test_the_candidate_query_uses_the_matching_ordering_operator(self):
        for metric, operator in (
            (METRIC.COSINE, "<=>"),
            (METRIC.DOT_PRODUCT, "<#>"),
            (METRIC.EUCLIDEAN, "<->"),
        ):
            with self.subTest(metric=metric):
                statement, _params = build_ann_candidate_sql(
                    application_scope_id=1, collection_ids=(2,), e1=E1_A,
                    dimension=4, metric=metric, query_values=(1.0, 0, 0, 0),
                )
                self.assertIn(f") {operator} (", render_sql(statement))


class CandidateQueryShapeTests(TestCase):
    def test_all_three_partition_keys_are_predicates(self):
        """Scope, collection and e1 all prune. That IS the security boundary."""
        statement, params = build_ann_candidate_sql(
            application_scope_id=5, collection_ids=(11, 12), e1=E1_A,
            dimension=4, metric=METRIC.COSINE, query_values=(1.0, 0, 0, 0),
        )
        rendered = render_sql(statement)
        self.assertIn("application_scope_id = %s", rendered)
        self.assertIn("collection_id = ANY(%s)", rendered)
        self.assertIn("e1 = %s", rendered)
        self.assertEqual(params[0], 5)
        self.assertEqual(params[1], [11, 12])
        self.assertEqual(params[2], E1_A)

    def test_no_knowledge_table_participates_in_candidate_selection(self):
        """A join would push authorization AFTER finite candidate selection."""
        rendered = render_sql(
            build_ann_candidate_sql(
                application_scope_id=1, collection_ids=(2,), e1=E1_A,
                dimension=4, metric=METRIC.COSINE, query_values=(1.0, 0, 0, 0),
            )[0]
        )
        self.assertNotIn("JOIN", rendered.upper())
        for table in (
            "ai_hub_knowledgedocument", "ai_hub_knowledgedocumentchunk",
            "ai_hub_knowledgecollection", "ai_hub_knowledgechunkembedding",
        ):
            self.assertNotIn(table, rendered)

    def test_the_candidate_pool_is_the_reference_value(self):
        self.assertEqual(ANN_CANDIDATE_POOL, 100)
        self.assertEqual(HNSW_EF_SEARCH, 100)
        self.assertEqual(MAX_ANN_RESULTS, 20)
        rendered = render_sql(
            build_ann_candidate_sql(
                application_scope_id=1, collection_ids=(2,), e1=E1_A,
                dimension=4, metric=METRIC.COSINE, query_values=(1.0, 0, 0, 0),
            )[0]
        )
        self.assertIn(f"LIMIT {ANN_CANDIDATE_POOL}", rendered)

    def test_the_query_vector_is_a_bound_parameter(self):
        _statement, params = build_ann_candidate_sql(
            application_scope_id=1, collection_ids=(2,), e1=E1_A,
            dimension=3, metric=METRIC.COSINE, query_values=(1.0, 2.0, 3.0),
        )
        self.assertEqual(params[3], "[1.0,2.0,3.0]")

    def test_iterative_scan_is_never_used_as_a_security_mechanism(self):
        """It is a recall feature. Authorization is structural, or it is nothing."""
        source = inspect.getsource(pgvector_ann)
        tree = ast.parse(source)
        literals = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        for literal in literals - docstrings:
            self.assertNotIn("iterative_scan", literal)


class DimensionCeilingTests(TestCase):
    def test_the_ceiling_is_the_pgvector_hnsw_limit(self):
        self.assertEqual(MAX_HNSW_VECTOR_DIMENSION, 2000)

    def test_a_dimension_inside_the_ceiling_is_accepted(self):
        identity = leaf_identity(1, 1, E1_A)
        for dimension in (1, 768, 1536, 2000):
            with self.subTest(dimension=dimension):
                ddl = render_sql(
                    leaf_index_sql(
                        identity, dimension=dimension, metric=METRIC.COSINE
                    )
                )
                self.assertIn(f"embedding::vector({dimension})", ddl)

    def test_a_dimension_above_the_ceiling_is_refused_never_reduced(self):
        identity = leaf_identity(1, 1, E1_A)
        for bad in (2001, 4096, 0, -1, 1.5, True, None):
            with self.subTest(dimension=bad):
                with self.assertRaises(PgvectorAnnError) as raised:
                    leaf_partition_sql(identity, dimension=bad)
                self.assertEqual(
                    raised.exception.category,
                    PgvectorFailureCategory.DIMENSION_UNSUPPORTED,
                )

    def test_the_leaf_carries_a_database_dimension_check(self):
        """Insertion code is not the only way rows arrive in a table."""
        identity = leaf_identity(1, 1, E1_A)
        statements = [
            render_sql(statement)
            for _name, statement in pgvector_ann.leaf_constraint_sql(
                identity, dimension=16
            )
        ]
        self.assertTrue(
            any("vector_dims(embedding) = 16" in s for s in statements)
        )

    def test_the_leaf_enforces_one_vector_per_source_and_per_chunk(self):
        identity = leaf_identity(1, 1, E1_A)
        pairs = pgvector_ann.leaf_constraint_sql(identity, dimension=4)
        statements = [render_sql(statement) for _name, statement in pairs]
        self.assertTrue(any("UNIQUE (source_embedding_id)" in s for s in statements))
        self.assertTrue(any("UNIQUE (chunk_id)" in s for s in statements))
        # Each constraint is NAMED, so provisioning can consult `pg_constraint`
        # instead of wrapping `ALTER TABLE` in a dollar-quoted `DO` block.
        for name, _statement in pairs:
            with self.subTest(name=name):
                self.assertTrue(name.startswith(identity.leaf_table))
                self.assertRegex(name, r"^[a-z0-9_]+$")


class CosineZeroVectorParityTests(TestCase):
    """The ANN backend must never answer where the exact oracle refuses.

    pgvector does not index zero vectors for cosine distance; S-21's exact
    scorer REFUSES them. Left alone those two facts combine into the worst
    outcome available: the oracle refuses while the backend silently omits the
    chunk and returns a confident, shorter ranking.
    """

    def test_zero_magnitude_is_defined_by_the_s21_oracle_not_re_derived(self):
        """One definition. A second could disagree at the edges."""
        self.assertTrue(is_cosine_unscorable((0.0, 0.0, 0.0, 0.0)))
        self.assertFalse(is_cosine_unscorable((1.0, 0.0, 0.0, 0.0)))
        self.assertFalse(is_cosine_unscorable((-1.0, 0.0, 0.0, 0.0)))
        # Tiny but non-zero has a direction, and S-21 accepts it. An epsilon
        # threshold invented here would refuse it and diverge from the oracle.
        self.assertFalse(is_cosine_unscorable((1e-30, 0.0, 0.0, 0.0)))

    def test_the_predicate_delegates_to_the_canonical_scorer(self):
        source = inspect.getsource(is_cosine_unscorable)
        tree = ast.parse(source.lstrip())
        called = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("cosine_similarity", called)
        for forbidden in ("sqrt", "sum", "abs", "isclose"):
            self.assertNotIn(forbidden, called)

    def test_the_backend_category_reuses_the_s21_string(self):
        from ai_hub.services.semantic_retrieval import RetrievalFailureCategory

        self.assertEqual(
            PgvectorFailureCategory.UNSCORABLE_ZERO_VECTOR,
            RetrievalFailureCategory.UNSCORABLE_ZERO_VECTOR,
        )

    def test_zero_vectors_are_not_banned_for_the_other_metrics(self):
        """Only cosine is undefined for them. Do not over-refuse."""
        self.assertEqual(dot_product_similarity((0.0, 0.0), (1.0, 2.0)), 0.0)
        self.assertAlmostEqual(
            euclidean_distance((0.0, 0.0), (3.0, 4.0)), 5.0, places=12
        )

    def test_the_zero_check_is_applied_only_to_cosine(self):
        for target in (
            pgvector_ann.rebuild_pgvector_ann_leaf,
            pgvector_ann.search_pgvector_ann_with_scope,
        ):
            with self.subTest(target=target.__name__):
                tree = ast.parse(inspect.getsource(target).lstrip())
                guarded = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "is_cosine_unscorable"
                ]
                self.assertEqual(len(guarded), 1)
                self.assertIn('"cosine"', inspect.getsource(target))

    def test_the_query_zero_check_precedes_the_candidate_query(self):
        """Structural: the refusal must be reachable before any ANN SQL."""
        source = inspect.getsource(pgvector_ann.search_pgvector_ann_with_scope)
        tree = ast.parse(source.lstrip())
        lines = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                lines.setdefault(node.func.id, node.lineno)
        self.assertLess(
            lines["is_cosine_unscorable"], lines["build_ann_candidate_sql"]
        )
        self.assertLess(lines["is_cosine_unscorable"], lines["leaf_readiness"])

    def test_the_rebuild_zero_check_precedes_leaf_state_persistence(self):
        source = inspect.getsource(pgvector_ann.rebuild_pgvector_ann_leaf)
        tree = ast.parse(source.lstrip())
        guard = next(
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "is_cosine_unscorable"
        )
        atomic = next(
            node.lineno for node in ast.walk(tree)
            if isinstance(node, (ast.With, ast.AsyncWith))
        )
        self.assertLess(
            guard, atomic,
            "a refused rebuild must not have written leaf state",
        )

    def test_the_backend_never_normalizes_or_substitutes_a_zero_vector(self):
        tree = ast.parse(inspect.getsource(pgvector_ann))
        names = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for forbidden in ("normalize_embedding_vector", "normalize"):
            self.assertNotIn(forbidden, names)


class VendorGuardTests(TestCase):
    @skipUnless(not POSTGRES, "only meaningful on a non-PostgreSQL backend")
    def test_every_backend_entry_point_refuses_on_sqlite(self):
        scope = ApplicationScope.objects.create(name="Guard", slug="guard")
        collection = KnowledgeCollection.objects.create(
            name="Guard Collection", application_scope=scope
        )
        provider = ProviderConfig.objects.create(
            name="Guard P", provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://ollama.internal:11434",
            declared_locality=LOCALITY.LOCAL,
        )
        config = EmbeddingModelConfig.objects.create(
            name="guard-embed", provider=provider,
            model_name="ollama/nomic-embed-text", model_revision="v1",
            vector_dimension=4, distance_metric=METRIC.COSINE,
            normalization=NORMALIZATION.NONE,
        )
        for operation in (
            lambda: provision_pgvector_ann_leaf(
                application_scope=scope, collection=collection,
                embedding_model_config=config,
            ),
            lambda: rebuild_pgvector_ann_leaf(
                application_scope=scope, collection=collection,
                embedding_model_config=config,
            ),
            lambda: pgvector_ann.pgvector_extension_version(),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(PgvectorAnnError) as raised:
                    operation()
                self.assertEqual(
                    raised.exception.category,
                    PgvectorFailureCategory.UNSUPPORTED_DATABASE_VENDOR,
                )

    def test_the_postgresql_campaign_cannot_silently_skip_on_postgresql(self):
        """A suite that skips S-24 on PostgreSQL would report green for nothing.

        The gate is one module-level boolean derived from the live connection,
        so this asserts the gate itself rather than trusting it.
        """
        self.assertEqual(POSTGRES, connection.vendor == "postgresql")
        if connection.vendor == "postgresql":
            self.assertTrue(
                POSTGRES,
                "the structural campaign MUST run on PostgreSQL",
            )
            for name in (
                "ExtensionAndSchemaTests", "PartitionHierarchyTests",
                "GenerationTriggerTests", "RebuildTests", "PlanIsolationTests",
                "AnnSearchTests", "OperatorParityTests",
                "CosineZeroVectorCampaignTests", "ProvisioningConcurrencyTests",
                "MigrationRoundTripTests",
            ):
                case = globals()[name]
                with self.subTest(case=name):
                    self.assertFalse(
                        getattr(case, "__unittest_skip__", False),
                        f"{name} must not be skipped on PostgreSQL",
                    )

    def test_the_minimum_pgvector_version_is_pinned(self):
        self.assertEqual(MIN_PGVECTOR_VERSION, (0, 8, 6))

    def test_the_backend_version_is_named_and_fixed(self):
        self.assertEqual(PGVECTOR_BACKEND_VERSION, "pgv-hnsw1")


class BackendIsolationTests(TestCase):
    """S-24 must not have reached into any established contract."""

    def _identifiers(self, module):
        tree = ast.parse(inspect.getsource(module))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        names |= {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
                names |= {alias.name for alias in node.names}
        return names

    def test_the_backend_never_resolves_authorization(self):
        """It consumes a frozen scope so a composing layer keeps one snapshot."""
        names = self._identifiers(pgvector_ann)
        self.assertNotIn("resolve_effective_knowledge_scope", names)

    def test_the_backend_calls_no_provider_and_never_re_embeds(self):
        names = self._identifiers(pgvector_ann)
        for forbidden in (
            "requests", "resolve_embedding_transport", "embed_text_via_ollama",
            "index_chunk_embedding_local", "canonical_query_embedding_text",
            "normalize_embedding_vector", "chunk_embedding_fingerprint",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_backend_reuses_canonical_inspection_and_decoding(self):
        names = self._identifiers(pgvector_ann)
        self.assertIn("inspect_vector_record", names)
        self.assertIn("decode_vector", names)
        self.assertIn("resolve_metric_scorer", names)

    def test_no_ivfflat_halfvec_or_quantization(self):
        """Scans executable SQL and identifiers, never prose.

        Both modules explain at length WHY these are excluded, so a text scan
        would match their own explanations.
        """
        migration_module = _migration_module()

        executable = {
            literal.lower()
            for literal in code_strings(inspect.getsource(pgvector_ann))
        }
        executable |= {
            migration_module.FORWARD_SQL.lower(),
            migration_module.REVERSE_SQL.lower(),
        }
        for forbidden in (
            "ivfflat", "halfvec", "binary_quantize", "sparsevec", "subvector",
        ):
            for literal in executable:
                with self.subTest(forbidden=forbidden):
                    self.assertNotIn(forbidden, literal)

        names = self._identifiers(pgvector_ann)
        for forbidden in ("ivfflat", "halfvec", "sparsevec", "binary_quantize"):
            self.assertNotIn(forbidden, names)

    def test_no_pgvector_python_dependency_and_no_vector_field(self):
        names = self._identifiers(pgvector_ann)
        for forbidden in ("pgvector", "VectorField", "numpy", "np", "scipy"):
            self.assertNotIn(forbidden, names)

        import pathlib

        requirements = (
            pathlib.Path(__file__).resolve().parent.parent / "requirements.txt"
        ).read_text(encoding="utf-8").lower()
        for forbidden in ("pgvector", "numpy", "scipy", "chromadb", "faiss"):
            self.assertNotIn(forbidden, requirements)

    def test_the_backend_persists_no_query_and_no_audit_row(self):
        names = self._identifiers(pgvector_ann)
        for forbidden in (
            "RetrievalRun", "RetrievalHit", "RetrievalOutcome",
            "RetrievalRunCollection", "retrieval_audit",
            "q1", "query_hash", "hashlib_query",
        ):
            self.assertNotIn(forbidden, names)

    def test_no_automatic_rebuild_signal_or_background_task(self):
        names = self._identifiers(pgvector_ann)
        for forbidden in (
            "post_save", "pre_save", "post_delete", "receiver", "signals",
            "shared_task", "celery", "threading", "schedule",
        ):
            self.assertNotIn(forbidden, names)

    def test_no_backend_autodetection_or_fallback(self):
        names = self._identifiers(pgvector_ann)
        for forbidden in (
            "VECTOR_BACKEND", "settings", "fallback", "prefer_backend",
            "auto_backend", "getenv", "environ",
        ):
            self.assertNotIn(forbidden, names)

    def test_no_threshold_no_answer_or_reranker(self):
        names = self._identifiers(pgvector_ann)
        for forbidden in (
            "threshold", "min_score", "no_answer", "rerank_model",
            "cross_encoder", "litellm", "golden",
        ):
            self.assertNotIn(forbidden, names)

    def test_the_retrieval_chain_does_not_reference_the_ann_backend(self):
        """No runtime cutover. S-22 and S-23 remain reference-backed."""
        from ai_hub.services import (
            hybrid_retrieval,
            knowledge_retrieval,
            knowledge_tooling,
            retrieval_audit,
            semantic_retrieval,
            vector_store,
        )
        from ai_hub.tools import knowledge as knowledge_tools

        for module in (
            semantic_retrieval, hybrid_retrieval, retrieval_audit,
            knowledge_retrieval, vector_store, knowledge_tooling, knowledge_tools,
        ):
            with self.subTest(module=module.__name__):
                names = self._identifiers(module)
                for forbidden in (
                    "pgvector_ann", "search_pgvector_ann_with_scope",
                    "provision_pgvector_ann_leaf", "rebuild_pgvector_ann_leaf",
                ):
                    self.assertNotIn(forbidden, names)

    def test_the_canonical_store_still_writes_f32le1_bytes(self):
        """No hidden dual-write from `store_chunk_vector`."""
        from ai_hub.services import vector_store

        names = self._identifiers(vector_store)
        self.assertNotIn("pgvector_ann", names)
        self.assertIn("encode_vector", names)

    def test_the_reference_oracle_constants_are_unchanged(self):
        from ai_hub.services import semantic_retrieval

        self.assertEqual(
            semantic_retrieval.MAX_REFERENCE_SEMANTIC_CANDIDATES, 1000
        )
        self.assertEqual(
            set(semantic_retrieval.METRIC_SCORERS),
            {METRIC.COSINE, METRIC.DOT_PRODUCT, METRIC.EUCLIDEAN},
        )

    def test_no_admin_registration_for_backend_tables(self):
        from django.contrib import admin

        registered = {model.__name__ for model in admin.site._registry}
        for forbidden in ("PgvectorAnnEmbedding", "RetrievalRun"):
            self.assertNotIn(forbidden, registered)




class MigrationContractTests(TestCase):
    """Asserted against the RESOLVED SQL, not the file text.

    The migration builds its SQL with f-strings, so the file contains
    `{ANN_PARENT_TABLE}` where the database will see the real table name.
    Reading the module's constants tests what actually executes.
    """

    def _sql(self):
        migration_module = _migration_module()

        return migration_module.FORWARD_SQL, migration_module.REVERSE_SQL

    def _source(self):
        import pathlib

        return (
            pathlib.Path(__file__).resolve().parent
            / "migrations" / "0030_pgvector_ann_foundation.py"
        ).read_text(encoding="utf-8")

    def test_the_current_leaf_is_the_pgvector_foundation(self):
        from django.db.migrations.loader import MigrationLoader

        loader = MigrationLoader(None, ignore_no_migrations=True)
        migrations = sorted(
            name for app, name in loader.disk_migrations if app == "ai_hub"
        )
        self.assertEqual(migrations[-1], "0030_pgvector_ann_foundation")
        self.assertEqual(
            len([name for name in migrations if name.startswith("0030")]), 1
        )

    def test_the_parent_is_the_retrieval_audit_foundation(self):
        from django.db.migrations.loader import MigrationLoader

        loader = MigrationLoader(None, ignore_no_migrations=True)
        migration = loader.disk_migrations[("ai_hub", "0030_pgvector_ann_foundation")]
        self.assertEqual(
            migration.dependencies, [("ai_hub", "0029_retrieval_audit_foundation")]
        )

    def test_each_earlier_slice_still_owns_its_own_migration(self):
        """A slice asserts ITS migration exists, not what the leaf happens to be.

        S-21 and S-22 each once carried a `leaf == 0028` assertion, which was
        really a claim about future slices and broke the moment one landed. The
        current-leaf assertion belongs here, with the newest migration.
        """
        from django.db.migrations.loader import MigrationLoader

        loader = MigrationLoader(None, ignore_no_migrations=True)
        names = {name for app, name in loader.disk_migrations if app == "ai_hub"}
        for expected in (
            "0028_knowledge_chunk_embedding",
            "0029_retrieval_audit_foundation",
            "0030_pgvector_ann_foundation",
        ):
            self.assertIn(expected, names)

    def test_the_migration_adds_no_django_model_state(self):
        """`models.py` must stay free of a PostgreSQL-only column type.

        Structural: the migration's own comments discuss `state_operations`
        while explaining its absence, so a text scan matches that prose.
        """
        migration_module = _migration_module()

        operations = migration_module.Migration.operations
        self.assertEqual(len(operations), 1)
        self.assertEqual(type(operations[0]).__name__, "RunPython")
        self.assertFalse(getattr(operations[0], "state_operations", None))

        # Structural again: `models.py` DISCUSSES pgvector in the S-19
        # `KnowledgeChunkEmbedding` docstring, explaining why the canonical
        # store is portable bytes rather than a vector column. Asserting on the
        # actual FIELD TYPES is what proves no PostgreSQL-only column exists.
        from django.apps import apps
        from django.db import models as django_models

        for model in apps.get_app_config("ai_hub").get_models():
            for field in model._meta.get_fields():
                if not getattr(field, "concrete", False):
                    continue
                with self.subTest(model=model.__name__, field=field.name):
                    self.assertTrue(
                        type(field).__module__.startswith("django."),
                        f"{type(field)!r} is not a stock Django field",
                    )
                    self.assertIsInstance(field, django_models.Field)
                    self.assertNotIn("vector", type(field).__name__.lower())

    def test_the_reverse_migration_never_drops_the_extension(self):
        """The extension is shared infrastructure, not AI Hub's to remove."""
        _forward, reverse = self._sql()
        self.assertNotIn("DROP EXTENSION", reverse.upper())
        self.assertIn("DROP TABLE IF EXISTS", reverse)
        self.assertIn("DROP TRIGGER IF EXISTS", reverse)
        self.assertIn("DROP FUNCTION IF EXISTS", reverse)
        for table in (
            ANN_PARENT_TABLE, ANN_GENERATION_TABLE, ANN_LEAF_STATE_TABLE
        ):
            self.assertIn(table, reverse)

    def test_the_forward_migration_creates_the_extension_conditionally(self):
        forward, _reverse = self._sql()
        self.assertIn("CREATE EXTENSION IF NOT EXISTS vector", forward)

    def test_the_migration_is_a_no_op_on_non_postgresql(self):
        source = self._source()
        self.assertIn('schema_editor.connection.vendor != "postgresql"', source)

    def test_the_parent_table_is_partitioned_by_scope(self):
        forward, _reverse = self._sql()
        self.assertIn("PARTITION BY LIST (application_scope_id)", forward)

    def test_the_parent_table_holds_no_knowledge_content(self):
        forward, _reverse = self._sql()
        parent = forward[
            forward.index(f"CREATE TABLE IF NOT EXISTS {ANN_PARENT_TABLE}"):
        ]
        parent = parent[:parent.index(") PARTITION BY")]
        for forbidden in (
            "content", "title", "snippet", "metadata", "tags", "curated_text",
            "JSONB", "JSON", "TEXT",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, parent)

    def test_the_mirror_cascades_from_the_canonical_store(self):
        forward, _reverse = self._sql()
        self.assertIn("REFERENCES ai_hub_knowledgechunkembedding (id)", forward)
        self.assertIn("ON DELETE CASCADE", forward)

    def test_the_generation_column_is_monotonic(self):
        forward, _reverse = self._sql()
        self.assertIn("CHECK (generation >= 1)", forward)
        self.assertIn("generation + 1", forward)
        self.assertNotIn("generation = 1 WHERE", forward)
        self.assertNotIn("generation - 1", forward)

    def test_every_required_trigger_is_installed(self):
        source, _reverse = self._sql()
        for table, trigger in (
            ("ai_hub_knowledgechunkembedding", "ai_hub_pgv_embedding_trg"),
            ("ai_hub_knowledgedocumentchunk", "ai_hub_pgv_chunk_trg"),
            ("ai_hub_knowledgedocument", "ai_hub_pgv_document_trg"),
            ("ai_hub_knowledgecollection", "ai_hub_pgv_collection_trg"),
        ):
            with self.subTest(trigger=trigger):
                self.assertIn(f"CREATE TRIGGER {trigger}", source)
                self.assertIn(table, source)

    def test_the_chunk_trigger_ignores_metadata_and_token_estimate(self):
        """Narrow by column AND by value, so a full-row save does not bump."""
        source, _reverse = self._sql()
        clause = source[source.index("CREATE TRIGGER ai_hub_pgv_chunk_trg"):]
        clause = clause[:clause.index("EXECUTE FUNCTION")]
        self.assertIn("UPDATE OF content, section_title, document_id", clause)
        self.assertNotIn("metadata", clause)
        self.assertNotIn("token_estimate", clause)
        self.assertIn("IS DISTINCT FROM", clause)


# ---------------------------------------------------------------------------
# PostgreSQL: the structural campaign
# ---------------------------------------------------------------------------

class PgvectorFixtureMixin:
    """A two-application corpus whose FORBIDDEN vectors are the best matches."""

    def build_corpus(self, *, metric=METRIC.COSINE, dimension=4, unauthorized_bulk=0):
        chat_provider = ProviderConfig.objects.create(
            name="Chat P", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        self.model_config = ModelConfig.objects.create(
            provider=chat_provider, model_name="training",
            temperature_default=Decimal("0.10"),
        )
        self.scope_a = ApplicationScope.objects.create(name="App A", slug="app-a")
        self.scope_b = ApplicationScope.objects.create(name="App B", slug="app-b")

        self.provider = ProviderConfig.objects.create(
            name="Embed P", provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://ollama.internal:11434",
            declared_locality=LOCALITY.LOCAL,
        )
        self.config = EmbeddingModelConfig.objects.create(
            name="embed-a", provider=self.provider,
            model_name="ollama/nomic-embed-text", model_revision="v1",
            vector_dimension=dimension, distance_metric=metric,
            normalization=NORMALIZATION.NONE,
        )
        # A SECOND vector space in the same collections, to prove e1 pruning.
        self.config_other = EmbeddingModelConfig.objects.create(
            name="embed-b", provider=self.provider,
            model_name="ollama/nomic-embed-text", model_revision="v2",
            vector_dimension=dimension, distance_metric=metric,
            normalization=NORMALIZATION.NONE,
        )

        self.coll_a1 = self._collection(self.scope_a, "A One")
        self.coll_a2 = self._collection(self.scope_a, "A Two")
        self.coll_a3 = self._collection(self.scope_a, "A Three")   # unassigned
        self.coll_b1 = self._collection(self.scope_b, "B One")

        self.a1 = self._chunk(self.coll_a1, "Doc A1", "Alpha", "alpha one")
        self.a2 = self._chunk(self.coll_a1, "Doc A1b", "Alpha", "alpha two")
        self.a4 = self._chunk(self.coll_a2, "Doc A2", "Alpha", "alpha four")
        self.unassigned = self._chunk(
            self.coll_a3, "Doc A3", "Alpha", "alpha unassigned"
        )
        self.foreign = self._chunk(self.coll_b1, "Doc B1", "Beta", "beta one")

        self.agent = AgentProfile.objects.create(
            name="A Agent", role="r", model_config=self.model_config,
            application_scope=self.scope_a, knowledge_max_chars=6000,
        )
        self.agent.knowledge_collections.add(self.coll_a1, self.coll_a2)

        pad = (0.0,) * (dimension - 2)
        self.vectors = {
            "a1": (0.6, 0.8) + pad,
            "a2": (0.0, 1.0) + pad,
            "a4": (0.8, 0.6) + pad,
            # Both forbidden chunks are PERFECT matches for the query below.
            "unassigned": (1.0, 0.0) + pad,
            "foreign": (1.0, 0.0) + pad,
        }
        for name, values in self.vectors.items():
            store_chunk_vector(
                application_scope=getattr(self, name)
                .document.collection.application_scope,
                chunk=getattr(self, name),
                embedding_model_config=self.config,
                vector=values,
            )

        # Bulk unauthorized vectors: enough that a GLOBAL graph would be
        # dominated by them, so isolation cannot pass by fixture accident.
        self.bulk_chunks = []
        for index in range(unauthorized_bulk):
            chunk = self._chunk(
                self.coll_a3, f"Bulk {index}", "Alpha", f"alpha bulk {index}"
            )
            self.bulk_chunks.append(chunk)
            store_chunk_vector(
                application_scope=self.scope_a, chunk=chunk,
                embedding_model_config=self.config,
                vector=(1.0, 0.0) + pad,
            )

        self.scope = resolve_effective_knowledge_scope(self.agent)
        self.contract = resolve_embedding_contract(self.config)
        self.query = (1.0, 0.0) + pad
        return self

    def _collection(self, scope, name):
        return KnowledgeCollection.objects.create(
            name=name, description="", application_scope=scope
        )

    def _chunk(self, collection, title, section, content):
        document = KnowledgeDocument.objects.create(
            collection=collection, title=title,
            curated_text=f"{KNOWLEDGE_SECRET} {content}",
            status=KnowledgeDocument.Status.ACTIVE,
        )
        return KnowledgeDocumentChunk.objects.create(
            document=document, chunk_index=1, section_title=section,
            content=f"{KNOWLEDGE_SECRET} {content}",
        )

    def provision_and_rebuild(self, collection, *, config=None):
        config = config or self.config
        provision_pgvector_ann_leaf(
            application_scope=collection.application_scope,
            collection=collection, embedding_model_config=config,
        )
        return rebuild_pgvector_ann_leaf(
            application_scope=collection.application_scope,
            collection=collection, embedding_model_config=config,
        )

    def prepare_authorized(self, *, config=None):
        for collection in (self.coll_a1, self.coll_a2):
            self.provision_and_rebuild(collection, config=config)

    def explain(self, statement, params):
        """The plan for the EXACT production query, not a lookalike.

        `enable_seqscan = off` is a TEST-only aid: the fixtures are tiny, and
        PostgreSQL would reasonably prefer a sequential scan over a handful of
        rows, which would hide whether the HNSW index is usable at all. It is
        never set on the production search path.
        """
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL enable_seqscan = off")
            cursor.execute(
                "EXPLAIN (FORMAT JSON) "
                + statement.as_string(cursor.connection),
                params,
            )
            return json.dumps(cursor.fetchone()[0])

    def leaf_names(self, collection, *, config=None):
        config = config or self.config
        contract = resolve_embedding_contract(config)
        return leaf_identity(
            collection.application_scope_id, collection.pk, contract.e1
        )


@requires_postgres
class ExtensionAndSchemaTests(PgvectorFixtureMixin, TestCase):
    def test_the_vector_extension_is_installed_and_recent_enough(self):
        version = pgvector_ann.require_pgvector_backend()
        self.assertGreaterEqual(version, MIN_PGVECTOR_VERSION)

    def test_the_backend_tables_exist(self):
        with connection.cursor() as cursor:
            for table in (
                ANN_PARENT_TABLE, ANN_GENERATION_TABLE, ANN_LEAF_STATE_TABLE
            ):
                cursor.execute("SELECT to_regclass(%s)", [table])
                self.assertIsNotNone(cursor.fetchone()[0], table)

    def test_the_parent_table_is_partitioned(self):
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT c.relkind FROM pg_class c WHERE c.relname = %s",
                [ANN_PARENT_TABLE],
            )
            self.assertEqual(cursor.fetchone()[0], "p", "a partitioned table")


@requires_postgres
class PartitionHierarchyTests(PgvectorFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def test_provisioning_creates_the_three_level_path(self):
        identity = self.leaf_names(self.coll_a1)
        provision_pgvector_ann_leaf(
            application_scope=self.scope_a, collection=self.coll_a1,
            embedding_model_config=self.config,
        )
        with connection.cursor() as cursor:
            for name, relkind in (
                (identity.scope_partition, "p"),
                (identity.collection_partition, "p"),
                (identity.leaf_table, "r"),
            ):
                cursor.execute(
                    "SELECT relkind FROM pg_class WHERE relname = %s", [name]
                )
                row = cursor.fetchone()
                self.assertIsNotNone(row, name)
                self.assertEqual(row[0], relkind, name)

    def test_provisioning_is_idempotent(self):
        for _ in range(3):
            provision_pgvector_ann_leaf(
                application_scope=self.scope_a, collection=self.coll_a1,
                embedding_model_config=self.config,
            )
        identity = self.leaf_names(self.coll_a1)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM pg_class WHERE relname = %s",
                [identity.leaf_table],
            )
            self.assertEqual(cursor.fetchone()[0], 1)

    def test_a_foreign_collection_cannot_be_provisioned_into_a_scope(self):
        with self.assertRaises(PgvectorAnnError) as raised:
            provision_pgvector_ann_leaf(
                application_scope=self.scope_a, collection=self.coll_b1,
                embedding_model_config=self.config,
            )
        self.assertEqual(
            raised.exception.category,
            PgvectorFailureCategory.COLLECTION_FOREIGN_SCOPE,
        )

    def test_the_hnsw_index_belongs_to_exactly_one_leaf(self):
        self.provision_and_rebuild(self.coll_a1)
        identity = self.leaf_names(self.coll_a1)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT i.relname, t.relname, am.amname "
                "FROM pg_index x "
                "JOIN pg_class i ON i.oid = x.indexrelid "
                "JOIN pg_class t ON t.oid = x.indrelid "
                "JOIN pg_am am ON am.oid = i.relam "
                "WHERE am.amname = 'hnsw'"
            )
            rows = cursor.fetchall()
        self.assertTrue(rows, "an HNSW index exists")
        for index_name, table_name, _am in rows:
            with self.subTest(index=index_name):
                self.assertNotEqual(table_name, ANN_PARENT_TABLE)
                self.assertNotEqual(table_name, identity.scope_partition)
                self.assertNotEqual(table_name, identity.collection_partition)
                self.assertRegex(table_name, r"^ah_pgv_s\d+_c\d+_e[0-9a-f]{12}$")

    def test_no_global_or_scope_wide_hnsw_index_exists(self):
        """The primary catalog proof. A graph may span ONE leaf and no more."""
        self.prepare_authorized()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT t.relname, t.relkind FROM pg_index x "
                "JOIN pg_class i ON i.oid = x.indexrelid "
                "JOIN pg_class t ON t.oid = x.indrelid "
                "JOIN pg_am am ON am.oid = i.relam "
                "WHERE am.amname = 'hnsw'"
            )
            for table_name, relkind in cursor.fetchall():
                with self.subTest(table=table_name):
                    self.assertEqual(
                        relkind, "r",
                        "an HNSW index on a partitioned table would span "
                        "multiple collections or scopes",
                    )

    def test_the_leaf_rejects_a_wrong_dimension_vector(self):
        self.provision_and_rebuild(self.coll_a1)
        identity = self.leaf_names(self.coll_a1)
        from django.db import InternalError, ProgrammingError, DataError
        from django.db.utils import IntegrityError

        with connection.cursor() as cursor:
            with self.assertRaises(
                (IntegrityError, DataError, InternalError, ProgrammingError)
            ):
                cursor.execute(
                    f'INSERT INTO "{identity.leaf_table}" '
                    "(source_embedding_id, application_scope_id, collection_id, "
                    "chunk_id, k1, e1, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s::vector)",
                    [
                        999999, self.scope_a.pk, self.coll_a1.pk, 999999,
                        "k1:sha256:" + "0" * 64, self.contract.e1, "[1,2,3]",
                    ],
                )


@requires_postgres
class GenerationTriggerTests(PgvectorFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()
        self.prepare_authorized()

    def generation(self, collection):
        return pgvector_ann.current_generation(self.scope_a.pk, collection.pk)

    def assertBumped(self, collection, before):
        self.assertGreater(self.generation(collection), before)
        readiness = leaf_readiness(self.scope_a.pk, collection.pk, self.contract)
        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "stale_generation")

    def test_a_ready_leaf_reports_ready(self):
        readiness = leaf_readiness(self.scope_a.pk, self.coll_a1.pk, self.contract)
        self.assertTrue(readiness.ready, readiness.reason)

    def test_editing_chunk_content_invalidates(self):
        before = self.generation(self.coll_a1)
        KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).update(
            content="alpha rewritten"
        )
        self.assertBumped(self.coll_a1, before)

    def test_editing_a_section_title_invalidates(self):
        before = self.generation(self.coll_a1)
        KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).update(
            section_title="Renamed"
        )
        self.assertBumped(self.coll_a1, before)

    def test_archiving_a_document_invalidates(self):
        before = self.generation(self.coll_a1)
        KnowledgeDocument.objects.filter(pk=self.a1.document_id).update(
            status=KnowledgeDocument.Status.ARCHIVED
        )
        self.assertBumped(self.coll_a1, before)

    def test_moving_a_document_invalidates_both_collections(self):
        before_a1 = self.generation(self.coll_a1)
        before_a2 = self.generation(self.coll_a2)
        KnowledgeDocument.objects.filter(pk=self.a1.document_id).update(
            collection=self.coll_a2
        )
        self.assertGreater(self.generation(self.coll_a1), before_a1)
        self.assertGreater(self.generation(self.coll_a2), before_a2)

    def test_deactivating_a_collection_invalidates(self):
        before = self.generation(self.coll_a1)
        KnowledgeCollection.objects.filter(pk=self.coll_a1.pk).update(
            is_active=False
        )
        self.assertGreater(self.generation(self.coll_a1), before)

    def test_mutating_a_canonical_vector_invalidates(self):
        before = self.generation(self.coll_a1)
        KnowledgeChunkEmbedding.objects.filter(chunk_id=self.a1.pk).update(
            k1="k1:sha256:" + "0" * 64
        )
        self.assertBumped(self.coll_a1, before)

    def test_deleting_a_canonical_vector_invalidates(self):
        before = self.generation(self.coll_a1)
        KnowledgeChunkEmbedding.objects.filter(chunk_id=self.a1.pk).delete()
        self.assertGreater(self.generation(self.coll_a1), before)

    def test_deleting_a_chunk_cascades_to_invalidation(self):
        before = self.generation(self.coll_a1)
        KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).delete()
        self.assertGreater(
            self.generation(self.coll_a1), before,
            "the cascaded embedding DELETE must bump the generation",
        )

    def test_metadata_and_token_estimate_do_not_invalidate(self):
        """The trigger is narrow by column AND by value, deliberately."""
        before = self.generation(self.coll_a1)
        KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).update(
            metadata={"note": "bookkeeping"}, token_estimate=42
        )
        self.assertEqual(self.generation(self.coll_a1), before)

    def test_a_full_row_save_that_changes_nothing_relevant_does_not_invalidate(self):
        before = self.generation(self.coll_a1)
        chunk = KnowledgeDocumentChunk.objects.get(pk=self.a1.pk)
        chunk.token_estimate = 7
        chunk.save()
        self.assertEqual(
            self.generation(self.coll_a1), before,
            "a value-comparing WHEN clause means save() alone does not bump",
        )

    def test_a_raw_sql_update_still_invalidates(self):
        """The reason this is a database trigger and not a Django signal."""
        before = self.generation(self.coll_a1)
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE ai_hub_knowledgedocumentchunk SET content = %s "
                "WHERE id = %s",
                ["bypassing the ORM entirely", self.a1.pk],
            )
        self.assertGreater(self.generation(self.coll_a1), before)


@requires_postgres
class RebuildTests(PgvectorFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()

    def leaf_rows(self, collection, *, config=None):
        identity = self.leaf_names(collection, config=config)
        with connection.cursor() as cursor:
            cursor.execute(
                f'SELECT chunk_id, k1 FROM "{identity.leaf_table}" ORDER BY chunk_id'
            )
            return cursor.fetchall()

    def test_a_rebuild_mirrors_only_current_and_retrievable_vectors(self):
        stale = self._chunk(self.coll_a1, "Doc stale", "Alpha", "alpha stale")
        store_chunk_vector(
            application_scope=self.scope_a, chunk=stale,
            embedding_model_config=self.config, vector=(0.1, 0.2, 0.0, 0.0),
        )
        KnowledgeDocumentChunk.objects.filter(pk=stale.pk).update(
            content="edited after indexing"
        )

        archived = self._chunk(self.coll_a1, "Doc archived", "Alpha", "alpha arch")
        store_chunk_vector(
            application_scope=self.scope_a, chunk=archived,
            embedding_model_config=self.config, vector=(0.3, 0.4, 0.0, 0.0),
        )
        KnowledgeDocument.objects.filter(pk=archived.document_id).update(
            status=KnowledgeDocument.Status.ARCHIVED
        )

        other_space = self._chunk(self.coll_a1, "Doc other", "Alpha", "alpha other")
        store_chunk_vector(
            application_scope=self.scope_a, chunk=other_space,
            embedding_model_config=self.config_other, vector=(0.5, 0.6, 0.0, 0.0),
        )

        result = self.provision_and_rebuild(self.coll_a1)
        mirrored = {row[0] for row in self.leaf_rows(self.coll_a1)}

        self.assertEqual(mirrored, {self.a1.pk, self.a2.pk})
        self.assertNotIn(stale.pk, mirrored)
        self.assertNotIn(archived.pk, mirrored)
        self.assertNotIn(other_space.pk, mirrored)
        self.assertEqual(result.source_count, 2)

    def test_an_inactive_collection_mirrors_nothing(self):
        KnowledgeCollection.objects.filter(pk=self.coll_a1.pk).update(
            is_active=False
        )
        self.coll_a1.refresh_from_db()
        result = self.provision_and_rebuild(self.coll_a1)
        self.assertEqual(result.source_count, 0)
        self.assertEqual(self.leaf_rows(self.coll_a1), [])

    def test_a_rebuild_contacts_no_provider(self):
        """ANN maintenance is local index work. Nothing is re-embedded."""
        def explode(*args, **kwargs):
            raise AssertionError("no provider may be contacted during a rebuild")

        with mock.patch(
            "ai_hub.services.semantic_retrieval.resolve_embedding_transport",
            explode,
        ):
            with mock.patch("requests.post", explode):
                with mock.patch("requests.get", explode):
                    result = self.provision_and_rebuild(self.coll_a1)
        self.assertEqual(result.source_count, 2)

    def test_a_rebuild_reads_only_the_canonical_store(self):
        result = self.provision_and_rebuild(self.coll_a1)
        canonical = set(
            KnowledgeChunkEmbedding.objects.filter(
                collection_id=self.coll_a1.pk, e1=self.contract.e1
            ).values_list("chunk_id", flat=True)
        )
        self.assertEqual({row[0] for row in self.leaf_rows(self.coll_a1)}, canonical)
        self.assertEqual(result.source_count, len(canonical))

    def test_a_rebuild_replaces_rather_than_appends(self):
        self.provision_and_rebuild(self.coll_a1)
        KnowledgeChunkEmbedding.objects.filter(chunk_id=self.a2.pk).delete()
        result = self.provision_and_rebuild(self.coll_a1)
        self.assertEqual({row[0] for row in self.leaf_rows(self.coll_a1)}, {self.a1.pk})
        self.assertEqual(result.source_count, 1)

    def test_a_rebuild_that_races_a_mutation_reports_not_ready(self):
        real = pgvector_ann._mirrorable_rows

        def mutate_mid_rebuild(scope_id, collection_id, contract):
            rows = real(scope_id, collection_id, contract)
            KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).update(
                content="changed during the rebuild"
            )
            return rows

        provision_pgvector_ann_leaf(
            application_scope=self.scope_a, collection=self.coll_a1,
            embedding_model_config=self.config,
        )
        with mock.patch.object(
            pgvector_ann, "_mirrorable_rows", mutate_mid_rebuild
        ):
            result = rebuild_pgvector_ann_leaf(
                application_scope=self.scope_a, collection=self.coll_a1,
                embedding_model_config=self.config,
            )
        self.assertFalse(result.ready, "a raced rebuild must not claim readiness")
        self.assertNotEqual(result.indexed_generation, result.current_generation)
        readiness = leaf_readiness(self.scope_a.pk, self.coll_a1.pk, self.contract)
        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "stale_generation")

    def test_a_failed_rebuild_leaves_no_partial_leaf(self):
        self.provision_and_rebuild(self.coll_a1)
        before = self.leaf_rows(self.coll_a1)
        self.assertTrue(before)

        class Boom(RuntimeError):
            pass

        def failing_rows(*args, **kwargs):
            raise Boom("rebuild failed midway")

        with mock.patch.object(pgvector_ann, "_vector_literal", failing_rows):
            with self.assertRaises(Boom):
                rebuild_pgvector_ann_leaf(
                    application_scope=self.scope_a, collection=self.coll_a1,
                    embedding_model_config=self.config,
                )
        self.assertEqual(
            self.leaf_rows(self.coll_a1), before,
            "the delete and the insert share one transaction",
        )


@requires_postgres
class PlanIsolationTests(PgvectorFixtureMixin, TestCase):
    """The load-bearing proofs. A plan, not a result list."""

    def setUp(self):
        self.build_corpus(unauthorized_bulk=200)
        self.prepare_authorized()
        # The forbidden leaves EXIST and are populated, so their absence from a
        # plan is a real property and not an artefact of them being empty.
        self.provision_and_rebuild(self.coll_a3)
        provision_pgvector_ann_leaf(
            application_scope=self.scope_b, collection=self.coll_b1,
            embedding_model_config=self.config,
        )
        rebuild_pgvector_ann_leaf(
            application_scope=self.scope_b, collection=self.coll_b1,
            embedding_model_config=self.config,
        )

    def plan_for(self, collection_ids):
        statement, params = build_ann_candidate_sql(
            application_scope_id=self.scope_a.pk,
            collection_ids=tuple(collection_ids),
            e1=self.contract.e1,
            dimension=self.contract.vector_dimension,
            metric=self.contract.distance_metric,
            query_values=self.query,
        )
        return self.explain(statement, params)

    def test_the_unauthorized_leaves_are_populated(self):
        """Proves the plan tests below are not vacuous."""
        for collection in (self.coll_a3, self.coll_b1):
            identity = self.leaf_names(collection)
            with connection.cursor() as cursor:
                cursor.execute(f'SELECT count(*) FROM "{identity.leaf_table}"')
                self.assertGreater(cursor.fetchone()[0], 0, identity.leaf_table)

    def test_a_single_collection_plan_touches_only_its_own_leaf(self):
        plan = self.plan_for([self.coll_a1.pk])
        wanted = self.leaf_names(self.coll_a1)
        forbidden = [
            self.leaf_names(self.coll_a2),
            self.leaf_names(self.coll_a3),
            self.leaf_names(self.coll_b1),
        ]
        self.assertIn(wanted.leaf_table, plan)
        for identity in forbidden:
            with self.subTest(leaf=identity.leaf_table):
                self.assertNotIn(identity.leaf_table, plan)
                self.assertNotIn(identity.leaf_index, plan)

    def test_a_same_scope_unassigned_collection_is_never_scanned(self):
        plan = self.plan_for([self.coll_a1.pk, self.coll_a2.pk])
        identity = self.leaf_names(self.coll_a3)
        self.assertNotIn(identity.leaf_table, plan)
        self.assertNotIn(identity.leaf_index, plan)

    def test_a_cross_scope_collection_is_never_scanned(self):
        plan = self.plan_for([self.coll_a1.pk, self.coll_a2.pk])
        identity = self.leaf_names(self.coll_b1)
        self.assertNotIn(identity.leaf_table, plan)
        self.assertNotIn(identity.leaf_index, plan)
        self.assertNotIn(f"ah_pgv_s{self.scope_b.pk}", plan)

    def test_multiple_authorized_collections_scan_exactly_their_leaves(self):
        plan = self.plan_for([self.coll_a1.pk, self.coll_a2.pk])
        self.assertIn(self.leaf_names(self.coll_a1).leaf_table, plan)
        self.assertIn(self.leaf_names(self.coll_a2).leaf_table, plan)
        for name in re.findall(r"ah_pgv_s\d+_c\d+_e[0-9a-f]{12}", plan):
            with self.subTest(leaf=name):
                self.assertIn(
                    name,
                    {
                        self.leaf_names(self.coll_a1).leaf_table,
                        self.leaf_names(self.coll_a2).leaf_table,
                    },
                )

    def test_a_different_e1_leaf_is_never_scanned(self):
        """Different vector spaces consume zero ANN capacity."""
        for chunk in (self.a1, self.a2):
            store_chunk_vector(
                application_scope=self.scope_a, chunk=chunk,
                embedding_model_config=self.config_other,
                vector=(1.0, 0.0, 0.0, 0.0),
            )
        self.provision_and_rebuild(self.coll_a1, config=self.config_other)
        other = self.leaf_names(self.coll_a1, config=self.config_other)

        plan = self.plan_for([self.coll_a1.pk])
        self.assertIn(self.leaf_names(self.coll_a1).leaf_table, plan)
        self.assertNotIn(other.leaf_table, plan)
        self.assertNotIn(other.leaf_index, plan)

    def test_the_hnsw_index_can_actually_be_used(self):
        """`enable_seqscan = off` is a TEST-only aid for a tiny fixture."""
        plan = self.plan_for([self.coll_a1.pk])
        self.assertIn("Index Scan", plan)
        self.assertIn(self.leaf_names(self.coll_a1).leaf_index, plan)

    def test_bulk_unauthorized_vectors_never_enter_the_candidate_pool(self):
        """Isolation must be structural, not a small-fixture accident."""
        result = search_pgvector_ann_with_scope(
            self.scope, query_values=self.query,
            embedding_model_config=self.config, limit=5,
        )
        returned = {match.chunk_id for match in result.matches}
        bulk = {chunk.pk for chunk in self.bulk_chunks}
        self.assertFalse(returned & bulk)
        self.assertNotIn(self.unassigned.pk, returned)
        self.assertNotIn(self.foreign.pk, returned)
        self.assertLessEqual(result.ann_candidates_returned, ANN_CANDIDATE_POOL)
        self.assertEqual(result.ann_candidates_returned, 3)


@requires_postgres
class AnnSearchTests(PgvectorFixtureMixin, TestCase):
    def setUp(self):
        self.build_corpus()
        self.prepare_authorized()

    def search(self, **kwargs):
        parameters = dict(
            query_values=self.query, embedding_model_config=self.config, limit=5
        )
        parameters.update(kwargs)
        return search_pgvector_ann_with_scope(self.scope, **parameters)

    def test_the_result_is_exactly_reranked_by_the_s21_scorer(self):
        result = self.search()
        expected = sorted(
            (
                (cosine_similarity(self.query, self.vectors[name]), chunk.pk)
                for name, chunk in (
                    ("a1", self.a1), ("a2", self.a2), ("a4", self.a4)
                )
            ),
            key=lambda entry: (-entry[0], entry[1]),
        )
        self.assertEqual(
            [match.chunk_id for match in result.matches],
            [chunk_id for _score, chunk_id in expected],
        )
        for match, (score, _chunk_id) in zip(result.matches, expected):
            self.assertAlmostEqual(match.metric_value, score, places=5)

    def test_the_backend_operator_value_never_becomes_the_metric_value(self):
        """`<#>` returns a NEGATED inner product. That must not escape."""
        config = EmbeddingModelConfig.objects.create(
            name="embed-dot", provider=self.provider,
            model_name="ollama/nomic-embed-text", model_revision="v9",
            vector_dimension=4, distance_metric=METRIC.DOT_PRODUCT,
            normalization=NORMALIZATION.NONE,
        )
        for chunk, name in ((self.a1, "a1"), (self.a2, "a2")):
            store_chunk_vector(
                application_scope=self.scope_a, chunk=chunk,
                embedding_model_config=config, vector=self.vectors[name],
            )
        self.provision_and_rebuild(self.coll_a1, config=config)
        self.provision_and_rebuild(self.coll_a2, config=config)

        result = search_pgvector_ann_with_scope(
            self.scope, query_values=self.query,
            embedding_model_config=config, limit=5,
        )
        for match in result.matches:
            with self.subTest(chunk=match.chunk_id):
                self.assertGreaterEqual(match.metric_value, 0.0)

    def test_limit_zero_runs_no_ann_query(self):
        result = self.search(limit=0)
        self.assertEqual(result.matches, ())
        self.assertEqual(result.ann_candidates_returned, 0)

    def test_an_out_of_range_limit_is_refused(self):
        for bad in (-1, MAX_ANN_RESULTS + 1, 1.5, True, None):
            with self.subTest(limit=bad):
                with self.assertRaises(PgvectorAnnError) as raised:
                    self.search(limit=bad)
                self.assertEqual(
                    raised.exception.category,
                    PgvectorFailureCategory.INVALID_LIMIT,
                )

    def test_a_wrong_dimension_query_vector_is_refused(self):
        with self.assertRaises(PgvectorAnnError) as raised:
            self.search(query_values=(1.0, 0.0))
        self.assertEqual(
            raised.exception.category,
            PgvectorFailureCategory.QUERY_VECTOR_INVALID,
        )

    def test_a_non_finite_query_vector_is_refused(self):
        with self.assertRaises(PgvectorAnnError):
            self.search(query_values=(float("nan"), 0.0, 0.0, 0.0))

    def test_collection_narrowing_restricts_the_search(self):
        result = self.search(collection_id=self.coll_a2.pk)
        self.assertEqual(result.collection_ids, (self.coll_a2.pk,))
        self.assertEqual(
            {match.chunk_id for match in result.matches}, {self.a4.pk}
        )

    def test_narrowing_can_never_widen(self):
        for requested in (self.coll_a3.pk, self.coll_b1.pk, 9_999_999):
            with self.subTest(requested=requested):
                result = self.search(collection_id=requested)
                self.assertEqual(result.matches, ())
                self.assertEqual(result.collection_ids, ())

    def test_one_unready_target_refuses_the_whole_search(self):
        """Never a silently narrower search over the collections that are ready."""
        KnowledgeDocumentChunk.objects.filter(pk=self.a4.pk).update(
            content="changed after indexing"
        )
        with self.assertRaises(PgvectorAnnError) as raised:
            self.search()
        self.assertEqual(
            raised.exception.category, PgvectorFailureCategory.LEAF_NOT_READY
        )

    def test_an_unprovisioned_target_refuses(self):
        identity = self.leaf_names(self.coll_a2)
        with connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {ANN_LEAF_STATE_TABLE} "
                "WHERE application_scope_id = %s AND collection_id = %s",
                [self.scope_a.pk, self.coll_a2.pk],
            )
        self.assertTrue(identity.leaf_table)
        with self.assertRaises(PgvectorAnnError) as raised:
            self.search()
        self.assertEqual(
            raised.exception.category, PgvectorFailureCategory.LEAF_NOT_READY
        )

    def test_a_backend_version_change_makes_a_leaf_unready(self):
        with connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {ANN_LEAF_STATE_TABLE} SET backend_version = %s",
                ["pgv-hnsw0"],
            )
        readiness = leaf_readiness(self.scope_a.pk, self.coll_a1.pk, self.contract)
        self.assertFalse(readiness.ready)
        self.assertEqual(readiness.reason, "backend_version_mismatch")

    def test_a_corrupt_mirror_row_refuses_rather_than_being_dropped(self):
        """Quietly dropping it would return a short ranking that looks complete."""
        identity = self.leaf_names(self.coll_a1)
        with connection.cursor() as cursor:
            cursor.execute(
                f'UPDATE "{identity.leaf_table}" SET k1 = %s WHERE chunk_id = %s',
                ["k1:sha256:" + "0" * 64, self.a1.pk],
            )
            # Deliberately corrupt the MIRROR only, so the generation is
            # untouched and readiness still passes - which is exactly the state
            # canonical revalidation exists to catch.
            cursor.execute(
                f'DELETE FROM "{identity.leaf_table}" WHERE chunk_id = %s',
                [self.a2.pk],
            )
            cursor.execute(
                f'INSERT INTO "{identity.leaf_table}" (source_embedding_id, '
                "application_scope_id, collection_id, chunk_id, k1, e1, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::vector)",
                [
                    10_000_001, self.scope_a.pk, self.coll_a1.pk, 10_000_001,
                    "k1:sha256:" + "0" * 64, self.contract.e1, "[1,0,0,0]",
                ],
            )
        with self.assertRaises(PgvectorAnnError) as raised:
            self.search()
        self.assertEqual(
            raised.exception.category,
            PgvectorFailureCategory.ANN_INTEGRITY_MISMATCH,
        )

    def test_a_source_change_during_search_refuses_the_result(self):
        real = pgvector_ann._rerank_candidates

        def mutate_mid_search(*args, **kwargs):
            matches = real(*args, **kwargs)
            KnowledgeDocumentChunk.objects.filter(pk=self.a1.pk).update(
                content="changed during the search"
            )
            return matches

        with mock.patch.object(
            pgvector_ann, "_rerank_candidates", mutate_mid_search
        ):
            with self.assertRaises(PgvectorAnnError) as raised:
                self.search()
        self.assertEqual(
            raised.exception.category,
            PgvectorFailureCategory.SOURCE_CHANGED_DURING_SEARCH,
        )

    def test_the_result_carries_no_query_content_or_vector(self):
        result = self.search()
        blob = json.dumps(
            {
                "result": {
                    key: value for key, value in result.__dict__.items()
                    if key != "matches"
                },
                "matches": [match.__dict__ for match in result.matches],
            },
            default=str,
        )
        for forbidden in (KNOWLEDGE_SECRET, "alpha", "embedding", "[1.0"):
            self.assertNotIn(forbidden, blob)
        self.assertEqual(
            set(result.matches[0].__dict__),
            {
                "rank", "chunk_id", "document_id", "collection_id",
                "application_scope_id", "k1", "e1", "metric", "metric_value",
            },
        )

    def test_the_search_writes_no_audit_row(self):
        from ai_hub.models import RetrievalHit, RetrievalRun

        self.search()
        self.assertEqual(RetrievalRun.objects.count(), 0)
        self.assertEqual(RetrievalHit.objects.count(), 0)


@requires_postgres
class OperatorParityTests(PgvectorFixtureMixin, TestCase):
    """PostgreSQL exact operators must agree with the S-21 Python oracle."""

    def _parity(self, metric, scorer, higher_is_better):
        self.build_corpus(metric=metric)
        config = self.config
        self.prepare_authorized(config=config)
        identity = self.leaf_names(self.coll_a1)

        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL enable_indexscan = off")
            cursor.execute("SET LOCAL enable_bitmapscan = off")
            backend = resolve_metric_backend(metric)
            cursor.execute(
                f'SELECT chunk_id FROM "{identity.leaf_table}" '
                f"ORDER BY (embedding::vector(4)) {backend.operator} "
                "(%s::vector(4)), chunk_id",
                [pgvector_ann._vector_literal(self.query)],
            )
            postgres_order = [row[0] for row in cursor.fetchall()]

        python_order = [
            chunk_id
            for _score, chunk_id in sorted(
                (
                    (scorer(self.query, self.vectors[name]), chunk.pk)
                    for name, chunk in (("a1", self.a1), ("a2", self.a2))
                ),
                key=lambda entry: (
                    -entry[0] if higher_is_better else entry[0], entry[1]
                ),
            )
        ]
        self.assertEqual(postgres_order, python_order)

    def test_cosine_parity(self):
        self._parity(METRIC.COSINE, cosine_similarity, True)

    def test_dot_product_parity(self):
        self._parity(METRIC.DOT_PRODUCT, dot_product_similarity, True)

    def test_euclidean_parity(self):
        self._parity(METRIC.EUCLIDEAN, euclidean_distance, False)

    def test_ties_resolve_on_chunk_id_after_canonical_reranking(self):
        self.build_corpus(metric=METRIC.COSINE)
        for chunk in (self.a1, self.a2):
            KnowledgeChunkEmbedding.objects.filter(chunk_id=chunk.pk).delete()
            store_chunk_vector(
                application_scope=self.scope_a, chunk=chunk,
                embedding_model_config=self.config, vector=(1.0, 0.0, 0.0, 0.0),
            )
        self.prepare_authorized()
        result = search_pgvector_ann_with_scope(
            self.scope, query_values=self.query,
            embedding_model_config=self.config, limit=5,
        )
        tied = [
            match.chunk_id for match in result.matches
            if abs(match.metric_value - 1.0) < 1e-9
        ]
        self.assertEqual(tied, sorted(tied))


@requires_postgres
class CosineZeroVectorCampaignTests(PgvectorFixtureMixin, TestCase):
    """The gap CI would otherwise find: ANN answering where S-21 refuses."""

    def ann_query_spy(self):
        """Records whether the HNSW candidate SQL was ever built."""
        calls = []
        real = pgvector_ann.build_ann_candidate_sql

        def spy(**kwargs):
            calls.append(kwargs)
            return real(**kwargs)

        return mock.patch.object(
            pgvector_ann, "build_ann_candidate_sql", spy
        ), calls

    def add_zero_vector(self, collection, chunk_label="zero"):
        chunk = self._chunk(collection, f"Doc {chunk_label}", "Alpha", chunk_label)
        store_chunk_vector(
            application_scope=collection.application_scope, chunk=chunk,
            embedding_model_config=self.config,
            vector=(0.0,) * self.contract.vector_dimension,
        )
        return chunk

    # -- A. cosine corpus zero ------------------------------------------

    def test_a_cosine_rebuild_refuses_an_eligible_zero_vector(self):
        self.build_corpus(metric=METRIC.COSINE)
        zero = self.add_zero_vector(self.coll_a1)

        provision_pgvector_ann_leaf(
            application_scope=self.scope_a, collection=self.coll_a1,
            embedding_model_config=self.config,
        )
        with self.assertRaises(PgvectorAnnError) as raised:
            rebuild_pgvector_ann_leaf(
                application_scope=self.scope_a, collection=self.coll_a1,
                embedding_model_config=self.config,
            )
        self.assertEqual(
            raised.exception.category,
            PgvectorFailureCategory.UNSCORABLE_ZERO_VECTOR,
        )

        readiness = leaf_readiness(self.scope_a.pk, self.coll_a1.pk, self.contract)
        self.assertFalse(readiness.ready, "the leaf must NOT become ready")
        self.assertEqual(readiness.reason, "leaf_absent")

        # And the zero vector is still in the canonical store, untouched.
        self.assertTrue(
            KnowledgeChunkEmbedding.objects.filter(chunk_id=zero.pk).exists()
        )

    def test_the_zero_vector_was_eligible_and_not_merely_filtered_out(self):
        """Proves the refusal is real, not a candidate that never qualified."""
        self.build_corpus(metric=METRIC.COSINE)
        zero = self.add_zero_vector(self.coll_a1)

        eligible = pgvector_ann._mirrorable_rows(
            self.scope_a.pk, self.coll_a1.pk, self.contract
        )
        eligible_ids = {record.chunk_id for record, _values in eligible}
        self.assertIn(
            zero.pk, eligible_ids,
            "it is CURRENT and RETRIEVABLE; only its magnitude disqualifies it",
        )

    def test_a_search_cannot_silently_return_the_survivors(self):
        self.build_corpus(metric=METRIC.COSINE)
        self.prepare_authorized()
        readiness = leaf_readiness(self.scope_a.pk, self.coll_a1.pk, self.contract)
        self.assertTrue(readiness.ready, "ready before the zero vector arrives")

        self.add_zero_vector(self.coll_a1)

        # The insert bumped the generation, so the previously-ready leaf is
        # already stale; a rebuild must not restore readiness by dropping it.
        with self.assertRaises(PgvectorAnnError):
            rebuild_pgvector_ann_leaf(
                application_scope=self.scope_a, collection=self.coll_a1,
                embedding_model_config=self.config,
            )
        with self.assertRaises(PgvectorAnnError) as raised:
            search_pgvector_ann_with_scope(
                self.scope, query_values=self.query,
                embedding_model_config=self.config, limit=5,
            )
        self.assertIn(
            raised.exception.category,
            {
                PgvectorFailureCategory.LEAF_NOT_READY,
                PgvectorFailureCategory.UNSCORABLE_ZERO_VECTOR,
            },
        )

    def test_an_archived_zero_vector_does_not_block_a_rebuild(self):
        """Only ELIGIBLE vectors matter. An unretrievable one is irrelevant."""
        self.build_corpus(metric=METRIC.COSINE)
        zero = self.add_zero_vector(self.coll_a1)
        KnowledgeDocument.objects.filter(pk=zero.document_id).update(
            status=KnowledgeDocument.Status.ARCHIVED
        )
        result = self.provision_and_rebuild(self.coll_a1)
        self.assertTrue(result.ready)
        self.assertEqual(result.source_count, 2)

    # -- B. cosine query zero -------------------------------------------

    def test_a_zero_cosine_query_refuses_before_any_ann_sql(self):
        self.build_corpus(metric=METRIC.COSINE)
        self.prepare_authorized()

        patcher, calls = self.ann_query_spy()
        with patcher:
            with self.assertRaises(PgvectorAnnError) as raised:
                search_pgvector_ann_with_scope(
                    self.scope,
                    query_values=(0.0,) * self.contract.vector_dimension,
                    embedding_model_config=self.config, limit=5,
                )
        self.assertEqual(
            raised.exception.category,
            PgvectorFailureCategory.UNSCORABLE_ZERO_VECTOR,
        )
        self.assertEqual(calls, [], "the HNSW candidate query was never built")

    def test_a_non_zero_cosine_query_still_reaches_the_ann_query(self):
        """Proves the spy above is not vacuous."""
        self.build_corpus(metric=METRIC.COSINE)
        self.prepare_authorized()
        patcher, calls = self.ann_query_spy()
        with patcher:
            search_pgvector_ann_with_scope(
                self.scope, query_values=self.query,
                embedding_model_config=self.config, limit=5,
            )
        self.assertEqual(len(calls), 1)

    # -- C / D. the other metrics are unaffected --------------------------

    def test_dot_product_accepts_zero_vectors_on_both_sides(self):
        self.build_corpus(metric=METRIC.DOT_PRODUCT)
        zero = self.add_zero_vector(self.coll_a1)
        self.prepare_authorized()

        result = search_pgvector_ann_with_scope(
            self.scope, query_values=self.query,
            embedding_model_config=self.config, limit=10,
        )
        scored = {match.chunk_id: match.metric_value for match in result.matches}
        self.assertIn(zero.pk, scored, "a zero candidate is valid for dot product")
        self.assertAlmostEqual(scored[zero.pk], 0.0, places=9)

        zero_query = search_pgvector_ann_with_scope(
            self.scope,
            query_values=(0.0,) * self.contract.vector_dimension,
            embedding_model_config=self.config, limit=10,
        )
        for match in zero_query.matches:
            self.assertAlmostEqual(match.metric_value, 0.0, places=9)

    def test_euclidean_accepts_zero_vectors_on_both_sides(self):
        self.build_corpus(metric=METRIC.EUCLIDEAN)
        zero = self.add_zero_vector(self.coll_a1)
        self.prepare_authorized()

        result = search_pgvector_ann_with_scope(
            self.scope, query_values=self.query,
            embedding_model_config=self.config, limit=10,
        )
        scored = {match.chunk_id: match.metric_value for match in result.matches}
        self.assertIn(zero.pk, scored)
        self.assertAlmostEqual(
            scored[zero.pk],
            euclidean_distance(
                self.query, (0.0,) * self.contract.vector_dimension
            ),
            places=5,
        )

        zero_query = search_pgvector_ann_with_scope(
            self.scope,
            query_values=(0.0,) * self.contract.vector_dimension,
            embedding_model_config=self.config, limit=10,
        )
        self.assertTrue(zero_query.matches)


@requires_postgres
class ProvisioningConcurrencyTests(TransactionTestCase):
    """Two REAL PostgreSQL connections provisioning the same leaf at once.

    `TransactionTestCase` rather than `TestCase`, because the threads need
    committed data and their own connections - inside a `TestCase` the fixture
    would live in an uncommitted transaction the other thread cannot see.

    Django's `connections` is thread-local, so each thread opens its own backend
    connection. The test asserts that from the server's own `pg_backend_pid()`
    rather than assuming it, and the real `pg_advisory_xact_lock` executes: it is
    deliberately not mocked, because the point is whether the LOCK works.
    """

    available_apps = None

    def setUp(self):
        self.scope = ApplicationScope.objects.create(
            name="Concurrency", slug="concurrency"
        )
        self.collection = KnowledgeCollection.objects.create(
            name="Concurrency Collection", application_scope=self.scope
        )
        provider = ProviderConfig.objects.create(
            name="Concurrency P",
            provider_type=ProviderConfig.ProviderType.OLLAMA,
            base_url="http://ollama.internal:11434",
            declared_locality=LOCALITY.LOCAL,
        )
        self.config = EmbeddingModelConfig.objects.create(
            name="concurrency-embed", provider=provider,
            model_name="ollama/nomic-embed-text", model_revision="v1",
            vector_dimension=4, distance_metric=METRIC.COSINE,
            normalization=NORMALIZATION.NONE,
        )
        self.contract = resolve_embedding_contract(self.config)
        self.identity = leaf_identity(
            self.scope.pk, self.collection.pk, self.contract.e1
        )

    def tearDown(self):
        # The ANN partitions live outside Django's model registry, so
        # `TransactionTestCase` truncation does not reach them.
        with connection.cursor() as cursor:
            cursor.execute(
                f'DROP TABLE IF EXISTS "{self.identity.scope_partition}" CASCADE'
            )
            cursor.execute(
                f"DELETE FROM {ANN_LEAF_STATE_TABLE} WHERE application_scope_id = %s",
                [self.scope.pk],
            )
            cursor.execute(
                f"DELETE FROM {ANN_GENERATION_TABLE} WHERE application_scope_id = %s",
                [self.scope.pk],
            )

    def test_two_connections_provision_the_same_leaf_without_conflict(self):
        barrier = threading.Barrier(2, timeout=30)
        errors = []
        backend_pids = []

        def worker():
            try:
                # Force this thread's own connection to open, and record the
                # server-side pid so the test can PROVE they differ.
                with connections["default"].cursor() as cursor:
                    cursor.execute("SELECT pg_backend_pid()")
                    backend_pids.append(cursor.fetchone()[0])
                barrier.wait()
                provision_pgvector_ann_leaf(
                    application_scope=self.scope, collection=self.collection,
                    embedding_model_config=self.config,
                )
            except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
                errors.append(exc)
            finally:
                connections["default"].close()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            self.assertFalse(
                thread.is_alive(), "provisioning deadlocked under the advisory lock"
            )

        self.assertEqual(errors, [], f"both calls must succeed: {errors}")
        self.assertEqual(len(backend_pids), 2)
        self.assertNotEqual(
            backend_pids[0], backend_pids[1],
            "the two threads must use independent PostgreSQL connections",
        )

        with connection.cursor() as cursor:
            for name, expected_kind in (
                (self.identity.scope_partition, "p"),
                (self.identity.collection_partition, "p"),
                (self.identity.leaf_table, "r"),
                (self.identity.leaf_index, "i"),
            ):
                cursor.execute(
                    "SELECT relkind, count(*) OVER () FROM pg_class "
                    "WHERE relname = %s",
                    [name],
                )
                rows = cursor.fetchall()
                self.assertEqual(len(rows), 1, f"exactly one {name}")
                self.assertEqual(rows[0][0], expected_kind, name)

            cursor.execute(
                "SELECT count(*) FROM pg_index x "
                "JOIN pg_class i ON i.oid = x.indexrelid "
                "JOIN pg_class t ON t.oid = x.indrelid "
                "JOIN pg_am am ON am.oid = i.relam "
                "WHERE am.amname = 'hnsw' AND t.relname = %s",
                [self.identity.leaf_table],
            )
            self.assertEqual(cursor.fetchone()[0], 1, "exactly one HNSW index")

            cursor.execute(
                "SELECT count(*) FROM pg_constraint WHERE conname LIKE %s",
                [f"{self.identity.leaf_table}%"],
            )
            self.assertEqual(
                cursor.fetchone()[0], 3, "dims + source + chunk, each once"
            )

            cursor.execute(
                f"SELECT count(*) FROM {ANN_LEAF_STATE_TABLE} "
                "WHERE application_scope_id = %s AND collection_id = %s",
                [self.scope.pk, self.collection.pk],
            )
            self.assertLessEqual(
                cursor.fetchone()[0], 1, "no duplicate leaf state"
            )

    def test_the_advisory_lock_is_not_mocked_away(self):
        source = inspect.getsource(pgvector_ann.provision_pgvector_ann_leaf)
        self.assertIn("pg_advisory_xact_lock", source)
        key = pgvector_ann._advisory_key(self.identity)
        self.assertEqual(key, pgvector_ann._advisory_key(self.identity))
        self.assertNotEqual(
            key,
            pgvector_ann._advisory_key(
                leaf_identity(self.scope.pk, self.collection.pk, E1_B)
            ),
        )
        self.assertTrue(-(2 ** 31) <= key < 2 ** 31)


@requires_postgres
class MigrationRoundTripTests(TestCase):
    """0029 -> 0030 -> 0029 -> 0030, with the extension deliberately surviving."""

    def test_the_migration_round_trips(self):
        from django.db.migrations.executor import MigrationExecutor

        def objects_present():
            with connection.cursor() as cursor:
                cursor.execute("SELECT to_regclass(%s)", [ANN_PARENT_TABLE])
                table = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT count(*) FROM pg_trigger WHERE tgname = %s",
                    ["ai_hub_pgv_embedding_trg"],
                )
                triggers = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT count(*) FROM pg_proc WHERE proname = %s",
                    ["ai_hub_pgv_bump"],
                )
                functions = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT count(*) FROM pg_extension WHERE extname = 'vector'"
                )
                extension = cursor.fetchone()[0]
            return table, triggers, functions, extension

        table, triggers, functions, extension = objects_present()
        self.assertIsNotNone(table)
        self.assertEqual(triggers, 1)
        self.assertEqual(functions, 1)
        self.assertEqual(extension, 1)

        executor = MigrationExecutor(connection)
        executor.migrate([("ai_hub", "0029_retrieval_audit_foundation")])
        table, triggers, functions, extension = objects_present()
        self.assertIsNone(table, "AI Hub ANN tables are removed")
        self.assertEqual(triggers, 0)
        self.assertEqual(functions, 0)
        self.assertEqual(
            extension, 1,
            "the vector extension is shared infrastructure and must survive",
        )

        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate([("ai_hub", "0030_pgvector_ann_foundation")])
        table, triggers, functions, extension = objects_present()
        self.assertIsNotNone(table, "0030 reapplies cleanly")
        self.assertEqual(triggers, 1)
        self.assertEqual(functions, 1)
