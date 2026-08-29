"""S-20 — local corpus embedding execution.

Every provider call in this module is mocked. **No real Ollama server, no real
network, no external API is ever contacted.**

The two properties worth reading the file for:

    locality and transport are INDEPENDENT axes
        declared_locality decides whether we may execute
        provider_type decides how we would execute

    a provider call happens outside the transaction
        so chunk identity (k1) and contract identity (e1) are snapshotted
        before dispatch and re-verified under lock afterwards
"""

import ast
import inspect
import math
from unittest import mock

from django.db import connection, transaction
from django.test import TestCase

import requests

from ai_hub.models import (
    ApplicationScope,
    EmbeddingModelConfig,
    KnowledgeChunkEmbedding,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    ProviderConfig,
    ProviderGrant,
)
from ai_hub.services import embedding_client, embedding_execution
from ai_hub.services.chunk_embedding_identity import (
    canonical_chunk_embedding_text,
    chunk_embedding_fingerprint,
)
from ai_hub.services.embedding_client import (
    EMBEDDING_TRANSPORTS,
    EmbeddingProviderExecutionError,
    EmbeddingProviderResult,
    ErrorCategory,
    embed_text_via_ollama,
    resolve_embedding_transport,
)
from ai_hub.services.embedding_contract import resolve_embedding_contract
from ai_hub.services.embedding_execution import (
    ChunkEmbeddingExecutionResult,
    EmbeddingExecutionError,
    ExecutionStatus,
    FailureCategory,
    index_chunk_embedding_local,
)
from ai_hub.services.vector_store import decode_vector

LOCALITY = ProviderConfig.DeclaredLocality
NORMALIZATION = EmbeddingModelConfig.Normalization

#: Where the execution module reaches the transport registry. Patching here
#: proves the call was or was not made, without any real socket.
TRANSPORT_PATH = "ai_hub.services.embedding_execution.resolve_embedding_transport"
OLLAMA_POST = "ai_hub.services.embedding_client.requests.post"


def fake_response(payload, *, status_code=200, invalid_json=False):
    response = mock.Mock()
    response.status_code = status_code
    if invalid_json:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = payload
    return response


class ExecutionFixtureMixin:
    """Re-entrant: several tests build more than one world per test method.

    Every globally-unique name is suffixed with a per-instance counter so a
    second `build_world()` cannot collide on `ApplicationScope.slug`,
    `KnowledgeCollection.name`, `ProviderConfig.name` or
    `EmbeddingModelConfig.name`.
    """

    _world_counter = 0

    def build_world(
        self,
        *,
        provider_type="ollama",
        locality=LOCALITY.LOCAL,
        grant=True,
        dimension=4,
        normalization=NORMALIZATION.NONE,
        status=KnowledgeDocument.Status.ACTIVE,
        section_title="Safety",
        content="Body text",
        base_url="http://ollama.internal:11434",
        max_input_chars=8000,
    ):
        self._world_counter += 1
        tag = self._world_counter
        self.scope = ApplicationScope.objects.create(
            name=f"App A{tag}", slug=f"app-a{tag}"
        )
        self.collection = KnowledgeCollection.objects.create(
            name=f"Collection A{tag}", application_scope=self.scope
        )
        self.document = KnowledgeDocument.objects.create(
            collection=self.collection, title="Doc", curated_text=content,
            status=status,
        )
        self.chunk = KnowledgeDocumentChunk.objects.create(
            document=self.document, chunk_index=1,
            section_title=section_title, content=content,
        )
        self.provider = ProviderConfig.objects.create(
            name=f"P{tag}", provider_type=provider_type, base_url=base_url,
            declared_locality=locality,
        )
        self.config = EmbeddingModelConfig.objects.create(
            name=f"embed{tag}", provider=self.provider, model_name="ollama/nomic-embed-text",
            model_revision="v1", vector_dimension=dimension,
            distance_metric=EmbeddingModelConfig.DistanceMetric.COSINE,
            normalization=normalization, max_input_chars=max_input_chars,
        )
        if grant is not None:
            ProviderGrant.objects.create(
                application_scope=self.scope, provider=self.provider,
                allow_embeddings=grant,
            )
        return self

    def transport_returning(self, values, *, hook=None):
        """A fake transport that records its calls and never touches a socket."""
        calls = []

        def fake_transport(*, provider, contract, text):
            calls.append({"provider": provider, "contract": contract, "text": text})
            if hook is not None:
                hook()
            return EmbeddingProviderResult(
                values=tuple(values), provider_type=provider.provider_type,
                provider_model="nomic-embed-text",
            )

        fake_transport.calls = calls
        return fake_transport

    def run_indexing(self, transport, **overrides):
        kwargs = dict(
            application_scope=self.scope, chunk=self.chunk,
            embedding_model_config=self.config,
        )
        kwargs.update(overrides)
        with mock.patch(TRANSPORT_PATH, return_value=transport):
            return index_chunk_embedding_local(**kwargs)


# ---------------------------------------------------------------------------
# The Ollama transport
# ---------------------------------------------------------------------------

class OllamaTransportTests(ExecutionFixtureMixin, TestCase):
    def setUp(self):
        self.build_world(dimension=3)
        self.contract = resolve_embedding_contract(self.config)

    def _post(self, payload, **kwargs):
        with mock.patch(OLLAMA_POST) as post:
            post.return_value = fake_response(payload, **kwargs)
            result = embed_text_via_ollama(
                provider=self.provider, contract=self.contract, text="Safety\n\nBody"
            )
        return result, post

    def test_the_request_is_exactly_the_contracted_shape(self):
        _, post = self._post({"embeddings": [[0.1, 0.2, 0.3]]})
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://ollama.internal:11434/api/embed")
        self.assertEqual(
            kwargs["json"],
            {
                "model": "nomic-embed-text",
                "input": "Safety\n\nBody",
                "truncate": False,
                "dimensions": 3,
            },
        )
        self.assertEqual(kwargs["timeout"], self.contract.request_timeout_seconds)

    def test_truncate_false_is_always_sent(self):
        """Load-bearing: Ollama truncates by default, which would embed text
        that `k1` does not describe."""
        _, post = self._post({"embeddings": [[0.1, 0.2, 0.3]]})
        self.assertIs(post.call_args.kwargs["json"]["truncate"], False)

    def test_the_ollama_prefix_is_stripped_for_the_request_only(self):
        _, post = self._post({"embeddings": [[0.1, 0.2, 0.3]]})
        self.assertEqual(post.call_args.kwargs["json"]["model"], "nomic-embed-text")
        # The persisted configuration is untouched...
        self.config.refresh_from_db()
        self.assertEqual(self.config.model_name, "ollama/nomic-embed-text")
        # ...and `e1` still uses the operator-declared contract.
        self.assertEqual(resolve_embedding_contract(self.config).e1, self.contract.e1)

    def test_a_model_name_without_the_prefix_is_unchanged(self):
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(
            model_name="mxbai-embed-large"
        )
        self.config.refresh_from_db()
        contract = resolve_embedding_contract(self.config)
        with mock.patch(OLLAMA_POST) as post:
            post.return_value = fake_response({"embeddings": [[0.1, 0.2, 0.3]]})
            embed_text_via_ollama(
                provider=self.provider, contract=contract, text="x"
            )
        self.assertEqual(post.call_args.kwargs["json"]["model"], "mxbai-embed-large")

    def test_a_valid_response_returns_bounded_values(self):
        result, _ = self._post({"embeddings": [[0.1, 0.2, 0.3]], "model": "nomic"})
        self.assertEqual(result.values, (0.1, 0.2, 0.3))
        self.assertEqual(result.provider_type, "ollama")
        self.assertEqual(result.provider_model, "nomic")

    # -- configuration ------------------------------------------------------

    def test_a_blank_base_url_is_refused_without_inventing_a_default(self):
        ProviderConfig.objects.filter(pk=self.provider.pk).update(base_url="")
        self.provider.refresh_from_db()
        with mock.patch(OLLAMA_POST) as post:
            with self.assertRaises(EmbeddingProviderExecutionError) as raised:
                embed_text_via_ollama(
                    provider=self.provider, contract=self.contract, text="x"
                )
        post.assert_not_called()
        self.assertEqual(
            raised.exception.category, ErrorCategory.INVALID_PROVIDER_CONFIGURATION
        )
        for invented in ("localhost", "127.0.0.1", "11434"):
            self.assertNotIn(invented, str(raised.exception))

    def test_no_credential_is_read_or_sent(self):
        ProviderConfig.objects.filter(pk=self.provider.pk).update(
            api_key_env_var="SECRET_EMBED_KEY"
        )
        self.provider.refresh_from_db()
        _, post = self._post({"embeddings": [[0.1, 0.2, 0.3]]})
        self.assertNotIn("headers", post.call_args.kwargs)
        self.assertNotIn(
            "api_key", str(post.call_args.kwargs.get("json", {}))
        )
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(inspect.getsource(embedding_client)))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("os", imported)

    # -- failures -----------------------------------------------------------

    def test_transport_failures_map_to_bounded_categories(self):
        cases = (
            (requests.Timeout("t"), ErrorCategory.PROVIDER_UNREACHABLE),
            (requests.ConnectionError("c"), ErrorCategory.PROVIDER_UNREACHABLE),
            (requests.RequestException("r"), ErrorCategory.PROVIDER_UNREACHABLE),
        )
        for exception, category in cases:
            with self.subTest(exception=type(exception).__name__):
                with mock.patch(OLLAMA_POST, side_effect=exception):
                    with self.assertRaises(EmbeddingProviderExecutionError) as raised:
                        embed_text_via_ollama(
                            provider=self.provider, contract=self.contract, text="x"
                        )
                self.assertEqual(raised.exception.category, category)

    def test_a_404_is_reported_as_model_not_found(self):
        with self.assertRaises(EmbeddingProviderExecutionError) as raised:
            self._post({}, status_code=404)
        self.assertEqual(raised.exception.category, ErrorCategory.MODEL_NOT_FOUND)

    def test_other_http_errors_are_status_only(self):
        for status in (400, 422, 500, 503):
            with self.subTest(status=status):
                with self.assertRaises(EmbeddingProviderExecutionError) as raised:
                    self._post({"error": "Safety\n\nBody was rejected"}, status_code=status)
                self.assertEqual(
                    raised.exception.category, ErrorCategory.PROVIDER_RETURNED_ERROR
                )
                # The provider body may echo the submitted text; never surface it.
                self.assertNotIn("Safety", str(raised.exception))
                self.assertNotIn("Body", str(raised.exception))

    def test_malformed_responses_are_refused(self):
        cases = {
            "invalid json": ({}, {"invalid_json": True}),
            "not an object": ([1, 2, 3], {}),
            "missing embeddings": ({"model": "x"}, {}),
            "embeddings not a list": ({"embeddings": "abc"}, {}),
            "zero embeddings": ({"embeddings": []}, {}),
            "two embeddings": ({"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}, {}),
            "inner not a list": ({"embeddings": [0.1]}, {}),
            "wrong dimension": ({"embeddings": [[0.1, 0.2]]}, {}),
            "non-numeric": ({"embeddings": [["a", "b", "c"]]}, {}),
            "bool component": ({"embeddings": [[True, False, True]]}, {}),
            "NaN": ({"embeddings": [[float("nan"), 0.1, 0.2]]}, {}),
            "+inf": ({"embeddings": [[float("inf"), 0.1, 0.2]]}, {}),
            "-inf": ({"embeddings": [[float("-inf"), 0.1, 0.2]]}, {}),
        }
        for label, (payload, kwargs) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(EmbeddingProviderExecutionError) as raised:
                    self._post(payload, **kwargs)
                self.assertEqual(
                    raised.exception.category, ErrorCategory.INVALID_PROVIDER_RESPONSE
                )
                self.assertNotIn("Safety", str(raised.exception))

    def test_the_result_carries_no_raw_response_or_input(self):
        for field in EmbeddingProviderResult.__dataclass_fields__:
            for forbidden in (
                "raw", "response", "body", "input", "text", "prompt",
                "api_key", "credential", "request",
            ):
                self.assertNotIn(forbidden, field)


# ---------------------------------------------------------------------------
# Transport capability vs locality
# ---------------------------------------------------------------------------

class TransportCapabilityTests(ExecutionFixtureMixin, TestCase):
    def test_only_ollama_is_implemented(self):
        self.assertEqual(
            set(EMBEDDING_TRANSPORTS), {ProviderConfig.ProviderType.OLLAMA}
        )

    def test_every_other_provider_type_is_unsupported(self):
        for provider_type in ("openai", "anthropic", "deepseek", "training", "other"):
            with self.subTest(provider_type=provider_type):
                provider = ProviderConfig(
                    provider_type=provider_type, declared_locality=LOCALITY.LOCAL
                )
                with self.assertRaises(EmbeddingProviderExecutionError) as raised:
                    resolve_embedding_transport(provider)
                self.assertEqual(
                    raised.exception.category,
                    ErrorCategory.UNSUPPORTED_EMBEDDING_TRANSPORT,
                )

    def test_transport_resolution_never_consults_locality_or_url(self):
        """Capability and security are independent axes."""
        referenced = {
            node.attr
            for node in ast.walk(
                ast.parse(inspect.getsource(resolve_embedding_transport).lstrip())
            )
            if isinstance(node, ast.Attribute)
        }
        for forbidden in ("declared_locality", "base_url", "name"):
            self.assertNotIn(forbidden, referenced)

    def test_the_client_never_parses_a_url_to_infer_locality(self):
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(inspect.getsource(embedding_client)))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        for forbidden in ("socket", "ipaddress", "urllib"):
            self.assertNotIn(forbidden, imported)


# ---------------------------------------------------------------------------
# The locality adversarial matrix
# ---------------------------------------------------------------------------

class LocalityAdversarialTests(ExecutionFixtureMixin, TestCase):
    def test_localhost_url_declared_external_is_refused(self):
        """Case 1: the URL looks local; the operator declared otherwise."""
        self.build_world(
            locality=LOCALITY.EXTERNAL, base_url="http://localhost:11434"
        )
        self.scope.allow_external_embedding_corpus_egress = True
        self.scope.save(update_fields=["allow_external_embedding_corpus_egress"])

        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category, FailureCategory.LOCAL_ONLY_EXECUTION_REQUIRED
        )
        self.assertEqual(len(transport.calls), 0)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_a_public_looking_url_declared_local_executes(self):
        """Case 2: the operator declaration is authoritative, not the URL."""
        self.build_world(
            locality=LOCALITY.LOCAL,
            base_url="https://some-public-looking-host.example",
        )
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        result = self.run_indexing(transport)
        self.assertEqual(result.status, ExecutionStatus.STORED)
        self.assertEqual(len(transport.calls), 1)

    def test_openai_declared_local_is_authorized_but_unsupported(self):
        """Case 3: locality and transport support are independent."""
        self.build_world(provider_type="openai", locality=LOCALITY.LOCAL)

        from ai_hub.services.embedding_egress import (
            PAYLOAD_CORPUS,
            ReasonCode,
            resolve_embedding_access,
        )

        decision = resolve_embedding_access(
            self.scope, self.provider,
            collection=self.collection, payload_kind=PAYLOAD_CORPUS,
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.ALLOWED_LOCAL)

        with mock.patch(OLLAMA_POST) as post:
            with self.assertRaises(EmbeddingProviderExecutionError) as raised:
                index_chunk_embedding_local(
                    application_scope=self.scope, chunk=self.chunk,
                    embedding_model_config=self.config,
                )
        self.assertEqual(
            raised.exception.category, ErrorCategory.UNSUPPORTED_EMBEDDING_TRANSPORT
        )
        post.assert_not_called()
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)


# ---------------------------------------------------------------------------
# Grant / policy matrix
# ---------------------------------------------------------------------------

class GrantPolicyMatrixTests(ExecutionFixtureMixin, TestCase):
    def _attempt(self, **world):
        self.build_world(**world)
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        try:
            self.run_indexing(transport)
            return None, transport
        except EmbeddingExecutionError as exc:
            return exc, transport

    def test_local_without_a_grant_makes_no_call(self):
        error, transport = self._attempt(grant=None)
        self.assertEqual(error.category, FailureCategory.EMBEDDING_NOT_AUTHORIZED)
        self.assertEqual(len(transport.calls), 0)

    def test_local_with_a_disabled_grant_makes_no_call(self):
        error, transport = self._attempt(grant=False)
        self.assertEqual(error.category, FailureCategory.EMBEDDING_NOT_AUTHORIZED)
        self.assertEqual(len(transport.calls), 0)

    def test_local_with_a_grant_executes(self):
        error, transport = self._attempt(grant=True)
        self.assertIsNone(error)
        self.assertEqual(len(transport.calls), 1)

    def test_unknown_locality_makes_no_call(self):
        error, transport = self._attempt(locality=LOCALITY.UNKNOWN, grant=True)
        self.assertEqual(error.category, FailureCategory.EMBEDDING_NOT_AUTHORIZED)
        self.assertEqual(len(transport.calls), 0)

    def test_external_without_egress_flags_makes_no_call(self):
        error, transport = self._attempt(locality=LOCALITY.EXTERNAL, grant=True)
        self.assertEqual(error.category, FailureCategory.EMBEDDING_NOT_AUTHORIZED)
        self.assertEqual(len(transport.calls), 0)

    def test_external_WITH_egress_flags_is_still_refused_by_S20(self):
        """The critical case: S-17 says yes, S-20 still does not implement it."""
        self.build_world(locality=LOCALITY.EXTERNAL, grant=True)
        self.scope.allow_external_embedding_corpus_egress = True
        self.scope.save(update_fields=["allow_external_embedding_corpus_egress"])

        from ai_hub.services.embedding_egress import (
            PAYLOAD_CORPUS,
            ReasonCode,
            resolve_embedding_access,
        )

        decision = resolve_embedding_access(
            self.scope, self.provider,
            collection=self.collection, payload_kind=PAYLOAD_CORPUS,
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason_code, ReasonCode.ALLOWED_EXTERNAL)

        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category, FailureCategory.LOCAL_ONLY_EXECUTION_REQUIRED
        )
        self.assertEqual(len(transport.calls), 0)

    def test_the_egress_module_was_not_modified(self):
        from ai_hub.services import embedding_egress

        source = inspect.getsource(embedding_egress)
        self.assertNotIn("embedding_execution", source)
        self.assertNotIn("embedding_client", source)


# ---------------------------------------------------------------------------
# Document eligibility
# ---------------------------------------------------------------------------

class DocumentEligibilityTests(ExecutionFixtureMixin, TestCase):
    def test_active_documents_are_indexed(self):
        self.build_world(status=KnowledgeDocument.Status.ACTIVE)
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(self.run_indexing(transport).status, ExecutionStatus.STORED)

    def test_draft_and_archived_make_no_call(self):
        for status in (
            KnowledgeDocument.Status.DRAFT, KnowledgeDocument.Status.ARCHIVED,
        ):
            with self.subTest(status=status):
                self.build_world(status=status)
                transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
                with self.assertRaises(EmbeddingExecutionError) as raised:
                    self.run_indexing(transport)
                self.assertEqual(
                    raised.exception.category, FailureCategory.DOCUMENT_NOT_ACTIVE
                )
                self.assertEqual(len(transport.calls), 0)
                self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_archival_after_indexing_leaves_the_vector_in_place(self):
        """Status is not part of `k1`/`e1`; retrieval excludes it separately."""
        self.build_world()
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        result = self.run_indexing(transport)
        KnowledgeDocument.objects.filter(pk=self.document.pk).update(
            status=KnowledgeDocument.Status.ARCHIVED
        )
        self.assertTrue(
            KnowledgeChunkEmbedding.objects.filter(pk=result.record_id).exists()
        )


# ---------------------------------------------------------------------------
# Input limits
# ---------------------------------------------------------------------------

class InputLimitTests(ExecutionFixtureMixin, TestCase):
    def test_exactly_at_the_limit_is_allowed(self):
        self.build_world(section_title="", content="x" * 50, max_input_chars=50)
        self.assertEqual(len(canonical_chunk_embedding_text(self.chunk)), 50)
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(self.run_indexing(transport).status, ExecutionStatus.STORED)

    def test_one_over_the_limit_is_refused_before_the_provider(self):
        self.build_world(section_title="", content="x" * 51, max_input_chars=50)
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category, FailureCategory.EMBEDDING_INPUT_TOO_LARGE
        )
        self.assertEqual(len(transport.calls), 0)

    def test_nothing_is_truncated_or_sliced(self):
        self.build_world(section_title="Title", content="y" * 40, max_input_chars=100)
        expected = canonical_chunk_embedding_text(self.chunk)
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        self.run_indexing(transport)
        self.assertEqual(transport.calls[0]["text"], expected)
        self.assertEqual(len(transport.calls[0]["text"]), len(expected))

    def test_empty_canonical_text_is_refused(self):
        self.build_world(section_title="", content="")
        self.assertEqual(canonical_chunk_embedding_text(self.chunk), "")
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category, FailureCategory.EMBEDDING_INPUT_EMPTY
        )
        self.assertEqual(len(transport.calls), 0)

    def test_whitespace_only_text_is_NOT_treated_as_empty(self):
        """`""` is compared exactly - `.strip()` would disagree with `k1`."""
        self.build_world(section_title="", content="   ")
        self.assertEqual(canonical_chunk_embedding_text(self.chunk), "   ")
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        result = self.run_indexing(transport)
        self.assertEqual(result.status, ExecutionStatus.STORED)
        self.assertEqual(transport.calls[0]["text"], "   ")

    def test_the_emptiness_check_does_not_strip(self):
        source = inspect.getsource(index_chunk_embedding_local)
        called = {
            node.func.attr
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for forbidden in ("strip", "lstrip", "rstrip"):
            self.assertNotIn(forbidden, called)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class NormalizationTests(ExecutionFixtureMixin, TestCase):
    def _stored_values(self, result):
        record = KnowledgeChunkEmbedding.objects.get(pk=result.record_id)
        return decode_vector(
            record.vector_bytes, expected_dimension=record.vector_dimension
        )

    def test_none_preserves_the_values(self):
        self.build_world(dimension=2, normalization=NORMALIZATION.NONE)
        result = self.run_indexing(self.transport_returning([3.0, 4.0]))
        values = self._stored_values(result)
        self.assertAlmostEqual(values[0], 3.0, places=5)
        self.assertAlmostEqual(values[1], 4.0, places=5)

    def test_none_does_not_normalize_just_because_the_metric_is_cosine(self):
        """Metric and normalization are separate explicit contract facts."""
        self.build_world(dimension=2, normalization=NORMALIZATION.NONE)
        self.assertEqual(
            self.config.distance_metric, EmbeddingModelConfig.DistanceMetric.COSINE
        )
        values = self._stored_values(
            self.run_indexing(self.transport_returning([3.0, 4.0]))
        )
        self.assertAlmostEqual(math.sqrt(values[0] ** 2 + values[1] ** 2), 5.0, places=5)

    def test_l2_normalizes_deterministically(self):
        self.build_world(dimension=2, normalization=NORMALIZATION.L2)
        values = self._stored_values(
            self.run_indexing(self.transport_returning([3.0, 4.0]))
        )
        self.assertAlmostEqual(values[0], 0.6, places=6)
        self.assertAlmostEqual(values[1], 0.8, places=6)
        self.assertAlmostEqual(math.sqrt(values[0] ** 2 + values[1] ** 2), 1.0, places=6)

    def test_a_zero_vector_under_l2_is_refused(self):
        self.build_world(dimension=2, normalization=NORMALIZATION.L2)
        transport = self.transport_returning([0.0, 0.0])
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category,
            FailureCategory.ZERO_VECTOR_CANNOT_L2_NORMALIZE,
        )
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_a_zero_vector_under_none_is_stored_unchanged(self):
        self.build_world(dimension=2, normalization=NORMALIZATION.NONE)
        values = self._stored_values(
            self.run_indexing(self.transport_returning([0.0, 0.0]))
        )
        self.assertEqual(values, (0.0, 0.0))

    def test_a_wrong_dimension_result_is_refused(self):
        self.build_world(dimension=3)
        transport = self.transport_returning([0.1, 0.2])
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category, FailureCategory.VECTOR_DIMENSION_MISMATCH
        )
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_non_finite_values_are_refused(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                self.build_world(dimension=2)
                transport = self.transport_returning([bad, 0.1])
                with self.assertRaises(EmbeddingExecutionError) as raised:
                    self.run_indexing(transport)
                self.assertEqual(
                    raised.exception.category, FailureCategory.VECTOR_NON_FINITE
                )
                self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_normalization_is_never_inferred(self):
        source = inspect.getsource(embedding_execution._normalize_vector)
        referenced = {
            node.attr
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Attribute)
        }
        for forbidden in ("distance_metric", "provider_type", "model_name"):
            self.assertNotIn(forbidden, referenced)


# ---------------------------------------------------------------------------
# Server-derived facts and the public API
# ---------------------------------------------------------------------------

class PublicApiTests(ExecutionFixtureMixin, TestCase):
    def setUp(self):
        self.build_world()

    def test_the_caller_supplies_only_scope_chunk_and_config(self):
        parameters = set(inspect.signature(index_chunk_embedding_local).parameters)
        self.assertEqual(
            parameters,
            {"application_scope", "chunk", "embedding_model_config"},
        )
        for forbidden in (
            "text", "collection", "provider", "k1", "e1",
            "vector_dimension", "vector_format", "normalization",
        ):
            self.assertNotIn(forbidden, parameters)

    def test_every_derived_fact_matches_the_canonical_source(self):
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        result = self.run_indexing(transport)
        contract = resolve_embedding_contract(self.config)
        self.assertEqual(result.collection_id, self.collection.pk)
        self.assertEqual(result.application_scope_id, self.scope.pk)
        self.assertEqual(result.provider_id, self.provider.pk)
        self.assertEqual(result.k1, chunk_embedding_fingerprint(self.chunk))
        self.assertEqual(result.e1, contract.e1)
        self.assertEqual(result.vector_dimension, contract.vector_dimension)
        self.assertEqual(
            transport.calls[0]["text"], canonical_chunk_embedding_text(self.chunk)
        )

    def test_a_foreign_scope_is_refused_before_any_call(self):
        other = ApplicationScope.objects.create(name="Other App", slug="other-app")
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport, application_scope=other)
        self.assertEqual(raised.exception.category, FailureCategory.SCOPE_MISMATCH)
        self.assertEqual(len(transport.calls), 0)

    def test_an_unsaved_chunk_is_refused(self):
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport, chunk=KnowledgeDocumentChunk())
        self.assertEqual(raised.exception.category, FailureCategory.CHUNK_MISSING)
        self.assertEqual(len(transport.calls), 0)

    def test_the_result_carries_no_text_vector_or_credential(self):
        for field in ChunkEmbeddingExecutionResult.__dataclass_fields__:
            for forbidden in (
                "text", "content", "values", "vector_bytes", "raw", "response",
                "api_key", "credential", "prompt", "embedding_text",
            ):
                self.assertNotIn(forbidden, field)
        self.assertEqual(
            set(ChunkEmbeddingExecutionResult.__dataclass_fields__),
            {
                "status", "record_id", "chunk_id", "collection_id",
                "application_scope_id", "embedding_model_config_id", "provider_id",
                "k1", "e1", "vector_dimension",
            },
        )

    def test_no_error_message_echoes_the_canonical_text(self):
        self.build_world(section_title="SECRET_TITLE", content="SECRET_BODY",
                         max_input_chars=5)
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        message = str(raised.exception)
        self.assertNotIn("SECRET_TITLE", message)
        self.assertNotIn("SECRET_BODY", message)

    def test_no_agent_authorization_is_required_or_consulted(self):
        source = inspect.getsource(embedding_execution)
        imported = {
            node.module
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        for forbidden in (
            "ai_hub.services.knowledge_authorization",
            "ai_hub.services.knowledge_retrieval",
        ):
            self.assertNotIn(forbidden, imported)
        referenced = {
            node.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Attribute)
        }
        for forbidden in ("knowledge_collections", "agents"):
            self.assertNotIn(forbidden, referenced)


# ---------------------------------------------------------------------------
# Already-current short circuit
# ---------------------------------------------------------------------------

class AlreadyCurrentTests(ExecutionFixtureMixin, TestCase):
    def setUp(self):
        self.build_world()

    def test_a_second_call_does_not_re_invoke_the_provider(self):
        first_transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        first = self.run_indexing(first_transport)
        self.assertEqual(first.status, ExecutionStatus.STORED)
        self.assertEqual(len(first_transport.calls), 1)

        second_transport = self.transport_returning([0.9, 0.9, 0.9, 0.9])
        second = self.run_indexing(second_transport)
        self.assertEqual(second.status, ExecutionStatus.ALREADY_CURRENT)
        self.assertEqual(len(second_transport.calls), 0)
        self.assertEqual(second.record_id, first.record_id)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 1)

    def test_authorization_is_checked_before_the_shortcut(self):
        """Chosen order: an unauthorized caller learns nothing about existence."""
        self.run_indexing(self.transport_returning([0.1, 0.2, 0.3, 0.4]))
        ProviderGrant.objects.filter(provider=self.provider).update(
            allow_embeddings=False
        )
        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category, FailureCategory.EMBEDDING_NOT_AUTHORIZED
        )
        self.assertEqual(len(transport.calls), 0)

    def test_changed_content_forces_a_new_provider_call(self):
        self.run_indexing(self.transport_returning([0.1, 0.2, 0.3, 0.4]))
        KnowledgeDocumentChunk.objects.filter(pk=self.chunk.pk).update(
            content="Rewritten body"
        )
        self.chunk.refresh_from_db()

        transport = self.transport_returning([0.5, 0.5, 0.5, 0.5])
        result = self.run_indexing(transport)
        self.assertEqual(result.status, ExecutionStatus.STORED)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 1)

    def test_a_changed_revision_creates_a_second_slot(self):
        first = self.run_indexing(self.transport_returning([0.1, 0.2, 0.3, 0.4]))
        EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(
            model_revision="v2"
        )
        self.config.refresh_from_db()

        transport = self.transport_returning([0.5, 0.5, 0.5, 0.5])
        second = self.run_indexing(transport)
        self.assertEqual(second.status, ExecutionStatus.STORED)
        self.assertEqual(len(transport.calls), 1)
        self.assertNotEqual(second.e1, first.e1)
        # The old slot survives - controlled reindex, not destruction.
        self.assertEqual(
            KnowledgeChunkEmbedding.objects.filter(chunk=self.chunk).count(), 2
        )
        self.assertTrue(
            KnowledgeChunkEmbedding.objects.filter(pk=first.record_id).exists()
        )

    def test_the_shortcut_uses_canonical_inspection(self):
        source = inspect.getsource(index_chunk_embedding_local)
        called = {
            node.func.id
            for node in ast.walk(ast.parse(source.lstrip()))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("inspect_vector_record", called)


# ---------------------------------------------------------------------------
# The in-flight races - the load-bearing tests
# ---------------------------------------------------------------------------

class InFlightRaceTests(ExecutionFixtureMixin, TestCase):
    def setUp(self):
        self.build_world()

    def test_a_chunk_mutated_during_the_provider_call_refuses_persistence(self):
        """vector(A) must never be stored under k1(B)."""
        def mutate():
            KnowledgeDocumentChunk.objects.filter(pk=self.chunk.pk).update(
                content="Rewritten while in flight"
            )

        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4], hook=mutate)
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category,
            FailureCategory.STALE_CHUNK_AFTER_PROVIDER_CALL,
        )
        self.assertEqual(len(transport.calls), 1)   # it WAS sent...
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)  # ...never stored

    def test_an_existing_row_is_byte_identical_after_a_stale_refusal(self):
        first = self.run_indexing(self.transport_returning([0.1, 0.2, 0.3, 0.4]))
        record = KnowledgeChunkEmbedding.objects.get(pk=first.record_id)
        original_bytes, original_k1 = bytes(record.vector_bytes), record.k1

        # Change the content so the shortcut does not fire, then change it AGAIN
        # while the provider is running.
        KnowledgeDocumentChunk.objects.filter(pk=self.chunk.pk).update(
            content="First edit"
        )
        self.chunk.refresh_from_db()

        def mutate():
            KnowledgeDocumentChunk.objects.filter(pk=self.chunk.pk).update(
                content="Second edit, mid-flight"
            )

        transport = self.transport_returning([0.9, 0.9, 0.9, 0.9], hook=mutate)
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category,
            FailureCategory.STALE_CHUNK_AFTER_PROVIDER_CALL,
        )
        record.refresh_from_db()
        self.assertEqual(bytes(record.vector_bytes), original_bytes)
        self.assertEqual(record.k1, original_k1)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 1)

    def test_a_revision_changed_during_the_call_refuses_persistence(self):
        """Never stamp an old vector with a new `e1`."""
        def mutate():
            EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(
                model_revision="v2-mid-flight"
            )

        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4], hook=mutate)
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category,
            FailureCategory.EMBEDDING_CONTRACT_CHANGED_AFTER_PROVIDER_CALL,
        )
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_a_dimension_changed_during_the_call_refuses_persistence(self):
        def mutate():
            EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(
                vector_dimension=8
            )

        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4], hook=mutate)
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category,
            FailureCategory.EMBEDDING_CONTRACT_CHANGED_AFTER_PROVIDER_CALL,
        )
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_config_deactivated_during_the_call_prevents_a_new_write(self):
        from ai_hub.services.embedding_contract import EmbeddingContractError

        def mutate():
            EmbeddingModelConfig.objects.filter(pk=self.config.pk).update(
                is_active=False
            )

        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4], hook=mutate)
        with self.assertRaises(EmbeddingContractError):
            self.run_indexing(transport)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_provider_deactivated_during_the_call_prevents_a_new_write(self):
        from ai_hub.services.embedding_contract import EmbeddingContractError

        def mutate():
            ProviderConfig.objects.filter(pk=self.provider.pk).update(is_active=False)

        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4], hook=mutate)
        with self.assertRaises(EmbeddingContractError):
            self.run_indexing(transport)
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_a_document_archived_during_the_call_refuses_persistence(self):
        def mutate():
            KnowledgeDocument.objects.filter(pk=self.document.pk).update(
                status=KnowledgeDocument.Status.ARCHIVED
            )

        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4], hook=mutate)
        with self.assertRaises(EmbeddingExecutionError) as raised:
            self.run_indexing(transport)
        self.assertEqual(
            raised.exception.category,
            FailureCategory.STALE_CHUNK_AFTER_PROVIDER_CALL,
        )
        self.assertEqual(KnowledgeChunkEmbedding.objects.count(), 0)

    def test_no_automatic_retry_or_re_embed_happens(self):
        calls = []

        def mutate():
            KnowledgeDocumentChunk.objects.filter(pk=self.chunk.pk).update(
                content=f"Edit {len(calls)}"
            )

        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4], hook=mutate)
        transport_calls = transport.calls
        with self.assertRaises(EmbeddingExecutionError):
            self.run_indexing(transport)
        # Exactly one dispatch. A refusal is not a trigger to try again.
        self.assertEqual(len(transport_calls), 1)


# ---------------------------------------------------------------------------
# Transaction shape
# ---------------------------------------------------------------------------

class TransactionShapeTests(ExecutionFixtureMixin, TestCase):
    def setUp(self):
        self.build_world()

    def test_the_provider_call_happens_outside_a_transaction(self):
        """Never hold a DB lock across model inference."""
        observed = {}

        def record_atomic():
            observed["in_atomic_block"] = connection.in_atomic_block

        transport = self.transport_returning([0.1, 0.2, 0.3, 0.4], hook=record_atomic)
        # TestCase wraps everything in a transaction, so compare against a
        # baseline captured the same way rather than asserting False naively.
        with mock.patch(TRANSPORT_PATH, return_value=transport):
            baseline = connection.in_atomic_block
            index_chunk_embedding_local(
                application_scope=self.scope, chunk=self.chunk,
                embedding_model_config=self.config,
            )
        self.assertEqual(observed["in_atomic_block"], baseline)

    def test_the_provider_call_is_not_nested_inside_the_commit_atomic_block(self):
        """Structural: the transport call precedes `transaction.atomic()`."""
        tree = ast.parse(inspect.getsource(index_chunk_embedding_local).lstrip())
        function = tree.body[0]

        atomic_line = None
        for node in ast.walk(function):
            if (
                isinstance(node, ast.With)
                and any(
                    isinstance(item.context_expr, ast.Call)
                    and getattr(item.context_expr.func, "attr", "") == "atomic"
                    for item in node.items
                )
            ):
                atomic_line = node.lineno
                # Nothing inside the atomic block may call the transport.
                inner_calls = {
                    getattr(inner.func, "id", "")
                    for inner in ast.walk(node)
                    if isinstance(inner, ast.Call)
                }
                self.assertNotIn("transport", inner_calls)
        self.assertIsNotNone(atomic_line, "the commit path must be atomic")

        transport_line = next(
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "transport"
        )
        self.assertLess(transport_line, atomic_line)

    def test_the_storage_section_is_atomic(self):
        source = inspect.getsource(index_chunk_embedding_local)
        self.assertIn("transaction.atomic()", source)
        self.assertIn("select_for_update", source)


# ---------------------------------------------------------------------------
# Canonical storage boundary, and absence
# ---------------------------------------------------------------------------

class BoundaryAndAbsenceTests(ExecutionFixtureMixin, TestCase):
    def test_persistence_goes_through_the_canonical_S19_store(self):
        source = inspect.getsource(embedding_execution)
        called = {
            getattr(node.func, "id", "")
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
        }
        self.assertIn("store_chunk_vector", called)
        referenced = {
            node.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Attribute)
        }
        for forbidden in ("vector_bytes", "vector_format"):
            self.assertNotIn(forbidden, referenced)

    def test_k1_e1_and_f32le1_are_not_reimplemented(self):
        for module in (embedding_execution, embedding_client):
            with self.subTest(module=module.__name__):
                imported = {
                    alias.name.split(".")[0]
                    for node in ast.walk(ast.parse(inspect.getsource(module)))
                    if isinstance(node, ast.Import)
                    for alias in node.names
                }
                for forbidden in ("hashlib", "struct"):
                    self.assertNotIn(forbidden, imported)

    def test_no_semantic_or_query_api_is_exposed(self):
        for module in (embedding_execution, embedding_client):
            with self.subTest(module=module.__name__):
                for forbidden in (
                    "embed_query", "semantic_query_vector", "search_embedding",
                    "similarity_search", "nearest_neighbors", "rank", "top_k",
                    "cosine_similarity", "hybrid_search", "rerank",
                ):
                    self.assertFalse(hasattr(module, forbidden), forbidden)

    def test_no_vector_backend_or_ann_dependency(self):
        for module in (embedding_execution, embedding_client):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                imported = {
                    alias.name.split(".")[0]
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Import)
                    for alias in node.names
                }
                for forbidden in (
                    "pgvector", "chromadb", "faiss", "numpy", "scipy",
                ):
                    self.assertNotIn(forbidden, imported)

    def test_no_signals_or_background_queue(self):
        for module in (embedding_execution, embedding_client):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                imported = {
                    node.module
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.ImportFrom) and node.module
                }
                referenced = {
                    node.attr
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Attribute)
                }
                for forbidden in (
                    "post_save", "pre_save", "post_delete", "receiver",
                    "celery", "delay", "apply_async",
                ):
                    self.assertNotIn(forbidden, referenced)
                self.assertNotIn("django.db.models.signals", imported)

    def test_the_lifecycle_contracts_are_untouched(self):
        for module in (embedding_execution, embedding_client):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                referenced = {
                    node.attr
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Attribute)
                }
                for forbidden in (
                    "chunk_authority_mode", "generation_input_fingerprint",
                    "generation_chunk_set_fingerprint",
                ):
                    self.assertNotIn(forbidden, referenced)

    def test_completion_modules_are_untouched_by_embedding_execution(self):
        import ai_hub.services.agent_runtime as agent_runtime
        import ai_hub.services.litellm_client as litellm_client
        import ai_hub.services.provider_registry as provider_registry

        for module in (provider_registry, litellm_client, agent_runtime):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("embedding_execution", source)
                self.assertNotIn("embedding_client", source)
                self.assertNotIn("ProviderGrant", source)


# ---------------------------------------------------------------------------
# Completion regression
# ---------------------------------------------------------------------------

class CompletionRegressionTests(TestCase):
    def test_completion_resolution_is_unchanged(self):
        from decimal import Decimal

        from ai_hub.models import ModelConfig
        from ai_hub.services.provider_registry import resolve_model_config

        provider = ProviderConfig.objects.create(
            name="Training", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        model_config = ModelConfig.objects.create(
            provider=provider, model_name="training",
            temperature_default=Decimal("0.30"),
        )
        resolved = resolve_model_config(model_config)
        self.assertEqual(resolved["model"], "training")
        self.assertEqual(resolved["temperature"], 0.30)
        for absent in ("embedding", "vector_dimension", "e1", "k1", "grant"):
            self.assertNotIn(absent, resolved)

    def test_completion_does_not_require_a_provider_grant(self):
        """Chat must not be routed through embedding permission."""
        from ai_hub.models import ModelConfig
        from ai_hub.services.provider_registry import resolve_model_config

        provider = ProviderConfig.objects.create(
            name="Chat", provider_type=ProviderConfig.ProviderType.TRAINING
        )
        model_config = ModelConfig.objects.create(
            provider=provider, model_name="training"
        )
        self.assertFalse(ProviderGrant.objects.exists())
        self.assertEqual(resolve_model_config(model_config)["model"], "training")
