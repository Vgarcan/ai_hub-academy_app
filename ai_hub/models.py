from decimal import Decimal

from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


# === REUSABLE AI PIPELINE CORE =============================================
# These models are generic orchestration primitives. They can be copied to
# another Django project without depending on Dreamsreader domain models.
class ApplicationScope(models.Model):
    """The root security boundary for ONE application hosted by this AI Hub.

    One AI Hub installation may serve several independent applications out of a
    single database and runtime. This model is the only thing that says which
    application a resource belongs to, and it is deliberately generic: Core
    never learns what a scope MEANS. A host maps its own domain object onto a
    scope from outside; no host, project or domain concept may appear here.

    S-14 establishes OWNERSHIP only. Answering "which scope owns this?" is in
    scope; answering "may this agent retrieve this collection in this
    execution?" is S-15's effective authorization policy and is deliberately
    NOT implemented here.

    There is no runtime default and no implicit scope. A resource without an
    explicitly supplied scope is a resource whose security boundary nobody
    decided, and the schema refuses it.
    """

    name = models.CharField(max_length=140, unique=True)
    # Stable machine identity. Names are edited; slugs are referenced by
    # migrations, fixtures and operators, so this is the durable handle.
    slug = models.SlugField(max_length=140, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    # --- External embedding egress policy (S-17) ---------------------------
    # The scope OWNS this decision. A collection may narrow it; nothing may
    # widen it. Both default to DENY, so a fresh install and every migrated
    # legacy scope authorize no external embedding egress at all until an
    # operator explicitly decides otherwise.
    #
    # These govern the FUTURE embedding capability only. Existing chat and
    # completion execution is deliberately untouched by them.
    allow_external_embedding_corpus_egress = models.BooleanField(default=False)
    # Query text is not curated: it may carry a user message, a client name or
    # a pasted document, and no review step exists. So it needs its own
    # permission AND can never be broader than the corpus permission - see the
    # constraint below, which is enforced in the database, not just in `clean`.
    allow_external_embedding_query_egress = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "1.0 Application scope"
        verbose_name_plural = "1.0 Application scopes - isolation boundaries"
        constraints = [
            # query egress ⊆ corpus egress. Expressed as: NOT (query AND NOT
            # corpus). A raw ORM write cannot produce the invalid combination.
            models.CheckConstraint(
                condition=models.Q(allow_external_embedding_query_egress=False)
                | models.Q(allow_external_embedding_corpus_egress=True),
                name="ai_hub_query_egress_subset_of_corpus",
            ),
        ]

    def __str__(self):
        return self.name

    def clean(self):
        if not str(self.name or "").strip():
            raise ValidationError({"name": "Application scope name is required."})
        if not str(self.slug or "").strip():
            raise ValidationError({"slug": "Application scope slug is required."})
        if (
            self.allow_external_embedding_query_egress
            and not self.allow_external_embedding_corpus_egress
        ):
            raise ValidationError(
                {
                    "allow_external_embedding_query_egress": (
                        "External query egress cannot be broader than external "
                        "corpus egress. Enable corpus egress first, or leave "
                        "both disabled."
                    )
                }
            )


class ProviderConfig(models.Model):
    class DeclaredLocality(models.TextChoices):
        """Where this provider sits relative to the deployment's trust boundary.

        An OPERATOR DECLARATION, and the only authoritative source for it. Core
        must never infer locality from `provider_type`, `base_url`, hostname, IP
        address, scheme, provider name or model name - all of them are wrong in
        realistic deployments:

            provider_type=ollama, declared_locality=external   (hosted Ollama)
            provider_type=openai, declared_locality=local      (internal gateway)

        There is deliberately no DNS lookup, no RFC1918 check and no localhost
        detection anywhere in this codebase.
        """

        # Not yet classified. Embedding use FAILS CLOSED - this is the default
        # for every existing and new row, and it is never auto-upgraded.
        UNKNOWN = "unknown", "Unknown (not classified - embeddings denied)"
        # Inside the approved internal trust boundary. NOT necessarily the same
        # machine, localhost or a particular vendor; it may be another server
        # the operator controls.
        LOCAL = "local", "Local (inside the declared trust boundary)"
        # Sending text here crosses the trust boundary.
        EXTERNAL = "external", "External (crosses the trust boundary)"

    class ProviderType(models.TextChoices):
        OPENAI = "openai", "OpenAI"
        OLLAMA = "ollama", "Ollama"
        DEEPSEEK = "deepseek", "DeepSeek"
        ANTHROPIC = "anthropic", "Anthropic"
        # Deterministic stub provider for demos and tests — no real API key needed
        TRAINING = "training", "Training (stub)"
        OTHER = "other", "Other"

    name = models.CharField(max_length=120, unique=True)
    provider_type = models.CharField(max_length=30, choices=ProviderType.choices)
    base_url = models.URLField(blank=True)
    api_key_env_var = models.CharField(max_length=100, blank=True)
    # S-17. Governs the FUTURE embedding capability only; existing completion
    # execution does not consult it, so an unclassified provider keeps working
    # for chat exactly as before while being refused for embeddings.
    declared_locality = models.CharField(
        max_length=20,
        choices=DeclaredLocality.choices,
        default=DeclaredLocality.UNKNOWN,
    )
    default_timeout = models.PositiveIntegerField(default=60)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "1.1 Provider config"
        verbose_name_plural = "1.1 Provider configs - AI service accounts"

    def __str__(self):
        return self.name


class ProviderGrant(models.Model):
    """Permission for ONE application scope to use ONE provider for embeddings.

    **Provider definition is not provider permission.** A `ProviderConfig` row
    describes an endpoint the deployment has configured; it never implies that
    any particular application may send that provider its data. This model is
    the explicit, per-scope grant that does.

    A row with `allow_embeddings=False` grants nothing - it exists so an
    operator can record a considered "no" rather than leaving an absence that
    reads the same as "never asked".

    Scope: FUTURE embedding use (S-17/S-18). Existing chat and completion
    execution is deliberately NOT routed through grants; doing so would be a
    much wider authorization redesign.
    """

    # A grant is subordinate configuration owned by its scope: deleting the
    # application removes its authorization records with it.
    application_scope = models.ForeignKey(
        ApplicationScope,
        on_delete=models.CASCADE,
        related_name="provider_grants",
    )
    # PROTECT the other way round: a globally configured provider that scopes
    # still explicitly reference should require intentional cleanup, not have
    # its authorization records silently erased underneath it.
    provider = models.ForeignKey(
        ProviderConfig,
        on_delete=models.PROTECT,
        related_name="scope_grants",
    )
    allow_embeddings = models.BooleanField(default=False)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["application_scope__name", "provider__name"]
        verbose_name = "1.2b Provider grant"
        verbose_name_plural = "1.2b Provider grants - scope embedding permission"
        constraints = [
            models.UniqueConstraint(
                fields=["application_scope", "provider"],
                name="ai_hub_unique_scope_provider_grant",
            ),
        ]

    def __str__(self):
        state = "embeddings allowed" if self.allow_embeddings else "no embedding use"
        return f"{self.application_scope.name} -> {self.provider.name} ({state})"


class ModelConfig(models.Model):
    provider = models.ForeignKey(ProviderConfig, on_delete=models.CASCADE, related_name="models")
    model_name = models.CharField(max_length=140)
    temperature_default = models.DecimalField(max_digits=4, decimal_places=2, default=Decimal("0.70"))
    max_tokens_default = models.PositiveIntegerField(default=1000)
    supports_tools = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("provider", "model_name")]
        ordering = ["provider__name", "model_name"]
        verbose_name = "1.2 Model config"
        verbose_name_plural = "1.2 Model configs - model choices"

    def __str__(self):
        return f"{self.provider.name} / {self.model_name}"

    def clean(self):
        if self.is_active and not self.provider.is_active:
            raise ValidationError("Cannot activate a model with an inactive provider.")
        # Keep one explicit namespace for deterministic Training models. Provider
        # routing itself is selected independently from ProviderConfig.provider_type.
        if self.provider.provider_type == ProviderConfig.ProviderType.TRAINING:
            if self.model_name != "training" and not self.model_name.startswith("training/"):
                raise ValidationError(
                    {
                        "model_name": (
                            "Training-provider models must be named 'training' or start "
                            "with 'training/' (e.g. 'training/assistant')."
                        )
                    }
                )


class EmbeddingModelConfig(models.Model):
    """What an embedding model MEANS to AI Hub - its vector-space contract.

    Deliberately NOT an extension of `ModelConfig`. A completion model is
    described by temperature, max tokens and tool support; an embedding model is
    described by dimension, distance metric and normalization. Neither set is
    meaningful for the other, so sharing one table would give every row a
    handful of misleading fields and still lack the ones that matter.

    A GLOBAL reusable definition. There is deliberately no `application_scope`,
    `knowledge_collection`, `agent` or `workspace` FK here: permission to use a
    provider already exists as `ProviderGrant` plus the ApplicationScope egress
    policy (S-17), and a second permission system would be one more thing that
    can disagree with the first.

    Configuring an embedding model grants nothing. Future embedding execution
    will require BOTH a resolved contract (this model) AND an allowed
    `EmbeddingAccessDecision`.
    """

    class DistanceMetric(models.TextChoices):
        """Core semantics, not backend operator names.

        A future vector backend maps these to its own implementation. There are
        deliberately no `pgvector_cosine` / `l2_ops` / `vector_ip_ops` members -
        binding the contract to one backend's vocabulary would make the contract
        unportable and would leak a storage decision into a semantic one.
        """

        COSINE = "cosine", "Cosine"
        DOT_PRODUCT = "dot_product", "Dot product"
        EUCLIDEAN = "euclidean", "Euclidean"

    class Normalization(models.TextChoices):
        """What a future execution pipeline must do before a vector is valid."""

        NONE = "none", "None (no additional transform)"
        L2 = "l2", "L2 normalize"

    # An operator-facing administrative label. Deliberately NOT part of `e1`:
    # renaming a configuration must never invalidate vectors produced under it.
    name = models.CharField(max_length=140, unique=True)
    # PROTECT, stricter than the legacy completion `ModelConfig` relationship
    # and intentionally so: an embedding contract must not silently disappear
    # because someone tried to delete a provider definition. Vectors produced
    # under a contract outlive attempts to tidy up configuration.
    provider = models.ForeignKey(
        ProviderConfig,
        on_delete=models.PROTECT,
        related_name="embedding_models",
    )
    model_name = models.CharField(max_length=140)
    # Operator-declared stable identity for the model REVISION, and therefore
    # for the vector space. Load-bearing, because the provider endpoint is not
    # part of `e1`: if the backing model may produce an incompatible vector
    # space, the operator MUST change this even when `model_name` is unchanged.
    # Core never contacts the provider to verify it, and never invents a
    # floating "latest" / "current" / provider-default revision.
    model_revision = models.CharField(max_length=140)
    vector_dimension = models.PositiveIntegerField()
    # No default on purpose: the operator decides the contract explicitly.
    distance_metric = models.CharField(max_length=20, choices=DistanceMetric.choices)
    normalization = models.CharField(max_length=20, choices=Normalization.choices)
    # A Core safety ceiling in CHARACTERS, not tokens. AI Hub has no embedding
    # tokenizer contract, and claiming a token count would be false precision.
    #
    # Future rule, recorded here because the enforcement point does not exist
    # yet: input over this limit must be REJECTED, never silently truncated.
    # Truncating would change the text being embedded without saying so.
    max_input_chars = models.PositiveIntegerField(default=8000)
    # Operational only, independent of completion behaviour. Not part of `e1`.
    request_timeout_seconds = models.PositiveIntegerField(default=60)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "1.2c Embedding model config"
        verbose_name_plural = "1.2c Embedding model configs - vector-space contracts"

    def __str__(self):
        return f"{self.name} ({self.model_name} @ {self.model_revision})"

    def clean(self):
        # Two rows may legitimately share every `e1` field while differing
        # operationally - a short-timeout and a long-timeout variant of the same
        # vector space. That is CORRECT, so there is deliberately no uniqueness
        # constraint over the semantic fields.
        errors = {}
        for field in ("name", "model_name", "model_revision"):
            if not str(getattr(self, field, "") or "").strip():
                errors[field] = f"{field.replace('_', ' ').capitalize()} is required."
        for field in ("vector_dimension", "max_input_chars", "request_timeout_seconds"):
            value = getattr(self, field, None)
            if value is not None and value <= 0:
                errors[field] = (
                    f"{field.replace('_', ' ').capitalize()} must be greater than zero."
                )
        if self.distance_metric not in self.DistanceMetric.values:
            errors["distance_metric"] = "Select a supported distance metric."
        if self.normalization not in self.Normalization.values:
            errors["normalization"] = "Select a supported normalization contract."
        if self.is_active and self.provider_id and not self.provider.is_active:
            errors["is_active"] = (
                "Cannot activate an embedding model whose provider is inactive."
            )
        if errors:
            raise ValidationError(errors)


class ToolDefinition(models.Model):
    class ToolKind(models.TextChoices):
        HTTP = "http", "HTTP"
        PYTHON_CALLABLE = "python_callable", "Python Callable"
        PROMPT_MACRO = "prompt_macro", "Prompt Macro"

    class RiskLevel(models.TextChoices):
        LOW = "low", "Low"
        MEDIUM = "medium", "Medium"
        HIGH = "high", "High"

    class OperationMode(models.TextChoices):
        READ = "read", "Read"
        DRAFT_WRITE = "draft_write", "Draft write"
        STATE_WRITE = "state_write", "State write"
        EXTERNAL_WRITE = "external_write", "External write"
        EXECUTE = "execute", "Execute"

    name = models.CharField(max_length=120, unique=True)
    label = models.CharField(max_length=160, blank=True)
    description = models.TextField(blank=True)
    tool_kind = models.CharField(max_length=30, choices=ToolKind.choices)
    input_schema = models.JSONField(default=dict, blank=True)
    output_schema = models.JSONField(default=dict, blank=True)
    config = models.JSONField(default=dict, blank=True)
    risk_level = models.CharField(max_length=20, choices=RiskLevel.choices, default=RiskLevel.LOW)
    operation_mode = models.CharField(max_length=30, choices=OperationMode.choices, default=OperationMode.READ)
    requires_approval = models.BooleanField(default=False)
    is_system_tool = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "1.6 Tool definition"
        verbose_name_plural = "1.6 Tool definitions - optional agent tools"

    def __str__(self):
        return self.name

    def clean(self):
        for field_name in ("input_schema", "output_schema", "config"):
            if not isinstance(getattr(self, field_name), dict):
                raise ValidationError({field_name: "Must be a JSON object."})
        if self.tool_kind == self.ToolKind.HTTP and isinstance(self.config, dict):
            from ai_hub.services.http_tool_policy import build_http_tool_configuration

            build_http_tool_configuration(self)


class Toolbox(models.Model):
    name = models.CharField(max_length=140, unique=True)
    slug = models.SlugField(max_length=140, unique=True)
    label = models.CharField(max_length=160)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "1.7 Toolbox"
        verbose_name_plural = "1.7 Toolboxes - reusable tool groups"

    def __str__(self):
        return self.label or self.name


class ToolboxTool(models.Model):
    toolbox = models.ForeignKey(Toolbox, on_delete=models.CASCADE, related_name="tool_entries")
    tool = models.ForeignKey(ToolDefinition, on_delete=models.CASCADE, related_name="toolbox_entries")
    is_enabled = models.BooleanField(default=True)
    default_enabled = models.BooleanField(default=True)
    display_order = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["toolbox__name", "display_order", "tool__name"]
        verbose_name = "1.8 Toolbox tool"
        verbose_name_plural = "1.8 Toolbox tools - toolbox membership"
        constraints = [
            models.UniqueConstraint(fields=["toolbox", "tool"], name="ai_hub_unique_toolbox_tool"),
        ]

    def __str__(self):
        status = "enabled" if self.is_enabled else "disabled"
        return f"{self.toolbox} -> {self.tool.name} ({status})"


class KnowledgeCollection(models.Model):
    class ExternalEmbeddingEgressPolicy(models.TextChoices):
        """A collection may only NARROW its scope's external egress decision.

        There is deliberately no `allow` / `force_allow` / `override_allow`
        member. A collection can accept the scope's decision or refuse external
        egress outright; it can never turn a scope-level DENY into an ALLOW.
        The security root stays the ApplicationScope.
        """

        INHERIT = "inherit", "Inherit the application scope decision"
        DENY = "deny", "Deny external embedding egress for this collection"

    # Root-owned: a collection belongs to exactly one application scope.
    # PROTECT, because a scope that still owns Knowledge must not vanish
    # through an accidental cascade - deleting a security boundary is a
    # deliberate operator act, not a side effect.
    application_scope = models.ForeignKey(
        ApplicationScope,
        on_delete=models.PROTECT,
        related_name="knowledge_collections",
    )
    # Global uniqueness is retained DELIBERATELY in S-14 and is stricter than
    # multi-application isolation requires. Name-based resolution paths still
    # exist in the runtime and assume a global namespace; relaxing this to
    # UniqueConstraint(application_scope, name) before those paths are
    # scope-aware would create ambiguous identity resolution. A later slice may
    # change it, and only then.
    name = models.CharField(max_length=140, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    # S-17. `inherit` grants nothing by itself: every scope migrates with both
    # external egress flags FALSE, so inheriting a DENY is still a DENY.
    external_embedding_egress_policy = models.CharField(
        max_length=20,
        choices=ExternalEmbeddingEgressPolicy.choices,
        default=ExternalEmbeddingEgressPolicy.INHERIT,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "1.3 Knowledge collection"
        verbose_name_plural = "1.3 Knowledge collections - knowledge groups"

    def __str__(self):
        return self.name


class KnowledgeDocument(models.Model):
    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        ACTIVE = "active", "Active"
        ARCHIVED = "archived", "Archived"

    class ChunkAuthorityMode(models.TextChoices):
        """Who is authoritative for this document's retrieval chunk set.

        Authority is a property of the chunk SET, not of an individual chunk,
        so it lives on the document that owns the set. A per-chunk mode would
        permit an incoherent half-derived document with no defined
        regeneration semantics.

        UNKNOWN  - legacy or ungoverned. Generation provenance may be absent
                   and must not be trusted. Never auto-repaired, never
                   silently reclassified. This is the safe default for every
                   pre-existing row and for any raw ORM write, until a future
                   governed write explicitly declares authority.
        DERIVED  - the set was generated from a source representation.
                   NOTE: this alone NEVER authorizes overwrite. Safe
                   regeneration also requires the current generation inputs to
                   match `generation_input_fingerprint` AND the current chunks
                   to match `generation_chunk_set_fingerprint`.
        EXPLICIT - the chunk set itself was deliberately authored and is
                   authoritative for retrieval. Generation provenance is
                   normally blank, and `curated_text` changing does not make
                   it stale.
        """

        UNKNOWN = "unknown", "Unknown (legacy or ungoverned)"
        DERIVED = "derived", "Derived from a source"
        EXPLICIT = "explicit", "Explicitly authored"

    collection = models.ForeignKey(KnowledgeCollection, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=180)
    curated_text = models.TextField(blank=True)
    source_file = models.FileField(upload_to="ai_hub/knowledge/", blank=True)
    tags = models.JSONField(default=list, blank=True)
    language = models.CharField(max_length=20, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)
    notes = models.TextField(blank=True)
    # --- Lifecycle facts. Persisted only; no runtime code reads them yet. ----
    # Deliberately unconstrained beyond types and defaults: the governed
    # mutation boundary does not exist, raw ORM stays an intentional escape
    # hatch, and a future preflight must be able to REPORT inconsistent
    # combinations rather than be prevented from observing them.
    chunk_authority_mode = models.CharField(
        max_length=20,
        choices=ChunkAuthorityMode.choices,
        default=ChunkAuthorityMode.UNKNOWN,
    )
    # Versioned fingerprints ("i1:<sha256>" / "c1:<sha256>"); see
    # ai_hub.services.knowledge_lifecycle. Blank unless mode is DERIVED.
    #
    # generation_input_fingerprint covers the COMPLETE mutable input set the
    # recorded generator reads - for curated_text_single_chunk that is BOTH
    # `title` (which becomes section_title) and `curated_text` (which becomes
    # content). A curated_text-only fingerprint could not detect a title change.
    generation_input_fingerprint = models.CharField(max_length=80, blank=True)
    generation_chunk_set_fingerprint = models.CharField(max_length=80, blank=True)
    generator_identity = models.CharField(max_length=64, blank=True)
    # Null - not zero - when no generator applies. A fake version would be a
    # provenance claim about a generator that never ran.
    generator_version = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["collection__name", "title"]
        verbose_name = "1.4 Knowledge document"
        verbose_name_plural = "1.4 Knowledge documents - curated files and text"
        indexes = [
            models.Index(fields=["collection", "status"], name="ai_hub_kdoc_collect_status_idx"),
            models.Index(fields=["language", "status"], name="ai_hub_kdoc_lang_status_idx"),
        ]

    def __str__(self):
        return f"{self.title} ({self.collection.name})"


class KnowledgeDocumentChunk(models.Model):
    document = models.ForeignKey(KnowledgeDocument, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.PositiveIntegerField()
    section_title = models.CharField(max_length=240, blank=True)
    content = models.TextField()
    token_estimate = models.PositiveIntegerField(default=0)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["document__collection__name", "document__title", "chunk_index"]
        verbose_name = "1.5 Knowledge document chunk"
        verbose_name_plural = "1.5 Knowledge document chunks - retrievable sections"
        constraints = [
            models.UniqueConstraint(fields=["document", "chunk_index"], name="ai_hub_unique_doc_chunk_index"),
        ]
        indexes = [
            models.Index(fields=["document", "chunk_index"], name="aihub_kchunk_doc_idx"),
            models.Index(fields=["section_title"], name="aihub_kchunk_section_idx"),
        ]

    def __str__(self):
        label = self.section_title or f"Chunk {self.chunk_index}"
        return f"{self.document.title} - {label}"

    def clean(self):
        if not isinstance(self.metadata, dict):
            raise ValidationError({"metadata": "Chunk metadata must be a JSON object."})


class AgentProfile(models.Model):
    class ExecutionMode(models.TextChoices):
        SYNC = "sync", "Sync"
        ASYNC = "async", "Async"
        INHERIT = "inherit", "Inherit"

    # Root-owned, and SINGLE-scope in V1. An AgentProfile is mutable
    # operational configuration - Knowledge assignments, model choice, tools,
    # runtime settings - so sharing one across security boundaries would make
    # its authorization ambiguous. If reusable agents are needed later they
    # need a separate template/definition abstraction; that is not S-14.
    application_scope = models.ForeignKey(
        ApplicationScope,
        on_delete=models.PROTECT,
        related_name="agent_profiles",
    )
    # Globally unique in S-14 by deliberate compatibility choice; see the note
    # on KnowledgeCollection.name.
    name = models.CharField(max_length=140, unique=True)
    role = models.CharField(max_length=140)
    system_prompt = models.TextField(blank=True)
    model_config = models.ForeignKey(ModelConfig, on_delete=models.PROTECT, related_name="agents")
    tools = models.ManyToManyField(ToolDefinition, blank=True, related_name="agents")
    knowledge_collections = models.ManyToManyField(KnowledgeCollection, blank=True, related_name="agents")
    knowledge_max_chars = models.PositiveIntegerField(default=6000)
    input_contract = models.JSONField(default=dict, blank=True)
    output_contract = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    execution_mode = models.CharField(max_length=20, choices=ExecutionMode.choices, default=ExecutionMode.INHERIT)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "1.9 Agent profile"
        verbose_name_plural = "1.9 Agent profiles - prompts and roles"

    def __str__(self):
        return self.name

    def clean(self):
        if self.is_active and (not self.model_config.is_active or not self.model_config.provider.is_active):
            raise ValidationError("Cannot activate an agent with inactive model/provider.")


class AgentToolboxAssignment(models.Model):
    agent = models.ForeignKey(AgentProfile, on_delete=models.CASCADE, related_name="toolbox_assignments")
    toolbox = models.ForeignKey(Toolbox, on_delete=models.CASCADE, related_name="agent_assignments")
    is_enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["agent__name", "toolbox__name"]
        verbose_name = "1.10 Agent toolbox assignment"
        verbose_name_plural = "1.10 Agent toolbox assignments - reusable access"
        constraints = [
            models.UniqueConstraint(fields=["agent", "toolbox"], name="ai_hub_unique_agent_toolbox"),
        ]

    def __str__(self):
        status = "enabled" if self.is_enabled else "disabled"
        return f"{self.agent.name} -> {self.toolbox} ({status})"


class AgentToolGrant(models.Model):
    class PermissionLevel(models.TextChoices):
        USE = "use", "Use"
        READ_ONLY = "read_only", "Read only"
        DRAFT_WRITE = "draft_write", "Draft write"
        STATE_WRITE = "state_write", "State write"
        EXTERNAL_WRITE = "external_write", "External write"
        EXECUTE = "execute", "Execute"

    agent = models.ForeignKey(AgentProfile, on_delete=models.CASCADE, related_name="tool_grants")
    tool = models.ForeignKey(ToolDefinition, on_delete=models.CASCADE, related_name="agent_grants")
    is_enabled = models.BooleanField(default=True)
    permission_level = models.CharField(max_length=30, choices=PermissionLevel.choices, default=PermissionLevel.USE)
    requires_approval_override = models.BooleanField(null=True, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["agent__name", "tool__name"]
        verbose_name = "1.11 Agent tool grant"
        verbose_name_plural = "1.11 Agent tool grants - individual overrides"
        constraints = [
            models.UniqueConstraint(fields=["agent", "tool"], name="ai_hub_unique_agent_tool_grant"),
        ]

    def __str__(self):
        status = "allowed" if self.is_enabled else "denied"
        return f"{self.agent.name} -> {self.tool.name} ({status})"


class PipelineDefinition(models.Model):
    name = models.CharField(max_length=140, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=False)
    entry_agent = models.ForeignKey(
        AgentProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="entry_for_pipelines",
    )
    global_input_contract = models.JSONField(default=dict, blank=True)
    global_output_contract = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "2.1 Pipeline definition"
        verbose_name_plural = "2.1 Pipeline definitions - ordered agent flows"

    def __str__(self):
        return self.name

    def clean(self):
        if self.is_active:
            if not self.pk:
                raise ValidationError("Save the pipeline and add steps before activating it.")
            steps = list(self.steps.order_by("order"))
            if not steps:
                raise ValidationError("Cannot activate pipeline without steps.")
            # An executable pipeline must belong to ONE application scope, or it
            # becomes a bridge between them: step 1 retrieves Scope A Knowledge
            # and the output mapping hands it to a Scope B agent. Configuration
            # time only - `execution_runner` re-checks before running, and that
            # runtime check is the authoritative one.
            from ai_hub.services.knowledge_authorization import (
                PipelineScopeError,
                require_coherent_pipeline_scope,
            )

            try:
                require_coherent_pipeline_scope(self)
            except PipelineScopeError as exc:
                raise ValidationError(str(exc)) from exc
            expected = list(range(1, len(steps) + 1))
            got = [s.order for s in steps]
            if got != expected:
                raise ValidationError("Pipeline step order must be continuous starting at 1.")
            for step in steps:
                agents = [("Agent", step.agent)]
                if (
                    step.on_error == step.OnError.FALLBACK_AGENT
                    and step.fallback_agent_id
                ):
                    agents.append(("Fallback agent", step.fallback_agent))
                for label, agent in agents:
                    if not agent.input_contract or not agent.output_contract:
                        raise ValidationError(
                            f"{label} '{agent.name}' must define input/output contracts "
                            "before activating pipeline."
                        )
                    if not step.input_mapping:
                        continue
                    available_inputs = set(step.input_mapping)
                    available_inputs.add("knowledge_context")
                    required_inputs = (
                        agent.input_contract.get("required", [])
                        if isinstance(agent.input_contract, dict)
                        else []
                    )
                    missing_inputs = [
                        key for key in required_inputs if key not in available_inputs
                    ]
                    if missing_inputs:
                        raise ValidationError(
                            f"{label} '{agent.name}' cannot receive required input keys: "
                            f"{', '.join(missing_inputs)}."
                        )


class PipelineStep(models.Model):
    class OnError(models.TextChoices):
        STOP = "stop", "Stop"
        CONTINUE = "continue", "Continue"
        FALLBACK_AGENT = "fallback_agent", "Fallback Agent"

    pipeline = models.ForeignKey(PipelineDefinition, on_delete=models.CASCADE, related_name="steps")
    agent = models.ForeignKey(AgentProfile, on_delete=models.PROTECT, related_name="pipeline_steps")
    order = models.PositiveIntegerField()
    input_mapping = models.JSONField(default=dict, blank=True)
    output_mapping = models.JSONField(default=dict, blank=True)
    on_error = models.CharField(max_length=20, choices=OnError.choices, default=OnError.STOP)
    fallback_agent = models.ForeignKey(
        AgentProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="fallback_steps",
    )

    class Meta:
        unique_together = [("pipeline", "order")]
        ordering = ["pipeline__name", "order"]
        verbose_name = "2.2 Pipeline step"
        verbose_name_plural = "2.2 Pipeline steps - agent connections"

    def __str__(self):
        return f"{self.pipeline.name} - step {self.order}"

    def clean(self):
        if self.agent_id and not self.agent.is_active:
            raise ValidationError("Pipeline step agent must be active.")
        if self.on_error == self.OnError.FALLBACK_AGENT and not self.fallback_agent:
            raise ValidationError("Fallback agent is required when on_error is fallback_agent.")
        if self.fallback_agent_id and not self.fallback_agent.is_active:
            raise ValidationError("Fallback agent must be active.")
        # Fallback agents are part of the security boundary: an error path that
        # switches to an agent in another application is still a bridge between
        # applications.
        if self.agent_id:
            scope_ids = {self.agent.application_scope_id}
            if self.fallback_agent_id:
                scope_ids.add(self.fallback_agent.application_scope_id)
            if self.pipeline_id and self.pipeline.entry_agent_id:
                scope_ids.add(self.pipeline.entry_agent.application_scope_id)
            if len(scope_ids) > 1:
                raise ValidationError(
                    "All agents in a pipeline must belong to the same "
                    "application scope."
                )


class GameWorkspace(models.Model):
    # Root-owned. A workspace belongs to a scope; it is NOT the security root
    # itself. It is a GAME execution environment, Orchestrator sessions have no
    # workspace at all, and its agent allow-list fails open when empty - none of
    # which is acceptable in a root boundary.
    application_scope = models.ForeignKey(
        ApplicationScope,
        on_delete=models.PROTECT,
        related_name="game_workspaces",
    )
    # Globally unique in S-14 by deliberate compatibility choice; see the note
    # on KnowledgeCollection.name.
    name = models.CharField(max_length=160, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    default_policy = models.JSONField(default=dict, blank=True)
    default_runtime_config = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "3.1 GAME workspace"
        verbose_name_plural = "3.1 GAME workspaces - environments"

    def __str__(self):
        return self.name

    def clean(self):
        if not isinstance(self.default_runtime_config, dict):
            raise ValidationError("GAME workspace default_runtime_config must be a JSON object.")
        from ai_hub.services.game_policy import validate_workspace_policy

        validate_workspace_policy(self.default_policy)


class GameGoal(models.Model):
    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        WAITING_INFO = "waiting_info", "Waiting for information"
        WAITING_APPROVAL = "waiting_approval", "Waiting for approval"
        BLOCKED = "blocked", "Blocked"
        COMPLETED = "completed", "Completed"
        PARTIAL = "partial", "Partial"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    workspace = models.ForeignKey(GameWorkspace, on_delete=models.CASCADE, related_name="goals")
    title = models.CharField(max_length=255)
    description = models.TextField()
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.QUEUED)
    base_priority = models.PositiveIntegerField(default=50, validators=[MaxValueValidator(999900)])
    calculated_priority = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0), MaxValueValidator(999999.99)],
    )
    due_at = models.DateTimeField(null=True, blank=True)
    queued_at = models.DateTimeField(default=timezone.now)
    success_criteria = models.JSONField(default=dict, blank=True)
    context = models.JSONField(default=dict, blank=True)
    result = models.JSONField(default=dict, blank=True)
    transition_metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["workspace__name", "-calculated_priority", "created_at"]
        verbose_name = "3.2 GAME goal"
        verbose_name_plural = "3.2 GAME goals - work items"
        indexes = [
            models.Index(fields=["workspace", "status"], name="aihub_goal_ws_status_idx"),
            models.Index(fields=["workspace", "calculated_priority"], name="aihub_goal_ws_priority_idx"),
            models.Index(fields=["due_at"], name="ai_hub_goal_due_idx"),
        ]

    def __str__(self):
        return f"{self.title} ({self.workspace.name})"


class GameGoalDependency(models.Model):
    goal = models.ForeignKey(GameGoal, on_delete=models.CASCADE, related_name="dependencies")
    depends_on = models.ForeignKey(GameGoal, on_delete=models.CASCADE, related_name="required_by")
    is_required = models.BooleanField(default=True)
    note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["goal_id", "depends_on_id"]
        verbose_name = "3.3 GAME goal dependency"
        verbose_name_plural = "3.3 GAME goal dependencies"
        constraints = [
            models.UniqueConstraint(fields=["goal", "depends_on"], name="ai_hub_unique_goal_dependency"),
            models.CheckConstraint(
                condition=~models.Q(goal=models.F("depends_on")),
                name="ai_hub_goal_dependency_not_self",
            ),
        ]

    def __str__(self):
        return f"{self.goal.title} depends on {self.depends_on.title}"

    def clean(self):
        if not self.goal_id or not self.depends_on_id:
            return
        if self.goal_id == self.depends_on_id:
            raise ValidationError("A goal cannot depend on itself.")
        if self.goal.workspace_id != self.depends_on.workspace_id:
            raise ValidationError("Goal dependencies must belong to the same workspace.")

        pending = [self.depends_on_id]
        visited = set()
        while pending:
            current_id = pending.pop()
            if current_id == self.goal_id:
                raise ValidationError("Circular goal dependencies are not allowed.")
            if current_id in visited:
                continue
            visited.add(current_id)
            next_ids = (
                GameGoalDependency.objects.filter(goal_id=current_id)
                .exclude(pk=self.pk)
                .values_list("depends_on_id", flat=True)
            )
            pending.extend(next_ids)


class GameActionDefinition(models.Model):
    class ActionType(models.TextChoices):
        INTERNAL = "internal", "Internal"
        CONTEXT_TOOL = "context_tool", "Context tool"
        TOOL = "tool", "Tool"
        PYTHON_CALLABLE = "python_callable", "Python callable"
        HTTP = "http", "HTTP"
        SUB_AGENT = "sub_agent", "Sub-agent"
        HUMAN_APPROVAL = "human_approval", "Human approval"

    name = models.SlugField(max_length=120, unique=True)
    label = models.CharField(max_length=160)
    description = models.TextField(blank=True)
    action_type = models.CharField(max_length=30, choices=ActionType.choices)
    tool = models.ForeignKey(
        ToolDefinition,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name="game_action_definitions",
    )
    input_contract = models.JSONField(default=dict, blank=True)
    output_contract = models.JSONField(default=dict, blank=True)
    config = models.JSONField(default=dict, blank=True)
    risk_level = models.CharField(max_length=20, default="low")
    requires_approval = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "3.6 GAME action definition"
        verbose_name_plural = "3.6 GAME action definitions - dispatcher registry"

    def __str__(self):
        return f"{self.name} ({self.get_action_type_display()})"

    def clean(self):
        for field_name in ("input_contract", "output_contract", "config"):
            if not isinstance(getattr(self, field_name), dict):
                raise ValidationError({field_name: "Must be a JSON object."})
        if self.risk_level not in {"low", "medium", "high"}:
            raise ValidationError({"risk_level": "Risk level must be low, medium, or high."})


class ExecutionSession(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        WAITING_ASYNC = "waiting_async", "Waiting Async Continuation"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    class RuntimeKind(models.TextChoices):
        ORCHESTRATOR = "orchestrator", "Orchestrator"
        GAME = "game", "GAME"

    class RuntimeMode(models.TextChoices):
        SYNC = "sync", "Sync"
        ASYNC = "async", "Async"
        HYBRID = "hybrid", "Hybrid"

    source_content_type = models.ForeignKey(
        ContentType,
        on_delete=models.CASCADE,
        related_name="ai_hub_execution_sessions",
        null=True,
        blank=True,
    )
    source_object_id = models.PositiveBigIntegerField(null=True, blank=True)
    source_object = GenericForeignKey("source_content_type", "source_object_id")
    source_label = models.CharField(max_length=255, blank=True)
    goal = models.ForeignKey(
        GameGoal,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name="execution_sessions",
    )
    pipeline = models.ForeignKey(
        PipelineDefinition,
        on_delete=models.PROTECT,
        related_name="execution_sessions",
        null=True,
        blank=True,
    )
    entry_agent = models.ForeignKey(
        AgentProfile,
        on_delete=models.PROTECT,
        related_name="entry_execution_sessions",
        null=True,
        blank=True,
    )
    triggered_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="triggered_ai_hub_sessions",
    )
    runtime_kind = models.CharField(max_length=20, choices=RuntimeKind.choices, default=RuntimeKind.ORCHESTRATOR)
    runtime_mode = models.CharField(max_length=20, choices=RuntimeMode.choices, default=RuntimeMode.ASYNC)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    goal_text = models.TextField(blank=True)
    runtime_config = models.JSONField(default=dict, blank=True)
    initial_context = models.JSONField(default=dict, blank=True)
    final_context = models.JSONField(default=dict, blank=True)
    goal_outcome_fingerprint = models.CharField(max_length=64, blank=True)
    error_detail = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "4.1 Execution session"
        verbose_name_plural = "4.1 Execution sessions - generic runs"
        indexes = [
            models.Index(fields=["runtime_kind", "status"], name="ai_hub_session_kind_status_idx"),
            models.Index(fields=["source_content_type", "source_object_id"], name="ai_hub_session_source_idx"),
            models.Index(fields=["created_at"], name="ai_hub_session_created_idx"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["goal"],
                condition=models.Q(
                    goal__isnull=False,
                    status__in=["pending", "running", "waiting_async"],
                ),
                name="aihub_unique_active_goal",
            ),
        ]

    def __str__(self):
        label = self.source_label or self.source_object or "no source"
        return f"Session #{self.pk} - {label}"

    def clean(self):
        if self.runtime_kind == self.RuntimeKind.ORCHESTRATOR and not self.pipeline_id:
            raise ValidationError("Orchestrator sessions require a pipeline.")
        if self.runtime_kind == self.RuntimeKind.GAME and not (self.entry_agent_id or self.pipeline_id):
            raise ValidationError("GAME sessions require an entry agent or pipeline.")
        if self.runtime_kind == self.RuntimeKind.GAME and self.runtime_mode == self.RuntimeMode.HYBRID:
            raise ValidationError("GAME Hybrid continuation is not enabled yet. Use sync or async mode.")
        if self.goal_id and self.runtime_kind != self.RuntimeKind.GAME:
            raise ValidationError("Only GAME execution sessions can be linked to a GAME goal.")


class ExecutionStepRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"

    session = models.ForeignKey(ExecutionSession, on_delete=models.CASCADE, related_name="step_runs")
    order = models.PositiveIntegerField()
    pipeline_step = models.ForeignKey(
        PipelineStep,
        on_delete=models.PROTECT,
        related_name="execution_step_runs",
        null=True,
        blank=True,
    )
    agent = models.ForeignKey(
        AgentProfile,
        on_delete=models.PROTECT,
        related_name="execution_step_runs",
        null=True,
        blank=True,
    )
    action_name = models.CharField(max_length=140, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    request_payload = models.JSONField(default=dict, blank=True)
    response_payload = models.JSONField(default=dict, blank=True)
    observation_payload = models.JSONField(default=dict, blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    error_detail = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("session", "order")]
        ordering = ["session_id", "order"]
        verbose_name = "4.2 Execution step run"
        verbose_name_plural = "4.2 Execution step runs - generic step logs"
        indexes = [
            models.Index(fields=["session", "status"], name="ai_hub_step_session_status_idx"),
            models.Index(fields=["agent", "status"], name="ai_hub_step_agent_status_idx"),
        ]

    def __str__(self):
        return f"Session #{self.session_id} - step {self.order}"


class ToolExecutionRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        WAITING_APPROVAL = "waiting_approval", "Waiting approval"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"

    class ApprovalState(models.TextChoices):
        NOT_REQUIRED = "not_required", "Not required"
        REQUIRED = "required", "Required"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"

    session = models.ForeignKey(
        ExecutionSession,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="tool_execution_runs",
    )
    step_run = models.ForeignKey(
        ExecutionStepRun,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="tool_execution_runs",
    )
    agent = models.ForeignKey(
        AgentProfile,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name="tool_execution_runs",
    )
    tool = models.ForeignKey(ToolDefinition, on_delete=models.PROTECT, related_name="execution_runs")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    input_payload = models.JSONField(default=dict, blank=True)
    output_payload = models.JSONField(default=dict, blank=True)
    error_detail = models.TextField(blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    risk_level = models.CharField(max_length=20, choices=ToolDefinition.RiskLevel.choices, default=ToolDefinition.RiskLevel.LOW)
    approval_state = models.CharField(
        max_length=30,
        choices=ApprovalState.choices,
        default=ApprovalState.NOT_REQUIRED,
    )
    idempotency_key = models.CharField(max_length=255, blank=True, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "4.3 Tool execution run"
        verbose_name_plural = "4.3 Tool execution runs - generic tool audit"
        indexes = [
            models.Index(fields=["session", "status"], name="aihub_tool_run_sess_status_idx"),
            models.Index(fields=["agent", "status"], name="aihub_tool_run_agent_stat_idx"),
            models.Index(fields=["tool", "status"], name="aihub_tool_run_tool_stat_idx"),
        ]

    def __str__(self):
        return f"Tool '{self.tool.name}' - {self.status}"

    def clean(self):
        for field_name in ("input_payload", "output_payload"):
            if not isinstance(getattr(self, field_name), dict):
                raise ValidationError({field_name: "Must be a JSON object."})


class GameActionRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        WAITING_APPROVAL = "waiting_approval", "Waiting approval"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"
        REJECTED = "rejected", "Rejected"

    session = models.ForeignKey(
        ExecutionSession,
        on_delete=models.CASCADE,
        related_name="game_action_runs",
    )
    step_run = models.ForeignKey(
        ExecutionStepRun,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="game_action_runs",
    )
    action = models.ForeignKey(
        GameActionDefinition,
        on_delete=models.PROTECT,
        related_name="game_action_runs",
    )
    idempotency_key = models.CharField(max_length=255, unique=True)
    action_name = models.CharField(max_length=120)
    iteration = models.PositiveIntegerField()
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    input_payload = models.JSONField(default=dict, blank=True)
    output_payload = models.JSONField(default=dict, blank=True)
    observation_payload = models.JSONField(default=dict, blank=True)
    error_detail = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        ordering = ["session_id", "iteration"]
        verbose_name = "4.4 GAME action run"
        verbose_name_plural = "4.4 GAME action runs - dispatcher history"
        indexes = [
            models.Index(fields=["session", "iteration"], name="aihub_action_run_sess_iter_idx"),
            models.Index(fields=["action", "status"], name="aihub_game_run_action_stat_idx"),
        ]

    def __str__(self):
        return f"Action '{self.action_name}' - session #{self.session_id} iter {self.iteration}"


class GameMemoryEntry(models.Model):
    from decimal import Decimal as _Dec

    class ScopeType(models.TextChoices):
        WORKSPACE = "workspace", "Workspace"
        GOAL = "goal", "Goal"
        SESSION = "session", "Session"
        ACTION_RESULT = "action_result", "Action result"

    workspace = models.ForeignKey(GameWorkspace, on_delete=models.CASCADE, related_name="memory_entries")
    goal = models.ForeignKey(
        GameGoal, on_delete=models.CASCADE, null=True, blank=True, related_name="memory_entries"
    )
    session = models.ForeignKey(
        ExecutionSession, on_delete=models.CASCADE, null=True, blank=True, related_name="memory_entries"
    )
    scope_type = models.CharField(max_length=20, choices=ScopeType.choices)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    importance_score = models.DecimalField(
        max_digits=4,
        decimal_places=2,
        default=_Dec("0.50"),
        validators=[MinValueValidator(_Dec("0.00")), MaxValueValidator(_Dec("1.00"))],
    )
    expires_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["workspace_id", "-importance_score", "created_at"]
        verbose_name = "3.9 GAME memory entry"
        verbose_name_plural = "3.9 GAME memory entries - scoped knowledge store"
        indexes = [
            models.Index(fields=["workspace", "scope_type"], name="aihub_mem_ws_scope_idx"),
            models.Index(fields=["goal", "scope_type"], name="aihub_mem_goal_scope_idx"),
            models.Index(fields=["workspace", "importance_score"], name="aihub_mem_ws_import_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(scope_type="workspace", goal__isnull=True, session__isnull=True)
                    | models.Q(scope_type__in=["goal", "action_result"], goal__isnull=False)
                    | models.Q(scope_type="session", session__isnull=False)
                ),
                name="aihub_memory_scope_shape",
            ),
        ]

    def __str__(self):
        return f"{self.scope_type} memory - workspace '{self.workspace.name}'"

    def clean(self):
        if self.scope_type == self.ScopeType.WORKSPACE:
            if self.goal_id is not None or self.session_id is not None:
                raise ValidationError("Workspace-scoped memory must not have a goal or session.")
        elif self.scope_type in {self.ScopeType.GOAL, self.ScopeType.ACTION_RESULT}:
            if not self.goal_id:
                raise ValidationError(
                    f"{self.get_scope_type_display()}-scoped memory requires a goal."
                )
            if self.goal_id and self.goal.workspace_id != self.workspace_id:
                raise ValidationError("Memory entry goal must belong to the same workspace.")
            if self.session_id and self.session.goal_id != self.goal_id:
                raise ValidationError("Goal/action-result memory session must belong to the same goal.")
        elif self.scope_type == self.ScopeType.SESSION:
            if not self.session_id:
                raise ValidationError("Session-scoped memory requires a session.")
            if self.session_id and self.session.goal_id:
                if self.session.goal.workspace_id != self.workspace_id:
                    raise ValidationError("Session memory must belong to the session goal's workspace.")
                if self.goal_id and self.session.goal_id != self.goal_id:
                    raise ValidationError("Memory entry goal must match session.goal when both are set.")
            elif self.session_id and self.goal_id:
                raise ValidationError(
                    "Memory for a legacy session without a goal cannot reference a GAME goal."
                )
        else:
            raise ValidationError({"scope_type": "Unknown GAME memory scope."})


class GameContinuationRequest(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RESOLVED = "resolved", "Resolved"
        EXPIRED = "expired", "Expired"
        CANCELLED = "cancelled", "Cancelled"

    class ReasonCode(models.TextChoices):
        NEEDS_INFORMATION = "needs_information", "Needs information"
        NEEDS_APPROVAL = "needs_approval", "Needs approval"
        EXTERNAL_RESULT_PENDING = "external_result_pending", "External result pending"
        RATE_LIMITED = "rate_limited", "Rate limited"
        MANUAL_REVIEW_REQUIRED = "manual_review_required", "Manual review required"

    session = models.ForeignKey(ExecutionSession, on_delete=models.CASCADE, related_name="continuation_requests")
    goal = models.ForeignKey(GameGoal, on_delete=models.CASCADE, related_name="continuation_requests")
    reason_code = models.CharField(max_length=80, choices=ReasonCode.choices)
    detail = models.TextField(blank=True)
    payload = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.PENDING)
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="resolved_continuation_requests",
    )

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "4.6 GAME continuation request"
        verbose_name_plural = "4.6 GAME continuation requests - pause records"
        indexes = [
            models.Index(fields=["session", "status"], name="aihub_cont_req_sess_stat_idx"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["session"],
                condition=models.Q(status="pending"),
                name="aihub_one_pending_continuation",
            ),
        ]

    def __str__(self):
        return f"Continuation #{self.pk} — {self.reason_code} ({self.status})"

    def clean(self):
        if self.session_id and self.goal_id and self.session.goal_id != self.goal_id:
            raise ValidationError("Continuation goal must match the session goal.")


class GameActionApprovalRequest(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"
        EXPIRED = "expired", "Expired"

    action_run = models.OneToOneField(GameActionRun, on_delete=models.CASCADE, related_name="approval_request")
    goal = models.ForeignKey(GameGoal, on_delete=models.CASCADE, related_name="approval_requests")
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.PENDING)
    requested_payload = models.JSONField(default=dict, blank=True)
    execution_intent_snapshot = models.JSONField(default=dict, blank=True)
    execution_intent_fingerprint = models.CharField(max_length=64, blank=True)
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="reviewed_approval_requests",
    )
    review_note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "4.7 GAME action approval request"
        verbose_name_plural = "4.7 GAME action approval requests - gated actions"
        permissions = [("approve_game_action", "Can approve GAME action requests")]
        indexes = [
            models.Index(fields=["goal", "status"], name="aihub_approval_goal_stat_idx"),
        ]

    def __str__(self):
        return f"Approval #{self.pk} — '{self.action_run.action_name}' ({self.status})"

    def clean(self):
        if (
            self.action_run_id
            and self.goal_id
            and self.action_run.session.goal_id != self.goal_id
        ):
            raise ValidationError("Approval goal must match the action run session goal.")


class GameWorkspaceAction(models.Model):
    workspace = models.ForeignKey(
        GameWorkspace, on_delete=models.CASCADE, related_name="workspace_actions"
    )
    action = models.ForeignKey(
        GameActionDefinition, on_delete=models.CASCADE, related_name="workspace_entries"
    )
    is_enabled = models.BooleanField(default=True)
    requires_approval_override = models.BooleanField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["workspace__name", "action__name"]
        verbose_name = "3.7 GAME workspace action"
        verbose_name_plural = "3.7 GAME workspace actions - allow-list"
        constraints = [
            models.UniqueConstraint(fields=["workspace", "action"], name="ai_hub_unique_ws_action"),
        ]

    def __str__(self):
        status = "enabled" if self.is_enabled else "disabled"
        return f"{self.workspace.name} → {self.action.name} ({status})"


class GameWorkspaceAgent(models.Model):
    workspace = models.ForeignKey(
        GameWorkspace, on_delete=models.CASCADE, related_name="workspace_agents"
    )
    agent = models.ForeignKey(
        AgentProfile, on_delete=models.CASCADE, related_name="workspace_entries"
    )
    is_enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["workspace__name", "agent__name"]
        verbose_name = "3.8 GAME workspace agent"
        verbose_name_plural = "3.8 GAME workspace agents - allow-list"
        constraints = [
            models.UniqueConstraint(fields=["workspace", "agent"], name="ai_hub_unique_ws_agent"),
        ]

    def __str__(self):
        status = "enabled" if self.is_enabled else "disabled"
        return f"{self.workspace.name} → {self.agent.name} ({status})"


class GameGoalPlan(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        COMPLETED = "completed", "Completed"
        ABANDONED = "abandoned", "Abandoned"

    goal = models.OneToOneField(GameGoal, on_delete=models.CASCADE, related_name="plan")
    version = models.PositiveIntegerField(default=1)
    summary = models.TextField(blank=True)
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.ACTIVE)
    revision_history = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "3.4 GAME goal plan"
        verbose_name_plural = "3.4 GAME goal plans - structured execution aids"
        constraints = [
            models.CheckConstraint(
                condition=models.Q(version__gte=1),
                name="aihub_plan_version_gte_1",
            ),
        ]

    def __str__(self):
        return f"Plan #{self.pk} — {self.goal.title} ({self.status})"

    def clean(self):
        if not isinstance(self.revision_history, list):
            raise ValidationError({"revision_history": "Plan revision history must be a JSON list."})


class GameGoalPlanStep(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        IN_PROGRESS = "in_progress", "In progress"
        COMPLETED = "completed", "Completed"
        SKIPPED = "skipped", "Skipped"
        BLOCKED = "blocked", "Blocked"

    plan = models.ForeignKey(GameGoalPlan, on_delete=models.CASCADE, related_name="steps")
    order = models.PositiveIntegerField()
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.PENDING)
    depends_on_step = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="required_by_steps",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["plan_id", "order"]
        verbose_name = "3.5 GAME goal plan step"
        verbose_name_plural = "3.5 GAME goal plan steps"
        constraints = [
            models.UniqueConstraint(
                fields=["plan", "order"], name="ai_hub_unique_plan_step_order"
            ),
            models.CheckConstraint(
                condition=models.Q(order__gte=1),
                name="aihub_plan_step_order_gte_1",
            ),
            models.CheckConstraint(
                condition=~models.Q(pk=models.F("depends_on_step")),
                name="aihub_plan_step_not_self",
            ),
        ]

    def __str__(self):
        return f"Step {self.order}: {self.title}"

    def clean(self):
        if self.depends_on_step_id:
            if self.pk and self.depends_on_step_id == self.pk:
                raise ValidationError("A step cannot depend on itself.")
            if self.depends_on_step.plan_id != self.plan_id:
                raise ValidationError("Step dependency must belong to the same plan.")
            if self.depends_on_step.order >= self.order:
                raise ValidationError("A plan step may depend only on an earlier step.")

            current = self.depends_on_step
            visited = set()
            while current is not None:
                if current.pk in visited or (self.pk and current.pk == self.pk):
                    raise ValidationError("Circular plan-step dependencies are not allowed.")
                visited.add(current.pk)
                current = current.depends_on_step


class GameDelegationRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"

    parent_action_run = models.OneToOneField(
        GameActionRun, on_delete=models.CASCADE, related_name="delegation_run"
    )
    parent_goal = models.ForeignKey(
        GameGoal, on_delete=models.CASCADE, related_name="delegation_runs"
    )
    delegated_session = models.OneToOneField(
        ExecutionSession,
        on_delete=models.PROTECT,
        related_name="delegation_run",
        null=True,
        blank=True,
    )
    target_agent = models.ForeignKey(
        AgentProfile, on_delete=models.PROTECT, related_name="delegation_runs"
    )
    status = models.CharField(max_length=30, choices=Status.choices, default=Status.PENDING)
    task = models.TextField()
    expected_result = models.TextField(blank=True)
    result_summary = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "4.5 GAME delegation run"
        verbose_name_plural = "4.5 GAME delegation runs - sub-agent history"
        indexes = [
            models.Index(
                fields=["parent_goal", "status"], name="aihub_delegation_goal_stat_idx"
            ),
        ]
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(
                        status__in=["pending", "running"],
                        finished_at__isnull=True,
                    )
                    | models.Q(
                        status__in=["success", "failed"],
                        finished_at__isnull=False,
                    )
                ),
                name="aihub_delegation_terminal_time",
            ),
        ]

    def __str__(self):
        return f"Delegation #{self.pk} → {self.target_agent.name} ({self.status})"

    def clean(self):
        if self.parent_action_run_id and self.parent_goal_id:
            parent_session = self.parent_action_run.session
            if parent_session.goal_id != self.parent_goal_id:
                raise ValidationError("Delegation parent goal must match the parent action session goal.")
            if self.parent_action_run.action.action_type != GameActionDefinition.ActionType.SUB_AGENT:
                raise ValidationError("Delegation parent action must be a sub-agent action.")
        if self.delegated_session_id:
            if self.delegated_session.goal_id is not None:
                raise ValidationError("Delegated sessions must not be linked directly to a GAME goal.")
            if self.delegated_session.entry_agent_id != self.target_agent_id:
                raise ValidationError("Delegated session entry agent must match the target agent.")
        if self.status in {self.Status.SUCCESS, self.Status.FAILED} and not self.finished_at:
            raise ValidationError("Terminal delegation runs require finished_at.")


class KnowledgeLifecycleEvent(models.Model):
    """Durable, append-only record of a COMMITTED Knowledge lifecycle change.

    WHAT THIS IS
    ------------
    Corpus history, not execution history. One row means: a governed Knowledge
    mutation **committed**. It is written inside the same transaction as the
    mutation it describes, so the pair is atomic in both directions - there is
    no committed governed mutation without its event, and no event claiming a
    mutation that rolled back.

    It is deliberately NOT an execution-attempt model. Rejected validation,
    stale-review conflicts and exceptions produce no row: nothing changed, so
    there is no state change to record. Durable auditing of *attempts* is a
    separate concern and is not built here.

    WHY NOT AN EXISTING AUDIT MODEL
    -------------------------------
    `ToolExecutionRun`, `GameActionRun`, `ExecutionSession` and
    `ExecutionStepRun` are all execution-scoped: each requires a session, step,
    tool or action to exist. A corpus mutation may have none of those - an
    operator adjudicating a legacy document acts outside any AI execution
    entirely - so reusing them would force a fabricated execution context and
    couple corpus history to AI runtime lifetimes.

    Numbered `1.12` rather than `4.x` on purpose: it belongs with the Knowledge
    corpus models (1.3-1.5), not with the execution audit family.

    REFERENCE-FIRST
    ---------------
    This model records facts and references. It never stores `curated_text`,
    chunk content, `source_file` bytes, retrieval snippets, or any other
    Knowledge body - not directly and not inside a generic payload. Every field
    is a bounded scalar, which is enforced by test rather than left to habit.

    WHAT IT DOES *NOT* AUDIT
    ------------------------
    This schema describes lifecycle facts: authority mode, status, generation
    inputs, chunk set, generator identity/version and chunk count. It does NOT
    carry before/after values for `collection`, so it cannot explain a
    collection move - which is an authorization-boundary change and a separate,
    security-sensitive future operation. The mutation foundation therefore
    REFUSES a mutation that changes the collection, rather than committing a
    change it cannot describe. Extend this schema first if that operation is
    ever built.

    Title and `curated_text` are likewise not copied here, but a change to
    either moves the observed `i1` fingerprint, so the event still states
    truthfully that the generation inputs moved.

    APPEND-ONLY
    -----------
    Conceptually immutable once written. Core exposes no update or delete API
    for these rows and does not register them in the Admin. Raw ORM can of
    course still reach them; the contract is about supported Core behavior, not
    about pretending the ORM can be locked out.
    """

    class PrincipalKind(models.TextChoices):
        """WHO initiated the mutation.

        Deliberately tiny and closed. This is an operator/system axis, not an
        Agent identity axis: an `AgentProfile` is never a lifecycle principal,
        and a model must never be able to declare one. Unlike `operation`, this
        vocabulary is not expected to grow, so `choices` costs no migration
        churn and buys validation.
        """

        HUMAN = "human", "Human operator"
        SYSTEM = "system", "System or service"

    # SET_NULL, never CASCADE: deleting a document must not erase the history of
    # what was done to it. `document_id_snapshot` keeps the row intelligible
    # after the FK is nulled.
    document = models.ForeignKey(
        KnowledgeDocument,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="lifecycle_events",
    )
    # PositiveBigIntegerField, not PositiveIntegerField: DEFAULT_AUTO_FIELD is
    # BigAutoField, so these ids are 64-bit. A durable snapshot must never have
    # a narrower numeric domain than the identifier it copies - the whole point
    # of the snapshot is that it stays valid after the FK is gone, and a 32-bit
    # column would start failing on a long-lived corpus exactly when the history
    # matters most.
    document_id_snapshot = models.PositiveBigIntegerField()
    # Bounded collection context: which collection owned the document when the
    # mutation committed. An id only - names and collection bodies are not
    # duplicated here. Collection MOVES are an authorization concern, are NOT a
    # governed operation in this slice, and are refused by the mutation
    # foundation precisely because this schema cannot describe one truthfully.
    collection_id_snapshot = models.PositiveBigIntegerField()

    # No `choices` on purpose. The operation vocabulary is known to grow as
    # later slices add adjudication, explicit authoring, derived generation,
    # regeneration and repair, and Django emits an AlterField migration for
    # every `choices` change - schema churn for a column whose storage never
    # changes. The service layer validates the slug shape instead, which keeps
    # the column queryable without inventing speculative operations now.
    operation = models.CharField(max_length=64)
    # Optional bounded machine code explaining WHY. Never free-form prose, and
    # never a place to smuggle content.
    reason_code = models.CharField(max_length=64, blank=True)

    # Principal snapshot, not a foreign key: the record must stay intelligible
    # after a host deletes the user, and Core must not depend on a host's user
    # model. `principal_identifier` is a stable opaque handle (a primary key or
    # username). Do not put an email address or any other personal contact
    # detail here.
    #
    # There is deliberately NO display-label column. A human-readable name is
    # convenience data, and audit rows are kept indefinitely, so a label column
    # is a standing invitation to persist personal data forever for no
    # correctness benefit. A host that wants to show a name can resolve it from
    # the identifier at display time. Data minimization wins absent a proven
    # requirement, and no requirement was found.
    principal_kind = models.CharField(max_length=20, choices=PrincipalKind.choices)
    principal_identifier = models.CharField(max_length=150)

    # --- Lifecycle facts BEFORE and AFTER, mirroring KnowledgeDocument -------
    previous_authority_mode = models.CharField(max_length=20)
    new_authority_mode = models.CharField(max_length=20)
    # Status is recorded because it gates retrievability (D-L3). Without it a
    # governed ACTIVE/ARCHIVED transition would commit unexplained.
    previous_status = models.CharField(max_length=20)
    new_status = models.CharField(max_length=20)
    previous_generation_input_fingerprint = models.CharField(max_length=80, blank=True)
    new_generation_input_fingerprint = models.CharField(max_length=80, blank=True)
    previous_generation_chunk_set_fingerprint = models.CharField(max_length=80, blank=True)
    new_generation_chunk_set_fingerprint = models.CharField(max_length=80, blank=True)
    previous_generator_identity = models.CharField(max_length=64, blank=True)
    new_generator_identity = models.CharField(max_length=64, blank=True)
    previous_generator_version = models.PositiveIntegerField(null=True, blank=True)
    new_generator_version = models.PositiveIntegerField(null=True, blank=True)
    previous_chunk_count = models.PositiveIntegerField()
    new_chunk_count = models.PositiveIntegerField()
    # Observed fingerprints, which are NOT the same as the recorded generation
    # fingerprints above: these are what the rows actually hashed to inside the
    # mutation transaction. Together the two pairs are the tamper evidence a
    # later reader needs - a recorded/observed divergence is exactly what
    # Preflight V2 reports as DERIVED_CHUNKS_MODIFIED.
    #
    # The observed INPUT pair is also what makes a title or curated_text edit
    # explicable from the audit alone: neither field is copied here, but a change
    # to either moves `i1`, so the event still says truthfully that the
    # generation inputs moved.
    previous_observed_input_fingerprint = models.CharField(max_length=80, blank=True)
    new_observed_input_fingerprint = models.CharField(max_length=80, blank=True)
    previous_observed_chunk_set_fingerprint = models.CharField(max_length=80, blank=True)
    new_observed_chunk_set_fingerprint = models.CharField(max_length=80, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        verbose_name = "1.12 Knowledge lifecycle event"
        verbose_name_plural = "1.12 Knowledge lifecycle events - corpus mutation history"
        # Two indexes, deliberately not a speculative set.
        #
        # `document_id_snapshot` first, because it - not the FK - is the DURABLE
        # history key: once the document is deleted the FK is NULL and the
        # snapshot is the only way to ask "what happened to document 42?".
        # Leaving that unindexed would make the one query the SET_NULL design
        # exists to support a full-table scan. Django already indexes the FK
        # itself, so no separate index for it is added.
        #
        # `operation` is indexed because the column is explicitly intended to be
        # queryable; both are paired with `created_at` since lifecycle history is
        # always read in time order.
        indexes = [
            models.Index(
                fields=["document_id_snapshot", "created_at"],
                name="aihub_klc_evt_docsnap_idx",
            ),
            models.Index(
                fields=["operation", "created_at"], name="aihub_klc_evt_op_time_idx"
            ),
        ]

    def __str__(self):
        return (
            f"{self.operation} on document #{self.document_id_snapshot}: "
            f"{self.previous_authority_mode} -> {self.new_authority_mode}"
        )


# === END REUSABLE AI PIPELINE CORE =========================================
