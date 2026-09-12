"""Shared Build Console wizard builder logic (P1 extraction).

The GAME and Orchestrator wizards both resolve the same object chain — engine
(provider + model), agent, toolboxes and knowledge — before their workspace-
specific steps. That shared logic lives here so the two admin builder functions
stay thin and cannot drift apart.

Each helper mutates the passed ``errors`` dict (field name -> message) and returns
the resolved object or ``None``. Callers are expected to run inside a
``transaction.atomic()`` block and to short-circuit when ``errors`` is non-empty,
so partial work rolls back.
"""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation

from django.utils.translation import gettext_lazy as _

from ai_hub.models import (
    AgentProfile,
    AgentToolboxAssignment,
    KnowledgeCollection,
    KnowledgeDocument,
    ModelConfig,
    ProviderConfig,
    Toolbox,
)
from ai_hub.services.knowledge_ingestion import ensure_initial_knowledge_chunk
from ai_hub.services.application_scope import require_single_active_scope


def parse_json_field(
    raw,
    default,
    *,
    errors=None,
    field_name: str = "",
    label: str = "JSON value",
):
    """Parse a JSON object and optionally report invalid input to the wizard."""
    if not raw:
        return default
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, dict):
            return parsed
        if errors is not None and field_name:
            errors[field_name] = _("%(label)s must be a JSON object.") % {
                "label": label
            }
        return default
    except (json.JSONDecodeError, AttributeError):
        if errors is not None and field_name:
            errors[field_name] = _("%(label)s must be valid JSON.") % {
                "label": label
            }
        return default


def resolve_engine(data, errors):
    """Reuse an existing ModelConfig or get_or_create provider + model.

    Returns the ModelConfig, or None when a field error was recorded.
    """
    engine_mode = data.get("engine_mode", "reuse")
    if engine_mode not in {"reuse", "create"}:
        errors["engine_mode"] = _("Select whether to reuse or create an engine.")
        return None
    if engine_mode == "reuse":
        mid = data.get("engine_reuse_model_id", "")
        try:
            return ModelConfig.objects.get(
                pk=int(mid),
                is_active=True,
                provider__is_active=True,
            )
        except (ModelConfig.DoesNotExist, ValueError, TypeError):
            errors["engine_reuse_model_id"] = _("Select an active model with an active provider.")
            return None

    provider_name = (data.get("engine_provider_name") or "").strip()
    if not provider_name:
        errors["engine_provider_name"] = _("Provider name is required.")
        return None
    provider_type = (
        data.get("engine_provider_type") or ProviderConfig.ProviderType.TRAINING
    )
    valid_provider_types = {
        choice for choice, _label in ProviderConfig.ProviderType.choices
    }
    if provider_type not in valid_provider_types:
        errors["engine_provider_type"] = _("Select a valid provider type.")
        return None

    model_name = (data.get("engine_model_name") or "").strip()
    if not model_name:
        errors["engine_model_name"] = _("Model name is required.")
        return None

    try:
        temp = Decimal(str(data.get("engine_temperature") or "0.30"))
    except (InvalidOperation, ValueError, TypeError):
        errors["engine_temperature"] = _("Temperature must be a number (e.g. 0.30).")
        return None
    if temp < Decimal("0") or temp > Decimal("2"):
        errors["engine_temperature"] = _("Temperature must be between 0 and 2.")
        return None

    provider, _created = ProviderConfig.objects.get_or_create(
        name=provider_name,
        defaults={
            "provider_type": provider_type,
            "is_active": True,
        },
    )
    if not provider.is_active:
        errors["engine_provider_name"] = _(
            "The provider with this name is inactive. Activate it or use another name."
        )
        return None
    if provider.provider_type == ProviderConfig.ProviderType.TRAINING:
        if model_name != "training" and not model_name.startswith("training/"):
            errors["engine_model_name"] = _(
                "Training-provider models must be named 'training' or start with 'training/'."
            )
            return None

    model_config, _created = ModelConfig.objects.get_or_create(
        provider=provider,
        model_name=model_name,
        defaults={"temperature_default": temp, "is_active": True},
    )
    if not model_config.is_active:
        errors["engine_model_name"] = _(
            "The model with this name is inactive. Activate it or use another name."
        )
        return None
    return model_config


def resolve_agent(data, model_config, errors, *, with_contracts):
    """Reuse an existing AgentProfile or create a new one bound to ``model_config``.

    ``with_contracts`` controls whether the Orchestrator-only input/output
    contract fields are parsed and stored. Returns the AgentProfile or None.
    """
    agent_mode = data.get("agent_mode", "reuse")
    if agent_mode not in {"reuse", "create"}:
        errors["agent_mode"] = _("Select whether to reuse or create an agent.")
        return None
    if agent_mode == "reuse":
        aid = data.get("agent_reuse_id", "")
        try:
            return AgentProfile.objects.get(
                pk=int(aid),
                is_active=True,
                model_config__is_active=True,
                model_config__provider__is_active=True,
            )
        except (AgentProfile.DoesNotExist, ValueError, TypeError):
            errors["agent_reuse_id"] = _(
                "Select an active agent whose model and provider are active."
            )
            return None

    agent_name = (data.get("agent_name") or "").strip()
    if not agent_name:
        errors["agent_name"] = _("Agent name is required.")
        return None
    if AgentProfile.objects.filter(name=agent_name).exists():
        errors["agent_name"] = _(
            "An agent with this name already exists. Choose reuse or use another name."
        )
        return None

    fields = {
        "name": agent_name,
        "role": (data.get("agent_role") or "").strip(),
        "system_prompt": (data.get("agent_prompt") or "").strip(),
        "model_config": model_config,
        "is_active": True,
    }
    if with_contracts:
        fields["input_contract"] = parse_json_field(
            data.get("agent_input_contract"),
            {},
            errors=errors,
            field_name="agent_input_contract",
            label=_("Agent input contract"),
        )
        fields["output_contract"] = parse_json_field(
            data.get("agent_output_contract"),
            {},
            errors=errors,
            field_name="agent_output_contract",
            label=_("Agent output contract"),
        )
        if errors:
            return None
    # Explicit ownership. The Build Console has no scope selector yet, so it
    # resolves the single active scope and refuses when that is ambiguous.
    fields["application_scope"] = require_single_active_scope()
    return AgentProfile.objects.create(**fields)


def attach_toolboxes(agent, data, errors):
    """Attach selected toolboxes to the agent (idempotent get_or_create)."""
    for tb_id in data.getlist("agent_toolbox_ids"):
        try:
            tb = Toolbox.objects.get(pk=int(tb_id), is_active=True)
            AgentToolboxAssignment.objects.get_or_create(
                agent=agent, toolbox=tb, defaults={"is_enabled": True}
            )
        except (Toolbox.DoesNotExist, ValueError, TypeError):
            errors["agent_toolbox_ids"] = _("One or more selected toolboxes could not be found.")


def attach_knowledge(agent, data, errors):
    """Reuse or create a knowledge collection and attach it to the agent."""
    knowledge_mode = data.get("knowledge_mode", "none")
    if knowledge_mode not in {"none", "reuse", "create"}:
        errors["knowledge_mode"] = _("Select a valid knowledge mode.")
        return
    if knowledge_mode == "reuse":
        try:
            coll = KnowledgeCollection.objects.get(
                pk=int(data.get("knowledge_collection_id", "")),
                is_active=True,
            )
            agent.knowledge_collections.add(coll)
        except (KnowledgeCollection.DoesNotExist, ValueError, TypeError):
            errors["knowledge_collection_id"] = _("The selected knowledge collection could not be found.")
    elif knowledge_mode == "create":
        coll_name = (data.get("knowledge_collection_name") or "").strip()
        doc_title = (data.get("knowledge_doc_title") or "").strip()
        doc_content = (data.get("knowledge_doc_content") or "").strip()
        if not coll_name:
            errors["knowledge_collection_name"] = _("Collection name is required.")
        elif KnowledgeCollection.objects.filter(name=coll_name).exists():
            errors["knowledge_collection_name"] = _(
                "A collection with this name already exists. Choose reuse or use another name."
            )
        if not doc_title:
            errors["knowledge_doc_title"] = _("Document title is required.")
        if not doc_content:
            errors["knowledge_doc_content"] = _(
                "Document content is required to create a retrievable chunk."
            )
        if errors:
            return
        coll = KnowledgeCollection.objects.create(
            name=coll_name,
            is_active=True,
            application_scope=require_single_active_scope(),
        )
        document = KnowledgeDocument.objects.create(
            collection=coll,
            title=doc_title,
            curated_text=doc_content,
            status=KnowledgeDocument.Status.ACTIVE,
        )
        ensure_initial_knowledge_chunk(document)
        agent.knowledge_collections.add(coll)
