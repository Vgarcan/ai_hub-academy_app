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


def parse_json_field(raw, default):
    """Parse a JSON object from a form field, falling back to ``default``."""
    if not raw:
        return default
    try:
        parsed = json.loads(raw.strip())
        return parsed if isinstance(parsed, dict) else default
    except (json.JSONDecodeError, AttributeError):
        return default


def resolve_engine(data, errors):
    """Reuse an existing ModelConfig or get_or_create provider + model.

    Returns the ModelConfig, or None when a field error was recorded.
    """
    engine_mode = data.get("engine_mode", "reuse")
    if engine_mode == "reuse":
        mid = data.get("engine_reuse_model_id", "")
        try:
            return ModelConfig.objects.get(pk=int(mid))
        except (ModelConfig.DoesNotExist, ValueError, TypeError):
            errors["engine_reuse_model_id"] = _("Select a valid model.")
            return None

    provider_name = (data.get("engine_provider_name") or "").strip()
    if not provider_name:
        errors["engine_provider_name"] = _("Provider name is required.")
        return None

    provider, _created = ProviderConfig.objects.get_or_create(
        name=provider_name,
        defaults={
            "provider_type": data.get("engine_provider_type") or ProviderConfig.ProviderType.TRAINING,
            "is_active": True,
        },
    )
    model_name = (data.get("engine_model_name") or "training/starter").strip()
    try:
        temp = Decimal(str(data.get("engine_temperature") or "0.30"))
    except (InvalidOperation, ValueError, TypeError):
        errors["engine_temperature"] = _("Temperature must be a number (e.g. 0.30).")
        temp = Decimal("0.30")
    model_config, _created = ModelConfig.objects.get_or_create(
        provider=provider,
        model_name=model_name,
        defaults={"temperature_default": temp, "is_active": True},
    )
    return model_config


def resolve_agent(data, model_config, errors, *, with_contracts):
    """Reuse an existing AgentProfile or create a new one bound to ``model_config``.

    ``with_contracts`` controls whether the Orchestrator-only input/output
    contract fields are parsed and stored. Returns the AgentProfile or None.
    """
    agent_mode = data.get("agent_mode", "reuse")
    if agent_mode == "reuse":
        aid = data.get("agent_reuse_id", "")
        try:
            return AgentProfile.objects.get(pk=int(aid))
        except (AgentProfile.DoesNotExist, ValueError, TypeError):
            errors["agent_reuse_id"] = _("Select a valid agent.")
            return None

    agent_name = (data.get("agent_name") or "").strip()
    if not agent_name:
        errors["agent_name"] = _("Agent name is required.")
        return None

    fields = {
        "name": agent_name,
        "role": (data.get("agent_role") or "").strip(),
        "system_prompt": (data.get("agent_prompt") or "").strip(),
        "model_config": model_config,
        "is_active": True,
    }
    if with_contracts:
        fields["input_contract"] = parse_json_field(data.get("agent_input_contract"), {})
        fields["output_contract"] = parse_json_field(data.get("agent_output_contract"), {})
    return AgentProfile.objects.create(**fields)


def attach_toolboxes(agent, data, errors):
    """Attach selected toolboxes to the agent (idempotent get_or_create)."""
    for tb_id in data.getlist("agent_toolbox_ids"):
        try:
            tb = Toolbox.objects.get(pk=int(tb_id))
            AgentToolboxAssignment.objects.get_or_create(
                agent=agent, toolbox=tb, defaults={"is_enabled": True}
            )
        except (Toolbox.DoesNotExist, ValueError, TypeError):
            errors["agent_toolbox_ids"] = _("One or more selected toolboxes could not be found.")


def attach_knowledge(agent, data, errors):
    """Reuse or create a knowledge collection and attach it to the agent."""
    knowledge_mode = data.get("knowledge_mode", "none")
    if knowledge_mode == "reuse":
        try:
            coll = KnowledgeCollection.objects.get(pk=int(data.get("knowledge_collection_id", "")))
            agent.knowledge_collections.add(coll)
        except (KnowledgeCollection.DoesNotExist, ValueError, TypeError):
            errors["knowledge_collection_id"] = _("The selected knowledge collection could not be found.")
    elif knowledge_mode == "create":
        coll_name = (data.get("knowledge_collection_name") or "").strip()
        doc_title = (data.get("knowledge_doc_title") or "").strip()
        if coll_name and doc_title:
            coll = KnowledgeCollection.objects.create(name=coll_name, is_active=True)
            document = KnowledgeDocument.objects.create(
                collection=coll,
                title=doc_title,
                curated_text=(data.get("knowledge_doc_content") or "").strip(),
                status=KnowledgeDocument.Status.ACTIVE,
            )
            ensure_initial_knowledge_chunk(document)
            agent.knowledge_collections.add(coll)
