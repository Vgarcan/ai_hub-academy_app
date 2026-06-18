# Project Overview

## Purpose

`ai_hub` is a reusable Django app for configuring, operating, and auditing AI agent systems from Django Admin.

It gives a host project an admin-driven AI operations layer where staff can define providers, models, agents, tools, knowledge, fixed workflows, autonomous sessions, and execution telemetry without scattering AI logic across views, tasks, and scripts.

## Problem It Solves

AI features become hard to maintain when prompts, provider calls, tool access, execution rules, and debug logs are hidden inside product code.

`ai_hub` separates those concerns:

- product code owns the business domain,
- `ai_hub` owns reusable AI configuration and execution primitives,
- admin users can inspect how AI is configured,
- developers can integrate AI workflows without rebuilding the platform layer each time,
- failed runs leave inspectable traces instead of disappearing into logs.

## Two Workspaces

`ai_hub` has two workspaces.

| Workspace | Mental Model | Use Case |
| --- | --- | --- |
| Orchestrator | Known ordered steps | Extract, classify, enrich, summarize, persist |
| GAME | Goal-driven decision loop | Decide, act, observe, remember, stop |

The workspaces are visually separated in Admin, but they share the same technical base:

- providers,
- models,
- agents,
- knowledge,
- tools,
- execution telemetry.

## Reusable Foundation

`ai_hub` supports:

- provider configuration,
- model configuration,
- agent profiles with prompts and contracts,
- knowledge collections and documents,
- tool definitions,
- sequential pipeline orchestration,
- GAME-style goal sessions,
- staff-only execution endpoints,
- execution sessions and step logs,
- admin control center metrics,
- guided forms and examples.

## What The App Does Not Own

`ai_hub` should not own:

- public product UI,
- billing,
- product-specific database objects,
- final domain persistence,
- user-facing business flows,
- private secrets.

Those responsibilities belong to the host project.

## Admin Experience

The Admin experience is part of the product.

`ai_hub` should feel like an operational backend, not only a raw Django model list. It uses:

- an AI Hub home dashboard,
- separate Orchestrator and GAME workspaces,
- shared resource sections,
- a visual connection graph,
- recent sessions and health metrics,
- guided changelists,
- guided forms with examples and placeholders,
- readable execution timelines.

The design goal is that a non-expert admin can understand what to create next and why.

## Portability Rule

If a feature only makes sense for one product domain, keep it in the host project or behind a small adapter.

If a feature helps any AI-enabled Django project, it belongs in `ai_hub`.
