# AI Hub Core Foundation Baseline

Status: **GREEN / CLOSED**

Date: 2026-07-27

Baseline:

```text
67f139140f846693af015a4f97643df819373d2a
```

## Closed Foundation Work

- Foundation Phases 1–4.
- FND-P1-06 — Approval Expiry Recovery Deadlock.
- FFPR-001 — HTTP Tool Credential Isolation.
- FFPR-002 — Legacy Provider/Model Guard.
- FFPR-003 — Legacy Provider Failure Tool Audit Omission.

The independently reviewed Final Foundation Proof Restart 4 completed without
a reproducible Core P0/P1.

## Validation Evidence

```text
SQLite:
550 tests
11 expected skips
0 failures

PostgreSQL 16:
550 tests
4 expected skips
0 failures

Real Ollama:
Direct Agent PASS
Orchestrator PASS
controlled primary failure -> fallback PASS
official live module 4/4 PASS

/admin/login/:
HTTP 200
```

Deterministic and adversarial validation covered Provider/Model/Agent routing,
Tool resolution and audit, HTTP credential isolation, approvals and expiry,
Orchestrator fallback, GAME, Knowledge boundaries, Admin controls, supported
legacy entry points and PostgreSQL locking-sensitive paths.

One diagnostic constrained `qwen3:8b` to an artificial 64-token output budget
and received no final content after that reasoning budget was exhausted. This
was not classified as a Core defect: the actual 512-token Model configuration
produced content and passed the Direct Agent, Orchestrator and controlled
fallback slices.

## Meaning Of Closure

**GREEN / CLOSED** means the agreed Foundation closure criteria were satisfied
on this exact baseline. It does not mean the software can never contain bugs.
Future reproducible Core P0/P1 findings are new defects measured against this
historical baseline; they do not retroactively falsify the evidence collected
for this closure.

## Boundary Statement

```text
AI Hub Core Foundations:
GREEN / CLOSED

Academy / host-specific security findings:
separate boundary
```

An Academy or host-project finding does not reopen the reusable AI Hub Core
Foundation unless evidence demonstrates the same defect inside `ai_hub`.
