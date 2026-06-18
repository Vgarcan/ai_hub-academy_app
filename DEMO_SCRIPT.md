# AI Hub Academy — 10-Minute Demo Script

## Setup (before the demo)

```bash
# 1. Apply migrations
python manage.py migrate

# 2. Seed training data (providers, agents, tutorials)
python manage.py seed_academy_training_data

# 3. Import documentation
python manage.py import_academy_docs

# 4. (Optional) Seed Ollama GAME agents
python manage.py seed_ollama_agents      # requires Ollama at 192.168.2.132:11434
python manage.py run_doc_sync            # run the doc sync agent once

# 5. Create admin user
python manage.py createsuperuser

# 6. Start server
python manage.py runserver
```

Open http://localhost:8000/ and log in as admin.

---

## 1. The Landing Page (1 min)

Open http://localhost:8000/

**Say:** "This is AI Hub Academy — a demonstration of how AI Hub can be used to build, run, and teach AI workflows inside any Django project."

Point to:
- System status (providers, agents, sessions)
- The three pillars: Docs, Tutorials, Assistant
- The value statement: "AI Hub not only executes workflows — it teaches them"

---

## 2. Documentation (1 min)

Open http://localhost:8000/docs/

**Say:** "All AI Hub documentation is stored in the database, imported from Markdown files. The team can update docs without redeploying."

- Show the page list (14 pages, 172 chunks)
- Open "Core Concepts"
- Show the sidebar navigation and section headings
- Search for "GAME" — show results with source links

---

## 3. Ask the AI Assistant (2 min)

Open http://localhost:8000/assistant/

**Say:** "The documentation assistant answers questions using the imported docs. Every answer goes through an AI Hub ExecutionSession — so every question is audited."

Ask: "What is the difference between Orchestrator and GAME?"

Show:
- The answer with source citations
- The sources panel at the bottom
- If logged in as staff, the Session link

**Say:** "Notice that the assistant cites its sources and never invents features. This is by design — the system prompt enforces factual grounding."

---

## 4. AI Hub Admin — The Control Room (2 min)

Open http://localhost:8000/admin/ai_hub/

**Say:** "This is where AI Hub lives. Every component is configured here — no code changes needed."

Show:
- Provider configs: Academy Training Provider (training mode, no API key)
- Model configs: training/assistant
- Agent profiles: Input Normalizer, Ticket Classifier, AI Hub Documentation Assistant
- Pipeline definitions: Ticket Triage Pipeline (2 steps)

Open the Ticket Triage Pipeline. Show:
- Step 1: Input Normalizer (normalises and validates)
- Step 2: Ticket Classifier (categorises and prioritises)
- Input/output contracts on each agent

**Say:** "Contracts enforce data quality. If step 1 doesn't produce the required fields, step 2 fails with a clear error — not a silent failure buried in a log."

---

## 5. Run a Pipeline on a Ticket (2 min)

Open http://localhost:8000/admin/support_demo/supportticket/

**Say:** "The Support Demo app connects real domain objects to AI Hub. Let's classify a ticket."

- Select "Login page returns 500 error after update"
- Run the **Run AI triage** action
- Open the ticket — show that it now has an AI session linked
- Click the session link

Show the ExecutionSession:
- Status: success
- Execution step runs: step 1 (Input Normalizer) and step 2 (Ticket Classifier)
- Expand step 2 — show request_payload, response_payload, latency_ms

**Say:** "This is full auditability. We know exactly what was sent to the model, what it returned, and how long it took. This is governance-grade tracing."

---

## 6. Tutorials (1 min)

Open http://localhost:8000/tutorials/

**Say:** "Academy includes an interactive tutorial system. Missions guide users through building their own AI Hub setup step by step."

Show:
- Module 0: Orientation → Enter The Control Room
- Module 1: Providers → Power Up The Lab, Install Your First Engine
- Module 4: Orchestrator → Build Your First Conveyor Belt
- Module 7: Capstone → Complete Workflow

Open a mission (e.g., "Build Your First Conveyor Belt"):
- Show the goal, instructions, related docs links
- Show the "Ask Assistant" suggested questions panel
- Show the Check Mission button with live feedback

**Say:** "When a user clicks Check Mission, the system queries the database to verify they actually completed the task in Admin. It's not a quiz — it validates real configuration."

---

## 7. Enterprise Value Summary (1 min)

**Say:**

> "What you've seen is a complete adoption platform built on top of AI Hub in Django:
>
> - **Auditability**: Every AI call is logged with full request/response telemetry. You know who ran what, when, with what data, and what the model returned.
>
> - **Governance**: Input/output contracts prevent bad data from flowing between agents. Training mode lets teams onboard without API costs.
>
> - **Reusability**: Providers, models, agents and pipelines are configured once and reused across any Django app in the organisation.
>
> - **Adoption**: The Academy layer means new developers can learn the platform from inside the platform — no separate documentation site needed.
>
> AI Hub is not just a tool for running AI. It's an operating model for AI in the enterprise."

---

## Appendix: Key URLs

| URL | Description |
|-----|-------------|
| `/` | Landing page |
| `/docs/` | Documentation browser |
| `/docs/search/?q=GAME` | Documentation search |
| `/assistant/` | AI Documentation Assistant |
| `/tutorials/` | Tutorial modules |
| `/progress/` | User mission progress |
| `/dashboard/` | Visual dashboard (Control Room) |
| `/dashboard/providers/` | Providers & Models |
| `/dashboard/agents/` | Agent Profiles |
| `/dashboard/pipelines/` | Pipeline Definitions |
| `/dashboard/sessions/` | Execution Sessions audit trail |
| `/dashboard/sessions/<pk>/` | GAME iteration detail view |
| `/admin/` | Django Admin |
| `/admin/ai_hub/` | AI Hub Admin home |
| `/admin/ai_hub/workspaces/orchestrator/` | Orchestrator workspace |
| `/admin/ai_hub/workspaces/game/` | GAME workspace |
| `/admin/support_demo/supportticket/` | Support tickets |

## Running tests

```bash
python manage.py test --verbosity=2   # runs all 74 tests
```
