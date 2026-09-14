# AI Transformation Bootcamp

This repository is the complete, week-by-week record of a hands-on AI
Transformation bootcamp: daily Python/ML/LLM exercises building up in
difficulty, followed by two capstone projects that put everything together
into working, testable software. Nothing here is a slide deck — every claim
in this README is backed by runnable code, tests, or a trained model
somewhere in this repo.

**The arc, in one line:** start with plain Python and APIs → learn classical
machine learning on a real business problem → learn LLMs, retrieval and
multi-agent orchestration → prove both skill sets with a full capstone each.

---

## Repository map

| Path | What it is |
|---|---|
| [`week_1/`](week_1/) | Days 0–5 — Python fundamentals, APIs, configuration |
| [`week_2/`](week_2/) | Days 6–8 — classical ML on an IT ticket dataset |
| [`week_3/`](week_3/) | Days 11–14 — LLM APIs, RAG, and multi-agent systems |
| [`projects/capstone_1/`](projects/capstone_1/) | **Capstone 1 — SignalCraft:** flight-delay risk prediction (classical ML) |
| [`projects/capstone_2/`](projects/capstone_2/) | **Capstone 2 — OpsMate:** agentic IT service-desk assistant (LangGraph + FastAPI + Streamlit) |
| [`data/`](data/) | Shared datasets and the 40-document knowledge base used by Week 3 and Capstone 2 |
| [`LEARNING_LOG.md`](LEARNING_LOG.md) | Daily "what I built / what broke / what I couldn't answer" journal |

---

## The learning arc

### Week 1 — Python & software foundations (Days 0–5)

The groundwork: no ML, no AI yet — just the engineering habits everything
later depends on.

| Day | Focus | Artefact |
|---|---|---|
| 0 | Environment setup | Python, VS Code, Git, GitHub, a virtualenv, a ServiceNow PDI |
| 1 | Python packages & modules | `week_1/day_1/day1_katas.py` — string/parsing katas |
| 2 | File handling & first Git workflow | `week_1/day_2/ticket_utils/` — a small ticket-processing package |
| 3 | Data profiling | `week_1/day_3/day3_profiling.ipynb` — 50,000-row synthetic incident dataset, SLA breach rates by category/priority |
| 4 | APIs, both sides | `week_1/day_4/` — calling external APIs with `requests`, then building one with **FastAPI** |
| 5 | Configuration & secrets | `week_1/day_5/` — `.env` / `.env.example`, `os.getenv()`, boolean env-var parsing |

### Week 2 — Classical ML on real ticket data (Days 6–8)

The first machine-learning problem: **will this IT ticket breach its SLA?**
This is where the bootcamp's central ML lesson lands hard.

- **Day 6** — a "predict the majority class" baseline scores 96% accuracy
  and is *useless*: it misses every single SLA breach (0% recall). This
  motivates every evaluation choice made for the rest of the bootcamp —
  accuracy alone is treated as a red flag, not a result, from here on.
- **Day 7** — a real feature-engineering + modelling workflow: time-of-day,
  business-hours, priority, reassignment and text features; three model
  families (Logistic Regression, Random Forest, Gradient Boosting) compared
  on **PR-AUC**, not accuracy. Gradient Boosting wins narrowly
  (0.7523 vs. 0.7518 for Logistic Regression).
- **Day 8** — cleaning and reading real ticket text, the stepping stone
  toward treating ticket descriptions as first-class model input.
- `week_2/day_7/app.py` + `week_2/day_7/features.py` — a small **Streamlit**
  app that scores a new ticket at intake using the Day 7 model, with the
  feature logic shared between training and serving so the two can never
  drift apart (the same principle Capstone 1's serving layer later scales
  up).

### Week 3 — LLMs, RAG, and multi-agent systems (Days 11–14)

This week is where the bootcamp pivots from "train a model" to "orchestrate
a language model safely" — and it directly rehearses everything Capstone 2
is built from.

| Day | Focus | What it introduces |
|---|---|---|
| 11 | First LLM API calls | Reading a response's metadata *before* trusting its output text |
| 12 | Retrieval-Augmented Generation | Embedding a 40-document knowledge base (30 articles + 10 runbooks) into **ChromaDB**; the token/dollar cost of embedding vs. querying |
| 13 | A tool-using agent | `week_3/day_13/Triage.py` — a **LangGraph** ReAct agent (`create_react_agent`) that calls tools, backed by an audit trace (`day13_trace.jsonl`) |
| 14 | Multi-agent + human-in-the-loop | `week_3/day_14/Ask.py` — adds `interrupt()` / `Command(resume=...)`: the agent can pause and wait for a **human decision** before acting |

Day 13 and Day 14 are, quite literally, Capstone 2's prototype: the same
libraries (`langgraph`, `langchain-chroma`, `langchain-openai`), the same
pattern (tool-calling agent → human approval gate), at a fraction of the
scale.

---

## Capstone 1 — SignalCraft: flight-delay risk prediction

**Question answered:** *given a flight's schedule, will it arrive more than
15 minutes late?* — answered the afternoon before departure (T-24h), for a
station duty manager deciding whether to pre-position standby ground crew.

- **Data:** the U.S. DOT's public BTS On-Time Reporting dataset —
  7,043,316 raw rows across 12 months (6,896,283 after cleaning). Public,
  no login wall, large enough for the evaluation to mean something.
- **Pipeline:** a single CLI (`python -m app.cli <stage>`) runs each stage
  as its own file-in/file-out step — `acquire → prepare → seal → features →
  train → evaluate → report` — so a failure late in the run never wastes an
  expensive earlier step.
- **Modelling:** 24 features (18 derived) with a hard leakage guard against
  any post-resolution information; three model families (Logistic
  Regression, Random Forest, LightGBM) validated with **expanding-window
  cross-validation** (never random k-fold, since this is time-series data);
  a decision threshold chosen from a target recall rather than the default
  0.5; SHAP explanations plus a written analysis of the 20 worst
  predictions.
- **Serving:** a **Streamlit** app (`streamlit_app.py`) a duty manager could
  actually use — every prediction ships with its SHAP explanation and an
  honest, visible caveat about where the model is least reliable (e.g.
  early-morning departures).
- **Engineering discipline:** the sealed holdout set is opened exactly once
  and the unseal is one-way and recorded; the data acquisition step
  documents its own source, licence and retrieval date
  (`data/manifest.json`); notebooks are generated *after* the package, from
  the package, so no logic is ever duplicated between "the real code" and
  "the demo notebook."

Eleven graded requirements (R1–R11) map onto this, from *reproducible
acquisition* through *decision framing* to *engineering quality* — see
[`projects/capstone_1/README.md`](projects/capstone_1/README.md) for the
full traceability table.

**Run it:** `python -m app.cli train` to reproduce the model, then
`streamlit run streamlit_app.py` to try the UI.

---

## Capstone 2 — OpsMate: an agentic IT service-desk assistant

**Question answered:** *can an AI investigate an IT ticket end-to-end and
still be safe to deploy?* — OpsMate's own tagline: *"It investigates freely.
It changes nothing without you."*

- **What it does:** a user describes an IT problem in a chat window ("U-1042
  is locked out after failed logins"). A pipeline of small, specialised
  agents — Intake, Diagnostic, Knowledge, Action, Supervisor, Handover —
  investigates using a real database and a 40-document knowledge base, then
  either resolves it, asks one clarifying question, or escalates to a human
  engineer.
- **The non-negotiable rule:** the AI can *propose* a fix (unlock an
  account, reset a password, restart a service…) but **a named human must
  click Approve** before anything real changes. This is enforced three
  separate ways at once (not just "the prompt says so"):
  1. `interrupt()` genuinely suspends the running process until a human
     answers.
  2. Every write tool independently re-checks a cryptographic approval
     ledger before touching anything — even a fully prompt-injected agent
     cannot bypass this.
  3. The final message the user reads is composed from verified state only,
     never from the original ticket text — so an injected instruction can
     never leak into a false "done!" message.
- **Where judgement is deliberately *not* an agent:** knowledge-base
  conflict resolution, the approval ledger, "never re-enable a leaver's
  account," and the audit-cost arithmetic are all plain, deterministic
  Python — because a model that can be persuaded is not a control.
- **Stack:** LangGraph (orchestration) · LangChain + `gpt-4o-mini` (agents)
  · ChromaDB + `text-embedding-3-small` (RAG) · MySQL (the estate + audit
  trail) · **FastAPI** (the HTTP service, 12 endpoints) · **Streamlit** (the
  operator console — chat, approval inbox, trace viewer, cost/latency
  dashboard) · pytest.
- **Proven, not claimed:** `python -m app.cli --attack` fires six real
  prompt-injection attempts (fake approvals, fake "maintenance mode," fake
  prior tool output, a leaver's account under social pressure) and reports
  **0 unapproved changes, 0 false claims of success** — a SQL join between
  the `action_log` and `approvals` tables makes that claim independently
  checkable, not just asserted.

Ten graded requirements (R1–R10) map onto this — see
[`projects/capstone_2/README.md`](projects/capstone_2/README.md) for the
full traceability table, the architecture diagram, and an honest "results
and known limitations" section (including a deliberately-cut evaluation
harness, described plainly rather than hidden).

**Run it:**
```
uvicorn app.api.main:app --port 8000 --workers 1     # terminal 1 — the API
streamlit run streamlit_app.py                        # terminal 2 — the console
```

---

## The throughline

| Skill introduced | First seen | Proven at scale in |
|---|---|---|
| APIs, both directions | Week 1, Day 4 (`requests` + FastAPI) | Capstone 2's 12-endpoint FastAPI service |
| "Accuracy lies, check the confusion matrix" | Week 2, Day 6 | Capstone 1's PR-AUC / recall-driven threshold |
| Feature engineering without leakage | Week 2, Day 7 | Capstone 1's leakage-guarded, 24-feature pipeline |
| Train/serve logic sharing one source of truth | Week 2's `app.py` + `features.py` | Capstone 1's `app/serving/predict.py` |
| Calling an LLM correctly | Week 3, Day 11 | Every agent call in Capstone 2 |
| RAG over a knowledge base | Week 3, Day 12 | Capstone 2's grounded, cited Knowledge agent |
| A tool-calling agent | Week 3, Day 13 (`Triage.py`) | Capstone 2's Diagnostic agent |
| Human-in-the-loop approval | Week 3, Day 14 (`Ask.py`) | Capstone 2's three-lock approval gate |

Two capstones, two very different kinds of AI risk, two different sets of
engineering answers: **Capstone 1** shows the discipline classical ML needs
(sealed holdouts, leakage guards, honest thresholds, explainability).
**Capstone 2** shows the discipline agentic AI needs (structural safety
gates, grounded knowledge, a deterministic core wrapped around a
probabilistic model, and a complete audit trail). Together they're the
argument this bootcamp is built to make: AI transformation work is mostly
disciplined software engineering, with the model as one component — not the
whole system.

---

## Further reading

- [`LEARNING_LOG.md`](LEARNING_LOG.md) — the daily journal this summary is drawn from
- [`projects/capstone_1/`](projects/capstone_1/) and [`projects/capstone_2/README.md`](projects/capstone_2/README.md) — full project documentation
