# PRD — Computer-Use Automation System

**Assignment:** interface.ai take-home, "Computer-Use Automation System"
**Repo:** RePlay-Interface
**Status:** planning complete, implementation not started

---

## 1. What we are building

A system that lets an AI agent operate legacy back-office banking applications that expose no API,
by driving their UI the way a human operator would.

The through-line the brief asks us to hold:

> The model discovers. The artifact becomes a reusable capability. Deterministic replay is how the AI agent invokes it in production.

Three phases, one vertical slice:

1. **Discovery** — an LLM-driven observe → decide → act loop completes a natural-language goal against a live UI.
2. **Artifact** — the successful run is distilled into a typed, versioned, reviewable capability (not a transcript dump).
3. **Replay** — that artifact executes deterministically with input parameters, no LLM in the decision loop, returning typed outputs or a classified outcome.

Wrapped around all three: safety guardrails, evidence, and a human-escalation path that can take over the *same* live session.

## 2. Why this shape

The grading weight, in the brief's own order, is: system design → correctness of the core loop → robustness and error handling → human-in-the-loop → generalization → safety → code quality → communication.

Feature breadth is explicitly *not* rewarded. Neither is scaling infrastructure. So: one narrow thread that touches every core requirement, with real depth on the three load-bearing pieces — **artifact schema**, **deterministic replay plus error handling**, and **the safety/escalation model**.

## 3. Decisions and rationale

| Decision | Choice | Why |
|---|---|---|
| Language / runtime | Python 3.12 via `uv` | Pydantic v2 gives runtime-validated, versioned artifact schemas with JSON Schema export — the artifact contract is the top-graded item. `uv` because system Python is 3.9.6. |
| Computer-use mechanism | Playwright (Python), **accessibility-tree first**, screenshot as the model's visual channel | The brief says bias toward what still works with no clean DOM. The a11y tree exists on desktop apps too, so the perceive/act seam extends beyond the browser — this *is* our answer to requirement 3.7. |
| Element targeting | Ranked **locator ladder**, recorded per step, tried in order at replay | Determinism and locator strategy are top-3 evaluated. One selector is a guess; a ranked ladder with recorded fallback tiers is a defensible strategy, and which tier fired becomes an observability signal. |
| LLM | OpenAI, behind a thin `LLMClient` interface | Constrained tool-calling over a small action vocabulary, not free-form. Provider is swappable but we target one. |
| Target application | Locally built hostile legacy app, **MERIDIAN CORE** | The brief's deliverables require a replay that hits an error state. A public site cannot be made to return "record not found" or a session timeout on demand. Local also means zero ToS risk and a grader can run it offline. |
| Hostility level | Genuine legacy: `<frameset>`, nested table layout, no test IDs, `<input name=f7>`, labels associated by table-cell adjacency, `confirm()` dialog, server-side session expiry | Makes the a11y-first argument demonstrably correct instead of merely asserted. CSS selectors are hopeless here; role+name still works. |
| Architecture | Single process. CLI-first, thin FastAPI for two jobs only | "Simpler is fine if justified." FastAPI earns its place twice: the agent-facing capability catalog, and the operator escalation console. No queues, no DB, no services. |
| Artifact storage | JSON files on disk under `artifacts/` | A capability is a reviewable document. Files diff in a PR; rows in a database do not. |
| Escalation | Headed browser + control-token operator console | The human acts in the real browser window — same `BrowserContext`, same cookies, same session. Automation blocks on a control token rather than tearing down and respawning. |
| Stretch goals (2 of 6) | Capability catalog; cross-tenant variant demo | The catalog closes the brief's own through-line. The variant demo converts requirement 3.7 from a prose essay into a running demonstration — 3.7 is otherwise our weakest, design-only section. |

## 4. Scope

### In scope — must work end to end

| # | Requirement | How we satisfy it |
|---|---|---|
| 3.1 | Goal-driven agent loop | `replay discover --goal "..." --target <url>`; observe (a11y snapshot + screenshot) → decide (LLM tool call) → act; stops on goal met, max steps, timeout, or dead-end. |
| 3.2 | Structured artifact | Pydantic `CapabilityArtifact`: ordered steps, per-target locator ladder **with recorded robustness rationale**, typed inputs, typed outputs, checkpoints, declared business outcomes, policy block, provenance. Semver-versioned. |
| 3.3 | Deterministic replay | `replay run <capability> --member-id 12345`. No LLM. Locator ladder + explicit waits + checkpoint assertion. Result contract separates business outcome / recoverable / hard failure. |
| 3.4 | Safety and policy | Configurable allowlist (domains, routes, action types). Three risk classes with distinct handling. Sensitive params never persisted to artifacts, logs, or screenshots. |
| 3.5 | Evidence | Structured JSONL run log plus screenshot and a11y snapshot per step; on failure, additionally the DOM snapshot and the failing condition's expected-vs-observed. |
| 3.6 | Escalation and handoff | Stuck detection → `InterventionRequest` with context → control token released → operator console → human acts in live session → resume → checkpoint re-asserted. Human actions captured into evidence. |
| 3.7 | Heterogeneity and scale | `Surface` protocol as the perceive/act seam (web implemented; legacy-web and desktop designed). Base artifact plus per-tenant override layer, demonstrated against a second app variant. |

### Deliberately mocked or stubbed, at a clean seam

- **Desktop surface** — `Surface` protocol defined and argued; only the web implementation ships.
- **Operator console UI** — functional but deliberately bare. Real mechanism, mock chrome.
- **Multi-tenant plumbing** — no tenant registry or per-tenant infra. The *artifact* is designed for reuse; the infrastructure around it is not built. The brief calls building it a negative.

### Explicitly out of scope

Real-time co-browsing console. Queues, clusters, workers. Any real bank system or real PII. Authentication beyond a mock login. Feature breadth of any kind.

## 5. Target application — MERIDIAN CORE

A local Flask app impersonating a 1990s credit-union teller console.

**Primary flow (the recorded capability):** member search → member detail → open sub-account → confirmation screen.

**Deliberate hostility:** `<frameset>` splitting nav from content; nested `<table>` layout; no `data-testid` anywhere; form fields named `f7`, `f12`; labels associated with inputs only by table-cell adjacency; a native `confirm()` on submit; server-side session expiry.

**Failure injection**, so replay error handling is demonstrable on demand:

| Flag | Simulates | Expected classification |
|---|---|---|
| `?inject=not_found` | Member ID does not exist | Business outcome — `MEMBER_NOT_FOUND` |
| `?inject=validation` | Server rejects a field | Business outcome — `VALIDATION_REJECTED` |
| `?inject=denied` | Operator lacks permission | Business outcome — `PERMISSION_DENIED` |
| `?inject=dialog` | Unexpected interstitial | Recoverable — dismiss and continue |
| `?inject=slow` | 8s response | Recoverable — wait and retry |
| `?inject=timeout` | Session expired mid-flow | Hard failure, or escalate |
| `?inject=error500` | App error | Hard failure |

**Variant app (`MERIDIAN CORE — Northgate FCU`)** for the cross-tenant demo: same vendor product, different branding, one relocated field, two renamed labels, different route prefix. The same artifact replays against it through a per-variant override layer.

## 6. Deliverables

Per the brief's exact paths — **verify these three strings against the PDF before submission**, our text extraction of the monospace font was lossy:

1. **`README.md`** — setup, config, keys, how to run without live services, and a demo path giving the exact commands for discovery then replay.
2. **`REPORT.md`** — 1–3 pages, using the brief's seven mandated headings verbatim:
   1. Architecture 2. Artifact schema 3. Determinism & error handling 4. Heterogeneity & multi-tenant 5. Escalation & handoff 6. Safety 7. Cuts
3. **`evidence/`** — a saved example artifact plus logs from a discovery run and a replay run, including **at least one replay that hits an error or exceptional state**.

Public GitHub repo. Repo URL on its own line, emailed to `assignments@interface.ai` from the address applied with. No zip. No secrets in the repo.

## 7. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Toolchain | `uv` installed, Python 3.12 project, Playwright browsers, `OPENAI_API_KEY` loaded from env only |
| M1 | MERIDIAN CORE | Hostile app serves the full flow; every injection flag verified by hand |
| M2 | Surface layer | `Surface` protocol; Playwright web impl; a11y snapshot across frames; screenshot; action primitives |
| M3 | Artifact schema | Pydantic models, JSON Schema export, round-trip tests, worked example committed |
| M4 | Discovery loop | Real LLM run completes the goal against the live app; transcript and evidence captured |
| M5 | Artifact synthesis | Successful run distilled into a versioned artifact, decoupled from the transcript |
| M6 | Replay engine | Deterministic replay with parameters; locator ladder; checkpoints; typed outputs |
| M7 | Error taxonomy | All seven injections classified correctly into the three-way result contract |
| M8 | Policy layer | Allowlist enforced; risk classes handled; redaction verified by test |
| M9 | Escalation | Stuck detection → console → human acts in live session → resume → checkpoint re-asserted |
| M10 | Capability catalog | `GET /capabilities`, `POST /capabilities/{name}:invoke`, one invocation demonstrated |
| M11 | Cross-tenant demo | Same artifact replays against the variant via override layer |
| M12 | Docs and evidence | `README.md`, `REPORT.md`, `evidence/` complete; fresh-clone run verified |

## 8. Acceptance criteria

- A single genuine LLM-driven discovery run completes the goal against the live app, with committed evidence proving it happened.
- The emitted artifact replays deterministically **N times with identical outputs**, LLM never invoked.
- Each of the seven injected conditions produces the correct classification — business outcome, recoverable, or hard failure — never a crash and never a silent pass.
- An unparameterised replay against an unknown member returns `MEMBER_NOT_FOUND` as a *result*, not an exception. (The brief names conflating these as the most common design mistake.)
- Attempting an action outside the allowlist is refused and logged.
- No sensitive value appears in any artifact, log, or screenshot — asserted by an automated test.
- Escalation transfers control of the live session to a human and resumes on the same session, with the human's actions recorded.
- `git clone` → documented setup → demo path works on a clean machine.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Accessibility snapshots across `<frameset>` are fiddly in Playwright | Prove frame traversal in M2 before anything depends on it |
| The one required real LLM run is flaky or expensive | Cache the transcript as a fixture the moment it succeeds; a `MockLLM` keeps everything else runnable and testable offline |
| Legacy app has near-zero accessible names, weakening role+name targeting | This is the actual finding to report — the ladder's lower tiers exist precisely for it, and which tier fires is logged as evidence |
| Scope creep across seven requirements | Thin-but-real everywhere, deep only on schema, replay/errors, and safety/escalation. Every cut documented in `REPORT.md` §7 |
| Over-building infrastructure | The brief penalises it. No queues, no DB, no tenant plumbing |
