# Project Architecture — RePlay

Companion to `PRD.md`. This is the build blueprint: module boundaries, the schemas, and the
contracts between the pieces.

---

## 1. Shape of the system

Single Python process. Three entry points, one core.

```
                    ┌──────────── discovery (LLM in the loop) ───────────┐
  goal + target ──▶ │  agent.loop:  observe → decide → act → repeat      │
                    │     observe = a11y snapshot + screenshot           │
                    │     decide  = LLM tool call, constrained vocabulary│
                    │     act     = Surface primitive                    │
                    └───────────────────────┬───────────────────────────┘
                                            │ success
                                            ▼
                                  artifact.synthesize
                                            │
                                            ▼
                            artifacts/<name>@<version>.json
                                            │
        ┌───────────────────────────────────┼───────────────────────────────┐
        ▼                                   ▼                               ▼
  CLI: replay run             API: POST /capabilities/{n}:invoke      variant override
        └───────────────────────────────────┬───────────────────────────────┘
                                            ▼
                        ┌──── replay (no LLM in the decision loop) ────┐
                        │  resolve params → for each step:             │
                        │    locator ladder → act → wait → checkpoint  │
                        │    classify anything unexpected              │
                        └───────────────────┬─────────────────────────┘
                                            ▼
                              ReplayResult  +  evidence/
                                            │
                              (if stuck) ───┴──▶ escalation → operator console
```

Both discovery and replay drive the **same** `Surface` abstraction and write the **same** evidence
format. That symmetry is deliberate: it is what makes the artifact a faithful recording of a real run
rather than a hand-written script.

## 2. Module layout

```
replay/
  cli.py              discover | run | capabilities | serve | validate
  api.py              FastAPI: capability catalog + operator console

  surface/
    base.py           Surface protocol — THE seam between perceive/act and the recorded flow
    web.py            Playwright implementation (frame-aware)
    snapshot.py       a11y tree normalisation, stable node identity, frame paths
    __future__.py     desktop surface — designed, documented, not built

  agent/
    loop.py           observe → decide → act, stopping conditions
    vocabulary.py     the constrained action set the LLM may emit
    llm.py            LLMClient protocol; OpenAI impl; MockLLM for offline tests
    prompt.py         system prompt, observation rendering, few-shot

  artifact/
    schema.py         Pydantic models (below) — the load-bearing file
    synthesize.py     successful run → artifact, decoupled from transcript
    store.py          load/save/list, semver, JSON Schema export
    overrides.py      per-tenant/variant override layer

  replay/
    executor.py       deterministic step execution
    locators.py       the ladder: resolution order, tier reporting
    conditions.py     checkpoint / outcome condition evaluation
    outcomes.py       three-way classification
    result.py         ReplayResult contract

  policy/
    allowlist.py      domains, routes, action types
    risk.py           safe / risky / irreversible classification + handling
    redaction.py      sensitive-value filter for artifacts, logs, screenshots

  escalation/
    detect.py         stuck detection
    control.py        control token state machine
    console.py        operator surface (bare, real mechanism)
    trace.py          capture what the human did

  evidence/
    recorder.py       JSONL log, screenshots, a11y dumps, DOM on failure

targets/
  meridian/           hostile legacy app (Flask)
  meridian_variant/   re-skinned variant — cross-tenant demo

artifacts/            saved capabilities (JSON, committed)
evidence/             run outputs (committed examples)
tests/
```

## 3. The Surface seam

This is the answer to requirement 3.7. Everything above `Surface` is surface-agnostic; everything
below is specific to a browser, a desktop app, or a terminal.

```python
class Surface(Protocol):
    def observe(self) -> Observation: ...          # a11y tree + screenshot + url/window id
    def act(self, action: Action) -> ActionResult: ...
    def resolve(self, target: TargetSpec) -> Handle | None: ...  # runs the ladder
    def evaluate(self, condition: Condition) -> bool: ...
    def release_control(self) -> None: ...         # for handoff
    def reacquire_control(self) -> None: ...
```

An `Observation` is **not** HTML. It is a normalised accessibility tree — `(role, name, value,
state, frame_path, bounds)` — plus a screenshot. A legacy web app, a modern SPA, and a Win32
desktop app can all produce this. HTML/DOM is available to the web implementation only, and only
the ladder's lowest textual tier is allowed to touch it.

## 4. Artifact schema

The focal point of the evaluation. Pydantic v2, JSON on disk, JSON Schema exported for agent
consumers.

```
CapabilityArtifact
  schema_version   "1.0"            # schema evolution
  name             "open_subaccount"
  version          "1.2.0"          # semver of THIS capability
  title, description                # for a human reviewer AND a calling agent
  app              AppRef { product, product_version, surface_kind, entry_url_pattern }
  inputs           [ParamSpec]
  outputs          [OutputSpec]
  steps            [Step]
  outcomes         [BusinessOutcome]   # declared, expected, non-crash results
  policy           PolicyBlock
  provenance       Provenance
  reliability      Reliability

ParamSpec    name, type, required, description, sensitive: bool, example, constraints
OutputSpec   name, type, description, source: {step_id, extraction}
Provenance   recorded_at, model, run_id, transcript_ref, recorded_by, app_version_seen
Reliability  replays, successes, last_verified_at, approval: draft|approved
PolicyBlock  max_risk, requires_approval, allowlist_ref
```

### Step

```
Step
  id           "s7"
  intent       "Submit the new sub-account form"   # natural language, for reviewers
  action       click | type | select | navigate | press | read | wait | assert | accept_dialog
  target       TargetSpec | null
  value        Literal | ParamRef        # ParamRef, never a raw sensitive value
  waits        [WaitSpec]
  checkpoint   Condition | null          # assert we actually got where we expected
  on_error     [RecoveryRule]
  risk         safe | risky | irreversible
```

### TargetSpec — the locator ladder

The brief asks for "how each target element/control is identified, **with your reasoning about
robustness**". So the rationale is a schema field, not a code comment.

```
TargetSpec
  description  "Member ID input"
  rationale    "No test IDs, no ARIA. Accessible name derives from the adjacent
                table cell, so role+name is viable but weak; the label-adjacency
                tier is the real workhorse on this app."
  frame_path   ["main", "content"]
  strategies   ordered list, tried in order at replay:
    1  role_name        role=textbox, name="Member ID"     portable, works on desktop
    2  aria_path        stable path through the a11y tree
    3  label_adjacent   label="Member ID", relation=right  legacy table layouts
    4  anchored_text    anchor="Account Details", nth=0
    5  css / xpath      brittle, recorded as evidence of what we saw
    6  coordinates      viewport-relative ratio, last resort
```

Replay records **which tier resolved**. A capability that has silently degraded from tier 1 to
tier 5 is drifting, and that is a reportable signal, not a hidden fact.

### Conditions

One small declarative language, shared by checkpoints, outcome detectors, and waits — so
`assert`, "did we succeed", and "is this a known business result" are the same machinery.

```
Condition = text_present | text_absent | role_name_visible | url_matches
          | element_state | frame_present | all_of | any_of | not
```

## 5. Result contract and error taxonomy

The brief calls conflating business outcomes with failures "the most common design mistake here".
The three-way split is therefore structural, not a convention.

```
ReplayResult
  status        success | business_outcome | failed
  capability    name@version
  run_id
  outputs       {...}                        # when success
  outcome       {code, message, detected_at_step}   # when business_outcome
  failure       {step_id, expected, observed, class, evidence_refs}  # when failed
  steps_executed
  locator_tiers {step_id: tier}              # drift signal
  escalation    {request_id, resolved_by, resumed_at} | null
  duration_ms
```

| Class | Meaning | Handling | Example |
|---|---|---|---|
| **Business outcome** | A legitimate answer the caller needs | Return as `status=business_outcome` with a declared code | `MEMBER_NOT_FOUND`, `VALIDATION_REJECTED`, `PERMISSION_DENIED` |
| **Recoverable** | Known transient or interstitial | Apply the step's `on_error` rule, retry bounded, continue, record it | Unexpected dialog, slow load, stale frame |
| **Hard failure** | Cannot proceed safely | Stop, capture rich evidence, return `status=failed` with expected-vs-observed | Session expired, app 500, checkpoint unmet after recovery |

Declared outcomes live in the artifact, so a calling agent knows the full result space *before*
invoking — that is what makes it a capability contract rather than a script.

## 6. Safety model

**Allowlist** — YAML, per environment; enforced in `Surface.act`, not at the call site, so nothing
can route around it.

```yaml
domains:  ["localhost:8080", "localhost:8081"]
routes:   ["/member/*", "/account/*", "/search"]
actions:  [click, type, select, navigate, press, read, wait, assert]
denied:   ["/admin/*", "/transfer/*"]
```

**Risk classes**

| Class | Definition | Handling |
|---|---|---|
| safe | Read-only or trivially reversible: read, navigate in-allowlist, type into a non-submitting field | Proceed |
| risky | Mutates state but is bounded: submit a form, create a record | Proceed only if the artifact is `approved` **or** `--allow-risky`; always logged |
| irreversible | Money movement, deletion, external notification | **Blocked by default.** Requires escalation to a human, who takes control and performs it |

Blocking irreversible actions by default is the conservative reading, and it composes with
escalation rather than duplicating it — the safety valve and the human-in-the-loop path are the
same mechanism.

**Redaction** — `ParamSpec.sensitive: true` means: the value is passed at invoke time and held in
memory only; the artifact stores a `ParamRef`; the log filter masks it; screenshots mask the
element's bounds before writing. Enforced by test, not by discipline.

## 7. Escalation and control transfer

```
                  ┌──────────────┐
                  │  AUTOMATION  │◀────────── resume (checkpoint re-asserted)
                  └──────┬───────┘                        ▲
        stuck detected   │                                │
                         ▼                                │
                  ┌──────────────┐                 ┌──────┴───────┐
                  │  ESCALATING  │────────────────▶│   OPERATOR   │
                  └──────────────┘  control token  └──────────────┘
                   emit InterventionRequest          human acts in the
                   release control                   SAME BrowserContext
                                                     actions → HumanActionTrace
```

**Stuck detection:** max steps exhausted, no state change across N actions, an unclassifiable
condition, a blocked irreversible step, or an explicit `escalate` action from the LLM.

**`InterventionRequest`** carries enough to act on: capability and goal, current step id and intent,
current screenshot, current URL, why it stopped, and the allowlist state.

**Control token** — a single explicit holder, `automation | operator`, with every transition logged.
The browser is headed and the context is never torn down, so "the same live session" is a
structural property, not a claim. On resume, the step's checkpoint is re-asserted before continuing
— we verify where the human actually left us rather than assuming.

## 8. Cross-tenant reuse

Base artifact plus a thin override layer. No re-recording per tenant.

```
artifacts/open_subaccount@1.2.0.json          # base, recorded against MERIDIAN CORE
overrides/northgate_fcu/open_subaccount.json  # variant deltas only
```

An override may replace `app.entry_url_pattern`, a `TargetSpec`, or a `Condition` — nothing else.
Steps, inputs, and outputs are fixed by the base, so a variant cannot silently change the
capability's contract. Drift is detected by watching locator-tier downgrades across replays.

## 9. Interfaces

```bash
replay discover --goal "look up member 12345 and read their savings balance" \
                --target http://localhost:8080 --out artifacts/
replay run lookup_balance --member-id 12345
replay run lookup_balance --member-id 99999          # → MEMBER_NOT_FOUND
replay run open_subaccount --member-id 12345 --tenant northgate_fcu
replay capabilities list | show <name>
replay serve                                          # catalog API + operator console
```

```
GET  /capabilities                      typed catalog, JSON Schema per capability
GET  /capabilities/{name}
POST /capabilities/{name}:invoke        typed args → ReplayResult
GET  /escalations                       pending intervention requests
POST /escalations/{id}/resume
```

## 10. Evidence layout

```
evidence/
  discovery-<run_id>/
    run.jsonl          structured decision log — what, why, which tier
    transcript.jsonl   LLM messages, redacted
    steps/step-NN.png
    a11y/step-NN.json
    result.json
  replay-<run_id>/     same shape, no transcript
  replay-error-<run_id>/
    dom/step-NN.html   richer signal, failure only
    failure.json       expected vs observed
```

## 11. Testing

| Layer | Approach |
|---|---|
| Artifact schema | Round-trip, semver, JSON Schema validity, rejection of malformed artifacts |
| Locator ladder | Unit tests per tier against fixture a11y trees, including degraded-tier reporting |
| Error taxonomy | One test per injection flag asserting the exact classification — the highest-value tests in the repo |
| Redaction | Assert no sensitive value appears in any artifact, log, or screenshot |
| Policy | Allowlist violations and irreversible actions are refused |
| Replay determinism | Same artifact + same params, N runs, identical outputs |
| Agent loop | `MockLLM` with a scripted tool-call sequence — full offline runnability |

The one real LLM run is captured as a committed fixture the moment it succeeds, so nothing in CI
depends on a live model.
