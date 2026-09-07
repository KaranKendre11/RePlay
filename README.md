# RePlay

Record-once, replay-many computer-use automation for legacy back-office applications.

An LLM discovers how to complete a goal by driving a real UI. The successful run is distilled
into a typed, versioned **capability artifact**. That artifact then replays deterministically —
no model in the decision loop, ~150 ms per run — and is callable by an AI agent by name with
typed arguments.

```
goal ──▶ LLM drives the UI ──▶ capability artifact ──▶ deterministic replay ──▶ typed result
         (once, ~30s, cents)    (typed, reviewable)     (every time, ~150ms, free)
```

- **Design write-up:** [`REPORT.md`](REPORT.md)
- **Evidence from real runs:** [`evidence/`](evidence/)
- **Saved capabilities:** [`artifacts/`](artifacts/)

---

## Setup

Requires [`uv`](https://docs.astral.sh/uv/). Nothing else — it installs Python itself.

```bash
uv sync                              # Python 3.12 + dependencies
uv run playwright install chromium   # the browser it drives
```

For the discovery step only, you need an OpenAI key:

```bash
cp .env.example .env
# then set OPENAI_API_KEY=sk-...
```

**Replay never needs a key.** Everything except discovery — replay, the error taxonomy, the
catalog, escalation, cross-tenant — runs offline and free.

---

## Demo path

Two terminals. This is the whole system in about a minute.

### Terminal 1 — the target application

`MERIDIAN CORE` is a deliberately hostile stand-in for a legacy credit-union teller console:
a real `<frameset>`, nested table layout, no test IDs, form fields named `f7`/`f12`, labels tied
to inputs only by cell adjacency, a native `confirm()` on submit, and server-side session expiry.

```bash
uv run python -m targets.meridian --port 8080                       # base deployment
uv run python -m targets.meridian --variant northgate --port 8081   # a second institution
```

Browse it at <http://127.0.0.1:8080>. `View Source` shows why it is hard.

### Terminal 2 — discovery, then replay

**1. Discover.** An LLM drives the live UI until the goal is met. Costs a few cents and ~30
seconds. Add `--headed` to watch it happen.

```bash
uv run replay discover \
  --goal "Look up member 12345 and read their current savings balance." \
  --target http://127.0.0.1:8080/ \
  --save-as lookup_balance \
  --profile targets/meridian/review.toml
```

`--profile` supplies the product knowledge one happy-path run cannot discover: the business
outcomes the capability can legitimately reach, the interstitials worth recovering from, and the
screen text that means the session died rather than the step being wrong. Omit it and the
capability still synthesises — it just declares no outcomes, and says so.

**2. Replay.** The same job, with no model anywhere in the path.

```bash
uv run replay run lookup_balance -p member_id=12345
```

```
success
outputs:   {"current_savings_balance": "4,211.03"}
tiers:     {"s2": 3, "s3": 1, "s4": 4}
duration:  169 ms
```

### Everything else, without a key

```bash
# A legitimate business answer — not a crash.
uv run replay run lookup_balance -p member_id=99999

# A hard failure, with expected-vs-observed and a DOM snapshot.
uv run replay run lookup_balance -p member_id=12345 \
    -t "http://127.0.0.1:8080/?inject=error500"

# A recoverable interstitial: dismissed, recorded, run completes.
uv run replay run lookup_balance -p member_id=12345 \
    -t "http://127.0.0.1:8080/?inject=dialog"

# The same artifact against a different institution, via overrides only.
uv run replay run lookup_balance -p member_id=12345 --tenant northgate

# Blocked: this capability opens an account, so it needs a human.
uv run replay run open_subaccount \
    -p member_id=12345 -p product_code=S02 -p opening_deposit=50.00

# ...unless one is available. Opens a headed browser and the operator console.
uv run replay run open_subaccount \
    -p member_id=12345 -p product_code=S02 -p opening_deposit=50.00 \
    --allow-risky --escalate
```

### The agent-facing surface

```bash
uv run replay capabilities          # what an agent can call
uv run replay serve                 # catalog + operator console
```

```bash
curl -s localhost:8000/capabilities | jq '.[].ref'
curl -s -XPOST localhost:8000/capabilities/lookup_balance:invoke \
     -H 'content-type: application/json' \
     -d '{"arguments": {"member_id": "12345"}}'
```

Catalog at `/capabilities`, operator console at `/operator`. Served invocations write
their evidence to `runs/` (gitignored), not to the curated `evidence/` set — pass
`--evidence-dir` to put it somewhere else.

---

## Failure injection

The target app can produce every runtime condition the brief names, on demand. Append
`?inject=<mode>` to the entry URL; it propagates through the flow.

| Mode | Simulates | Replay reports |
|---|---|---|
| `not_found` | member does not exist | business outcome `MEMBER_NOT_FOUND` |
| `validation` | server rejects a field | business outcome `VALIDATION_REJECTED` |
| `denied` | insufficient authority | business outcome `PERMISSION_DENIED` |
| `dialog` | unexpected interstitial | recovered, recorded, run completes |
| `slow` | 3s stall | absorbed by the declared waits |
| `timeout` | session expired mid-flow | hard failure `SESSION_LOST` |
| `error500` | application error | hard failure `APPLICATION_ERROR` |

`targets/meridian/inject.py` declares the expected classification for each, and a test asserts
replay actually agrees — so the app and the engine cannot drift apart silently.

---

## Tests

```bash
uv run pytest        # 338 tests, ~100s, no API key needed
uv run ruff check .
```

The suite boots the real app and drives a real browser. Discovery is covered offline by a
`MockLLM` replaying scripted tool calls, so the one paid run is not the only thing that ever
exercises the loop.

---

## What is in here

```
src/replay/
  surface/      perceive and act — accessibility tree, never markup
  agent/        the discovery loop; the only place a model sits
  artifact/     the capability schema, store, and per-tenant overrides
  synthesis.py  run trace → capability
  engine/       deterministic replay and the result contract
  policy/       allowlist, risk classes, redaction
  escalation/   intervention requests, control transfer, operator console
  api.py        the capability catalog an agent calls
targets/meridian/   the hostile legacy app, and a second tenant's variant
  review.toml       product knowledge a happy-path run cannot discover
artifacts/          saved capabilities (JSON, reviewable, diffable)
overrides/          per-tenant deltas
evidence/           committed runs — four real gpt-5 discoveries, seven replays
policy.toml         the guardrails
```

## Evidence

Committed, from real runs, not reconstructed.

| Directory | What it shows |
|---|---|
| `discovery-20260906T085541Z` | **Real gpt-5 run** — the read flow, 4 actions |
| `discovery-20260906T091017Z` | **Real gpt-5 run** — the model proposed a checkpoint that only held for one member; the loop caught it |
| `discovery-20260906T211130Z` | **Real gpt-5 run** — the write flow, 9 actions, irreversible |
| `discovery-20260907T065628Z` | **Real gpt-5 run** — same failure, opposite outcome: the checkpoint was substituted, not refused |
| `replay-01-success` | Deterministic replay with typed outputs |
| `replay-02-business-outcome` | `MEMBER_NOT_FOUND` returned as a result |
| `replay-03-hard-failure` | Application error, with DOM snapshot |
| `replay-04-recovered` | Interstitial dismissed and recorded |
| `replay-05-cross-tenant` | Same artifact, second institution |
| `replay-06-policy-refused` | Irreversible capability blocked before touching the app |
| `replay-07-escalation` | A human takes the live session, finishes the step, hands it back |

Each contains a structured `run.jsonl`, per-step screenshots and accessibility snapshots, a
`result.json`, and a `capability.json` copy of the artifact that run produced or replayed —
`artifacts/` stays the source of truth. Discovery runs also keep the model transcript — referenced
by the artifact, never embedded in it.

## Notes

- No secret reaches the repo. `.env` is gitignored, evidence is redaction-filtered on write, and
  a test walks `git ls-files` to enforce it.
- The target app is local. Nothing here touches a real institution, and no real credentials or
  PII exist anywhere in it.
- `PRD.md` and `project-arch.md` are the plan this was built from, kept for provenance.
