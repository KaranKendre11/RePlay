# Design write-up

## 1. Architecture

One Python process: a CLI, plus a thin HTTP surface for the two things that genuinely need one —
the capability catalog an agent calls and the operator console a human uses. No queues, no
database, no services; artifacts are JSON files on disk. Single-process and single-operator by
choice, and every seam that would have to become a service — intervention queue, artifact store,
catalog — is already an interface.

```
goal ─▶ agent/        observe → decide → act; the only place a model sits
          ▼
     synthesis.py     run trace → capability   (never reads the transcript)
          ▼
     artifacts/*.json
          ▼
     engine/          deterministic replay, no model, ~150 ms
          │
   policy/ ──────── escalation/
   allowlist, risk,  intervention, control transfer,
   redaction         operator console
```

Two decisions hold the `surface/` seam, and both are argued where they are enforced
(`surface/base.py`, `surface/inventory.py`). **Observations are accessibility trees, never markup**
— a tree of roles, names and values is something a browser, a screen reader and a Win32 window can
all produce, and a test asserts the rendered observation carries no tags. **The model never writes
a selector** — the surface enumerates candidates with a ranked locator ladder each and the model
picks an index, so it chooses *which* control while the surface decides *how to name it*.

## 2. Artifact schema

The artifact is a **contract, not a script**. `inputs`, `outputs` and `outcomes` are declared up
front, so a caller knows what a capability needs, returns, and can legitimately result in *before*
invoking it. A step list alone would make this a macro.

Three points earned their place:

**Robustness reasoning is a required field.** `TargetSpec.rationale` has a minimum length and
lives in the JSON. The artifact is the reviewable unit and a reviewer cannot read the recorder's
code.

**The ladder is ranked, validated as ordered, and ranked per target.** Recording a raw selector
ahead of role+name is structurally impossible. Measured on the target app: tier 1 resolves every
button and **not one text input**, because nothing associates the visible label with the control.
A flat "use role+name" policy and a flat "use XPath" policy are both wrong there.

**Business outcomes are declared, not inferred.** One happy-path run cannot discover the failure
space, so synthesis does not invent one; outcomes are attached at review and the artifact ships
`draft`. Nothing synthesised is ever `approved` — a capability that has replayed zero times has
earned nothing.

Most validators exist to *reject*, each matching a failure we would otherwise meet in production:
no checkpoint anywhere, an output sourced from a click, a sensitive parameter carrying a persisted
example, a policy permitting less risk than the steps take.

Synthesis reads only the run result, never the transcript — possible because the loop recorded
durable targets rather than model prose. A test compares the synthesised artifact against a
hand-authored one; they agree tier for tier.

## 3. Determinism & error handling

Replay constructs no model — a test monkeypatches `openai.OpenAI` to explode and deletes the key.
Targets resolve through the recorded ladder with a bounded poll; waits are declared, not slept;
every checkpoint is asserted. A click that raises no error has demonstrated nothing, and a replay
reporting success whenever nothing crashed is the failure mode that makes UI automation
untrustworthy. Three consecutive runs produce identical outputs *and identical tiers*.

The result contract splits three ways, because conflating an expected answer with a crash is the
mistake this problem invites:

| status | Meaning |
|---|---|
| `success` | Use `outputs`. |
| `business_outcome` | A declared, expected answer. Branch on the code. Nothing went wrong. |
| `failed` | Broken: step, class, expected, observed, evidence. |

Recoverable conditions are deliberately *not* a status — a dismissed interstitial is not something
the caller asked about. They are still recorded per step, because "recovered silently" and "never
happened" must not look the same afterwards.

Outcomes are checked after **every** step: "no such member" appears immediately after the search
click, and waiting until the flow ends means running four more steps against a screen that is
already answering.

Session loss and application errors are classified **above** whatever step was running. Reporting
a timeout as `CHECKPOINT_UNMET` sends someone hunting for a drifted locator when we were simply
logged out — which needs a re-login or a human, never a retry.

All seven injectable conditions are asserted against the classification table the target app
itself publishes, so app and engine cannot drift apart silently.

**UI drift**, secondarily: synthesis records the tier that actually resolved each control at record
time, and replay compares. Resolving below tier 1 is uninteresting on a legacy app — most controls
never resolved there. Resolving *worse than recorded* is the signal, because the capability is then
one vendor release from failing and passes every test until it does.

## 4. Heterogeneity & multi-tenant

**Surface abstraction.** Extending to a desktop app means writing one more `Surface`, touching
neither the schema nor the engine: steps address controls by role, name and layout relationships,
all of which a desktop accessibility API supplies. The two web-specific pieces — `SelectorLocator`
and `html_of()` — are the ladder's last resort and a failure-only dump. The honest limit is
coordinates: recorded as tier 6, never exercised, and the tier that would matter most on a
screenshot-only surface.

**Nothing in the engine knows which product it is driving.** The text meaning a session expired,
the outcomes a capability can legitimately reach, the interstitials worth recovering from — all
declared per product and carried by the artifact, because every vendor spells them differently and
an engine holding one product's error codes silently stops classifying against every other. A
product that declares nothing gets no reclassification: honest degradation rather than confident
mislabelling, and a test strips the declaration to prove the answer changes.

**Multi-tenant.** A capability is recorded once and *specialised* per tenant by overrides that may
change the entry point, a target, a checkpoint or an outcome detector — never the steps, inputs or
outputs, which are the contract (`artifact/overrides.py`). `apply_override` re-derives that
contract afterwards and refuses anything that moved, so a tenant needing a different flow needs a
different capability.

Demonstrated, not asserted: the Northgate variant renames the member field, the search button, the
balance column and the form field, and mounts the product under `/tlr` — each breaking a different
ladder tier. The base artifact **fails** against it without overrides; with a 40-line override it
succeeds at the tiers it was recorded at.

## 5. Escalation & handoff

**Detecting stuck.** Discovery: `give_up`, a repeated identical decision, or a model error. Replay:
a step policy will not take unattended, an expired session, a checkpoint that will not come true, a
vanished control. All of them reach a person. Deliberately *not* escalated: a malformed argument
nobody can fix, a business outcome, which is a correct answer, and a spent step or time budget —
paging for any of them teaches operators to ignore the queue.

**Control transfer.** One explicit holder, `automation` or `operator`, every transition logged —
not a lock, not two booleans that can disagree. The browser context is never torn down, so "the
human operates the same session" is structural: same cookies, same server-side session, same page.
Escalation **blocks**; a run that raises a request and carries on has not escalated, it has logged.
On hand-back the step is judged as if we had performed it — outcome first, then the checkpoint,
re-asserted rather than assumed — after waiting for the screen, because a person's click carries no
navigation promise to wait on and control returns while their submit is still in flight. A stuck
discovery resumes the same way, then warns that a human did part of the work: those steps are not
in the trace synthesis reads, so the result is not a replayable capability.

**What the human did** is captured, not self-reported: a capture-phase listener in every frame
reports clicks, changes and Enter presses, recording *which control was touched, never what was
typed*. The console is deliberately bare and does not stream the screen — the browser is headed and
the operator is sitting in front of it.

## 6. Safety

**Default-deny, enforced structurally.** A missing or empty allowlist permits nothing, and every
command that opens a surface — `discover`, `run`, `serve` — refuses to start without a policy file
rather than falling back to permissive. The allowlist lives inside `Surface.act` and the risk gate
inside the executor, so discovery, replay, recovery rules and the HTTP catalog are all covered
without knowing the guardrails exist — a guardrail checked by its callers is one a future caller
forgets. Discovery matters most there, being the one path where a model rather than a recording
chooses the URL: a refused entry point stops the run, while a destination the model chose becomes a
line in its action log, so it sees the boundary and routes around it. Deny beats allow, because a deny rule exists precisely when
a general rule was too generous. Each guardrail below is argued at length under `policy/`.

**Three risk classes, not a slider.** `safe` proceeds; `risky` needs approval or an explicit opt-in;
`irreversible` is **blocked by default**, and `--allow-risky` does not imply `--allow-irreversible`
— committing money is not "more of" creating a record. Blocking rather than prompting makes the
safety valve and the human-in-the-loop path the same mechanism. Refusal happens before a browser
opens, so a blocked run leaves the application untouched.

**Redaction in two layers.** Explicit masks cover values we were handed; shape-based patterns cover
what appears on screen and lands in an observation dump — or in a screenshot, which is pixels and is
masked too. Short digit strings are deliberately left alone: a member ID is the caller's argument,
not a secret.

**Limits.** The allowlist is host-and-path based, so it cannot distinguish a legitimate
`POST /member/12345/subaccount` from a malicious one. Risk is classified at record time from
observable signals — a step answering a confirmation dialog is treated as irreversible — a good
heuristic, not a guarantee: an application that commits without asking would be classified `risky`.
The redaction patterns are narrow and will miss institution-specific formats. And nothing defends
against a *compromised artifact*: an approved capability is trusted, so artifact review is a real
control, which is why the schema works so hard to keep them readable.

## 7. Cuts

**Not built, deliberately.** A real co-browsing console, and multi-tenant infrastructure — no
registry, no per-tenant deployment; the *artifact* is designed for reuse, the plumbing is not.
Stretch goals declined: multi-run flakiness scoring, code generation, and assisted LLM fallback,
the last of which would muddy the "no model in the replay loop" claim determinism rests on.
`LabelAdjacentLocator` returns `None` for above/below rather than guessing at column arithmetic.
And folding an operator's manual steps into the artifact when they unstick a discovery — a
capability part-discovered and part-demonstrated is the interesting version, and a much bigger
change; the run warns instead.

**Next, in order.**

1. **A desktop surface**, to prove the seam rather than argue it. The accessibility-tree bet is the
   load-bearing claim here, and the one thing reasoned rather than demonstrated.
2. **Approval workflow with reliability scoring.** The schema carries `draft`/`approved` and a
   replay counter; nothing moves a capability between them. Auto-promotion after N clean replays
   makes the guardrail self-maintaining.
3. **A drift dashboard.** The tier comparison is the right signal and currently only reaches a run
   log; across hundreds of tenants it wants to be a ranked list of capabilities sliding down the
   ladder.
4. **Shared sub-flows.** Both capabilities duplicate a search prefix.

**One thing I got wrong.** A real `gpt-5` run offered `"4,211.03"` — the balance it had just read —
as proof of success (`evidence/discovery-20260906T091017Z`): true for member 12345, false for
everyone else, so a capability asserting it would pass once and fail forever. The loop catches what
the model cannot, because it knows which values were parameters and which were outputs; synthesis
substitutes stable screen text, records why, and refuses outright when there is none.
