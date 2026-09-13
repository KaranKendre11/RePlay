# Design write-up

## 1. Architecture

One Python process: a CLI, plus a thin HTTP surface for the two things that need one — the
capability catalog an agent calls, the operator console a human uses. No queues, no database, no
services; artifacts are JSON files on disk, and every seam that would have to become a service is
already an interface.

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

Two decisions hold the `surface/` seam, both argued where they are enforced (`surface/base.py`,
`surface/inventory.py`). **Observations are accessibility trees, never markup** — roles, names and
values are what a browser, a screen reader and a Win32 window can all produce, and a test asserts
the observation carries no tags. **The model never writes a selector** — the surface enumerates
candidates, each with a ranked locator ladder, and the model picks an index: it chooses *which*
control, the surface decides *how to name it*.

## 2. Artifact schema

The artifact is a **contract, not a script**. `inputs`, `outputs` and `outcomes` are declared up
front, so a caller knows what a capability needs, returns and can legitimately result in *before*
invoking it. A step list alone would be a macro.

**Robustness reasoning is a required field.** `TargetSpec.rationale` has a minimum length and lives
in the JSON: the artifact is the reviewable unit, and a reviewer cannot read the recorder's code.

**The ladder is ranked, validated as ordered, and ranked per target.** Recording a raw selector
ahead of role+name is structurally impossible. On the target app, tier 1 resolves every button and
**not one text input** — nothing associates a visible label with its control. A flat "use
role+name" policy and a flat "use XPath" policy are both wrong there.

**Business outcomes are declared, not inferred.** One happy-path run cannot discover the failure
space, so synthesis does not invent one: outcomes are attached at review and the artifact ships
`draft`.

Most validators exist to *reject* — no checkpoint anywhere, no checkpoint on a step above `safe`
risk, an output sourced from a click, a policy permitting less risk than the steps take. Synthesis
reads only the run result, never the transcript, which the loop makes possible by recording durable
targets rather than model prose; a test holds the synthesised artifact against a hand-authored one,
tier for tier.

## 3. Determinism & error handling

Replay constructs no model — a test monkeypatches `openai.OpenAI` to explode and deletes the key.
Targets resolve through the recorded ladder with a bounded poll, waits are declared rather than
slept, and every checkpoint is asserted: a click that raised no error has demonstrated nothing, and
"nothing crashed, so it worked" is the failure mode that makes UI automation untrustworthy. Three
consecutive runs produce identical outputs *and identical tiers*.

The result contract splits three ways, because conflating an expected answer with a crash is the
mistake this problem invites:

| status | Meaning |
|---|---|
| `success` | Use `outputs`. |
| `business_outcome` | A declared, expected answer. Branch on the code. Nothing went wrong. |
| `failed` | Broken: step, class, expected, observed, evidence. |

Recoverable conditions are deliberately *not* a status — a dismissed interstitial is not something
the caller asked about — but they are recorded per step, so "recovered silently" and "never
happened" do not look the same afterwards.

Outcomes are checked after **every** step: "no such member" appears immediately after the search
click, and waiting for the end of the flow runs four more steps against a screen that is already
answering. Session loss and application errors are classified **above** the running step — a
timeout reported as `CHECKPOINT_UNMET` sends someone hunting a drifted locator when we were simply
logged out, which needs a re-login or a human, never a retry. All seven injectable conditions are
asserted against the classification table the target app itself publishes, so app and engine cannot
drift apart silently.

**UI drift**, secondarily: synthesis records the tier that actually resolved each control and
replay compares. Resolving below tier 1 is uninteresting on a legacy app — most controls never
resolved there. Resolving *worse than recorded* is the signal: the capability is one vendor release
from failing, and passes every test until it does.

## 4. Heterogeneity & multi-tenant

**Surface abstraction.** A desktop app means writing one more `Surface`, touching neither schema
nor engine: steps address controls by role, name and layout relationship, all of which a desktop
accessibility API supplies. The two web-specific pieces — `SelectorLocator` and `html_of()` — are
opt-in: a surface offering neither is refused nothing, while one missing anything *required* is
rejected when the engine is built, not halfway through a replay. A test replays a capability end to
end through a surface that is nothing but text on a screen. The honest limit is coordinates:
recorded as tier 6, never exercised, and the tier that would matter most on a screenshot-only
surface.

**Nothing in the engine knows which product it is driving.** Session-expiry text, reachable
outcomes, recoverable interstitials — all declared per product and carried by the artifact, because
every vendor spells them differently. A product declaring nothing gets no reclassification: honest
degradation rather than confident mislabelling.

**Multi-tenant.** A capability is recorded once and *specialised* per tenant by overrides that may
change the entry point, a target, a checkpoint or an outcome detector — never the steps, inputs or
outputs, which are the contract (`artifact/overrides.py`). `apply_override` re-derives that
contract afterwards and refuses anything that moved: a tenant needing a different flow needs a
different capability. A checkpoint may be *reworded*, not weakened, or a tenant file could swap the
proof a sub-account opened for a condition that is always true. The specialised artifact carries
the tenant in its ref, so a Northgate replay is evidence about Northgate alone.

Demonstrated, not asserted: the Northgate variant renames the member field, the search button, the
balance column and the form field, and mounts the product under `/tlr` — each breaking a different
ladder tier. The base artifact **fails** against it without overrides; with one override file —
three targets, one checkpoint, a new entry point, no steps — it succeeds at the tiers it was
recorded at.

## 5. Escalation & handoff

**Detecting stuck.** Discovery: `give_up`, a repeated identical decision, a model error. Replay: a
step policy will not take unattended, an expired session, a checkpoint that will not come true, a
vanished control. Deliberately *not* escalated: a malformed argument nobody can fix, a business
outcome — a correct answer — and a spent budget; paging for any of them teaches operators to ignore
the queue.

**Control transfer.** One explicit holder, `automation` or `operator`, every transition logged —
not a lock, not two booleans that can disagree. The browser context is never torn down, so "the
human operates the same session" is structural: same cookies, same server-side session, same page.
Escalation **blocks**; a run that raises a request and carries on has not escalated, it has logged.
On hand-back the step is judged as if we had performed it — outcome first, then checkpoint — after
waiting for the screen, since a person's click carries no navigation promise and control returns
while their submit is still in flight. A stuck discovery resumes the same way but warns that a
human did part of the work: those steps never reach the trace synthesis reads, so the result is not
a replayable capability.

**What the human did** is captured, not self-reported: a capture-phase listener in every frame
records *which control was touched, never what was typed*. The console is bare and does not stream
the screen — the browser is headed and the operator is in front of it.

## 6. Safety

**Default-deny, enforced structurally.** An empty or missing allowlist permits nothing, and no
command opens a surface without a policy file. The allowlist lives inside `Surface.act` and the
risk gate inside the executor, so discovery, replay, recovery and the HTTP catalog are covered
without knowing the guardrails exist — a guardrail checked by its callers is one a future caller
forgets. Discovery matters most, being the one path where a model picks the URL: a refused
destination becomes a line in its action log, so it routes around the boundary rather than
stopping. Deny beats allow, because a deny rule exists precisely when a general rule was too
generous.

**Three risk classes, not a slider.** `safe` proceeds; `risky` needs approval or an opt-in;
`irreversible` is **blocked by default**, and `--allow-risky` does not imply `--allow-irreversible`
— committing money is not "more of" creating a record. Blocking rather than prompting makes the
safety valve and the human-in-the-loop path one mechanism, and refusal happens before a browser
opens.

**Approval is earned, not typed.** Counters are derived from run evidence, never stored — a stored
counter merely *claims* five clean replays happened. `replay approve` needs five recent runs, no
failures, no drift, three full successes and more than one argument set. A business outcome
exercises only a prefix of the flow and counts apart: `lookup_balance` is unapprovable until it has
met a member who does not exist. Approval gates *unattended* use — a reachable operator satisfies
it, or nothing needing approval could ever earn one.

**Redaction in two layers.** Explicit masks cover values we were handed; shape patterns cover what
lands in an observation dump — or a screenshot, which is pixels and is masked too. Short digit
strings are left alone: a member ID is the caller's argument, not a secret. The converse bit us — a
pattern eating `name@version` as an email address redacted every capability name in the evidence,
including the one shown to an operator asked to take over.

**Limits.** The allowlist is host-and-path based, so it cannot tell a legitimate
`POST /member/12345/subaccount` from a malicious one. Risk is classified at record time from
observable signals — a step answering a confirmation dialog is irreversible, and a click on a
control that names itself a commit ("Submit", "Post", "Transfer", "Approve") is `risky` even when
the application commits without asking — and both are heuristics: a submit button labelled "Go" is
`safe`, and only a reviewer catches it. Redaction patterns will miss institution-specific formats.
And nothing defends against a *compromised artifact* — an approved capability is trusted, which is
why the schema works so hard to keep them readable.

## 7. Cuts

**Built, of the stretch goals.** Two deliberately — the agent-facing capability catalog and the
cross-tenant variant — each load-bearing for a claim made above. The draft → approved gate is a
third only by name: `Reliability` was in the schema and `RiskGate` already enforced it, with
nothing but hand-edited JSON able to move it; what was added is a tally derived from evidence the
replays already write, not a new subsystem.

**Not built, deliberately.** A real co-browsing console; multi-tenant infrastructure — the
*artifact* is designed for reuse, the plumbing is not. Declined: a standalone flakiness score, code
generation, and assisted LLM fallback, the last of which would muddy the "no model in the replay
loop" claim determinism rests on. And folding an operator's manual steps into the artifact when
they unstick a discovery: part-discovered, part-demonstrated is a much bigger change, so the run
warns instead.

**Next, in order.** A real desktop surface — the seam is exercised against a text-only stub, not a
live accessibility API, and that last step is the load-bearing bet still taken on trust. A drift
dashboard: the tier comparison currently reaches only a run log, and across hundreds of tenants it
wants to be a ranked list of capabilities sliding down the ladder. Then shared sub-flows, since
both capabilities duplicate a search prefix.

**One thing I got wrong, and the thing I got wrong fixing it.** A real `gpt-5` run offered
`"4,211.03"` — the balance it had just read — as proof of success
(`evidence/discovery-20260906T091017Z`): true for member 12345, false for everyone else. The loop
catches what the model cannot, because it knows which values were parameters and which were
outputs. The first fix then traded a checkpoint that was too specific for one too generic — `"Open
Sub-Account"` is a link on *every* member's page — because it treated parameters and outputs as one
"volatile" set. They are opposites: an output is unknown until the run produces it, while a
**parameter** is supplied before the browser opens, so a checkpoint may name it, and it is the only
assertion on that screen that says whose it is. Checkpoint text may now be a `ParamRef` resolved at
replay time; outputs are still refused outright.
