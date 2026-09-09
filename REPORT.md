# Design write-up

## 1. Architecture

One Python process: a CLI, plus a thin HTTP surface for the two things that need one — the
capability catalog an agent calls and the operator console a human uses. No queues, no database,
no services; artifacts are JSON files on disk. Single-process and single-operator by choice, and
every seam that would have to become a service is already an interface.

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
and `html_of()` — are opt-in protocol extensions, not attributes the engine reaches for and hopes
to find: a surface offering neither is refused nothing and the log says what it could not do, while
one missing anything *required* — screen text included, since the taxonomy runs on it — is rejected
when the engine is built, not halfway through a replay. A test replays a capability end to end
through a surface that is nothing but text on a screen. The honest limit is coordinates: recorded
as tier 6, never exercised, and the tier that would matter most on a screenshot-only surface.

**Nothing in the engine knows which product it is driving.** The text meaning a session expired,
the outcomes a capability can reach, the interstitials worth recovering from — all declared per
product and carried by the artifact, because every vendor spells them differently and an engine
holding one product's error codes silently stops classifying against every other. A product
declaring nothing gets no reclassification: honest degradation rather than confident mislabelling,
and a test strips the declaration to prove the answer changes.

**Multi-tenant.** A capability is recorded once and *specialised* per tenant by overrides that may
change the entry point, a target, a checkpoint or an outcome detector — never the steps, inputs or
outputs, which are the contract (`artifact/overrides.py`). `apply_override` re-derives that
contract afterwards and refuses anything that moved: a tenant needing a different flow needs a
different capability. A checkpoint may be *reworded*, not weakened — the replacement has to assert
the same kind of thing, or a tenant file could swap the proof a sub-account opened for a condition
that is always true. A named tenant the overrides root has never heard of is refused rather than
quietly run as the base, and the specialised artifact carries the tenant in its ref, so a
Northgate replay is evidence about Northgate and about nothing else.

Demonstrated, not asserted: the Northgate variant renames the member field, the search button, the
balance column and the form field, and mounts the product under `/tlr` — each breaking a different
ladder tier. The base artifact **fails** against it without overrides; with a 40-line override it
succeeds at the tiers it was recorded at.

## 5. Escalation & handoff

**Detecting stuck.** Discovery: `give_up`, a repeated identical decision, a model error. Replay: a
step policy will not take unattended, an expired session, a checkpoint that will not come true, a
vanished control. All reach a person. Deliberately *not* escalated: a malformed argument nobody can
fix, a business outcome, which is a correct answer, and a spent budget — paging for any of them
teaches operators to ignore the queue.

**Control transfer.** One explicit holder, `automation` or `operator`, every transition logged —
not a lock, not two booleans that can disagree. The browser context is never torn down, so "the
human operates the same session" is structural: same cookies, same server-side session, same page.
Escalation **blocks**; a run that raises a request and carries on has not escalated, it has logged.
On hand-back the step is judged as if we had performed it — outcome first, then checkpoint,
re-asserted rather than assumed — after waiting for the screen, since a person's click carries no
navigation promise and control returns while their submit is still in flight. A stuck discovery
resumes the same way, then warns that a human did part of the work: those steps never reach the
trace synthesis reads, so the result is not a replayable capability.

**What the human did** is captured, not self-reported: a capture-phase listener in every frame
records *which control was touched, never what was typed*. The console is bare and does not stream
the screen — the browser is headed and the operator is in front of it.

## 6. Safety

**Default-deny, enforced structurally.** An empty or missing allowlist permits nothing, and every
command opening a surface refuses to start without a policy file. The allowlist lives inside
`Surface.act` and the risk gate inside the executor, so discovery, replay, recovery and the HTTP
catalog are covered without knowing the guardrails exist — a guardrail checked by its callers is
one a future caller forgets. Discovery matters most, being the one path where a model picks the
URL: a refused entry point stops the run, a refused destination becomes a line in the model's
action log, so it routes around the boundary. Deny beats allow, because a deny rule exists
precisely when a general rule was too generous.

**Three risk classes, not a slider.** `safe` proceeds; `risky` needs approval or an opt-in;
`irreversible` is **blocked by default**, and `--allow-risky` does not imply `--allow-irreversible`
— committing money is not "more of" creating a record. Blocking rather than prompting makes the
safety valve and the human-in-the-loop path one mechanism. Refusal happens before a browser opens.

**Approval is earned, not typed.** Counters are derived from run evidence, never stored — a stored
counter merely *claims* five clean replays happened. The threshold lives in `store.approve`, which
takes the evidence rather than a reliability block, so every caller routes through it instead of
only the one an operator happens to type. `replay approve` needs five recent runs, no failures, no
drift, three full successes and more than one argument set. A business outcome
exercises only a prefix of the flow, so it counts apart: `lookup_balance` is unapprovable until it
has met a member who does not exist. Approval gates *unattended* use, so a reachable operator
satisfies it — otherwise a capability needing approval could never earn one, having never been
allowed to run.

**Redaction in two layers.** Explicit masks cover values we were handed; shape patterns cover what
lands in an observation dump — or a screenshot, which is pixels and is masked too. Short digit
strings are left alone: a member ID is the caller's argument, not a secret. The converse bit us —
a pattern that ate `name@version` as an email address redacted every capability name in the
evidence, including the one shown to an operator being asked to take over. Test both directions.

**Limits.** The allowlist is host-and-path based, so it cannot distinguish a legitimate
`POST /member/12345/subaccount` from a malicious one. Risk is classified at record time from
observable signals — a step answering a confirmation dialog is treated as irreversible, and a click
on a control that names itself a commit ("Submit", "Post", "Transfer", "Approve") is `risky` even
when the application asks nothing. Both are heuristics, not guarantees: a submit button labelled
"Go" is classified `safe` and only a reviewer catches it.
The redaction patterns are narrow and will miss institution-specific formats. And nothing defends
against a *compromised artifact*: an approved capability is trusted, so artifact review is a real
control, which is why the schema works so hard to keep them readable.

## 7. Cuts

**Not built, deliberately.** A real co-browsing console; multi-tenant infrastructure — the
*artifact* is designed for reuse, the plumbing is not. Stretch goals declined: flakiness scoring,
code generation, and assisted LLM fallback, the last of which would muddy the "no model in the
replay loop" claim determinism rests on. `LabelAdjacentLocator` returns `None` for above/below
rather than guessing at column arithmetic. And folding an operator's manual steps into the artifact
when they unstick a discovery — part-discovered, part-demonstrated is the interesting version and a
much bigger change; the run warns instead.

**Next, in order.**

1. **A real desktop surface.** The seam is now exercised — the engine replays through a
   text-only surface with no browser under it — but against a stub, not a live accessibility API.
   That last step is the load-bearing bet still taken on trust.
2. **A drift dashboard.** The tier comparison is the right signal and currently only reaches a run
   log; across hundreds of tenants it wants to be a ranked list of capabilities sliding down the
   ladder.
3. **Shared sub-flows.** Both capabilities duplicate a search prefix.

**One thing I got wrong, and the thing I got wrong fixing it.** A real `gpt-5` run offered
`"4,211.03"` — the balance it had just read — as proof of success
(`evidence/discovery-20260906T091017Z`): true for member 12345, false for everyone else, so a
capability asserting it would pass once and fail forever. The loop catches what the model cannot,
because it knows which values were parameters and which were outputs.

The first fix substituted stable screen text — and swapped a checkpoint that was too specific for
one that was too generic. `"Open Sub-Account"` is a link on *every* member's page, so it proved a
member screen was loaded and never *which*, and the balance is read from whatever SAVINGS row is in
the work frame. Both member-independent: any failure leaving the wrong page in `workframe` returned
someone else's balance as `success`. The cause was treating parameters and outputs as one
"volatile" set. They are opposites. An output is unknown until the run produces it; a **parameter**
is supplied by the caller before the browser opens, so a checkpoint may name it — and it is the
only assertion on that screen that says whose it is. A checkpoint text may now be a `ParamRef`
resolved from the caller's arguments at replay time, synthesis prefers a parameter-bearing
checkpoint over screen chrome (pairing the two where it has both), and outputs are still refused
outright.
