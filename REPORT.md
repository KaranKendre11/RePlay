# Design write-up

## 1. Architecture

One Python process. A CLI, plus a thin HTTP surface for the two things that genuinely need one:
the capability catalog an agent calls, and the operator console a human uses. No queues, no
database, no services — artifacts are JSON files on disk.

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

Everything above `surface/` is surface-agnostic. `Surface` is a six-method protocol — `observe`,
`resolve`, `act`, `evaluate`, `release_control`, `reacquire_control` — and the whole heterogeneity
story rests on it.

**Observations are accessibility trees, never markup.** An HTML string is a web fact; a tree of
roles, names and values is something a browser, a screen reader and a Win32 window can all
produce. A test asserts the rendered observation contains no tags. Markup appears in exactly one
place — a DOM dump on failure — because it is useless for deciding what to do and invaluable for
working out afterwards why something broke.

**The model never writes a selector.** The surface enumerates what is on screen and computes a
locator ladder per candidate; the model picks an index. It chooses *which* control, the surface
decides *how to name it*. A model that can emit arbitrary selectors emits what it imagines the
markup looks like, which on a legacy screen is usually fiction.

Trade-off: this is single-process and single-operator, deliberately. Every seam that would need to
become a service — intervention queue, artifact store, catalog — is already an interface.

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

**Surface abstraction.** Extending to a desktop app means writing one more `Surface`. Nothing in
the schema or the engine mentions a browser: steps address controls by role, name and layout
relationships, all of which a desktop accessibility API supplies. The two web-specific pieces —
`SelectorLocator` and `html_of()` — are the ladder's last resort and a failure-only dump. The
honest limit is coordinates: recorded as tier 6, never exercised, and the tier that would matter
most on a screenshot-only surface.

**Multi-tenant.** A capability is recorded once and *specialised* per tenant by a thin override
layer. An override may change the entry point, a target, a checkpoint, an outcome detector. It may
**not** change steps, inputs or outputs — those are the contract, and letting an override alter
what `lookup_balance` does means a caller can no longer rely on what the name means. Enforced by
re-deriving the contract afterwards and refusing anything that moved. A tenant needing a different
flow needs a different capability.

Demonstrated, not asserted: the Northgate variant renames the member field, the search button, the
balance column and the form field, and mounts the product under `/tlr` — each breaking a different
ladder tier. The base artifact **fails** against it without overrides; with a 40-line override it
succeeds at the tiers it was recorded at.

## 5. Escalation & handoff

**Detecting stuck.** Discovery: max steps, a repeated identical decision, or the model calling
`give_up`. Replay: a step policy will not take unattended, an expired session, a checkpoint that
will not come true, a vanished control. Deliberately *not* escalated: a caller's malformed
argument, which nobody can fix, and a business outcome, which is a correct answer. Paging for
either teaches operators to ignore the queue.

**Taking control.** One explicit holder — `automation` or `operator` — every transition logged.
Not a lock, not two booleans that can disagree. The browser context is never torn down, so "the
human operates the same session" is structural: same cookies, same server-side session, same page.
Escalation **blocks**; a run that raises a request and carries on has not escalated, it has logged.

**Handing back.** The operator resumes from the console, automation reacquires, and the checkpoint
is re-asserted rather than assumed — the entire point of a checkpoint is not to trust that we are
where we think we are.

**Recording what they did** is captured, not self-reported: a capture-phase listener in every frame
reports clicks, changes and Enter presses. It records *which control was touched, never what was
typed* — an operator handling a bank escalation is often typing exactly the data this system must
not persist.

The console is bare and does not stream the screen. The browser is headed and the operator is in
front of it; streaming pixels would be a nicer product and a worse demonstration of the mechanism.

## 6. Safety

**Default-deny.** A missing or empty allowlist permits nothing; the CLI refuses to start without a
policy file rather than falling back to permissive.

**Enforcement is not optional.** The allowlist lives inside `Surface.act`, the risk gate inside the
executor, so discovery, replay, recovery rules and the HTTP catalog are covered without knowing the
guardrails exist. A control checked by its callers is one that a future caller forgets. Deny beats
allow, because a deny rule exists precisely when a general rule was too generous.

**Three risk classes, not a slider.** `safe` proceeds; `risky` needs approval or an explicit opt-in;
`irreversible` is **blocked by default**. `--allow-risky` does not imply `--allow-irreversible` —
committing money is not "more of" creating a record. Blocking rather than prompting makes the
safety valve and the human-in-the-loop path the *same* mechanism instead of two that can disagree,
and a confirmation flag would move the decision to whoever wrote the calling code. Refusal happens
before a browser opens, so a blocked run leaves the application untouched.

**Redaction in two layers.** Explicit masks cover values we were handed; shape-based patterns cover
what appears on screen and lands in an observation dump. Screenshots are masked too — a screenshot
is pixels. Short digit strings are deliberately left alone: a member ID is the caller's argument,
not a secret, and a redactor that eats every identifier makes debugging impossible while protecting
nothing.

**Limits.** The allowlist is host-and-path based, so it cannot distinguish a legitimate
`POST /member/12345/subaccount` from a malicious one. Risk is classified at record time from
observable signals — a step answering a confirmation dialog is treated as irreversible — a good
heuristic, not a guarantee; an application that commits without asking would be classified `risky`.
The redaction patterns are narrow and will miss institution-specific formats. And nothing defends
against a *compromised artifact*: an approved capability is trusted, so artifact review is a real
control — which is why the schema works so hard to keep them readable.

## 7. Cuts

**Not built, deliberately.** A desktop `Surface` (designed and argued, only web ships). A real
co-browsing console (out of scope; the control transfer is real, the chrome is not). Multi-tenant
infrastructure — no registry, no per-tenant deployment; the *artifact* is designed for reuse, the
plumbing is not. Stretch goals not chosen: multi-run flakiness scoring, code generation, and
assisted LLM fallback — the last would muddy the "no model in the replay loop" claim determinism
rests on. `LabelAdjacentLocator` supports right/left/same-row and returns `None` for above/below
rather than guessing at column arithmetic.

**Next, in order.**

1. **Approval workflow with reliability scoring.** The schema carries `draft`/`approved` and a
   replay counter; nothing moves a capability between them. Auto-promotion after N clean replays
   makes the guardrail self-maintaining instead of manual.
2. **A drift dashboard.** The tier comparison is the right signal and currently only appears in a
   run log. Across hundreds of tenants it wants to be a ranked list of capabilities sliding down
   the ladder.
3. **A desktop surface**, to prove the seam rather than argue it. The accessibility-tree bet is the
   load-bearing architectural claim here and the one thing reasoned rather than demonstrated.
4. **Shared sub-flows.** Both capabilities duplicate a search prefix.

**One thing I got wrong.** The first real `gpt-5` run offered `"4,211.03"` — the balance itself — as
proof of success. True for member 12345, false for everyone else; a capability asserting it would
pass once and fail forever. Synthesis now substitutes stable screen text, records why, and refuses
outright when none exists. The model cannot easily see this; the loop can, because it knows which
values were parameters and which were outputs.
