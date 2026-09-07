# Evidence

Committed output from real runs. Nothing here is reconstructed or hand-written.

Every directory contains:

- `run.jsonl` — structured decision log: what happened, in order, and why
- `result.json` — the run's typed result
- `capability.json` — the capability the run produced, or the one it replayed

Discovery runs additionally carry a snapshot per step:

- `observations/step-NN.json` — the accessibility snapshot the system saw
- `steps/step-NN.png` — the screen at that moment

A replay carries those only where something went wrong. Capturing a screenshot at every step of
a 150 ms run would cost more than the run and prove nothing; the interesting state is the one the
flow failed on. So `replay-03-hard-failure` and `replay-07-escalation` have snapshots and the
other five do not.

**`artifacts/` is the source of truth for capabilities.** Each `capability.json` here is a
*point-in-time copy* of what that run produced or replayed, so that a reviewer opening `evidence/`
finds an artifact alongside the discovery and replay logs without leaving the directory. Edit a
copy and nothing happens; edit `artifacts/` and the catalog changes, because the catalog *is* that
directory — there is no registry that can disagree with it. Copies are never read by the system.

One directory has no `capability.json`: `discovery-20260906T091017Z`, where synthesis refuses to
produce one. It carries `no-capability.txt` instead, so the absence reads as the result it is
rather than as an oversight.

Discovery runs additionally keep `transcript.jsonl`, the model's messages. The artifact
references it by path and never embeds it — the brief asks for a capability decoupled from the
raw transcript, and keeping the pointer means a reviewer can still audit the distillation.

Failure runs additionally keep `dom/<step>.html`. Markup is useless for deciding what to do,
which is why observations carry an accessibility tree instead — but it is invaluable for working
out afterwards why something broke, so it is captured here and nowhere else.

## Discovery — a model driving the UI

| Directory | Goal | Result | Capability produced |
|---|---|---|---|
| `discovery-20260906T085541Z` | Look up member 12345 and read their savings balance | `goal_met`, 4 actions | `lookup_balance@1.1.0` |
| `discovery-20260906T091017Z` | The same goal, and the model proposed a checkpoint that was rejected | `goal_met`, 4 actions | **none — synthesis refuses** |
| `discovery-20260906T211130Z` | Open a sub-account and reach the confirmation screen | `goal_met`, 9 actions | `open_subaccount@1.0.0` |

All three are genuine `gpt-5` runs against the live application. Each copied `capability.json`
names its origin run in `provenance.run_id`, so the pairing above is checkable rather than
asserted.

The third contains an irreversible step, and the model had to discover that the submit button
raises a confirmation dialog which must be accepted.

The second is the one worth reading against `REPORT.md` §7. Asked for text proving the goal was
met, the model offered `"4,211.03"` — the balance it had just read. That is true for member 12345
and false for every other member, so a capability asserting it would pass once and fail forever.
The loop catches this because it knows which values were parameters and which were outputs, and
the warning is in `result.json`:

> checkpoint `'4,211.03'` contains the value read as output `'current_savings_balance'`; it
> asserts this run's data rather than the state reached

Synthesising a capability from this run then fails outright:

```
$ uv run replay synthesize evidence/discovery-20260906T091017Z --name lookup_balance
synthesis failed: no stable checkpoint available: the model proposed '4,211.03', which varies
per invocation, and no stable text was captured on the success screen
```

That is why this is the one directory with no `capability.json`. Refusing is the right answer here,
and it is the third of the three behaviours `REPORT.md` §7 describes. This run predates the loop
capturing stable alternatives from the success screen, so there is genuinely nothing to
substitute — `result.json` carries no `checkpoint_candidates`,
where the later `discovery-20260906T211130Z` carries eleven and synthesis substitutes instead of
refusing. Emitting a capability whose success condition only holds for member 12345 would be
worse than emitting none.

Compare with `discovery-20260906T085541Z`: same goal, same loop, and the model proposed
`"Open Sub-Account"` unprompted, so no warning fired and the capability synthesised cleanly. Two
runs against the same screen, two different choices by the model. That difference is the whole
case for verifying the checkpoint rather than trusting it.

## Replay — the same work, without a model

| Directory | Shows |
|---|---|
| `replay-01-success` | Typed outputs, tiers 3/1/4, ~150 ms |
| `replay-02-business-outcome` | `MEMBER_NOT_FOUND` returned as a **result**, not an exception |
| `replay-03-hard-failure` | `APPLICATION_ERROR`, with expected-vs-observed and a DOM snapshot |
| `replay-04-recovered` | An unexpected interstitial dismissed, recorded, run completes |
| `replay-05-cross-tenant` | The same artifact against a second institution, via overrides only |
| `replay-06-policy-refused` | An irreversible capability blocked before the browser opened |
| `replay-07-escalation` | The same capability finished by a human, on the same live session |

`replay-01` through `replay-05` all run `lookup_balance@1.1.0`; `replay-06` and `replay-07` run
`open_subaccount@1.0.0`, the irreversible one. `replay-05-cross-tenant` also
carries `override.json`, the Northgate deltas it was run with — without them its `capability.json`
would show the base targets rather than the ones actually resolved.

Worth comparing 01 against 02 and 03: all three are the same capability with the same arguments
shape, and the difference between "an answer you did not expect" and "something is broken" is
visible in the result rather than inferred from an exception type.

### The handoff, in `replay-07-escalation`

Read `run.jsonl` in order. It is the whole control-transfer model in one file:

```
step ×6 → policy_refused → escalation_raised → control_transferred(operator)
        → control_transferred(automation) → escalation_resolved → step → replay_finished
```

Six steps run unattended. Step `s7` is irreversible, so the guardrail refuses it — and because a
person is reachable, the refusal becomes an intervention rather than a failure. The automation
releases the session, a human performs the step **in that same browser context** — same cookies,
same server-side session, same page, nothing recreated — and hands it back. The automation
reacquires and re-asserts the checkpoint rather than assuming the person succeeded, then reads the
new account number and returns it: `{"new_account_no": "12345046"}`.

What the operator did is captured, not self-reported:

```json
"human_actions": [
  {"kind": "change", "label": "Opening Deposit field"},
  {"kind": "click",  "label": "Submit"}
]
```

Note what is *not* there. The listener records which control was touched and never what was typed
— the deposit field is named, its contents are not. An operator handling a bank escalation is
often typing exactly the data this system must not persist.

**What is mocked here, and what is not.** The control transfer is entirely real: the same
executor, the same guardrail, the same browser context, the same release-and-reacquire the CLI
uses under `--escalate`. The only stand-in is *who does the clicking* — `ScriptedOperator`, the
same one the test suite uses, because a live handoff cannot be committed to a repository without
a person at a keyboard at the moment it is recorded. Run it yourself with a real person:

```bash
uv run replay run open_subaccount \
    -p member_id=12345 -p product_code=S02 -p opening_deposit=50.00 \
    --allow-risky --escalate
```

The capability also carries `approval: draft`, and an unapproved capability requiring approval is
refused. This run marks it approved first, standing in for the review step the workflow expects
before unattended use.
