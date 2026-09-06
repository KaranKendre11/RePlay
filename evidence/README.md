# Evidence

Committed output from real runs. Nothing here is reconstructed or hand-written.

Every directory contains:

- `run.jsonl` — structured decision log: what happened, in order, and why
- `result.json` — the run's typed result
- `observations/step-NN.json` — the accessibility snapshot the system saw
- `steps/step-NN.png` — the screen at that moment

Discovery runs additionally keep `transcript.jsonl`, the model's messages. The artifact
references it by path and never embeds it — the brief asks for a capability decoupled from the
raw transcript, and keeping the pointer means a reviewer can still audit the distillation.

Failure runs additionally keep `dom/<step>.html`. Markup is useless for deciding what to do,
which is why observations carry an accessibility tree instead — but it is invaluable for working
out afterwards why something broke, so it is captured here and nowhere else.

## Discovery — a model driving the UI

| Directory | Goal | Result |
|---|---|---|
| `discovery-20260906T085541Z` | Look up member 12345 and read their savings balance | `goal_met`, 4 actions |
| `discovery-20260906T211130Z` | Open a sub-account and reach the confirmation screen | `goal_met`, 9 actions |

Both are genuine `gpt-5` runs against the live application. The second is the interesting one: it
contains an irreversible step, and the model had to discover that the submit button raises a
confirmation dialog which must be accepted.

## Replay — the same work, without a model

| Directory | Shows |
|---|---|
| `replay-01-success` | Typed outputs, tiers 3/1/4, ~150 ms |
| `replay-02-business-outcome` | `MEMBER_NOT_FOUND` returned as a **result**, not an exception |
| `replay-03-hard-failure` | `APPLICATION_ERROR`, with expected-vs-observed and a DOM snapshot |
| `replay-04-recovered` | An unexpected interstitial dismissed, recorded, run completes |
| `replay-05-cross-tenant` | The same artifact against a second institution, via overrides only |
| `replay-06-policy-refused` | An irreversible capability blocked before the browser opened |

Worth comparing 01 against 02 and 03: all three are the same capability with the same arguments
shape, and the difference between "an answer you did not expect" and "something is broken" is
visible in the result rather than inferred from an exception type.
