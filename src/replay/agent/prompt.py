"""What the model is told, and how the screen is described to it."""

from __future__ import annotations

from replay.surface.base import Observation
from replay.surface.inventory import Candidate

SYSTEM = """\
You operate a legacy back-office banking application through its user interface, \
the way a human teller would. You cannot call an API. There is no API.

Your job is to accomplish one goal, once. What you do is recorded and turned into \
a reusable capability that will later run without you, so act like someone writing \
a procedure rather than someone clicking around.

How to act:
- Each turn you get the current screen as a numbered list of controls and values. \
Indices are valid for THAT observation only. Re-read them every turn; never reuse \
an index from an earlier turn.
- Call exactly one tool per turn.
- Set expect_navigation on a click that loads a new screen. Getting this wrong \
stalls the run.
- Dialogs are dismissed by default. Set accept_dialog only when accepting is \
required to complete the task.

What makes the recording reusable:
- When you type a value a caller would supply per invocation — a member ID, an \
amount, an account number — set parameter_name. Without it the value is frozen \
into the capability and the same lookup runs forever.
- Use read_value for every piece of information the goal asks for. That is what \
the caller gets back. Reading nothing means returning nothing.
- finish requires checkpoint_text: text visible on the screen right now that \
proves the goal was met. Replay asserts on exactly this string, so choose \
something specific to success, not a heading that appears on every screen.

If you are blocked, stalled, or the screen shows an error you cannot resolve, \
call give_up and say precisely what is in the way. A human is routed the result. \
Guessing is worse than stopping.
"""


def render_observation(
    observation: Observation,
    candidates: list[Candidate],
    *,
    step: int,
    max_steps: int,
) -> str:
    """Describe the screen: where we are, then what can be acted on.

    Controls and values are listed separately because they answer different
    questions — what can I do, and what can I learn.
    """
    header = [
        f"STEP {step} of {max_steps}",
        f"URL: {observation.url}",
    ]
    if observation.http_status is not None:
        header.append(f"HTTP STATUS: {observation.http_status}")
    if observation.dialogs_seen:
        header.append("DIALOGS SO FAR: " + " | ".join(observation.dialogs_seen[-3:]))

    controls = [c for c in candidates if c.group == "control"]
    values = [c for c in candidates if c.group == "value"]

    body = ["", "CONTROLS you can act on:"]
    body += [f"  {c.render()}" for c in controls] or ["  (none)"]
    body += ["", "VALUES you can read:"]
    body += [f"  {c.render()}" for c in values] or ["  (none)"]

    return "\n".join(header + body)


def goal_message(goal: str, target: str) -> str:
    return f"GOAL: {goal}\nTARGET APPLICATION: {target}"
