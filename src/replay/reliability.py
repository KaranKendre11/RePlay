"""How well a capability has actually replayed, and when that earns approval.

``Reliability`` on the artifact has always declared ``replays``, ``successes``
and ``last_verified_at``. Nothing wrote to them, so ``RiskGate.check_capability``
enforced an approval state that only a human editing JSON could ever change.
This module is the thing that maintains it.

The design question is where the counters live
---------------------------------------------

An artifact is deliberately a reviewable document — ``artifact/store.py`` says
so, and ``save`` refuses to clobber a published version because *a caller that
pinned* ``lookup_balance@1.0.0`` *must keep getting the same behaviour*. So the
obvious implementation — increment a field in the JSON after every run — is the
one thing this must not do. It would mean the README demo dirties the working
tree, the repo's own committed artifacts drift whenever anyone runs anything,
and ``save``'s immutability rule gets routed around with ``overwrite=True`` on
the hot path, at which point it protects nothing.

**So nothing is written back at all. The tally is derived.** Every replay
already writes a ``replay_finished`` event into its own evidence directory,
carrying the capability ref, the status, the per-step tier comparison and the
arguments it was called with. That is the record. :func:`tally` reads it. A
replay therefore mutates exactly what it mutates today — its own new evidence
directory — and the counters still move the moment it finishes.

Three things fall out of that, and they are the reason this beat a sidecar
counter file:

* There is no second source of truth to disagree with the evidence. A stored
  counter is a number that *claims* five clean replays happened; a derived one
  cannot be right unless the five runs are on disk to be read.
* Deleting evidence lowers the tally, which is correct. No evidence, no
  approval.
* Reliability is per *version and deployment*, keyed on the exact
  ``name@version#tenant`` ref, so a new version starts at zero — which is what
  the immutability rule already implies and a mutable counter would have
  quietly broken. The tenant belongs in the key because an override changes the
  host, the mount point, the selectors and the checkpoints: five clean
  ``--tenant northgate`` replays are not evidence about the base capability,
  and a base capability's approval is not evidence about Northgate.

The one thing that *is* written back is the approval itself, because that is a
decision rather than an observation, and decisions belong in the diff. See
:meth:`replay.artifact.store.ArtifactStore.approve`.

What the threshold requires
---------------------------

Promotion is never implicit. ``replay approve`` is typed by a person; this
module only decides whether to let them. Over the most recent :data:`WINDOW`
runs of that exact ref:

1. there have to *be* :data:`WINDOW` of them — a capability with two runs has
   not been observed, it has been sampled;
2. none of them failed;
3. none of them drifted — a step resolving below the tier it was recorded at
   still passes, but it is one vendor release from not passing, and approving
   something already sliding down the locator ladder is approving a countdown;
4. at least :data:`MIN_SUCCESSES` were *full* successes; and
5. they were not all the same call.

Rules 4 and 5 are where the interesting question sits. A replay that reaches a
declared business outcome is a successful invocation — ``ReplayResult.ok``
treats it that way and it is right to, because the caller got the answer it
asked for. But it exercises a *prefix* of the flow: ``MEMBER_NOT_FOUND`` stops
at the search box and never touches the steps that read a balance. Ten of those
prove the search works and say nothing about the rest of the capability, so
they are counted, reported, and deliberately not allowed to satisfy rule 4.

They do count for rule 5, and that is the point of it. The failure mode worth
blocking is a capability replayed N times against one happy-path fixture
promoting itself on volume alone; requiring the window to cover more than one
argument set is what makes "it works" mean something wider than "it works for
member 12345". Business outcomes are the cheapest honest source of that
diversity, which is a nice result: you cannot approve ``lookup_balance`` until
you have also replayed it against a member who does not exist.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from replay.artifact.schema import ApprovalState, CapabilityArtifact, Reliability

#: How many recent runs are examined. Small enough that a capability fixed
#: yesterday is approvable today, large enough that a single lucky run is not a
#: track record.
WINDOW = 5

#: How many of those must be full successes rather than business outcomes.
MIN_SUCCESSES = 3

#: The event a replay writes when it finishes, in ``<evidence>/<run>/run.jsonl``.
FINISHED = "replay_finished"

#: The event carrying the arguments the run was invoked with.
STARTED = "replay_started"


@dataclass(frozen=True)
class RunRecord:
    """One replay, as the evidence log recorded it."""

    run_id: str
    at: str
    status: str
    drifting_steps: tuple[str, ...]
    arguments: str
    evidence_dir: str

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def business_outcome(self) -> bool:
        return self.status == "business_outcome"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    @property
    def drifted(self) -> bool:
        return bool(self.drifting_steps)


@dataclass(frozen=True)
class ReliabilityTally:
    """What the evidence says about one capability version, right now."""

    ref: str
    runs: tuple[RunRecord, ...] = field(default_factory=tuple)

    @property
    def replays(self) -> int:
        return len(self.runs)

    @property
    def successes(self) -> int:
        return sum(1 for r in self.runs if r.succeeded)

    @property
    def outcomes(self) -> int:
        return sum(1 for r in self.runs if r.business_outcome)

    @property
    def failures(self) -> int:
        return sum(1 for r in self.runs if r.failed)

    @property
    def last_verified_at(self) -> datetime | None:
        """When this version last ran without malfunctioning.

        A run that failed verifies nothing, so it does not move the stamp.
        """
        for run in reversed(self.runs):
            if not run.failed:
                return datetime.fromisoformat(run.at)
        return None

    @property
    def window(self) -> tuple[RunRecord, ...]:
        return self.runs[-WINDOW:]

    def summary(self) -> str:
        """One line, for a terminal. Failures are named because they matter."""
        return (
            f"{self.replays} replays / {self.successes} success / "
            f"{self.outcomes} outcome / {self.failures} failed"
        )

    def snapshot(self, *, approval: ApprovalState = ApprovalState.DRAFT) -> Reliability:
        """The block to stamp into the artifact when a decision is recorded.

        Lifetime totals rather than the window: the artifact is citing the
        evidence the approval rested on, and a reviewer reading the diff should
        see the whole record, including the runs that failed.
        """
        return Reliability(
            replays=self.replays,
            successes=self.successes,
            outcomes=self.outcomes,
            last_verified_at=self.last_verified_at,
            approval=approval,
        )


def tally(evidence_root: Path | str, ref: str) -> ReliabilityTally:
    """Read every recorded replay of ``ref`` out of an evidence directory."""
    runs = sorted(_scan(Path(evidence_root), ref), key=lambda r: r.at)
    return ReliabilityTally(ref=ref, runs=tuple(runs))


def _scan(root: Path, ref: str) -> Iterator[RunRecord]:
    if not root.exists():
        return
    for log in sorted(root.glob("*/run.jsonl")):
        try:
            lines = log.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        arguments = "{}"
        for line in lines:
            # Evidence, not accounting: read what is legible and move on. A run
            # killed mid-write leaves a partial last line, and a hand-edited or
            # foreign log can hold a legible JSON line that is not an object
            # (AttributeError on ``.get``), one with no ``ts`` (KeyError), or
            # one whose stamp will not parse (ValueError) — none of which is a
            # reason for ``replay run``'s post-run report and ``replay approve``
            # to die.
            try:
                record = json.loads(line)

                if record.get("kind") == STARTED:
                    arguments = json.dumps(record.get("arguments", {}), sort_keys=True)
                    continue
                if record.get("kind") != FINISHED or record.get("capability") != ref:
                    continue

                # A run refused at the door, or rejected for bad arguments,
                # never touched the application: zero steps ran. Counting it
                # would be counting the guardrail rather than the capability,
                # and one refusal would poison the window of the capabilities
                # the guardrail exists to protect.
                if not record.get("steps"):
                    continue

                at = record["ts"]
                # Parsed here rather than trusted: it is the sort key for the
                # whole window and the value ``last_verified_at`` returns.
                datetime.fromisoformat(at)
                run = RunRecord(
                    run_id=record.get("run_id", log.parent.name),
                    at=at,
                    status=record.get("status", "failed"),
                    drifting_steps=tuple(record.get("drifting_steps") or ()),
                    arguments=arguments,
                    evidence_dir=log.parent.name,
                )
            except (ValueError, AttributeError, KeyError, TypeError):
                continue
            yield run


def promotion_blockers(
    tallied: ReliabilityTally,
    artifact: CapabilityArtifact,
    *,
    window: int = WINDOW,
    min_successes: int = MIN_SUCCESSES,
) -> list[str]:
    """Why this capability may not be approved yet. Empty means it may.

    Reasons rather than a boolean, because an operator refused promotion needs
    to know what to go and do about it.
    """
    if artifact.reliability.approval is ApprovalState.APPROVED:
        return [f"{artifact.ref} is already approved"]

    recent = tallied.runs[-window:]
    if len(recent) < window:
        return [
            f"only {len(recent)} recorded {_runs(len(recent))} of {artifact.ref}; "
            f"{window} are needed before there is a track record to judge"
        ]

    blockers: list[str] = []

    failed = [r for r in recent if r.failed]
    if failed:
        blockers.append(
            f"{len(failed)} of the last {window} runs failed "
            f"({', '.join(r.evidence_dir for r in failed)})"
        )

    drifted = [r for r in recent if r.drifted]
    if drifted:
        steps = sorted({s for r in drifted for s in r.drifting_steps})
        blockers.append(
            f"{len(drifted)} of the last {window} runs resolved a step below the tier it was "
            f"recorded at ({', '.join(steps)}); re-record before approving something "
            f"already drifting"
        )

    successes = sum(1 for r in recent if r.succeeded)
    if successes < min_successes:
        blockers.append(
            f"only {successes} of the last {window} runs fully succeeded, {min_successes} needed; "
            f"business outcomes are successful invocations but exercise fewer steps, so they "
            f"do not count towards this"
        )

    # A capability with no arguments has only one call to make, so requiring
    # variety in it would make approval unreachable rather than harder.
    if artifact.inputs and len({r.arguments for r in recent}) < 2:
        blockers.append(
            f"the last {window} runs were all the same call; replay it against at least one "
            f"other argument set so approval means more than 'it works for one fixture'"
        )

    return blockers


def _runs(count: int) -> str:
    return "run" if count == 1 else "runs"
