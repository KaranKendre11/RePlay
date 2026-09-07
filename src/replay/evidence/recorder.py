"""Run evidence: what happened, and enough to argue about it afterwards.

Discovery and replay write the same shape, deliberately. A reviewer comparing a
recorded run against a replayed one should not have to learn two formats, and
the symmetry is what makes "the artifact faithfully describes the run" a
checkable claim rather than a hopeful one.

Redaction happens on the way in, not on the way out. A value that is masked
only when someone remembers to mask it is a value that eventually gets written.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_hex
from typing import Any

from replay.policy.redaction import Redactor

DEFAULT_ROOT = Path("evidence")


def new_run_id(prefix: str) -> str:
    """An id no other run can share.

    Second resolution was not enough: a replay takes ~150 ms, so back-to-back
    runs landed in one directory and overwrote each other's result and
    screenshots. Microseconds keep the ids in run order; the random suffix is
    what makes them unique rather than merely unlikely to repeat, which matters
    now that a shared directory is refused rather than silently merged.
    """
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%f')}Z-{token_hex(3)}"


class EvidenceRecorder:
    """Writes one run's evidence directory."""

    def __init__(
        self,
        run_id: str,
        *,
        root: Path | str = DEFAULT_ROOT,
        mask: Sequence[str] = (),
    ) -> None:
        self.run_id = run_id
        self.dir = Path(root) / run_id
        # A run id names one run. Recording a second run into an existing
        # directory leaves a log describing two runs beside a result.json
        # describing one, and screenshots from whichever ran last — so re-using
        # a --label is refused, as ArtifactStore.save refuses to clobber a
        # published version. Generated ids are unique and never reach this.
        if self.dir.exists() and any(self.dir.iterdir()):
            raise FileExistsError(
                f"{self.dir} already holds a run; delete it or choose another "
                f"label rather than recording two runs into one directory"
            )
        self.steps_dir = self.dir / "steps"
        self.obs_dir = self.dir / "observations"
        for directory in (self.dir, self.steps_dir, self.obs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.redactor = Redactor(mask)
        self._log = (self.dir / "run.jsonl").open("a", encoding="utf-8")
        # Opened on first use. Replay has no model, so a replay's evidence
        # should not contain an empty transcript implying otherwise.
        self._transcript: Any = None

    # -- redaction --------------------------------------------------------

    def add_mask(self, value: str | None) -> None:
        """Register a value that must never appear in evidence."""
        self.redactor.add(value)

    def redact(self, payload: Any) -> Any:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        return json.loads(self.redactor.scrub(text))

    # -- writing ----------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> None:
        record = self.redact({"ts": datetime.now(UTC).isoformat(), "kind": kind, **fields})
        self._log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._log.flush()

    def message(self, role: str, content: Any) -> None:
        """Append to the model transcript, kept separate from the run log.

        The artifact is decoupled from the transcript; the evidence keeps both
        so a reviewer can check the distillation without the capability
        carrying the model's reasoning around forever.
        """
        if self._transcript is None:
            self._transcript = (self.dir / "transcript.jsonl").open("a", encoding="utf-8")
        record = self.redact(
            {"ts": datetime.now(UTC).isoformat(), "role": role, "content": content}
        )
        self._transcript.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._transcript.flush()

    def observation(self, step: int, observation: Any, rendered: str) -> dict[str, str]:
        """Persist a screenshot and the structured observation for one step."""
        refs: dict[str, str] = {}
        stem = f"step-{step:02d}"

        if getattr(observation, "screenshot", None):
            shot = self.steps_dir / f"{stem}.png"
            shot.write_bytes(observation.screenshot)
            refs["screenshot"] = str(shot.relative_to(self.dir))

        payload = self.redact({"rendered": rendered, **observation.to_dict()})
        obs_path = self.obs_dir / f"{stem}.json"
        obs_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        refs["observation"] = str(obs_path.relative_to(self.dir))
        return refs

    def snapshot_text(self, name: str, text: str) -> str:
        """Store a richer failure signal, e.g. a DOM dump."""
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.redactor.scrub(text), encoding="utf-8")
        return str(path.relative_to(self.dir))

    def result(self, payload: dict[str, Any]) -> Path:
        path = self.dir / "result.json"
        path.write_text(json.dumps(self.redact(payload), indent=2, ensure_ascii=False) + "\n")
        return path

    def close(self) -> None:
        for handle in (self._log, self._transcript):
            if handle is not None and not handle.closed:
                handle.close()

    def __enter__(self) -> EvidenceRecorder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
