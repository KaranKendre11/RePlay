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
from typing import Any

REDACTED = "«redacted»"
DEFAULT_ROOT = Path("evidence")


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


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
        self.steps_dir = self.dir / "steps"
        self.obs_dir = self.dir / "observations"
        for directory in (self.dir, self.steps_dir, self.obs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._mask = [m for m in mask if m]
        self._log = (self.dir / "run.jsonl").open("a", encoding="utf-8")
        self._transcript = (self.dir / "transcript.jsonl").open("a", encoding="utf-8")

    # -- redaction --------------------------------------------------------

    def add_mask(self, value: str | None) -> None:
        """Register a value that must never appear in evidence."""
        if value:
            self._mask.append(value)

    def redact(self, payload: Any) -> Any:
        text = json.dumps(payload, ensure_ascii=False, default=str)
        for secret in self._mask:
            text = text.replace(secret, REDACTED)
        return json.loads(text)

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
        for secret in self._mask:
            text = text.replace(secret, REDACTED)
        path.write_text(text, encoding="utf-8")
        return str(path.relative_to(self.dir))

    def result(self, payload: dict[str, Any]) -> Path:
        path = self.dir / "result.json"
        path.write_text(json.dumps(self.redact(payload), indent=2, ensure_ascii=False) + "\n")
        return path

    def close(self) -> None:
        for handle in (self._log, self._transcript):
            if not handle.closed:
                handle.close()

    def __enter__(self) -> EvidenceRecorder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
