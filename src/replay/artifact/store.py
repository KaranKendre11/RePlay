"""Artifact persistence.

Files on disk, one JSON document per capability version. Not a database, and
that is the point: a capability is a reviewable document. It should show up in a
pull request diff, be arguable line by line, and be readable without running
anything. Rows in a table are none of those.

Filenames are ``<name>@<version>.json``, so the store's directory listing is
already the catalogue.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from replay.artifact.schema import CapabilityArtifact, Reliability

DEFAULT_ROOT = Path("artifacts")

log = logging.getLogger(__name__)


class ArtifactNotFound(LookupError):
    pass


class ArtifactInvalid(ValueError):
    """A file is present but is not the capability artifact it claims to be.

    Distinct from :class:`ArtifactNotFound` on purpose. "There is nothing here"
    and "there is something here and it is broken" call for different reactions
    from an operator, and collapsing them into one error — or letting the raw
    ``ValidationError`` escape — is how a single bad file gets reported as a
    server fault rather than as the one capability that needs fixing.
    """


class ArtifactStore:
    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    # -- paths ------------------------------------------------------------

    def path_for(self, name: str, version: str) -> Path:
        return self.root / f"{name}@{version}.json"

    # -- read -------------------------------------------------------------

    def load_path(self, path: Path | str) -> CapabilityArtifact:
        """Read one artifact file, or say precisely why it is not one."""
        path = Path(path)
        try:
            return CapabilityArtifact.model_validate_json(path.read_text())
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError, OSError) as bad:
            raise ArtifactInvalid(f"{path} is not a readable capability artifact: {bad}") from bad

    def load(self, name: str, version: str | None = None) -> CapabilityArtifact:
        """Load a capability. Without a version, the highest semver wins."""
        if version is not None:
            path = self.path_for(name, version)
            if not path.exists():
                raise ArtifactNotFound(f"{name}@{version} not found in {self.root}")
            return self.load_path(path)

        candidates = [a for a in self.list_all() if a.name == name]
        if not candidates:
            raise ArtifactNotFound(f"no versions of {name!r} in {self.root}")
        return max(candidates, key=lambda a: a.version_tuple)

    def list_all(self) -> list[CapabilityArtifact]:
        return sorted(self._iter_all(), key=lambda a: (a.name, a.version_tuple))

    def _iter_all(self) -> Iterator[CapabilityArtifact]:
        """Every readable artifact in the directory, skipping the ones that are not.

        The catalogue is the directory — drop a file in, it is callable; delete
        it, it is gone. One unreadable file has to mean *that* capability is
        gone, not all of them: propagating the error would take ``list_all``,
        ``names`` and every unversioned ``load`` down with it, so a corrupt
        ``lookup_balance@1.1.0.json`` would make ``open_subaccount``
        uninvocable too, and ``GET /capabilities`` a 500.

        Logged rather than swallowed. A capability that silently stops being
        offered is the same class of problem in the other direction.
        """
        if not self.root.exists():
            return
        for path in sorted(self.root.glob("*.json")):
            try:
                yield self.load_path(path)
            except ArtifactInvalid as bad:
                log.warning("skipping unreadable artifact: %s", bad)

    def names(self) -> list[str]:
        return sorted({a.name for a in self.list_all()})

    # -- write ------------------------------------------------------------

    def save(self, artifact: CapabilityArtifact, *, overwrite: bool = False) -> Path:
        """Write an artifact.

        Refuses to clobber an existing version by default. A published capability
        version is immutable — a caller that pinned ``lookup_balance@1.0.0`` must
        keep getting the same behaviour. Change means a new version.
        """
        if artifact.tenant:
            raise ValueError(
                f"{artifact.ref} is a tenant specialisation, not a publishable "
                "capability; it would be written over the base recording it was "
                "derived from. Save the base, and the deltas as an override."
            )
        path = self.path_for(artifact.name, artifact.version)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{artifact.ref} already exists; bump the version rather than "
                f"editing a published capability (or pass overwrite=True)"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        _write_atomically(path, serialize(artifact))
        return path

    def approve(self, name: str, version: str, reliability: Reliability) -> Path:
        """Record an approval decision against an already-published version.

        The single sanctioned in-place edit of a published artifact, and it is
        narrow on purpose. ``save`` refuses to clobber because a caller that
        pinned ``lookup_balance@1.0.0`` must keep getting the same behaviour —
        and approving it does not change the behaviour. The steps, inputs,
        outputs and policy are identical afterwards; what changed is that the
        organisation has said it trusts them unattended. Publishing ``1.0.1``
        instead would force every pinned caller to re-pin for a change that is
        not one, and would make the semver advertise a difference that does not
        exist.

        So the artifact is re-read from disk and only its reliability block is
        replaced. A caller cannot smuggle a step edit through here alongside an
        approval, which is exactly the thing the immutability rule is defending
        against, and the resulting diff is the reliability block and nothing
        else — which is what makes it reviewable.
        """
        published = self.load(name, version)
        updated = published.model_copy(update={"reliability": reliability})
        path = self.path_for(name, version)
        _write_atomically(path, serialize(updated))
        return path


def _write_atomically(path: Path, text: str) -> None:
    """Write via a sibling temp file and rename.

    ``os.replace`` is atomic, so a crash, a kill, or a full disk part-way
    through leaves the previous file intact instead of a truncated one — and a
    truncated artifact is precisely the input that used to take the whole
    catalogue down. The store's own writes should not be able to manufacture
    the corruption the reader has to tolerate.

    Sibling rather than ``/tmp`` because ``os.replace`` is only atomic within a
    filesystem, and the temp name is dot-prefixed so a half-written artifact is
    never picked up by the ``*.json`` scan.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def serialize(artifact: CapabilityArtifact) -> str:
    """Stable, diff-friendly JSON.

    Two-space indent and a trailing newline so a version bump shows up as the
    lines that actually changed.
    """
    payload = artifact.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def json_schema() -> dict:
    """JSON Schema for the artifact.

    Exported so a calling agent can validate arguments, and so the contract is
    consumable by something that is not Python.
    """
    return CapabilityArtifact.model_json_schema()


def invocation_schema(artifact: CapabilityArtifact) -> dict:
    """JSON Schema for *invoking* this capability.

    The catalogue surface an agent sees (M10): the typed argument object, derived
    from the declared inputs.
    """
    json_types = {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "money": "string",
    }
    properties: dict[str, dict] = {}
    for param in artifact.inputs:
        prop: dict[str, object] = {
            "type": json_types[param.type.value],
            "description": param.description,
        }
        if param.pattern:
            prop["pattern"] = param.pattern
        if param.example is not None:
            prop["examples"] = [param.example]
        properties[param.name] = prop

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": artifact.ref,
        "description": artifact.description,
        "type": "object",
        "properties": properties,
        "required": [p.name for p in artifact.inputs if p.required],
        "additionalProperties": False,
    }
