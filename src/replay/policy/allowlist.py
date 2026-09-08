"""What the automation is permitted to touch.

Enforced inside :meth:`~replay.surface.web.WebSurface.act`, not at the call
site. That placement is the whole design: a guardrail checked by the caller is
a guardrail every future caller must remember to check, and one of them
eventually will not. Putting it at the point of action means discovery, replay,
recovery rules and anything written later are all covered without knowing the
allowlist exists.

Deny beats allow. An explicitly denied route is refused even if a broader allow
rule would have permitted it, because the reason someone writes a deny rule is
that a general rule was too generous.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from replay.artifact.schema import Action

DEFAULT_POLICY_FILE = Path("policy.toml")


class PolicyRefused(PermissionError):
    """A guardrail blocked this. Not a malfunction — a refusal.

    Kept distinct from every other error in the system so it can never be
    retried, recovered from, or mistaken for the application misbehaving.
    """

    def __init__(self, what: str, why: str) -> None:
        self.what = what
        self.why = why
        super().__init__(f"refused {what}: {why}")


@dataclass(frozen=True)
class Allowlist:
    """Where the automation may go and what it may do there."""

    #: Every field defaults to empty, and empty means "nothing is permitted".
    #: The failure mode of a missing config should be a refusal, not free rein,
    #: and that has to hold for each field separately. An absent ``actions``
    #: key quietly meaning "all nine" made the list an operator is most likely
    #: to leave out the list that granted the most.
    domains: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    actions: frozenset[Action] = frozenset()
    denied_routes: tuple[str, ...] = ()

    @property
    def permits_nothing(self) -> bool:
        return not self.domains

    # -- checks -----------------------------------------------------------

    def check_navigation(self, url: str) -> None:
        parts = urlsplit(url)
        host = parts.netloc
        path = parts.path or "/"

        if self.permits_nothing:
            raise PolicyRefused(f"navigation to {url}", "no domains are allowlisted")

        if not any(fnmatch.fnmatch(host, pattern) for pattern in self.domains):
            raise PolicyRefused(f"navigation to {url}", f"host {host!r} is not in the allowlist")

        for pattern in self.denied_routes:
            if fnmatch.fnmatch(path, pattern):
                raise PolicyRefused(
                    f"navigation to {url}", f"route matches the deny rule {pattern!r}"
                )

        if not any(fnmatch.fnmatch(path, pattern) for pattern in self.routes):
            raise PolicyRefused(f"navigation to {url}", f"route {path!r} is not in the allowlist")

    def check_action(self, action: Action) -> None:
        # Consulted here as well as in check_navigation. Without it, a config
        # with no domains still permitted clicking and typing on whatever page
        # happened to be open — the navigation that opened it being the only
        # thing that was ever refused.
        if self.permits_nothing:
            raise PolicyRefused(f"action {action.value!r}", "no domains are allowlisted")
        if action not in self.actions:
            raise PolicyRefused(f"action {action.value!r}", "this action type is not permitted")

    # -- loading ----------------------------------------------------------

    @classmethod
    def permissive(cls, *domains: str) -> Allowlist:
        """For tests and local development. Never a default.

        Spelled out rather than leaning on the field defaults, because those
        now refuse everything — which is the point of them.
        """
        return cls(domains=tuple(domains) or ("*",), routes=("*",), actions=frozenset(Action))

    @classmethod
    def from_file(cls, path: Path | str = DEFAULT_POLICY_FILE) -> Allowlist:
        raw = tomllib.loads(Path(path).read_text())
        return cls.from_dict(raw.get("allowlist", {}))

    @classmethod
    def from_dict(cls, raw: dict) -> Allowlist:
        return cls(
            domains=tuple(raw.get("domains", ())),
            routes=tuple(raw.get("routes", ())),
            actions=frozenset(Action(a) for a in raw.get("actions", ())),
            denied_routes=tuple(raw.get("denied_routes", ())),
        )

    def describe(self) -> dict[str, list[str]]:
        """For the evidence log and the operator console.

        A refusal is only useful if whoever reads it can see what the rules
        were at the time.
        """
        return {
            "domains": list(self.domains),
            "routes": list(self.routes),
            "denied_routes": list(self.denied_routes),
            "actions": sorted(a.value for a in self.actions),
        }
