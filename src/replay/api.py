"""The capability catalog: what an AI agent actually talks to.

This closes the brief's own through-line — *the model discovers, the artifact
becomes a reusable capability, deterministic replay is how the agent invokes it
in production*. Everything up to here produced artifacts; this is where they
become callable by name with typed arguments.

Three things it deliberately does.

**The catalog is the artifacts directory.** There is no registry to keep in
sync, because a registry that can disagree with the files eventually does. Drop
an artifact in, it is callable; delete it, it is gone.

**Arguments are validated before a browser opens.** Each capability publishes a
JSON Schema derived from its declared inputs, so a caller can check locally and
a bad call costs nothing. An agent that has to invoke something to find out what
it takes is not being offered a contract.

**The API cannot outrank the guardrails.** Invocation goes through the same
executor, the same allowlist and the same risk gate as the CLI. An HTTP endpoint
that quietly permits more than the command line would make the guardrails
decorative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from replay.artifact import (
    ArtifactInvalid,
    ArtifactNotFound,
    ArtifactStore,
    invocation_schema,
    specialise,
)
from replay.artifact.overrides import DEFAULT_ROOT as OVERRIDES_ROOT
from replay.artifact.overrides import OverrideRejected, TenantUnknown
from replay.artifact.schema import CapabilityArtifact, unrunnable_on_a_browser
from replay.engine import ReplayExecutor, ReplayStatus
from replay.escalation import ConsoleEscalation, InterventionQueue
from replay.escalation.console import create_console
from replay.evidence import EvidenceRecorder, new_run_id
from replay.policy import Allowlist, RiskGate
from replay.surface import WebSurface


class InvokeRequest(BaseModel):
    """One call. Arguments are checked against the capability's own schema."""

    # An unknown field is refused rather than dropped. Silently ignoring one
    # means a caller that sends `allow_irreversible` believes it raised the
    # ceiling and is told nothing — which is worse than the field existing.
    model_config = ConfigDict(extra="forbid")

    arguments: dict[str, Any] = Field(default_factory=dict)
    target: str | None = Field(
        default=None,
        description="Run against this origin instead of the recorded one.",
    )
    tenant: str | None = Field(
        default=None,
        description="Apply this tenant's overrides. The capability's contract is unchanged.",
    )
    # No allow_risky / allow_irreversible. They were plain booleans in an
    # unauthenticated request body, which is the arrangement policy/risk.py
    # argues against by name: a flag the caller can set moves the decision to
    # whoever wrote the calling code. The ceiling is a deployment's, and lives
    # in policy.toml beside the allowlist.
    escalate: bool = Field(
        default=False,
        description=(
            "Route blocks to a human operator and wait. Off by default: an agent "
            "calling an API expects an answer, not a fifteen-minute hold while "
            "someone is found."
        ),
    )
    escalation_timeout_s: float = Field(default=300.0, gt=0)


def _as_json(status: int):
    """One error shape for every failure the catalog knows how to name."""

    def handler(_request: Request, failure: Exception) -> JSONResponse:
        return JSONResponse({"error": str(failure)}, status_code=status)

    return handler


def summarise(artifact: CapabilityArtifact) -> dict[str, Any]:
    """What an agent needs to decide whether to call this at all.

    Outcomes are included on purpose: a caller should know the full space of
    results — including the legitimate non-successes — before invoking, not
    after.
    """
    return {
        "name": artifact.name,
        "version": artifact.version,
        "ref": artifact.ref,
        "title": artifact.title,
        "description": artifact.description,
        "application": artifact.app.product,
        "product_version": artifact.app.product_version,
        "surface": artifact.app.surface.value,
        "arguments": invocation_schema(artifact),
        "returns": [
            {"name": o.name, "type": o.type.value, "description": o.description}
            for o in artifact.outputs
        ],
        "outcomes": [{"code": o.code, "message": o.message} for o in artifact.outcomes],
        "risk": artifact.max_step_risk.value,
        "requires_approval": artifact.policy.requires_approval,
        "approval": artifact.reliability.approval.value,
        # The snapshot recorded when this version was last decided on, not a
        # live count. The catalog is served from the artifacts directory and
        # should answer identically for every caller; deriving the running
        # tally here would make a read of the catalog depend on which evidence
        # root the server happened to be started with. `replay approve` is
        # where the live numbers are read, because that is where they change
        # something.
        "reliability": {
            "replays": artifact.reliability.replays,
            "successes": artifact.reliability.successes,
            "outcomes": artifact.reliability.outcomes,
        },
    }


def create_api(
    *,
    artifacts_dir: Path | str = "artifacts",
    policy_file: Path | str = "policy.toml",
    # Not "evidence": that directory is a curated deliverable and it is what the
    # approval tally is read from, so a server producing evidence continuously
    # from agent traffic would both dirty it and move numbers nobody meant to
    # move. A one-shot `replay run` is a deliberate act and still defaults there.
    evidence_dir: Path | str = "runs",
    # Explicit, because the default is relative to the process's working
    # directory: a server started anywhere but the repo root silently found no
    # overrides at all and ran base capabilities against tenant deployments.
    overrides_dir: Path | str = OVERRIDES_ROOT,
    queue: InterventionQueue | None = None,
    headed: bool = False,
    allowlist: Allowlist | None = None,
    gate: RiskGate | None = None,
) -> FastAPI:
    store = ArtifactStore(artifacts_dir)
    # Loaded from the shipped policy unless a caller supplies one. Tests bind to
    # an ephemeral port, which the production policy rightly does not allow.
    allowlist = allowlist if allowlist is not None else Allowlist.from_file(policy_file)
    # One ceiling for the life of the server, from the same file as the
    # allowlist. Raising it is a config change someone reviews, not a field a
    # caller sets per request.
    gate = gate if gate is not None else RiskGate.from_file(policy_file)
    queue = queue if queue is not None else InterventionQueue()

    app = FastAPI(
        title="RePlay capability catalog",
        description="Saved capabilities, callable by name with typed arguments.",
        version="1.0",
    )

    # Registered once, rather than repeated at every endpoint. A per-route
    # handler list is one somebody forgets on the next route, and that is
    # exactly what happened: `get_artifact` and `invoke` caught only
    # `ArtifactNotFound` — raised solely from a `path.exists()` check — so a
    # file that existed but did not parse came back as a 500 with a traceback,
    # and the difference between 404 and 500 was an existence oracle for paths
    # on the host, from an unauthenticated endpoint.
    for failure, status in (
        (ArtifactNotFound, 404),
        (TenantUnknown, 404),
        (OverrideRejected, 409),
        (ArtifactInvalid, 500),
    ):
        app.add_exception_handler(failure, _as_json(status))

    @app.get("/capabilities")
    def list_capabilities() -> JSONResponse:
        return JSONResponse([summarise(a) for a in store.list_all()])

    @app.get("/capabilities/{name}")
    def get_capability(name: str, version: str | None = None) -> JSONResponse:
        return JSONResponse(summarise(store.load(name, version)))

    @app.get("/capabilities/{name}/artifact")
    def get_artifact(name: str, version: str | None = None) -> JSONResponse:
        """The whole capability, for a human reviewer or a diff."""
        artifact = store.load(name, version)
        return JSONResponse(artifact.model_dump(mode="json", exclude_none=True))

    @app.post("/capabilities/{name}:invoke")
    def invoke(name: str, request: InvokeRequest, version: str | None = None) -> JSONResponse:
        """Run a capability. This is the production path an agent triggers."""
        artifact = specialise(store.load(name, version), request.tenant, root=overrides_dir)
        if (wrong_surface := unrunnable_on_a_browser(artifact)) is not None:
            return JSONResponse({"error": wrong_surface}, status_code=409)

        # The declared types, applied. They are published to every calling
        # agent as this capability's contract and nothing enforced them:
        # `bind_parameters` checks `required` and `pattern` and then does
        # `str(value)`, and `arguments` is `dict[str, Any]`, so
        # {"product_code": ["S0", "2"]} was typed into the bank application as
        # the literal string "['S0', '2']". Names this capability does not
        # declare are left alone — `bind_parameters` reports those, and better.
        declared = {p.name: p for p in artifact.inputs}
        try:
            arguments = {
                name: declared[name].check(value) if name in declared else value
                for name, value in request.arguments.items()
            }
        except ValueError as mistyped:
            return JSONResponse({"error": str(mistyped)}, status_code=422)

        run_id = new_run_id("invoke")
        # Without opt-in there is no operator, so a blocked capability is
        # refused immediately rather than held open waiting for one.
        handler = (
            ConsoleEscalation(queue, timeout_s=request.escalation_timeout_s)
            if request.escalate
            else None
        )

        try:
            with (
                EvidenceRecorder(run_id, root=evidence_dir) as recorder,
                WebSurface(headed=headed, allowlist=allowlist) as surface,
            ):
                result = ReplayExecutor(
                    surface,
                    artifact,
                    recorder=recorder,
                    base_url=request.target,
                    gate=gate,
                    escalation=handler,
                ).run(arguments)
        except OSError as unrecordable:
            # `EvidenceRecorder.__init__` refuses a directory that already holds
            # a run, and can fail on a read-only or full disk. Either way the
            # answer is to refuse rather than to touch the application anyway: a
            # run nobody can audit afterwards is worse than a run that did not
            # happen.
            return JSONResponse(
                {"error": f"cannot record this run, so it was not started: {unrecordable}"},
                status_code=503,
            )

        # 200 for success and for a declared business outcome — both are the
        # capability working. Only a malfunction is an error status.
        status = 200 if result.status is not ReplayStatus.FAILED else 422
        return JSONResponse(result.to_dict(), status_code=status)

    console = create_console(queue)
    app.mount("/operator", console)
    return app


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    artifacts_dir: Path | str = "artifacts",
    policy_file: Path | str = "policy.toml",
    evidence_dir: Path | str = "runs",
    overrides_dir: Path | str = OVERRIDES_ROOT,
    headed: bool = False,
) -> None:
    import uvicorn

    uvicorn.run(
        create_api(
            artifacts_dir=artifacts_dir,
            policy_file=policy_file,
            evidence_dir=evidence_dir,
            overrides_dir=overrides_dir,
            headed=headed,
        ),
        host=host,
        port=port,
        log_level="info",
    )
