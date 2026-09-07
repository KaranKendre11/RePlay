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

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from replay.artifact import ArtifactNotFound, ArtifactStore, invocation_schema, specialise
from replay.artifact.overrides import OverrideRejected
from replay.artifact.schema import CapabilityArtifact
from replay.engine import ReplayExecutor, ReplayStatus
from replay.escalation import ConsoleEscalation, InterventionQueue
from replay.escalation.console import create_console
from replay.evidence import EvidenceRecorder, new_run_id
from replay.policy import Allowlist, RiskGate
from replay.surface import WebSurface


class InvokeRequest(BaseModel):
    """One call. Arguments are checked against the capability's own schema."""

    arguments: dict[str, Any] = Field(default_factory=dict)
    target: str | None = Field(
        default=None,
        description="Run against this origin instead of the recorded one.",
    )
    tenant: str | None = Field(
        default=None,
        description="Apply this tenant's overrides. The capability's contract is unchanged.",
    )
    allow_risky: bool = False
    allow_irreversible: bool = False
    escalate: bool = Field(
        default=False,
        description=(
            "Route blocks to a human operator and wait. Off by default: an agent "
            "calling an API expects an answer, not a fifteen-minute hold while "
            "someone is found."
        ),
    )
    escalation_timeout_s: float = Field(default=300.0, gt=0)


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
    queue: InterventionQueue | None = None,
    headed: bool = False,
    allowlist: Allowlist | None = None,
) -> FastAPI:
    store = ArtifactStore(artifacts_dir)
    # Loaded from the shipped policy unless a caller supplies one. Tests bind to
    # an ephemeral port, which the production policy rightly does not allow.
    allowlist = allowlist if allowlist is not None else Allowlist.from_file(policy_file)
    queue = queue if queue is not None else InterventionQueue()

    app = FastAPI(
        title="RePlay capability catalog",
        description="Saved capabilities, callable by name with typed arguments.",
        version="1.0",
    )

    @app.get("/capabilities")
    def list_capabilities() -> JSONResponse:
        return JSONResponse([summarise(a) for a in store.list_all()])

    @app.get("/capabilities/{name}")
    def get_capability(name: str, version: str | None = None) -> JSONResponse:
        try:
            return JSONResponse(summarise(store.load(name, version)))
        except ArtifactNotFound as missing:
            return JSONResponse({"error": str(missing)}, status_code=404)

    @app.get("/capabilities/{name}/artifact")
    def get_artifact(name: str, version: str | None = None) -> JSONResponse:
        """The whole capability, for a human reviewer or a diff."""
        try:
            artifact = store.load(name, version)
        except ArtifactNotFound as missing:
            return JSONResponse({"error": str(missing)}, status_code=404)
        return JSONResponse(artifact.model_dump(mode="json", exclude_none=True))

    @app.post("/capabilities/{name}:invoke")
    def invoke(name: str, request: InvokeRequest, version: str | None = None) -> JSONResponse:
        """Run a capability. This is the production path an agent triggers."""
        try:
            artifact = specialise(store.load(name, version), request.tenant)
        except ArtifactNotFound as missing:
            return JSONResponse({"error": str(missing)}, status_code=404)
        except OverrideRejected as rejected:
            return JSONResponse({"error": str(rejected)}, status_code=409)

        run_id = new_run_id("invoke")
        gate = RiskGate(
            allow_risky=request.allow_risky,
            allow_irreversible=request.allow_irreversible,
        )
        # Without opt-in there is no operator, so a blocked capability is
        # refused immediately rather than held open waiting for one.
        handler = (
            ConsoleEscalation(queue, timeout_s=request.escalation_timeout_s)
            if request.escalate
            else None
        )

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
            ).run(request.arguments)

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
    headed: bool = False,
) -> None:
    import uvicorn

    uvicorn.run(
        create_api(
            artifacts_dir=artifacts_dir,
            policy_file=policy_file,
            evidence_dir=evidence_dir,
            headed=headed,
        ),
        host=host,
        port=port,
        log_level="info",
    )
