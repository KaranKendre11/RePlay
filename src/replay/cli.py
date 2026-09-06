"""Command-line entry point.

Subcommands are added as their milestones land:
    discover      M4  run the LLM loop against a live surface
    run           M6  replay a saved capability with typed parameters
    capabilities  M10 list and inspect the capability catalog
    serve         M9  operator console and catalog API
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from replay import __version__

app = typer.Typer(
    name="replay",
    help="Record-once, replay-many computer-use automation.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Root callback.

    Present so Typer keeps subcommand dispatch while only one command exists,
    and as the future home of global options such as --config.
    """
    # Loaded here rather than at import time so tests and library users are not
    # silently affected by whatever .env happens to be lying around.
    load_dotenv(Path.cwd() / ".env")


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


def _load_profile(path: Path | None) -> dict:
    """Product knowledge a single happy-path run cannot discover.

    The business outcomes a capability can legitimately reach, the interstitials
    worth recovering from, and the screen text that means the session died
    rather than the step being wrong. All three are properties of the product,
    not of the run, so a run cannot infer them and synthesis refuses to invent
    them.

    Read from a file beside the application rather than held here. This module
    should not know which products exist, and an engine carrying one vendor's
    error strings classifies correctly against that vendor and silently stops
    classifying against every other.
    """
    from replay.synthesis import declare_interstitial, declare_outcome

    if path is None:
        return {}
    if not path.exists():
        typer.secho(f"no profile at {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    raw = tomllib.loads(path.read_text())
    return {
        "product": raw.get("product", "unknown"),
        "outcomes": [
            declare_outcome(o["code"], o["text"], o["message"]) for o in raw.get("outcome", [])
        ],
        "recoveries": [
            declare_interstitial(r["when_text"], r["link_name"], r.get("frame_path"))
            for r in raw.get("recovery", [])
        ],
        "session_lost_markers": raw.get("session_lost_markers", []),
        "application_error_markers": raw.get("application_error_markers", []),
    }


#: Where the profile for the bundled target application lives.
PROFILE_HELP = "Product knowledge to attach at review, e.g. targets/meridian/review.toml."


@app.command()
def discover(
    goal: Annotated[
        str, typer.Option("--goal", "-g", help="What to accomplish, in plain language.")
    ],
    target: Annotated[
        str, typer.Option("--target", "-t", help="Entry point URL of the application.")
    ],
    model: Annotated[str | None, typer.Option("--model", help="Override REPLAY_MODEL.")] = None,
    max_steps: Annotated[
        int, typer.Option("--max-steps", help="Stop after this many decisions.")
    ] = 25,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Wall-clock budget in seconds.")
    ] = 300.0,
    headed: Annotated[bool, typer.Option("--headed", help="Show the browser window.")] = False,
    vision: Annotated[
        bool, typer.Option("--vision/--no-vision", help="Send screenshots to the model.")
    ] = True,
    evidence_dir: Annotated[Path, typer.Option("--evidence-dir")] = Path("evidence"),
    save_as: Annotated[
        str | None,
        typer.Option("--save-as", help="Synthesise the run into a capability with this name."),
    ] = None,
    profile: Annotated[Path | None, typer.Option("--profile", help=PROFILE_HELP)] = None,
) -> None:
    """Run the LLM-driven discovery loop against a live application.

    This is the only path that spends model tokens. Everything downstream —
    replay, the capability catalog — runs without a model.
    """
    from replay.agent import DiscoveryLoop, LLMError, OpenAIClient, StopReason
    from replay.evidence import EvidenceRecorder, new_run_id
    from replay.surface import WebSurface

    try:
        llm = OpenAIClient(model=model)
    except LLMError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    run_id = new_run_id("discovery")
    typer.secho(f"run {run_id}  model {llm.name}", fg=typer.colors.CYAN)

    with (
        EvidenceRecorder(run_id, root=evidence_dir) as recorder,
        WebSurface(headed=headed) as surface,
    ):
        loop = DiscoveryLoop(
            surface,
            llm,
            recorder,
            max_steps=max_steps,
            timeout_s=timeout,
            vision=vision,
        )
        result = loop.run(goal, target)

    colour = typer.colors.GREEN if result.succeeded else typer.colors.YELLOW
    typer.secho(f"\n{result.status.value}: {result.summary or result.reason}", fg=colour)
    if result.parameters:
        typer.echo(f"parameters: {json.dumps(result.parameters)}")
    if result.outputs:
        typer.echo(f"outputs:    {json.dumps(result.outputs)}")
    typer.echo(f"evidence:   {result.evidence_dir}")
    for warning in result.warnings:
        typer.secho(f"warning:    {warning}", fg=typer.colors.YELLOW)

    if result.status is not StopReason.GOAL_MET:
        raise typer.Exit(code=1)

    if save_as:
        _synthesise_and_save(result, save_as, profile=profile)


@app.command(name="run")
def run_capability(
    name: Annotated[str, typer.Argument(help="Capability name.")],
    param: Annotated[
        list[str] | None,
        typer.Option("--param", "-p", help="Argument as key=value. Repeatable."),
    ] = None,
    capability_version: Annotated[
        str | None, typer.Option("--capability-version", help="Pin a version.")
    ] = None,
    target: Annotated[
        str | None,
        typer.Option("--target", "-t", help="Run against this origin instead of the recorded one."),
    ] = None,
    tenant: Annotated[
        str | None,
        typer.Option("--tenant", help="Apply this tenant's overrides to the base capability."),
    ] = None,
    headed: Annotated[bool, typer.Option("--headed", help="Show the browser window.")] = False,
    evidence_dir: Annotated[Path, typer.Option("--evidence-dir")] = Path("evidence"),
    policy_file: Annotated[Path, typer.Option("--policy", help="Allowlist file.")] = Path(
        "policy.toml"
    ),
    allow_risky: Annotated[
        bool, typer.Option("--allow-risky", help="Permit steps that change state.")
    ] = False,
    allow_irreversible: Annotated[
        bool,
        typer.Option(
            "--allow-irreversible",
            help="Permit irreversible steps. Blocked by default; prefer human escalation.",
        ),
    ] = False,
    escalate: Annotated[
        bool,
        typer.Option("--escalate", help="Open the operator console and route blocks to a human."),
    ] = False,
    label: Annotated[
        str | None,
        typer.Option("--label", help="Name this run's evidence directory, instead of a timestamp."),
    ] = None,
    console_port: Annotated[int, typer.Option("--console-port")] = 8765,
) -> None:
    """Replay a saved capability. No model is involved.

    This is the path an AI agent triggers in production.
    """
    from replay.artifact import ArtifactNotFound, ArtifactStore, specialise
    from replay.engine import ReplayExecutor
    from replay.escalation import ConsoleEscalation, InterventionQueue, serve_console
    from replay.evidence import EvidenceRecorder, new_run_id
    from replay.policy import RiskGate
    from replay.surface import WebSurface

    arguments: dict[str, str] = {}
    for pair in param or []:
        key, sep, value = pair.partition("=")
        if not sep:
            typer.secho(f"--param expects key=value, got {pair!r}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)
        arguments[key.strip()] = value

    try:
        artifact = specialise(ArtifactStore().load(name, capability_version), tenant)
    except ArtifactNotFound as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    allowlist = _load_allowlist(policy_file)
    gate = RiskGate(allow_risky=allow_risky, allow_irreversible=allow_irreversible)

    handler = None
    if escalate:
        queue = InterventionQueue()
        serve_console(queue, port=console_port)
        handler = ConsoleEscalation(queue)
        typer.secho(f"operator console: http://127.0.0.1:{console_port}", fg=typer.colors.MAGENTA)

    run_id = f"replay-{label}" if label else new_run_id("replay")
    scope = f"  tenant {tenant}" if tenant else ""
    typer.secho(f"run {run_id}  capability {artifact.ref}{scope}", fg=typer.colors.CYAN)

    with (
        EvidenceRecorder(run_id, root=evidence_dir) as recorder,
        # Headed whenever a human might be asked to take over: they act in the
        # real window, on the same session.
        WebSurface(headed=headed or escalate, allowlist=allowlist) as surface,
    ):
        result = ReplayExecutor(
            surface,
            artifact,
            recorder=recorder,
            base_url=target,
            gate=gate,
            escalation=handler,
        ).run(arguments)

    _report(result)
    raise typer.Exit(code=0 if result.ok else 1)


def _load_allowlist(path: Path):
    """Load the allowlist, refusing to run without one.

    A missing policy file must not mean "permit everything". The failure mode
    of a misconfiguration should be a refusal.
    """
    from replay.policy import Allowlist

    if not path.exists():
        typer.secho(
            f"no policy file at {path}; refusing to run without an allowlist",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    return Allowlist.from_file(path)


def _report(result) -> None:
    from replay.engine import ReplayStatus

    colour = {
        ReplayStatus.SUCCESS: typer.colors.GREEN,
        ReplayStatus.BUSINESS_OUTCOME: typer.colors.BLUE,
        ReplayStatus.FAILED: typer.colors.RED,
    }[result.status]
    typer.secho(f"\n{result.status.value}", fg=colour, bold=True)

    if result.outputs:
        typer.echo(f"outputs:   {json.dumps(result.outputs)}")
    if result.outcome:
        typer.echo(f"outcome:   {result.outcome.code} — {result.outcome.message}")
        typer.echo(f"           detected at step {result.outcome.detected_at_step}")
    if result.failure:
        typer.echo(f"failed at: {result.failure.step_id} ({result.failure.failure_class.value})")
        typer.echo(f"expected:  {result.failure.expected}")
        typer.echo(f"observed:  {result.failure.observed[:200]}")

    recovered = [s for s in result.steps if s.recovered]
    if recovered:
        for step in recovered:
            typer.secho(
                f"recovered: {step.step_id} — {', '.join(step.recovered)}",
                fg=typer.colors.YELLOW,
            )

    typer.echo(f"tiers:     {json.dumps(result.locator_tiers)}")
    if result.drifting_steps:
        typer.secho(
            f"DRIFT:     {result.drifting_steps} resolved worse than when recorded",
            fg=typer.colors.RED,
        )
    elif result.degraded_steps:
        typer.secho(
            f"degraded:  {result.degraded_steps} resolved below tier 1, as recorded",
            fg=typer.colors.YELLOW,
        )
    if result.escalation:
        escalation = result.escalation
        typer.secho(
            f"escalated: {escalation['reason']} → {escalation['resolution']}",
            fg=typer.colors.MAGENTA,
        )
        for action in escalation["human_actions"]:
            typer.echo(f"  operator: {action['kind']} {action['label']}")

    typer.echo(f"duration:  {result.duration_ms} ms")
    typer.echo(f"evidence:  {result.evidence_dir}")


@app.command()
def capabilities(
    name: Annotated[str | None, typer.Argument(help="Show one capability in detail.")] = None,
) -> None:
    """List the saved capabilities, or show one.

    The same view an agent gets from GET /capabilities — derived from the
    artifacts directory, so there is no registry that can disagree with it.
    """
    from replay.api import summarise
    from replay.artifact import ArtifactNotFound, ArtifactStore

    store = ArtifactStore()

    if name:
        try:
            typer.echo(json.dumps(summarise(store.load(name)), indent=2))
        except ArtifactNotFound as missing:
            typer.secho(str(missing), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from missing
        return

    found = store.list_all()
    if not found:
        typer.secho("no capabilities saved yet", fg=typer.colors.YELLOW)
        return

    for artifact in found:
        risk = artifact.max_step_risk.value
        colour = typer.colors.RED if risk == "irreversible" else typer.colors.GREEN
        typer.secho(f"{artifact.ref}", fg=typer.colors.CYAN, nl=False)
        typer.echo(f"  {artifact.title}")
        typer.secho(f"    risk={risk}", fg=colour, nl=False)
        typer.echo(
            f"  approval={artifact.reliability.approval.value}"
            f"  args=[{', '.join(p.name for p in artifact.inputs)}]"
            f"  returns=[{', '.join(o.name for o in artifact.outputs)}]"
        )


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    policy_file: Annotated[Path, typer.Option("--policy")] = Path("policy.toml"),
    headed: Annotated[
        bool, typer.Option("--headed", help="Show the browser, so an operator can take over.")
    ] = False,
) -> None:
    """Serve the capability catalog and the operator console.

    Catalog at /capabilities, console at /operator. One process: a
    single-operator handoff needs no more, and the brief is explicit that
    building scaling infrastructure is not rewarded.
    """
    from replay.api import serve as run_server

    _load_allowlist(policy_file)
    typer.secho(f"catalog:  http://{host}:{port}/capabilities", fg=typer.colors.CYAN)
    typer.secho(f"operator: http://{host}:{port}/operator", fg=typer.colors.MAGENTA)
    run_server(host=host, port=port, policy_file=policy_file, headed=headed)


@app.command()
def synthesize(
    evidence_dir: Annotated[Path, typer.Argument(help="A discovery run's evidence directory.")],
    name: Annotated[str, typer.Option("--name", "-n", help="Capability name.")],
    version: Annotated[
        str, typer.Option("--version", help="Semver for this capability.")
    ] = "1.0.0",
    profile: Annotated[Path | None, typer.Option("--profile", help=PROFILE_HELP)] = None,
) -> None:
    """Distil a recorded discovery run into a capability artifact.

    Reads only the run result, never the transcript. Re-runnable against
    committed evidence, so regenerating the artifact costs nothing.
    """
    from replay.agent.loop import DiscoveryResult

    payload = json.loads((evidence_dir / "result.json").read_text())
    _synthesise_and_save(
        DiscoveryResult.from_dict(payload), name, version=version, profile=profile
    )


def _synthesise_and_save(
    result, name: str, *, version: str = "1.0.0", profile: Path | None = None
) -> None:
    from replay.artifact import ArtifactStore
    from replay.synthesis import SynthesisError
    from replay.synthesis import synthesize as distil

    try:
        synthesis = distil(result, name=name, version=version, **_load_profile(profile))
    except SynthesisError as exc:
        typer.secho(f"synthesis failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    for note in synthesis.notes:
        typer.secho(f"note:       {note}", fg=typer.colors.YELLOW)

    if synthesis.needs_outcomes:
        typer.secho(
            "note:       no business outcomes declared, so this capability tells a "
            "caller nothing about how it can legitimately not-succeed; pass --profile",
            fg=typer.colors.YELLOW,
        )

    path = ArtifactStore().save(synthesis.artifact, overwrite=True)
    typer.secho(f"capability: {synthesis.artifact.ref} → {path}", fg=typer.colors.GREEN)
    typer.echo(f"checkpoint: {synthesis.checkpoint_text!r}")


if __name__ == "__main__":
    app()
