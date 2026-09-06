"""Command-line entry point.

Subcommands are added as their milestones land:
    discover      M4  run the LLM loop against a live surface
    run           M6  replay a saved capability with typed parameters
    capabilities  M10 list and inspect the capability catalog
    serve         M9  operator console and catalog API
"""

from __future__ import annotations

import json
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


MERIDIAN_OUTCOMES = [
    ("MEMBER_NOT_FOUND", "MEMBER_NOT_FOUND", "No member on file for the supplied ID."),
    ("PERMISSION_DENIED", "PERMISSION_DENIED", "Teller authority is insufficient."),
    (
        "VALIDATION_REJECTED",
        "VALIDATION_REJECTED",
        "The application rejected the submitted values.",
    ),
]


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
        _synthesise_and_save(result, save_as)


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
    headed: Annotated[bool, typer.Option("--headed", help="Show the browser window.")] = False,
    evidence_dir: Annotated[Path, typer.Option("--evidence-dir")] = Path("evidence"),
) -> None:
    """Replay a saved capability. No model is involved.

    This is the path an AI agent triggers in production.
    """
    from replay.artifact import ArtifactNotFound, ArtifactStore
    from replay.engine import ReplayExecutor
    from replay.evidence import EvidenceRecorder, new_run_id
    from replay.surface import WebSurface

    arguments: dict[str, str] = {}
    for pair in param or []:
        key, sep, value = pair.partition("=")
        if not sep:
            typer.secho(f"--param expects key=value, got {pair!r}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)
        arguments[key.strip()] = value

    try:
        artifact = ArtifactStore().load(name, capability_version)
    except ArtifactNotFound as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    run_id = new_run_id("replay")
    typer.secho(f"run {run_id}  capability {artifact.ref}", fg=typer.colors.CYAN)

    with (
        EvidenceRecorder(run_id, root=evidence_dir) as recorder,
        WebSurface(headed=headed) as surface,
    ):
        result = ReplayExecutor(surface, artifact, recorder=recorder, base_url=target).run(
            arguments
        )

    _report(result)
    raise typer.Exit(code=0 if result.ok else 1)


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

    typer.echo(f"tiers:     {json.dumps(result.locator_tiers)}")
    if result.degraded_steps:
        typer.secho(
            f"degraded:  {result.degraded_steps} resolved below tier 1",
            fg=typer.colors.YELLOW,
        )
    typer.echo(f"duration:  {result.duration_ms} ms")
    typer.echo(f"evidence:  {result.evidence_dir}")


@app.command()
def synthesize(
    evidence_dir: Annotated[Path, typer.Argument(help="A discovery run's evidence directory.")],
    name: Annotated[str, typer.Option("--name", "-n", help="Capability name.")],
    version: Annotated[
        str, typer.Option("--version", help="Semver for this capability.")
    ] = "1.0.0",
) -> None:
    """Distil a recorded discovery run into a capability artifact.

    Reads only the run result, never the transcript. Re-runnable against
    committed evidence, so regenerating the artifact costs nothing.
    """
    from replay.agent.loop import DiscoveryResult

    payload = json.loads((evidence_dir / "result.json").read_text())
    _synthesise_and_save(DiscoveryResult.from_dict(payload), name, version=version)


def _synthesise_and_save(result, name: str, *, version: str = "1.0.0") -> None:
    from replay.artifact import ArtifactStore
    from replay.synthesis import SynthesisError, declare_outcome
    from replay.synthesis import synthesize as distil

    outcomes = [declare_outcome(*row) for row in MERIDIAN_OUTCOMES]
    try:
        synthesis = distil(
            result,
            name=name,
            version=version,
            product="MERIDIAN CORE",
            outcomes=outcomes,
        )
    except SynthesisError as exc:
        typer.secho(f"synthesis failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    for note in synthesis.notes:
        typer.secho(f"note:       {note}", fg=typer.colors.YELLOW)

    path = ArtifactStore().save(synthesis.artifact, overwrite=True)
    typer.secho(f"capability: {synthesis.artifact.ref} → {path}", fg=typer.colors.GREEN)
    typer.echo(f"checkpoint: {synthesis.checkpoint_text!r}")


if __name__ == "__main__":
    app()
