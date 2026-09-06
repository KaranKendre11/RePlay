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

    if result.status is not StopReason.GOAL_MET:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
