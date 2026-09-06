"""Command-line entry point.

Subcommands are added as their milestones land:
    discover      M4  run the LLM loop against a live surface
    run           M6  replay a saved capability with typed parameters
    capabilities  M10 list and inspect the capability catalog
    serve         M9  operator console and catalog API
"""

import typer

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


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
