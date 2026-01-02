from typer import Typer

from projective_geometry import entrypoints
from projective_geometry.entrypoints import register_entrypoints

cli_app = Typer()

register_entrypoints(entrypoints, cli_app.command())


# Program entry point redirection
if __name__ == "__main__":
    cli_app()

