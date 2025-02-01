#!/usr/bin/env python3

import importlib.metadata
import logging
import os
import sys
import typer

from pathlib import Path
from tstiler import __app_name__
from typing import List, Optional, Tuple

LOGGER: logging.Logger = logging.getLogger(__name__)

PREFIX: str = f"{__app_name__.upper()}"

app = typer.Typer(pretty_exceptions_show_locals=False)

def map_verbosity(enabled: bool) -> str:
    if enabled:
        return "DEBUG"
    else:
        return "INFO"


def version_callback(value: bool):
    if value:
        version = importlib.metadata.version(__app_name__)
        print(f"{__app_name__} {version}")
        raise typer.Exit()


@app.callback()
def main(
    config_path: Optional[List[Path]] = typer.Option(
        None, "--config-path", "-c", help="Configuration file path"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print debugging statements to STDOUT.",
        envvar=f"{PREFIX}_VERBOSE",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Prints the version to STDOUT",
        callback=version_callback,
        is_eager=True,
    ),
):
    logging.basicConfig(level=map_verbosity(verbose))
    LOGGER.debug(f"version={version}")
    LOGGER.debug(f"config_path={config_path}")
    if config_path is None:
        config_paths = [
            f"/etc/{__app_name__}",
            os.path.join(
                os.getenv("XDG_CONFIG_HOME", Path.home().joinpath(".config")),
                __app_name__,
            ),
        ]
        dotenv_config = find_dotenv()
        LOGGER.debug(f"dotenv_config={dotenv_config}")
        if dotenv_config:
            config_paths.append(dotenv_config)
    else:
        config_paths = [c.expanduser().resolve() for c in config_path]
    LOGGER.debug(f"config_paths={config_paths}")
    configs = {}
    for path in config_paths:
        new_configs = dotenv_values(dotenv_path=path)
        configs = configs | new_configs
    LOGGER.debug(f"configs={configs}")
    for k, v in configs.items():
        if v is not None:
            os.environ[k] = v


if __name__ == "__main__":
    app(prog_name=__app_name__)
