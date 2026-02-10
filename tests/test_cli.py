#!/usr/bin/env python3

import importlib.metadata
import pytest
import zipfile

from mist import __app_name__
from mist.cli import (
    app,
    map_verbosity,
)
from typer.testing import CliRunner

runner = CliRunner()

@pytest.fixture
def zip_file(blank_png, blank_npy, blank_tif, tmp_path):
    zip_path = tmp_path.joinpath("images.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(blank_png)
        zf.write(blank_npy)
        zf.write(blank_tif)
    yield zip_path


def test_map_verbosity_false():
    actual = map_verbosity(False)
    assert actual == "INFO"


def test_map_verbosity_true():
    actual = map_verbosity(True)
    assert actual == "DEBUG"


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_app_version():
    version = importlib.metadata.version(__app_name__)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} {version}" in result.stdout


def test_app_image(blank_png, weights_file):
    result = runner.invoke(
        app, ["--device=cpu", "--no-show", str(weights_file), str(blank_png)]
    )
    assert result.exit_code == 0


def test_app_directory(tmp_path, weights_file):
    result = runner.invoke(
        app, ["--device=cpu", "--no-show", str(weights_file), str(tmp_path)]
    )
    assert result.exit_code == 0


def test_app_zip(zip_file, weights_file):
    result = runner.invoke(
        app, ["--device=cpu", "--no-show", str(weights_file), str(zip_file)]
    )
    assert result.exit_code == 0


def test_app_visualize(mocker, blank_png, weights_file):
    def mock_visualizing_run(*args, **kwargs):
        _ = args
        _ = kwargs

        return None

    mocker.patch("mist.visualizing.run", mock_visualizing_run)
    result = runner.invoke(
        app, ["--device=cpu", str(weights_file), str(blank_png)]
    )
    assert result.exit_code == 0

   
def test_app_visualize_show_tiles(mocker, blank_png, weights_file):
    def mock_visualizing_run(*args, **kwargs):
        _ = args
        _ = kwargs

        return None

    mocker.patch("mist.visualizing.run", mock_visualizing_run)
    result = runner.invoke(
        app, ["--device=cpu", "--show-tiles", str(weights_file), str(blank_png)]
    )
    assert result.exit_code == 0
   
