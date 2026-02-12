#!/usr/bin/env python3

import importlib.metadata
import pytest
import zipfile

from mist import __app_name__
from mist.cli import (
    app,
    expand_sources,
    map_verbosity,
)
from pathlib import Path
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


@pytest.fixture
def mock_visualizing_run(mocker):
    def mock_visualizing_run(*args, **kwargs):
        _ = args
        _ = kwargs

        return None

    mocker.patch("mist.visualizing.run", mock_visualizing_run)


def test_map_verbosity_false():
    actual = map_verbosity(False)
    assert actual == "INFO"


def test_map_verbosity_true():
    actual = map_verbosity(True)
    assert actual == "DEBUG"


def test_expand_sources_with_single_supported_file(blank_png):
    actual = expand_sources([blank_png])
    assert len(actual) == 1
    assert blank_png in actual


def test_expand_sources_with_multiple_supported_files(blank_png, bus_jpg):
    actual = expand_sources([blank_png, bus_jpg])
    assert len(actual) == 2
    assert blank_png in actual
    assert bus_jpg in actual


def test_expand_sources_with_no_supported_file(tmp_path):
    txt_file = tmp_path.joinpath("test.txt")
    with open(txt_file, "+w") as fp:
        fp.write("Hello World")
    actual = expand_sources([txt_file])
    assert len(actual) == 0


def test_expand_sources_with_zip_file(zip_file):
    actual = expand_sources([zip_file])
    assert len(actual) == 3
    for path in actual:
        assert isinstance(path, Path)


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


def test_app_visualize(mock_visualizing_run, blank_png, weights_file):
    _ = mock_visualizing_run
    result = runner.invoke(
        app, ["--device=cpu", str(weights_file), str(blank_png)]
    )
    assert result.exit_code == 0

   
def test_app_visualize_show_tiles(mock_visualizing_run, blank_png, weights_file):
    _ = mock_visualizing_run
    result = runner.invoke(
        app, ["--device=cpu", "--show-tiles", str(weights_file), str(blank_png)]
    )
    assert result.exit_code == 0
   
