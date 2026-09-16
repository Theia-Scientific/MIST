#!/usr/bin/env python3

import importlib.metadata
import pytest
import shutil
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
def text_file(tmp_path: Path) -> Path:
    txt_file = tmp_path.joinpath("test.txt")
    with open(txt_file, "+w") as fp:
        _ = fp.write("Hello World")
    return txt_file


@pytest.fixture
def zip_file(
    blank_jpg: Path, blank_png: Path, blank_npy: Path, blank_tif: Path, tmp_path: Path
) -> Path:
    zip_path = tmp_path.joinpath("images.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        _ = zf.write(blank_jpg)
        _ = zf.write(blank_png)
        _ = zf.write(blank_npy)
        _ = zf.write(blank_tif)
    return zip_path


@pytest.fixture
def dir_with_images(
    blank_jpg: Path, blank_png: Path, blank_npy: Path, blank_tif: Path, tmp_path: Path
) -> Path:
    _ = shutil.move(blank_jpg, tmp_path.joinpath("image0.jpg"))
    _ = shutil.move(blank_png, tmp_path.joinpath("image1.png"))
    _ = shutil.move(blank_npy, tmp_path.joinpath("image2.npy"))
    _ = shutil.move(blank_tif, tmp_path.joinpath("image3.tif"))
    return tmp_path


@pytest.fixture
def dir_with_images_and_text(
    blank_jpg: Path,
    blank_png: Path,
    blank_npy: Path,
    blank_tif: Path,
    text_file: Path,
    tmp_path: Path,
) -> Path:
    _ = shutil.move(blank_jpg, tmp_path.joinpath("image0.jpg"))
    _ = shutil.move(blank_png, tmp_path.joinpath("image1.png"))
    _ = shutil.move(blank_npy, tmp_path.joinpath("image2.npy"))
    _ = shutil.move(blank_tif, tmp_path.joinpath("image3.tif"))
    _ = shutil.move(text_file, tmp_path.joinpath("text.txt"))
    return tmp_path


def test_map_verbosity():
    assert map_verbosity(0) == "WARNING"
    assert map_verbosity(1) == "INFO"
    assert map_verbosity(2) == "DEBUG"
    assert map_verbosity(3) == "DEBUG"


def test_expand_sources_with_single_supported_file(blank_png: Path):
    actual = expand_sources([blank_png])
    assert len(actual) == 1
    assert blank_png in actual


def test_expand_sources_with_multiple_supported_files(blank_png: Path, bus_jpg: Path):
    actual = expand_sources([blank_png, bus_jpg])
    assert len(actual) == 2
    assert blank_png in actual
    assert bus_jpg in actual


def test_expand_sources_with_no_supported_file(text_file: Path):
    actual = expand_sources([text_file])
    assert len(actual) == 0


def test_expand_sources_with_zip_file(zip_file: Path):
    paths = expand_sources([zip_file])
    assert len(paths) == 4
    for path in paths:
        assert isinstance(path, Path)


def test_expand_sources_with_directory(dir_with_images: Path):
    paths = expand_sources([dir_with_images])
    assert len(paths) == 4
    for path in paths:
        assert isinstance(path, Path)


def test_expand_sources_with_directory_unsupported(dir_with_images_and_text: Path):
    paths = expand_sources([dir_with_images_and_text])
    assert len(paths) == 4
    for path in paths:
        assert isinstance(path, Path)


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_app_version():
    version = importlib.metadata.version(__app_name__)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} {version}" in result.stdout


def test_app_image(blank_png: Path, tmp_path: Path, weights_file: Path):
    result = runner.invoke(
        app,
        ["--device=cpu", "--output", str(tmp_path), str(weights_file), str(blank_png)],
    )
    assert result.exit_code == 0


def test_app_directory(dir_with_images: Path, tmp_path: Path, weights_file: Path):
    result = runner.invoke(
        app,
        [
            "--device=cpu",
            "--output",
            str(tmp_path),
            str(weights_file),
            str(dir_with_images),
        ],
    )
    assert result.exit_code == 0


def test_app_zip(zip_file: Path, tmp_path: Path, weights_file: Path):
    result = runner.invoke(
        app,
        ["--device=cpu", "--output", str(tmp_path), str(weights_file), str(zip_file)],
    )
    assert result.exit_code == 0
