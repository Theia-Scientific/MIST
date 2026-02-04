#!/usr/bin/env python3

import cv2
import importlib.metadata
import numpy as np
import os
import pytest
import zipfile

from matplotlib.figure import Figure
from mist import __app_name__
from mist.cli import (
    app,
    correct_cv_image,
    create_tiles,
    decode_data,
    map_verbosity,
    NPY_MIME_TYPE,
    read_image_file,
    TIFF_MIME_TYPE,
    UnknownMimeTypeError,
)
from typer.testing import CliRunner
from ultralytics.utils.downloads import attempt_download_asset

runner = CliRunner()


@pytest.fixture(scope="session")
def assets(tmp_path_factory):
    return tmp_path_factory.mktemp("assets")


@pytest.fixture(scope="session")
def weights_file(assets):
    weights_file = attempt_download_asset(
        "weights/yolov8n-seg.pt", dir=assets, progress=False
    )
    return assets.joinpath(weights_file)


@pytest.fixture
def blank_image() -> np.ndarray:
    return np.zeros((4096, 4096, 3), dtype=np.uint8)


@pytest.fixture
def blank_grayscale_image() -> np.ndarray:
    return np.zeros((4096, 4096), dtype=np.uint8)


@pytest.fixture
def blank_float32_image() -> np.ndarray:
    return np.zeros((4096, 4096, 3), dtype=np.float32)


@pytest.fixture
def random_image() -> np.ndarray:
    return np.random.randint(255, size=(4096, 4096, 3), dtype=np.uint8)


@pytest.fixture
def random_float32_image() -> np.ndarray:
    return np.random.random((4096, 4096, 3)).astype(np.float32)


@pytest.fixture
def blank_png(blank_image, tmp_path):
    png_file = tmp_path.joinpath("image.png")
    cv2.imwrite(str(png_file), blank_image)
    yield png_file


@pytest.fixture
def blank_npy(blank_image, tmp_path):
    npy_file = tmp_path.joinpath("image.npy")
    np.save(npy_file, blank_image)
    yield npy_file


@pytest.fixture
def blank_tif(blank_image, tmp_path):
    tif_file = tmp_path.joinpath("image.tif")
    cv2.imwrite(str(tif_file), blank_image)
    yield tif_file


@pytest.fixture
def unknown_image_file(blank_png):
    unknown_image_file = blank_png.with_suffix(".XXYY")
    os.rename(blank_png, unknown_image_file)
    yield unknown_image_file


@pytest.fixture
def zip_file(blank_png, blank_npy, blank_tif, tmp_path):
    zip_path = tmp_path.joinpath("images.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(blank_png)
        zf.write(blank_npy)
        zf.write(blank_tif)
    yield zip_path


def test_read_image_file(blank_image, blank_png):
    actual = read_image_file(blank_png)
    assert (actual == blank_image).all()


def test_correct_cv_image(blank_image):
    actual = correct_cv_image(blank_image)
    assert (actual == blank_image).all()


def test_correct_cv_image_grayscale(blank_image, blank_grayscale_image):
    actual = correct_cv_image(blank_grayscale_image)
    assert (actual == blank_image).all()


def test_correct_cv_image_normalized_blank(blank_image, blank_float32_image):
    actual = correct_cv_image(blank_float32_image)
    assert (actual == blank_image).all()


def test_correct_cv_image_normalized_random(random_float32_image):
    actual = correct_cv_image(random_float32_image)
    assert actual.dtype == np.uint8


def test_read_image_file_fail_unknown_mime_type(unknown_image_file):
    with pytest.raises(UnknownMimeTypeError):
        read_image_file(unknown_image_file)


def test_decode_data_npy(blank_image, blank_npy):
    with open(blank_npy, "rb+") as f:
        data = f.read()
    actual = decode_data(data, NPY_MIME_TYPE)
    assert (actual == blank_image).all()


def test_decode_data_tif(blank_image, blank_tif):
    with open(blank_tif, "rb+") as f:
        data = f.read()
    actual = decode_data(data, TIFF_MIME_TYPE)
    assert (actual == blank_image).all()


def test_map_verbosity_false():
    actual = map_verbosity(False)
    assert actual == "INFO"


def test_map_verbosity_true():
    actual = map_verbosity(True)
    assert actual == "DEBUG"


def test_create_tiles_show(mocker, blank_image):
    def mock_figure(*args, **kwargs):
        _ = args
        _ = kwargs
        return mocker.MagicMock(spec=Figure)

    mocker.patch("matplotlib.pyplot.figure", mock_figure)
    create_tiles(blank_image)


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
