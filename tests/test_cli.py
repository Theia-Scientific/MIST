#!/usr/bin/env python3

import cv2
import importlib.metadata
import numpy as np
import os
import pytest
import torch

from tstiler import __app_name__
from tstiler.cli import (
    app,
    correct_cv_image,
    decode_data,
    main,
    map_verbosity,
    NPY_MIME_TYPE,
    read_image_file,
    UnknownMimeTypeError,
)
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def weights_file(tmp_path):
    weights_file = tmp_path.joinpath("weights_file.pt")
    x = torch.tensor([0, 1, 2, 3, 4])
    torch.save(x, weights_file)
    yield weights_file


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
def unknown_image_file(blank_png):
    unknown_image_file = blank_png.with_suffix(".abc")
    os.rename(blank_png, unknown_image_file)
    yield unknown_image_file


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


def test_decode_data(blank_image, blank_npy):
    with open(blank_npy, "rb+") as f:
        data = f.read()
    actual = decode_data(data, NPY_MIME_TYPE)
    assert (actual == blank_image).all()


def test_map_verbosity_false():
    actual = map_verbosity(False)
    assert actual == "INFO"


def test_map_verbosity_true():
    actual = map_verbosity(True)
    assert actual == "DEBUG"


def test_main(blank_png, weights_file):
    main(weights_file, [blank_png], verbose=False, version=False)


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_app_version():
    version = importlib.metadata.version(__app_name__)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} {version}" in result.stdout
