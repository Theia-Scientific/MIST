#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
import os
import pytest

from mist.utils import (
    ImageDecodeError,
    correct_cv_image,
    decode_data,
    NPY_MIME_TYPE,
    read_image_file,
    TIFF_MIME_TYPE,
    UnsupportedImageFile,
)
from pathlib import Path


@pytest.fixture
def blank_float32_image() -> npt.NDArray[np.float32]:
    return np.zeros((4096, 4096, 3), dtype=np.float32)


@pytest.fixture
def blank_grayscale_image() -> npt.NDArray[np.uint8]:
    return np.zeros((4096, 4096), dtype=np.uint8)


@pytest.fixture
def random_image() -> npt.NDArray[np.uint8]:
    return np.random.randint(255, size=(4096, 4096, 3), dtype=np.uint8)


@pytest.fixture
def random_float32_image() -> npt.NDArray[np.float32]:
    return np.random.random((4096, 4096, 3)).astype(np.float32)


@pytest.fixture
def unknown_image_file(blank_png: Path) -> Path:
    unknown_image_file = blank_png.with_suffix(".XXYY")
    os.rename(blank_png, unknown_image_file)
    return unknown_image_file


def test_read_image_file(blank_image: npt.NDArray[np.uint8], blank_png: Path):
    actual = read_image_file(blank_png)
    assert np.all(actual == blank_image)


def test_correct_cv_image(blank_image: npt.NDArray[np.uint8]):
    actual = correct_cv_image(blank_image)
    assert np.all(actual == blank_image)


def test_correct_cv_image_grayscale(
    blank_image: npt.NDArray[np.uint8], blank_grayscale_image: npt.NDArray[np.uint8]
):
    actual = correct_cv_image(blank_grayscale_image)
    assert np.all(actual == blank_image)


def test_correct_cv_image_normalized_blank(
    blank_image: npt.NDArray[np.uint8], blank_float32_image: npt.NDArray[np.float32]
):
    actual = correct_cv_image(blank_float32_image)
    assert np.all(actual == blank_image)


def test_correct_cv_image_normalized_random(
    random_float32_image: npt.NDArray[np.float32],
):
    actual = correct_cv_image(random_float32_image)
    assert actual.dtype == np.uint8


def test_read_image_file_fail_unknown_mime_type(unknown_image_file: Path):
    with pytest.raises(UnsupportedImageFile):
        _ = read_image_file(unknown_image_file)


def test_decode_data_npy(blank_image: npt.NDArray[np.uint8], blank_npy: Path):
    with open(blank_npy, "rb+") as f:
        data = f.read()
    actual = decode_data(data, NPY_MIME_TYPE)
    assert np.all(actual == blank_image)


def test_decode_data_tif(blank_image: npt.NDArray[np.uint8], blank_tif: Path):
    with open(blank_tif, "rb+") as f:
        data = f.read()
    actual = decode_data(data, TIFF_MIME_TYPE)
    assert np.all(actual == blank_image)


def test_decode_data_fails(blank_npy: Path):
    with open(blank_npy, "rb+") as f:
        data = f.read()
    with pytest.raises(ImageDecodeError):
        _ = decode_data(data, "image/jpeg")
