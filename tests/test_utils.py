#!/usr/bin/env python3

import cv2
import numpy as np
import os
import pytest

from mist.utils import (
    correct_cv_image,
    decode_data,
    NPY_MIME_TYPE,
    read_image_file,
    TIFF_MIME_TYPE,
    UnknownMimeTypeError
)

@pytest.fixture
def blank_float32_image() -> np.ndarray:
    return np.zeros((4096, 4096, 3), dtype=np.float32)


@pytest.fixture
def blank_grayscale_image() -> np.ndarray:
    return np.zeros((4096, 4096), dtype=np.uint8)


@pytest.fixture
def random_image() -> np.ndarray:
    return np.random.randint(255, size=(4096, 4096, 3), dtype=np.uint8)


@pytest.fixture
def random_float32_image() -> np.ndarray:
    return np.random.random((4096, 4096, 3)).astype(np.float32)


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


