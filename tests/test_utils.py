#!/usr/bin/env python3

import numpy as np
import pytest

from mist.utils import (
    correct_cv_image,
    decode_data,
    NPY_MIME_TYPE,
    read_image_file,
    TIFF_MIME_TYPE,
    UnknownMimeTypeError
)

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


