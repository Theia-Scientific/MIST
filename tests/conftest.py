#!/usr/bin/env python3

import cv2
import numpy as np
import pytest

@pytest.fixture
def blank_image() -> np.ndarray:
    return np.zeros((4096, 4096, 3), dtype=np.uint8)


@pytest.fixture
def blank_png(blank_image, tmp_path):
    png_file = tmp_path.joinpath("image.png")
    cv2.imwrite(str(png_file), blank_image)
    yield png_file


