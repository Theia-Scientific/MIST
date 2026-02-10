#!/usr/bin/env python3

import cv2
import numpy as np
import pytest

from ultralytics.utils.downloads import attempt_download_asset

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
def blank_npy(blank_image, tmp_path):
    npy_file = tmp_path.joinpath("image.npy")
    np.save(npy_file, blank_image)
    yield npy_file


@pytest.fixture
def blank_png(blank_image, tmp_path):
    png_file = tmp_path.joinpath("image.png")
    cv2.imwrite(str(png_file), blank_image)
    yield png_file


@pytest.fixture
def blank_tif(blank_image, tmp_path):
    tif_file = tmp_path.joinpath("image.tif")
    cv2.imwrite(str(tif_file), blank_image)
    yield tif_file


