#!/usr/bin/env python3

import cv2
import numpy as np
import numpy.typing as npt
import pytest
import supervision as sv

from pathlib import Path
from typing import Callable
from ultralytics.models import YOLO
from ultralytics.utils.downloads import attempt_download_asset, download


@pytest.fixture(scope="session")
def assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("assets")


@pytest.fixture(scope="session")
def weights_file(assets: Path) -> Path:
    weights_file = attempt_download_asset(
        "weights/yolov8n-seg.pt", dir=assets, progress=False
    )
    return assets.joinpath(weights_file)


@pytest.fixture(scope="session")
def bus_jpg(assets: Path) -> Path:
    bus_jpg = "bus.jpg"
    download(f"https://www.ultralytics.com/images/{bus_jpg}", dir=assets)
    return assets.joinpath(bus_jpg)


@pytest.fixture
def blank_image() -> npt.NDArray[np.uint8]:
    return np.zeros((4096, 4096, 3), dtype=np.uint8)


@pytest.fixture
def blank_jpg(blank_image: npt.NDArray[np.uint8], tmp_path: Path) -> Path:
    jpg_file = tmp_path.joinpath("image.jpg")
    assert cv2.imwrite(str(jpg_file), blank_image)
    return jpg_file


@pytest.fixture
def blank_npy(blank_image: npt.NDArray[np.uint8], tmp_path: Path) -> Path:
    npy_file = tmp_path.joinpath("image.npy")
    np.save(npy_file, blank_image)
    return npy_file


@pytest.fixture
def blank_png(blank_image: npt.NDArray[np.uint8], tmp_path: Path) -> Path:
    png_file = tmp_path.joinpath("image.png")
    assert cv2.imwrite(str(png_file), blank_image)
    return png_file


@pytest.fixture
def blank_tif(blank_image: npt.NDArray[np.uint8], tmp_path: Path) -> Path:
    tif_file = tmp_path.joinpath("image.tif")
    assert cv2.imwrite(str(tif_file), blank_image)
    return tif_file


@pytest.fixture(scope="session")
def model(
    weights_file: Path,
) -> tuple[Callable[[npt.NDArray[np.uint8]], sv.Detections], list[str]]:
    model = YOLO(weights_file)
    class_names = [name for _, name in sorted(model.names.items())]

    def predict(image: npt.NDArray[np.uint8]) -> sv.Detections:
        return sv.Detections.from_ultralytics(
            list(
                model(
                    image,
                    agnostic_nms=False,
                    device="cpu",
                    conf=0.35,
                    imgsz=640,
                    iou=0.7,
                    max_det=1000,
                    retina_masks=True,
                    verbose=False,
                )
            )[0]
        )

    return predict, class_names


@pytest.fixture(scope="session")
def empty_detections() -> (
    tuple[Callable[[npt.NDArray[np.uint8]], sv.Detections], list[str]]
):
    def predict(image: npt.NDArray[np.uint8]) -> sv.Detections:
        _ = image

        return sv.Detections.empty()

    return predict, ["object"]
