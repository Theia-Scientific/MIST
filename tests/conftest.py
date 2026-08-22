#!/usr/bin/env python3

import cv2
import numpy as np
import pytest
import supervision as sv

from typing import Any, Callable, Dict, List, Tuple
from ultralytics.models import YOLO
from ultralytics.utils.downloads import attempt_download_asset, download


@pytest.fixture(scope="session")
def assets(tmp_path_factory):
    return tmp_path_factory.mktemp("assets")


@pytest.fixture(scope="session")
def weights_file(assets):
    weights_file = attempt_download_asset(
        "weights/yolov8n-seg.pt", dir=assets, progress=False
    )
    return assets.joinpath(weights_file)


@pytest.fixture(scope="session")
def bus_jpg(assets):
    bus_jpg = "bus.jpg"
    download(f"https://www.ultralytics.com/images/{bus_jpg}", dir=assets)
    return assets.joinpath(bus_jpg)


@pytest.fixture
def blank_image() -> np.ndarray:
    return np.zeros((4096, 4096, 3), dtype=np.uint8)


@pytest.fixture
def blank_jpg(blank_image, tmp_path):
    jpg_file = tmp_path.joinpath("image.jpg")
    cv2.imwrite(str(jpg_file), blank_image)
    yield jpg_file


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


@pytest.fixture(scope="session")
def model(
    weights_file,
) -> Tuple[Callable[[np.ndarray, Dict[str, Any]], sv.Detections], List[str]]:
    model = YOLO(weights_file)
    class_names = [name for _, name in sorted(model.names.items())]

    def predict(image: np.ndarray, parameters: Dict[str, Any]) -> sv.Detections:
        return sv.Detections.from_ultralytics(
            model(
                image,
                agnostic_nms=parameters.get("agnostic_nms", False),
                device=parameters.get("device", "cpu"),
                classes=parameters.get("classes", None),
                conf=parameters.get("confidence", 0.35),
                imgsz=parameters.get("image_size", 640),
                iou=parameters.get("iou", 0.7),
                max_det=parameters.get("maximum_detections", 1000),
                retina_masks=parameters.get("retina_masks", True),
                verbose=parameters.get("verbose", False),
            )[0]
        )

    return predict, class_names


@pytest.fixture(scope="session")
def empty_detections() -> (
    Tuple[Callable[[np.ndarray, Dict[str, Any]], sv.Detections], List[str]]
):
    def predict(image: np.ndarray, parameters: Dict[str, Any]) -> sv.Detections:
        _ = image
        _ = parameters

        return sv.Detections(xyxy=np.array([[0, 0, 100, 100]]))

    return predict, ["object"]
