#!/usr/bin/env python3

import cv2
import numpy as np
import numpy.typing as npt
import os
import pytest
import supervision as sv

from pathlib import Path
from supervision.config import CLASS_NAME_DATA_FIELD
from typing import Callable


@pytest.fixture
def assets() -> Path:
    return Path(os.getcwd()).joinpath("tests", "assets")


@pytest.fixture
def bus_jpg(assets: Path) -> Path:
    return assets.joinpath("bus.jpg")


@pytest.fixture
def bus_txt(assets: Path) -> Path:
    return assets.joinpath("bus.txt")


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


@pytest.fixture
def weights_file() -> str:
    return "yolo26n-seg.pt"


@pytest.fixture
def model(
    bus_txt: Path,
) -> tuple[Callable[[npt.NDArray[np.uint8]], sv.Detections], list[str]]:
    class_names = [
        "bus",
        "cat",
        "dog",
        "horse",
        "bird",
        "car",
        "cellphone",
        "tv",
        "clock",
        "person",
    ]

    with open(bus_txt, "r") as txt:
        masks: list[npt.NDArray[np.uint8]] = []
        class_ids: list[int] = []
        for line in txt:
            current_line = line.strip()
            data = current_line.split(" ")
            class_ids.append(int(data.pop(0)))
            polygon = np.array(
                [
                    [
                        int(float(x) * int(640)),
                        int(float(y) * int(640)),
                    ]
                    for x, y in zip(data[0::2], data[1::2])
                ]
            )
            masks.append(sv.polygon_to_mask(polygon, resolution_wh=(640, 640)))
    mask = np.array([mask.astype(np.bool) for mask in masks])

    def predict(image: npt.NDArray[np.uint8]) -> sv.Detections:
        _ = image

        return sv.Detections(
            class_id=np.array(class_ids),
            confidence=None,
            data={
                CLASS_NAME_DATA_FIELD: np.array(
                    [class_names[class_id] for class_id in class_ids]
                ),
            },
            mask=mask,
            tracker_id=None,
            xyxy=sv.mask_to_xyxy(mask),
        )

    return predict, class_names


@pytest.fixture
def empty_detections() -> (
    tuple[Callable[[npt.NDArray[np.uint8]], sv.Detections], list[str]]
):
    def predict(image: npt.NDArray[np.uint8]) -> sv.Detections:
        _ = image

        detections = sv.Detections.empty()
        detections.class_id = None
        assert detections.class_id is None
        return detections

    return predict, ["object"]
