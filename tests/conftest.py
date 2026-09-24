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
def bus_class_ids(bus_txt: Path) -> list[int]:
    with open(bus_txt, "r") as txt:
        return [int(line.strip().split(" ").pop(0)) for line in txt]


@pytest.fixture
def bus_image(bus_jpg: Path) -> npt.NDArray[np.uint8]:
    img = cv2.imread(bus_jpg)
    assert img is not None
    return np.asarray(img, dtype=np.uint8)


@pytest.fixture
def bus_masks(bus_txt: Path) -> npt.NDArray[np.bool]:
    mask_width_px = 640
    mask_height_px = 640
    with open(bus_txt, "r") as txt:
        masks: list[npt.NDArray[np.uint8]] = []
        for line in txt:
            current_line = line.strip()
            data = current_line.split(" ")
            data.pop(0)
            polygon = np.array(
                [
                    [
                        int(float(x) * int(mask_width_px)),
                        int(float(y) * int(mask_height_px)),
                    ]
                    for x, y in zip(data[0::2], data[1::2])
                ]
            )
            masks.append(
                sv.polygon_to_mask(
                    polygon, resolution_wh=(mask_width_px, mask_height_px)
                )
            )
    return np.array([mask.astype(np.bool) for mask in masks])


@pytest.fixture
def model(
    bus_class_ids: list[int], bus_masks: npt.NDArray[np.bool]
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

    def predict(image: npt.NDArray[np.uint8]) -> sv.Detections:
        _ = image

        return sv.Detections(
            class_id=np.array(bus_class_ids),
            confidence=None,
            data={
                CLASS_NAME_DATA_FIELD: np.array(
                    [class_names[class_id] for class_id in bus_class_ids]
                ),
            },
            mask=bus_masks,
            tracker_id=None,
            xyxy=sv.mask_to_xyxy(bus_masks),
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
