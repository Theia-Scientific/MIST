#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
import os
import supervision as sv

from mist.detecting import run
from mist.dump import MaskConfiguration
from mist.erosion import Configuration as ErosionConfiguration
from pathlib import Path
from typing import Callable, TypeAlias

Model: TypeAlias = tuple[Callable[[npt.NDArray[np.uint8]], sv.Detections], list[str]]


def test_run(bus_image: npt.NDArray[np.uint8], model: Model):
    predict, class_names = model
    detections = run(bus_image, predict, class_names)
    assert detections is not None


def test_run_with_no_predictions(
    blank_image: npt.NDArray[np.uint8], empty_detections: Model
):
    predict, class_names = empty_detections
    detections = run(blank_image, predict, class_names)
    assert detections.is_empty()


def test_run_with_erosion(bus_image: npt.NDArray[np.uint8], model: Model):
    predict, class_names = model
    detections = run(
        bus_image,
        predict,
        class_names,
        erosion=ErosionConfiguration(enabled=True),
    )
    assert detections is not None


def test_run_with_dump_masks(
    bus_image: npt.NDArray[np.uint8], model: Model, tmp_path: Path
):
    predict, class_names = model
    detections = run(
        bus_image,
        predict,
        class_names,
        dump_masks=MaskConfiguration(
            clazz=True, data=True, erode=True, instance=True, to=tmp_path
        ),
    )
    assert detections is not None
    assert len(os.listdir(tmp_path)) > 0


def test_empty_detections(blank_image: npt.NDArray[np.uint8], empty_detections: Model):
    predict, class_names = empty_detections
    detections = run(blank_image, predict, class_names)
    assert detections.is_empty()
