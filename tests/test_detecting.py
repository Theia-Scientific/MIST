#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt
import os
import supervision as sv

from mist.detecting import run
from mist.dump import MaskConfiguration
from mist.erosion import Configuration as ErosionConfiguration
from pathlib import Path
from typing import Any, Callable, TypeAlias

Model: TypeAlias = tuple[
    Callable[[npt.NDArray[np.uint8], dict[str, Any]], sv.Detections], list[str]
]


def test_run(bus_jpg: Path, model: Model):
    predict, class_names = model
    result = run(bus_jpg, predict, class_names)
    assert len(result.class_names) > 0
    assert len(result.instances) > 0


def test_run_with_no_predictions(blank_png: Path, model: Model):
    predict, class_names = model
    result = run(blank_png, predict, class_names)
    assert len(result.class_names) > 0
    assert len(result.instances) == 0


def test_run_with_erosion(bus_jpg: Path, model: Model):
    predict, class_names = model
    result = run(
        bus_jpg,
        predict,
        class_names,
        erosion=ErosionConfiguration(enabled=True),
    )
    assert len(result.class_names) > 0
    assert len(result.instances) > 0


def test_run_with_dump_masks(bus_jpg: Path, model: Model, tmp_path: Path):
    predict, class_names = model
    result = run(
        bus_jpg,
        predict,
        class_names,
        dump_masks=MaskConfiguration(
            clazz=True, data=True, erode=True, instance=True, to=tmp_path
        ),
    )
    assert len(result.class_names) > 0
    assert len(result.instances) > 0
    assert len(os.listdir(tmp_path)) > 0


def test_empty_detections(blank_png: Path, empty_detections: Model):
    predict, class_names = empty_detections
    result = run(blank_png, predict, class_names)

    assert len(result.class_names) == 1
    assert len(result.instances) == 0
