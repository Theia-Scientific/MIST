#!/usr/bin/env python3

import os

from mist.detecting import run
from mist.dump import MaskConfiguration
from mist.erosion import Configuration as ErosionConfiguration


def test_run(bus_jpg, model):
    predict, class_names = model
    result = run(bus_jpg, predict, class_names)
    assert len(result.class_names) > 0
    assert len(result.instances) > 0


def test_run_with_no_predictions(blank_png, model):
    predict, class_names = model
    result = run(blank_png, predict, class_names)
    assert len(result.class_names) > 0
    assert len(result.instances) == 0


def test_run_with_erosion(bus_jpg, model):
    predict, class_names = model
    result = run(
        bus_jpg,
        predict,
        class_names,
        erosion=ErosionConfiguration(enabled=True),
    )
    assert len(result.class_names) > 0
    assert len(result.instances) > 0


def test_run_with_dump_masks(bus_jpg, model, tmp_path):
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


def test_empty_detections(blank_png, empty_detections):
    predict, class_names = empty_detections
    result = run(blank_png, predict, class_names)

    assert len(result.class_names) == 1
    assert len(result.instances) == 0
