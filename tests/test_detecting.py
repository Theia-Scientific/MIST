#!/usr/bin/env python3

import os

from mist.detecting import run
from ultralytics.models import YOLO

def test_run(bus_jpg, weights_file):
    result = run(bus_jpg, YOLO(weights_file), device="cpu")
    assert len(result.class_names) > 0
    assert len(result.instances) > 0


def test_run_with_no_predictions(blank_png, weights_file):
    result = run(blank_png, YOLO(weights_file), device="cpu")
    assert len(result.class_names) > 0
    assert len(result.instances) == 0


def test_run_with_dump_masks(bus_jpg, tmp_path, weights_file):
    result = run(
        bus_jpg,
        YOLO(weights_file),
        device="cpu",
        dump_masks=True,
        dump_masks_to=tmp_path
    )
    assert len(result.class_names) > 0
    assert len(result.instances) > 0
    assert len(os.listdir(tmp_path)) > 0
