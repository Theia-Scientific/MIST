#!/usr/bin/env python3

import os
import pytest

from mist.detecting import run
from mist.models import Inference, yolo


@pytest.fixture
def model(weights_file) -> Inference:
    return yolo.Model(weights_file, device="cpu")


def test_run(bus_jpg, model):
    result = run(bus_jpg, model)
    assert len(result.class_names) > 0
    assert len(result.instances) > 0


def test_run_with_no_predictions(blank_png, model):
    result = run(blank_png, model)
    assert len(result.class_names) > 0
    assert len(result.instances) == 0


def test_run_with_dump_masks(bus_jpg, model, tmp_path):
    result = run(
        bus_jpg,
        model,
        dump_masks=True,
        dump_masks_to=tmp_path,
    )
    assert len(result.class_names) > 0
    assert len(result.instances) > 0
    assert len(os.listdir(tmp_path)) > 0
