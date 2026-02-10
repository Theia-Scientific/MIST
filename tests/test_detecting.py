#!/usr/bin/env python3

from mist.detecting import run
from ultralytics.models import YOLO

def test_run(bus_jpg, weights_file):
    run(bus_jpg, YOLO(weights_file), device="cpu")


def test_run_with_no_predictions(blank_png, weights_file):
    run(blank_png, YOLO(weights_file), device="cpu")
