#!/usr/bin/env python3

from mist.detecting import run
from ultralytics.models import YOLO

def test_run_with_segmentation_results(blank_png, weights_file):
    run(blank_png, YOLO(weights_file))
