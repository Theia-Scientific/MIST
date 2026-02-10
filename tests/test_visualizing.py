#!/usr/bin/env python3

from mist import detecting, visualizing
from ultralytics.models import YOLO

def test_run(mocker, bus_jpg, weights_file):
    def mock_plt_show(*args, **kwargs):
        _ = args
        _ = kwargs

        return None

    mocker.patch("matplotlib.pyplot.show", mock_plt_show)
    result = detecting.run(bus_jpg, YOLO(weights_file), "cpu")
    labeled_image = visualizing.run(result.instances, result.original_image, result.class_names)
    assert labeled_image.any()
