#!/usr/bin/env python3

import pytest

from mist import detecting, visualizing
from ultralytics.models import YOLO

@pytest.fixture
def mock_plt_show(mocker):
    def mock_plt_show(*args, **kwargs):
        _ = args
        _ = kwargs

        return None

    mocker.patch("matplotlib.pyplot.show", mock_plt_show)
   

def test_run(mock_plt_show, bus_jpg, weights_file):
    _ = mock_plt_show
    result = detecting.run(bus_jpg, YOLO(weights_file), "cpu")
    labeled_image = visualizing.run(result.instances, result.original_image, result.class_names)
    assert labeled_image.any()


def test_run_with_no_class_names(mock_plt_show, bus_jpg, weights_file):
    _ = mock_plt_show
    result = detecting.run(bus_jpg, YOLO(weights_file), "cpu")
    labeled_image = visualizing.run(result.instances, result.original_image, [])
    assert labeled_image.any()


def test_run_with_show_classes_list(mock_plt_show, bus_jpg, weights_file):
    _ = mock_plt_show
    result = detecting.run(bus_jpg, YOLO(weights_file), "cpu")
    labeled_image = visualizing.run(
        result.instances,
        result.original_image,
        result.class_names,
        show_classes_list=[0]
    )
    assert labeled_image.any()
