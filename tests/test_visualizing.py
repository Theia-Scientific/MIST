#!/usr/bin/env python3

import pytest

from mist import detecting, visualizing


@pytest.fixture(scope="module")
def bus_result(bus_jpg, model):
    return detecting.run(bus_jpg, model)


@pytest.fixture
def mock_plt_show(mocker):
    def mock_plt_show(*args, **kwargs):
        _ = args
        _ = kwargs

        return None

    mocker.patch("matplotlib.pyplot.show", mock_plt_show)


def test_run(mock_plt_show, bus_result):
    _ = mock_plt_show
    labeled_image = visualizing.run(
        bus_result.instances, bus_result.original_image, bus_result.class_names
    )
    assert labeled_image.any()


def test_run_with_no_class_names(mock_plt_show, bus_result):
    _ = mock_plt_show
    labeled_image = visualizing.run(bus_result.instances, bus_result.original_image, [])
    assert labeled_image.any()


def test_run_with_show_classes_list(mock_plt_show, bus_result):
    _ = mock_plt_show
    labeled_image = visualizing.run(
        bus_result.instances,
        bus_result.original_image,
        bus_result.class_names,
        show_classes_list=[0],
    )
    assert labeled_image.any()


def test_run_with_no_random_colors(mock_plt_show, bus_result):
    _ = mock_plt_show
    labeled_image = visualizing.run(
        bus_result.instances,
        bus_result.original_image,
        bus_result.class_names,
        random_object_colors=False,
        list_of_class_colors=None,
    )
    assert labeled_image.any()


def test_run_with_class_colors(mock_plt_show, bus_result):
    _ = mock_plt_show
    labeled_image = visualizing.run(
        bus_result.instances,
        bus_result.original_image,
        bus_result.class_names,
        random_object_colors=False,
        list_of_class_colors=[(255, 0, 0) for _ in bus_result.class_names],
    )
    assert labeled_image.any()


def test_run_with_opaque_mask_color(mock_plt_show, bus_result):
    _ = mock_plt_show
    labeled_image = visualizing.run(
        bus_result.instances, bus_result.original_image, bus_result.class_names, alpha=1
    )
    assert labeled_image.any()


def test_run_with_tiles(mock_plt_show, bus_result):
    _ = mock_plt_show
    labeled_image = visualizing.run(
        bus_result.instances,
        bus_result.original_image,
        bus_result.class_names,
        tiles=bus_result.visual_tiles,
    )
    assert labeled_image.any()


def test_run_with_boxes(mock_plt_show, bus_result):
    _ = mock_plt_show
    labeled_image = visualizing.run(
        bus_result.instances,
        bus_result.original_image,
        bus_result.class_names,
        show_boxes=True,
    )
    assert labeled_image.any()


def test_run_with_show_classes(mock_plt_show, bus_result):
    _ = mock_plt_show
    labeled_image = visualizing.run(
        bus_result.instances,
        bus_result.original_image,
        bus_result.class_names,
        show_class=True,
    )
    assert labeled_image.any()
