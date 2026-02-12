#!/usr/bin/env python3

from matplotlib.figure import Figure
from mist.tiling import run


def test_create_tiles_show(mocker, blank_image):
    def mock_figure(*args, **kwargs):
        _ = args
        _ = kwargs
        return mocker.MagicMock(spec=Figure)

    mocker.patch("matplotlib.pyplot.figure", mock_figure)
    run(blank_image)
