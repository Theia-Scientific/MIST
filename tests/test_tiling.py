#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt

from matplotlib.figure import Figure
from mist.tiling import run
from pytest_mock import MockerFixture
from typing import Any


def test_create_tiles_show(mocker: MockerFixture, blank_image: npt.NDArray[np.uint8]):
    def mock_figure(*args: Any, **kwargs: Any):
        _ = args
        _ = kwargs
        return mocker.MagicMock(spec=Figure)

    _ = mocker.patch("matplotlib.pyplot.figure", mock_figure)
    _ = run(blank_image)
