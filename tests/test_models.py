#!/usr/bin/env python3

import numpy as np

from mist.models import Inference, Result
from typing import List


class Mock(Inference):
    def __call__(
        self,
        image: np.ndarray,
        offset_x: int,
        offset_y: int,
        tile_height: int,
        tile_width: int,
    ) -> Result:
        _ = image
        _ = offset_x
        _ = offset_y
        _ = tile_height
        _ = tile_width
        return Result(masks=[], class_indices=[])

    @property
    def names(self) -> List[str]:
        return ["hello", "world"]


def test_inference():
    model = Mock()
    result = model(np.zeros((10, 10)), 0, 0, 0, 0)
    assert len(result.masks) == 0
    assert len(result.class_indices) == 0
    assert model.names == ["hello", "world"]
