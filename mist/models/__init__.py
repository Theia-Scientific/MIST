#!/usr/bin/env python3

import numpy as np

from abc import ABC, abstractmethod
from mist import merging
from pydantic import BaseModel
from typing import List


class Result(BaseModel):
    class_indices: List[int]
    masks: List[merging.Mask]


class Inference(ABC):

    @abstractmethod
    def __call__(
        self,
        image: np.ndarray,
        offset_x: int,
        offset_y: int,
        tile_height: int,
        tile_width: int,
    ) -> Result:  # pragma: no cover
        pass

    @property
    @abstractmethod
    def names(self) -> List[str]:  # pragma: no cover
        pass
