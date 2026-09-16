#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt

from pydantic import BaseModel


class Instance(BaseModel, arbitrary_types_allowed=True):
    box: list[int]
    class_index: int
    id: int
    mask: npt.NDArray[np.uint8]
