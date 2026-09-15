#!/usr/bin/env python3

import numpy as np
import numpy.typing as npt

from pydantic import BaseModel, ConfigDict


class Instance(BaseModel):
    box: list[int]
    class_index: int
    id: int
    mask: npt.NDArray[np.uint8]

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
