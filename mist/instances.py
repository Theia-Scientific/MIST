#!/usr/bin/env python3

import numpy as np

from pydantic import BaseModel, ConfigDict
from typing import List


class Instance(BaseModel):
    box: List[int]
    class_index: int
    id: int
    mask: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)
