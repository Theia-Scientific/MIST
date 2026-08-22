#!/usr/bin/env python3

import numpy as np
import supervision as sv

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Inference(ABC):

    @abstractmethod
    def __call__(self, image: np.ndarray, parameters: Dict[str, Any]) -> sv.Detections:
        pass

    @property
    @abstractmethod
    def names(self) -> List[str]:
        pass
