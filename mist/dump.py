#!/usr/bin/env python3

import cv2
import numpy as np

from pathlib import Path
from pydantic import BaseModel

DEFAULT_MASK_CLASS: bool = False
DEFAULT_MASK_DATA: bool = False
DEFAULT_MASK_ERODE: bool = False
DEFAULT_MASK_INSTANCE: bool = False
DEFAULT_MASK_TO: Path = Path("tmp")


class MaskConfiguration(BaseModel):
    clazz: bool = DEFAULT_MASK_CLASS
    data: bool = DEFAULT_MASK_DATA
    erode: bool = DEFAULT_MASK_ERODE
    instance: bool = DEFAULT_MASK_INSTANCE
    to: Path = Path("tmp")

    @property
    def enabled(self) -> bool:
        return self.clazz or self.data or self.erode or self.instance

    def write(
        self, img: np.ndarray, class_index: int, instance_index: int, suffix: str
    ) -> bool:
        return not cv2.imwrite(
            str(
                self.to.joinpath(str(class_index)).joinpath(
                    f"{instance_index}{suffix}.png"
                )
            ),
            img * 255,
        )

    def write_class(
        self, img: np.ndarray, class_index: int, instance_index: int
    ) -> bool:
        if self.clazz:
            return self.write(img, class_index, instance_index, "c")
        else:
            return False

    def write_data(
        self, img: np.ndarray, class_index: int, instance_index: int
    ) -> bool:
        if self.data:
            return self.write(img, class_index, instance_index, "m")
        else:
            return False

    def write_erode(
        self, img: np.ndarray, class_index: int, instance_index: int
    ) -> bool:
        if self.erode:
            return self.write(img, class_index, instance_index, "e")
        else:
            return False

    def write_instance(
        self, img: np.ndarray, class_index: int, instance_index: int
    ) -> bool:
        if self.instance:
            return self.write(img, class_index, instance_index, "i")
        else:
            return False
