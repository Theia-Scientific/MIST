#!/usr/bin/env python3

import numpy as np

from mist.merging import Mask
from mist.models import Inference, Result
from pathlib import Path
from typing import Union, List
from ultralytics.models import YOLO

DEFAULT_CONFIDENCE: float = 0.35
DEFAULT_DEVICE: str = "cuda:0"
DEFAULT_IOU: float = 0.7
DEFAULT_IMAGE_SIZE: int = 640
DEFAULT_MAX_DETECTIONS: int = 1000
DEFAULT_SILENT: bool = False


class Model(Inference):
    def __init__(
        self,
        weights_file: Union[Path, str],
        confidence: float = DEFAULT_CONFIDENCE,
        device: str = DEFAULT_DEVICE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        iou: float = DEFAULT_IOU,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        silent: bool = DEFAULT_SILENT,
    ):
        self.device = device
        self.confidence = confidence
        self.image_size = image_size
        self.iou = iou
        self.max_detections = max_detections
        self.silent = silent

        self.model = YOLO(weights_file)

    @property
    def names(self) -> List[str]:
        return [name for _, name in sorted(self.model.names.items())]

    def __call__(
        self,
        image: np.ndarray,
        offset_x: int,
        offset_y: int,
        tile_height: int,
        tile_width: int,
    ) -> Result:
        results = self.model(
            image,
            agnostic_nms=False,
            device=self.device,
            classes=None,
            conf=self.confidence,
            half=False,
            imgsz=self.image_size,
            iou=self.iou,
            max_det=self.max_detections,
            retina_masks=True,
            verbose=not self.silent,
        )
        pred = results[0]
        tile_class_indices = pred.boxes.cls.cpu().int().tolist()
        if pred.masks is None:
            masks_data = np.zeros((len(tile_class_indices), tile_height, tile_width))
        else:
            masks_data = pred.masks.data.cpu().numpy().astype(np.uint8)
        return Result(
            class_indices=tile_class_indices,
            masks=[
                Mask(data=data, offset_x=offset_x, offset_y=offset_y)
                for data in masks_data
            ],
        )
