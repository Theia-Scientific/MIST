#!/usr/bin/env python3

import numpy as np
import supervision as sv

from mist.models import Inference
from pathlib import Path
from typing import Any, Dict, Union, List
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

    def __call__(self, image: np.ndarray, parameters: Dict[str, Any]) -> sv.Detections:
        return sv.Detections.from_ultralytics(
            self.model(
                image,
                agnostic_nms=parameters.get("agnostic_nms", False),
                device=parameters.get("device", "cuda:0"),
                classes=parameters.get("classes", None),
                conf=parameters.get("confidence", 0.35),
                imgsz=parameters.get("image_size", 640),
                iou=parameters.get("iou", 0.7),
                max_det=parameters.get("maximum_detections", 1000),
                retina_masks=parameters.get("retina_masks", True),
                verbose=parameters.get("verbose", False),
            )
        )
