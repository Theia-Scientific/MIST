#!/usr/bin/env python3

import logging
import numpy as np

from collections import Counter
from mist import merging, tiling
from mist.instances import Instance
from mist.utils import read_image_file
from mist.visualizing import Tile as VisualTile
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from ultralytics.models import YOLO
from typing import List

DEFAULT_DEVICE: str = "cuda:0"
DEFAULT_DUMP_MASKS: bool = False
DEFAULT_DUMP_MASKS_TO: Path = Path("tmp")
DEFAULT_INFERENCE_CONFIDENCE: float = 0.35
DEFAULT_INFERENCE_IMAGE_SIZE: int = 640
DEFAULT_INFERENCE_IOU: float = 0.7
DEFAULT_INFERENCE_MAX_DETECTIONS: int = 1000
DEFAULT_INFERENCE_SILENT: bool = False
DEFAULT_MERGE_CLASSES: List[int] = []
DEFAULT_OVERLAP_HEIGHT: float = 0.2
DEFAULT_OVERLAP_WIDTH: float = 0.2
DEFAULT_SHOW_TILES: bool = False
DEFAULT_TILE_HEIGHT: int = 640
DEFAULT_TILE_WIDTH: int = 640

LOGGER: logging.Logger = logging.getLogger(__name__)

class Stats(BaseModel):
    merged: Counter
    unmerged: Counter


class Result(BaseModel):
    class_names: List[str]
    instances: List[Instance]
    original_image: np.ndarray
    stats: Stats
    visual_tiles: List[VisualTile]

    model_config = ConfigDict(arbitrary_types_allowed=True)


def run(
    src: Path,
    model: YOLO,
    device: str = DEFAULT_DEVICE,
    dump_masks: bool = DEFAULT_DUMP_MASKS,
    dump_masks_to: Path = DEFAULT_DUMP_MASKS_TO,
    inference_confidence: float = DEFAULT_INFERENCE_CONFIDENCE,
    inference_image_size: int = DEFAULT_INFERENCE_IMAGE_SIZE,
    inference_iou: float = DEFAULT_INFERENCE_IOU,
    inference_max_detections: int = DEFAULT_INFERENCE_MAX_DETECTIONS,
    inference_silent: bool = DEFAULT_INFERENCE_SILENT,
    merge_classes: List[int] = DEFAULT_MERGE_CLASSES,
    overlap_height: float = DEFAULT_OVERLAP_HEIGHT,
    overlap_width: float = DEFAULT_OVERLAP_WIDTH,
    tile_height: int = DEFAULT_TILE_HEIGHT,
    tile_width: int = DEFAULT_TILE_WIDTH,
    logger: logging.Logger = LOGGER
) -> Result:
    logger.info("Reading image file...")
    original_img = read_image_file(src)
    logger.info("Reading image file...DONE")
    orig_height, orig_width, *_ = original_img.shape
    orig_size = (orig_width, orig_height)
    logger.info("Creating tiles...")
    tiles = tiling.run(
        original_img,
        tile_size=(tile_width, tile_height),
        overlap=(overlap_width, overlap_height),
    )
    logger.info("Creating tiles...DONE")
    masks = []
    class_indices = []
    visual_tiles = []
    for index, tile in enumerate(tiles):
        logger.info(f"Running inference on {index} tile...")
        results = model(
            tile.img,
            agnostic_nms=False,
            device=device,
            classes=None,
            conf=inference_confidence,
            half=False,
            imgsz=inference_image_size,
            iou=inference_iou,
            max_det=inference_max_detections,
            retina_masks=True,
            verbose=not inference_silent,
        )
        logger.info(f"Running inference on {index} tile...DONE")
        pred = results[0]
        tile_class_indices = pred.boxes.cls.cpu().int().tolist()
        class_indices.extend(tile_class_indices)
        if pred.masks is None:
            masks_data = np.zeros(
                (len(tile_class_indices), tile_height, tile_width)
            )
        else:
            masks_data = pred.masks.data.cpu().numpy().astype(np.uint8)
        for data in masks_data:
            masks.append(
                merging.Mask(
                    data=data, offset_x=tile.x_start, offset_y=tile.y_start
                )
            )
        visual_tiles.append(
            VisualTile(
                x_min=tile.x_start,
                y_min=tile.y_start,
                x_max=tile.x_start + tile_width,
                y_max=tile.y_start + tile_height,
            )
        )
    logger.info("Merging results...")
    instances = merging.run(
        class_indices,
        masks,
        orig_size,
        (tile_width, tile_height),
        dump_masks=dump_masks,
        dump_masks_to=dump_masks_to,
        merge_classes=merge_classes,
    )
    logger.info("Merging results...DONE")
    class_names = [name for _, name in sorted(model.names.items())]
    logger.debug(f"class_names={class_names}")
    all_class_names = [class_names[i] for i in class_indices]
    instance_class_names = [class_names[i.class_index] for i in instances]
    return Result(
        class_names=class_names,
        instances=instances,
        original_image=original_img,
        stats=Stats(
            merged = Counter(instance_class_names),
            unmerged = Counter(all_class_names)
        ),
        visual_tiles=visual_tiles
    )
