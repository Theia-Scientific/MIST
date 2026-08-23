#!/usr/bin/env python3

import logging
import numpy as np
import supervision as sv

from collections import Counter
from mist import erosion, merging, tiling
from mist.instances import Instance
from mist.utils import read_image_file
from mist.visualizing import Tile as VisualTile
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from typing import Callable, Any, Dict, List

DEFAULT_DUMP_MASKS: bool = False
DEFAULT_DUMP_MASKS_TO: Path = Path("tmp")
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
    model: Callable[[np.ndarray, Dict[str, Any]], sv.Detections],
    class_names: List[str],
    dump_masks: bool = DEFAULT_DUMP_MASKS,
    dump_masks_to: Path = DEFAULT_DUMP_MASKS_TO,
    erosion: erosion.Configuration = erosion.Configuration(),
    merge_classes: List[int] = DEFAULT_MERGE_CLASSES,
    overlap_height: float = DEFAULT_OVERLAP_HEIGHT,
    overlap_width: float = DEFAULT_OVERLAP_WIDTH,
    tile_height: int = DEFAULT_TILE_HEIGHT,
    tile_width: int = DEFAULT_TILE_WIDTH,
    logger: logging.Logger = LOGGER,
    parameters: Dict[str, Any] = {},
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
        detections = model(tile.img, parameters)
        if detections.class_id is None:
            class_indices.extend([0 for _ in range(len(detections))])
        else:
            class_indices.extend(detections.class_id.tolist())
        if detections.mask is None:
            masks_data = np.zeros((len(detections), tile_height, tile_width))
        else:
            masks_data = detections.mask
        masks.extend(
            [
                merging.Mask(data=data, offset_x=tile.x_start, offset_y=tile.y_start)
                for data in masks_data
            ]
        )
        logger.info(f"Running inference on {index} tile...DONE")
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
        erosion=erosion,
        dump_masks=dump_masks,
        dump_masks_to=dump_masks_to,
        merge_classes=merge_classes,
    )
    logger.info("Merging results...DONE")
    all_class_names = [class_names[i] for i in class_indices]
    instance_class_names = [class_names[i.class_index] for i in instances]
    return Result(
        class_names=class_names,
        instances=instances,
        original_image=original_img,
        stats=Stats(
            merged=Counter(instance_class_names), unmerged=Counter(all_class_names)
        ),
        visual_tiles=visual_tiles,
    )
