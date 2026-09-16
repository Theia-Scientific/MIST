#!/usr/bin/env python3

import logging
import numpy as np
import numpy.typing as npt
import supervision as sv

from collections import Counter
from mist import dump, erosion, merging, tiling
from mist.instances import Instance
from pydantic import BaseModel
from typing import Callable, Any

DEFAULT_EROSION_CONFIGURATION: erosion.Configuration = erosion.Configuration()
DEFAULT_MASK_CONFIGURATION: dump.MaskConfiguration = dump.MaskConfiguration()
DEFAULT_MERGE_CLASSES: list[int] = []
DEFAULT_OVERLAP_HEIGHT: float = 0.2
DEFAULT_OVERLAP_WIDTH: float = 0.2
DEFAULT_PARAMETERS: dict[str, Any] = {}
DEFAULT_SHOW_TILES: bool = False
DEFAULT_TILE_HEIGHT: int = 640
DEFAULT_TILE_WIDTH: int = 640

LOGGER: logging.Logger = logging.getLogger(__name__)


class Stats(BaseModel):
    merged: Counter[str]
    unmerged: Counter[str]


class Result(BaseModel, arbitrary_types_allowed=True):
    class_names: list[str]
    instances: list[Instance]
    original_image: npt.NDArray[np.uint8]
    stats: Stats


def run(
    img: npt.NDArray[np.uint8],
    model: Callable[[npt.NDArray[np.uint8], dict[str, Any]], sv.Detections],
    class_names: list[str],
    dump_masks: dump.MaskConfiguration = DEFAULT_MASK_CONFIGURATION,
    erosion: erosion.Configuration = DEFAULT_EROSION_CONFIGURATION,
    logger: logging.Logger = LOGGER,
    merge_classes: list[int] = DEFAULT_MERGE_CLASSES,
    overlap_height: float = DEFAULT_OVERLAP_HEIGHT,
    overlap_width: float = DEFAULT_OVERLAP_WIDTH,
    parameters: dict[str, Any] = DEFAULT_PARAMETERS,
    tile_height: int = DEFAULT_TILE_HEIGHT,
    tile_width: int = DEFAULT_TILE_WIDTH,
) -> Result:
    logger.debug(f"{img=}")
    logger.debug(f"{class_names=}")
    logger.debug(f"{dump_masks=}")
    logger.debug(f"{erosion=}")
    logger.debug(f"{merge_classes=}")
    logger.debug(f"{overlap_height=}")
    logger.debug(f"{overlap_width=}")
    logger.debug(f"{parameters=}")
    logger.debug(f"{tile_height=}")
    logger.debug(f"{tile_width=}")
    img_shape: tuple[int, ...] = img.shape
    orig_height, orig_width, *_ = img_shape
    logger.debug(f"{orig_height=}")
    logger.debug(f"{orig_width=}")
    orig_size = (orig_width, orig_height)
    logger.debug(f"{orig_size=}")
    logger.info("Creating tiles...")
    tiles = tiling.run(
        img,
        tile_size=(tile_width, tile_height),
        overlap=(overlap_width, overlap_height),
    )
    logger.info("Creating tiles...DONE")
    masks: list[merging.Mask] = []
    class_indices: list[int] = []
    for index, tile in enumerate(tiles):
        logger.info(f"Running inference on {index} tile...")
        detections = model(tile.img, parameters)
        if detections.class_id is None:
            class_indices.extend([0 for _ in range(len(detections))])
        else:
            class_indices.extend(detections.class_id.tolist())
        if detections.mask is None:
            masks_data = np.zeros(
                (len(detections), tile_height, tile_width), dtype=bool
            )
        else:
            masks_data = detections.mask
        masks.extend(
            [
                merging.Mask(data=data, offset_x=tile.x_start, offset_y=tile.y_start)
                for data in masks_data
            ]
        )
        logger.info(f"Running inference on {index} tile...DONE")
    logger.info("Merging results...")
    instances = merging.run(
        class_indices,
        masks,
        orig_size,
        (tile_width, tile_height),
        erosion=erosion,
        dump_masks=dump_masks,
        merge_classes=merge_classes,
    )
    logger.info("Merging results...DONE")
    all_class_names = [class_names[i] for i in class_indices]
    logger.debug(f"{all_class_names=}")
    instance_class_names = [class_names[i.class_index] for i in instances]
    logger.debug(f"{instance_class_names=}")
    return Result(
        class_names=class_names,
        instances=instances,
        original_image=img,
        stats=Stats(
            merged=Counter(instance_class_names), unmerged=Counter(all_class_names)
        ),
    )
