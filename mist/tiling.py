#!/usr/bin/env python3

import logging
import numpy as np

from pydantic import BaseModel, ConfigDict
from typing import List, Tuple

LOGGER: logging.Logger = logging.getLogger(__name__)

class Tile(BaseModel):
    img: np.ndarray
    index: int
    x_start: int
    y_start: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


def run(
    src_img: np.ndarray,
    tile_size: Tuple[int, int] = (640, 640),
    overlap: Tuple[float, float] = (0.2, 0.2),
    logger: logging.Logger = LOGGER
) -> List[Tile]:
    logger.debug(f"{src_img=}")
    logger.debug(f"{tile_size=}")
    logger.debug(f"{overlap=}")
    image_height, image_width, *_ = src_img.shape
    tile_width, tile_height = tile_size
    overlap_width_ratio, overlap_height_ratio = overlap
    tiles = []
    y_max = 0
    y_min = 0
    y_overlap = int(overlap_height_ratio * tile_height)
    x_overlap = int(overlap_width_ratio * tile_width)
    logger.debug(f"{y_overlap=}")
    logger.debug(f"{x_overlap=}")
    count = 0
    while y_max < image_height:
        logger.debug(f"{count=}")
        x_min = 0
        x_max = 0
        y_max = y_min + tile_height
        logger.debug(f"{y_max=}")
        while x_max < image_width:
            x_max = x_min + tile_width
            logger.debug(f"{x_max=}")
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - tile_width)
                ymin = max(0, ymax - tile_height)
            else:
                xmax = x_max
                ymax = y_max
                xmin = x_min
                ymin = y_min
            tile_img = src_img[ymin:ymax, xmin:xmax]
            count += 1
            tiles.append(Tile(img=tile_img, index=count, x_start=xmin, y_start=ymin))
            x_min = x_max - x_overlap
            logger.debug(f"{x_min=}")
        y_min = y_max - y_overlap
        logger.debug(f"{y_min=}")
    return tiles


