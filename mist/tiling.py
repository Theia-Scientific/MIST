#!/usr/bin/env python3

import logging
import numpy as np
import numpy.typing as npt

from pydantic import BaseModel

LOGGER: logging.Logger = logging.getLogger(__name__)


class Tile(BaseModel, arbitrary_types_allowed=True):
    img: npt.NDArray[np.uint8]
    index: int
    x_start: int
    y_start: int


def run(
    src_img: npt.NDArray[np.uint8],
    logger: logging.Logger = LOGGER,
    tile_size: tuple[int, int] = (640, 640),
    overlap: tuple[float, float] = (0.2, 0.2),
) -> list[Tile]:
    logger.debug(f"{src_img=}")
    logger.debug(f"{tile_size=}")
    logger.debug(f"{overlap=}")
    src_img_shape: tuple[int, ...] = src_img.shape
    image_height, image_width, *_ = src_img_shape
    tile_width, tile_height = tile_size
    overlap_width_ratio, overlap_height_ratio = overlap
    tiles: list[Tile] = []
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
