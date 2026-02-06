#!/usr/bin/env python3

import numpy as np

from pydantic import BaseModel, ConfigDict
from typing import List, Tuple

class Tile(BaseModel):
    img: np.ndarray
    index: int
    x_start: int
    y_start: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TileMask(BaseModel):
    data: np.ndarray
    offset_x: int
    offset_y: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TileVisual(BaseModel):
    color: Tuple[int, int, int] = (0, 0, 255)  # BGR
    thickness: int = 3
    x_min: int
    y_min: int
    x_max: int
    y_max: int


def create_tiles(
    src_img: np.ndarray,
    tile_size: Tuple[int, int] = (640, 640),
    overlap: Tuple[float, float] = (0.2, 0.2),
) -> List[Tile]:
    image_height, image_width, *_ = src_img.shape
    tile_width, tile_height = tile_size
    overlap_width_ratio, overlap_height_ratio = overlap
    tiles = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * tile_height)
    x_overlap = int(overlap_width_ratio * tile_width)
    count = 0
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + tile_height
        while x_max < image_width:
            x_max = x_min + tile_width
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
        y_min = y_max - y_overlap
    return tiles


