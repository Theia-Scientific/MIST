#!/usr/bin/env python3

import cv2
import logging
import numpy as np
import os
import torch

from pathlib import Path
from mist.instances import Instance
from pydantic import BaseModel, ConfigDict
from typing import List, Tuple

LOGGER: logging.Logger = logging.getLogger(__name__)


class Mask(BaseModel):
    data: np.ndarray
    offset_x: int
    offset_y: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


def run(
    class_indices: List[int],
    masks: List[Mask],
    src_image_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    dump_masks: bool = False,
    dump_masks_to: Path = Path("tmp"),
    merge_classes: List[int] = [],
    logger: logging.Logger = LOGGER
) -> List[Instance]:
    logger.debug(f"{class_indices=}")
    logger.debug(f"{masks=}")
    logger.debug(f"{src_image_size=}")
    logger.debug(f"{tile_size=}")
    logger.debug(f"{dump_masks=}")
    logger.debug(f"{dump_masks_to=}")
    logger.debug(f"{merge_classes=}")
    tile_width, tile_height = tile_size
    src_image_width, src_image_height = src_image_size
    tensor_class_indices = torch.tensor(class_indices)
    instance_id = 0
    instances = []
    for cls_index in torch.unique(tensor_class_indices):
        logger.debug(f"{cls_index=}")
        cls_index_int = cls_index.item()
        logger.debug(f"{cls_index_int=}")
        if (cls_index in merge_classes and len(merge_classes) > 0) or len(
            merge_classes
        ) == 0:
            if dump_masks:
                os.makedirs(dump_masks_to.joinpath(str(cls_index_int)), exist_ok=True)
            cls_indexes = torch.where(tensor_class_indices == cls_index)[0]
            class_masks = [masks[i] for i in cls_indexes]
            logger.debug(f"class masks count = {len(class_masks)}")
            class_mask = np.zeros((src_image_height, src_image_width))
            logger.debug(f"class_mask.shape = {class_mask.shape}")
            for i, mask in enumerate(class_masks):
                class_mask[
                    mask.offset_y:mask.offset_y + tile_height,
                    mask.offset_x:mask.offset_x + tile_width,
                ] += mask.data
                if dump_masks:
                    dst = dump_masks_to.joinpath(str(cls_index_int))
                    logger.debug(f"{dst=}")
                    cv2.imwrite(
                        str(dst.joinpath(f"{i}c.png")), class_mask.astype(np.uint8) * 255
                    )
                    cv2.imwrite(
                        str(dst.joinpath(f"{i}m.png")), mask.data.astype(np.uint8) * 255
                    )
            class_mask = class_mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            logger.debug(f"contours count={len(contours)}")
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                instance_mask = np.zeros(
                    (src_image_height, src_image_width), dtype=np.uint8
                )
                cv2.fillPoly(instance_mask, [contour], 1)
                if dump_masks:
                    cv2.imwrite(
                        str(dump_masks_to.joinpath(str(cls_index_int), f"{instance_id}i.png")), instance_mask * 255
                    )
                instance = Instance(
                    box=[x, y, x + w, y + h],
                    class_index=cls_index,
                    id=instance_id,
                    mask=instance_mask,
                )
                instances.append(instance)
                instance_id += 1
    logger.debug(f"instances count = {len(instances)}")
    return instances
