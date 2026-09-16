#!/usr/bin/env python3

import cv2
import logging
import numpy as np
import numpy.typing as npt
import os
import torch

from mist import dump, erosion
from mist.instances import Instance
from pydantic import BaseModel

LOGGER: logging.Logger = logging.getLogger(__name__)

DEFAULT_EROSION_CONFIGURATION: erosion.Configuration = erosion.Configuration()
DEFAULT_MASK_CONFIGURATION: dump.MaskConfiguration = dump.MaskConfiguration()
DEFAULT_MERGE_CLASSES: list[int] = []


class Mask(BaseModel, arbitrary_types_allowed=True):
    data: npt.NDArray[np.bool]
    offset_x: int
    offset_y: int


def run(
    class_indices: list[int],
    masks: list[Mask],
    src_image_size: tuple[int, int],
    tile_size: tuple[int, int],
    erosion: erosion.Configuration = DEFAULT_EROSION_CONFIGURATION,
    dump_masks: dump.MaskConfiguration = DEFAULT_MASK_CONFIGURATION,
    logger: logging.Logger = LOGGER,
    merge_classes: list[int] = DEFAULT_MERGE_CLASSES,
) -> list[Instance]:
    logger.debug(f"{class_indices=}")
    logger.debug(f"{masks=}")
    logger.debug(f"{src_image_size=}")
    logger.debug(f"{tile_size=}")
    logger.debug(f"{dump_masks=}")
    logger.debug(f"{merge_classes=}")
    tile_width, tile_height = tile_size
    src_image_width, src_image_height = src_image_size
    tensor_class_indices = torch.tensor(class_indices)
    instance_id = 0
    instances: list[Instance] = []
    if erosion.enabled:
        erosion_kernel = np.ones((erosion.size, erosion.size), np.uint8)
    else:
        erosion_kernel = None
    for cls_index in torch.unique(tensor_class_indices):
        logger.debug(f"{cls_index=}")
        cls_index_int = int(cls_index.item())
        logger.debug(f"{cls_index_int=}")
        if (cls_index in merge_classes and len(merge_classes) > 0) or len(
            merge_classes
        ) == 0:
            if dump_masks.enabled:
                os.makedirs(dump_masks.to.joinpath(str(cls_index_int)), exist_ok=True)
            cls_indexes = torch.where(tensor_class_indices == cls_index)[0]
            class_masks = [masks[i] for i in cls_indexes]
            logger.debug(f"{len(class_masks)=}")
            class_mask = np.zeros((src_image_height, src_image_width))
            logger.debug(f"{class_mask.shape=}")
            for i, mask in enumerate(class_masks):
                if erosion_kernel is None:
                    mask_data = mask.data
                else:
                    erode_img = cv2.erode(
                        mask.data.astype(np.uint8),
                        erosion_kernel,
                        iterations=erosion.iterations,
                    )
                    _ = dump_masks.write_erode(erode_img, cls_index_int, i)
                    mask_data = erode_img.astype(bool)
                class_mask[
                    mask.offset_y : mask.offset_y + tile_height,
                    mask.offset_x : mask.offset_x + tile_width,
                ] += mask_data
                _ = dump_masks.write_class(
                    class_mask.astype(np.uint8), cls_index_int, i
                )
                _ = dump_masks.write_data(
                    mask.data.astype(np.uint8),
                    cls_index_int,
                    i,
                )
            class_mask = class_mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            logger.debug(f"{len(contours)=}")
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                instance_mask = np.zeros(
                    (src_image_height, src_image_width), dtype=np.uint8
                )
                _ = cv2.fillPoly(instance_mask, [contour], 1)
                _ = dump_masks.write_instance(instance_mask, cls_index_int, instance_id)
                instance = Instance(
                    box=[x, y, x + w, y + h],
                    class_index=cls_index_int,
                    id=instance_id,
                    mask=instance_mask,
                )
                instances.append(instance)
                instance_id += 1
    logger.debug(f"instances count = {len(instances)}")
    return instances
