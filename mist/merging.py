#!/usr/bin/env python3

import cv2
import logging
import numpy as np
import numpy.typing as npt
import os

from mist import dump, erosion
from mist.instances import Instance
from pydantic import BaseModel

LOGGER: logging.Logger = logging.getLogger(__name__)

DEFAULT_EROSION_CONFIGURATION: erosion.Configuration = erosion.Configuration()
DEFAULT_GROUPS: list[list[int]] = []
DEFAULT_MASK_CONFIGURATION: dump.MaskConfiguration = dump.MaskConfiguration()
DEFAULT_MERGE_CLASSES: list[int] = []


class Mask(BaseModel, arbitrary_types_allowed=True):
    data: npt.NDArray[np.bool]
    offset_x: int
    offset_y: int


def flatten_groups(
    class_id: int, groups: list[list[int]], logger: logging.Logger = LOGGER
) -> list[int]:
    logger.debug(f"{class_id=}")
    logger.debug(f"{groups=}")
    group_cls_ids = [c for group in groups for c in group if class_id in group]
    group_cls_ids.append(class_id)
    logger.debug(f"{group_cls_ids=}")
    return list(set(group_cls_ids))


def find_class_indices(
    class_ids: list[int], group_cls_ids: list[int], logger: logging.Logger = LOGGER
) -> list[int]:
    logger.debug(f"{class_ids=}")
    logger.debug(f"{group_cls_ids=}")
    result = np.nonzero(np.isin(np.array(class_ids), np.array(group_cls_ids)))
    logger.debug(f"{result=}")
    return [int(i) for i in result[0]]


def run(
    class_ids: list[int],
    masks: list[Mask],
    src_image_size: tuple[int, int],
    tile_size: tuple[int, int],
    dump_masks: dump.MaskConfiguration = DEFAULT_MASK_CONFIGURATION,
    erosion: erosion.Configuration = DEFAULT_EROSION_CONFIGURATION,
    groups: list[list[int]] = DEFAULT_GROUPS,
    logger: logging.Logger = LOGGER,
    merge_classes: list[int] = DEFAULT_MERGE_CLASSES,
) -> list[Instance]:
    logger.debug(f"{class_ids=}")
    logger.debug(f"{masks=}")
    logger.debug(f"{src_image_size=}")
    logger.debug(f"{tile_size=}")
    logger.debug(f"{dump_masks=}")
    logger.debug(f"{erosion=}")
    logger.debug(f"{groups=}")
    logger.debug(f"{merge_classes=}")
    tile_width, tile_height = tile_size
    src_image_width, src_image_height = src_image_size
    instance_id = 0
    instances: list[Instance] = []
    if erosion.enabled:
        erosion_kernel = np.ones((erosion.size, erosion.size), np.uint8)
    else:
        erosion_kernel = None
    for cls_id in set(class_ids):
        logger.debug(f"{cls_id=}")
        if (cls_id in merge_classes and len(merge_classes) > 0) or len(
            merge_classes
        ) == 0:
            if dump_masks.enabled:
                os.makedirs(dump_masks.to.joinpath(str(cls_id)), exist_ok=True)
            group_cls_ids = [c for group in groups for c in group if cls_id in group]
            group_cls_ids.append(cls_id)
            logger.debug(f"{group_cls_ids=}")
            cls_indices = find_class_indices(
                class_ids, flatten_groups(cls_id, groups, logger=logger), logger=logger
            )
            class_masks = [masks[i] for i in cls_indices]
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
                    _ = dump_masks.write_erode(erode_img, cls_id, i)
                    mask_data = erode_img.astype(bool)
                class_mask[
                    mask.offset_y : mask.offset_y + tile_height,
                    mask.offset_x : mask.offset_x + tile_width,
                ] += mask_data
                _ = dump_masks.write_class(class_mask.astype(np.uint8), cls_id, i)
                _ = dump_masks.write_data(
                    mask.data.astype(np.uint8),
                    cls_id,
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
                _ = dump_masks.write_instance(instance_mask, cls_id, instance_id)
                instance = Instance(
                    box=[x, y, x + w, y + h],
                    class_index=cls_id,
                    id=instance_id,
                    mask=instance_mask.astype(np.bool),
                )
                instances.append(instance)
                instance_id += 1
    logger.debug(f"instances count = {len(instances)}")
    return instances
