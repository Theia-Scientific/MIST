#!/usr/bin/env python3

import cv2
import logging
import numpy as np
import os
import torch

from pathlib import Path
from mist import erosion
from mist.instances import Instance
from pydantic import BaseModel, ConfigDict
from typing import List, Tuple

LOGGER: logging.Logger = logging.getLogger(__name__)


class DumpMaskConfiguration(BaseModel):
    clazz: bool = False
    data: bool = False
    erode: bool = False
    instance: bool = False
    to: Path = Path("tmp")

    @property
    def enabled(self) -> bool:
        return self.clazz or self.data or self.erode or self.instance

    def write(
        self, img: np.ndarray, class_index: int, instance_index: int, suffix: str
    ) -> bool:
        return cv2.imwrite(
            str(
                self.to.joinpath(str(class_index)).joinpath(
                    f"{instance_index}{suffix}.png"
                )
            ),
            img * 255,
        )

    def write_class(
        self, img: np.ndarray, class_index: int, instance_index: int
    ) -> bool:
        if self.clazz:
            return self.write(img, class_index, instance_index, "c")
        else:
            return False

    def write_data(
        self, img: np.ndarray, class_index: int, instance_index: int
    ) -> bool:
        if self.data:
            return self.write(img, class_index, instance_index, "m")
        else:
            return False

    def write_erode(
        self, img: np.ndarray, class_index: int, instance_index: int
    ) -> bool:
        if self.erode:
            return self.write(img, class_index, instance_index, "e")
        else:
            return False

    def write_instance(
        self, img: np.ndarray, class_index: int, instance_index: int
    ) -> bool:
        if self.instance:
            return self.write(img, class_index, instance_index, "i")
        else:
            return False


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
    erosion: erosion.Configuration = erosion.Configuration(),
    dump_masks: DumpMaskConfiguration = DumpMaskConfiguration(),
    logger: logging.Logger = LOGGER,
    merge_classes: List[int] = [],
) -> List[Instance]:
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
    instances = []
    if erosion.enabled:
        erosion_kernel = np.ones((erosion.size, erosion.size), np.uint8)
    else:
        erosion_kernel = None
    for cls_index in torch.unique(tensor_class_indices):
        logger.debug(f"{cls_index=}")
        cls_index_int = cls_index.item()
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
                    dump_masks.write_erode(erode_img, cls_index_int, i)
                    mask_data = erode_img.astype(bool)
                class_mask[
                    mask.offset_y : mask.offset_y + tile_height,
                    mask.offset_x : mask.offset_x + tile_width,
                ] += mask_data
                dump_masks.write_class(class_mask.astype(np.uint8), cls_index_int, i)
                dump_masks.write_data(
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
                cv2.fillPoly(instance_mask, [contour], 1)
                dump_masks.write_instance(instance_mask, cls_index_int, instance_id)
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
