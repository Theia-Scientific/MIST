#!/usr/bin/env python3

import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import statistics

from mist.instances import Instance
from pydantic import BaseModel
from typing import List, Optional, Tuple

LOGGER: logging.Logger = logging.getLogger(__name__)

class Tile(BaseModel):
    color: Tuple[int, int, int] = (0, 0, 255)  # BGR
    thickness: int = 3
    x_min: int
    y_min: int
    x_max: int
    y_max: int


def visualize(
    instances: List[Instance],
    img: np.ndarray,
    class_names: List[str],
    tiles: Optional[List[Tile]] = None,
    segment: bool = True,
    show_boxes: bool = False,
    show_class: bool = False,
    fill_mask: bool = True,
    alpha: float = 0.3,
    color_class_background: Tuple[int, int, int] = (0, 0, 255),
    color_class_text: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 4,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.5,
    delta_colors: int = 3,
    dpi: int = 150,
    random_object_colors=True,
    show_confidences=False,
    show_classes_list=[],
    list_of_class_colors=None,
    logger: logging.Logger = LOGGER
):
    logger.debug(f"{instances=}")
    logger.debug(f"{img=}")
    logger.debug(f"{class_names=}")
    logger.debug(f"{tiles=}")
    logger.debug(f"{segment=}")
    logger.debug(f"{show_boxes=}")
    logger.debug(f"{show_class=}")
    logger.debug(f"{fill_mask=}")
    logger.debug(f"{alpha=}")
    logger.debug(f"{color_class_background=}")
    logger.debug(f"{color_class_text=}")
    logger.debug(f"{thickness=}")
    logger.debug(f"{font=}")
    logger.debug(f"{font_scale=}")
    logger.debug(f"{delta_colors=}")
    logger.debug(f"{dpi=}")
    logger.debug(f"{random_object_colors=}")
    logger.debug(f"{show_confidences=}")
    logger.debug(f"{show_classes_list=}")
    logger.debug(f"{list_of_class_colors=}")
    labeled_image = img.copy()
    if random_object_colors:
        random.seed(int(delta_colors))
    for instance in instances:
        if len(class_names) > 0:
            class_name = str(class_names[instance.class_index])
        else:
            class_name = str(instance.class_index)
        if show_classes_list and int(instance.class_index) not in show_classes_list:
            continue
        if random_object_colors:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        elif list_of_class_colors is None:
            random.seed(int(instance.class_index + delta_colors))
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        else:
            color = list_of_class_colors[instance.class_index]
        box = instance.box
        x_min, y_min, x_max, y_max = box
        logger.debug(f"{x_min=}")
        logger.debug(f"{y_min=}")
        logger.debug(f"{x_max=}")
        logger.debug(f"{y_max=}")
        if segment:
            mask = instance.mask.astype(np.uint8)
            mask_resized = cv2.resize(
                np.array(mask),
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_contours, _ = cv2.findContours(
                mask_resized,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if fill_mask:
                if alpha == 1:
                    cv2.fillPoly(labeled_image, pts=mask_contours, color=color)
                else:
                    color_mask = np.zeros_like(img)
                    color_mask[mask_resized > 0] = color
                    labeled_image = cv2.addWeighted(
                        labeled_image, 1, color_mask, alpha, 0
                    )
            cv2.drawContours(labeled_image, mask_contours, -1, color, thickness)
        if tiles is not None:
            for tile in tiles:
                cv2.rectangle(
                    labeled_image,
                    (tile.x_min, tile.y_min),
                    (tile.x_max, tile.y_max),
                    tile.color,
                    tile.thickness,
                )
        if show_boxes:
            cv2.rectangle(
                labeled_image, (x_min, y_min), (x_max, y_max), color, thickness
            )
        if show_class:
            if show_confidences:
                label = f"{str(class_name)} {statistics.fmean(instance.scores):.2}"
            else:
                label = str(class_name)
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            background_color = (
                color_class_background[instance.class_index]
                if isinstance(color_class_background, list)
                else color_class_background
            )
            cv2.rectangle(
                labeled_image,
                (x_min, y_min),
                (x_min + text_width + 5, y_min + text_height + 5),
                background_color,
                -1,
            )
            cv2.putText(
                labeled_image,
                label,
                (x_min + 5, y_min + text_height),
                font,
                font_scale,
                color_class_text,
                thickness=thickness,
            )
    plt.figure(figsize=(8, 8), dpi=dpi)
    labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
    plt.imshow(labeled_image)
    plt.axis("off")
    plt.show()
