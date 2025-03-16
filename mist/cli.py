#!/usr/bin/env python3

import cv2
import importlib.metadata
import io
import json
import logging
import matplotlib.pyplot as plt
import mimetypes
import numpy as np
import os
import random
import statistics
import tifffile
import torch
import typer
import zipfile

from collections import Counter
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from mist import __app_name__
from typing import List, Optional, Tuple
from ultralytics import YOLO

logging.getLogger("matplotlib.font_manager").disabled = True

LOGGER: logging.Logger = logging.getLogger(__name__)

BIT_DEPTH_DTYPE: str = "uint8"
COLOR_CHANNEL_COUNT: int = 3
NPY_MIME_TYPE: str = "application/numpy"
PREFIX: str = f"{__app_name__.upper()}"
TIFF_MIME_TYPE: str = "image/tiff"

app = typer.Typer(pretty_exceptions_show_locals=False)


class UnknownMimeTypeError(Exception):
    def __init__(self, file_name: str):
        self.file_name = file_name


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


class Instance(BaseModel):
    box: List[int]
    class_index: int
    id: int
    mask: np.ndarray
    scores: List[float]

    model_config = ConfigDict(arbitrary_types_allowed=True)


def map_verbosity(enabled: bool) -> str:
    if enabled:
        return "DEBUG"
    else:
        return "INFO"


def version_callback(value: bool):
    if value:
        version = importlib.metadata.version(__app_name__)
        print(f"{__app_name__} {version}")
        raise typer.Exit()


def count_channels(img: np.ndarray) -> int:
    return img.shape[-1] if img.ndim == 3 else 1


def correct_cv_image(src: np.ndarray) -> np.ndarray:
    if src.dtype == BIT_DEPTH_DTYPE:
        corrected_image = src
    else:
        src_max = np.max(src)
        src_min = np.min(src)
        if src_max == src_min:
            normalized_image = src.astype(np.float32)
        else:
            normalized_image = ((src - src_min) / (src_max - src_min)).astype(
                np.float32
            )
        corrected_image = np.round(normalized_image * 256).astype(BIT_DEPTH_DTYPE)
    if count_channels(corrected_image) < COLOR_CHANNEL_COUNT:
        three_channel_image = cv2.cvtColor(corrected_image, cv2.COLOR_GRAY2BGR)
    else:
        three_channel_image = corrected_image
    return three_channel_image


def decode_data(data: bytes, mime_type: str) -> np.ndarray:
    if mime_type == NPY_MIME_TYPE:
        return np.load(io.BytesIO(data), allow_pickle=True)
    elif mime_type == TIFF_MIME_TYPE:
        return correct_cv_image(tifffile.imread(io.BytesIO(data)))
    else:
        return correct_cv_image(
            cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        )


def read_image_file(source: Path) -> np.ndarray:
    with open(source, "rb+") as f:
        data = f.read()
    mime_type = mimetypes.guess_type(source)[0]
    if mime_type is None:
        raise UnknownMimeTypeError(source.name)
    return decode_data(data, mime_type)


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


def combine(
    class_indices: List[int],
    masks: List[TileMask],
    src_image_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    dump_masks: bool = False,
    merge_classes: List[int] = [],
) -> List[Instance]:
    LOGGER.info("Combining...")
    tile_width, tile_height = tile_size
    src_image_width, src_image_height = src_image_size
    tensor_class_indices = torch.tensor(class_indices)
    instance_id = 0
    instances = []
    for cls_index in torch.unique(tensor_class_indices):
        LOGGER.debug(f"cls_index={cls_index}")
        if (cls_index in merge_classes and len(merge_classes) > 0) or len(
            merge_classes
        ) == 0:
            if dump_masks:
                os.makedirs(f"tmp/{cls_index}", exist_ok=True)
            cls_indexes = torch.where(tensor_class_indices == cls_index)[0]
            class_masks = [masks[i] for i in cls_indexes]
            LOGGER.debug(f"class masks count = {len(class_masks)}")
            class_mask = np.zeros((src_image_height, src_image_width))
            LOGGER.debug(f"class_mask.shape = {class_mask.shape}")
            for i, mask in enumerate(class_masks):
                class_mask[
                    mask.offset_y : mask.offset_y + tile_height,
                    mask.offset_x : mask.offset_x + tile_width,
                ] += mask.data
                if dump_masks:
                    cv2.imwrite(
                        f"tmp/{cls_index}/{i}c.png", class_mask.astype(np.uint8) * 255
                    )
                    cv2.imwrite(
                        f"tmp/{cls_index}/{i}m.png", mask.data.astype(np.uint8) * 255
                    )
            class_mask = class_mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            LOGGER.debug(f"contours count={len(contours)}")
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                instance_mask = np.zeros(
                    (src_image_height, src_image_width), dtype=np.uint8
                )
                cv2.fillPoly(instance_mask, [contour], 1)  # pyright: ignore
                if dump_masks:
                    cv2.imwrite(
                        f"tmp/{cls_index}/{instance_id}i.png", instance_mask * 255
                    )
                instance = Instance(
                    box=[x, y, x + w, y + h],
                    class_index=cls_index,
                    id=instance_id,
                    mask=instance_mask,
                    scores=[],
                )
                instances.append(instance)
                instance_id += 1
    LOGGER.debug(f"instances count = {len(instances)}")
    LOGGER.info("Combining...DONE")
    return instances


def visualize(
    instances: List[Instance],
    img: np.ndarray,
    class_names: List[str],
    tiles: Optional[List[TileVisual]] = None,
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
):
    LOGGER.info("Visualizing results...")
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
    LOGGER.info("Visualizing results...DONE")


@app.command()
def main(
    weights_file: Path = typer.Argument(help="The path to the YOLO weights file."),
    sources: List[Path] = typer.Argument(
        help="The images to run tiled inference with the weights file."
    ),
    device: str = typer.Option("cuda:0", help="The device to use for inference."),
    dump_masks: bool = typer.Option(
        False, help="Creates PNGs of masks during merging."
    ),
    inference_confidence: float = typer.Option(
        0.35, help="The confidence threshold as a ratio between 0.0. and 1.0."
    ),
    inference_iou: float = typer.Option(
        0.7, help="The Intersection-over-Union for inference."
    ),
    inference_image_size: int = typer.Option(
        640, help="The size of the image for the YOLO model."
    ),
    inference_max_detections: int = typer.Option(
        1000, help="The maximum number of detections for inference."
    ),
    inference_silent: bool = typer.Option(
        False, help="Silence the output for inference."
    ),
    merge_classes: List[int] = typer.Option(
        [],
        "--merge-class",
        "-c",
        help="Only merge instances with these class indices.",
    ),
    overlap_height: float = typer.Option(
        0.2,
        help="The amount of overlap in the Y direction as a ratio between 0.0 and 1.0.",
    ),
    overlap_width: float = typer.Option(
        0.2,
        help="The amount of overlap in the X direction as a ratio between 0.0 and 1.0.",
    ),
    random_object_colors: bool = typer.Option(
        False,
        help="Use random colors for each instance; otherwise, select random color for each class.",
    ),
    show_tiles: bool = typer.Option(False, help="Show tiles in visualization"),
    tile_height: int = typer.Option(640, help="The height of a tile in pixels."),
    tile_width: int = typer.Option(640, help="The width of a tile in pixels."),
    visualize_classes: List[int] = typer.Option(
        [],
        "--visualize-classes",
        "-C",
        help="Only visualize instances with these class indices.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print debugging statements to STDOUT.",
        envvar=f"{PREFIX}_VERBOSE",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Prints the version to STDOUT",
        callback=version_callback,
        is_eager=True,
    ),
):
    logging.basicConfig(level=map_verbosity(verbose))
    LOGGER.debug(f"version={version}")
    LOGGER.debug(f"weights_file={weights_file}")
    LOGGER.debug(f"inference_iou={inference_iou}")
    model = YOLO(weights_file)
    for source in sources:
        LOGGER.debug(f"source={sources}")
        src = source.expanduser().resolve()
        LOGGER.debug(f"src={src}")
        if src.is_dir():
            pass
        elif zipfile.is_zipfile(src):
            pass
        else:
            original_img = read_image_file(src)
            orig_height, orig_width, *_ = original_img.shape
            orig_size = (orig_width, orig_height)
            tiles = create_tiles(
                original_img,
                tile_size=(tile_width, tile_height),
                overlap=(overlap_width, overlap_height),
            )
            masks = []
            class_indices = []
            visual_tiles = []
            for tile in tiles:
                results = model(
                    tile.img,
                    agnostic_nms=False,
                    device=device,
                    classes=None,
                    conf=inference_confidence,
                    half=False,
                    imgsz=inference_image_size,
                    iou=inference_iou,
                    max_det=inference_max_detections,
                    retina_masks=True,
                    verbose=not inference_silent,
                )
                pred = results[0]
                tile_class_indices = pred.boxes.cls.cpu().int().tolist()
                class_indices.extend(tile_class_indices)
                if pred.masks is None:
                    masks_data = np.zeros(
                        (len(tile_class_indices), tile_height, tile_width)
                    )
                else:
                    masks_data = pred.masks.data.cpu().numpy().astype(np.uint8)
                for data in masks_data:
                    masks.append(
                        TileMask(
                            data=data, offset_x=tile.x_start, offset_y=tile.y_start
                        )
                    )
                if show_tiles:
                    visual_tiles.append(
                        TileVisual(
                            x_min=tile.x_start,
                            y_min=tile.y_start,
                            x_max=tile.x_start + tile_width,
                            y_max=tile.y_start + tile_height,
                        )
                    )
            instances = combine(
                class_indices,
                masks,
                orig_size,
                (tile_width, tile_height),
                dump_masks=dump_masks,
                merge_classes=merge_classes,
            )
            class_names = [name for _, name in sorted(model.names.items())]
            LOGGER.debug(f"class_names={class_names}")
            all_class_names = [class_names[i] for i in class_indices]
            stats = {"unmerged": Counter(all_class_names)}
            instance_class_names = [class_names[i.class_index] for i in instances]
            stats["merged"] = Counter(instance_class_names)
            print(json.dumps(stats, indent=2))
            visualize(
                instances,
                original_img,
                class_names,
                tiles=visual_tiles,
                random_object_colors=random_object_colors,
                show_classes_list=visualize_classes,
            )


if __name__ == "__main__":
    app(prog_name=__app_name__)
