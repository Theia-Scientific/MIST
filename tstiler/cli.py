#!/usr/bin/env python3

import cv2
import importlib.metadata
import io
import logging
import matplotlib.pyplot as plt
import mimetypes
import numpy as np
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
from tstiler import __app_name__
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


class Metric(Enum):
    IOU = "IoU"
    IOS = "IoS"

    def calculate_bbox(
        self,
        rem_areas: torch.Tensor,
        intersection_area: torch.Tensor,
        areas: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        if self == Metric.IOU:
            union = (rem_areas - intersection_area) + areas[idx]
            return intersection_area / union
        elif self == Metric.IOS:
            smaller = torch.min(rem_areas, areas[idx])
            return intersection_area / smaller
        else:
            raise ValueError("Unknown matching metric")

    def calculate_mask(
        self,
        masks: List[np.ndarray],
        filtered_masks: List[np.ndarray],
        nms_threshold: float,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        if self == Metric.IOU:
            mask_iou = calculate_mask_iou(masks[idx], filtered_masks)
            return mask_iou > nms_threshold
        elif self == Metric.IOS:
            mask_ios = calculate_mask_ios(masks[idx], filtered_masks)
            return mask_ios > nms_threshold
        else:
            raise ValueError("Unknown matching metric")


class UnknownMimeTypeError(Exception):
    def __init__(self, file_name: str):
        self.file_name = file_name


class Tile(BaseModel):
    img: np.ndarray
    index: int
    x_start: int
    y_start: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TileResult(BaseModel):
    boxes: List[List[int]]
    class_indices: List[int]
    masks: np.ndarray
    scores: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TileVisual(BaseModel):
    color: Tuple[int, int, int] = (0, 0, 255)  # BGR
    thickness: int = 3
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class GlobalResult(BaseModel):
    boxes: List[List[int]] = []
    masks: List[np.ndarray] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Instance(BaseModel):
    box: List[int]
    class_index: int
    id: int
    mask: np.ndarray
    scores: List[float]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CombineResult(BaseModel):
    boxes: List[List[int]]
    class_indices: List[int]
    masks: List[np.ndarray]
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


def create_sahi_tiles(
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


def create_patched_tiles(
    src_img: np.ndarray,
    tile_shape: Tuple[int, int] = (640, 640),
    overlap: Tuple[float, float] = (0.2, 0.2),
    show: bool = False,
) -> List[Tile]:
    src_height, src_width, *_ = src_img.shape
    tile_width, tile_height = tile_shape
    overlap_x, overlap_y = overlap
    cross_koef_x = 1 - overlap_x
    cross_koef_y = 1 - overlap_y
    tiles = []
    x_steps = int((src_width - tile_width) / (tile_width * cross_koef_x)) + 1
    y_steps = int((src_height - tile_height) / (tile_height * cross_koef_y)) + 1
    if show:
        plt.figure(figsize=(x_steps * 0.9, y_steps * 0.9))
    count = 0
    for i in range(y_steps):
        for j in range(x_steps):
            x_start = int(tile_width * j * cross_koef_x)
            y_start = int(tile_height * i * cross_koef_y)
            if x_start + tile_width > src_width:
                LOGGER.warning("Error in generating crops along the x-axis")
                continue
            if y_start + tile_height > src_height:
                LOGGER.warning("Error in generating crops along the y-axis")
                continue
            tile_img = src_img[
                y_start : y_start + tile_height, x_start : x_start + tile_width
            ]
            if show:
                plt.subplot(y_steps, x_steps, i * x_steps + j + 1)
                plt.imshow(cv2.cvtColor(tile_img.copy(), cv2.COLOR_BGR2RGB))
                plt.axis("off")
            count += 1
            tiles.append(
                Tile(
                    img=tile_img,
                    index=count,
                    x_start=x_start,
                    y_start=y_start,
                )
            )
    if show:
        plt.show()
    LOGGER.info(f"Number of generated tiles: {count}")
    return tiles


def resize_result(
    global_result: GlobalResult,
    original_image_size: Tuple[int, int],
    resized_image_size: Tuple[int, int],
) -> GlobalResult:
    LOGGER.info("Resizing global result...")
    original_width, original_height = original_image_size
    resized_width, resized_height = resized_image_size
    resized_xyxy = []
    resized_masks = []

    for bbox in global_result.boxes:
        x_min, y_min, x_max, y_max = bbox
        x_min_resized = int(x_min * (original_width / resized_width))
        y_min_resized = int(y_min * (original_height / resized_height))
        x_max_resized = int(x_max * (original_width / resized_width))
        y_max_resized = int(y_max * (original_height / resized_height))
        resized_xyxy.append(
            [x_min_resized, y_min_resized, x_max_resized, y_max_resized]
        )
    for mask in global_result.masks:
        mask_resized = cv2.resize(
            mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        resized_masks.append(mask_resized.astype(np.uint8))
    LOGGER.info("Resizing global result...DONE")
    return GlobalResult(boxes=resized_xyxy, masks=resized_masks)


def calculate_global_result(
    tile: Tile, tile_result: TileResult, src_image_size: Tuple[int, int]
) -> GlobalResult:
    global_result = GlobalResult()
    global_x_start = tile.x_start
    global_y_start = tile.y_start
    global_width, global_height = src_image_size
    tile_height, tile_width, *_ = tile.img.shape
    for bbox in tile_result.boxes:
        tile_x_min, tile_y_min, tile_x_max, tile_y_max = bbox
        global_x_min = tile_x_min + global_x_start
        global_y_min = tile_y_min + global_y_start
        global_x_max = tile_x_max + global_x_start
        global_y_max = tile_y_max + global_y_start
        global_result.boxes.append(
            [global_x_min, global_y_min, global_x_max, global_y_max]
        )
    for mask in tile_result.masks:
        black_image = np.zeros((global_height, global_width))
        mask_resized = cv2.resize(
            np.array(mask).copy(),
            (tile_width, tile_height),
            interpolation=cv2.INTER_NEAREST,
        )
        black_image[
            global_y_start : global_y_start + tile_height,
            global_x_start : global_x_start + tile_width,
        ] = mask_resized
        global_result.masks.append(black_image.astype(np.uint8))
    return global_result


def calculate_mask_iou(mask: np.ndarray, masks: List[np.ndarray]) -> torch.Tensor:
    iou_scores = []
    for other_mask in masks:
        intersection = np.logical_and(mask, other_mask).sum()
        union = np.logical_or(mask, other_mask).sum()
        iou = intersection / union if union != 0 else 0
        iou_scores.append(iou)
    return torch.tensor(iou_scores)


def calculate_mask_ios(mask: np.ndarray, masks: List[np.ndarray]) -> torch.Tensor:
    ios_scores = []
    for other_mask in masks:
        intersection = np.logical_and(mask, other_mask).sum()
        smaller_area = min(mask.sum(), other_mask.sum())
        ios = intersection / smaller_area if smaller_area != 0 else 0
        ios_scores.append(ios)
    return torch.tensor(ios_scores)


def apply_nms(
    boxes: torch.Tensor,
    class_indices: torch.Tensor,
    confidences: torch.Tensor,
    masks: List[np.ndarray],
    match_metric: Metric = Metric.IOS,
    nms_threshold: float = 0.3,
) -> List:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = confidences.argsort()
    keep = []
    while len(order) > 0:
        idx = order[-1]
        keep.append(idx.tolist())
        order = order[:-1]
        if len(order) == 0:
            break
        xx1 = torch.index_select(x1, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
        intersection_width = torch.clamp(xx2 - xx1, min=0.0)
        intersection_height = torch.clamp(yy2 - yy1, min=0.0)
        intersection_area = intersection_width * intersection_height
        rem_areas = torch.index_select(areas, dim=0, index=order)
        match_metric_value = match_metric.calculate_bbox(
            rem_areas, intersection_area, areas, idx
        )
        if len(masks) > 0 and torch.any(match_metric_value > 0):
            mask_mask = match_metric_value > 0
            order_2 = order[mask_mask]
            filtered_masks = [masks[i] for i in order_2]
            mask_mask = match_metric.calculate_mask(
                masks, filtered_masks, nms_threshold, idx
            )
            order_2 = order_2[mask_mask]
            inverse_mask = ~torch.isin(order, order_2)
            order = order[inverse_mask]
        else:
            mask = match_metric_value < nms_threshold
            order = order[mask]
    if class_indices is not None:
        keep = [class_indices[i] for i in keep]
    return keep


def apply_class_nms(
    boxes: torch.Tensor,
    class_indices: torch.Tensor,
    confidences: torch.Tensor,
    masks: List[np.ndarray],
    match_metric: Metric = Metric.IOS,
    nms_threshold: float = 0.3,
):
    all_keeps = []
    for cls_index in torch.unique(class_indices):
        cls_indexes = torch.where(class_indices == cls_index)[0]
        if len(masks) > 0:
            class_masks = [masks[i] for i in cls_indexes]
        else:
            class_masks = []
        keep_indexes = apply_nms(
            boxes[cls_indexes],
            cls_indexes,
            confidences[cls_indexes],
            class_masks,
            match_metric,
            nms_threshold,
        )
        all_keeps.extend(keep_indexes)
    return all_keeps


def combine_results(
    boxes: List[List[int]],
    class_indices: List[int],
    confidences: List[float],
    masks: List[np.ndarray],
    match_metric: Metric = Metric.IOS,
    merge: bool = True,
    merge_classes: List[int] = [],
    nms_threshold: float = 0.3,
) -> List[Instance]:
    LOGGER.info("Combining results...")
    LOGGER.info("Applying NMS...")
    nms_filtered_indices = apply_class_nms(
        torch.tensor(boxes),
        torch.tensor(class_indices),
        torch.tensor(confidences),
        [],
        match_metric,
        nms_threshold,
    )
    LOGGER.info("Applying NMS...DONE")
    instances = []
    instance_id = 0
    if merge:
        LOGGER.info("Merging instances...")
        visited = []
        if len(merge_classes) > 0:
            indices_to_merge = [
                i for i in nms_filtered_indices if class_indices[i] in merge_classes
            ]
        else:
            indices_to_merge = nms_filtered_indices
        for i in indices_to_merge:
            if i not in visited:
                visited.append(i)
                class_i = class_indices[i]
                class_filtered_indices = [
                    c
                    for c in nms_filtered_indices
                    if class_indices[c] == class_i and c not in visited
                ]
                instance = Instance(
                    box=boxes[i],
                    class_index=class_i,
                    id=instance_id,
                    mask=masks[i].copy(),
                    scores=[confidences[i]],
                )
                for j in class_filtered_indices:
                    mask_j = masks[j]
                    # TODO: Possibly change to IOU threshold or something
                    if np.logical_and(instance.mask, mask_j).sum() != 0:
                        x_min_i, y_min_i, x_max_i, y_max_i = instance.box
                        x_min_j, y_min_j, x_max_j, y_max_j = boxes[j]
                        instance.box = [
                            min(x_min_i, x_min_j),
                            min(y_min_i, y_min_j),
                            max(x_max_i, x_max_j),
                            max(y_max_i, y_max_j),
                        ]
                        instance.mask = np.logical_or(instance.mask, mask_j)
                        instance.scores.append(confidences[j])
                        visited.append(j)
                instances.append(instance)
                instance_id += 1
        LOGGER.info("Merging instances...DONE")
    else:
        for i in nms_filtered_indices:
            instance = Instance(
                box=boxes[i],
                class_index=class_indices[i],
                id=instance_id,
                mask=masks[i].copy(),
                scores=[confidences[i]],
            )
            instance_id += 1
    LOGGER.info("Combining results...DONE")
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
                mask_resized.astype(np.uint8),
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
    merge: bool = typer.Option(True, help="Enable or disable merging instances."),
    overlap_height: float = typer.Option(
        0.2,
        help="The amount of overlap in the Y direction as a ratio between 0.0 and 1.0.",
    ),
    overlap_width: float = typer.Option(
        0.2,
        help="The amount of overlap in the X direction as a ratio between 0.0 and 1.0.",
    ),
    show_tiles: bool = typer.Option(False, help="Show tiles in visualization"),
    tile_height: int = typer.Option(640, help="The height of a tile in pixels."),
    tile_width: int = typer.Option(640, help="The width of a tile in pixels."),
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
            tiles = create_sahi_tiles(
                original_img,
                tile_size=(tile_width, tile_height),
                overlap=(overlap_width, overlap_height),
            )
            confidences = []
            boxes = []
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
                if pred.masks is None:
                    masks_data = np.zeros(tile.img.shape)
                else:
                    masks_data = pred.masks.data.cpu().numpy().astype(np.uint8)
                tile_result = TileResult(
                    boxes=pred.boxes.xyxy.cpu().int().tolist(),
                    class_indices=pred.boxes.cls.cpu().int().tolist(),
                    masks=masks_data,
                    scores=pred.boxes.conf.cpu().numpy(),
                )
                global_result = calculate_global_result(tile, tile_result, orig_size)
                confidences.extend(tile_result.scores)
                boxes.extend(global_result.boxes)
                masks.extend(global_result.masks)
                class_indices.extend(tile_result.class_indices)
                if show_tiles:
                    visual_tiles.append(
                        TileVisual(
                            x_min=tile.x_start,
                            y_min=tile.y_start,
                            x_max=tile.x_start + tile_width,
                            y_max=tile.y_start + tile_height,
                        )
                    )
            instances = combine_results(
                boxes, class_indices, confidences, masks, merge=merge
            )
            class_names = [name for _, name in sorted(model.names.items())]
            LOGGER.debug(f"class_names={class_names}")
            all_class_names = [class_names[i] for i in class_indices]
            LOGGER.debug(f"Unmerged Class Counts={Counter(all_class_names)}")
            instance_class_names = [class_names[i.class_index] for i in instances]
            LOGGER.debug(f"Merged Class Counts={Counter(instance_class_names)}")
            visualize(
                instances,
                original_img,
                class_names,
                tiles=visual_tiles,
            )


if __name__ == "__main__":
    app(prog_name=__app_name__)
