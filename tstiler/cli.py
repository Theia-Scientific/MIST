#!/usr/bin/env python3

import cv2
import importlib.metadata
import io
import logging
import matplotlib.pyplot as plt
import mimetypes
import numpy as np
import random
import tifffile
import torch
import typer
import zipfile

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


class GlobalResult(BaseModel):
    boxes: List[List[int]] = []
    masks: List[np.ndarray] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FilteredResult(BaseModel):
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


def create_tiles(
    src_img: np.ndarray,
    tile_shape: Tuple[int, int] = (640, 640),
    overlap: Tuple[float, float] = (0.25, 0.25),
    show: bool = False,
):
    LOGGER.debug(f"tile_shape={tile_shape}")
    LOGGER.debug(f"overlap={overlap}")
    LOGGER.debug(f"show={show}")
    src_height, src_width, *_ = src_img.shape
    tile_width, tile_height = tile_shape
    overlap_x, overlap_y = overlap
    cross_koef_x = 1 - overlap_x
    cross_koef_y = 1 - overlap_y
    data_all_crops = []
    x_steps = int((src_width - tile_width) / (tile_width * cross_koef_x)) + 1
    LOGGER.debug(f"x_steps={x_steps}")
    y_steps = int((src_height - tile_height) / (tile_height * cross_koef_y)) + 1
    LOGGER.debug(f"y_steps={y_steps}")
    # Resizing original image to multiple of tiles. I don't think this is needed
    # for us.
    x_new = round((x_steps - 1) * (tile_width * cross_koef_x) + tile_width)
    LOGGER.debug(f"x_new={x_new}")
    y_new = round((y_steps - 1) * (tile_height * cross_koef_y) + tile_height)
    LOGGER.debug(f"y_new={y_new}")
    resized_img = cv2.resize(src_img, (x_new, y_new))
    if show:
        plt.figure(figsize=(x_steps * 0.9, y_steps * 0.9))
    count = 0
    total_steps = y_steps * x_steps  # Total number of tiles
    LOGGER.debug(f"total_steps={total_steps}")
    for i in range(y_steps):
        for j in range(x_steps):
            x_start = int(tile_width * j * cross_koef_x)
            y_start = int(tile_height * i * cross_koef_y)
            tile_img = resized_img[
                y_start : y_start + tile_height, x_start : x_start + tile_width
            ]
            if show:
                plt.subplot(y_steps, x_steps, i * x_steps + j + 1)
                plt.imshow(cv2.cvtColor(tile_img.copy(), cv2.COLOR_BGR2RGB))
                plt.axis("off")
            count += 1
            data_all_crops.append(
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
    return data_all_crops


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
        # Compute intersection and area of smaller mask
        intersection = np.logical_and(mask, other_mask).sum()
        smaller_area = min(mask.sum(), other_mask.sum())
        # Compute IoU score over smaller area, avoiding division by zero
        ios = intersection / smaller_area if smaller_area != 0 else 0
        ios_scores.append(ios)
    return torch.tensor(ios_scores)


def apply_nms(
    confidences: torch.Tensor,
    boxes: torch.Tensor,
    masks: List[np.ndarray],
    match_metric="IOS",
    nms_threshold=0.3,
) -> List:
    LOGGER.info("Applying NMS...")
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
        if match_metric == "IOU":
            union = (rem_areas - intersection_area) + areas[idx]
            match_metric_value = intersection_area / union
        elif match_metric == "IOS":
            smaller = torch.min(rem_areas, areas[idx])
            match_metric_value = intersection_area / smaller
        else:
            raise ValueError("Unknown matching metric")
        if len(masks) > 0 and torch.any(match_metric_value > 0):
            mask_mask = match_metric_value > 0
            order_2 = order[mask_mask]
            filtered_masks = [masks[i] for i in order_2]
            if match_metric == "IOU":
                mask_iou = calculate_mask_iou(masks[idx], filtered_masks)
                mask_mask = mask_iou > nms_threshold
            elif match_metric == "IOS":
                mask_ios = calculate_mask_ios(masks[idx], filtered_masks)
                mask_mask = mask_ios > nms_threshold
            order_2 = order_2[mask_mask]
            inverse_mask = ~torch.isin(order, order_2)
            order = order[inverse_mask]
        else:
            mask = match_metric_value < nms_threshold
            order = order[mask]
    LOGGER.info("Applying NMS...DONE")
    return keep


def combine_results(
    confidences: List[float],
    boxes: List[List[int]],
    masks: List[np.ndarray],
    class_indices: List[int],
) -> FilteredResult:
    LOGGER.info("Combining results...")
    filtered_indices = apply_nms(torch.tensor(confidences), torch.tensor(boxes), masks)
    LOGGER.info("Combining results...DONE")
    return FilteredResult(
        boxes=[boxes[i] for i in filtered_indices],
        class_indices=[class_indices[i] for i in filtered_indices],
        masks=[masks[i] for i in filtered_indices],
        scores=[confidences[i] for i in filtered_indices],
    )


def visualize(
    results: FilteredResult,
    img: np.ndarray,
    class_names=[str],
    segment=True,
    show_boxes=True,
    show_class=True,
    fill_mask=False,
    alpha=0.3,
    color_class_background=(0, 0, 255),
    color_class_text=(255, 255, 255),
    thickness=4,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.5,
    delta_colors=3,
    dpi=150,
    random_object_colors=False,
    show_confidences=False,
    axis_off=True,
    show_classes_list=[],
    list_of_class_colors=None,
):
    labeled_image = img.copy()
    if random_object_colors:
        random.seed(int(delta_colors))
    for i in range(len(results.class_indices)):
        if len(class_names) > 0:
            class_name = str(class_names[results.class_indices[i]])
        else:
            class_name = str(results.class_indices[i])
        if show_classes_list and int(results.class_indices[i]) not in show_classes_list:
            continue
        if random_object_colors:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        elif list_of_class_colors is None:
            random.seed(int(results.class_indices[i] + delta_colors))
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        else:
            color = list_of_class_colors[results.class_indices[i]]
        box = results.boxes[i]
        x_min, y_min, x_max, y_max = box
        if segment and len(results.masks) > 0:
            mask = results.masks[i]
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
        if show_boxes:
            cv2.rectangle(
                labeled_image, (x_min, y_min), (x_max, y_max), color, thickness
            )
        if show_class:
            if show_confidences:
                label = f"{str(class_name)} {results.scores[i]:.2}"
            else:
                label = str(class_name)
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            background_color = (
                color_class_background[results.class_indices[i]]
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
    if axis_off:
        plt.axis("off")
    plt.show()


@app.command()
def main(
    weights_file: Path = typer.Argument(help="The path to the YOLO weights file."),
    sources: List[Path] = typer.Argument(
        help="The images to run tiled inference with the weights file."
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
            tiles = create_tiles(original_img, show=verbose)
            confidences = []
            boxes = []
            masks = []
            class_indices = []
            for tile in tiles:
                results = model(
                    tile.img,
                    agnostic_nms=False,
                    device="cuda:0",
                    classes=None,
                    conf=0.35,
                    half=False,
                    imgsz=640,
                    iou=0.7,
                    max_det=1000,
                    retina_masks=True,
                    verbose=True,
                )
                pred = results[0]
                tile_result = TileResult(
                    boxes=pred.boxes.xyxy.cpu().int().tolist(),
                    class_indices=pred.boxes.cls.cpu().int().tolist(),
                    masks=pred.masks.data.cpu().numpy().astype(np.uint8),
                    scores=pred.boxes.conf.cpu().numpy(),
                )
                global_result = calculate_global_result(
                    tile, tile_result, (orig_width, orig_height)
                )
                confidences.extend(tile_result.scores)
                boxes.extend(global_result.boxes)
                masks.extend(global_result.masks)
                class_indices.extend(tile_result.class_indices)
            filtered_results = combine_results(confidences, boxes, masks, class_indices)
            visualize(
                filtered_results,
                original_img,
                [name for _, name in sorted(model.names.items())],
            )


if __name__ == "__main__":
    app(prog_name=__app_name__)
