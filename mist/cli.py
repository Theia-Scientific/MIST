#!/usr/bin/env python3

import importlib.metadata
import json
import logging
import numpy as np
import typer
import zipfile

from collections import Counter
from pathlib import Path
from mist import __app_name__
from mist.merging import merge
from mist.tiling import create_tiles, TileMask, TileVisual
from mist.utils import read_image_file
from mist.visualizing import visualize
from typing import List, Optional
from ultralytics.models import YOLO

logging.getLogger("matplotlib.font_manager").disabled = True

LOGGER: logging.Logger = logging.getLogger(__name__)

PREFIX: str = f"{__app_name__.upper()}"

app = typer.Typer(pretty_exceptions_show_locals=False)


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
    show: bool = typer.Option(True, help="Show visualization"),
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
            LOGGER.info("Reading image file...")
            original_img = read_image_file(src)
            LOGGER.info("Reading image file...DONE")
            orig_height, orig_width, *_ = original_img.shape
            orig_size = (orig_width, orig_height)
            LOGGER.info("Creating tiles...")
            tiles = create_tiles(
                original_img,
                tile_size=(tile_width, tile_height),
                overlap=(overlap_width, overlap_height),
            )
            LOGGER.info("Creating tiles...DONE")
            masks = []
            class_indices = []
            visual_tiles = []
            for index, tile in enumerate(tiles):
                LOGGER.info(f"Running inference on {index} tile...")
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
                LOGGER.info(f"Running inference on {index} tile...DONE")
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
            LOGGER.info("Merging results...")
            instances = merge(
                class_indices,
                masks,
                orig_size,
                (tile_width, tile_height),
                dump_masks=dump_masks,
                merge_classes=merge_classes,
            )
            LOGGER.info("Merging results...DONE")
            class_names = [name for _, name in sorted(model.names.items())]
            LOGGER.debug(f"class_names={class_names}")
            all_class_names = [class_names[i] for i in class_indices]
            stats = {"unmerged": Counter(all_class_names)}
            instance_class_names = [class_names[i.class_index] for i in instances]
            stats["merged"] = Counter(instance_class_names)
            print(json.dumps(stats, indent=2))
            if show:
                LOGGER.info("Visualizing results...")
                visualize(
                    instances,
                    original_img,
                    class_names,
                    tiles=visual_tiles,
                    random_object_colors=random_object_colors,
                    show_classes_list=visualize_classes,
                )
                LOGGER.info("Visualizing results...DONE")


if __name__ == "__main__":
    app(prog_name=__app_name__)
