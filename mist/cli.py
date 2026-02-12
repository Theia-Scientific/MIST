#!/usr/bin/env python3

import importlib.metadata
import json
import logging
import os
import tempfile
import typer
import zipfile

from mist import __app_name__, detecting, utils, visualizing
from natsort import natsorted
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from ultralytics.models import YOLO

logging.getLogger("matplotlib.font_manager").disabled = True

LOGGER: logging.Logger = logging.getLogger(__name__)
PREFIX: str = f"{__app_name__.upper()}"

app = typer.Typer(pretty_exceptions_show_locals=False)


class Result(BaseModel):
    source: Path
    stats: detecting.Stats

    
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


def expand_sources(sources: List[Path]) -> List[Path]:
    expanded_sources = []
    for source in sources:
        LOGGER.debug(f"{source=}")
        src = source.expanduser().resolve()
        LOGGER.debug(f"{src=}")
        if src.is_dir():
            # TODO: Add support for running inference on folder of images
            pass
        elif zipfile.is_zipfile(src):
            zip_dir = tempfile.mkdtemp()
            LOGGER.debug(f"{zip_dir=}")
            with zipfile.ZipFile(src, "r") as zip_fp:
                names = natsorted(zip_fp.namelist())
                for name in names:
                    LOGGER.debug(f"{name=}")
                    mime_type = utils.is_image_file_supported(os.path.basename(name))
                    LOGGER.debug(f"{mime_type=}")
                    if mime_type:
                        expanded_sources.append(Path(zip_fp.extract(name, path=zip_dir)))
        else:
            mime_type = utils.is_image_file_supported(src.name)
            if mime_type:
                expanded_sources.append(src)
    return expanded_sources

   
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
    dump_masks_to: Path = typer.Option(
        Path("tmp"), help="Location to create PNGs of masks during merging."
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
    LOGGER.debug(f"{version=}")
    model = YOLO(weights_file)
    results = []
    for src in expand_sources(sources):
        LOGGER.info("Detecting...")
        result = detecting.run(
            src,
            model,
            device,
            dump_masks=dump_masks,
            dump_masks_to=dump_masks_to,
            inference_confidence=inference_confidence,
            inference_image_size=inference_image_size,
            inference_iou=inference_iou,
            inference_max_detections=inference_max_detections,
            inference_silent=inference_silent,
            merge_classes=merge_classes,
            overlap_height=overlap_height,
            overlap_width=overlap_width,
            tile_height=tile_height,
            tile_width=tile_width,
            logger=LOGGER
        )
        LOGGER.info("Detecting...DONE")
        if show:
            LOGGER.info("Visualizing results...")
            visual_tiles = []
            if show_tiles:
                visual_tiles = result.visual_tiles
            else:
                visual_tiles = []
            visualizing.run(
                result.instances,
                result.original_image,
                result.class_names,
                tiles=visual_tiles,
                random_object_colors=random_object_colors,
                show_classes_list=visualize_classes,
            )
            LOGGER.info("Visualizing results...DONE")
        results.append(Result(source=src, stats=result.stats).model_dump())
    json.dumps(results)


if __name__ == "__main__":
    app(prog_name=__app_name__)
