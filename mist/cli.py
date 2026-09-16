#!/usr/bin/env python3

import importlib.metadata
import json
import logging
import numpy as np
import os
import supervision as sv
import sys
import tempfile
import typer
import zipfile

from mist import __app_name__, detecting, dump, erosion, utils
from natsort import natsorted
from pathlib import Path
from pydantic import BaseModel
from typing import Annotated, Any
from ultralytics.models import YOLO

LOGGER: logging.Logger = logging.getLogger(__name__)
PREFIX: str = f"{__app_name__.upper()}"

app = typer.Typer(pretty_exceptions_show_locals=False)


class Result(BaseModel):
    source: str
    stats: detecting.Stats


def map_verbosity(count: int) -> str:
    if count == 1:
        log_level = "INFO"
    elif count >= 2:
        log_level = "DEBUG"
    else:
        log_level = "WARNING"
    return log_level


def version_callback(value: bool):
    if value:
        version = importlib.metadata.version(__app_name__)
        print(f"{__app_name__} {version}")
        raise typer.Exit()


def expand_sources(sources: list[Path]) -> list[Path]:
    expanded_sources: list[Path] = []
    for source in sources:
        LOGGER.debug(f"{source=}")
        src = source.expanduser().resolve()
        LOGGER.debug(f"{src=}")
        if src.is_dir():
            expanded_sources.extend(
                [
                    src.joinpath(path)
                    for path in os.listdir(src)
                    if utils.is_image_file_supported(path)
                ]
            )
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
                        expanded_sources.append(
                            Path(zip_fp.extract(name, path=zip_dir))
                        )
        else:
            mime_type = utils.is_image_file_supported(src.name)
            if mime_type:
                expanded_sources.append(src)
    return expanded_sources


@app.command()
def main(
    weights_file: Annotated[
        Path, typer.Argument(help="The path to the YOLO weights file.")
    ],
    sources: Annotated[
        list[Path],
        typer.Argument(help="The images to run tiled inference with the weights file."),
    ],
    device: Annotated[
        str,
        typer.Option(
            help="The device to use for inference. Use 'mps' for Apple Silicon.",
        ),
    ] = "cuda:0",
    dump_class_masks: Annotated[
        bool, typer.Option(help="Creates PNGs of class masks during merging.")
    ] = dump.DEFAULT_MASK_CLASS,
    dump_data_masks: Annotated[
        bool, typer.Option(help="Creates PNGs of data masks during merging.")
    ] = dump.DEFAULT_MASK_DATA,
    dump_erosion_masks: Annotated[
        bool, typer.Option(help="Creates PNGs of erosion masks during merging.")
    ] = dump.DEFAULT_MASK_ERODE,
    dump_instance_masks: Annotated[
        bool,
        typer.Option(
            help="Creates PNGs of instance masks during merging.",
        ),
    ] = dump.DEFAULT_MASK_INSTANCE,
    dump_masks_to: Annotated[
        Path,
        typer.Option(
            help="Location to create PNGs of masks during merging.",
        ),
    ] = dump.DEFAULT_MASK_TO,
    erosion_enabled: Annotated[
        bool,
        typer.Option(
            "--erosion/--no-erosion",
            help=(
                "Enable an erode morphological operation on each instance mask "
                "before merging."
            ),
        ),
    ] = erosion.DEFAULT_ENABLED,
    erosion_iterations: Annotated[
        int,
        typer.Option(
            help="Number of erode operations to execute. Ignored if erosion is disabled.",
        ),
    ] = erosion.DEFAULT_ITERATIONS,
    erosion_size: Annotated[
        int,
        typer.Option(
            help=(
                "Size of the square kernel to use during the erosion operation. "
                "Ignored if erosion is disabled."
            ),
        ),
    ] = erosion.DEFAULT_SIZE,
    inference_confidence: Annotated[
        float,
        typer.Option(
            help="The confidence threshold as a ratio between 0.0. and 1.0.",
        ),
    ] = 0.35,
    inference_iou: Annotated[
        float, typer.Option(help="The Intersection-over-Union for inference.")
    ] = 0.7,
    inference_image_size: Annotated[
        int, typer.Option(help="The size of the image for the YOLO model.")
    ] = 640,
    inference_max_detections: Annotated[
        int,
        typer.Option(
            help="The maximum number of detections for inference.",
        ),
    ] = 1000,
    inference_silent: Annotated[
        bool, typer.Option(help="Silence the output for inference.")
    ] = False,
    merge_classes: Annotated[
        list[int],
        typer.Option(
            "--merge-class",
            "-c",
            help="Only merge instances with these class indices.",
        ),
    ] = detecting.DEFAULT_MERGE_CLASSES,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="The destination for saving annotated images and detections.",
        ),
    ] = None,
    overlap_height: Annotated[
        float,
        typer.Option(
            help=(
                "The amount of overlap in the Y direction as a ratio between 0.0 "
                "and 1.0."
            ),
        ),
    ] = detecting.DEFAULT_OVERLAP_HEIGHT,
    overlap_width: Annotated[
        float,
        typer.Option(
            help=(
                "The amount of overlap in the X direction as a ratio between 0.0 "
                "and 1.0."
            ),
        ),
    ] = detecting.DEFAULT_OVERLAP_WIDTH,
    tile_height: Annotated[
        int, typer.Option(help="The height of a tile in pixels.")
    ] = detecting.DEFAULT_TILE_HEIGHT,
    tile_width: Annotated[
        int, typer.Option(help="The width of a tile in pixels.")
    ] = detecting.DEFAULT_TILE_WIDTH,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose", "-v", help="Print debugging statements to STDOUT.", count=True
        ),
    ] = 0,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            help="Prints the version to STDOUT",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
):
    logging.basicConfig(level=map_verbosity(verbose))
    LOGGER.debug(f"{version=}")
    model = YOLO(weights_file)

    def predict(image: np.ndarray, parameters: dict[str, Any]) -> sv.Detections:
        return sv.Detections.from_ultralytics(
            list(
                model(
                    image,
                    agnostic_nms=parameters.get("agnostic_nms", False),
                    device=parameters.get("device", "cuda:0"),
                    classes=parameters.get("classes", None),
                    conf=parameters.get("confidence", 0.35),
                    imgsz=parameters.get("image_size", 640),
                    iou=parameters.get("iou", 0.7),
                    max_det=parameters.get("maximum_detections", 1000),
                    retina_masks=parameters.get("retina_masks", True),
                    verbose=parameters.get("verbose", False),
                )
            )[0]
        )

    if output is None:
        dst = Path(os.getcwd())
    else:
        dst = output
    LOGGER.debug(f"{dst=}")
    results: list[dict[str, Any]] = []
    for src in expand_sources(sources):
        LOGGER.info("Detecting...")
        result = detecting.run(
            utils.read_image_file(src),
            predict,
            class_names=[name for _, name in sorted(model.names.items())],
            dump_masks=dump.MaskConfiguration(
                clazz=dump_class_masks,
                data=dump_data_masks,
                erode=dump_erosion_masks,
                instance=dump_instance_masks,
                to=dump_masks_to,
            ),
            erosion=erosion.Configuration(
                enabled=erosion_enabled,
                iterations=erosion_iterations,
                size=erosion_size,
            ),
            merge_classes=merge_classes,
            overlap_height=overlap_height,
            overlap_width=overlap_width,
            tile_height=tile_height,
            tile_width=tile_width,
            logger=LOGGER,
            parameters={
                "confidence": inference_confidence,
                "device": device,
                "image_size": inference_image_size,
                "iou": inference_iou,
                "maximum_detections": inference_max_detections,
                "verbose": not inference_silent,
            },
        )
        LOGGER.info("Detecting...DONE")
        results.append(Result(source=str(src), stats=result.stats).model_dump())
        LOGGER.info("Saving...")
        # TODO: Add annotating with Supervision
        # TODO: Save to disk
        LOGGER.info("Saving...DONE")

    json.dump(results, sys.stdout)


if __name__ == "__main__":
    app(prog_name=__app_name__)
