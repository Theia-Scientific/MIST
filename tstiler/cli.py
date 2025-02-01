#!/usr/bin/env python3

import cv2
import importlib.metadata
import io
import logging
import matplotlib.pyplot as plt
import mimetypes
import numpy as np
import tifffile
import typer
import zipfile

from pathlib import Path
from pydantic import BaseModel, ConfigDict
from tstiler import __app_name__
from typing import List, Optional, Tuple

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
        normalized_image = ((src - np.min(src)) / (np.max(src) - np.min(src))).astype(
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
            # Check for residuals
            if x_start + tile_width > src_width:
                LOGGER.warning("Error in generating tiles along the x-axis")
                continue
            if y_start + tile_height > src_height:
                LOGGER.warning("Error in generating crops along the y-axis")
                continue
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
    for source in sources:
        LOGGER.debug(f"source={sources}")
        src = source.expanduser().resolve()
        LOGGER.debug(f"src={src}")
        if src.is_dir():
            pass
        elif zipfile.is_zipfile(src):
            pass
        else:
            create_tiles(read_image_file(src), show=verbose)


if __name__ == "__main__":
    app(prog_name=__app_name__)
