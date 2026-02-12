#!/usr/bin/env python3

import cv2
import io
import mimetypes
import numpy as np
import tifffile

from pathlib import Path
from typing import Optional

BIT_DEPTH_DTYPE: str = "uint8"
COLOR_CHANNEL_COUNT: int = 3
JPEG_MIME_TYPE: str = "image/jpeg"
NPY_MIME_TYPE: str = "application/numpy"
PNG_MIME_TYPE: str = "image/png"
TIFF_MIME_TYPE: str = "image/tiff"

mimetypes.add_type(NPY_MIME_TYPE, ".npy")

class ImageDecodeError(Exception):
    pass


class UnsupportedImageFile(Exception):
    def __init__(self, path: Path):
        self.path = path


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
        src = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if src is None:
            raise ImageDecodeError()
        return correct_cv_image(src)

    
def is_image_file_supported(file_name: str) -> Optional[str]:
    SUPPORTED_MIME_TYPES = [JPEG_MIME_TYPE, NPY_MIME_TYPE, PNG_MIME_TYPE, TIFF_MIME_TYPE]
    mime_type, _ = mimetypes.guess_type(file_name)
    if mime_type in SUPPORTED_MIME_TYPES:
        return mime_type
    else:
        return None
   

def read_image_file(source: Path) -> np.ndarray:
    mime_type = is_image_file_supported(source.name)
    if mime_type is None:
        raise UnsupportedImageFile(source)
    with open(source, "rb+") as f:
        data = f.read()
    return decode_data(data, mime_type)


