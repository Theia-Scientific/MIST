#!/usr/bin/env python3

import cv2
import importlib.metadata
import numpy as np
import numpy.typing as npt
import os
import pytest
import shutil
import supervision as sv
import torch
import ultralytics
import zipfile

from mist import __app_name__
from mist.cli import (
    app,
    expand_sources,
    map_verbosity,
)
from pathlib import Path
from pytest_mock import MockerFixture
from typer.testing import CliRunner
from typing import Any, Iterator
from ultralytics.engine.results import Results

runner = CliRunner()


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    txt_file = tmp_path.joinpath("test.txt")
    with open(txt_file, "+w") as fp:
        _ = fp.write("Hello World")
    return txt_file


@pytest.fixture
def zip_file(
    blank_jpg: Path, blank_png: Path, blank_npy: Path, blank_tif: Path, tmp_path: Path
) -> Path:
    zip_path = tmp_path.joinpath("images.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        _ = zf.write(blank_jpg)
        _ = zf.write(blank_png)
        _ = zf.write(blank_npy)
        _ = zf.write(blank_tif)
    return zip_path


@pytest.fixture
def dir_with_images(
    blank_jpg: Path, blank_png: Path, blank_npy: Path, blank_tif: Path, tmp_path: Path
) -> Path:
    _ = shutil.move(blank_jpg, tmp_path.joinpath("image0.jpg"))
    _ = shutil.move(blank_png, tmp_path.joinpath("image1.png"))
    _ = shutil.move(blank_npy, tmp_path.joinpath("image2.npy"))
    _ = shutil.move(blank_tif, tmp_path.joinpath("image3.tif"))
    return tmp_path


@pytest.fixture
def dir_with_images_and_text(
    blank_jpg: Path,
    blank_png: Path,
    blank_npy: Path,
    blank_tif: Path,
    text_file: Path,
    tmp_path: Path,
) -> Path:
    _ = shutil.move(blank_jpg, tmp_path.joinpath("image0.jpg"))
    _ = shutil.move(blank_png, tmp_path.joinpath("image1.png"))
    _ = shutil.move(blank_npy, tmp_path.joinpath("image2.npy"))
    _ = shutil.move(blank_tif, tmp_path.joinpath("image3.tif"))
    _ = shutil.move(text_file, tmp_path.joinpath("text.txt"))
    return tmp_path


def test_map_verbosity():
    assert map_verbosity(0) == "WARNING"
    assert map_verbosity(1) == "INFO"
    assert map_verbosity(2) == "DEBUG"
    assert map_verbosity(3) == "DEBUG"


def test_expand_sources_with_single_supported_file(blank_png: Path):
    actual = expand_sources([blank_png])
    assert len(actual) == 1
    assert blank_png in actual


def test_expand_sources_with_multiple_supported_files(blank_png: Path, bus_jpg: Path):
    actual = expand_sources([blank_png, bus_jpg])
    assert len(actual) == 2
    assert blank_png in actual
    assert bus_jpg in actual


def test_expand_sources_with_no_supported_file(text_file: Path):
    actual = expand_sources([text_file])
    assert len(actual) == 0


def test_expand_sources_with_zip_file(zip_file: Path):
    paths = expand_sources([zip_file])
    assert len(paths) == 4
    for path in paths:
        assert isinstance(path, Path)


def test_expand_sources_with_directory(dir_with_images: Path):
    paths = expand_sources([dir_with_images])
    assert len(paths) == 4
    for path in paths:
        assert isinstance(path, Path)


def test_expand_sources_with_directory_unsupported(dir_with_images_and_text: Path):
    paths = expand_sources([dir_with_images_and_text])
    assert len(paths) == 4
    for path in paths:
        assert isinstance(path, Path)


def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_app_version():
    version = importlib.metadata.version(__app_name__)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"{__app_name__} {version}" in result.stdout


def test_app_image(
    bus_class_ids: list[int],
    bus_image: npt.NDArray[np.uint8],
    bus_jpg: Path,
    bus_masks: npt.NDArray[np.bool],
    mocker: MockerFixture,
    tmp_path: Path,
    weights_file: Path,
    yolo: ultralytics.YOLO,
):
    boxes = torch.tensor(
        [
            [box[0], box[1], box[2], box[3], 0.9, class_id]
            for box, class_id in zip(sv.mask_to_xyxy(bus_masks), bus_class_ids)
        ]
    )
    class_names = [
        "bus",
        "cat",
        "dog",
        "horse",
        "bird",
        "car",
        "cellphone",
        "tv",
        "clock",
        "person",
    ]
    names = {index: name for index, name in enumerate(class_names)}
    results = [
        Results(
            bus_image,
            str(bus_jpg),
            names,
            boxes=boxes,
            masks=torch.tensor(bus_masks),
            probs=None,
            obb=None,
            speed=None,
            semantic_mask=None,
            depth=None,
        )
    ]

    mock_names = mocker.patch(
        "ultralytics.YOLO.names", new_callable=mocker.PropertyMock
    )
    mock_names.return_value = names

    def mock_init(
        self, model: str | Path, task: str | None = None, verbose: bool = False
    ) -> None:
        _ = self
        _ = model
        _ = task
        _ = verbose

    def mock_call(
        self,
        source: npt.NDArray[np.uint8],
        stream: bool = False,
        **kwargs: Any,
    ) -> Iterator[Results | torch.Tensor] | list[Results] | list[torch.Tensor]:
        _ = self
        _ = source
        _ = stream
        _ = kwargs
        return results

    mocker.patch.object(ultralytics.YOLO, "__init__", mock_init)
    mocker.patch.object(ultralytics.YOLO, "__call__", mock_call)
    yolo = ultralytics.YOLO("yolo26n-seg.pt")
    assert hasattr(yolo, "names")
    assert yolo.names == names
    actual = yolo(bus_image)
    assert isinstance(actual, list)
    assert actual == results
    # result = runner.invoke(
    #     app,
    #     ["--device=cpu", "--output", str(tmp_path), str(weights_file), str(blank_png)],
    # )
    # assert result.exit_code == 0


def test_app_no_output(
    bus_jpg: Path, mocker: MockerFixture, tmp_path: Path, weights_file: Path
):

    def mock_os_getcwd() -> str:
        return str(tmp_path)

    _ = mocker.patch("os.getcwd", mock_os_getcwd)

    result = runner.invoke(
        app,
        ["--device=cpu", str(weights_file), str(bus_jpg)],
    )
    assert result.exit_code == 0
    assert len(os.listdir(tmp_path)) == 1
    assert tmp_path.joinpath(bus_jpg.stem + "_mist.png").exists()


def test_app_directory(dir_with_images: Path, tmp_path: Path, weights_file: Path):
    result = runner.invoke(
        app,
        [
            "--device=cpu",
            "--output",
            str(tmp_path),
            str(weights_file),
            str(dir_with_images),
        ],
    )
    assert result.exit_code == 0


def test_app_zip(zip_file: Path, tmp_path: Path, weights_file: Path):
    result = runner.invoke(
        app,
        ["--device=cpu", "--output", str(tmp_path), str(weights_file), str(zip_file)],
    )
    assert result.exit_code == 0


def test_app_fail_to_save_image(
    bus_jpg: Path, mocker: MockerFixture, tmp_path: Path, weights_file: Path
):
    def mock_cv2_imwrite(dst: str, img: cv2.typing.MatLike) -> bool:
        _ = dst
        _ = img
        return False

    _ = mocker.patch("cv2.imwrite", mock_cv2_imwrite)
    result = runner.invoke(
        app,
        ["--device=cpu", "--output", str(tmp_path), str(weights_file), str(bus_jpg)],
    )
    assert result.exit_code == 0
    assert len(os.listdir(tmp_path)) == 0
