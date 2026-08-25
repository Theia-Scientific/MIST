#!/usr/bin/env python3

import numpy as np
import pytest

from mist.merging import DumpMaskConfiguration


@pytest.fixture
def mock_write(mocker):
    def mock_write(
        self, img: np.ndarray, class_index: int, instance_index: int, suffix: str
    ) -> bool:
        _ = self
        _ = img
        _ = class_index
        _ = instance_index
        _ = suffix

        return False

    mocker.patch("mist.merging.DumpMaskConfiguration.write", mock_write)


def test_dump_mask_cfg_enabled():
    assert DumpMaskConfiguration(clazz=True).enabled
    assert DumpMaskConfiguration(data=True).enabled
    assert DumpMaskConfiguration(erode=True).enabled
    assert DumpMaskConfiguration(instance=True).enabled
    assert DumpMaskConfiguration(clazz=True, data=True, erode=True, instance=True)


def test_dump_mask_cfg_write(blank_image, tmp_path):
    assert DumpMaskConfiguration(to=tmp_path).write(blank_image, 0, 0, "t")
    assert tmp_path.joinpath("0t.png").exists


def test_dump_mask_cfg_write_class(blank_image, tmp_path):
    assert DumpMaskConfiguration(clazz=True, to=tmp_path).write_class(blank_image, 0, 0)
    assert tmp_path.joinpath("0c.png").exists


def test_dump_mask_cfg_write_class_fail(blank_image, mock_write, tmp_path):
    _ = mock_write
    assert not DumpMaskConfiguration(clazz=True, to=tmp_path).write_class(
        blank_image, 0, 0
    )


def test_dump_mask_cfg_write_class_disabled(blank_image, tmp_path):
    assert not DumpMaskConfiguration(clazz=False, to=tmp_path).write_class(
        blank_image, 0, 0
    )


def test_dump_mask_cfg_write_data(blank_image, tmp_path):
    assert DumpMaskConfiguration(data=True, to=tmp_path).write_data(blank_image, 0, 0)
    assert tmp_path.joinpath("0d.png").exists


def test_dump_mask_cfg_write_data_fail(blank_image, mock_write, tmp_path):
    _ = mock_write
    assert not DumpMaskConfiguration(data=True, to=tmp_path).write_data(
        blank_image, 0, 0
    )


def test_dump_mask_cfg_write_data_disabled(blank_image, tmp_path):
    assert not DumpMaskConfiguration(data=False, to=tmp_path).write_data(
        blank_image, 0, 0
    )


def test_dump_mask_cfg_write_erode(blank_image, tmp_path):
    assert DumpMaskConfiguration(erode=True, to=tmp_path).write_erode(blank_image, 0, 0)
    assert tmp_path.joinpath("0e.png").exists


def test_dump_mask_cfg_write_erode_fail(blank_image, mock_write, tmp_path):
    _ = mock_write
    assert not DumpMaskConfiguration(erode=True, to=tmp_path).write_erode(
        blank_image, 0, 0
    )


def test_dump_mask_cfg_write_erode_disabled(blank_image, tmp_path):
    assert not DumpMaskConfiguration(erode=False, to=tmp_path).write_erode(
        blank_image, 0, 0
    )


def test_dump_mask_cfg_write_instance(blank_image, tmp_path):
    assert DumpMaskConfiguration(instance=True, to=tmp_path).write_instance(
        blank_image, 0, 0
    )
    assert tmp_path.joinpath("0i.png").exists


def test_dump_mask_cfg_write_instance_fail(blank_image, mock_write, tmp_path):
    _ = mock_write
    assert not DumpMaskConfiguration(instance=True, to=tmp_path).write_instance(
        blank_image, 0, 0
    )


def test_dump_mask_cfg_write_instance_disabled(blank_image, tmp_path):
    assert not DumpMaskConfiguration(instance=False, to=tmp_path).write_instance(
        blank_image, 0, 0
    )
