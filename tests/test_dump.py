#!/usr/bin/env python3

import numpy as np
import pytest

from mist.dump import MaskConfiguration


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

    mocker.patch("mist.dump.MaskConfiguration.write", mock_write)


def test_mask_cfg_enabled():
    assert MaskConfiguration(clazz=True).enabled
    assert MaskConfiguration(data=True).enabled
    assert MaskConfiguration(erode=True).enabled
    assert MaskConfiguration(instance=True).enabled
    assert MaskConfiguration(clazz=True, data=True, erode=True, instance=True)


def test_mask_cfg_write(blank_image, tmp_path):
    assert MaskConfiguration(to=tmp_path).write(blank_image, 0, 0, "t")
    assert tmp_path.joinpath("0t.png").exists


def test_mask_cfg_write_class(blank_image, tmp_path):
    assert MaskConfiguration(clazz=True, to=tmp_path).write_class(blank_image, 0, 0)
    assert tmp_path.joinpath("0c.png").exists


def test_mask_cfg_write_class_fail(blank_image, mock_write, tmp_path):
    _ = mock_write
    assert not MaskConfiguration(clazz=True, to=tmp_path).write_class(blank_image, 0, 0)


def test_mask_cfg_write_class_disabled(blank_image, tmp_path):
    assert not MaskConfiguration(clazz=False, to=tmp_path).write_class(
        blank_image, 0, 0
    )


def test_mask_cfg_write_data(blank_image, tmp_path):
    assert MaskConfiguration(data=True, to=tmp_path).write_data(blank_image, 0, 0)
    assert tmp_path.joinpath("0d.png").exists


def test_mask_cfg_write_data_fail(blank_image, mock_write, tmp_path):
    _ = mock_write
    assert not MaskConfiguration(data=True, to=tmp_path).write_data(blank_image, 0, 0)


def test_mask_cfg_write_data_disabled(blank_image, tmp_path):
    assert not MaskConfiguration(data=False, to=tmp_path).write_data(blank_image, 0, 0)


def test_mask_cfg_write_erode(blank_image, tmp_path):
    assert MaskConfiguration(erode=True, to=tmp_path).write_erode(blank_image, 0, 0)
    assert tmp_path.joinpath("0e.png").exists


def test_mask_cfg_write_erode_fail(blank_image, mock_write, tmp_path):
    _ = mock_write
    assert not MaskConfiguration(erode=True, to=tmp_path).write_erode(blank_image, 0, 0)


def test_mask_cfg_write_erode_disabled(blank_image, tmp_path):
    assert not MaskConfiguration(erode=False, to=tmp_path).write_erode(
        blank_image, 0, 0
    )


def test_mask_cfg_write_instance(blank_image, tmp_path):
    assert MaskConfiguration(instance=True, to=tmp_path).write_instance(
        blank_image, 0, 0
    )
    assert tmp_path.joinpath("0i.png").exists


def test_mask_cfg_write_instance_fail(blank_image, mock_write, tmp_path):
    _ = mock_write
    assert not MaskConfiguration(instance=True, to=tmp_path).write_instance(
        blank_image, 0, 0
    )


def test_mask_cfg_write_instance_disabled(blank_image, tmp_path):
    assert not MaskConfiguration(instance=False, to=tmp_path).write_instance(
        blank_image, 0, 0
    )
