#!/usr/bin/env python3

from mist.merging import DumpMaskConfiguration


def test_bump_mask_cfg_enabled():
    assert DumpMaskConfiguration(clazz=True).enabled
    assert DumpMaskConfiguration(data=True).enabled
    assert DumpMaskConfiguration(erode=True).enabled
    assert DumpMaskConfiguration(instance=True).enabled
    assert DumpMaskConfiguration(clazz=True, data=True, erode=True, instance=True)
