#!/usr/bin/env python3


from mist.merging import flatten_groups


def test_flatten_groups():
    result = flatten_groups(0, [])
    assert len(result) == 1
    assert result[0] == 0

    result = flatten_groups(0, [[0, 2, 6]])
    assert len(result) == 3
    assert result[0] == 0
    assert result[1] == 2
    assert result[2] == 6

    result = flatten_groups(1, [[0, 2, 6]])
    assert len(result) == 1
    assert result[0] == 1

    result = flatten_groups(0, [[0, 2, 6], [1, 3, 5]])
    assert len(result) == 3
    assert result[0] == 0
    assert result[1] == 2
    assert result[2] == 6

    result = flatten_groups(1, [[0, 2, 6], [1, 3, 5]])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 3
    assert result[2] == 5
