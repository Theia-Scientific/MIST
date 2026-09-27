#!/usr/bin/env python3


from mist.merging import find_class_indices, flatten_groups


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


def test_find_class_indices():
    class_ids = [6, 6, 3, 3, 3, 1, 1, 1, 1, 0, 2, 2, 2, 4, 5, 5, 5]

    class_indices = find_class_indices(
        class_ids, flatten_groups(0, [[0, 2, 4, 6], [1, 3, 5]])
    )
    assert len(class_indices) == 7
    assert class_indices == [0, 1, 9, 10, 11, 12, 13]

    class_indices = find_class_indices(
        class_ids, flatten_groups(1, [[0, 2, 4, 6], [1, 3, 5]])
    )
    assert len(class_indices) == 10
    assert class_indices == [2, 3, 4, 5, 6, 7, 8, 14, 15, 16]

    class_indices = find_class_indices(
        class_ids, flatten_groups(0, [[2, 4, 6], [1, 3, 5]])
    )
    assert len(class_indices) == 1
    assert class_indices == [9]

    class_indices = find_class_indices(class_ids, flatten_groups(0, []))
    assert len(class_indices) == 1
    assert class_indices == [9]

    class_indices = find_class_indices(class_ids, flatten_groups(2, []))
    assert len(class_indices) == 3
    assert class_indices == [10, 11, 12]
