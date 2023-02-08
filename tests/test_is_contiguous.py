from __future__ import annotations

import pytest

import nope

from array_info import ArrayInfo


CONTIGUOUS_ARRAYS_INFO = (
    ArrayInfo((), (), 1),
    ArrayInfo((1, 3), (12, 4), 4),
    ArrayInfo((2, 3, 4), (48, 16, 4), 4),
    ArrayInfo((5, 12, 10), (240, 20, 2), 2),
    ArrayInfo((3, 5, 12, 10), (1200, 240, 20, 2), 2)
)

NOT_CONTIGUOUS_ARRAYS_INFO = (
    ArrayInfo((2, ), (8, ), 4),
    ArrayInfo((3, 2), (10, 3), 1),
    ArrayInfo((3, 4, 10), (240, 30, 1), 1)
)


@pytest.mark.parametrize("array_info", CONTIGUOUS_ARRAYS_INFO, ids=str)
def test_contiguous_arrays(array_info: ArrayInfo):
    assert nope.is_contiguous(array_info.shape, array_info.strides,
                              array_info.element_size)


@pytest.mark.parametrize("array_info", NOT_CONTIGUOUS_ARRAYS_INFO)
def test_not_contiguous_arrays(array_info: ArrayInfo):
    assert not nope.is_contiguous(array_info.shape, array_info.strides,
                                  array_info.element_size)


def test_contiguous_check_throws_exception_on_lengths_mismatch():
    with pytest.raises(ValueError):
        nope.is_contiguous((1, 2), (23, 3, 3), 4)
