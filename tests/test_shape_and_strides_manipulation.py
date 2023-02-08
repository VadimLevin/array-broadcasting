from __future__ import annotations

import pytest

import nope

from array_info import ArrayInfo

ArrayInfoPair = tuple[ArrayInfo, ArrayInfo]

ARRAY_INFO_PAIRS = (
    (ArrayInfo((), (), 1), ArrayInfo((), (), 1)),
    (ArrayInfo((1, 1, 1, 1), (4, 4, 4, 4), 4), ArrayInfo((1,), (4,), 4)),
    (ArrayInfo((1, 4), (16, 4), 4), ArrayInfo((4, ), (4,), 4)),
    (ArrayInfo((2, 3, 4), (48, 16, 4), 4), ArrayInfo((24, ), (4,), 4)),
    (ArrayInfo((3, 1, 1, 5), (20, 20, 20, 4), 4), ArrayInfo((15, ), (4,), 4)),
    (ArrayInfo((2, 4, 4, 2), (512, 64, 16, 8), 4), ArrayInfo((2, 32), (512, 8), 4)),
    (ArrayInfo((2, 2, 1, 5), (80, 40, 20, 4), 4), ArrayInfo((4, 5), (40, 4), 4)),
    (ArrayInfo((3, 5, 12, 10), (1200, 240, 20, 2), 2), ArrayInfo((1800, ), (2,), 2)),
    (ArrayInfo((10, 4, 12, 5), (2880, 480, 60, 8), 4),
     ArrayInfo((10, 4, 12, 5), (2880, 480, 60, 8), 4))
)


@pytest.mark.parametrize('array_info_pair', ARRAY_INFO_PAIRS, ids=str)
def test_calculate_effective_shape_and_strides(array_info_pair: ArrayInfoPair):
    in_array_info, expected_array_info = array_info_pair
    out_shape, out_strides = nope.calculate_effective_shape_and_strides(
        in_array_info.shape, in_array_info.strides
    )
    assert tuple(out_shape) == expected_array_info.shape, 'Shapes mismatch'
    assert tuple(out_strides) == expected_array_info.strides, \
        'Strides mismatch'
