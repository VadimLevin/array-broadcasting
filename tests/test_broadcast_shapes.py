from __future__ import annotations

import pytest
import numpy as np

import nope


BROADCASTABLE_SHAPES = (
    ((256, 256, 3), (3, )),
    ((7, 1, 5), (8, 1, 6, 1)),
    ((1,), (5, 4)),
    ((5, 4), (4,)),
    ((15, 3, 5), (15, 1, 5)),
    ((15, 3, 5), (3, 5)),
    ((3, 1), (15, 3, 10)),
    ((2, 1, 4), (5, 1, 4, 4)),
    ((1, 2, 1), (3, 2, 4), (2, 4))
)

NOT_BROADCASTABLE_SHAPES = (
    ((1, 2, 3), (), (1, 2, 3)),
    ((3, ), (4, )),
    ((2, 1), (3, 2)),
    ((5, ), (3, 4)),
    ((4, 3), (3, 2, 3)),
    ((4, 2, 4), (4, 3, 4)),
)


@pytest.mark.parametrize("shapes", BROADCASTABLE_SHAPES, ids=str)
def test_broadcastable_shapes(shapes: tuple[tuple[int, ...], ...]) -> None:
    expected_out_shape = np.broadcast_shapes(*shapes)
    actual_out_shape = tuple(nope.broadcast_shapes(shapes))
    assert expected_out_shape == actual_out_shape


@pytest.mark.parametrize("shapes", NOT_BROADCASTABLE_SHAPES, ids=str)
def test_not_broadcastable_shapes(shapes: tuple[tuple[int, ...], ...]) -> None:
    with pytest.raises(ValueError):
        nope.broadcast_shapes(shapes)
