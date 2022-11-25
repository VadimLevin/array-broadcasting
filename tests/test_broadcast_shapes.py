from typing import Tuple

import pytest
import numpy as np

from nope import broadcast_shapes


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


def _shapes_to_name(shapes: Tuple[Tuple[int, ...], ...]) -> str:
    return "__".join("_".join(map(str, shape)) for shape in shapes)


@pytest.mark.parametrize("shapes", BROADCASTABLE_SHAPES, ids=_shapes_to_name)
def test_broadcastable_shapes(shapes: tuple) -> None:
    expected_out_shape = np.broadcast_shapes(*shapes)
    actual_out_shape = tuple(broadcast_shapes(shapes))
    assert expected_out_shape == actual_out_shape


@pytest.mark.parametrize("shapes", NOT_BROADCASTABLE_SHAPES,
                         ids=_shapes_to_name)
def test_not_broadcastable_shapes(shapes: tuple) -> None:
    with pytest.raises(ValueError):
        broadcast_shapes(shapes)
