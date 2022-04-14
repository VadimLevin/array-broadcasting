import pytest
import numpy as np

from nope import elemwise_sum


SHAPES_SET = ((13, ), (123, ), (1027,), (1, 123), (8, 256, 128, 100))
TYPES_SET = (np.float32, np.float64,)


def shape_to_str(value):
    if isinstance(value, tuple):
        return str(value)


@pytest.mark.parametrize("shape", SHAPES_SET, ids=shape_to_str)
@pytest.mark.parametrize("dtype", TYPES_SET)
def test_benchmark_elementwise_sum_continuous_matching_shapes_numpy(benchmark,
                                                                    shape,
                                                                    dtype):
    def test_func():
        return a + b

    a = np.zeros(shape, dtype=dtype)
    b = np.ones(shape, dtype=dtype)

    result = benchmark(test_func)


@pytest.mark.parametrize("shape", SHAPES_SET, ids=shape_to_str)
@pytest.mark.parametrize("dtype", TYPES_SET)
def test_benchmark_elementwise_sum_continuous_matching_shapes_nope(benchmark,
                                                                   shape,
                                                                   dtype):
    a = np.zeros(shape, dtype=dtype)
    b = np.ones(shape, dtype=dtype)

    result = benchmark(elemwise_sum, a, b)

    np.testing.assert_equal(result, a + b)
