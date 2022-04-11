import pytest
import numpy as np
from nope import elemwise_sum, elemwise_mul

def elementwise_sum_op(a, b):
    expected = a + b
    actual = elemwise_sum(a, b)
    np.testing.assert_allclose(actual, expected,
                               err_msg=f"Test failed for input:\n{a=}\n{b=}")

def elementwise_mul_op(a, b):
    expected = a * b
    actual = elemwise_mul(a, b)
    np.testing.assert_allclose(actual, expected,
                               err_msg=f"Test failed for input:\n{a=}\n{b=}")

OPERATIONS_SET = (elementwise_sum_op, elementwise_mul_op)


@pytest.mark.parametrize("test_op", OPERATIONS_SET)
def test_elementwise_op_1d_and_1d_continuous(test_op):
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 5])

    test_op(a, b)


@pytest.mark.parametrize("test_op", OPERATIONS_SET)
def test_elementwise_op_1d_and_1d_non_continuous(test_op):
    base = np.arange(10, dtype=np.int32)
    a = base[::2]
    b = base[:5]

    test_op(a, b)


@pytest.mark.parametrize("test_op", OPERATIONS_SET)
def test_elementwise_op_2d_and_2d_continuous(test_op):
    a = np.hstack([10 * np.arange(4, dtype=np.int32), ] * 4)
    b = np.hstack([np.arange(4, dtype=np.int32), ] * 4)

    test_op(a, b)


@pytest.mark.parametrize("test_op", OPERATIONS_SET)
def test_elementwise_op_2d_and_2d_non_continuous(test_op):
    b = np.arange(16).reshape((4, 4))
    a = b * 10

    test_op(a[::2, ::2], b[::2, ::2])


@pytest.mark.parametrize("test_op", OPERATIONS_SET)
def test_elementwise_op_3d_and_3d_continuous(test_op):
    b = np.arange(5 * 5 * 3, dtype=np.int32).reshape((5, 5, 3))
    a = b * 10

    test_op(a, b)


@pytest.mark.parametrize("test_op", OPERATIONS_SET)
def test_elementwise_op_1d_and_1d_with_newaxis(test_op):
    a = 10 * np.arange(4)
    b = np.arange(3)

    test_op(a[:, np.newaxis], b[np.newaxis, :])


@pytest.mark.parametrize("test_op", OPERATIONS_SET)
def test_elementwise_op_2d_and_1d_continuous(test_op):
    a = np.arange(5 * 4, dtype=np.float32).reshape((4, 5))
    b = np.arange(5, dtype=np.float32)

    test_op(a, b)


@pytest.mark.parametrize("test_op", OPERATIONS_SET)
def test_elementwise_op_4d_and_3d_non_continuous(test_op):
    a = 100 * np.arange(8 * 1 * 6 * 1, dtype=np.int32).reshape((8, 1, 6, 1))
    b_src = np.arange(7 * 3 * 5, dtype=np.int32).reshape(7, 3, 5)
    b = b_src[:, ::3, :]

    test_op(a, b)
