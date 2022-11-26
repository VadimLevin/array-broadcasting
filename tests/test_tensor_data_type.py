import pytest

import nope

TENSOR_DATA_TYPE_NAMES = (
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "float32",
    "float64"
)


@pytest.mark.parametrize("type_name", TENSOR_DATA_TYPE_NAMES)
def test_tensor_data_type_is_available(type_name: str) -> None:
    assert hasattr(nope, type_name), \
        f"{type_name} definition is missing"

    dtype = getattr(nope, type_name)

    assert isinstance(dtype, nope.TensorDataType), \
        f"Wrong type for {type_name}. Got: {type(dtype)}"
    assert dtype.size >= 1
    assert hasattr(dtype, "value")
