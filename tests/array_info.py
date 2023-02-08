from dataclasses import dataclass


@dataclass
class ArrayInfo:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    element_size: int
