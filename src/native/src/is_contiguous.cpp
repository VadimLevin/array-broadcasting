#include "nope/is_contiguous.h"

#include <stdexcept>
#include <iostream>

namespace nope {
namespace detail {
bool isContiguous(const int64_t* shape,
                  const int64_t* strides,
                  const int64_t dims,
                  const int64_t elem_size) {
    if (dims == 0) {
        return true;
    }
    if (strides[dims - 1] != elem_size) {
        return false;
    }
    for (int64_t dim = dims - 2; dim >= 0; --dim) {
        if (shape[dim + 1] * strides[dim + 1] != strides[dim]) {
            return false;
        }
    }
    return true;
}
} // namespace detail

bool isContiguous(const std::vector<int64_t>& shape,
                  const std::vector<int64_t>& strides,
                  size_t element_size) {
    if (shape.size() != strides.size()) {
        throw std::length_error("Shape and strides have different lengths");
    }
    return detail::isContiguous(shape.data(),
                                strides.data(),
                                static_cast<int64_t>(shape.size()),
                                static_cast<int64_t>(element_size));
}
} // namespace nope
