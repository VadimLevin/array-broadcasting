#include "nope/shape_and_strides_manipulation.h"

namespace nope {
void calculateEffectiveShapeAndStrides(std::vector<int64_t>& shape,
                                       std::vector<int64_t>& strides) {
    if (shape.size() != strides.size()) {
        throw std::length_error("Shape and strides have different lengths");
    }
    if (shape.empty()) {
        return;
    }
    auto shape_it = shape.begin();
    auto strides_it = strides.begin();
    for (size_t i = 1; i < shape.size(); ++i) {
        if (shape[i] * strides[i] == strides[i - 1]) {
            *shape_it *= shape[i];
            *strides_it = strides[i];
        } else {
            ++shape_it;
            *shape_it = shape[i];
            ++strides_it;
        }
    }
    if (strides_it != strides.end()) {
        *strides_it = strides.back();
    }
    shape.erase(shape_it + 1, shape.end());
    strides.erase(strides_it + 1, strides.end());
}

void fillContiguousStrides(const int64_t* shape,
                           int64_t* strides,
                           int64_t dims,
                           int64_t element_size) {
    strides[dims - 1] = element_size;
    for (int64_t dim = dims - 2; dim >= 0; --dim) {
        strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
}

std::vector<int64_t> createContiguousStrides(const std::vector<int64_t>& shape,
                                             int64_t element_size) {
    std::vector<int64_t> strides(shape.size());
    // clang-format off
    fillContiguousStrides(shape.data(), strides.data(),
                         static_cast<int64_t>(shape.size()), element_size);
    // clang-format on
    return strides;
}
} // namespace nope
