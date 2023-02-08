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
    std::vector<int64_t> effective_shape(1, shape.front());
    std::vector<int64_t> effective_strides(1, strides.front());
    for (size_t i = 1; i < shape.size(); ++i) {
        if (shape[i] * strides[i] == strides[i - 1]) {
            effective_shape.back() *= shape[i];
            effective_strides.back() = strides[i];
        } else {
            effective_shape.push_back(shape[i]);
            effective_strides.push_back(strides[i]);
        }
    }
    if (effective_strides.size() != effective_shape.size()) {
        effective_strides.push_back(strides.back());
    }
    shape = std::move(effective_shape);
    strides = std::move(effective_strides);
}
} // namespace nope
