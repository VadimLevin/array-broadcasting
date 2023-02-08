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
} // namespace nope
