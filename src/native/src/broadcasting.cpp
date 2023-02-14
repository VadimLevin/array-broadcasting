#include "nope/broadcasting.h"

namespace nope {
namespace detail {
bool broadcastShapes(const int64_t* const* in_shapes,
                     int64_t* in_shapes_dims,
                     int64_t n_inputs,
                     int64_t* out_shape,
                     int64_t out_shape_dims) noexcept {
    // for each dimension of output shape starting from the trailing one
    for (int64_t out_dim = out_shape_dims - 1; out_dim >= 0; --out_dim) {
        int64_t dim_candidate{1};
        // for each input shape
        for (int64_t i = 0; i < n_inputs; ++i) {
            int64_t in_dim = 1;
            // get the shape dimension if index is non negative
            // otherwise set it to 1
            // this is equivalent to padding all input shapes with leading 1
            // to the max len shape
            if (auto& dim = in_shapes_dims[i]; dim > 0) {
                in_dim = in_shapes[i][--dim];
            }
            // Apply broadcasting rules:
            // - Each dimension should be equal
            // - Or equal to 1
            // - Or doesn't exist (handled as "padding" 1)
            if (dim_candidate != in_dim && in_dim != 1) {
                if (dim_candidate != 1) {
                    return false;
                }
                dim_candidate = in_dim;
            }
        }
        out_shape[out_dim] = dim_candidate;
    }
    return true;
}
} // namespace detail

std::vector<int64_t> broadcastShapes(const std::vector<std::vector<int64_t>>& shapes) noexcept {
    std::vector<const int64_t*> input_shapes_ptr;
    input_shapes_ptr.reserve(shapes.size());
    std::vector<int64_t> input_shapes_dims;
    input_shapes_dims.reserve(shapes.size());

    std::vector<int64_t> output_shape;

    size_t out_dims = 0;
    for (auto&& shape : shapes) {
        input_shapes_ptr.push_back(shape.data());
        const size_t size = shape.size();
        input_shapes_dims.push_back(static_cast<int64_t>(size));
        if (size > out_dims) {
            out_dims = size;
        } else if (size == 0) {
            return output_shape;
        }
    }

    output_shape.resize(out_dims);
    if (!detail::broadcastShapes(input_shapes_ptr.data(),
                                 input_shapes_dims.data(),
                                 static_cast<int64_t>(shapes.size()),
                                 output_shape.data(),
                                 static_cast<int64_t>(out_dims))) {
        output_shape.clear();
    }
    return output_shape;
}

} // namespace nope
