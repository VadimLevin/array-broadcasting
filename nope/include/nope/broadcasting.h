#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace nope {
namespace detail {
bool broadcastShapes(const int64_t* const* in_shapes,
                     int64_t* in_shapes_dims,
                     int64_t n_inputs,
                     int64_t* out_shape,
                     int64_t out_shape_dims) noexcept;
} // namespace detail

/**
 * \brief Tries to broadcast all input shapes into a single one.
 *
 * 2 input shapes are compatible (broadcastable) when each dimension:
 *  - Both equal, or
 *  - One of them is 1, or
 *  - One of them doesn't exist
 * If one of input shapes are empty - they are not broadcastable.
 * Above rules can be intuitively extended to more than 2 inputs
 *
 * \tparam Shapes Shapes containers types with \a data() and \a size() methods.
 *
 * \param shapes Input shapes.
 *
 * \return Non-empty broadcasted output shape if input shapes are broadcastable,
 *      empty otherwise.
 */
template <class... Shapes>
std::vector<int64_t> broadcastShapes(const Shapes&... shapes) noexcept {
    static constexpr size_t kNInputs = sizeof...(Shapes);

    const std::array<const int64_t*, kNInputs> input_shapes_ptr{shapes.data()...};
    // clang-format off
    std::array<int64_t, kNInputs> input_shapes_dims{
        static_cast<int64_t>(shapes.size())...
    };
    // clang-format on
    std::vector<int64_t> output_shape;
    // if one of input shapes is empty...
    if (((shapes.size() == 0) || ...)) {
        return output_shape;
    }
    output_shape.resize(std::max({shapes.size()...}));
    if (!detail::broadcastShapes(input_shapes_ptr.data(),
                                 input_shapes_dims.data(),
                                 static_cast<int64_t>(input_shapes_dims.size()),
                                 output_shape.data(),
                                 static_cast<int64_t>(output_shape.size()))) {
        output_shape.clear();
    }
    return output_shape;
}

/**
 * \brief Tries to broadcast all input shapes into a single one.
 *
 * \overload \a broadcastShapes for 1 input shape
 *
 * \tparam Shape Shape container type with \a data() and \a size() methods.
 *
 * \param shape Input shape.
 *
 * \return \a shape converted to \a std::vector<int64_t>
 */
template <class Shape>
std::vector<int64_t> broadcastShapes(const Shape& shape) noexcept {
    return {shape.data(), shape.data() + shape.size()};
}

/**
 * \brief Tries to broadcast all input shapes into a single one
 *
 * \overload
 *
 * \param shapes Input shapes.
 *
 * \return Non-empty broadcasted output shape if input shapes are broadcastable,
 *      empty otherwise.
 */
std::vector<int64_t>
broadcastShapes(const std::vector<std::vector<int64_t>>& shapes) noexcept;

} // namespace nope
