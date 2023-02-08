#pragma once

#include <cstdint>
#include <vector>

namespace nope {
/**
 * \brief Calculates effective \a shape and \a strides trying to reduce number
 * of dimensions and increase continuous memory blocks.
 *
 * Adjacent dimensions \a dim_0 and \a dim_1 can be coalesced if they refer
 * to contiguous block of memory. In other words:
 * - \code shape[dim_1] == 1 || shape[dim_0] == 1 \endcode
 * - \code shape[dim_1] * strides[dim_1] == strides[dim_0] \endcode
 *
 * \param shape Tensor shape, that possible be updated.
 * \param strides Tensor strides, that possible be updated.
 */
void calculateEffectiveShapeAndStrides(std::vector<int64_t>& shape,
                                       std::vector<int64_t>& strides);
} // namespace nope
