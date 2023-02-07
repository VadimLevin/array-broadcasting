#pragma once

#include <cstdint>
#include <vector>

namespace nope {
/**
 * \brief Performs check whenever tensor holding elements with byte size
 * \a element_size with given \a shape and \a strides is contiguous or not.
 *
 * \param shape Shape of the N-dimensional tensor.
 * \param strides Strides of the N-dimensional tensor.
 * \param element_size Size of the tensor element in bytes.
 *
 * \return true if tensor is contiguous, false otherwise.
 */
bool isContiguous(const std::vector<int64_t>& shape,
                  const std::vector<int64_t>& strides,
                  size_t element_size);
} // namespace nope
