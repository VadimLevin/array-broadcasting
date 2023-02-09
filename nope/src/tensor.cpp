#include "nope/tensor.h"

#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "nope/shape_and_strides_manipulation.h"

namespace nope {
namespace detail {
size_t calcDataSize(const std::vector<int64_t>& shape, int64_t element_size) noexcept {
    // clang-format off
    return static_cast<size_t>(
        std::accumulate(shape.begin(), shape.end(), element_size)
    );
    // clang-format on
}

void freeNothing(std::byte*) {
}

void validateStrides(const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& strides,
                     int64_t element_size) {
    if (shape.size() != strides.size()) {
        throw std::length_error("Shape and strides have different lengths");
    }
    if (strides.back() < element_size) {
        throw std::logic_error("Last stride can't be less than element size");
    }
}
} // namespace detail

Tensor::Tensor(std::vector<int64_t> shape, TensorDataType dtype)
    : storage_{Storage::allocateContiguous(shape, dtype.ssize())},
      shape_{std::move(shape)},
      strides_{createContiguousStrides(shape_, dtype.ssize())},
      dtype_{dtype} {
}

Tensor::Tensor(std::byte* bytes,
               std::vector<int64_t> shape,
               std::vector<int64_t> strides,
               TensorDataType dtype,
               BytesFree bytes_free)
    : storage_{Storage::fromBytes(
        bytes, detail::calcDataSize(shape, dtype.ssize()), bytes_free)},
      shape_{std::move(shape)},
      strides_{std::move(strides)},
      dtype_{dtype} {
}

std::shared_ptr<Tensor::Storage> Tensor::Storage::allocateContiguous(
    const std::vector<int64_t>& shape, int64_t element_size) {
    const auto size = detail::calcDataSize(shape, element_size);
    DataPtr data_ptr(new std::byte[size], [](std::byte* bytes) {
        delete[] bytes;
    });
    return std::make_shared<Storage>(std::move(data_ptr), size);
}

std::shared_ptr<Tensor::Storage> Tensor::Storage::fromBytes(std::byte* bytes,
                                                            size_t bytes_size,
                                                            BytesFree bytes_free) {
    return std::make_shared<Storage>(DataPtr{bytes, bytes_free}, bytes_size);
}

template <class T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec) {
    stream << "(";
    if (!vec.empty()) {
        stream << vec.front();
        for (auto it = vec.begin() + 1; it != vec.end(); ++it) {
            stream << ", " << *it;
        }
    }
    return stream << ")";
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    return stream << "Tensor(shape=" << tensor.shape() << ", strides=" << tensor.strides()
                  << ", data_addr=" << tensor.data() << ", dtype=" << tensor.dtype()
                  << ")";
}
} // namespace nope
