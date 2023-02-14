#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>
#include <iosfwd>

#include "nope/tensor_data_type.h"

namespace nope {
namespace detail {
void freeNothing(std::byte* bytes);
}

class TypesMismatchError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Tensor {
public:
    using BytesFree = void (*)(std::byte*);

    explicit Tensor(std::vector<int64_t> shape,
                    TensorDataType dtype = TensorDataType::Float32);

    Tensor(std::byte* bytes,
           std::vector<int64_t> shape,
           std::vector<int64_t> strides,
           TensorDataType dtype,
           BytesFree bytes_free = &detail::freeNothing);

    Tensor(const Tensor& /* that */) = default;

    Tensor& operator=(const Tensor& /* that */) = default;

    Tensor(Tensor&& /* that */) noexcept = default;

    Tensor& operator=(Tensor&& /* that */) noexcept = default;

    ~Tensor() = default;

    const std::vector<int64_t>& shape() const noexcept {
        return shape_;
    }

    const std::vector<int64_t>& strides() const noexcept {
        return strides_;
    }

    TensorDataType dtype() const noexcept {
        return dtype_;
    }

    size_t itemSize() const noexcept {
        return dtype_.size();
    }

    size_t dims() const noexcept {
        return shape_.size();
    }

    int64_t dim(size_t i) const noexcept {
        return shape_[i];
    }

    // SECTION: Data pointer access
    std::byte* data() noexcept {
        return storage_->data.get();
    }

    const std::byte* data() const noexcept {
        return storage_->data.get();
    }

    template <class T>
    T* unsafeData() noexcept {
        return static_cast<T*>(data()) + storage_offset_;
    }

    template <class T>
    const T* unsafeData() const noexcept {
        return static_cast<const T*>(data()) + storage_offset_;
    }

    template <class T>
    T* safeData() {
        if (dtype_ != TensorDataType::of<T>) {
            throw TypesMismatchError("Trying to reinterpret tensor data as wrong type");
        }
        return unsafeData<T>();
    }


    template <class T>
    const T* safeData() const {
        if (dtype_ != TensorDataType::of<T>) {
            throw TypesMismatchError("Trying to reinterpret tensor data as wrong type");
        }
        return unsafeData<T>();
    }

private:
    struct Storage {
        using DataPtr = std::unique_ptr<std::byte, BytesFree>;
        DataPtr data;
        size_t size;

        Storage(DataPtr bytes, size_t data_size)
            : data{std::move(bytes)}, size{data_size} {
        }

        static std::shared_ptr<Storage>
        allocateContiguous(const std::vector<int64_t>& shape, int64_t element_size);

        static std::shared_ptr<Storage> fromBytes(std::byte* bytes,
                                                  size_t bytes_size,
                                                  BytesFree bytes_free);
    };

    std::shared_ptr<Storage> storage_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t storage_offset_{0};
    TensorDataType dtype_;
};

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);
} // namespace nope
