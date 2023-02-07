#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include "nope/broadcasting.h"
#include "nope/is_contiguous.h"
#include "nope/tensor_data_type.h"

#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// template <class T>
// std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec) {
//     stream << '[';
//     if (!vec.empty()) {
//         stream << vec.front();
//     }
//     for (size_t i = 1, size = vec.size(); i < size; ++i) {
//         stream << ", " << vec[i];
//     }
//     return stream << ']';
// }

// template <class T>
// struct NDArray {
//     std::unique_ptr<T[]> data;
//     std::vector<ssize_t> shape;
//     std::vector<ssize_t> strides;

//     explicit NDArray(std::vector<ssize_t> shape_descr) : shape{std::move(shape_descr)}
//     {
//         const size_t total_bytes = this->total_bytes();
//         data = std::make_unique<T[]>(total_bytes / sizeof(T));
//         const size_t ndims = dims();
//         strides.reserve(ndims);
//         for (size_t i = 0, dims_bytes = total_bytes; i < ndims; ++i) {
//             dims_bytes /= static_cast<size_t>(shape[i]);
//             strides.push_back(static_cast<ssize_t>(dims_bytes));
//         }
//     }

//     size_t dims() const noexcept {
//         return shape.size();
//     }

//     T* ptr() noexcept {
//         return data.get();
//     }

//     const T* ptr() const noexcept {
//         return data.get();
//     }

//     size_t total_bytes() const noexcept {
//         return static_cast<size_t>(total()) * sizeof(T);
//     }

//     ssize_t total() const noexcept {
//         return std::accumulate(
//             shape.begin(), shape.end(), ssize_t{1}, std::multiplies<>{});
//     }

//     py::array numpy() && {
//         py::capsule delete_handle = {data.get(), [](void* data_ptr) {
//                                          delete[] static_cast<T*>(data_ptr);
//                                      }};
//         py::array arr{py::dtype::of<T>(),
//                       /* shape */ shape,
//                       /* strides */ strides,
//                       /* data_ptr */ data.get(),
//                       /* delete_handle */ delete_handle};
//         static_cast<void>(data.release());
//         return arr;
//     }
// };

// template <class T>
// std::ostream& operator<<(std::ostream& stream, const NDArray<T>& ndarray) {
//     return stream << "NDArray(type=" << typeid(T).name() << ", shape=" << ndarray.shape
//                   << ", strides=" << ndarray.strides << ')';
// }

// template <class T>
// class NDArrayIterator {
// public:
//     using value_type = std::remove_cv_t<T>;
//     using reference_type = const value_type&;
//     using const_reference_type = reference_type;
//     using pointer_type = const value_type*;
//     using const_pointer_type = pointer_type;
//     using iterator_type = std::input_iterator_tag;

//     NDArrayIterator(const NDArrayIterator<T>&) = default;
//     NDArrayIterator& operator=(const NDArrayIterator<T>&) = default;

//     NDArrayIterator(NDArrayIterator<T>&&) noexcept = default;
//     NDArrayIterator& operator=(NDArrayIterator<T>&&) noexcept = default;

//     ~NDArrayIterator() = default;

//     NDArrayIterator(T* base,
//                     const std::vector<ssize_t>& shape,
//                     const std::vector<ssize_t>& strides)
//         : ndindex_(shape.size()),
//           shape_{shape.data()},
//           strides_{strides.data()},
//           base_{reinterpret_cast<const std::byte*>(base)} {
//         std::cerr << "Dims: " << dims_ << '\n';
//     }

//     reference_type operator*() const {
//         return *reinterpret_cast<pointer_type>(ptr_);
//     }

//     NDArrayIterator& operator++() {
//         for (ssize_t i = dims_ - 1; i >= 0; --i) {
//             const ssize_t dim_size = shape_[i];
//             ptr_ += strides_[i];

//             if (++ndindex_[static_cast<size_t>(i)] < dim_size) {
//                 return *this;
//             }

//             ptr_ = ptr_ - dim_size * strides_[i];
//             ndindex_[static_cast<size_t>(i)] = 0;
//         }
//         return *this;
//     }

//     NDArrayIterator operator++(int) {
//         NDArrayIterator it(*this);
//         ++*this;
//         return it;
//     }

//     bool operator==(const NDArrayIterator<T>& other) {
//         return (ptr_ == other.ptr_) && (base_ == other.base_);
//     }

//     bool operator!=(const NDArrayIterator<T>& other) {
//         return !(*this == other);
//     }

// private:
//     std::vector<ssize_t> ndindex_;
//     ssize_t dims_{static_cast<ssize_t>(ndindex_.size())};
//     const ssize_t* shape_;
//     const ssize_t* strides_;
//     const std::byte* base_;
//     const std::byte* ptr_{base_};
// };

// template <class OutputIt, class InputIt1, class InputIt2, class Operation>
// void applyElemwise(OutputIt out_first,
//                    OutputIt out_last,
//                    InputIt1 first1,
//                    InputIt2 first2,
//                    Operation&& op) {
//     for (; out_first != out_last; ++first1, ++first2) {
//         *out_first++ = op(*first1, *first2);
//     }
// }

// template <class T, class Operation>
// NDArray<T> applyElemwise(const T* l_data,
//                          const std::vector<ssize_t>& l_shape,
//                          const std::vector<ssize_t>& l_strides,
//                          const T* r_data,
//                          const std::vector<ssize_t>& r_shape,
//                          const std::vector<ssize_t>& r_strides,
//                          Operation&& op) {
//     const ssize_t l_dims = static_cast<ssize_t>(l_shape.size());
//     const ssize_t r_dims = static_cast<ssize_t>(r_shape.size());
//     const ssize_t output_dims = std::max(l_dims, r_dims);
//     std::vector<ssize_t> broadcasted_shape(static_cast<size_t>(output_dims));
//     std::vector<ssize_t> broadcasted_l_strides(broadcasted_shape.size());
//     std::vector<ssize_t> broadcasted_r_strides(broadcasted_shape.size());

//     for (ssize_t i = output_dims - 1, l = l_dims - 1, r = r_dims - 1; i >= 0;
//          --i, --l, --r) {
//         ssize_t l_dim = 1;
//         ssize_t l_stride = 0;
//         if (l >= 0) {
//             l_dim = l_shape[static_cast<size_t>(l)];
//             if (l_dim > 1) {
//                 l_stride = l_strides[static_cast<size_t>(l)];
//             }
//         }
//         ssize_t r_dim = 1;
//         ssize_t r_stride = 0;
//         if (r >= 0) {
//             r_dim = r_shape[static_cast<size_t>(r)];
//             if (r_dim > 1) {
//                 r_stride = r_strides[static_cast<size_t>(r)];
//             }
//         }
//         broadcasted_shape[static_cast<size_t>(i)] = std::max(l_dim, r_dim);
//         broadcasted_l_strides[static_cast<size_t>(i)] = l_stride;
//         broadcasted_r_strides[static_cast<size_t>(i)] = r_stride;
//     }

//     NDArray<T> result(broadcasted_shape);

//     NDArrayIterator l_it(l_data, broadcasted_shape, broadcasted_l_strides);
//     NDArrayIterator r_it(r_data, broadcasted_shape, broadcasted_r_strides);

//     applyElemwise(result.ptr(),
//                   result.ptr() + result.total(),
//                   l_it,
//                   r_it,
//                   std::forward<Operation>(op));

//     return result;
// }

// template <class OperationType>
// py::array arrayElemwiseOp(const py::array& lhs, const py::array& rhs) {
//     const py::buffer_info lhs_buffer_info = lhs.request();
//     const py::buffer_info rhs_buffer_info = rhs.request();

//     const py::dtype& common_dtype = lhs.dtype();
//     if (common_dtype.not_equal(rhs.dtype())) {
//         py::pybind11_fail("Types mismatch");
//     }

// #define TYPE_DISPATCH(T)                                           \
//     if (common_dtype.equal(py::dtype::of<T>())) {                  \
//         return applyElemwise(static_cast<T*>(lhs_buffer_info.ptr), \
//                              lhs_buffer_info.shape,                \
//                              lhs_buffer_info.strides,              \
//                              static_cast<T*>(rhs_buffer_info.ptr), \
//                              rhs_buffer_info.shape,                \
//                              rhs_buffer_info.strides,              \
//                              OperationType{})                      \
//             .numpy();                                              \
//     }

//     TYPE_DISPATCH(float);
//     TYPE_DISPATCH(double);
//     TYPE_DISPATCH(int);
//     TYPE_DISPATCH(long);

// #undef TYPE_DISPATCH
//     py::pybind11_fail("Unsupported dtype");
// }

void registerTensorDataType(py::module_& module) {
    py::class_<nope::TensorDataType>(module, "TensorDataType")
        .def_property_readonly("value", &nope::TensorDataType::typeId)
        .def_property_readonly("size", &nope::TensorDataType::size)
        .def("__str__", [](const nope::TensorDataType& dtype) {
            using std::to_string;

            return to_string(dtype);
        });

#define DEFINE_TENSOR_DATA_TYPE_AS_MODULE_CONSTANT(name, value) \
    module.attr(name) = nope::TensorDataType(nope::TensorDataType::value)

    DEFINE_TENSOR_DATA_TYPE_AS_MODULE_CONSTANT("int8", Int8);
    DEFINE_TENSOR_DATA_TYPE_AS_MODULE_CONSTANT("uint8", UInt8);
    DEFINE_TENSOR_DATA_TYPE_AS_MODULE_CONSTANT("int16", Int16);
    DEFINE_TENSOR_DATA_TYPE_AS_MODULE_CONSTANT("uint16", UInt16);
    DEFINE_TENSOR_DATA_TYPE_AS_MODULE_CONSTANT("int32", Int32);
    DEFINE_TENSOR_DATA_TYPE_AS_MODULE_CONSTANT("uint32", UInt32);
    DEFINE_TENSOR_DATA_TYPE_AS_MODULE_CONSTANT("float32", Float32);
    DEFINE_TENSOR_DATA_TYPE_AS_MODULE_CONSTANT("float64", Float64);

#undef DEFINE_TENSOR_DATA_TYPE_AS_MODULE_CONSTANT
}

PYBIND11_MODULE(nope, nope_module) {
    nope_module.def(
        "broadcast_shapes",
        [](const std::vector<std::vector<int64_t>>& shapes) -> std::vector<int64_t> {
            auto out_shape = nope::broadcastShapes(shapes);
            if (out_shape.empty()) {
                throw py::value_error("Failed to broadcast input shape");
            }
            return out_shape;
        },
        py::arg("input_shapes"));
    nope_module.def(
        "is_contiguous",
        [](const std::vector<int64_t>& shape,
           const std::vector<int64_t>& strides,
           size_t element_size) -> bool {
            return nope::isContiguous(shape, strides, element_size);
        },
        py::arg("shape"),
        py::arg("strides"),
        py::arg("element_size"));
    registerTensorDataType(nope_module);
}
