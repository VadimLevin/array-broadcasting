#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define NOPE_MARK_UNUSED(arg) static_cast<void>(arg)

namespace py = pybind11;

template <class T>
std::ostream &operator<<(std::ostream &stream, const std::vector<T> &vec) {
  stream << '[';
  if (!vec.empty()) {
    stream << vec.front();
  }
  for (size_t i = 1, size = vec.size(); i < size; ++i) {
    stream << ", " << vec[i];
  }
  return stream << ']';
}

template <class T> struct NDArray {
  std::unique_ptr<T[]> data;
  std::vector<ssize_t> shape;
  std::vector<ssize_t> strides;

  explicit NDArray(std::vector<ssize_t> shape_descr)
      : shape{std::move(shape_descr)} {
    const size_t total_bytes = this->total_bytes();
    data = std::make_unique<T[]>(total_bytes / sizeof(T));
    const size_t ndims = dims();
    strides.reserve(ndims);
    for (size_t i = 0, dims_bytes = total_bytes; i < ndims; ++i) {
      dims_bytes /= shape[i];
      strides.push_back(dims_bytes);
    }
  }

  size_t dims() const noexcept { return shape.size(); }

  T *ptr() noexcept { return data.get(); }

  const T *ptr() const noexcept { return data.get(); }

  size_t total_bytes() const noexcept {
    return static_cast<size_t>(total()) * sizeof(T);
  }

  ssize_t total() const noexcept {
    return std::accumulate(shape.begin(), shape.end(), ssize_t{1},
                           std::multiplies<>{});
  }

  py::array numpy() && {
    py::capsule delete_handle = {data.get(), [](void *data_ptr) {
                                   delete[] static_cast<T *>(data_ptr);
                                 }};
    py::array arr{py::dtype::of<T>(),
                  /* shape */ shape,
                  /* strides */ strides,
                  /* data_ptr */ data.get(),
                  /* delete_handle */ delete_handle};
    static_cast<void>(data.release());
    return arr;
  }
};

template <class T>
std::ostream &operator<<(std::ostream &stream, const NDArray<T> &ndarray) {
  return stream << "NDArray(type=" << typeid(T).name()
                << ", shape=" << ndarray.shape
                << ", strides=" << ndarray.strides << ')';
}

// template <class T>
// bool isContinuous(const std::vector<ssize_t> &shape,
//                   const std::vector<ssize_t> &strides) {
//   size_t expected_stride = std::accumulate(
//       shape.begin() + 1, shape.end(), ssize_t{sizeof(T)},
//       std::multiplies<>{});
//   for (size_t i = 1, dims = shape.size(); i < dims; ++i) {
//     const ssize_t dim = shape[i];
//     if (strides[i] != expected_stride) {
//       return false;
//     }
//     expected_stride /= dim;
//   }
//   return expected_stride == strides.back();
// }

template <class T> class NDArrayIterator {
public:
  using value_type = std::remove_cv_t<T>;
  using reference_type = const value_type &;
  using const_reference_type = reference_type;
  using pointer_type = const value_type *;
  using const_pointer_type = pointer_type;
  using iterator_type = std::input_iterator_tag;

  NDArrayIterator(const NDArrayIterator<T> &) = default;
  NDArrayIterator &operator=(const NDArrayIterator<T> &) = default;

  NDArrayIterator(NDArrayIterator<T> &&) noexcept = default;
  NDArrayIterator &operator=(NDArrayIterator<T> &&) noexcept = default;

  ~NDArrayIterator() = default;

  NDArrayIterator(T *base, const std::vector<ssize_t> &shape,
                  const std::vector<ssize_t> &strides)
      : ndindex_(shape.size()), shape_{shape.data()}, strides_{strides.data()},
        base_{reinterpret_cast<const std::byte *>(base)} {
            std::cerr << "Dims: " << dims_ << '\n';
        }

  reference_type operator*() const {
    return *reinterpret_cast<pointer_type>(ptr_);
  }

  NDArrayIterator &operator++() {
    for (ssize_t i = dims_ - 1; i >= 0; --i) {
      const ssize_t dim_size = shape_[i];
      ptr_ += strides_[i];

      if (++ndindex_[static_cast<size_t>(i)] < dim_size) {
        return *this;
      }

      ptr_ = ptr_ - dim_size * strides_[i];
      ndindex_[static_cast<size_t>(i)] = 0;
    }
    return *this;
  }

  NDArrayIterator operator++(int) {
    NDArrayIterator it(*this);
    ++*this;
    return it;
  }

  bool operator==(const NDArrayIterator<T> &other) {
    return (ptr_ == other.ptr_) && (base_ == other.base_);
  }

  bool operator!=(const NDArrayIterator<T> &other) { return !(*this == other); }

private:
  std::vector<ssize_t> ndindex_;
  ssize_t dims_{static_cast<ssize_t>(ndindex_.size())};
  const ssize_t *shape_;
  const ssize_t *strides_;
  const std::byte *base_;
  const std::byte *ptr_{base_};
};

std::vector<ssize_t> calculateOutputShape(const std::vector<ssize_t> &a_shape,
                                          const std::vector<ssize_t> &b_shape) {
  using VectorRef = const std::vector<ssize_t> &;

  const auto &[smallest_shape, largest_shape] =
      [&a_shape, &b_shape]() -> std::pair<VectorRef, VectorRef> {
    if (a_shape.size() > b_shape.size()) {
      return {b_shape, a_shape};
    }
    return {a_shape, b_shape};
  }();

  std::vector<ssize_t> expanded_shape(largest_shape.size());
  for (size_t i = 0; i < largest_shape.size() - smallest_shape.size(); ++i) {
    expanded_shape[i] = 1;
  }
  expanded_shape.insert(expanded_shape.begin() + smallest_shape.size(),
                        smallest_shape.begin(), smallest_shape.end());

  std::vector<ssize_t> output_shape(largest_shape.size());
  const ssize_t dims = static_cast<ssize_t>(output_shape.size() - 1);
  for (ssize_t i = dims; i >= 0; ++i) {
    output_shape[i] = std::max(largest_shape[static_cast<size_t>(i)],
                               expanded_shape[static_cast<size_t>(i)]);
  }

  return output_shape;
}

template <class OutputIt, class InputIt1, class InputIt2, class Operation>
void applyElemwise(OutputIt out_first, OutputIt out_last, InputIt1 first1,
                   InputIt2 first2, Operation&& op) {
  for (; out_first != out_last; ++first1, ++first2) {
    *out_first++ = op(*first1, *first2);
  }
}

template <class T, class Operation>
NDArray<T> applyElemwise(const T *l_data, const std::vector<ssize_t> &l_shape,
                         const std::vector<ssize_t> &l_strides, const T *r_data,
                         const std::vector<ssize_t> &r_shape,
                         const std::vector<ssize_t> &r_strides,
                         Operation&& op) {
  const ssize_t l_dims = static_cast<ssize_t>(l_shape.size());
  const ssize_t r_dims = static_cast<ssize_t>(r_shape.size());
  const ssize_t output_dims = std::max(l_dims, r_dims);
  std::vector<ssize_t> broadcasted_shape(static_cast<size_t>(output_dims));
  std::vector<ssize_t> broadcasted_l_strides(broadcasted_shape.size());
  std::vector<ssize_t> broadcasted_r_strides(broadcasted_shape.size());

  for (ssize_t i = output_dims - 1, l = l_dims - 1, r = r_dims - 1; i >= 0;
       --i, --l, --r) {
    ssize_t l_dim = 1;
    ssize_t l_stride = 0;
    if (l >= 0) {
        l_dim = l_shape[l];
        if (l_dim > 1) {
            l_stride = l_strides[l];
        }
    }
    ssize_t r_dim = 1;
    ssize_t r_stride = 0;
    if (r >= 0) {
        r_dim = r_shape[r];
        if (r_dim > 1) {
            r_stride = r_strides[r];
        }
    }
    broadcasted_shape[static_cast<size_t>(i)] = std::max(l_dim, r_dim);
    broadcasted_l_strides[static_cast<size_t>(i)] = l_stride;
    broadcasted_r_strides[static_cast<size_t>(i)] = r_stride;
  }

  NDArray<T> result(broadcasted_shape);

  NDArrayIterator l_it(l_data, broadcasted_shape, broadcasted_l_strides);
  NDArrayIterator r_it(r_data, broadcasted_shape, broadcasted_r_strides);

  applyElemwise(result.ptr(), result.ptr() + result.total(), l_it, r_it,
                std::forward<Operation>(op));

  return result;
}

template <class OperationType>
py::array arrayElemwiseOp(const py::array &lhs, const py::array &rhs) {
  const py::buffer_info lhs_buffer_info = lhs.request();
  const py::buffer_info rhs_buffer_info = rhs.request();

  const py::dtype &common_dtype = lhs.dtype();
  if (common_dtype.not_equal(rhs.dtype())) {
    py::pybind11_fail("Types mismatch");
  }

#define TYPE_DISPATCH(T)                                                       \
  if (common_dtype.equal(py::dtype::of<T>())) {                                \
    return applyElemwise(                                                      \
               static_cast<T *>(lhs_buffer_info.ptr), lhs_buffer_info.shape,   \
               lhs_buffer_info.strides, static_cast<T *>(rhs_buffer_info.ptr), \
               rhs_buffer_info.shape, rhs_buffer_info.strides, OperationType{})  \
        .numpy();                                                              \
  }

  TYPE_DISPATCH(float);
  TYPE_DISPATCH(double);
  TYPE_DISPATCH(int);
  TYPE_DISPATCH(long);

#undef TYPE_DISPATCH
  py::pybind11_fail("Unsupported dtype");
}

PYBIND11_MODULE(nope, nope_module) {
  nope_module.def("elemwise_sum", &arrayElemwiseOp<std::plus<>>);
  nope_module.def("elemwise_mul", &arrayElemwiseOp<std::multiplies<>>);
}
