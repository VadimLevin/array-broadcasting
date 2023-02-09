#include "tensor_bindings.h"

#include <stdexcept>
#include <sstream>

#include "nope/tensor.h"
#include "nope/tensor_data_type.h"

#include <pybind11/attr.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>

namespace py = pybind11;

namespace nope {
void registerTensorDataType(py::module_& module) {
    py::class_<nope::TensorDataType>(module, "TensorDataType")
        .def_property_readonly("value", &TensorDataType::typeId)
        .def_property_readonly("size", &TensorDataType::size)
        .def("__str__", [](const TensorDataType& dtype) {
            using std::to_string;

            return to_string(dtype);
        });
#define DEFINE_TENSOR_DATA_TYPE_AS_MODULE_CONSTANT(name, value) \
    module.attr(name) = TensorDataType(TensorDataType::value)

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

std::string tensorDataTypeToFormatDescriptor(const TensorDataType& dtype) {
#define SWITCH_TYPE_ID_CASE(type, type_id) \
    case TensorDataType::type_id:          \
        return py::format_descriptor<type>::format();

    switch (dtype.typeId()) {
        SWITCH_TYPE_ID_CASE(int8_t, Int8);
        SWITCH_TYPE_ID_CASE(uint8_t, UInt8);
        SWITCH_TYPE_ID_CASE(int16_t, Int16);
        SWITCH_TYPE_ID_CASE(uint16_t, UInt16);
        SWITCH_TYPE_ID_CASE(int32_t, Int32);
        SWITCH_TYPE_ID_CASE(uint32_t, UInt32);
        SWITCH_TYPE_ID_CASE(float, Float32);
        SWITCH_TYPE_ID_CASE(double, Float64);
        default:
            throw std::logic_error("Unknown tensor data type id: " + to_string(dtype));
    }
#undef SWITCH_TYPE_ID_CASE
}

TensorDataType formatDescriptorToTensorDataType(const std::string& format) {
#define CHECK_IF_FORMAT_REFER_TO(type)                   \
    if (format == py::format_descriptor<type>::format()) \
    return TensorDataType::of<type>()

    CHECK_IF_FORMAT_REFER_TO(int8_t);
    CHECK_IF_FORMAT_REFER_TO(uint8_t);
    CHECK_IF_FORMAT_REFER_TO(int16_t);
    CHECK_IF_FORMAT_REFER_TO(uint16_t);
    CHECK_IF_FORMAT_REFER_TO(int32_t);
    CHECK_IF_FORMAT_REFER_TO(uint32_t);
    CHECK_IF_FORMAT_REFER_TO(float);
    CHECK_IF_FORMAT_REFER_TO(double);

#undef CHECK_IF_FORMAT_REFER_TO

    throw std::runtime_error("Unknown tensor data type format: " + format);
    return TensorDataType::Float32;
}

std::vector<py::ssize_t> convertToSSizeVector(const std::vector<int64_t>& src) {
    std::vector<py::ssize_t> dst;
    dst.reserve(src.size());
    std::transform(src.begin(), src.end(), std::back_inserter(dst), [](int64_t val) {
        if (val > static_cast<int64_t>(std::numeric_limits<py::ssize_t>::max())
            || val < static_cast<int64_t>(std::numeric_limits<py::ssize_t>::min())) {
            throw std::out_of_range("Failed to convert int64_t to ssize_t. Value "
                                    + std::to_string(val));
        }
        return static_cast<py::ssize_t>(val);
    });
    return dst;
}

std::vector<int64_t> convertToInt64Vector(const std::vector<py::ssize_t>& src) {
    std::vector<int64_t> dst;
    dst.reserve(src.size());
    std::transform(src.begin(), src.end(), std::back_inserter(dst), [](py::ssize_t val) {
        return static_cast<int64_t>(val);
    });
    return dst;
}

void registerTensorBindings(py::module_& module) {
    registerTensorDataType(module);

    py::class_<Tensor>(module, "Tensor", py::buffer_protocol())
        .def_buffer([](Tensor& t) -> py::buffer_info {
            return py::buffer_info{
                t.data(),
                static_cast<py::ssize_t>(t.itemSize()),
                tensorDataTypeToFormatDescriptor(t.dtype()),
                static_cast<py::ssize_t>(t.dims()),
                convertToSSizeVector(t.shape()),
                convertToSSizeVector(t.strides())
            };
        })
        .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();

            return Tensor(static_cast<std::byte*>(info.ptr),
                          convertToInt64Vector(info.shape),
                          convertToInt64Vector(info.strides),
                          formatDescriptorToTensorDataType(info.format));
        }))
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("strides", &Tensor::strides)
        .def_property_readonly("dims", &Tensor::dims)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("item_size", &Tensor::itemSize)
        .def("__str__", [](const Tensor& t) {
            std::ostringstream stream;
            stream << t;
            return stream.str();
        });
}
} // namespace nope
