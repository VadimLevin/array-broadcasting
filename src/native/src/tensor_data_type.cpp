#include "nope/tensor_data_type.h"

#include <iostream>
#include <sstream>

namespace nope {
namespace detail {
inline size_t sizeOfTypeId(TensorDataType::TypeId type_id) noexcept {
    switch (type_id) {
        case TensorDataType::Int8:
            [[fallthrough]];
        case TensorDataType::UInt8:
            return 1;
        case TensorDataType::Int16:
            [[fallthrough]];
        case TensorDataType::UInt16:
            return 2;
        case TensorDataType::Int32:
            [[fallthrough]];
        case TensorDataType::UInt32:
            return 4;
        case TensorDataType::Int64:
            [[fallthrough]];
        case TensorDataType::UInt64:
            return 8;
        case TensorDataType::TypeId::Float32:
            return 4;
        case TensorDataType::TypeId::Float64:
            return 8;
        default:
            return 0;
    }
}
} // namespace detail

size_t TensorDataType::size() const noexcept {
    return detail::sizeOfTypeId(type_id_);
}

std::ostream& operator<<(std::ostream& stream, const TensorDataType& dtype) {
#define DATA_TYPE_CASE(data_type)   \
    case TensorDataType::data_type: \
        return stream << #data_type;

    switch (dtype.typeId()) {
        DATA_TYPE_CASE(Int8);
        DATA_TYPE_CASE(UInt8);
        DATA_TYPE_CASE(Int16);
        DATA_TYPE_CASE(UInt16);
        DATA_TYPE_CASE(Int32);
        DATA_TYPE_CASE(UInt32);
        DATA_TYPE_CASE(Int64);
        DATA_TYPE_CASE(UInt64);
        DATA_TYPE_CASE(Float32);
        DATA_TYPE_CASE(Float64);
        default:
            return stream << "<uknown(" << dtype.typeId() << ")>";
    }
#undef DATA_TYPE_CASE
}

std::string to_string(const TensorDataType& dtype) {
    std::ostringstream stream;
    stream << dtype;
    return stream.str();
}
} // namespace nope
