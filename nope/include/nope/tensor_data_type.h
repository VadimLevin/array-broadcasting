#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

namespace nope {
class TensorDataType {
public:
    enum TypeId : uint8_t {
        Int8 = 0,
        UInt8 = 1,
        Int16 = 2,
        UInt16 = 3,
        Int32 = 4,
        UInt32 = 5,
        Int64 = 6,
        UInt64 = 7,
        Float32 = 8,
        Float64 = 9
    };

    TensorDataType() = default;

    // NOLINTNEXTLINE(google-explicit-constructor)
    TensorDataType(TypeId type_id) : type_id_{type_id} {
    }

    template <class T>
    [[nodiscard]] static constexpr TypeId typeIdOf() noexcept {
        static_assert(sizeof(T) == 0, "Type ID is not specialized for this type");
        return TypeId::Int8;
    }

    template <class T>
    [[nodiscard]] static TensorDataType of() noexcept {
        return typeIdOf<T>();
    }

    [[nodiscard]] uint8_t typeId() const noexcept {
        return type_id_;
    }

    [[nodiscard]] size_t size() const noexcept;

private:
    TypeId type_id_{TypeId::Float32};
};

inline bool operator==(const TensorDataType& lhs, const TensorDataType& rhs) noexcept {
    return lhs.typeId() == rhs.typeId();
}

inline bool operator!=(const TensorDataType& lhs, const TensorDataType& rhs) noexcept {
    return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& stream, const TensorDataType& dtype);

std::string to_string(const TensorDataType& dtype);

#ifndef REGISTER_TENSOR_DATA_TYPE
    #define REGISTER_TENSOR_DATA_TYPE(builtin_type, type_id) \
        template <>                                          \
        [[nodiscard]] constexpr TensorDataType::TypeId       \
        TensorDataType::typeIdOf<builtin_type>() noexcept {  \
            return TypeId::type_id;                          \
        }
#else
    #error "Macros with name REGISTER_TENSOR_DATA_TYPE is already defined"
#endif

// macro usage
REGISTER_TENSOR_DATA_TYPE(int8_t, Int8)
REGISTER_TENSOR_DATA_TYPE(uint8_t, UInt8)
REGISTER_TENSOR_DATA_TYPE(int16_t, Int16)
REGISTER_TENSOR_DATA_TYPE(uint16_t, UInt16)
REGISTER_TENSOR_DATA_TYPE(int32_t, Int32)
REGISTER_TENSOR_DATA_TYPE(uint32_t, UInt32)
REGISTER_TENSOR_DATA_TYPE(int64_t, Int64)
REGISTER_TENSOR_DATA_TYPE(uint64_t, UInt64)
REGISTER_TENSOR_DATA_TYPE(float, Float32)
REGISTER_TENSOR_DATA_TYPE(double, Float64)

#undef REGISTER_TENSOR_DATA_TYPE
} // namespace nope
