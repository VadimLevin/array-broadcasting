target_sources(nope
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/broadcasting.cpp
        ${CMAKE_CURRENT_LIST_DIR}/is_contiguous.cpp
        ${CMAKE_CURRENT_LIST_DIR}/module.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tensor_data_type.cpp
        ${CMAKE_CURRENT_LIST_DIR}/shape_and_strides_manipulation.cpp
)
