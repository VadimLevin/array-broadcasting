cmake_minimum_required(VERSION 3.11)

# In-source builds might be risky and lead to unexpected things
# so it is good practice to prohibit them in configuration phase
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Do not build in-source! "
            "Please remove CMakeCache.txt and the CMakeFiles/ directory. "
            "Then build out-of-source.")
endif()


# Semantic versioning is used https://semver.org
project(nope
    VERSION 0.0.0
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

find_package(pybind11 REQUIRED)

pybind11_add_module(nope module.cpp)

set_target_properties(nope
    PROPERTIES
        CXX_STANDARD          17
        CXX_EXTENSIONS        OFF
        CXX_STANDARD_REQUIRED ON
)

include(cmake/warnings_definition.cmake)

target_compile_options(nope
    PRIVATE
        ${project_cxx_warnings}
)
