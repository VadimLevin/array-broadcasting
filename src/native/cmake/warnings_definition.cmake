include_guard(GLOBAL)

option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

# Try to add as much warnings as possible if compiler supports them
# Warnings are added to compile options of the target, so they are not inflate
# global CXX_FLAGS
function(populate_project_cxx_warnings warnings_flags)
    if(WARNINGS_AS_ERRORS)
        if(MSVC)
            set(warnings /WX)
        else()
            set(warnings -Werror)
        endif()
    endif()
    if(MSVC)
        list(APPEND warnings
            # Warnings baseline for Visual C++ compiler /W4 might be too noisy
            # but it is recommended for new projects
            /W4
            # 'classname': class has virtual functions, but destructor is not virtual
            /w14265
            # 'identifier': conversion from 'type1' to 'type2' possible loss of data
            /w14242
            # 'variable': pointer truncation from 'type1' to 'type2'
            /w14311
            # '':
            /w14826
            # 'function': member function doesn't override any base class
            # virtual member function
            /w14263
            # 'variable': loop control variable declard in the for-loop is used
            # outside the loop scope (nonstandard extension is used)
            /we4289
            # 'operator': expression is always 'boolean_value'
            /w14296
            # '
        )
    else()
        list(APPEND warnings
            # Warnings baseline
            -Wall
            -Wextra
            -pedantic
            # Pointer alignment is increasing (char* -> int*)
            -Wcast-align
            # Do not compare floats with == and != (error-prone)
            -Wfloat-equal
            # A requested optimization pass is disabled
            -Wdisabled-optimization
            # Enables additional format checks
            -Wformat=2
            # Uninitialzied variables are inititalized with themselves
            -Winit-self
            # User-supplied include directory does not exist
            -Wmissing-include-dirs
            # Several declarations
            -Wredundant-decls
            # Shadowing of variable, parameter, type...
            -Wshadow
            # Implicit conversions that may alter a value
            -Wconversion
            # Optimization based on the assumption that signed overflow doesn't occur
            -Wstrict-overflow=5
            # Function declaration hides virtual function from a base class
            -Woverloaded-virtual
            # Use only new-style casts (*_cast)
            -Wold-style-cast
            # Member initialization reordering
            -Wreorder
            # sizeof operator is applied to a parameter that is declared as an array in a function definition
            -Wsizeof-array-argument
        )
        if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
            # Clang specific flags go here
            list(APPEND warnings

            )
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            # GCC specific flags go here
            list(APPEND warnings
            )
        else()
            message(AUTHOR_WARNING "Warnings flags for ${CMAKE_CXX_COMPILER_ID} is not populated")
        endif()
    endif()
    set(${warnings_flags} ${warnings} PARENT_SCOPE)
endfunction()

populate_project_cxx_warnings(project_cxx_warnings)
