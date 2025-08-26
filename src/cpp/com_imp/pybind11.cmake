include(FetchContent)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.4
)

FetchContent_GetProperties(pybind11)

# CMake 3.14+
FetchContent_MakeAvailable(pybind11)

# CMake 3.11+
#if(NOT pybind11_POPULATED)
#  FetchContent_Populate(pybind11_sources)
#
#  add_subdirectory(
#    ${pybind11_sources_SOURCE_DIR}
#    ${pybind11_sources_BINARY_DIR}
#    )
#endif()