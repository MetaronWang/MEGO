# ESSENTIAL MATRIX BENCHMARK

# minimum version of cmake
cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD "17")
if (NOT CMAKE_CXX_STANDARD_REQUIRED)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

# release
set(CMAKE_BUILD_TYPE "Release")

# faster evaluation
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native" CACHE STRING "" FORCE)

# faster evaluation of math
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math" CACHE STRING "" FORCE)

# faster evaluation of Eigen
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx -mfma")

# build in lib folder
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/anchor_selection)
# set the project
project(4P3V_PROBLEM)

# add packages
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

find_package(PythonInterp REQUIRED)

include(FetchContent)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.4
)

FetchContent_GetProperties(pybind11)

# CMake 3.14+
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(LM4P3V 4p3v/interface.cpp 4p3v/calculate_cover.hpp 4p3v/evaluate_anchor.hpp 4p3v/sample_instances.hpp)
target_link_libraries(LM4P3V PUBLIC pybind11::module)

pybind11_add_module(LM5P2V 5p2v/interface.cpp 5p2v/calculate_cover.hpp 5p2v/evaluate_anchor.hpp 5p2v/sample_instances.hpp)
target_link_libraries(LM5P2V PUBLIC pybind11::module)