cmake_minimum_required(VERSION 3.20)

project(COMIMP LANGUAGES CXX)

# Specified the language standard
# require C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compile with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# set python path
# set(PYTHON_EXECUTABLE "/data/anaconda3/bin/python3")

#set(CMAKE_BUILD_TYPE Debug)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/com_imp)

find_package(Boost REQUIRED COMPONENTS system filesystem thread chrono log random)
include_directories(${Boost_INCLUDE_DIR})

file(GLOB SOURCE_FILES "src/*.cpp")
file(GLOB HEADER_FILES "src/*.h")
file(GLOB PYTHON_FILES "wrap/*.cpp" "wrap/*.hpp")

find_package(PythonInterp REQUIRED)
include(pybind11.cmake)

# Set up such that XCode organizes the files
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES})

add_subdirectory(wrap)
include(test_python.cmake)

include(test_cpp.cmake)