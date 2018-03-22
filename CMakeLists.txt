cmake_minimum_required(VERSION 3.5)
project(ceptron)

set (CMAKE_CXX_STANDARD 14)

set (ceptron_VERSION_MAJOR 0)
set (ceptron_VERSION_MINOR 0)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})


# manually include eigen location, which by default is installed in /usr/local/include/eigen3
include_directories(include ${CMAKE_INSTALL_PREFIX}/include/eigen3)

# source files can be manually added:
# set(SOURCES src/source0.cpp src/source1.cpp ...)
# but wildcarding is cleaner:
file(GLOB SOURCES src/*.cpp)
# file(GLOB_RECURSE SOURCES src/*.cpp) # recursive, for when we want to organize a bit
# this is almost entirely a header-only package, so we probably will remove the library
add_library(ceptron ${SOURCES})

# add_executable(hello ${SOURCES})

# there must be a more efficient way of organizing these
add_executable(test_min test/test_min.cpp)
target_link_libraries(test_min ceptron)

add_executable(test_net test/test_net.cpp)
target_link_libraries(test_net ceptron)

# add_definitions(-DBOOST_LOG_DYN_LINK)
# find_package(Boost REQUIRED)


# this will print out all the CMake variables
# get_cmake_property(_variableNames VARIABLES)
# foreach(_variableName ${_variableNames})
#   message( STATUS "${_variableName}=${${_variableName}}" )
# endforeach()
