cmake_minimum_required(VERSION 3.5)
project(ceptron)

set (CMAKE_CXX_STANDARD 14)

set (ceptron_VERSION_MAJOR 0)
set (ceptron_VERSION_MINOR 0)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# set(CMAKE_PREFIX_PATH )

# manually include eigen location, which by default is installed in /usr/local/include/eigen3
include_directories(include ${CMAKE_INSTALL_PREFIX}/include/eigen3)

# source files can be manually added:
# set(SOURCES src/source0.cxx src/source1.cxx)
# but wildcarding is cleaner:
file(GLOB SOURCES src/*.cpp)
# file(GLOB_RECURSE SOURCES src/*.cxx) # for when we want to organize a bit?

# add_executable(hello hello.cxx)
add_executable(hello ${SOURCES})

# get_cmake_property(_variableNames VARIABLES)
# foreach(_variableName ${_variableNames})
#   message( STATUS "${_variableName}=${${_variableName}}" )
# endforeach()
