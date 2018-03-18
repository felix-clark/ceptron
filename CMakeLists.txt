cmake_minimum_required(VERSION 3.5)
project(ceptron)

set (ceptron_VERSION_MAJOR 0)
set (ceptron_VERSION_MINOR 0)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories(include)

# source files can be manually added:
# set(SOURCES src/source0.cxx src/source1.cxx)
# but wildcarding is cleaner:
file(GLOB SOURCES src/*.cxx)
# file(GLOB_RECURSE SOURCES src/*.cxx) # for when we want to organize a bit?

# add_executable(hello hello.cxx)
add_executable(hello ${SOURCES})


# # find Eigen package
#actually don't need to so long as we "make install" eigen (which puts it in /usr/local/)
# find_package(Eigen)
# if (EIGEN_FOUND)
#   include_directories(${EIGEN_INCLUDE_DIRS})
#   # target_link_libraries(ceptron ${EIGEN_LIBRARIES}) # Eigen is header-only 
# endif (EIGEN_FOUND)
