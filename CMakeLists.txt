cmake_minimum_required(VERSION 3.5)
project(ceptron)

if(NOT CMAKE_BUILD_TYPE)
#set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set (CMAKE_CXX_STANDARD 14)

set (ceptron_VERSION_MAJOR 0)
set (ceptron_VERSION_MINOR 0)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

#find Eigen3 installation.it's header-only; no libraries necessary.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#we don't need to manually include the eigen include dir if we "link" it below
#include_directories(${EIGEN3_INCLUDE_DIR })

#we need the - pthread option
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

#make sure to install libboost - all - dev(not just libboost - dev)
#we need this definition for the log, or it fails to link
add_definitions(-DBOOST_LOG_DYN_LINK)
find_package(Boost 1.58 COMPONENTS log)
if (Boost_FOUND)
#we can work around not finding boost and just do a dumb global logging system
  add_definitions(-DBOOST_AVAILABLE)
endif()

#add our include directory
include_directories(include)

#source files can be manually added:
#set(SOURCES src / source0.cpp src / source1.cpp...)
#but wildcarding is easier:
file(GLOB SOURCES src/*.cpp)
# file(GLOB_RECURSE SOURCES src/*.cpp) # recursive, for when we want to organize a bit

# set compiler options
# checking for g++ can be done w/ something like if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX) ... endif()
set(MY_DEBUG_OPTIONS -g -O0 -Wall -Wextra -pedantic)
set(MY_RELEASE_OPTIONS -O3 -Wall -Wextra -pedantic)

# define our library
#  we will feasibly have multiple applications using the same lib,
# but static is generally better if we can get away with it (it's default for a reason)
# this should probably be a build option for CMake. it might be nice to produce both.
add_library(ceptron STATIC ${SOURCES})
# add_library(ceptron SHARED ${SOURCES})
# it appears that if we list the Eigen target here, we don't have to manually include the headers
target_link_libraries(ceptron Eigen3::Eigen ${Boost_LOG_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
# PUBLIC should propagate these flags down to dependencies
target_compile_options(ceptron PUBLIC "$<$<CONFIG:DEBUG>:${MY_DEBUG_OPTIONS}>")
target_compile_options(ceptron PUBLIC "$<$<CONFIG:RELEASE>:${MY_RELEASE_OPTIONS}>")


file(GLOB TESTSRCS test/*.cpp)
foreach(_TESTSRC ${TESTSRCS})
  string( REPLACE ".cpp" "" _TESTSRCNOEXT ${_TESTSRC} )
  # remove the prefix from the test source files. this may not be very robust:
  string( REPLACE "${CMAKE_SOURCE_DIR}/test/" "" _TESTOUT ${_TESTSRCNOEXT} )
  message( STATUS ${_TESTOUT} )
  add_executable(${_TESTOUT} ${_TESTSRC})
  target_link_libraries(${_TESTOUT} ceptron)
  # target_compile_definitions(${_TESTOUT} PRIVATE NOSTATIC) # turn off the static template testers right now just for speed
endforeach()


# # this will print out all the CMake variables
# get_cmake_property(_variableNames VARIABLES)
# foreach(_variableName ${_variableNames})
#   message( STATUS "${_variableName}=${${_variableName}}" )
# endforeach()
