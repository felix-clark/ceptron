# ceptron

## 1. Introduction

A machine-learning playground with a particular focus (for now) on simple feed-forward neural nets.

## 2. Requirements

### Eigen 3

[Eigen](http://eigen.tuxfamily.org/index.php) is an expression template library responsible for the linear algebra performed in ceptron. It is potentially quite fast, and should allow this library to have competetive performance for the simple cases it supports. (This statement has not actually been tested at the time of writing.)

### Boost.Log v2

The Boost dependency is now optional, and should only be invoked if CMake finds Boost. In this case, `BOOST_AVAILABLE` will be defined.

## 3. Installation

CMake is used to handle building and dependencies. It is highly recommended to not build in the source area. Something like the following is prefered (from this directory):

```
mkdir build
cd build
cmake ..
make -j4
```

Additional configuration options can be passed to cmake at the configuration stage. For instance, to compile with `clang` instead of g++, replace the 3rd line with `cmake .. -DCMAKE_CXX_COMPILER=clang++`.

## 4. Testing

After compilation, a testing trainer can be ran on some animal data:

`../ceptron/bin/test_train ../ceptron/data/zoo.data`

There are a few other validation testing utilities with source files lying in `test/`. They have yet to be integrated in any sort of systematic unit test suite.

## 5. Use

There are two major classes of NNs defined in this library; those with static (compile-time) and dynamic (run-time) definition. More on this will be described here as work progresses and interfaces solidify.
