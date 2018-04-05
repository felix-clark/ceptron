# ceptron

A machine-learning playground

Requirements: Eigen
Optional: Boost.Log

Ideally, install the testing utilities with

`mkdir build`

`cd build`

`cmake ..`

`make -j4`

After compilation, a testing trainer can be ran on the animal data:

`../ceptron/bin/test_train ../ceptron/data/zoo.data`
