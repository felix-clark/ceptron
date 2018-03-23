#pragma once
#include <Eigen/Core>

namespace ceptron {

  template <size_t M, size_t N>
  using Mat = Eigen::Matrix<double, M, N>;
  template <size_t M>
  using Vec = Mat<M, 1>;
  template <size_t M>
  using BatchVec = Eigen::Matrix<double, M, Eigen::Dynamic>;
  
  template <size_t M> // don't think we need anything but column arrays
  using Array = Eigen::Array<double, M, 1>;
  template <size_t M> // we might use this type explicitly for element-wise computations
  using BatchArray = Eigen::Array<double, M, Eigen::Dynamic>;

  using Eigen::Map;

  // struct for returning a function value with its gradient.
  // often we are only interested in the gradient
  template <size_t M>
  struct func_grad_res {
    double f=0.;
    Array<M> g = Array<M>::Zero();
  };
  
} // namespace ceptron
