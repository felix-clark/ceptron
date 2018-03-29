#pragma once
#include <Eigen/Core>

namespace ceptron {

  template <int M, int N>
  using Mat = Eigen::Matrix<double, M, N>;
  template <int M=Eigen::Dynamic>
  using Vec = Mat<M, 1>;
  template <int M=Eigen::Dynamic>
  using BatchVec = Eigen::Matrix<double, M, Eigen::Dynamic>;
  
  template <int M=Eigen::Dynamic> // don't think we need anything but column arrays
  using Array = Eigen::Array<double, M, 1>;
  template <int M=Eigen::Dynamic> // we might use this type explicitly for element-wise computations
  using BatchArray = Eigen::Array<double, M, Eigen::Dynamic>;

  using Eigen::Map;

  // struct for returning a function value with its gradient.
  // often we are only interested in the gradient
  template <int M=Eigen::Dynamic>
  struct func_grad_res {
    double f=0.;
    Array<M> g = Array<M>::Zero(M);
  };
  
} // namespace ceptron
