#pragma once
#include <Eigen/Core>

namespace ceptron {

  template <int M, int N>
  using Mat = Eigen::Matrix<double, M, N>;
  using MatX = Mat<Eigen::Dynamic, Eigen::Dynamic>;
  template <int M=Eigen::Dynamic>
  using Vec = Mat<M, 1>;
  using VecX = Vec<Eigen::Dynamic>;
  template <int M=Eigen::Dynamic>
  using BatchVec = Eigen::Matrix<double, M, Eigen::Dynamic>;
  using BatchVecX = BatchVec<Eigen::Dynamic>;
  
  template <int M=Eigen::Dynamic> // don't think we need anything but column arrays
  using Array = Eigen::Array<double, M, 1>;
  using ArrayX = Array<Eigen::Dynamic>;
  template <int M=Eigen::Dynamic> // we might use this type explicitly for element-wise computations
  using BatchArray = Eigen::Array<double, M, Eigen::Dynamic>;
  using BatchArrayX = BatchArray<Eigen::Dynamic>;

  using Eigen::Map;

  // struct for returning a function value with its gradient.
  // often we are only interested in the gradient
  struct func_grad_res {
    double f=0.;
    ArrayX g=ArrayX::Zero(0);
  };
  
} // namespace ceptron
