#pragma once
#include <Eigen/Core>

namespace ceptron {

  template <size_t M, size_t N>
  using Mat = Eigen::Matrix<double, M, N>;
  template <size_t M>
  using Vec = Mat<M, 1>;
  
  template <size_t M> // don't think we need anything but column arrays
  using Array = Eigen::Array<double, M, 1>;

  using Eigen::Map;

  
} // namespace ceptron
