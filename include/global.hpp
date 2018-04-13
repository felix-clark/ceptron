#pragma once
#include <Eigen/Core>

namespace ceptron {

// ML typically uses single-precision just fine.
// it might be interesting to let this be customizable but for now we'll leave
// it as a package-wide typedef.
using scalar = float;

template <int M, int N>
using Mat = Eigen::Matrix<scalar, M, N>;
using MatX = Mat<Eigen::Dynamic, Eigen::Dynamic>;
template <int M>
using Vec = Mat<M, 1>;
using VecX = Vec<Eigen::Dynamic>;
template <int M>
using BatchVec = Eigen::Matrix<scalar, M, Eigen::Dynamic>;
using BatchVecX = BatchVec<Eigen::Dynamic>;

template <int M>  // don't think we need anything but column arrays
using Array = Eigen::Array<scalar, M, 1>;
using ArrayX = Array<Eigen::Dynamic>;
  
  // we might use this type explicitly for element-wise computations
template <int M>
using BatchArray = Eigen::Array<scalar, M, Eigen::Dynamic>;
using BatchArrayX = BatchArray<Eigen::Dynamic>;

using Eigen::Map;

// struct for returning a function value with its gradient.
// often we are only interested in the gradient
struct func_grad_res {
  scalar f = 0.;
  ArrayX g = ArrayX::Zero(0);
};

}  // namespace ceptron
