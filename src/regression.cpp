#include "regression.hpp"

namespace {
using namespace ceptron;
}  // namespace

BatchArrayX ceptron::outputGate(RegressionType reg,
                                const Eigen::Ref<const BatchArrayX>& aout) {
  // ArrayX ceptron::outputGate(RegressionType reg, const Eigen::Ref<const
  // ArrayX>& aout) {
  switch (reg) {
    case RegressionType::Categorical:
      return Regressor<RegressionType::Categorical>::outputGate(aout);
    case RegressionType::LeastSquares:
      return Regressor<RegressionType::LeastSquares>::outputGate(aout);
    case RegressionType::Poisson:
      return Regressor<RegressionType::Poisson>::outputGate(aout);
  }
  throw std::runtime_error("unimplemented runtime output gate function");
}

ceptron::scalar ceptron::costFuncVal(RegressionType reg,
                                     const Eigen::Ref<const BatchArrayX>& xout,
                                     const Eigen::Ref<const BatchArrayX>& yin) {
  switch (reg) {
    case RegressionType::Categorical:
      return Regressor<RegressionType::Categorical>::costFuncVal(xout, yin);
    case RegressionType::LeastSquares:
      return Regressor<RegressionType::LeastSquares>::costFuncVal(xout, yin);
    case RegressionType::Poisson:
      return Regressor<RegressionType::Poisson>::costFuncVal(xout, yin);
  }
  throw std::runtime_error("unimplemented runtime cost function");
}
