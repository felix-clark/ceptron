#include "regression.hpp"

namespace {
  using namespace ceptron;
} // namespace

  
BatchArrayX outputGate(RegressionType reg, const Eigen::Ref<const BatchArrayX>& aout) {
  switch (reg) {
  case RegressionType::Categorical:
    return Regressor<RegressionType::Categorical>::outputGate(aout);
  case RegressionType::LeastSquares:
    return Regressor<RegressionType::LeastSquares>::outputGate(aout);
  case RegressionType::Poisson:
    return Regressor<RegressionType::Poisson>::outputGate(aout);
  }
  throw std::runtime_error( "unimplemented runtime output gate function" );
}



double costFuncVal(RegressionType reg,
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
  throw std::runtime_error( "unimplemented runtime cost function" );
}

