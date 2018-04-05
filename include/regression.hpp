#pragma once
#include "global.hpp"
#include <Eigen/Dense>

namespace ceptron {
  enum class RegressionType {Categorical, LeastSquares, Poisson};
  // "Categorical" means "exclusive categorical", meaning y should be one-hot or zero and an implied "none" category is used for |y| < 1.
  // for non-exclusive classification, independent NNs can be used.

  // runtime versions
  ceptron::BatchArrayX outputGate(RegressionType, const Eigen::Ref<const ceptron::BatchArrayX>& aout);
  // ceptron::ArrayX outputGate(RegressionType, const Eigen::Ref<const ceptron::ArrayX>& aout);
  double costFuncVal(RegressionType, const Eigen::Ref<const ceptron::BatchArrayX>& xout, const Eigen::Ref<const ceptron::BatchArrayX>& yin);

  template <RegressionType Reg>
  class Regressor
  {
  public:
    template <typename ArrT> static ArrT outputGate(const ArrT& aout);
    template <typename ArrT> static double costFuncVal(const ArrT& xout, const ArrT& yin); // compares y from data to output x of NN
    // we don't have to provide the derivative, it cancels out nicely for both regression types
    //  this statement is possibly somewhat general but it does depend on both the cost function and the output function of x
  };

  template <> // class template specialization
  template <typename ArrT>
  double Regressor<RegressionType::Categorical>::costFuncVal(const ArrT& xout, const ArrT& yin) {
    return  - (yin*log(xout)).sum() - ((1.0-yin.colwise().sum())*log1p(-xout.colwise().sum())).sum();
    // return  - (yin*log(xout)).sum() - (1.0-yin.sum())*log1p(-xout.sum());
  }

  template <>
  template <typename ArrT>
  double Regressor<RegressionType::LeastSquares>::costFuncVal(const ArrT& xout, const ArrT& yin) {
    // the convention in ML is to divide by factor of 2.
    // also makes backprop have same factors.
    // max likelihood of gaussian w/ variance 1
    //  note we also did not multiply by the factor of 2 in log-likelihood for categorical, as is convention in physics.
    return  (xout - yin).square().sum()/2.0;
  }

  template <>
  template <typename ArrT>
  double Regressor<RegressionType::Poisson>::costFuncVal(const ArrT& xout, const ArrT& yin) {
    // the additional log could be avoided if we passed in the value before applying the output gate (aout)
    return  (xout - yin*log(xout)).sum();
  }


  template <>
  template <typename ArrT>
  ArrT Regressor<RegressionType::Categorical>::outputGate(const ArrT& aout) {
    using Eigen::exp;
    // eval() actually totally screws it up! temporary object? don't call eval inside a function! (except perhaps in specific cases)
    ArrT expvals = exp(aout);
    // (1.0 + expvals.colwise().sum()) is a dynamic-range row vector w/ the normalization factor for each column
    ArrT result = expvals.rowwise() / (1.0 + expvals.colwise().sum());
    // traditionally, softmax does not have this extra 1.0 term, but we want to generalize cleanly from 1D case  
    return result;
  }

  template <>
  template <typename ArrT>
  ArrT Regressor<RegressionType::LeastSquares>::outputGate(const ArrT& aout) {
    // for a least-squares cost function, we need the output gate to be the identity in order to make backprop work the same way
    return aout;
  }

  template <>
  template <typename ArrT>
  ArrT Regressor<RegressionType::Poisson>::outputGate(const ArrT& aout) {
    // for poisson regression, the lambda parameter is typically an exponential of the gate value, which makes the value positive
    return exp(aout);
  }



} // namespace ceptron
