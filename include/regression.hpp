#include <Eigen/Dense>

namespace {
  template <size_t M>
  using Array = Eigen::Array<double, M, 1>;
}

enum class RegressionType {Categorical, LeastSquares};
// "Categorical" means "exclusive categorical", meaning |y| <= 1 and an implied "none" category is used for |y| < 1.
// an input is meant 
// for non-exclusive classification, independent NNs can be used.

template <RegressionType Reg>
class Regressor
{
public:
  template <typename ArrT> ArrT outputGate(const ArrT& aout) const;
  template <typename ArrT> double costFuncVal(const ArrT& xout, const ArrT& yin) const; // compares y from data to output x of NN
  // we don't have to provide the derivative, it cancels out nicely for both regression types
  //  this statement is possibly somewhat general but it does depend on both the cost function and the output function of x
};

template <> // class template specialization
template <typename ArrT>
double Regressor<RegressionType::Categorical>::costFuncVal(const ArrT& xout, const ArrT& yin) const {
  return  - (yin*log(xout)).sum() - (1.0-yin.sum())*log1p(-xout.sum());
}

template <>
template <typename ArrT>
double Regressor<RegressionType::LeastSquares>::costFuncVal(const ArrT& xout, const ArrT& yin) const {
  // the convention in ML is to divide by factor of 2.
  // also makes backprop have same factors.
  // max likelihood of gaussian w/ variance 1
  //  note we also did not multiply by the factor of 2 in log-likelihood for categorical, as is convention in physics.
  return  (xout - yin).square().sum()/2.0;
}

template <>
template <typename ArrT>
ArrT Regressor<RegressionType::Categorical>::outputGate(const ArrT& aout) const {
  using Eigen::exp;
  ArrT expvals = exp(aout).eval();
  return expvals / (1.0 + expvals.sum());
  // traditionally, softmax does not have this extra 1.0 term, but we want to generalize cleanly from 1D case  
}

template <>
template <typename ArrT>
ArrT Regressor<RegressionType::LeastSquares>::outputGate(const ArrT& aout) const {
  // for a least-squares cost function, we need the output gate to be the identity in order to make backprop work the same way
  return aout;
}


// move this code to a class determining regression type
// exclusive softmax: it assumes that categories are exclusive and that "none" is a possible category.
// most direct extension of logit for P > 1 dimensions.
// intended for use at output layer only : derivative has convenient cancellation
// template <typename Derived>
// Eigen::ArrayBase<Derived> softmax_ex(const Derived& in) { // i think this should work fine, but does require template argument to function
//   using Eigen::exp;
//   // Derived expvals = exp(in);
//   // // Array<1> expvals = exp(in);
//   // double normfact = 1.0 + expvals.sum();
//   // return expvals / normfact; // traditionally, softmax does not have this extra 1 term
//   return exp(in)/(1.0 + exp(in).sum());
// }
// for some reason we have to specify the Array below and can't use the pattern above... too tired to get it rn
// template <size_t N>
// Array<N> softmax_ex(const Array<N>& in) { // i think this should work fine, but does require template argument to function
//   using Eigen::exp;
//   Array<N> expvals = exp(in).eval();
//   return expvals / (1.0 + expvals.sum()); // traditionally, softmax does not have this extra 1 term
// }


