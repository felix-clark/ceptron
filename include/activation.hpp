#pragma once
// class implementing activation functions and their derivatives
/// as of now we don't actually need to include global.hpp
#include <Eigen/Core> // math functions?
// #include <Eigen/Dense>
// #include <type_traits>

// namespace {
//   using Eigen::exp;
//   using Eigen::log;
//   using Eigen::tanh;
// }

// logit functions are not typically used for internal activation functions,
// but it's a pedagogical choice so we'll keep it as an option.
// Tanh is better (since it's centered around 0) but rectified linear units (ReLU) and variations are popular now.
// Softplus is a smooth version of an ReLU but is not as trivial to get the derivative of.
enum class InternalActivator {Logit, Tanh, ReLU, Softplus, LReLU};
/// a number of activation functions can be generalized with hyperparameters, even on a per-channel level.
// implementing this will require some interface changes, but it's not yet a priority in terms of making this library useful.
// see https://arxiv.org/pdf/1710.05941.pdf for an exploration of various other activation functions

// see https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
// though we did have trouble working with Eigen::ArrayBase<Derived>, even at times getting incorrect values despite compilation (!)


template <InternalActivator Act>
class ActivFunc
{
public:
// there should possibly be some additional voodoo to ensure that ArrT is an Eigen::ArrayBase
// note that that seemed to bite us when we tried
  template <typename ArrT> ArrT activ(const ArrT& in);
  template <typename ArrT> ArrT activToD(const ArrT& act); // sig -> sig', not x -> sig'
  // another function like dactive_from_in would compute the derivative directly from the input,
  //  which might be necessary for the less nice activation choices
  // we could consider returning both the activated layer and it's derivative simultaneously
};

// ---- logit ----

// do template specialization: 
template <>
template <typename ArrT>
// might need to take an Eigen::Ref since we're passing it in segments?
/// something like ArrT -> Eigen::ArrayBase<ArrT> ? but that did lead to some very strange results at one point so maybe not
ArrT ActivFunc<InternalActivator::Logit>::activ(const ArrT& in) {
  // // something like the following assertion would be good to restrict to Array types, but I'm not getting it to work atm
  // static_assert( std::is_base_of<Eigen::ArrayBase, ArrT>::value,
  // 		 "activation function must take an Eigen::Array type" );
  return 1.0/(1.0 + exp(-in));  
}

template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::Logit>::activToD(const ArrT& act) {
  return act*(1.0-act);
}

// ---- tanh ----

template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::Tanh>::activ(const ArrT& in) {
  return tanh(in);  
}

template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::Tanh>::activToD(const ArrT& act) {
  return 1.0-act.square();
}

// ---- ReLU ----
// a popular choice but nodes often die out (this may actually help convergence)
template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::ReLU>::activ(const ArrT& in) {
  return (in >= 0).select(in, ArrT::Zero());
}

template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::ReLU>::activToD(const ArrT& act) {
  // maybe return 0.5 at zero exactly to give nets that are zero-initialized some gradient
  // somehow nested selects is screwing up the result??
  // return (act > 0).select(ArrT::Ones(),
  // 			  (act < 0).select(ArrT::Zero(),
  // 					   ArrT::Constant(0.5)));
  // return (act > 0).select(ArrT::Ones(),
  // 			  (act < 0).select(ArrT::Zero(),
  // 					   0.5*ArrT::Ones())); // this also fails
  return (act > 0).select(ArrT::Ones(), ArrT::Zero());
}

// ---- softplus ----
// a smooth approximation of ReLU
// this one could likely benefit from an approach where we compute the layer and its derivative simultaneously.

template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::Softplus>::activ(const ArrT& in) {
  return log(1.0 + exp(in));
}

template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::Softplus>::activToD(const ArrT& act) {
  return 1.0 - exp(-act);
}

// ---- LReLU ----
// a ReLU with a "leaky" term at x < 0. helps gradients be nonzero.
// the slope at x < 0 should really be an adjustable hyperparameter, and can even be tweaked on a per-channel basis.

template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::LReLU>::activ(const ArrT& in) {
  constexpr double alpha = 1.0/128.0;
  return (in >= 0).select(in, alpha*in);
}

template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::LReLU>::activToD(const ArrT& act) {
  constexpr double alpha = 1.0/128.0;
  // maybe return 0.5 at zero exactly to give nets that are zero-initialized some gradient
  return (act > 0).select(ArrT::Ones(), alpha*ArrT::Ones());
  // need to address why this is breaking at some point.
  // return (act > 0).select(ArrT::Ones(),
  // 			  (act < 0).select(alpha*ArrT::Ones(),
  // 					   0.5*(1+alpha)*ArrT::Ones()));
}
