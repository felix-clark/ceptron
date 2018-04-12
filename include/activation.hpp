#pragma once
#include "global.hpp"
// class implementing activation functions and their derivatives
#include <Eigen/Core> // math functions?
// #include <Eigen/Dense>
// #include <type_traits>

namespace ceptron {

  // logit functions are not typically used for internal activation functions,
  // but it's a pedagogical choice so we'll keep it as an option. (also used in log reg, tho output nodes are currently specialized)
  // Tanh is better (since it's centered around 0) but rectified linear units (ReLU) and variations are popular now.
  // Softplus is a smooth version of an ReLU but is not as trivial to get the derivative of.
  enum class InternalActivator {Identity, Logit, Tanh, ReLU, Softplus, LReLU, Softsign};
  /// a number of activation functions can be generalized with hyperparameters, even on a per-channel level.
  // implementing this will require some interface changes, but it's not yet a priority in terms of making this library useful.
  // see https://arxiv.org/pdf/1710.05941.pdf for an exploration of various other activation functions
  
  // see https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
  // though we did have trouble working with Eigen::ArrayBase<Derived>, even at times getting incorrect values despite compilation (!)



  // a version w/ runtime lookup of the activation choice
  // there's no reason to not use the dynamic BatchArrayX, except perhaps for when a single BatchArray is passed ?
  // it may end up being feasible to do the calculations in-place in some cases, but commiting to that model may not be worth the loss of potential flexibility
  ceptron::BatchArrayX activ(InternalActivator, const Eigen::Ref<const ceptron::BatchArrayX>&);
  ceptron::BatchArrayX activToD(InternalActivator,  const Eigen::Ref<const ceptron::BatchArrayX>&);



  template <InternalActivator Act>
  class ActivFunc
  {
  public:
    // there should possibly be some additional voodoo to ensure that ArrT is an Eigen::ArrayBase
    // note that that seemed to bite us when we tried
    // these need to be static (and therefore not const) in order to call them without an instance
    template <typename ArrT> static inline ArrT activ(const ArrT& in);
    template <typename ArrT> static inline ArrT activToD(const ArrT& act); // sig -> sig', not x -> sig'
    // another function like dactive_from_in would compute the derivative directly from the input,
    //  which might be necessary for the less nice activation choices
    // we could consider returning both the activated layer and it's derivative simultaneously
  };

  // ---- identity function ----
  // useful for least-squares regression
  // not useful for internal activation functions

  template <>
  template <typename ArrT>
  ArrT ActivFunc<InternalActivator::Identity>::activ(const ArrT& in) {
    return in;
  }

  template <>
  template <typename ArrT>
  ArrT ActivFunc<InternalActivator::Identity>::activToD(const ArrT& act) {
    // this may not be properly optimized for static-size ArrT
    return ArrT::Constant(act.rows(), act.cols(), 1.0);
  }

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
#if EIGEN_VERSION_AT_LEAST(3,3,0)
    return tanh(in);
#else
    // tanh did not have a special definition in versions <= 3.2.
    ArrT expSq = exp(2*in);
    return (expSq-1)/(expSq+1);
#endif
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
    // the methods evaluating the size of in shouldn't be called when using static-sized inputs -- they're a redundant part of the interface for convenience
    return (in >= 0).select(in, ArrT::Zero(in.cols(), in.rows()));
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
    return (act > 0).select(ArrT::Ones(act.cols(), act.rows()),
			    ArrT::Zero(act.cols(), act.rows()));
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
    constexpr scalar alpha = 1.0/128.0;
    return (in >= 0).select(in, alpha*in);
  }

  template <>
  template <typename ArrT>
  ArrT ActivFunc<InternalActivator::LReLU>::activToD(const ArrT& act) {
    constexpr scalar alpha = 1.0/128.0;
    // maybe return 0.5 at zero exactly to give nets that are zero-initialized some gradient
    return (act > 0).select(ArrT::Ones(act.cols(), act.rows()),
			    alpha*ArrT::Ones(act.cols(), act.rows()));
    // need to address why this is breaking at some point.
    // the temporary object created by each condition might not last long enough
    // return (act > 0).select(ArrT::Ones(),
    // 			  (act < 0).select(alpha*ArrT::Ones(),
    // 					   0.5*(1+alpha)*ArrT::Ones()));
  }

  // ---- Softsign ----
  // this choice may be useful when dealing w/ vanishing gradients, since
  //  the derivatives are only polynomial-suppressed at large values.
  
  template <>
  template <typename ArrT>
  ArrT ActivFunc<InternalActivator::Softsign>::activ(const ArrT& in) {
    return in/(1.0+abs(in));
  }

  template <>
  template <typename ArrT>
  ArrT ActivFunc<InternalActivator::Softsign>::activToD(const ArrT& act) {
    // this one is easier to compute in terms of input rather than the activation itself
    // return (1/(1+abs(in)).square());
    return (abs(act)-1).square();
  }

  
} // namespace ceptron
