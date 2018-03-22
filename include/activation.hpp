// class implementing activation functions and their derivatives
#include <Eigen/Core> // math functions?
// #include <Eigen/Dense>
// #include <type_traits>

enum class InternalActivator {Logit, Tanh};

// see https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
// though we did have trouble working with Eigen::ArrayBase<Derived>, even at times getting incorrect values despite compilation (!)


template <InternalActivator Act>
// there should possibly be some additional voodoo to ensure that ArrT is an Eigen::ArrayBase
// note that that seemed to bite us when we tried
class ActivFunc
{
public:
  template <typename ArrT> ArrT activ(const ArrT& in);
  template <typename ArrT> ArrT activToD(const ArrT& act); // sig -> sig', not x -> sig'
  // another function like dactive_from_in would compute the derivative directly from the input,
  //  which might be necessary for the less nice activation choices
};


// do template specialization: 
template <>
template <typename ArrT>
// might need to take an Eigen::Ref since we're passing it in segments?
/// something like ArrT -> Eigen::ArrayBase<ArrT> ? but that did lead to some very strange results at one point so maybe not
ArrT ActivFunc<InternalActivator::Logit>::activ(const ArrT& in) {
  // // something like the following assertion would be good to restrict to Array types, but I'm not getting it to work atm
  // static_assert( std::is_base_of<Eigen::ArrayBase, ArrT>::value,
  // 		 "activation function must take an Eigen::Array type" );
  using Eigen::exp;
  return 1.0/(1.0 + exp(-in));  
}

template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::Logit>::activToD(const ArrT& act) {
  using Eigen::exp;
  return act*(1.0-act);
}

template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::Tanh>::activ(const ArrT& in) {
  using Eigen::tanh;
  return tanh(in);  
}

template <>
template <typename ArrT>
ArrT ActivFunc<InternalActivator::Tanh>::activToD(const ArrT& act) {
  return 1.0-act.square();
}


