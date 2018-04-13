#pragma once
#include "activation.hpp"
#include "global.hpp"
#include "regression.hpp"
// we don't really want these recursive layer definitions to be publicly
// exposed, but it's too messy to try to make them private member classes of
// FfnStatic.
namespace ceptron {

// a layer should be able to be defined solely by its size and its activation
// function.
//  internally it may need to know how many elements are in the layer in front
//  or back of it, but its definition should require only these two.
template <size_t n, InternalActivator act>
struct FfnLayerDef : public ActivFunc<act> {
  FfnLayerDef() = default;
  ~FfnLayerDef() = default;
  static constexpr size_t size = n;
  // import activation functions and derivatives
  // std::function<Array<n>(const Array<n>&)> activ = ActivFunc<act>::activ;
  // we don't actually need to do this w/ inheritance, and in fact it probably
  //  requires less header code to remove them.
  using ActivFunc<act>::activ;
  using ActivFunc<act>::activToD;
};

// create a different type which specializes output layers
template <size_t n, RegressionType reg>
struct FfnOutputLayerDef : public Regressor<reg> {
  FfnOutputLayerDef() = default;
  ~FfnOutputLayerDef() = default;
  static constexpr size_t size = n;
  // import activation functions and derivatives
  // pull regression functions into local class namespace
  using Regressor<reg>::outputGate;
  using Regressor<reg>::costFuncVal;  // so far we don't require a special
                                      // derivative function
};

// another type of layer could be defined which acts as a dropout mask.
//   this could also be a property of the activation function.
//  it would need to be instantiated to set the float dropout probability

// forward-declare recursive template so that we can specialize later
template <size_t N_, typename L0_, typename... Rest_>
class LayerRec;

// we need to define a traits class to make a static interface for the layer
// helper classes, since the base case needs access to these traits but they
// aren't yet defined when the derived classes are defined
template <typename Derived>
struct LayerTraits;
template <size_t N_, typename L0_>
struct LayerTraits<LayerRec<N_, L0_>> {
  // this specialization is for the output layer
  static constexpr size_t inputs = N_;       // number of incoming values
  static constexpr size_t size = L0_::size;  // number of values in this layer
  static constexpr size_t outputs = size;
};
template <size_t N_, typename L0_, typename L1_, typename... Rest_>
struct LayerTraits<LayerRec<N_, L0_, L1_, Rest_...>> {
  static constexpr size_t inputs = N_;
  static constexpr size_t size = L0_::size;
  using next_layer_t = LayerRec<size, L1_, Rest_...>;
  // static constexpr size_t outputs = LayerTraits< typename
  // LayerRec<N_,L0_,L1_,Rest_...>::next_layer_t >::outputs;
  static constexpr size_t outputs = LayerTraits<next_layer_t>::outputs;
};

// static interface class for Layer classes
// it doesn't appear to be doing anything right now. so perhaps we don't even
// need this or the traits class.
// it might be useful if we have functions that are implemented the same for
// both internal and output layers.
template <typename Derived>
struct LayerBase {
 private:
  // we only have one base class so we can just define this helper function
  // in-class instead of making an additional helper class.
  // we may not need this non-const version
  // Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

 protected:
  // it will be convenient to export these into derived classes,
  //  but they will need to be "used" explicitly
  static constexpr size_t inputs = LayerTraits<Derived>::inputs;
  static constexpr size_t size = LayerTraits<Derived>::size;
  static constexpr size_t outputs = LayerTraits<Derived>::outputs;

 public:
  // get the output as a function of the inputs
  BatchVec<size> operator()(const Eigen::Ref<const ArrayX>& net,
                            const BatchVec<inputs>& xin) const {
    BatchVec<size> a1 = weight(net) * xin;
    a1.colwise() += bias(net);
    return activation(a1);
  }
  // so far, these do not need to be in the base case
  // scalar costFuncRecurse(const Eigen::Ref<const ArrayX>& net,
  //                        const BatchVec<inputs>& xin,
  //                        const BatchVec<outputs>& yin) const {
  //   return derived().costFuncRecurse(net, xin, yin);
  // }
  // Vec<outputs> predictRecurse(const Eigen::Ref<const ArrayX>& net,
  //                             const Vec<inputs>& xin) const {
  //   return derived().predictRecurse(net, xin);
  // }

 protected:
  inline BatchVec<size> activation(const BatchVec<size>& xin) const {
    return derived().activation(xin);
  }
  // convenience function for extracting bias term from net parameters
  inline Vec<size> bias(const Eigen::Ref<const ArrayX>& net) const {
    return Eigen::Map<const Vec<size>>(net.segment<size>(0).data());
  }
  // convenience function for extracting weight matrix
  inline Mat<size, inputs> weight(const Eigen::Ref<const ArrayX>& net) const {
    return Eigen::Map<const Mat<size, inputs>>(
        net.segment<inputs * size>(size).data());
  }
  // returns the rest of the array, which holds the remaining net parameters not
  // used by this
  inline ArrayX remainingNetPars(const Eigen::Ref<const ArrayX>& net) const {
    // will we have an issue with this being treated as a temporary object?
    return net.segment(size * (inputs + 1), net.size() - size * (inputs + 1));
  }
};  // class LayerBase

// perhaps should generalize InternalActivator to Activator which includes
// internal and output activation functions
// this would necessitate migrating some functionality currently in the
// regression file into this new class.
// cost function computation would be external (property of net only?) and
// we'd need more work to specialize in order to cancel common factors in e.g.
// logistic regression backprop
// recursive definition
template <size_t N_, typename L0_, typename L1_, typename... Rest_>
class LayerRec<N_, L0_, L1_, Rest_...>
    : public LayerBase<LayerRec<N_, L0_, L1_, Rest_...>> {
  using this_t = LayerRec<N_, L0_, L1_, Rest_...>;
  using traits_t = LayerTraits<this_t>;
  using base_t = LayerBase<this_t>;

 public:
  using base_t::inputs;
  using base_t::size;
  using base_t::outputs;
  // LayerRec() = default;
  scalar costFuncRecurse(const Eigen::Ref<const ArrayX>& net,
                         const BatchVec<N_>& xin,
                         const BatchVec<this_t::outputs>& yin) const;
  // function returns the prediction or a given input
  BatchVec<this_t::outputs> predictRecurse(
      const Eigen::Ref<const ArrayX>& net,
      const BatchVec<this_t::inputs>& xin) const;

  inline BatchVec<this_t::size> activation(
      const BatchVec<this_t::size>& xin) const {
    return L0_::template activ<BatchArray<size>>(xin.array()).matrix();
  }

 private:
  using next_layer_t = LayerRec<size, L1_, Rest_...>;
  next_layer_t next_layer_;
  using base_t::bias;
  using base_t::weight;
  using base_t::remainingNetPars;

};  // class LayerRec (internal case)

// end case for recursive layer: output layer will have RegressionType instead
// of InternalActivator
template <size_t N_, typename L0_>
class LayerRec<N_, L0_> : public LayerBase<LayerRec<N_, L0_>> {
  using this_t = LayerRec<N_, L0_>;
  using traits_t = LayerTraits<this_t>;
  using base_t = LayerBase<this_t>;
  // these traits may or may not need to be exposed for member function
  // definitions. probably yes if we want to define member functions below
  // (i.e. outside of the FfnStatic class definition).
  using base_t::inputs;  // could also get from traits_t::inputs, though not
                         // with "using" since this doesn't inherit
  using base_t::size;    // we should choose only one of these to use
  using base_t::outputs;

 public:
  // it's not optimal, but the template signatures make this difficult to
  // define below
  scalar costFuncRecurse(const Eigen::Ref<const ArrayX>& net,
                         const BatchVec<N_>& xin,
                         // const BatchVec<L0_::size>& yin) const;
                         const BatchVec<this_t::outputs>& yin) const;
  // function returns the prediction or a given input
  BatchVec<this_t::outputs> predictRecurse(
      const Eigen::Ref<const ArrayX>& net,
      const BatchVec<this_t::inputs>& xin) const;

  inline BatchVec<this_t::size> activation(
      const BatchVec<this_t::size>& xin) const {
    return L0_::template outputGate<BatchArray<outputs>>(xin.array()).matrix();
  }

 private:
  using base_t::bias;
  using base_t::weight;

};  // class LayerRec (output case)

template <size_t N, typename L0>
BatchVec<LayerRec<N, L0>::outputs> LayerRec<N, L0>::predictRecurse(
    const Eigen::Ref<const ArrayX>& net,
    const BatchVec<LayerRec<N, L0>::inputs>& xin) const {
  ///*auto*/ -> BatchVec<LayerRec<N, L0>::outputs> { // unfortunately this c++14
  ///syntax does not work w/ clang.. ?
  return (*this)(net, xin);
}

template <size_t N, typename L0, typename L1, typename... Rest>
BatchVec<LayerRec<N, L0, L1, Rest...>::outputs>
LayerRec<N, L0, L1, Rest...>::predictRecurse(
    const Eigen::Ref<const ArrayX>& net,
    const BatchVec<LayerRec<N, L0, L1, Rest...>::inputs>& xin) const {
  // auto -> BatchVec<LayerRec<N, L0, L1, Rest...>::outputs> { // this doesn't
  // work in clang...
  return next_layer_.predictRecurse(remainingNetPars(net), (*this)(net, xin));
}

template <size_t N, typename L0, typename L1, typename... Rest>
scalar LayerRec<N, L0, L1, Rest...>::costFuncRecurse(
    const Eigen::Ref<const ArrayX>& net, const BatchVec<N>& xin,
    const BatchVec<LayerRec<N, L0, L1, Rest...>::outputs>& yin) const {
  return next_layer_.costFuncRecurse(remainingNetPars(net), (*this)(net, xin),
                                     yin);
}

template <size_t N, typename L0>
scalar LayerRec<N, L0>::costFuncRecurse(
    const Eigen::Ref<const ArrayX>& net, const BatchVec<N>& xin,
    const BatchVec<LayerRec<N, L0>::outputs>& yin) const {
  // const BatchVec<L0::size>& yin) const {
  return L0::template costFuncVal<BatchArray<size>>((*this)(net, xin).array(),
                                                    yin.array());
}

}  // namespace ceptron
