#pragma once
// the static (compile-time) version of an arbitrary-size FFNN
#include <Eigen/Dense>
#include "activation.hpp"
#include "global.hpp"
#include "log.hpp"
#include "regression.hpp"

namespace ceptron {

// a layer should be able to be defined solely by its size and its activation
// function.
//  internally it may need to know how many elements are in the layer in front
//  or back of it,
//   but its definition should require only these two.
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

// ---- FfnStatic ----
// a template class defining a static-size feedforward net of arbitrarily many
// layers.
template <size_t N, typename... Layers>
class FfnStatic {
  static_assert(sizeof...(Layers) > 0, "a FFNN needs more than an input layer");

 private:
  // forward-declare recursive template so that we can specialize later
  template <size_t N_, typename L1_, typename... Rest_>
  class LayerRec;

  // we need to define a traits class to make a static interface for the layer
  // helper classes,
  //  since the base case needs access to these traits but they aren't yet
  //  defined when the derived classes are defined
  template <typename Derived>
  struct LayerTraits;
  template <size_t N_, typename L1_>
  struct LayerTraits<LayerRec<N_, L1_>> {
    static constexpr size_t inputs = N_;       // number of incoming values
    static constexpr size_t size = L1_::size;  // number of values in this layer
    // this specialization is for the output layer
    static constexpr size_t outputs = size;
  };
  template <size_t N_, typename L1_, typename L2_, typename... Rest_>
  struct LayerTraits<LayerRec<N_, L1_, L2_, Rest_...>> {
    static constexpr size_t inputs = N_;
    static constexpr size_t size = L1_::size;
    using next_layer_t = LayerRec<size, L2_, Rest_...>;
    // static constexpr size_t outputs = LayerTraits< typename
    // LayerRec<N_,L1_,L2_,Rest_...>::next_layer_t >::outputs;
    static constexpr size_t outputs = LayerTraits<next_layer_t>::outputs;
  };

  // static interface class for Layer classes
  template <typename Derived>
  struct LayerBase {
   private:
    // we only have one base class so we can just define this helper function
    // in-class instead of making an additional helper class.
    // will we need a const version?
    // Derived& derived() const {return *static_cast<Derived*>(this);}
    Derived& derived() const { return static_cast<Derived>(*this); }

   protected:
    // it will be convenient to export these into derived classes,
    //  but they will need to be "used" explicitly
    static constexpr size_t inputs = LayerTraits<Derived>::inputs;
    static constexpr size_t size = LayerTraits<Derived>::size;
    static constexpr size_t outputs = LayerTraits<Derived>::outputs;

   public:
    scalar costFuncRecurse(const Eigen::Ref<const ArrayX>& net,
                           const BatchVec<inputs>& xin,
                           const BatchVec<outputs>& yin) const {
      // return static_cast< Derived* >(this)->costFuncRecurse(net, xin, yin);
      return derived().costFuncRecurse(net, xin, yin);
    }
    Vec<outputs> predictRecurse(const Eigen::Ref<const ArrayX>& net,
                                const Vec<inputs>& xin) const {
      return derived()->predictRecurse(net, xin);
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
      : public LayerBase<LayerRec<N_, L0_, L1_, Rest_...>>,
        public L0_  // this direct inheritance is unnecessary, since we can
                    // access the static functions of ActivFunc<> regardless
  {
    using this_t = LayerRec<N_, L0_, L1_, Rest_...>;
    using traits_t = LayerTraits<this_t>;
    using base_t = LayerBase<this_t>;

   public:
    using base_t::inputs;
    using base_t::size;
    using base_t::outputs;
    // LayerRec() = default;
    // unfortunately for now we'll define member functions inside the template
    // definition, since it's quite difficult to figure out the signature
    // otherwise
    // they should probably be made non-member classes.
    scalar costFuncRecurse(const Eigen::Ref<const ArrayX>& net,
                           const BatchVec<inputs>& xin,
                           const BatchVec<outputs>& yin) const {
      Vec<size> bias = Eigen::Map<const Vec<size>>(net.segment<size>(0).data());
      Mat<size, inputs> weights = Eigen::Map<const Mat<size, inputs>>(
          net.segment<inputs * size>(size).data());
      BatchVec<size> a1 = weights * xin;  // + bias;
      a1.colwise() += bias;
      BatchVec<size> x1 =
          L0_::template activ<BatchArray<size>>(a1.array()).matrix();
      const Eigen::Ref<const ArrayX>& remNet =
          net.segment(size * (inputs + 1), net.size() - size * (inputs + 1));
      return next_layer_.costFuncRecurse(remNet, x1, yin);
    }

   private:
    // a private constructor may be appropriate if we will only instantiate this
    // layer class from within the public FfnStatic class

    using L0_::activ;
    using L0_::activToD;

    using next_layer_t = LayerRec<size, L1_, Rest_...>;
    next_layer_t next_layer_;

  };  // class LayerRec (internal case)

  // end case for recursive layer: output layer will have RegressionType instead
  // of InternalActivator
  template <size_t N_, typename L0_>
  class LayerRec<N_, L0_> : public LayerBase<LayerRec<N_, L0_>>, public L0_ {
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
                           const BatchVec<inputs>& xin,
                           const BatchVec<outputs>& yin) const {
      Vec<size> bias = Eigen::Map<const Vec<size>>(net.segment<size>(0).data());
      Mat<size, inputs> weights = Eigen::Map<const Mat<size, inputs>>(
          net.segment<inputs * size>(size).data());
      BatchVec<size> a1 = weights * xin;  // + bias;
      a1.colwise() += bias;
      BatchVec<size> x1 =
          L0_::template outputGate<BatchArray<size>>(a1.array()).matrix();
      return L0_::template costFuncVal<BatchArray<size>>(x1.array(),
                                                         yin.array());
    }

   private:
    // // there is no next layer. we'll define functions based on the regression
    // type.
    // using L0_::outputGate; // we'd need to inherit from L0 (which is an
    // FfnOutputLayerDef)
    // using L0_::costFuncVal;
  };  // class LayerRec (output case)

  LayerRec<N, Layers...> first_layer_;  // clang requires this member to be
                                        // declared after the definitions. g++
                                        // just segfaults w/ the CRTP.

  // definitions of FfnStatic
 private:
  static constexpr size_t outputs = decltype(first_layer_)::outputs;

 public:
  scalar costFunc(const ArrayX& net, const BatchVec<N>& xin,
                  const BatchVec<outputs>& yin) const {
    const int batchSize = xin.cols();
    scalar cost = first_layer_.costFuncRecurse(net, xin, yin);
    return cost / batchSize;
  }
};

}  // namespace ceptron

// this syntax is ridiculously out-of-hand, and I'm not even sure what the
// proper syntax is.
// it's probably best to just define all these functions in the declaration
// itself
// template <size_t N, typename... Layers, size_t N_, typename L0_>
// ceptron::scalar ceptron::FfnStatic<N, Layers...>::LayerRec<N_,
// L0_>::costFuncRecurse(const Eigen::Ref<const ArrayX>& net, const
// BatchVec<ceptron::FfnStatic<N, Layers...>::LayerRec<N_, L0_>::inputs>& xin,
// const BatchVec<ceptron::FfnStatic<N, Layers...>::LayerRec<N_, L0_>::outputs>&
// yin) const {
//   // dummy:
// }
