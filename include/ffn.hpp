#pragma once
// the static (compile-time) version of an arbitrary-size FFNN
#include <Eigen/Dense>
#include "activation.hpp"
#include "ffn_layer.hpp"
#include "global.hpp"
#include "log.hpp"
#include "regression.hpp"

namespace ceptron {

// ---- FfnStatic ----
// a template class defining a static-size feedforward net of arbitrarily many
// layers.
template <size_t Nin, typename... Layers>
class FfnStatic {
  static_assert(sizeof...(Layers) > 0, "a FFNN needs more than an input layer");
 private:
  // this member is recursively-defined first layer, which holds remaining layers.
  using first_layer_t = LayerRec<Nin, Layers...>;
  first_layer_t first_layer_;
 public:
  static constexpr size_t inputs = Nin;
  static constexpr size_t outputs = decltype(first_layer_)::outputs;
  static constexpr size_t size = LayerTraits<first_layer_t>::netSize;
  // get output for an input, in other words, the net's prediction
  BatchVec<outputs> operator()(const ArrayX& net,
                               const BatchVec<Nin>& xin) const {
    return first_layer_.predictRecurse(net, xin);
  }
  // return the value of the cost function given net data, an input batch,
  //  and observed output.
  scalar costFunc(const ArrayX& net, const BatchVec<Nin>& xin,
                  const BatchVec<outputs>& yin) const;
  ArrayX costFuncGrad(const ArrayX& net, const BatchVec<Nin>& xin,
		      const BatchVec<outputs>& yin) const;

private:
  // some helper template structs for function specialization of activationLayer
  template <size_t Nl, typename DUMMY=void> struct activationLayer {
    static BatchVecX func(const first_layer_t& fl,
			  const ArrayX& net,
			  const BatchVec<inputs>& xin) {
      return fl.template activationInLayer<Nl-1>(net, xin);
    }
  };
  template <typename DUMMY> struct activationLayer<0,DUMMY> {
    static BatchVecX func(const first_layer_t&,
			  const ArrayX&,
			  const BatchVec<inputs>& xin) {
      return xin;
    }
  };
public:
  // returns the activation Nl layers deep for the given input
  template <size_t Nl>
  BatchVecX activationInLayer(const ArrayX& net,
                              const BatchVec<inputs>& xin) const {
    return activationLayer<Nl>::func(first_layer_, net, xin);
  }
  
  ArrayX randomWeights() const {return first_layer_.randParsRecurse();}
};

template <size_t Nin, typename... Ts>
scalar FfnStatic<Nin, Ts...>::costFunc(
    const ArrayX& net, const BatchVec<Nin>& xin,
    const BatchVec<FfnStatic<Nin, Ts...>::outputs>& yin) const {
  const int batchSize = xin.cols();
  scalar cost = first_layer_.costFuncRecurse(net, xin, yin);
  return cost / batchSize;
}

template <size_t Nin, typename... Ts>
ArrayX FfnStatic<Nin, Ts...>::costFuncGrad(
    const ArrayX& net, const BatchVec<Nin>& xin,
    const BatchVec<FfnStatic<Nin, Ts...>::outputs>& yin) const {
  const int batchSize = xin.cols();
  constexpr size_t gsize = LayerTraits<first_layer_t>::netSize;
  ArrayX grad = ArrayX(gsize);
  // the return value of this recursive function could be interesting when compared to the input data
  first_layer_.costFuncGradBackprop(net, xin, yin, grad);
  return grad / batchSize;
}

} // namespace ceptron
