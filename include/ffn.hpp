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
  LayerRec<Nin, Layers...> first_layer_;

  // definitions of FfnStatic
 private:
  static constexpr size_t inputs = Nin;
  static constexpr size_t outputs = decltype(first_layer_)::outputs;

 public:
  BatchVec<outputs> operator()(const ArrayX& net,
                               const BatchVec<Nin>& xin) const;
  // return the value of the cost function given net data, an input batch,
  //  and observed output.
  scalar costFunc(const ArrayX& net, const BatchVec<Nin>& xin,
                  const BatchVec<outputs>& yin) const;
};

template <size_t Nin, typename... Ts>
auto FfnStatic<Nin, Ts...>::operator()(const ArrayX& net,
                                       const BatchVec<Nin>& xin) const
    -> BatchVec<FfnStatic<Nin, Ts...>::outputs> {
  return first_layer_.predictRecurse(net, xin);
}

template <size_t Nin, typename... Ts>
scalar FfnStatic<Nin, Ts...>::costFunc(
    const ArrayX& net, const BatchVec<Nin>& xin,
    const BatchVec<FfnStatic<Nin, Ts...>::outputs>& yin) const {
  const int batchSize = xin.cols();
  scalar cost = first_layer_.costFuncRecurse(net, xin, yin);
  return cost / batchSize;
}

}  // namespace ceptron
