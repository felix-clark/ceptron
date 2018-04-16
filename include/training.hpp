#pragma once
#include <functional>
#include "global.hpp"
#include "min_step.hpp"
#include "slfn.hpp"

// long term it's probably not best organization to include all the training
// functions in here,
//  because then to use a trainer we have to indirectly include the whole
//  library.

namespace ceptron {
// we may want to return the entire net value so that the input can be declared
// const,
//  but it's possible this would result in extra copies
template <typename Net>
void trainSlfnStatic(Net& net, ArrayX& par, IMinStep& ms,
                     const BatchVec<Net::inputs>& x,
                     const BatchVec<Net::outputs>& y) {
  func_t f = std::bind(costFunc<Net>, net, std::placeholders::_1, x, y);
  grad_t g = [&](const ArrayX& in) {
    return costFuncAndGrad<Net>(net, in, x, y).g;
  };  // strips gradient from combined result
  ArrayX dp = ms.getDeltaPar(f, g, par);
  par += dp;  // increment net
}

template <typename Net>
void trainFfnStatic(Net& net, ArrayX& par, IMinStep& ms,
                     const BatchVec<Net::inputs>& x,
                     const BatchVec<Net::outputs>& y) {
  func_t f = std::bind(Net::costFunc, net, std::placeholders::_1, x, y);
  grad_t g = std::bind(Net::costFuncGrad, net, std::placeholders::_1, x, y);
  ArrayX dp = ms.getDeltaPar(f, g, par);
  par += dp;  // increment net
}

}  // namespace ceptron
