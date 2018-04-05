#pragma once
#include "global.hpp"
#include "slfn.hpp"
#include "min_step.hpp"
#include <boost/log/trivial.hpp>
#include <functional>

// long term it's probably not best organization to include all the training functions in here,
//  because then to use a trainer we have to indirectly include the whole library.

namespace ceptron {
  // we may want to return the entire net value so that the input can be declared const,
  //  but it's possible this would result in extra copies
  template <typename Net>
  void trainSlfnStatic(Net& net, IMinStep& ms,
		       const BatchVec<Net::inputs>& x, const BatchVec<Net::outputs>& y) {
    const ArrayX& par = net.getNetValue();
    
    // these bindings only work because of the implicit conversion in the constructor from Array<size_> for the nets.
    //  this means this may incur extra copies and be inefficient.
    func_t f = std::bind(costFunc<Net>, std::placeholders::_1, x, y);
    grad_t g = [&](const ArrayX& in){return costFuncAndGrad<Net>(in, x, y).g;}; // strips gradient from combined result
    ArrayX dp = ms.getDeltaPar( f, g, par );
    assert(net.accessNetValue().size() == dp.size());
    net.accessNetValue() += dp; // increment net
  }

} // namespace ceptron
