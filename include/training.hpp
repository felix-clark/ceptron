#pragma once
#include "global.hpp"
#include "slfn.hpp"
#include "ffn_dyn.hpp"
#include "min_step.hpp"
#include <boost/log/trivial.hpp>
#include <functional>

// long term it's probably not best organization to include all the training functions in here,
//  because then to use a trainer we have to indirectly include the whole library.

namespace ceptron {
  // we may want to return the entire net value so that the input can be declared const,
  //  but it's possible this would result in extra copies
  template <typename Net,
	    RegressionType Reg, InternalActivator Act>
  void trainSlfnStatic(Net& net, IMinStep<Net::size>& ms,
		       const BatchVec<Net::inputs>& x, const BatchVec<Net::outputs>& y, double l2reg) {
    // constexpr int Npar = Eigen::Dynamic;// Net::size;
    const Array</*Npar*/>& par = net.getNetValue();
    // for compile testing only:
    // double fres = costFunc<N, M, P, Reg, Act>(net, x, y, l2reg);
    
    // these bindings only work because of the implicit conversion in the constructor from Array<size_> for the nets.
    //  this means this may incur extra copies and be inefficient.
    func_t<> f = std::bind(costFunc<Net, Reg, Act>, std::placeholders::_1, x, y, l2reg);
    fg_t<> fg = std::bind(costFuncAndGrad<Net, Reg, Act>, std::placeholders::_1, x, y, l2reg);
    grad_t<> g = [&](const Array</*Npar*/>& in){return fg(in).g;}; // strips gradient from combined result
    Array<> dp = ms.getDeltaPar( f, g, par );
    assert(net.accessNetValue().size() == dp.size());
    net.accessNetValue() += dp; // increment net
  }

  // can be defined in .cpp
  // void trainFfnDyn(FfnDyn& net, ...);
  
} // namespace ceptron
