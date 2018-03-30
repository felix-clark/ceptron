#pragma once
#include "global.hpp"
#include "net.hpp"
#include "min_step.hpp"
#include <boost/log/trivial.hpp>
#include <functional>


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
  func_t</*Npar*/> f = std::bind(costFunc<Net, Reg, Act>, std::placeholders::_1, x, y, l2reg);
  fg_t</*Npar*/> fg = std::bind(costFuncAndGrad<Net, Reg, Act>, std::placeholders::_1, x, y, l2reg);
  grad_t</*Npar*/> g = [&](const Array</*Npar*/>& in){return fg(in).g;}; // strips gradient from combined result
  Array</*Npar*/> dp = ms.getDeltaPar( f, g, par );
  assert(net.accessNetValue().size() == dp.size());
  net.accessNetValue() += dp; // increment net
}
