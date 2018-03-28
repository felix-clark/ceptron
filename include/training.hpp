#pragma once
#include "global.hpp"
#include "net.hpp"
#include "min_step.hpp"
#include <functional>

// // serves as a wrapper to get a function of the parameters from a batch and a small net
// template <int Npar=Eigen::Dynamic>
// class IParFunc {
// public:
//   virtual ceptron::func_grad_res<Npar> getFuncGrad( const Array<Npar>& ) const = 0;
//   virtual double getFuncOnly( const Array<Npar>& ) const = 0;
// };

// template <int Npar, int Nin, int Nout>
// class ParFunc : public IParFunc<Npar>
// {
// public:
//   ceptron::func_grad_res<Npar> getFuncGrad( const Array<Npar>& );
//   double getFuncOnly( const Array<Npar>& );
// };


// we may want to return the entire net value so that the input can be declared const,
//  but it's possible this would result in extra copies
template <size_t N, size_t M, size_t P,
	  RegressionType Reg, InternalActivator Act>
void trainSlfnStatic(SingleHiddenLayerStatic<N, M, P>& net, IMinStep<SingleHiddenLayerStatic<N,M,P>::size()>& ms,
		     const BatchVec<N>& x, const BatchVec<M>& y, double l2reg) {
  // constexpr int Npar = Eigen::Dynamic;// SingleHiddenLayerStatic<N,M,P>::size();
  constexpr int Npar = SingleHiddenLayerStatic<N,M,P>::size();
  const Array<Npar>& par = net.getNetValue();
  // for compile testing only:
  // double fres = costFunc<N, M, P, Reg, Act>(net, x, y, l2reg);

  // these bindings only work because of the implicit conversion in the constructor from Array<size_> for the nets.
  //  this means this may incur extra copies and be inefficient.
  func_t<Npar> f = std::bind(costFunc<N, M, P, Reg, Act>, std::placeholders::_1, x, y, l2reg);
  fg_t<Npar> fg = std::bind(costFuncAndGrad<N, M, P, Reg, Act>, std::placeholders::_1, x, y, l2reg);
  grad_t<Npar> g = [&](const Array<Npar>& in){return fg(in).g;}; // strips gradient from combined result
  Array<Npar> dp = ms.getDeltaPar( f, g, par );
  assert(net.accessNetValue().size() == dp.size());
  net.accessNetValue() += dp; // increment net
}
