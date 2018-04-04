#include "train_dyn.hpp"
#include <functional>

namespace {
  using namespace ceptron;
} // namespace

void ceptron::trainFfnDyn(const FfnDyn& net, ArrayX& par, IMinStep<>& ms,
			  const BatchVecX& x, const BatchVecX& y) {
  // to get pointer to member function, std::bind takes additional argument that is the address of the member to call it from.
  // a lambda would likely work just as well.
  func_t<> f = std::bind(&FfnDyn::costFunc, &net, std::placeholders::_1, x, y);
  grad_t<> g = std::bind(&FfnDyn::costFuncGrad, &net, std::placeholders::_1, x, y);
  ArrayX dp = ms.getDeltaPar( f, g, par );
  assert(par.size() == dp.size());
  par += dp; // increment net
}
