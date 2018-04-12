#pragma once
#include "ffn_dyn.hpp"
#include "global.hpp"
#include "min_step.hpp"

namespace ceptron {

void trainFfnDyn(const FfnDyn& net, ArrayX& par, IMinStep& ms,
                 const BatchVecX& x, const BatchVecX& y);

}  // namespace ceptron
