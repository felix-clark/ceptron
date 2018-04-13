#pragma once
// in this header we will define some simple functions for reading and writing
// nets values,
//  which are simply stored as dynamic-length Eigen::Array's.
#include "global.hpp"

namespace ceptron {

void toFile(const ArrayX& net, const std::string& fname);
ArrayX fromFile(const std::string& fname);

}  // namespace ceptron
