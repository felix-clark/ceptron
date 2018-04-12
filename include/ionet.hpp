#pragma once
// in this header we will define some simple functions for reading and writing
// nets values,
//  which are simply stored as dynamic-length Eigen::Array's.
#include "global.hpp"

namespace ceptron {

void toFile(const ceptron::ArrayX& net, const std::string& fname);
ceptron::ArrayX fromFile(const std::string& fname);

}  // namespace ceptron
