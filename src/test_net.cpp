#include "net.hpp"

#include <Eigen/Core>
#include <iostream>

using std::cout;
using std::endl;

int main(int argc, char** argv) {

  SingleHiddenLayer<8, 4> testNet;
  testNet.randomInit();

  Eigen::Matrix<double, 8, 1> input;
  input.setRandom();

  Eigen::IOFormat my_fmt(2, // first value is the precision
			 0, ", ", "\n", "[", "]");
  
  cout << "output of random network is:  " << testNet.getOutput(input) << endl;
  cout << "first layer:\n" << testNet.getFirstSynapses().format(my_fmt) << endl;
  cout << "second layer:\n" << testNet.getSecondSynapses().format(my_fmt) << endl;
  auto& pars = testNet.getNetValue();
  // cout << "net value of array:\n" << testNet.getNetValue() << endl;
  cout << "array has " << pars.size() << " parameters." << endl;
  
  
  return 0;
}
