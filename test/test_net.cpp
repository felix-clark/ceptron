#include "global.hpp"
#include "slfn.hpp"
#include "ffn_dyn.hpp"
#include "ffn.hpp"
#include "ionet.hpp"
#include "log.hpp"
#include <Eigen/Core>
#include <iostream>
#include <functional>
#include <cstdlib>

using namespace ceptron;

namespace {
  // a local formatting option block for convenience
  const Eigen::IOFormat my_fmt(3, // first value is the precision
			       0, ", ", "\n", "[", "]");
}

// borrow this function from other testing utility to check the neural net's derivative
// we will really want to check element-by-element
template <typename Net>
void check_gradient(Net& net, const ArrayX& p, const BatchVec<Net::inputs>& xin, const BatchVec<Net::outputs>& yin, double ep=1e-4, double tol=1e-8) {
  constexpr size_t Npar = Net::size;
  func_grad_res fgvals = costFuncAndGrad(net, p, xin, yin); // this must be done before
  double fval = fgvals.f; // don't think we actually need this, but it might be a nice check
  ArrayX evalgrad = fgvals.g;

  LOG_TRACE("about to try to compute numerical derivative");
  
  ArrayX df(Npar);// maybe can do this slickly with colwise() or rowwise() ?
  for (size_t i_f=0; i_f<Npar; ++i_f) {
    ArrayX dp = ArrayX::Zero(Npar);
    double dpi = ep*sqrt(1.0 + p(i_f)*p(i_f));
    dp(i_f) = dpi;
    double fplusi = costFunc(net, p+dp, xin, yin);
    double fminusi = costFunc(net, p-dp, xin, yin);
    df(i_f) = (fplusi - fminusi)/(2*dpi);
  }
  
  LOG_TRACE("about to check with analytic gradient");

  // now check this element-by element
  double sqSumDiff = (df-evalgrad).square().sum();
  if ( sqSumDiff > tol*tol*1
       || !df.isFinite().all()
       || !evalgrad.isFinite().all()) {
    LOG_WARNING("gradient check failed at tolerance level " << tol << " (" << sqrt(sqSumDiff) << ")");
    LOG_WARNING("f(p) = " << fval);
    LOG_WARNING("numerical derivative = " << df.transpose().format(my_fmt));
    LOG_WARNING("analytic gradient = " << evalgrad.transpose().format(my_fmt));
    LOG_WARNING("difference = " << (df - evalgrad).transpose().format(my_fmt));
  }
  
}

// version for dynamic nets
void check_gradient(const FfnDyn& net, const ArrayX& p, const BatchVecX& xin, const BatchVecX& yin, double ep=1e-4, double tol=1e-8) {
  assert( xin.rows() == net.numInputs() );
  assert( yin.rows() == net.numOutputs() );
  const int Npar = net.num_weights();
  assert( p.size() == Npar );
  LOG_TRACE("about to call costFunc");
  double fval = net.costFunc(p, xin, yin);
  LOG_TRACE("about to call costFuncGrad");
  ArrayX gradval = net.costFuncGrad(p, xin, yin);

  LOG_TRACE("about to try to compute numerical derivative");
  
  ArrayX df(Npar);// maybe can do this assignment slickly with colwise() or rowwise() ?
  for (int i_f=0; i_f<Npar; ++i_f) {
    ArrayX dp = ArrayX::Zero(Npar);
    double dpi = ep*sqrt(1.0 + p(i_f)*p(i_f));
    dp(i_f) = dpi;
    double fplusi = net.costFunc(p+dp, xin, yin);
    double fminusi = net.costFunc(p-dp, xin, yin);
    df(i_f) = (fplusi - fminusi)/(2*dpi);
  }
  
  LOG_TRACE("about to check with analytic gradient");

  // now check this element-by element
  double sqSumDiff = (df-gradval).square().sum();
  if ( sqSumDiff > tol*tol*1
       || !df.isFinite().all()
       || !gradval.isFinite().all()) {
    LOG_WARNING("gradient check failed at tolerance level " << tol << " (" << sqrt(sqSumDiff) << ")");
    LOG_WARNING("f(p) = " << fval);
    LOG_WARNING("numerical derivative = " << df.transpose().format(my_fmt));
    LOG_WARNING("analytic gradient = " << gradval.transpose().format(my_fmt));
    LOG_WARNING("difference = " << (df - gradval).transpose().format(my_fmt));
  }
  
}

int main(int, char**) {

  SET_LOG_LEVEL(debug);
  
  // set seed for random initialization (Eigen uses std::rand())
  std::srand( 3490 ); // could also use std::time(nullptr) from ctime
  
  constexpr size_t Nin = 8;
  constexpr size_t Nout = 4;
  constexpr size_t Nh = 6; // number of nodes in hidden layer
  // constexpr RegressionType Reg=RegressionType::Categorical;
  constexpr RegressionType Reg=RegressionType::LeastSquares;
  // constexpr RegressionType Reg=RegressionType::Poisson;
  // constexpr InternalActivator Act=InternalActivator::Tanh;
  // constexpr InternalActivator Act=InternalActivator::ReLU; // we had gradient issues w/ ReLU because we were overriding it accidentally
  // however the nested select() to attempt to check for x=0 screws up the gradient completely
  // constexpr InternalActivator Act=InternalActivator::LReLU;
  // constexpr InternalActivator Act=InternalActivator::Softplus;
  constexpr InternalActivator Act=InternalActivator::Softsign;
  constexpr int batchSize = 16;

  BatchVec<Nin> input(Nin, batchSize); // i guess we need the length in the constructor still?
  input.setRandom();
  
  BatchVec<Nout> output(Nout, batchSize);
  // with a large-ish batch there are too many numbers to plug in manually
  // output << 0.5, 0.25, 0.1, 0.01;
  output.setRandom();
  
  {
    FfnDyn netd(Reg, Act, Nin, Nh, Nout);
    LOG_DEBUG("size of dynamic net: " << netd.num_weights());
    ArrayX randpar = netd.randomWeights(); //ArrayX::Random(netd.num_weights());
    netd.setL2Reg(0.01);
    // LOG_DEBUG(randpar);
    check_gradient(netd, randpar, input, output);
  }
  // to speed up compilation we can disable the static tests while we develop the dynamic case
#ifndef NOSTATIC

  { // static single-layer net test
    // defines the architecture of our test NN
    using Net = SlfnStatic<Nin, Nout, Nh>;
    Net testNet;
    ArrayX pars = randomWeights<Net>();
    testNet.setL2Reg(0.01);
  
    // we need to change the interface to let us spit out intermediate-layer activations
    // "activation" only has a useful debug meaning for a single layer
    LOG_INFO("input data is:\n" << input.format(my_fmt));
    LOG_INFO("first weights:\n" << testNet.getFirstSynapses(pars).format(my_fmt));
    LOG_INFO("second weights:\n" << testNet.getSecondSynapses(pars).format(my_fmt));
    // LOG_INFO("net value of array:\n" << testNet.getNetValue().format(my_fmt));
    LOG_INFO("array has " << pars.size() << " parameters.");

    // do we need to feed in the Net template parameter, or can it be inferred from testNet?
    check_gradient( testNet, pars, input, output );

    pars.setRandom();
    input.setRandom();
    // output << 1e-6, 0.02, 0.2, 0.01;
    output.setRandom();
    check_gradient( testNet, pars, input, output );

    pars.setRandom();
    input.setRandom(); // should also check pathological x values
    // output << 0.9, -0.1, 10.0, 0.1; // this one has nonsensical values but we can check the gradient regardless (it may give warning messages)
    output.setRandom();
    check_gradient( testNet, pars, input, output );

    toFile(pars, "copytest.net");
    decltype(pars) parsCopy = fromFile("copytest.net");
    if ( (pars != parsCopy).any() ) {
      LOG_WARNING("loaded net is not the same!");
      LOG_WARNING("original:\n" << pars.transpose().format(my_fmt));
      LOG_WARNING("copy:\n" << parsCopy.transpose().format(my_fmt));
      auto diff = pars - parsCopy;
      LOG_WARNING("difference:\n" << diff.transpose().format(my_fmt));
    }
  } // static single-layer net test

  { // general-size static net
    using Net = FfnStatic<4,4>;// this template signature will need to change
    Net net;
  } // general-size static net

#else
  LOG_INFO("skipping static nets.");
#endif // ifndef NO_STATIC
  
  return 0;
}

