#include "global.hpp"
#include "slfn.hpp"
#include "ffn_dyn.hpp"
#include "ionet.hpp"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
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
void check_gradient(Net& net, const BatchVec<Net::inputs>& xin, const BatchVec<Net::outputs>& yin, double ep=1e-4, double tol=1e-8) {
  constexpr size_t Npar = Net::size;
  const ArrayX p = net.getNetValue(); // don't make this a reference: the internal data will change!
  // func_grad_res</*Npar*/> fgvals = costFuncAndGrad<Nin, Nout, Nhid, Reg, Act>(net, xin, yin, l2reg); // this must be done before 
  func_grad_res fgvals = costFuncAndGrad(net, xin, yin); // this must be done before
  // or does the following work? :
  // func_grad_res</*Npar*/> fgvals = costFuncAndGrad<Reg, Act>(net, xin, yin, l2reg); // this must be done before 
  double fval = fgvals.f; // don't think we actually need this, but it might be a nice check
  ArrayX evalgrad = fgvals.g;

  BOOST_LOG_TRIVIAL(trace) << "about to try to compute numerical derivative";
  
  ArrayX df(Npar);// maybe can do this slickly with colwise() or rowwise() ?
  for (size_t i_f=0; i_f<Npar; ++i_f) {
    ArrayX dp = ArrayX::Zero(Npar);
    // BOOST_LOG_TRIVIAL(trace) << "declaring dpi";
    double dpi = ep*sqrt(1.0 + p(i_f)*p(i_f));
    dp(i_f) = dpi;
    ArrayX pplus = p + dp;
    ArrayX pminus = p - dp;
    net.accessNetValue() = pplus;
    double fplusi = costFunc(net, xin, yin);
    net.accessNetValue() = pminus;
    double fminusi = costFunc(net, xin, yin);
    df(i_f) = (fplusi - fminusi)/(2*dpi);
  }
  
  BOOST_LOG_TRIVIAL(trace) << "about to check with analytic gradient";

  // now check this element-by element
  double sqSumDiff = (df-evalgrad).square().sum();
  if ( sqSumDiff > tol*tol*1
       || !df.isFinite().all()
       || !evalgrad.isFinite().all()) {
    BOOST_LOG_TRIVIAL(warning) << "gradient check failed at tolerance level " << tol << " (" << sqrt(sqSumDiff) << ")";
    BOOST_LOG_TRIVIAL(warning) << "f(p) = " << fval;
    BOOST_LOG_TRIVIAL(warning) << "numerical derivative = " << df.transpose().format(my_fmt);
    BOOST_LOG_TRIVIAL(warning) << "analytic gradient = " << evalgrad.transpose().format(my_fmt);
    BOOST_LOG_TRIVIAL(warning) << "difference = " << (df - evalgrad).transpose().format(my_fmt);
  }
  
}

// version for dynamic nets
void check_gradient(const FfnDyn& net, const ArrayX& p, const BatchVecX& xin, const BatchVecX& yin, double ep=1e-4, double tol=1e-8) {
  assert( xin.rows() == net.numInputs() );
  assert( yin.rows() == net.numOutputs() );
  const int Npar = net.num_weights();
  assert( p.size() == Npar );
  BOOST_LOG_TRIVIAL(trace) << "about to call costFunc";
  double fval = net.costFunc(p, xin, yin);
  BOOST_LOG_TRIVIAL(trace) << "about to call costFuncGrad";
  ArrayX gradval = net.costFuncGrad(p, xin, yin);

  BOOST_LOG_TRIVIAL(trace) << "about to try to compute numerical derivative";
  
  ArrayX df(Npar);// maybe can do this assignment slickly with colwise() or rowwise() ?
  for (int i_f=0; i_f<Npar; ++i_f) {
    ArrayX dp = ArrayX::Zero(Npar);
    double dpi = ep*sqrt(1.0 + p(i_f)*p(i_f));
    dp(i_f) = dpi;
    double fplusi = net.costFunc(p+dp, xin, yin);
    double fminusi = net.costFunc(p-dp, xin, yin);
    df(i_f) = (fplusi - fminusi)/(2*dpi);
  }
  
  BOOST_LOG_TRIVIAL(trace) << "about to check with analytic gradient";

  // now check this element-by element
  double sqSumDiff = (df-gradval).square().sum();
  if ( sqSumDiff > tol*tol*1
       || !df.isFinite().all()
       || !gradval.isFinite().all()) {
    BOOST_LOG_TRIVIAL(warning) << "gradient check failed at tolerance level " << tol << " (" << sqrt(sqSumDiff) << ")";
    BOOST_LOG_TRIVIAL(warning) << "f(p) = " << fval;
    BOOST_LOG_TRIVIAL(warning) << "numerical derivative = " << df.transpose().format(my_fmt);
    BOOST_LOG_TRIVIAL(warning) << "analytic gradient = " << gradval.transpose().format(my_fmt);
    BOOST_LOG_TRIVIAL(warning) << "difference = " << (df - gradval).transpose().format(my_fmt);
  }
  
}

int main(int, char**) {
  // we can adjust the log level like so:
  namespace logging = boost::log;
  logging::core::get()->set_filter
    (logging::trivial::severity >= logging::trivial::debug);

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
  constexpr InternalActivator Act=InternalActivator::Softplus;
  constexpr int batchSize = 16;

  FfnDyn netd(Reg, Act, Nin, Nh, Nout);
  BOOST_LOG_TRIVIAL(debug) << "size of dynamic net: " << netd.num_weights();
  ArrayX randpar = netd.randomWeights(); //ArrayX::Random(netd.num_weights());
  netd.setL2Reg(0.01);
  // BOOST_LOG_TRIVIAL(debug) << randpar;

  
  BatchVec<Nin> input(Nin, batchSize); // i guess we need the length in the constructor still?
  input.setRandom();

  BatchVec<Nout> output(Nout, batchSize);
  // with a large-ish batch there are too many numbers to plug in manually
  // output << 0.5, 0.25, 0.1, 0.01;
  output.setRandom();

  check_gradient(netd, randpar, input, output);

  // to speed up compilation we can disable the static tests while we develop the dynamic case
#ifndef NOSTATIC

  // defines the architecture of our test NN
  using Net = SlfnStatic<Nin, Nout, Nh>;
  
  
  Net testNet;
  testNet.accessNetValue() = randomWeights<Net>();
  netd.setL2Reg(0.01);
  
  // we need to change the interface to let us spit out intermediate-layer activations
  // "activation" only has a useful debug meaning for a single layer
  BOOST_LOG_TRIVIAL(info) << "input data is:\n" << input.format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "first weights:\n" << testNet.getFirstSynapses().format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "second weights:\n" << testNet.getSecondSynapses().format(my_fmt);
  auto& pars = testNet.getNetValue();
  // BOOST_LOG_TRIVIAL(info) << "net value of array:\n" << testNet.getNetValue().format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "array has " << pars.size() << " parameters.";

  // do we need to feed in the Net template parameter, or can it be inferred from testNet?
  check_gradient( testNet, input, output );

  testNet.accessNetValue() = randomWeights<Net>();
  input.setRandom();
  // output << 1e-6, 0.02, 0.2, 0.01;
  output.setRandom();
  check_gradient( testNet, input, output );

  testNet.accessNetValue() = randomWeights<Net>();
  input.setRandom(); // should also check pathological x values
  // output << 0.9, -0.1, 10.0, 0.1; // this one has nonsensical values but we can check the gradient regardless (it may give warning messages)
  output.setRandom();
  check_gradient( testNet, input, output );

  toFile(testNet.getNetValue(), "testcopy.net");
  Net netCopy;
  netCopy.accessNetValue() = fromFile("testcopy.net");
  if ( testNet != netCopy ) {
    BOOST_LOG_TRIVIAL(warning) << "loaded net is not the same!";
    BOOST_LOG_TRIVIAL(warning) << "original:\n" << testNet.getNetValue().transpose().format(my_fmt);
    BOOST_LOG_TRIVIAL(warning) << "copy:\n" << netCopy.getNetValue().transpose().format(my_fmt);
    auto diff = testNet.getNetValue() - netCopy.getNetValue();
    BOOST_LOG_TRIVIAL(warning) << "difference:\n" << diff.transpose().format(my_fmt);
  }

#else
  BOOST_LOG_TRIVIAL(info) << "skipping static nets.";
#endif // ifndef NO_STATIC
  
  return 0;
}

