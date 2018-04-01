#include "global.hpp"
#include "net.hpp"
#include "net_dyn.hpp"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <Eigen/Core>
#include <iostream>
#include <functional>

using namespace ceptron;

namespace {
  // a local formatting option block for convenience
  const Eigen::IOFormat my_fmt(3, // first value is the precision
			       0, ", ", "\n", "[", "]");
}

// borrow this function from other testing utility to check the neural net's derivative
// we will really want to check element-by-element
template <typename Net,
// template <size_t Nin, size_t Nout, size_t Nhid,
	  RegressionType Reg, InternalActivator Act>
void check_gradient(Net& net, const BatchVec<Net::inputs>& xin, const BatchVec<Net::outputs>& yin, double l2reg=0.01, double ep=1e-4, double tol=1e-8) {
  constexpr size_t Npar = Net::size;
  // const/*expr*/ size_t Npar = net.size();
  const Array</*Npar*/> p = net.getNetValue(); // don't make this a reference: the internal data will change!
  // func_grad_res</*Npar*/> fgvals = costFuncAndGrad<Nin, Nout, Nhid, Reg, Act>(net, xin, yin, l2reg); // this must be done before 
  func_grad_res</*Npar*/> fgvals = costFuncAndGrad<Net, Reg, Act>(net, xin, yin, l2reg); // this must be done before
  // or does the following work? :
  // func_grad_res</*Npar*/> fgvals = costFuncAndGrad<Reg, Act>(net, xin, yin, l2reg); // this must be done before 
  double fval = fgvals.f; // don't think we actually need this, but it might be a nice check
  Array</*Npar*/> evalgrad = fgvals.g;

  BOOST_LOG_TRIVIAL(trace) << "about to try to compute numerical derivative";
  
  Array</*Npar*/> df(Npar);// maybe can do this slickly with colwise() or rowwise() ?
  for (size_t i_f=0; i_f<Npar; ++i_f) {
    Array</*Npar*/> dp = Array<Npar>::Zero(Npar);
    BOOST_LOG_TRIVIAL(trace) << "declaring dpi";
    double dpi = ep*sqrt(1.0 + p(i_f)*p(i_f));
    dp(i_f) = dpi;
    Array</*Npar*/> pplus = p + dp;
    Array</*Npar*/> pminus = p - dp;
    net.accessNetValue() = pplus;
    double fplusi = costFunc<Net, Reg, Act>(net, xin, yin, l2reg);
    net.accessNetValue() = pminus;
    double fminusi = costFunc<Net, Reg, Act>(net, xin, yin, l2reg);
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

int main(int, char**) {
  constexpr size_t Nin = 8;
  constexpr size_t Nout = 4;
  constexpr size_t Nh = 6; // number of nodes in hidden layer
  constexpr RegressionType Reg=RegressionType::Categorical;
  // constexpr RegressionType Reg=RegressionType::LeastSquares;
  // constexpr InternalActivator Act=InternalActivator::Tanh;
  // constexpr InternalActivator Act=InternalActivator::ReLU; // we had gradient issues w/ this because we were overriding it accidentally
  // however the nested select() to attempt to check for x=0 screws up the gradient completely
  // constexpr InternalActivator Act=InternalActivator::LReLU;
  constexpr InternalActivator Act=InternalActivator::Softplus;
  constexpr int batchSize = 16;

  FfnDyn netd(Nin, Nout);
  
  // to speed up compilation we can disable the static tests while we develop the dynamic case
#ifndef NOSTATIC

  // defines the architecture of our test NN
  using Net = SingleHiddenLayerStatic<Nin, Nout, Nh>;
  
  // we can adjust the log level like so:
  namespace logging = boost::log;
  logging::core::get()->set_filter
    (logging::trivial::severity >= logging::trivial::debug);
  
  Net testNet;
  testNet.randomInit();

  BatchVec<Nin> input(Nin, batchSize); // i guess we need the length in the constructor still?
  input.setRandom();

  BatchVec<Nout> output(Nout, batchSize);
  // with a large-ish batch there are too many numbers to plug in manually
  // output << 0.5, 0.25, 0.1, 0.01;
  output.setRandom();
  
  // we need to change the interface here to spit out intermediate-layer activations
  // "activation" only has a useful debug meaning for a single layer
  // testNet.propagateData( input, output );
  BOOST_LOG_TRIVIAL(info) << "input data is:\n" << input.format(my_fmt);
  // BOOST_LOG_TRIVIAL(info) << "input layer cache is:  " << testNet.getInputLayerActivation().transpose().format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "first weights:\n" << testNet.getFirstSynapses().format(my_fmt);
  // BOOST_LOG_TRIVIAL(info) << "hidden activation:\n" << testNet.getHiddenLayerActivation().transpose().format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "second weights:\n" << testNet.getSecondSynapses().format(my_fmt);
  // BOOST_LOG_TRIVIAL(info) << "output of random network is:  " << testNet.getPrediction().transpose().format(my_fmt);
  auto& pars = testNet.getNetValue();
  // BOOST_LOG_TRIVIAL(info) << "net value of array:\n" << testNet.getNetValue().format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "array has " << pars.size() << " parameters.";

  // do we need to feed in the Net template parameter, or can it be inferred from testNet?
  check_gradient<Net, Reg, Act>( testNet, input, output );

  testNet.randomInit();
  input.setRandom();
  // output << 1e-6, 0.02, 0.2, 0.01;
  output.setRandom();
  check_gradient<Net, Reg, Act>( testNet, input, output );

  testNet.randomInit();
  input.setRandom(); // should also check pathological x values
  // output << 0.9, -0.1, 10.0, 0.1; // this one has nonsensical values but we can check the gradient regardless (it may give warning messages)
  output.setRandom();
  check_gradient<Net, Reg, Act>( testNet, input, output );

  testNet.toFile("testcopy.net");
  Net netCopy;
  netCopy.fromFile("testcopy.net");
  if ( testNet != netCopy) {
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

