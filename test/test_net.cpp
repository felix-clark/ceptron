#include "global.hpp"
#include "net.hpp"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <Eigen/Core>
#include <iostream>
#include <functional>

using namespace ceptron;

template <size_t N>
using func_t = std::function<double(Array<N>)>;
template <size_t N>
using grad_t = std::function<Array<N>(Array<N>)>;

const Eigen::IOFormat my_fmt(3, // first value is the precision
			     0, ", ", "\n", "[", "]");


// borrow this function from other testing utility to check the neural net's derivative
// we will really want to check element-by-element
template <size_t Nin, size_t Nout, size_t Ntot>
void check_gradient(IFeedForward<Nin, Nout, Ntot>& net, const Vec<Nin>& xin, const Vec<Nout>& yin, double ep=1e-4, double tol=1e-4) {
  constexpr size_t Npar = Ntot;
  const Array<Npar> p = net.getNetValue(); // don't make this a reference: the internal data will change!

  net.propagateData(xin, yin); // this must be done before 
  double fval = net.getCostFuncVal(); // don't think we actually need this, but it might be a nice check

  Array<Npar> evalgrad = net.getCostFuncGrad();
  double gradmag = sqrt(evalgrad.square().sum());
  Array<Npar> dir = (1.0+p.square()).sqrt()*evalgrad/gradmag; // vector along direction of greatest change
  double dirmag = sqrt(dir.square().sum());
  Array<Npar> dpar = ep*dir;

  net.accessNetValue() += dpar;
  net.propagateData(xin, yin);
  double fplus = net.getCostFuncVal();

  net.accessNetValue() = p-dpar;
  net.propagateData(xin, yin);
  double fminus = net.getCostFuncVal();


  Array<Npar> df;// maybe can do this slickly with colwise() or rowwise() ?
  for (size_t i_f=0; i_f<Npar; ++i_f) {
    Array<Npar> dp = Array<Npar>::Zero();
    double dpi = ep*sqrt(1.0 + p(i_f)*p(i_f));
    dp(i_f) = dpi;
    Array<Npar> pplus = p + dp;
    Array<Npar> pminus = p - dp;
    net.accessNetValue() = pplus;
    net.propagateData(xin, yin);
    double fplusi = net.getCostFuncVal();
    net.accessNetValue() = pminus;
    net.propagateData(xin, yin);
    double fminusi = net.getCostFuncVal();
    df(i_f) = (fplusi - fminusi)/(2*dpi);
  }

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
  
  // we should actually check the numerical derivative element-by-element
  // double deltaf = (f(p+ep*dir) - f(p-ep*dir))/(2*ep*dirmag); // should be close to |gradf|
  double deltaf = (fplus - fminus)/(2*ep*dirmag);
  if ( fabs(deltaf) - (dir*evalgrad).sum()/dirmag > tol*fabs(deltaf + (dir*evalgrad).sum()/dirmag) ) {
    BOOST_LOG_TRIVIAL(info) << "gradient magnitude check failed!";
    BOOST_LOG_TRIVIAL(info) << "f(p) = " << fval;
    BOOST_LOG_TRIVIAL(info) << "numerical derivative along grad = " << deltaf;
    BOOST_LOG_TRIVIAL(info) << "analytic gradient magnitude = " << (dir*evalgrad).sum()/dirmag;
    BOOST_LOG_TRIVIAL(info) << "difference = " << fabs(deltaf) - (dir*evalgrad).sum()/dirmag;
  }
}

int main(int argc, char** argv) {

  constexpr size_t Nin = 8;
  constexpr size_t Nh = 4;
  constexpr size_t Nout = 4;
  constexpr RegressionType Reg=RegressionType::Categorical;
  // constexpr RegressionType Reg=RegressionType::LeastSquares;
  // constexpr InternalActivator Act=InternalActivator::Tanh;
  // constexpr InternalActivator Act=InternalActivator::ReLU; // we had gradient issues w/ this because we were overriding it accidentally
  // however the nested select() to attempt to check for x=0 screws up the gradient completely
  // constexpr InternalActivator Act=InternalActivator::LReLU;
  constexpr InternalActivator Act=InternalActivator::Softplus;

  // we can adjust the log level like so: (e.g. turn on debug)
  namespace logging = boost::log;
  logging::core::get()->set_filter
    (logging::trivial::severity >= logging::trivial::info);
  
  SingleHiddenLayer<Nin, Nh, Nout, Reg, Act> testNet;
  constexpr size_t netsize = testNet.size();
  testNet.randomInit();

  Vec<Nin> input;
  input.setRandom();

  Vec<Nout> output;
  output << 0.5, 0.25, 0.1, 0.01;
  
  testNet.propagateData( input, output );
  BOOST_LOG_TRIVIAL(info) << "input data is:  " << input.transpose().format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "input layer cache is:  " << testNet.getInputLayerActivation().transpose().format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "first weights:\n" << testNet.getFirstSynapses().format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "hidden activation:\n" << testNet.getHiddenLayerActivation().transpose().format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "second weights:\n" << testNet.getSecondSynapses().format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "output of random network is:  " << testNet.getOutput().transpose().format(my_fmt);
  auto& pars = testNet.getNetValue();
  // BOOST_LOG_TRIVIAL(info) << "net value of array:\n" << testNet.getNetValue().format(my_fmt);
  BOOST_LOG_TRIVIAL(info) << "array has " << pars.size() << " parameters.";

  check_gradient<Nin, Nout, netsize>( testNet, input, output );

  testNet.randomInit();
  input.setRandom();
  output << 1e-6, 0.02, 0.2, 0.01;
  check_gradient<Nin, Nout, netsize>( testNet, input, output );

  testNet.randomInit();
  input.setRandom(); // should also check pathological x values
  // output << 0.9, -0.1, 10.0, 0.1; // this one has nonsensical values but we can check the gradient regardless (it may give warning messages)
  output.setRandom();
  check_gradient<Nin, Nout, netsize>( testNet, input, output );

  testNet.toFile("testcopy.net");
  SingleHiddenLayer<Nin, Nh, Nout, Reg, Act> netCopy;
  netCopy.fromFile("testcopy.net");
  if ( testNet != netCopy) {
    BOOST_LOG_TRIVIAL(warning) << "loaded net is not the same!";
    BOOST_LOG_TRIVIAL(warning) << "original:\n" << testNet.getNetValue().transpose().format(my_fmt);
    BOOST_LOG_TRIVIAL(warning) << "copy:\n" << netCopy.getNetValue().transpose().format(my_fmt);
    auto diff = testNet.getNetValue() - netCopy.getNetValue();
    BOOST_LOG_TRIVIAL(warning) << "difference:\n" << diff.transpose().format(my_fmt);
  }
  
  return 0;
}

