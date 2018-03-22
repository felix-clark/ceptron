#include "net.hpp"

#include <Eigen/Core>
#include <iostream>

#include <functional>

using std::cout;
using std::endl;

template <size_t N>
using parvec = Eigen::Array<double, N, 1>;

template <size_t N>
using func_t = std::function<double(parvec<N>)>;
template <size_t N>
using grad_t = std::function<parvec<N>(parvec<N>)>;

const Eigen::IOFormat my_fmt(3, // first value is the precision
			     0, ", ", "\n", "[", "]");


// template <size_t N>
// using Vec = Eigen::Matrix<double, N, 1>;

// borrow this function from other testing utility to check the neural net's derivative
// we will really want to check element-by-element
// template <size_t N>
// void check_gradient(func_t<N> f, grad_t<N> g, const parvec<N>& p, double ep=1e-6, double tol=1e-2) {
template <size_t N, size_t M, size_t P, InternalActivator Act>
void check_gradient(SingleHiddenLayer<N, M, P, Act>& net, const Vec<N>& xin, const Vec<P>& yin, double ep=1e-4, double tol=1e-4) {
  // assume that a data point has already been put in
  constexpr size_t Npar = net.size();
  assert( Npar == M*(N+1) + P*(M+1) );
  const parvec<Npar> p = net.getNetValue(); // don't make this a reference: the internal data will change!

  net.propagateData(xin, yin); // this must be done before 
  double fval = net.getCostFuncVal(); // don't think we actually need this, but it might be a nice check

  parvec<Npar> evalgrad = net.getCostFuncGrad();
  double gradmag = sqrt(evalgrad.square().sum());
  parvec<Npar> dir = (1.0+p.square()).sqrt()*evalgrad/gradmag; // vector along direction of greatest change
  double dirmag = sqrt(dir.square().sum());
  parvec<Npar> dpar = ep*dir;

  net.accessNetValue() += dpar;
  net.propagateData(xin, yin);
  double fplus = net.getCostFuncVal();

  net.accessNetValue() = p-dpar;
  net.propagateData(xin, yin);
  double fminus = net.getCostFuncVal();


  parvec<Npar> df;// maybe can do this slickly with colwise() or rowwise() ?
  for (size_t i_f=0; i_f<Npar; ++i_f) {
    parvec<Npar> dp = Array<Npar>::Zero();
    double dpi = ep*sqrt(1.0 + p(i_f)*p(i_f));
    dp(i_f) = dpi;
    parvec<Npar> pplus = p + dp;
    parvec<Npar> pminus = p - dp;
    net.accessNetValue() = pplus;
    net.propagateData(xin, yin);
    double fplusi = net.getCostFuncVal();
    net.accessNetValue() = pminus;
    net.propagateData(xin, yin);
    double fminusi = net.getCostFuncVal();
    df(i_f) = (fplusi - fminusi)/(2*dpi);
    // std::cout << dp << std::endl;
  }

  // now check this element-by element
  double sqSumDiff = (df-evalgrad).square().sum();
  if ( sqSumDiff > tol*tol*1
       || !df.isFinite().all()
       || !evalgrad.isFinite().all()) {
    cout << "gradient check failed at tolerance level " << tol << " (" << sqrt(sqSumDiff) << ")" << endl;
    cout << "f(p) = " << fval << endl;
    cout << "numerical derivative = " << df.transpose().format(my_fmt) << endl;
    cout << "analytic gradient = " << evalgrad.transpose().format(my_fmt) << endl;
    cout << "difference = " << (df - evalgrad).transpose().format(my_fmt) << endl;
  }
  
  // we should actually check the numerical derivative element-by-element
  // double deltaf = (f(p+ep*dir) - f(p-ep*dir))/(2*ep*dirmag); // should be close to |gradf|
  double deltaf = (fplus - fminus)/(2*ep*dirmag);
  if ( fabs(deltaf) - (dir*evalgrad).sum()/dirmag > tol*fabs(deltaf + (dir*evalgrad).sum()/dirmag) ) {
    cout << "gradient magnitude check failed!" << endl;
    cout << "f(p) = " << fval << endl;
    cout << "numerical derivative along grad = " << deltaf << endl;
    cout << "analytic gradient magnitude = " << (dir*evalgrad).sum()/dirmag << endl;
    cout << "difference = " << fabs(deltaf) - (dir*evalgrad).sum()/dirmag << endl;
  }
}

int main(int argc, char** argv) {

  constexpr size_t Nin = 6;
  constexpr size_t Nh = 4;
  constexpr size_t Nout = 2;
  constexpr InternalActivator Act=InternalActivator::Tanh;
  
  SingleHiddenLayer<Nin, Nh, Nout, Act> testNet;
  constexpr size_t netsize = testNet.size();
  testNet.randomInit();

  Vec<Nin> input;
  input.setRandom();

  Vec<Nout> output;
  output << 0.5, 0.5;
  
  testNet.propagateData( input, output );
  cout << "input data is:  " << input.transpose().format(my_fmt) << endl;
  cout << "input layer cache is:  " << testNet.getInputLayerActivation().transpose().format(my_fmt) << endl;
  cout << "first weights:\n" << testNet.getFirstSynapses().format(my_fmt) << endl;
  cout << "hidden activation:\n" << testNet.getHiddenLayerActivation().transpose().format(my_fmt) << endl;
  cout << "second weights:\n" << testNet.getSecondSynapses().format(my_fmt) << endl;
  cout << "output of random network is:  " << testNet.getOutput().transpose().format(my_fmt) << endl;
  auto& pars = testNet.getNetValue();
  // cout << "net value of array:\n" << testNet.getNetValue().format(my_fmt) << endl;
  cout << "array has " << pars.size() << " parameters." << endl;

  check_gradient<Nin, Nh, Nout, Act>( testNet, input, output );

  testNet.randomInit();
  input.setRandom();
  output << 0.02, 0.2;
  check_gradient<Nin, Nh, Nout, Act>( testNet, input, output );

  testNet.randomInit();
  input.setRandom(); // should also check pathological x values
  output << 0.99, 10.0; // yval > 1 should actually do something weird
  check_gradient<Nin, Nh, Nout, Act>( testNet, input, output );
  
  return 0;
}

