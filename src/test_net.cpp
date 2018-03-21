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

// template <size_t N>
// using Vec = Eigen::Matrix<double, N, 1>;

// borrow this function from other testing utility to check the neural net's derivative
// we will really want to check element-by-element
// template <size_t N>
// void check_gradient(func_t<N> f, grad_t<N> g, const parvec<N>& p, double ep=1e-6, double tol=1e-2) {
void check_gradient(SingleHiddenLayer<8, 4>& net, const Vec<8>& xin, const double yin, double ep=1e-4, double tol=1e-4) {
  // assume that a data point has already been put in
  constexpr size_t N = net.size();
  parvec<N> p = net.getNetValue(); // don't make this a reference: the internal data will change!

  net.propagateData(xin, yin); // this must be done before 
  double fval = net.getCostFuncVal(); // don't think we actually need this, but it might be a nice check

  parvec<N> evalgrad = net.getCostFuncGrad();
  double gradmag = sqrt(evalgrad.square().sum());
  parvec<N> dir = (1.0+p.square()).sqrt()*evalgrad/gradmag; // vector along direction of greatest change
  double dirmag = sqrt(dir.square().sum());
  parvec<N> dpar = ep*dir;

  net.accessNetValue() += dpar;
  net.propagateData(xin, yin);
  double fplus = net.getCostFuncVal();

  net.accessNetValue() = p-dpar;
  net.propagateData(xin, yin);
  double fminus = net.getCostFuncVal();


  parvec<N> df;// maybe can do this slickly with colwise() or rowwise() ?
  for (size_t i_f=0; i_f<N; ++i_f) {
    parvec<N> dp = Array<N>::Zero();
    double dpi = ep*sqrt(1.0 + p(i_f)*p(i_f));
    dp(i_f) = dpi;
    parvec<N> pplus = p + dp;
    parvec<N> pminus = p - dp;
    net.accessNetValue() = pplus;
    net.propagateData(xin, yin);
    double fplusi = net.getCostFuncVal();
    net.accessNetValue() = pminus;
    net.propagateData(xin, yin);
    double fminusi = net.getCostFuncVal();
    df(i_f) = (fplusi - fminusi)/(2*dpi);
  }

  // now check this element-by element
  if ( (df-evalgrad).square().sum() > tol*tol*1
       || !df.isFinite().all()
       || !evalgrad.isFinite().all()) {
    cout << "gradient check failed!" << endl;
    cout << "f(p) = " << fval << endl;
    cout << "numerical derivative = " << df << endl;
    cout << "analytic gradient = " << evalgrad << endl;
    cout << "difference = " << df - evalgrad << endl;
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

  SingleHiddenLayer<8, 4/*, 2*/> testNet;
  constexpr size_t netsize = testNet.size();
  testNet.randomInit();

  Vec<8> input;
  input.setRandom();

  Vec<2> output;
  output << 0, 1;
  double yval = 0.5;// testing single-dimension for now
  
  Eigen::IOFormat my_fmt(2, // first value is the precision
			 0, ", ", "\n", "[", "]");

  testNet.propagateData( input, yval/*output*/ );
  cout << "output of random network is:  " << testNet.getOutput() << endl;
  cout << "first layer:\n" << testNet.getFirstSynapses().format(my_fmt) << endl;
  cout << "second layer:\n" << testNet.getSecondSynapses().format(my_fmt) << endl;
  auto& pars = testNet.getNetValue();
  // cout << "net value of array:\n" << testNet.getNetValue() << endl;
  cout << "array has " << pars.size() << " parameters." << endl;

  check_gradient( testNet, input, yval/*output*/ );

  input.setRandom();
  yval = 0.02;
  check_gradient( testNet, input, yval );

  input.setRandom();
  yval = 0.99;
  check_gradient( testNet, input, yval );
  
  return 0;
}

