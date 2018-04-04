#include "min_step.hpp"
#include <boost/log/trivial.hpp>
#include <utility> // for std::swap

using Eigen::sqrt;


// will perform a 1D simplex minimum search to quickly find a step size
// could possibly use instead a golden search on the interval (0,1)
double ceptron::line_search( std::function<double(double)> f, double xa, double xb, size_t maxiter, double tol ) {
  constexpr double alpha = 1.0;
  constexpr double gamma = 2.0;
  constexpr double rho = 0.5;
  // constexpr double sigma = 0.5;
  constexpr double sigma = 0.25; // if this is equal to rho then the 1D algorithm has some redundant checks

  double fa = f(xa);
  double fb = f(xb);

  for (size_t i_iter=0; i_iter<maxiter; ++i_iter ) {
    // check condition for standard deviation of f values being small, either absolutely or relatively depending on how large f is
    if (std::isnan(fa) || std::isnan(fb)) {
      // don't waste time looping if the values are nonsense
      break;
    }
    if ( (fa-fb)*(fa-fb) < tol*tol*(1.0 + (fa+fb)*(fa+fb)) ) {
      // tolerance reached
      BOOST_LOG_TRIVIAL(trace) << "exiting line search due to tolerance reached";
      break;
    }
    
    if (fa > fb) {
      std::swap( xa, xb );
      std::swap( fa, fb );
    }
    if ( !( fa <= fb) ) {
      // we shouldn't get here -- possibly NaNs?
      BOOST_LOG_TRIVIAL(warning) << "f(xa) = " << fa << ", f(xb) = " << fb;
      BOOST_LOG_TRIVIAL(warning) << "iteration number " << i_iter;
    }
    assert( !(fa > fb) ); // xa is now the best value

    double xr = xa + alpha*(xa - xb); // reflected
    double fr = f(xr);
    if (fr < fa) {
      // reflected point is best so far, replace worst point with the either it or this expanded point:
      double xe = xa + gamma*(xr - xa); // = xa + gamma*alpha*(xa-xb)
      double fe = f(xe);
      if (fe < fr) {
	xb = xe;
	fb = fe;
      } else {
	xb = xr;
	fb = fr;
      }
      continue; // go to next step of loop
    }

    assert( !(fr < fa) );
    // contracted point (which is just average)
    double xc = xa + rho*(xb - xa);
    double fc = f(xc);
    // assert( fc == fc );
    if (fc < fb) {
      // replace worst point with contracted one
      xb = xc;
      fb = fc;
      continue;
    }

    // shrink
    // in 1D this is almost redundant with the contraction
    // will change value of sigma from wikipedia's recommended value to change this
    // it's actually a problem if we get here: it means there's a local maximum between xa and xb
    // and we do indeed get here...
    xb = xa + sigma*(xb - xa);
    fb = f(xb);

  }

  return fb < fa ? xb : xa;
}
