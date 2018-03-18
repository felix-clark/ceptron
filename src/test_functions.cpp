#include "test_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

using Eigen::sin;
using Eigen::cos;

double sphere( const parvec& pars )
{
  return pars.square().sum();
}

parvec grad_sphere( const parvec& pars )
{
  return 2.0*pars;
}

double rosenbrock( const parvec& pars, double scale )
{
  size_t ntake = pars.size()-1;
  parvec first = pars.head(ntake);
  parvec last = pars.tail(ntake);
  return scale*(last - first.square()).square().sum() + (1 - first).square().sum();
}

parvec grad_rosenbrock( const parvec& pars, double scale )
{
  size_t psize = pars.size();
  parvec shift_forward = parvec(psize);
  parvec shift_back = parvec(psize);
  shift_forward.segment(1,psize-1) = pars.segment(0,psize-1); // shift_forward[k] == pars[k-1]
  shift_back.segment(0,psize-1) = pars.segment(1,psize-1);    // shift_back[k] == pars[k+1]
  parvec grad = scale*2.0*(pars - shift_forward.square() - 2*pars*(shift_back - pars.square())) - 2*(1-pars);
  grad[psize-1] = 2.0*scale*(pars[psize-1] - pars[psize-2]*pars[psize-2]); // the last element is a special case since it's not included in the main sum
  return grad;
}

double rastrigin( const parvec& pars, double scale )
{
  size_t n = pars.size();
  return scale*n + (pars.square() - scale*cos(2*M_PI*pars)).sum();
}

parvec grad_rastrigin( const parvec& pars, double scale )
{
  return 2*pars + scale*2*M_PI*sin(2*M_PI*pars);
}
