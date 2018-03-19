#include "test_functions.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

using Eigen::sin;
using Eigen::cos;

// // iostream should not be included in general -- this is used temporarily for debugging only!
// #include <iostream>

double sphere( const parvec& pars )
{
  return pars.square().sum();
}

parvec grad_sphere( const parvec& pars )
{
  return 2.0*pars;
}

double ellipse( const parvec& pars, double scale )
{
  parvec coef = parvec::LinSpaced(pars.size(), 1.0, scale*pars.size());
  return (coef*pars.square()).sum();
}

parvec grad_ellipse( const parvec& pars, double scale )
{
  parvec coef = parvec::LinSpaced(pars.size(), 1.0, scale*pars.size());
  return 2.0*coef*pars;
}

double rosenbrock( const parvec& pars, double scale )
{
  size_t ntake = pars.size()-1;
  parvec first = pars.head(ntake);
  parvec last = pars.tail(ntake);
  auto result = scale*(last - first.square()).square().sum() + (1 - first).square().sum();
  
  return result;
}

parvec grad_rosenbrock( const parvec& pars, double scale )
{
  size_t psize = pars.size();
  parvec shift_forward = parvec::Zero(psize);
  parvec shift_back = parvec::Zero(psize);
  shift_forward.segment(1,psize-1) = pars.segment(0,psize-1); // shift_forward[k] == pars[k-1]
  shift_back.segment(0,psize-1) = pars.segment(1,psize-1);    // shift_back[k] == pars[k+1]

  parvec grad = scale*2.0*(pars - shift_forward.square() - 2*pars*(shift_back - pars.square())) - 2*(1-pars);
  // first and last elements are special because they are not coupled
  grad[0] = -4*scale*pars[0]*(pars[1] - pars[0]*pars[0]) - 2*(1-pars[0]);
  grad[psize-1] = 2.0*scale*(pars[psize-1] - pars[psize-2]*pars[psize-2]);

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
