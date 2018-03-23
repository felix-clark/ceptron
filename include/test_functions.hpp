// test functions to compare minimizers with
#include "global.hpp"
#include <Eigen/Dense>

#define _USE_MATH_DEFINES
#include <cmath>

namespace {
  using Eigen::sin;
  using Eigen::cos;
  using ceptron::Array;
}


// // an easy case that should converge to the origin
template <size_t N>
double sphere( const Array<N>& pars )
{
  return pars.square().sum();
}

template <size_t N>
Array<N> grad_sphere( const Array<N>& pars )
{
  return 2.0*pars;
}

template <size_t N>
double ellipse( const Array<N>& pars, double scale=16.0 )
{
  Array<N> coef = Array<N>::LinSpaced(1.0, scale*pars.size());
  return (coef*pars.square()).sum();
}

template <size_t N>
Array<N> grad_ellipse( const Array<N>& pars, double scale=16.0 )
{
  Array<N> coef = Array<N>::LinSpaced(1.0, scale*N);
  return 2.0*coef*pars;
}

template <size_t N>
double rosenbrock( const Array<N>& pars, double scale=100.0 )
{
  static_assert( N > 1, "rosenbrock function must have multiple parameters" );
  Array<N-1> first = pars.template head<N-1>();
  Array<N-1> last = pars.template tail<N-1>();
  auto result = scale*(last - first.square()).square().sum() + (1 - first).square().sum();
  
  return result;
}

// // should converge to all parameters having value 1.0
template <size_t N>
Array<N> grad_rosenbrock( const Array<N>& pars, double scale=100.0 )
{
  static_assert( N > 1, "rosenbrock function must have multiple parameters" );
  Array<N> shift_forward = Array<N>::Zero();
  Array<N> shift_back = Array<N>::Zero();
  shift_forward.template segment<N-1>(1) = pars.template segment<N-1>(0); // shift_forward[k] == pars[k-1]
  shift_back.template segment<N-1>(0) = pars.template segment<N-1>(1);    // shift_back[k] == pars[k+1]

  Array<N> grad = scale*2.0*(pars - shift_forward.square() - 2*pars*(shift_back - pars.square())) - 2*(1-pars);
  // first and last elements are special because they are not coupled
  grad[0] = -4*scale*pars[0]*(pars[1] - pars[0]*pars[0]) - 2*(1-pars[0]);
  grad[N-1] = 2.0*scale*(pars[N-1] - pars[N-2]*pars[N-2]);

  return grad;
}

// // a highly oscillating function.
// // global minimum at origin but has many local minima.
// // probably too pathological for ML minimizers
template <size_t N>
double rastrigin( const Array<N>& pars, double scale )
{
  // size_t n = pars.size();
  return scale*N + (pars.square() - scale*cos(2*M_PI*pars)).sum();
}

template <size_t N>
Array<N> grad_rastrigin( const Array<N>& pars, double scale )
{
  return 2*pars + scale*2*M_PI*sin(2*M_PI*pars);
}
