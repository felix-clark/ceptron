#include "global.hpp"
#include "min_step.hpp"
#include "test_functions.hpp"
#include <boost/log/trivial.hpp>
#include <iostream>

namespace {
  using namespace ceptron;
} // namespace

template <size_t N>
void check_gradient(func_t<N> f, grad_t<N> g, Array<N> p, double ep=1e-6, double tol=1e-2) {
  Array<N> evalgrad = g(p);
  double gradmag = sqrt(evalgrad.square().sum());
  Array<N> dir = (1.0+p.square()).sqrt()*evalgrad/gradmag; // vector along direction of greatest change
  double dirmag = sqrt(dir.square().sum());
  // we should actually check the numerical derivative element-by-element
  double deltaf = (f(p+ep*dir) - f(p-ep*dir))/(2*ep*dirmag); // should be close to |gradf|
  if ( fabs(deltaf) - (dir*evalgrad).sum()/dirmag > tol*fabs(deltaf + (dir*evalgrad).sum()/dirmag) ) {
    BOOST_LOG_TRIVIAL(error) << "gradient check failed!";
    BOOST_LOG_TRIVIAL(error) << "f(p) = " << f(p);
    BOOST_LOG_TRIVIAL(error) << "numerical derivative = " << deltaf;
    BOOST_LOG_TRIVIAL(error) << "analytic gradient magnitude = " << (dir*evalgrad).sum()/dirmag;
    BOOST_LOG_TRIVIAL(error) << "difference = " << fabs(deltaf) - (dir*evalgrad).sum()/dirmag;
  }
}

template <size_t N>
struct min_result
{
  Array<N> x;
  double f;
  Array<N> grad;
  size_t iterations = 0;
  int fail = 0;
};

template <size_t N>
min_result<N> test_min_step( IMinStep<N>& minstep, func_t<N> f, grad_t<N> g, Array<N> initpars,
			  double abstol=1e-10, size_t maxsteps=64000 ) {
  minstep.resetCache(); // need to make sure caches are reset!
  
  Array<N> pars = initpars;
  size_t n_iterations = 0;
  bool finished = false;
  while( !finished ) {
    check_gradient<N>( f, g, pars );
    Array<N> dpars = minstep.getDeltaPar( f, g, pars );
    if (isnan(dpars).any()) {
      BOOST_LOG_TRIVIAL(error) << "NaN in proposed delta pars!";
      
      return {pars, f(pars), g(pars), n_iterations, 1};
    }

    pars += dpars;
    ++n_iterations;
    // just check the gradient magnitude for now
    // ArrayXd does not have a squaredNorm() function like VectorXd
    if (g(pars).square().sum() < abstol*abstol || n_iterations >= maxsteps) {
      finished = true;
    }
  }
  return {pars, f(pars), g(pars), n_iterations};
}

template <size_t N>
void print_pars( Array<N> pars ) {
  for (size_t i_par=0; i_par<N; ++i_par) {
    BOOST_LOG_TRIVIAL(info) << pars[i_par] << ", ";
  }
  BOOST_LOG_TRIVIAL(info);
}

template <size_t N>
void print_result( min_result<N> res ) {
  BOOST_LOG_TRIVIAL(info);
  if (res.fail) {
    BOOST_LOG_TRIVIAL(error) << "fit did not converge!";
  }
  double f = res.f;
  BOOST_LOG_TRIVIAL(info) << "f(x) = " << f;
  BOOST_LOG_TRIVIAL(info) << "took " << res.iterations << " steps.";
  BOOST_LOG_TRIVIAL(info) << "gradient:\t";
  print_pars<N>( res.grad );
  BOOST_LOG_TRIVIAL(info) << "x:\t";
  print_pars<N>( res.x );
  BOOST_LOG_TRIVIAL(info);
}

template <size_t N>
void run_test_functions( IMinStep<N>& minstep, Array<N> initpars ) {
  // assert( minstep != nullptr );
  using namespace mintest;
  BOOST_LOG_TRIVIAL(info) << "testing basic ellipse function";
  // basic gradient descent has trouble with a large scale in ellipse function
  // use lambda function to automatically use default scale parameter
  // auto f_ellipse = [](Array<N> p){return ellipse<N>(p);};
  std::function<double(Array<N>)> f_ellipse = [](Array<N> p){return ellipse<N>(p);};
  // auto g_ellipse = [](Array<N> p){return grad_ellipse<N>(p);};
  std::function<Array<N>(Array<N>)> g_ellipse = [](Array<N> p){return grad_ellipse<N>(p);};
  min_result<N> gd_result = test_min_step<N>( minstep, f_ellipse, g_ellipse, initpars );
  BOOST_LOG_TRIVIAL(info) << "final parameters for ellipse: ";
  print_result( gd_result );

  BOOST_LOG_TRIVIAL(info) << "testing rosenbrock function";
  auto f_rosen = [](Array<N> p){return rosenbrock<N>(p);};
  auto g_rosen = [](Array<N> p){return grad_rosenbrock<N>(p);};  
  min_result<N> rosen_result = test_min_step<N>( minstep, f_rosen, g_rosen, initpars );
  BOOST_LOG_TRIVIAL(info) << "final parameters for rosenbrock: ";
  print_result( rosen_result );
}

int main( int, char** ) {

  // constexpr size_t ndim = 4;
  constexpr size_t ndim = 2;
  Array<ndim> initpars;
  initpars = 2 * Array<ndim>::Random();
  BOOST_LOG_TRIVIAL(info) << "initial parameters: ";
  print_pars<ndim>( initpars );
  
  GradientDescent<ndim> gd;
  gd.setLearnRate(0.001);
  BOOST_LOG_TRIVIAL(info) << "  ... gradient descent ...";
  run_test_functions<ndim>( gd, initpars );

  GradientDescentWithMomentum<ndim> gdm;
  gdm.setLearnRate(0.001); // default learn rate is too large for rosenbrock
  BOOST_LOG_TRIVIAL(info) << "  ... MOM ...";
  run_test_functions<ndim>( gdm, initpars );

  AcceleratedGradient<ndim> ag;
  // ag.setMomentumScale(0.5);
  ag.setLearnRate(0.001); // we need to turn down the learn rate significantly or it diverges on rosenbrock
  BOOST_LOG_TRIVIAL(info) << "  ... NAG ...";
  run_test_functions<ndim>( ag, initpars );

  // ADADELTA converges in much fewer steps for basic sphere.
  // it seems to diverge for rosenbrock function right now...
  BOOST_LOG_TRIVIAL(info) << "  ... ADADELTA ...";
  AdaDelta<ndim> ad;
  // ad.setLearnRate(0.5);
  // ad.setDecayScale(0.5);
  run_test_functions<ndim>( ad, initpars );

  // BFGS isn't going to be useful for machine learning but let's compare it to other minimizers
  // it's possible that LBFGS can be used, but may need some adaptation for stochastic/mini-batch minimization
  BOOST_LOG_TRIVIAL(info) << "  ... BFGS ...";
  Bfgs<ndim> bfgs;
  run_test_functions<ndim>( bfgs, initpars );
  
  return 0;
}
