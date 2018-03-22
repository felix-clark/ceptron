#include "min_step.hpp"
#include "test_functions.hpp"

#include <iostream>

using std::cout;
using std::endl;

template <size_t N>
void check_gradient(func_t<N> f, grad_t<N> g, parvec<N> p, double ep=1e-6, double tol=1e-2) {
  parvec<N> evalgrad = g(p);
  double gradmag = sqrt(evalgrad.square().sum());
  parvec<N> dir = (1.0+p.square()).sqrt()*evalgrad/gradmag; // vector along direction of greatest change
  double dirmag = sqrt(dir.square().sum());
  // we should actually check the numerical derivative element-by-element
  double deltaf = (f(p+ep*dir) - f(p-ep*dir))/(2*ep*dirmag); // should be close to |gradf|
  if ( fabs(deltaf) - (dir*evalgrad).sum()/dirmag > tol*fabs(deltaf + (dir*evalgrad).sum()/dirmag) ) {
    cout << "gradient check failed!" << endl;
    cout << "f(p) = " << f(p) << endl;
    cout << "numerical derivative = " << deltaf << endl;
    cout << "analytic gradient magnitude = " << (dir*evalgrad).sum()/dirmag << endl;
    cout << "difference = " << fabs(deltaf) - (dir*evalgrad).sum()/dirmag << endl;
  }
}

template <size_t N>
struct min_result
{
  parvec<N> x;
  double f;
  parvec<N> grad;
  size_t iterations = 0;
  int fail = 0;
};

template <size_t N>
min_result<N> test_min_step( IMinStep<N>& minstep, func_t<N> f, grad_t<N> g, parvec<N> initpars,
			  double abstol=1e-10, size_t maxsteps=64000 ) {
  minstep.resetCache(); // need to make sure caches are reset!
  
  parvec<N> pars = initpars;
  size_t n_iterations = 0;
  bool finished = false;
  while( !finished ) {
    check_gradient<N>( f, g, pars );
    parvec<N> dpars = minstep.getDeltaPar( f, g, pars );
    if (isnan(dpars).any()) {
      cout << "NaN in proposed delta pars!" << endl;
      
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
void print_pars( parvec<N> pars ) {
  for (size_t i_par=0; i_par<N; ++i_par) {
    cout << pars[i_par] << ", ";
  }
  cout << endl;
}

template <size_t N>
void print_result( min_result<N> res ) {
  cout << endl;
  if (res.fail) {
    cout << "fit did not converge!" << endl;
  }
  double f = res.f;
  cout << "f(x) = " << f << endl;
  cout << "took " << res.iterations << " steps." << endl;
  cout << "gradient:\t";
  print_pars<N>( res.grad );
  cout << "x:\t";
  print_pars<N>( res.x );
  cout << endl;
}

template <size_t N>
void run_test_functions( IMinStep<N>& minstep, parvec<N> initpars ) {
  // assert( minstep != nullptr );
  
  cout << "testing basic ellipse function" << endl;
  // basic gradient descent has trouble with a large scale in ellipse function
  // use lambda function to automatically use default scale parameter
  // auto f_ellipse = [](parvec<N> p){return ellipse<N>(p);};
  std::function<double(parvec<N>)> f_ellipse = [](parvec<N> p){return ellipse<N>(p);};
  // auto g_ellipse = [](parvec<N> p){return grad_ellipse<N>(p);};
  std::function<parvec<N>(parvec<N>)> g_ellipse = [](parvec<N> p){return grad_ellipse<N>(p);};
  min_result<N> gd_result = test_min_step<N>( minstep, f_ellipse, g_ellipse, initpars );
  cout << "final parameters for ellipse: ";
  print_result( gd_result );

  cout << "testing rosenbrock function" << endl;
  auto f_rosen = [](parvec<N> p){return rosenbrock<N>(p);};
  auto g_rosen = [](parvec<N> p){return grad_rosenbrock<N>(p);};  
  min_result<N> rosen_result = test_min_step<N>( minstep, f_rosen, g_rosen, initpars );
  cout << "final parameters for rosenbrock: ";
  print_result( rosen_result );
}

int main( int argc, char** argv ) {

  // constexpr size_t ndim = 4;
  constexpr size_t ndim = 2;
  parvec<ndim> initpars;
  initpars = 2 * parvec<ndim>::Random();
  cout << "initial parameters: ";
  print_pars<ndim>( initpars );
  
  GradientDescent<ndim> gd;
  gd.setLearnRate(0.001);
  cout << "  ... gradient descent ..." << endl;
  run_test_functions<ndim>( gd, initpars );

  GradientDescentWithMomentum<ndim> gdm;
  gdm.setLearnRate(0.001); // default learn rate is too large for rosenbrock
  cout << "  ... MOM ..." << endl;
  run_test_functions<ndim>( gdm, initpars );

  AcceleratedGradient<ndim> ag;
  // ag.setMomentumScale(0.5);
  ag.setLearnRate(0.001); // we need to turn down the learn rate significantly or it diverges on rosenbrock
  cout << "  ... NAG ..." << endl;
  run_test_functions<ndim>( ag, initpars );

  // ADADELTA converges in much fewer steps for basic sphere.
  // it seems to diverge for rosenbrock function right now...
  cout << "  ... ADADELTA ..." << endl;
  AdaDelta<ndim> ad;
  // ad.setLearnRate(0.5);
  // ad.setDecayScale(0.5);
  run_test_functions<ndim>( ad, initpars );

  // BFGS isn't going to be useful for machine learning but let's compare it to other minimizers
  // it's possible that LBFGS can be used, but may need some adaptation for stochastic/mini-batch minimization
  cout << "  ... BFGS ..." << endl;
  Bfgs<ndim> bfgs;
  run_test_functions<ndim>( bfgs, initpars );
  
  return 0;
}
