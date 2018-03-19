#include "min_step.hpp"
#include "test_functions.hpp"

#include <iostream>

using std::cout;
using std::endl;

void check_gradient(func_t f, grad_t g, parvec p, double ep=1e-6, double tol=1e-2) {
  parvec evalgrad = g(p);
  double gradmag = sqrt(evalgrad.square().sum());
  parvec dir = (1.0+p.square()).sqrt()*evalgrad/gradmag; // vector along direction of greatest change
  // we should actually check the numerical derivative element-by-element
  double deltaf = (f(p+ep*dir) - f(p-ep*dir))/(2*ep*sqrt(dir.square().sum())); // should be close to |gradf|
  if ( fabs(deltaf) - gradmag > tol*fabs(deltaf + gradmag) ) {
    cout << "gradient check failed!" << endl;
    cout << "f(p) = " << f(p) << endl;
    cout << "numerical derivative = " << deltaf << endl;
    cout << "analytic gradient magnitude = " << gradmag << endl;
    cout << "difference = " << fabs(deltaf) - gradmag << endl;
  }
}


struct min_result
{
  parvec x;
  double f;
  parvec grad;
  size_t iterations = 0;
  int fail = 0;
};

min_result test_min_step( IMinStep* minstep, func_t f, grad_t g, parvec initpars,
			  double abstol=1e-10, size_t maxsteps=64000 ) {
  assert( minstep != nullptr );
  minstep->resetCache(); // need to make sure caches are reset!
  
  parvec pars = initpars;
  size_t n_iterations = 0;
  bool finished = false;
  while( !finished ) {
    check_gradient( f, g, pars );
    parvec dpars = minstep->getDeltaPar( f, g, pars );
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

void print_pars( parvec pars ) {
  for (size_t i_par=0; i_par<pars.size(); ++i_par) {
    cout << pars[i_par] << ", ";
  }
  cout << endl;  
}

void print_result( min_result res ) {
  cout << endl;
  if (res.fail) {
    cout << "fit did not converge!" << endl;
  }
  double f = res.f;
  cout << "f(x) = " << f << endl;
  cout << "took " << res.iterations << " steps." << endl;
  cout << "gradient:\t";
  print_pars( res.grad );
  cout << "x:\t";
  print_pars( res.x );
  cout << endl;
}

void run_test_functions( IMinStep* minstep, parvec initpars ) {
  assert( minstep != nullptr );
  
  cout << "testing basic ellipse function" << endl;
  // basic gradient descent has trouble with a large scale in ellipse function
  // use lambda function to automatically use default scale parameter
  auto f_ellipse = [](parvec p){return ellipse(p);};
  auto g_ellipse = [](parvec p){return grad_ellipse(p);};
  min_result gd_result = test_min_step( minstep, f_ellipse, g_ellipse, initpars );
  cout << "final parameters for ellipse: ";
  print_result( gd_result );

  cout << "testing rosenbrock function" << endl;
  auto f_rosen = [](parvec p){return rosenbrock(p);};
  auto g_rosen = [](parvec p){return grad_rosenbrock(p);};  
  min_result rosen_result = test_min_step( minstep, f_rosen, g_rosen, initpars );
  cout << "final parameters for rosenbrock: ";
  print_result( rosen_result );
}

int main( int argc, char** argv ) {

  // size_t ndim = 4;
  size_t ndim = 2;
  parvec initpars(ndim);
  initpars = 2 * parvec::Random(ndim);
  cout << "initial parameters: ";
  print_pars( initpars );
  
  GradientDescent gd(0.001);
  cout << "  ... gradient descent ..." << endl;
  run_test_functions( &gd, initpars );

  GradientDescentWithMomentum gdm(ndim);
  gdm.setLearnRate(0.001); // default learn rate is too large for rosenbrock
  cout << "  ... MOM ..." << endl;
  run_test_functions( &gdm, initpars );

  AcceleratedGradient ag(ndim);
  // ag.setMomentumScale(0.5);
  ag.setLearnRate(0.001); // we need to turn down the learn rate significantly or it diverges on rosenbrock
  cout << "  ... NAG ..." << endl;
  run_test_functions( &ag, initpars );

  // ADADELTA converges in much fewer steps for basic sphere.
  // it seems to diverge for rosenbrock function right now...
  cout << "  ... ADADELTA ..." << endl;
  AdaDelta ad(ndim);
  // ad.setLearnRate(0.5);
  // ad.setDecayScale(0.5);
  run_test_functions( &ad, initpars );

  // BFGS isn't going to be useful for machine learning but let's compare it to other minimizers
  // it's possible that LBFGS can be used
  cout << "  ... BFGS ..." << endl;
  Bfgs bfgs(ndim);
  run_test_functions( &bfgs, initpars );
  
  return 0;
}
