#include <iostream>
#include <limits>
#include "global.hpp"
#include "log.hpp"
#include "min_step.hpp"
#include "test_functions.hpp"

namespace {
using namespace ceptron;
// a local formatting option block for convenience
const Eigen::IOFormat my_fmt(3,  // first value is the precision
                             0, ", ", "\n", "[", "]");
constexpr scalar scalar_ep = std::numeric_limits<ceptron::scalar>::epsilon();
const scalar local_epsilon = std::sqrt(scalar_ep);
const scalar default_tol = std::sqrt(scalar_ep);

}  // namespace

void check_gradient(func_t f, grad_t g, ArrayX p, double ep = local_epsilon,
                    double tol = default_tol) {
  const int Npar = p.size();
  // // TODO: borrow element-by-element version from test_net to beef this up
  // ArrayX evalgrad = g(p);
  // double gradmag = sqrt(evalgrad.square().sum());
  // ArrayX dir = (1.0+p.square()).sqrt()*evalgrad/gradmag; // vector along
  // direction of greatest change
  // double dirmag = sqrt(dir.square().sum());
  // // we should actually check the numerical derivative element-by-element
  // double deltaf = (f(p+ep*dir) - f(p-ep*dir))/(2*ep*dirmag); // should be
  // close to |gradf|
  // if ( fabs(deltaf) - (dir*evalgrad).sum()/dirmag > tol*fabs(deltaf +
  // (dir*evalgrad).sum()/dirmag) ) {
  //   LOG_ERROR("gradient check failed!");
  //   LOG_ERROR("f(p) = " << f(p));
  //   LOG_ERROR("numerical derivative = " << deltaf);
  //   LOG_ERROR("analytic gradient magnitude = " <<
  //   (dir*evalgrad).sum()/dirmag);
  //   LOG_ERROR("difference = " << fabs(deltaf) - (dir*evalgrad).sum()/dirmag);
  // }

  // ---- element-by-element
  scalar fval = f(p);
  ArrayX df(Npar);  // maybe can do this assignment slickly with colwise() or
                    // rowwise() ?
  for (int i_f = 0; i_f < Npar; ++i_f) {
    ArrayX dp = ArrayX::Zero(Npar);
    scalar dpi = ep * sqrt(1.0 + p(i_f) * p(i_f));
    dp(i_f) = dpi;
    // scalar fplusi = net.costFunc(p+dp, xin, yin);
    // scalar fminusi = net.costFunc(p-dp, xin, yin);
    scalar fplusi = f(p + dp);
    scalar fminusi = f(p - dp);
    df(i_f) = (fplusi - fminusi) / (2 * dpi);
  }

  LOG_TRACE("about to check with analytic gradient");
  ArrayX gradval = g(p);
  // now check this element-by element
  scalar sqSumDiff = (df - gradval).square().sum();
  if (sqSumDiff >
          tol * tol * (1.0 + df.square().sum() + gradval.square().sum()) ||
      !df.isFinite().all() || !gradval.isFinite().all()) {
    LOG_WARNING("gradient check failed at tolerance level "
                << tol << " (" << sqrt(sqSumDiff) << ")");
    LOG_WARNING("f(p) = " << fval);
    LOG_WARNING("numerical derivative = " << df.transpose().format(my_fmt));
    LOG_WARNING("analytic gradient = " << gradval.transpose().format(my_fmt));
    LOG_WARNING("difference = " << (df - gradval).transpose().format(my_fmt));
  }
}

struct min_result {
  ArrayX x;
  double f;
  ArrayX grad;
  size_t iterations = 0;
  int fail = 0;
};

min_result test_min_step(IMinStep& minstep, func_t f, grad_t g, ArrayX initpars,
                         double abstol = default_tol, size_t maxsteps = 64000) {
  minstep.resetCache();  // need to make sure caches are reset!

  ArrayX pars = initpars;
  size_t n_iterations = 0;
  bool finished = false;
  while (!finished) {
    check_gradient(f, g, pars);
    ArrayX dpars = minstep.getDeltaPar(f, g, pars);
    if (isnan(dpars).any()) {
      LOG_ERROR("NaN in proposed delta pars!");

      return {pars, f(pars), g(pars), n_iterations, 1};
    }

    pars += dpars;
    ++n_iterations;
    // just check the gradient magnitude for now
    // ArrayXd does not have a squaredNorm() function like VectorXd
    if (g(pars).square().sum() < abstol * abstol || n_iterations >= maxsteps) {
      finished = true;
    }
  }
  return {pars, f(pars), g(pars), n_iterations};
}

// template <size_t N>
void print_pars(ArrayX pars) {
  for (int i_par = 0; i_par < pars.size(); ++i_par) {
    LOG_INFO(pars[i_par] << ", ");
  }
  LOG_INFO(" ");
}

// template <size_t N>
void print_result(min_result res) {
  LOG_INFO(" ");
  if (res.fail) {
    LOG_ERROR("fit did not converge!");
  }
  double f = res.f;
  LOG_INFO("f(x) = " << f);
  LOG_INFO("took " << res.iterations << " steps.");
  LOG_INFO("gradient:\t");
  print_pars(res.grad);
  LOG_INFO("x:\t");
  print_pars(res.x);
  LOG_INFO(" ");
}

template <size_t N>
void run_test_functions(IMinStep& minstep, Array<N> initpars) {
  using namespace mintest;
  LOG_INFO("testing basic ellipse function");
  // basic gradient descent has trouble with a large scale in ellipse function
  // use lambda function to automatically use default scale parameter
  // auto f_ellipse = [](Array<N> p){return ellipse<N>(p);};
  // std::function<double(Array<N>)> f_ellipse = [](Array<N> p){return
  // ellipse<N>(p);};
  func_t f_ellipse = [](Array<N> p) { return ellipse<N>(p); };
  // auto g_ellipse = [](Array<N> p){return grad_ellipse<N>(p);};
  // std::function<Array<N>(Array<N>)> g_ellipse = [](Array<N> p){return
  // grad_ellipse<N>(p);};
  grad_t g_ellipse = [](Array<N> p) { return grad_ellipse<N>(p); };
  min_result gd_result = test_min_step(minstep, f_ellipse, g_ellipse, initpars);
  LOG_INFO("final parameters for ellipse: ");
  print_result(gd_result);

  LOG_INFO("testing rosenbrock function");
  auto f_rosen = [](Array<N> p) { return rosenbrock<N>(p); };
  auto g_rosen = [](Array<N> p) { return grad_rosenbrock<N>(p); };
  min_result rosen_result = test_min_step(minstep, f_rosen, g_rosen, initpars);
  LOG_INFO("final parameters for rosenbrock: ");
  print_result(rosen_result);
}

int main(int, char**) {
  SET_LOG_LEVEL(debug);

  // constexpr size_t ndim = 4;
  constexpr size_t ndim = 2;
  Array<ndim> initpars;
  initpars = 2 * Array<ndim>::Random();
  LOG_INFO("initial parameters:");
  print_pars(initpars);

  GradientDescent gd;
  gd.setLearnRate(0.001);
  LOG_INFO("  ... gradient descent ...");
  run_test_functions<ndim>(gd, initpars);

  GradientDescentWithMomentum gdm(ndim);
  gdm.setLearnRate(0.001);  // default learn rate is too large for rosenbrock
  LOG_INFO("  ... MOM ...");
  run_test_functions<ndim>(gdm, initpars);

  AcceleratedGradient ag(ndim);
  // ag.setMomentumScale(0.5);
  ag.setLearnRate(0.001);  // we need to turn down the learn rate significantly
                           // or it diverges on rosenbrock
  LOG_INFO("  ... NAG ...");
  run_test_functions<ndim>(ag, initpars);

  // ADADELTA converges in much fewer steps for basic sphere.
  // it seems to diverge for rosenbrock function right now...
  LOG_INFO("  ... ADADELTA ...");
  AdaDelta ad(ndim);
  // ad.setLearnRate(0.5);
  // ad.setDecayScale(0.5);
  run_test_functions<ndim>(ad, initpars);

  // BFGS isn't going to be useful for machine learning but let's compare it to
  // other minimizers
  // it's possible that LBFGS can be used, but may need some adaptation for
  // stochastic/mini-batch minimization
  LOG_INFO("  ... BFGS ...");
  Bfgs bfgs(ndim);
  run_test_functions<ndim>(bfgs, initpars);

  return 0;
}
