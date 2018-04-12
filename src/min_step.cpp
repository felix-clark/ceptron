#include "min_step.hpp"
#include <utility>  // for std::swap
#include "log.hpp"

namespace {
using namespace ceptron;
}  // namespace

using Eigen::sqrt;

namespace ceptron {

// will perform a 1D simplex minimum search to quickly find a step size
// could possibly use instead a golden search on the interval (0,1)
scalar line_search(std::function<scalar(scalar)> f, scalar xa, scalar xb,
                   size_t maxiter, scalar tol) {
  constexpr scalar alpha = 1.0;
  constexpr scalar gamma = 2.0;
  constexpr scalar rho = 0.5;
  // constexpr scalar sigma = 0.5;
  constexpr scalar sigma = 0.25;  // if this is equal to rho then the 1D
                                  // algorithm has some redundant checks

  scalar fa = f(xa);
  scalar fb = f(xb);

  for (size_t i_iter = 0; i_iter < maxiter; ++i_iter) {
    // check condition for standard deviation of f values being small, either
    // absolutely or relatively depending on how large f is
    if (std::isnan(fa) || std::isnan(fb)) {
      // don't waste time looping if the values are nonsense
      break;
    }
    if ((fa - fb) * (fa - fb) < tol * tol * (1.0 + (fa + fb) * (fa + fb))) {
      // tolerance reached
      LOG_TRACE("exiting line search due to tolerance reached");
      break;
    }
    if (fa > fb) {
      std::swap(xa, xb);
      std::swap(fa, fb);
    }
    if (!(fa <= fb)) {
      // we shouldn't get here -- possibly NaNs?
      LOG_WARNING("f(xa) = " << fa << ", f(xb) = " << fb);
      LOG_WARNING("iteration number " << i_iter);
    }
    assert(!(fa > fb));                  // xa is now the best value
    scalar xr = xa + alpha * (xa - xb);  // reflected
    scalar fr = f(xr);
    if (fr < fa) {
      // reflected point is best so far, replace worst point with the either it
      // or this expanded point:
      scalar xe = xa + gamma * (xr - xa);  // = xa + gamma*alpha*(xa-xb)
      scalar fe = f(xe);
      if (fe < fr) {
        xb = xe;
        fb = fe;
      } else {
        xb = xr;
        fb = fr;
      }
      continue;  // go to next step of loop
    }
    assert(!(fr < fa));
    // contracted point (which is just average)
    scalar xc = xa + rho * (xb - xa);
    scalar fc = f(xc);
    // assert( fc == fc );
    if (fc < fb) {
      // replace worst point with contracted one
      xb = xc;
      fb = fc;
      continue;
    }
    // shrink
    // in 1D this is almost redundant with the contraction
    // will change value of sigma from wikipedia's recommended value to change
    // this
    // it's actually a problem if we get here: it means there's a local maximum
    // between xa and xb
    // and we do indeed get here...
    xb = xa + sigma * (xb - xa);
    fb = f(xb);
  }

  return fb < fa ? xb : xa;
}

ArrayX GradientDescent::getDeltaPar(func_t, grad_t g, ArrayX pars) {
  return -learn_rate_ * g(pars);
}

ArrayX GradientDescentWithMomentum::getDeltaPar(func_t, grad_t g, ArrayX pars) {
  momentum_term_ = momentum_scale_ * momentum_term_ + learn_rate_ * g(pars);
  return -momentum_term_;
}

ArrayX AcceleratedGradient::getDeltaPar(func_t, grad_t g, ArrayX pars) {
  momentum_term_ *= momentum_scale_;  // exponentially reduce momentum term
  momentum_term_ += learn_rate_ * g(pars - momentum_term_);
  // momentum_term_ += learn_rate_ * g(pars - 0.5*momentum_term_); // using an
  // average between new and old points can help with convergence on rosenbrock
  // function
  return -momentum_term_;
}

void AdaDelta::resetCache() {
  accum_grad_sq_.setZero();
  accum_dpar_sq_.setZero();
  last_delta_par_.setZero();
}

ArrayX AdaDelta::getDeltaPar(func_t /*f*/, grad_t g, ArrayX pars) {
  ArrayX grad =
      g(pars);  // an improvement might be to use an accelerated version
  // element-wise learn rate
  ArrayX adj_learn_rate = sqrt((accum_dpar_sq_ + ep_) / (accum_grad_sq_ + ep_));

  // scaling down helps convergence but does seem to slow things down.
  // perhaps a line search (golden section) would be an improvement
  ArrayX dp = -learn_rate_ * adj_learn_rate * grad;
  // we can do an explicit check to keep from over-stepping

  // we should remove the line search in the general step
  // if ( f(pars+dp) > f(pars) ) {
  //   // a rapidly-truncated golden section search would probably be better
  //   since we can restrict 0 < alpha < 1
  //   // this actually ends up slowing down easy convergences so perhaps it's
  //   not the best approach
  //   // it prevents blowups but seems to negate most of the advantages of
  //   AdaDelta in the first place
  //   // practically, it might make sense to run a few iterations w/ line
  //   search then turn it off.
  //   auto f_line = [&](scalar x){return f(pars + x*dp.array());};
  //   scalar alpha_step = line_search( f_line, 0.7071067, 1.0 );

  //   dp *= alpha_step;
  //   grad *= alpha_step;
  // }

  // now we update for the next iteration
  accum_grad_sq_ *= decay_scale_;
  accum_grad_sq_ += (1.0 - decay_scale_) * (grad.square());
  accum_dpar_sq_ *= decay_scale_;
  accum_dpar_sq_ += (1.0 - decay_scale_) * (last_delta_par_.square());

  last_delta_par_ = dp;

  // this parameter can be saved directly for the next iteration
  return dp;
}

void Bfgs::resetCache() { hessian_approx_.setIdentity(); }

ArrayX Bfgs::getDeltaPar(func_t f, grad_t g, ArrayX par) {
  VecX grad = g(par).matrix();
  // the hessian approximation should remain positive definite. LLT requires
  // positive-definiteness.
  // LLT actually yields NaN solutions at times so perhaps our hessian is not
  // always pos-def.
  // using a line search seems to have alleviated this issue.
  VecX deltap = -learn_rate_ * hessian_approx_.llt().solve(grad);
  // VectorXd deltap = - learn_rate_ * hessian_approx_.ldlt().solve(grad);
  // VectorXd deltap = - learn_rate_ *
  // hessian_approx_.householderQr().solve(grad); // no requirements on matrix
  // for this

  // we might only want to do this line search if we see an increasing value
  // f(p+dp) > f(p)
  // , in which case a golden section search might be more efficient
  auto f_line = [&](scalar x) { return f(par + x * deltap.array()); };
  scalar alpha_step = line_search(f_line, 0.7071067, 1.0);

  deltap *= alpha_step;

  VecX deltagrad = (g(par + deltap.array()).matrix() - grad);

  // storing hessian by its square root could  possibly help numerical stability
  // might be good to check for divide-by-zero here, even if it's not likely to
  // happen
  hessian_approx_ +=
      (deltagrad * deltagrad.transpose()) / deltagrad.dot(deltap) -
      hessian_approx_ * deltap * deltap.transpose() * hessian_approx_ /
          (deltap.transpose() * hessian_approx_ * deltap);

  return deltap.array();
}

}  // namespace ceptron
