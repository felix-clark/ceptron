#include "min_step.hpp"
#include <utility> // for std::swap

using Eigen::sqrt;

// we don't actually want this in here, this is just for debugging
#include <iostream>

// ---- basic gradient descent ----

GradientDescent::GradientDescent(double lr)
  : learn_rate_(lr)
{
}

GradientDescent::~GradientDescent() {}

parvec GradientDescent::getDeltaPar( func_t, grad_t g, parvec pars ) {
  return - learn_rate_ * g(pars);
}

// ---- adding momentum term to naive gradient descent to help with "canyons" ----

GradientDescentWithMomentum::GradientDescentWithMomentum(size_t npar, double lr, double ms)
  : momentum_term_(parvec::Zero(npar))
  , learn_rate_(lr)
  , momentum_scale_(ms)
{
}

GradientDescentWithMomentum::~GradientDescentWithMomentum() {}

parvec GradientDescentWithMomentum::getDeltaPar( func_t, grad_t g, parvec pars ) {
  momentum_term_ = momentum_scale_ * momentum_term_ + learn_rate_ * g(pars);
  return - momentum_term_;
}

// ---- Nesterov's accelerated gradient ----
// an improved version of the momentum addition

AcceleratedGradient::AcceleratedGradient(size_t npar, double lr, double ms)
  : momentum_term_(parvec::Zero(npar))
  , learn_rate_(lr)
  , momentum_scale_(ms)
{
  momentum_term_.setZero(npar);
}

AcceleratedGradient::~AcceleratedGradient() {}

parvec AcceleratedGradient::getDeltaPar( func_t, grad_t g, parvec pars ) {
  momentum_term_ *= momentum_scale_; // exponentially reduce momentum term
  momentum_term_ += learn_rate_ * g(pars - momentum_term_);
  // momentum_term_ += learn_rate_ * g(pars - 0.5*momentum_term_); // using an average between new and old points can help with convergence on rosenbrock function
  return - momentum_term_;
}

// ---- ADADELTA has an adaptive, unitless learning rate ----
// it is probably often a good first choice because it tends to be insensitive to the hyperparameters

AdaDelta::AdaDelta(size_t npar, double decay_scale, double learn_rate, double epsilon)
  : accum_grad_sq_(parvec::Zero(npar))
  , accum_dpar_sq_(parvec::Zero(npar))
  , last_delta_par_(parvec::Zero(npar))
  , decay_scale_(decay_scale)
  , learn_rate_(learn_rate)
  , ep_(epsilon)
{
}

AdaDelta::~AdaDelta() {}

void AdaDelta::resetCache()
{
  accum_grad_sq_.setZero(accum_grad_sq_.size());
  accum_dpar_sq_.setZero(accum_dpar_sq_.size());
  last_delta_par_.setZero(last_delta_par_.size());
}

parvec AdaDelta::getDeltaPar( func_t, grad_t g, parvec pars )
{
  parvec grad = g(pars); // an improvement might be to use an accelerated version
  // element-wise learn rate
  parvec adj_learn_rate = sqrt((accum_dpar_sq_ + ep_)/(accum_grad_sq_ + ep_));

  // now we update for the next iteration
  accum_grad_sq_ *= decay_scale_;
  accum_grad_sq_ += (1.0-decay_scale_)*(grad.square());  
  accum_dpar_sq_ *= decay_scale_;
  accum_dpar_sq_ += (1.0-decay_scale_)*(last_delta_par_.square());

  
  // static int dbg = 0;
  // if (dbg < 1000 && dbg % 1 == 0) {
  //   std::cout << "lr: " << learn_rate_ << std::endl << "grad:";
  //   for (size_t i=0; i<pars.size(); ++i) {
  //     std::cout << grad[i] << ", ";
  //   }
  //   std::cout << std::endl << "last dp: ";
  //   for (size_t i=0; i<pars.size(); ++i) {
  //     std::cout << last_delta_par_[i] << ", ";
  //   }
  //   std::cout << std::endl << "adj lr: ";
  //   for (size_t i=0; i<pars.size(); ++i) {
  //     std::cout << adj_learn_rate[i] << ", ";
  //   }
  //   std::cout << std::endl;
  // }
  // dbg++;
  
  // this parameter can be saved directly for the next iteration
  last_delta_par_ = - learn_rate_ * adj_learn_rate * grad;
  return last_delta_par_;
  
  // return - learn_rate_ * sqrt((accum_dpar_sq_ + ep_)/(accum_grad_sq_ + ep_)) * grad;
}

// ---- BFGS ----
// a quasi-Newton method that uses an approximation for the Hessian, but uses a lot of memory to store it (npar^2)

Bfgs::Bfgs(size_t npar)
  : hessian_approx_(Eigen::MatrixXd::Identity(npar,npar))
  // , lastpar_(parvec::Zero(npar))
  // , lastgrad_(parvec::Zero(npar))
{
}

Bfgs::~Bfgs() {}

void Bfgs::resetCache()
{
  hessian_approx_ = Eigen::MatrixXd::Identity(hessian_approx_.rows(),
					      hessian_approx_.cols());
  // lastpar_ = parvec::Zero(lastpar_.size());
  // lastgrad_ = parvec::Zero(lastgrad_.size());
}

parvec Bfgs::getDeltaPar( func_t f, grad_t g, parvec par )
{
  Eigen::VectorXd grad = g(par).matrix();
  // the hessian approximation should remain positive definite. LLT requires positive-definiteness.
  // LLT actually yields NaN solutions at times so perhaps our hessian is not always pos-def.
  // Eigen::VectorXd deltap = - learn_rate_ * hessian_approx_.llt().solve(grad);
  Eigen::VectorXd deltap = - learn_rate_ * hessian_approx_.ldlt().solve(grad);
  // Eigen::VectorXd deltap = - learn_rate_ * hessian_approx_.householderQr().solve(grad); // no requirements on matrix for this

  // typically a line search is done to optimize the step size in the above direction. we'll just use a learning rate of 1 for now.
  // a line search might be necessary because Newton's method can diverge if the hessian is negative.
  // it might be good to check for a NaN or infinity in the solution

  // if ( ! deltap.array().isFinite().all() ) {
  //   std::cout << "solution is not finite" << std::endl;
  //   if ( ! grad.array().isFinite().all() ) {
  //     std::cout << "probably because the gradient is not finite either." << std::endl;
  //   }  else {
  //     std::cout << "gradient is finite though..." << std::endl;
  //   }
  //   if ( ! hessian_approx_.array().isFinite().all() ) {
  //     std::cout << "hessian itself is not finite!" << std::endl;
  //   }
  //   if ( ! par.array().isFinite().all() ) {
  //     std::cout << "and neither are the parameters" << std::endl;
  //   }
  // }
  
  auto f_line = [&](double x){return f(par + x*deltap.array());};
  double alpha_step = line_search( f_line, 0.7071067, 1.0 );

  // if (fabs(alpha_step) < 1e-2) {
  //   std::cout << "small step in line search: " << alpha_step << std::endl;
  //   for (double xl = -2.0e-2; xl < 0.1; xl += 1.0e-2) {
  //     std::cout << "f(" << xl << ") = " << f_line(xl) << std::endl;
  //   }
  // }
  
  Eigen::VectorXd deltagrad = (g(par+deltap.array()).matrix() - grad);

  // storing hessian by its square root could  possibly help numerical stability
  // might be good to check for divide-by-zero here, even if it's not likely to happen
  hessian_approx_ += (deltagrad * deltagrad.transpose())/deltagrad.dot(deltap)
    - hessian_approx_ * deltap * deltap.transpose() * hessian_approx_ / (deltap.transpose() * hessian_approx_ * deltap);
  
  return deltap.array();
}

// will perform a 1D simplex minimum search to quickly find a step size
double line_search( std::function<double(double)> f, double xa, double xb, size_t maxiter, double tol ) {
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
      // don't waste time looping
      break;
    }
    if ( (fa-fb)*(fa-fb) < tol*tol*(1.0 + (fa+fb)*(fa+fb)) ) {
      // tolerance reached
      // std::cout << "exiting line search due to tolerance reached" << std::endl;
      break;
    }
    
    if (fa > fb) {
      std::swap( xa, xb );
      std::swap( fa, fb );
    }
    if ( !( fa <= fb) ) {
      // we shouldn't get here -- possibly NaNs?
      std::cout << "f(xa) = " << fa << ", f(xb) = " << fb << std::endl;
      std::cout << "iteration number " << i_iter << std::endl;
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
