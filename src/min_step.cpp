#include "min_step.hpp"

using Eigen::sqrt;

// // we don't actually want this in here, this is just for debugging
// #include <iostream>

// ---- basic gradient descent ----

GradientDescent::GradientDescent(double lr)
  : learn_rate_(lr)
{
}

GradientDescent::~GradientDescent() {}

parvec GradientDescent::getDeltaPar( grad_t g, parvec pars ) {
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

parvec GradientDescentWithMomentum::getDeltaPar( grad_t g, parvec pars ) {
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

parvec AcceleratedGradient::getDeltaPar( grad_t g, parvec pars ) {
  momentum_term_ *= momentum_scale_; // exponentially reduce momentum term
  momentum_term_ += learn_rate_ * g(pars - momentum_term_);
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

parvec AdaDelta::getDeltaPar( grad_t g, parvec pars )
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

parvec Bfgs::getDeltaPar( grad_t g, parvec par )
{
  Eigen::VectorXd grad = g(par).matrix();
  // the hessian approximation should remain positive definite. LLT requires positive-definiteness.
  Eigen::VectorXd deltap = - learn_rate_ * hessian_approx_.llt().solve(grad);
  // typically a line search is done to optimize the step size in the above direction. we'll just use a learning rate of 1 for now.
  // it might be good to check for a NaN or infinity in the solution
  
  Eigen::VectorXd deltagrad = (g(par+deltap.array()).matrix() - grad);

  hessian_approx_ += (deltagrad * deltagrad.transpose())/deltagrad.dot(deltap)
    - hessian_approx_ * deltap * deltap.transpose() * hessian_approx_ / (deltap.transpose() * hessian_approx_ * deltap);
  
  return deltap.array();
}
