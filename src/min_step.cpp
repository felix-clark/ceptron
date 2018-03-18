#include "min_step.hpp"

using Eigen::sqrt;


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
  : momentum_term_(npar)
  , learn_rate_(lr)
  , momentum_scale_(ms)
{
  momentum_term_.setZero(npar); // probably redundant, but we'll be safe.
}

GradientDescentWithMomentum::~GradientDescentWithMomentum() {}

parvec GradientDescentWithMomentum::getDeltaPar( grad_t g, parvec pars ) {
  momentum_term_ = momentum_scale_ * momentum_term_ + learn_rate_ * g(pars);
  return - momentum_term_;
}

// ---- Nesterov's accelerated gradient ----
// an improved version of the momentum addition

AcceleratedGradient::AcceleratedGradient(size_t npar, double lr, double ms)
  : momentum_term_(npar)
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
  : accum_grad_sq_(npar)
  , accum_dpar_sq_(npar)
  , last_par_(npar)
  , decay_scale_(decay_scale)
  , learn_rate_(learn_rate)
  , ep_(epsilon)
{
}

AdaDelta::~AdaDelta() {}

void AdaDelta::resetCache()
{
  accum_grad_sq_.setZeros(accum_grad_sq_.size());
  accum_dpar_sq_.setZeros(accum_dpar_sq_.size());
  have_last_par_ = false;
  last_par_.setZeros(last_par_.size());
}

parvec AdaDelta::getDeltaPar( grad_t g, parvec pars )
{
  grad = g(pars); // an improvement might be to use an accelerated version
  accum_grad_sq_ *= decay_scale_;
  accum_grad_sq_ += (1.0-decay_scale_)*g.square();  
  if (have_last_par_) {
    accum_dpar_sq_ *= decay_scale_;
    accum_dpar_sq_ += (1.0-decay_scale_)*(pars-last_par_).square();
  }
  have_last_par_ = true;
  last_par_ = pars; // cache these values for the next step
  return - learn_rate_ * sqrt((accum_dpar_sq_ + ep_)/(accum_grad_sq + ep_)) * grad;
}
