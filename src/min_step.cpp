#include "min_step.hpp"


GradientDescent::GradientDescent(double lr)
  : learn_rate_(lr)
{
}

GradientDescent::~GradientDescent() {}

parvec GradientDescent::stepPars( func_t f, grad_t g, parvec pars ) {
  return pars - learn_rate_ * g(pars);
}

GradientDescentWithMomentum::GradientDescentWithMomentum(size_t npar, double lr, double ms)
  : momentum_term_(npar)
  , learn_rate_(lr)
  , momentum_scale_(ms)
{
  momentum_term_.setZero(npar); // probably redundant, but we'll be safe.
}

GradientDescentWithMomentum::~GradientDescentWithMomentum() {}

parvec GradientDescentWithMomentum::stepPars( func_t f, grad_t g, parvec pars ) {
  momentum_term_ *= momentum_scale_; // exponentially reduce momentum term
  momentum_term_ = momentum_term_ + learn_rate_ * g(pars - momentum_term_);
  return pars - momentum_term_;
}
