// header file for minimizers
#pragma once
#include "global.hpp"
#include <functional>
#include <Eigen/Dense>

namespace ceptron {
  using func_t = std::function<scalar(ArrayX)>;
  using grad_t = std::function<ArrayX(ArrayX)>;

  // we should also implement a golden section search
  scalar line_search( std::function<scalar(scalar)>, scalar, scalar, size_t maxiter = 16, scalar tol=1e-6 );

  // in the future some may report a Hessian approximation
  // this should be a separate interface
  // a full Newton method would take a Hessian as well as a gradient (probably not worth implementing)

  // an interface class for each of these that take a gradient function and a vector of parameters
  // template <int N=Eigen::Dynamic>
  class IMinStep
  {
  public:
    // do we need to initialize with the number of parameters?
    IMinStep(){}
    virtual ~IMinStep(){}
    // step_pars returns the next value of the parameter vector
    virtual ArrayX getDeltaPar( func_t, grad_t, ArrayX ) = 0;
    // virtual ArrayX getDeltaPar( const ArrayX&, fg_t, func_t ) = 0;
    // remove any cached information, which some minimizers use to optimize the step rates
    virtual void resetCache() = 0;
  };

  // ---- basic gradient descent ----

  class GradientDescent : public IMinStep
  {
  public:
    GradientDescent() = default;
    ~GradientDescent() = default;
    virtual ArrayX getDeltaPar( func_t, grad_t, ArrayX ) override;
    virtual void resetCache() override {}; // simple gradient descent uses no cache
    void setLearnRate(scalar lr) {learn_rate_ = lr;}
  private:
    scalar learn_rate_ = 0.01;
  };

  // ---- adding momentum term to naive gradient descent to help with "canyons" ----

  class GradientDescentWithMomentum : public IMinStep
  {
  public:
    GradientDescentWithMomentum(int npar)
      : momentum_term_(ArrayX::Zero(npar)) {};
    ~GradientDescentWithMomentum() = default;
    virtual ArrayX getDeltaPar( func_t, grad_t, ArrayX ) override;
    virtual void resetCache() override {momentum_term_.setZero();}
    void setLearnRate(scalar lr) {learn_rate_ = lr;}
    void setMomentumScale(scalar ms) {momentum_scale_ = ms;}
  private:
    ArrayX momentum_term_;// = ArrayX::Zero();
    scalar learn_rate_ = 0.005;
    scalar momentum_scale_ = 0.875;
  };

  // // ---- Nesterov's accelerated gradient ----
  // // an improved version of the momentum addition

  class AcceleratedGradient : public IMinStep
  {
  public:
    AcceleratedGradient(int npar)
      : momentum_term_(ArrayX::Zero(npar)) {};
    // AcceleratedGradient(scalar learn_rate=0.01,
    // 		      scalar momentum_scale=0.875);
    ~AcceleratedGradient() = default;
    virtual ArrayX getDeltaPar( func_t, grad_t, ArrayX ) override;
    virtual void resetCache() override {momentum_term_.setZero();}
    void setLearnRate(scalar lr) {learn_rate_ = lr;}
    void setMomentumScale(scalar ms) {momentum_scale_ = ms;}
  private:
    ArrayX momentum_term_;
    scalar learn_rate_ = 0.005;
    scalar momentum_scale_ = 0.875;
  };

  // ---- ADADELTA has an adaptive, unitless learning rate ----
  // it is probably often a good first choice because it tends to be insensitive to the hyperparameters

  class AdaDelta : public IMinStep
  {
  public:
    AdaDelta(int npar)
      : accum_grad_sq_(ArrayX::Zero(npar))
      , accum_dpar_sq_(ArrayX::Zero(npar))
      , last_delta_par_(ArrayX::Zero(npar)) {};
    ~AdaDelta() = default;
    virtual ArrayX getDeltaPar( func_t, grad_t, ArrayX ) override;
    virtual void resetCache() override;
    void setDecayScale(scalar ds) {decay_scale_ = ds;}
    void setLearnRate(scalar lr) {learn_rate_ = lr;}
    void setEpsilon(scalar ep) {ep_ = ep;}
  private:
    ArrayX accum_grad_sq_;
    ArrayX accum_dpar_sq_;
    ArrayX last_delta_par_;
    scalar decay_scale_ = 0.9375; // similar to window average of last 16 values. 0.875 for scale of 8 previous values
    scalar learn_rate_ = 1.0; // a default value that can be adjusted down if necessary
    scalar ep_ = 1e-6;
  };


  // ---- BFGS ----
  // a quasi-Newton method that uses an approximation for the Hessian, but uses a lot of memory to store it (N^2)

  class Bfgs : public IMinStep
  {
  public:
    Bfgs(int npar)
      : hessian_approx_(MatX::Identity(npar,npar)) {};
    ~Bfgs() {};
    virtual ArrayX getDeltaPar( func_t, grad_t, ArrayX ) override;
    virtual void resetCache() override;
  private:
    MatX hessian_approx_;
    scalar learn_rate_ = 1.0;
  };


} // namespace ceptron
