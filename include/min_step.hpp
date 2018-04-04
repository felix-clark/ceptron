// header file for minimizers
#pragma once
#include "global.hpp"
#include <functional>
#include <Eigen/Dense>

namespace ceptron {
  // template <int N=Eigen::Dynamic>
  using fg_t = std::function<func_grad_res<>(ArrayX)>;

  // these may not be used elsewhere yet... keep them local?
  /// we will move away from this and towards a function returning both func and gradient result
  // template <int N=Eigen::Dynamic>
  using func_t = std::function<double(ArrayX)>;
  // template <int N=Eigen::Dynamic>
  using grad_t = std::function<ArrayX(ArrayX)>;


  // we should also implement a golden section search
  double line_search( std::function<double(double)>, double, double, size_t maxiter = 16, double tol=1e-6 );

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
    // virtual ArrayX getDeltaPar( const ArrayX&, fg_t/*<N>*/, func_t ) = 0;
    // remove any cached information, which some minimizers use to optimize the step rates
    virtual void resetCache() = 0;
  };

  // ---- basic gradient descent ----

  // template <int N>
  class GradientDescent : public IMinStep/*<N>*/
  {
  public:
    GradientDescent(/*int npar*/) {}
    ~GradientDescent() {};
    virtual ArrayX getDeltaPar( func_t, grad_t, ArrayX ) override;
    // virtual ArrayX getDeltaPar( const ArrayX&, fg_t/*<N>*/, func_t ) override;
    virtual void resetCache() override {}; // simple gradient descent uses no cache
    void setLearnRate(double lr) {learn_rate_ = lr;}
  private:
    double learn_rate_ = 0.01;
  };

  // ---- adding momentum term to naive gradient descent to help with "canyons" ----

  // template <int N>
  class GradientDescentWithMomentum : public IMinStep/*<N>*/
  {
  public:
    GradientDescentWithMomentum(int npar)
      : momentum_term_(ArrayX::Zero(npar)) {};
    ~GradientDescentWithMomentum() {};
    virtual ArrayX getDeltaPar( func_t, grad_t, ArrayX ) override;
    virtual void resetCache() override {momentum_term_.setZero();}
    void setLearnRate(double lr) {learn_rate_ = lr;}
    void setMomentumScale(double ms) {momentum_scale_ = ms;}
  private:
    ArrayX momentum_term_;// = ArrayX::Zero();
    double learn_rate_ = 0.005;
    double momentum_scale_ = 0.875;
  };

  // // ---- Nesterov's accelerated gradient ----
  // // an improved version of the momentum addition

  // template <int N>
  class AcceleratedGradient : public IMinStep/*<N>*/
  {
  public:
    AcceleratedGradient(int npar)
      : momentum_term_(ArrayX::Zero(npar)) {};
    // AcceleratedGradient(double learn_rate=0.01,
    // 		      double momentum_scale=0.875);
    ~AcceleratedGradient() {};
    virtual ArrayX getDeltaPar( func_t, grad_t, ArrayX ) override;
    virtual void resetCache() override {momentum_term_.setZero();}
    void setLearnRate(double lr) {learn_rate_ = lr;}
    void setMomentumScale(double ms) {momentum_scale_ = ms;}
  private:
    ArrayX momentum_term_;// = ArrayX::Zero();
    double learn_rate_ = 0.005;
    double momentum_scale_ = 0.875;
  };

  // ---- ADADELTA has an adaptive, unitless learning rate ----
  // it is probably often a good first choice because it tends to be insensitive to the hyperparameters

  // template <int N>
  class AdaDelta : public IMinStep/*<N>*/
  {
  public:
    AdaDelta(int npar) {
      accum_grad_sq_ = ArrayX::Zero(npar);
      accum_dpar_sq_ = ArrayX::Zero(npar);
      last_delta_par_ = ArrayX::Zero(npar);
    };
    ~AdaDelta() {};
    virtual ArrayX getDeltaPar( func_t, grad_t, ArrayX ) override;
    virtual void resetCache() override;
    void setDecayScale(double ds) {decay_scale_ = ds;}
    void setLearnRate(double lr) {learn_rate_ = lr;}
    void setEpsilon(double ep) {ep_ = ep;}
  private:
    ArrayX accum_grad_sq_;// = ArrayX::Zero();
    ArrayX accum_dpar_sq_;// = ArrayX::Zero();
    ArrayX last_delta_par_;// = ArrayX::Zero();
    double decay_scale_ = 0.9375; // similar to window average of last 16 values. 0.875 for scale of 8 previous values
    double learn_rate_ = 1.0; // a default value that can be adjusted down if necessary
    double ep_ = 1e-6;
  };


  // ---- BFGS ----
  // a quasi-Newton method that uses an approximation for the Hessian, but uses a lot of memory to store it (N^2)

  // template <int N>
  class Bfgs : public IMinStep/*<N>*/
  {
  public:
    Bfgs(int npar) {hessian_approx_=MatX::Identity(npar,npar);};
    ~Bfgs() {};
    virtual ArrayX getDeltaPar( func_t, grad_t, ArrayX ) override;
    virtual void resetCache() override;
  private:
    MatX hessian_approx_;// = decltype(hessian_approx_)::Identity();
    double learn_rate_ = 1.0;
  };


} // namespace ceptron
