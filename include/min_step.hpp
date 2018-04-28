// header file for minimizers
#pragma once
#include <Eigen/Dense>
#include <functional>
#include "global.hpp"

namespace ceptron {
using func_t = std::function<scalar(ArrayX)>;
using grad_t = std::function<ArrayX(ArrayX)>;

// we should also implement a golden section search
scalar line_search(std::function<scalar(scalar)>, scalar, scalar,
                   size_t maxiter = 16, scalar tol = 1e-6);

// in the future some may report a Hessian approximation
// this should be a separate interface
// a full Newton method would take a Hessian as well as a gradient (probably not
// worth implementing)

// an interface class for each of these that take a gradient function and a
// vector of parameters
// template <int N=Eigen::Dynamic>
class IMinStep {
 public:
  // do we need to initialize with the number of parameters?
  IMinStep() {}
  virtual ~IMinStep() {}
  // step_pars returns the next value of the parameter vector
  virtual ArrayX getDeltaPar(func_t, grad_t, ArrayX) = 0;
  // virtual ArrayX getDeltaPar( const ArrayX&, fg_t, func_t ) = 0;
  // remove any cached information, which some minimizers use to optimize the
  // step rates
  virtual void resetCache() = 0;
};

// ---- basic gradient descent ----

class GradientDescent : public IMinStep {
 public:
  GradientDescent() = default;
  ~GradientDescent() = default;
  virtual ArrayX getDeltaPar(func_t, grad_t, ArrayX) override;
  virtual void resetCache()
      override{};  // simple gradient descent uses no cache
  void setLearnRate(scalar lr) { learn_rate_ = lr; }

 private:
  scalar learn_rate_ = 0.01;
};

// ---- adding momentum term to naive gradient descent to help with "canyons"
// ----

class GradientDescentWithMomentum : public IMinStep {
 public:
  explicit GradientDescentWithMomentum(int npar) : momentum_term_(ArrayX::Zero(npar)){};
  ~GradientDescentWithMomentum() = default;
  virtual ArrayX getDeltaPar(func_t, grad_t, ArrayX) override;
  virtual void resetCache() override { momentum_term_.setZero(); }
  void setLearnRate(scalar lr) { learn_rate_ = lr; }
  void setMomentumScale(scalar ms) { momentum_scale_ = ms; }

 private:
  scalar learn_rate_ = 0.005;
  scalar momentum_scale_ = 0.875;
  ArrayX momentum_term_;  // = ArrayX::Zero();
};

// // ---- Nesterov's accelerated gradient ----
// // an improved version of the momentum addition

class AcceleratedGradient : public IMinStep {
 public:
  explicit AcceleratedGradient(int npar) : momentum_term_(ArrayX::Zero(npar)){};
  // AcceleratedGradient(scalar learn_rate=0.01,
  // 		      scalar momentum_scale=0.875);
  ~AcceleratedGradient() = default;
  virtual ArrayX getDeltaPar(func_t, grad_t, ArrayX) override;
  virtual void resetCache() override { momentum_term_.setZero(); }
  void setLearnRate(scalar lr) { learn_rate_ = lr; }
  void setMomentumScale(scalar ms) { momentum_scale_ = ms; }

 private:
  scalar learn_rate_ = 0.005;
  scalar momentum_scale_ = 0.875;
  ArrayX momentum_term_;
};

// ---- ADADELTA has an adaptive, unitless learning rate ----
// it is probably often a good first choice because it tends to be insensitive
// to the hyperparameters

class AdaDelta : public IMinStep {
 public:
  explicit AdaDelta(int npar)
      : accum_grad_sq_(ArrayX::Zero(npar)),
        accum_dpar_sq_(ArrayX::Zero(npar)),
        last_delta_par_(ArrayX::Zero(npar)){};
  ~AdaDelta() = default;
  virtual ArrayX getDeltaPar(func_t, grad_t, ArrayX) override;
  virtual void resetCache() override;
  void setDecayScale(scalar ds) { decay_scale_ = ds; }
  void setLearnRate(scalar lr) { learn_rate_ = lr; }
  void setEpsilon(scalar ep) { ep_ = ep; }

 private:
  scalar decay_scale_ = 0.9375;  // similar to window average of last 16 values.
                                 // 0.875 for scale of 8 previous values
  // the learn rate can be adjusted down if necessary
  scalar learn_rate_ = 1.0;
  scalar ep_ = 1e-6;
  ArrayX accum_grad_sq_;
  ArrayX accum_dpar_sq_;
  ArrayX last_delta_par_;
};

// ---- Adam uses adaptive 1st and 2nd gradient moments ----
// it 

class Adam : public IMinStep {
 public:
  explicit Adam(int npar)
      : accum_m_(ArrayX::Zero(npar)),
        accum_v_(ArrayX::Zero(npar)) {};
  ~Adam() = default;
  virtual ArrayX getDeltaPar(func_t, grad_t, ArrayX) override;
  virtual void resetCache() override;
  void setLearnRate(scalar lr) { learn_rate_ = lr; }
  void setBeta1(scalar b) { beta1_ = b; }
  void setBeta2(scalar b) { beta2_ = b; }
  void setEpsilon(scalar ep) { ep_ = ep; }

 private:
  // hyperparameters
  scalar learn_rate_ = 1.0/(1 << 10); // ~ 0.000977
  scalar beta1_ = 0.9375;
  scalar beta2_ = 1.0 - 1.0/(1 << 10);
  scalar ep_ = 1e-8;
  // saved quantities
  // exponential average of gradient mean moment
  ArrayX accum_m_;
  // exponential average of gradient variance moment
  ArrayX accum_v_;
  // beta parameters to the (time) power, used for bias-correction
  scalar beta1t_ = 1;
  scalar beta2t_ = 1;
};

// ---- BFGS ----
// a quasi-Newton method that uses an approximation for the Hessian, but uses a
// lot of memory to store it (N^2)

class Bfgs : public IMinStep {
 public:
  explicit Bfgs(int npar) : hessian_approx_(MatX::Identity(npar, npar)){};
  ~Bfgs(){};
  virtual ArrayX getDeltaPar(func_t, grad_t, ArrayX) override;
  virtual void resetCache() override;

 private:
  MatX hessian_approx_;
  scalar learn_rate_ = 1.0;
};

}  // namespace ceptron
