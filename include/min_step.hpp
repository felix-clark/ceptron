// header file for minimizers
#include <functional>
#include <Eigen/Dense>

// it might be nice to use compile-time fixed-size arrays at some point. would necessitate template parameters.
using parvec = Eigen::ArrayXd;
// // these algorithms do not require the actual function, but we'll make this typedef for convenience in source files that include this
using func_t = std::function<double(parvec)>;
using grad_t = std::function<parvec(parvec)>;

// forward-declare classes for readability
class GradientDescent;
class GradientDescentWithMomentum;
class AcceleratedGradient;
class AdaDelta;

// in the future some may report a Hessian approximation
// this should be a separate interface
// a full Newton method would take a Hessian as well as a gradient (probably not worth implementing)

// an interface class for each of these that take a gradient function and a vector of parameters
class IMinStep
{
public:
  // do we need to initialize with the number of parameters? some steppers
  IMinStep(){}
  virtual ~IMinStep(){}
  // step_pars returns the next value of the parameter vector
  virtual parvec getDeltaPar( grad_t, parvec ) = 0;
  // remove any cached information, which some minimizers use to optimize the step rates
  virtual void resetCache() = 0;
};

class GradientDescent : public IMinStep
{
public:
  GradientDescent();
  GradientDescent(double);
  ~GradientDescent();
  virtual parvec getDeltaPar( grad_t, parvec ) override;
  virtual void resetCache() override {}; // nothing to do with gradient descent
  void setLearnRate(double lr) {learn_rate_ = lr;}
private:
  double learn_rate_ = 0.01;
};
  
class GradientDescentWithMomentum : public IMinStep
{
public:
  // GradientDescentWithMomentum();
  GradientDescentWithMomentum(size_t npar, double learn_rate, double momentum_scale);
  ~GradientDescentWithMomentum();
  virtual parvec getDeltaPar( grad_t, parvec ) override;
  virtual void resetCache() override {momentum_term_.setZero(momentum_term_.size());}
  void setLearnRate(double lr) {learn_rate_ = lr;}
  void setMomentumScale(double ms) {momentum_scale_ = ms;}
private:
  parvec momentum_term_;
  double learn_rate_ = 0.01;
  double momentum_scale_ = 0.875;
};

class AcceleratedGradient : public IMinStep
{
public:
  // AcceleratedGradient();
  AcceleratedGradient(size_t npar, double learn_rate, double momentum_scale);
  ~AcceleratedGradient();
  virtual parvec getDeltaPar( grad_t, parvec ) override;
  virtual void resetCache() override {momentum_term_.setZero(momentum_term_.size());}
  void setLearnRate(double lr) {learn_rate_ = lr;}
  void setMomentumScale(double ms) {momentum_scale_ = ms;}
private:
  parvec momentum_term_;
  double learn_rate_ = 0.01;
  double momentum_scale_ = 0.875;
};

class AdaDelta : public IMinStep
{
public:
  AdaDelta(size_t npar, double decay_scale, double epsilon, double learn_rate);
  ~AdaDelta();
  virtual parvec getDeltaPar( grad_t, parvec ) override;
  virtual void resetCache() override;
  void setDecayScale(double ds) {decay_scale_ = ds;}
  void setLearnRate(double lr) {learn_rate_ = lr;}
  void setEpsilon(double ep) {ep_ = ep;}
private:
  parvec accum_grad_sq_;
  parvec accum_dpar_sq_;
  bool have_last_par_ = false;
  parvec last_par_;
  double decay_scale_ = 0.9375; // similar to window average of last 16 values. 0.875 for scale of 8 previous values
  double learn_rate_ = 1.0; // a default value that can be adjusted down if necessary
  double ep_ = 1e-6;
};
