// header file for minimizers
#include <functional>
#include <Eigen/Dense>

// it might be nice to use compile-time fixed-size arrays at some point. would necessitate template parameters.
using parvec = Eigen::ArrayXd;
using func_t = std::function<double(parvec)>;
using grad_t = std::function<parvec(parvec)>;

// need to figure out how best to organize this.
// we want each minimizer to only go one step at a time, but some of them will cache some values.

// some may report a Hessian approximation
// a full Newton method would take a Hessian as well as a gradient (probably not worth implementing)

class IMinStep
{
public:
  // do we need to initialize with the number of parameters? some steppers
  IMinStep(){}
  virtual ~IMinStep(){}
  // step_pars returns the next value of the parameter vector
  virtual parvec stepPars( func_t, grad_t, parvec ) = 0;
  virtual void reset() = 0;
};

class GradientDescent : public IMinStep
{
public:
  GradientDescent();
  GradientDescent(double);
  ~GradientDescent();
  virtual parvec stepPars( func_t, grad_t, parvec ) override;
  virtual void reset() override {}; // nothing to do with gradient descent
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
  virtual parvec stepPars( func_t, grad_t, parvec ) override;
  virtual void reset() override {momentum_term_.setZero(momentum_term_.size());}
  void setLearnRate(double lr) {learn_rate_ = lr;}
  void setMomentumScale(double ms) {momentum_scale_ = ms;}
private:
  parvec momentum_term_;
  double learn_rate_ = 0.01;
  double momentum_scale_ = 0.875;
};
