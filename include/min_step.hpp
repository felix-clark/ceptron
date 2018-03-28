// header file for minimizers
#pragma once
#include "global.hpp"
#include <functional>
#include <Eigen/Dense>

// #include "training.hpp" // for IParStep interface. this might not be what we use permanently.

namespace {
  using ceptron::Array;
  using ceptron::Mat;
  using ceptron::func_grad_res;
  
} // namespace

namespace ceptron {
  template <int N=Eigen::Dynamic>
  using fg_t = std::function<func_grad_res<N>(Array<N>)>;
} // namespace ceptron

// these may not be used elsewhere yet... keep them local?
/// we will move away from this and towards a function returning both func and gradient result
template <int N=Eigen::Dynamic>
using func_t = std::function<double(Array<N>)>;
template <int N=Eigen::Dynamic>
using grad_t = std::function<Array<N>(Array<N>)>;


// we should also implement a golden section search
double line_search( std::function<double(double)>, double, double, size_t maxiter = 16, double tol=1e-6 );

// in the future some may report a Hessian approximation
// this should be a separate interface
// a full Newton method would take a Hessian as well as a gradient (probably not worth implementing)

// an interface class for each of these that take a gradient function and a vector of parameters
template <int N=Eigen::Dynamic>
class IMinStep
{
public:
  // do we need to initialize with the number of parameters?
  IMinStep(){}
  virtual ~IMinStep(){}
  // step_pars returns the next value of the parameter vector
  virtual Array<N> getDeltaPar( func_t<N>, grad_t<N>, Array<N> ) = 0;
  // virtual Array<N> getDeltaPar( const IParFunc<N>&, Array<N> ) = 0;
  // virtual Array<N> getDeltaPar( const Array<N>&, fg_t<N>, func_t<N> ) = 0;
  // remove any cached information, which some minimizers use to optimize the step rates
  virtual void resetCache() = 0;
};

// ---- basic gradient descent ----

template <int N>
class GradientDescent : public IMinStep<N>
{
public:
  GradientDescent() {}
  ~GradientDescent() {};
  virtual Array<N> getDeltaPar( func_t<N>, grad_t<N>, Array<N> ) override;
  // virtual Array<N> getDeltaPar( const Array<N>&, fg_t<N>, func_t<N> ) override;
  virtual void resetCache() override {}; // simple gradient descent uses no cache
  void setLearnRate(double lr) {learn_rate_ = lr;}
private:
  double learn_rate_ = 0.01;
};

template <int N>
Array<N> GradientDescent<N>::getDeltaPar( func_t<N>, grad_t<N> g, Array<N> pars ) {
  return - learn_rate_ * g(pars);
}

// template <int N>
// Array<N> GradientDescent<N>::getDeltaPar( const IParFunc<N>& funcgrad, Array<N> pars ) {
//   return - learn_rate_ * funcgrad.getFuncGrad(pars).g;
// }

// ---- adding momentum term to naive gradient descent to help with "canyons" ----

template <int N>
class GradientDescentWithMomentum : public IMinStep<N>
{
public:
  GradientDescentWithMomentum() {};
  ~GradientDescentWithMomentum() {};
  virtual Array<N> getDeltaPar( func_t<N>, grad_t<N>, Array<N> ) override;
  // virtual Array<N> getDeltaPar( const IParFunc<N>&, Array<N> ) override;
  virtual void resetCache() override {momentum_term_.setZero();}
  void setLearnRate(double lr) {learn_rate_ = lr;}
  void setMomentumScale(double ms) {momentum_scale_ = ms;}
private:
  Array<N> momentum_term_ = Array<N>::Zero();
  double learn_rate_ = 0.005;
  double momentum_scale_ = 0.875;
};

template <int N>
Array<N> GradientDescentWithMomentum<N>::getDeltaPar( func_t<N>, grad_t<N> g, Array<N> pars ) {
  momentum_term_ = momentum_scale_ * momentum_term_ + learn_rate_ * g(pars);
  return - momentum_term_;
}

// template <int N>
// Array<N> GradientDescentWithMomentum<N>::getDeltaPar( const IParFunc<N>& fg, Array<N> pars ) {
//   momentum_term_ = momentum_scale_ * momentum_term_ + learn_rate_ * fg.getFuncGrad(pars).g;
//   return - momentum_term_;  
// }

// // ---- Nesterov's accelerated gradient ----
// // an improved version of the momentum addition

template <int N>
class AcceleratedGradient : public IMinStep<N>
{
public:
  AcceleratedGradient() {};
  // AcceleratedGradient(double learn_rate=0.01,
  // 		      double momentum_scale=0.875);
  ~AcceleratedGradient() {};
  virtual Array<N> getDeltaPar( func_t<N>, grad_t<N>, Array<N> ) override;
  // virtual Array<N> getDeltaPar( const IParFunc<N>&, Array<N> ) override;
  virtual void resetCache() override {momentum_term_.setZero();}
  void setLearnRate(double lr) {learn_rate_ = lr;}
  void setMomentumScale(double ms) {momentum_scale_ = ms;}
private:
  Array<N> momentum_term_ = Array<N>::Zero();
  double learn_rate_ = 0.005;
  double momentum_scale_ = 0.875;
};

template <int N>
Array<N> AcceleratedGradient<N>::getDeltaPar( func_t<N>, grad_t<N> g, Array<N> pars ) {
  momentum_term_ *= momentum_scale_; // exponentially reduce momentum term
  momentum_term_ += learn_rate_ * g(pars - momentum_term_);
  // momentum_term_ += learn_rate_ * g(pars - 0.5*momentum_term_); // using an average between new and old points can help with convergence on rosenbrock function
  return - momentum_term_;
}

// template <int N>
// Array<N> AcceleratedGradient<N>::getDeltaPar( const IParFunc<N>& f, Array<N> pars ) {
//   momentum_term_ *= momentum_scale_; // exponentially reduce momentum term
//   momentum_term_ += learn_rate_ * f.getFuncGrad(pars - momentum_term_).g;
//   return - momentum_term_;
// }

// ---- ADADELTA has an adaptive, unitless learning rate ----
// it is probably often a good first choice because it tends to be insensitive to the hyperparameters

template <int N>
class AdaDelta : public IMinStep<N>
{
public:
  AdaDelta() {};
  ~AdaDelta() {};
  virtual Array<N> getDeltaPar( func_t<N>, grad_t<N>, Array<N> ) override;
  // virtual Array<N> getDeltaPar( const IParFunc<N>&, Array<N> ) override;
  virtual void resetCache() override;
  void setDecayScale(double ds) {decay_scale_ = ds;}
  void setLearnRate(double lr) {learn_rate_ = lr;}
  void setEpsilon(double ep) {ep_ = ep;}
private:
  Array<N> accum_grad_sq_ = Array<N>::Zero();
  Array<N> accum_dpar_sq_ = Array<N>::Zero();
  Array<N> last_delta_par_ = Array<N>::Zero();
  double decay_scale_ = 0.9375; // similar to window average of last 16 values. 0.875 for scale of 8 previous values
  double learn_rate_ = 1.0; // a default value that can be adjusted down if necessary
  double ep_ = 1e-6;
};

template <int N>
void AdaDelta<N>::resetCache()
{
  accum_grad_sq_.setZero();
  accum_dpar_sq_.setZero();
  last_delta_par_.setZero();
}

template <int N>
Array<N> AdaDelta<N>::getDeltaPar( func_t<N> /*f*/, grad_t<N> g, Array<N> pars )
{
  Array<N> grad = g(pars); // an improvement might be to use an accelerated version
  // element-wise learn rate
  Array<N> adj_learn_rate = sqrt((accum_dpar_sq_ + ep_)/(accum_grad_sq_ + ep_));

  // scaling down helps convergence but does seem to slow things down.
  // perhaps a line search (golden section) would be an improvement
  Array<N> dp = - learn_rate_ * adj_learn_rate * grad;
  // we can do an explicit check to keep from over-stepping

  // we should remove the line search in the general step
  // if ( f(pars+dp) > f(pars) ) {
  //   // a rapidly-truncated golden section search would probably be better since we can restrict 0 < alpha < 1
  //   // this actually ends up slowing down easy convergences so perhaps it's not the best approach
  //   // it prevents blowups but seems to negate most of the advantages of AdaDelta in the first place
  //   // practically, it might make sense to run a few iterations w/ line search then turn it off.
  //   auto f_line = [&](double x){return f(pars + x*dp.array());};
  //   double alpha_step = line_search( f_line, 0.7071067, 1.0 );

  //   dp *= alpha_step;
  //   grad *= alpha_step;
  // }
  
  // now we update for the next iteration
  accum_grad_sq_ *= decay_scale_;
  accum_grad_sq_ += (1.0-decay_scale_)*(grad.square());  
  accum_dpar_sq_ *= decay_scale_;
  accum_dpar_sq_ += (1.0-decay_scale_)*(last_delta_par_.square());

  last_delta_par_ = dp;

  // this parameter can be saved directly for the next iteration
  return dp;
  
}

// template <int N>
// Array<N> AdaDelta<N>::getDeltaPar( const IParFunc<N>& f, Array<N> pars ) {
//   Array<N> grad = f.getFuncGrad(pars).g; // an improvement might be to use an accelerated version w/ momentum
//   // element-wise learn rate
//   Array<N> adj_learn_rate = sqrt((accum_dpar_sq_ + ep_)/(accum_grad_sq_ + ep_));

//   // scaling down helps convergence but does seem to slow things down.
//   // perhaps a line search (golden section) would be an improvement
//   Array<N> dp = - learn_rate_ * adj_learn_rate * grad;
//   // we can do an explicit check to keep from over-stepping

//   // now we update for the next iteration
//   accum_grad_sq_ *= decay_scale_;
//   accum_grad_sq_ += (1.0-decay_scale_)*(grad.square());  
//   accum_dpar_sq_ *= decay_scale_;
//   accum_dpar_sq_ += (1.0-decay_scale_)*(last_delta_par_.square());

//   last_delta_par_ = dp;

//   // this parameter can be saved directly for the next iteration
//   return dp;
// }


// ---- BFGS ----
// a quasi-Newton method that uses an approximation for the Hessian, but uses a lot of memory to store it (N^2)

template <int N>
class Bfgs : public IMinStep<N>
{
public:
  Bfgs() {};
  ~Bfgs() {};
  virtual Array<N> getDeltaPar( func_t<N>, grad_t<N>, Array<N> ) override;
  // virtual Array<N> getDeltaPar( const IParFunc<N>&, Array<N> ) override;
  virtual void resetCache() override;
private:
  Mat<N, N> hessian_approx_ = decltype(hessian_approx_)::Identity();
  double learn_rate_ = 1.0;
};

template <int N>
void Bfgs<N>::resetCache()
{
  hessian_approx_.setIdentity();
}

template <int N>
Array<N> Bfgs<N>::getDeltaPar( func_t<N> f, grad_t<N> g, Array<N> par )
{
  Mat<N, 1> grad = g(par).matrix();
  // the hessian approximation should remain positive definite. LLT requires positive-definiteness.
  // LLT actually yields NaN solutions at times so perhaps our hessian is not always pos-def.
  // using a line search seems to have alleviated this issue.
  Mat<N, 1> deltap = - learn_rate_ * hessian_approx_.llt().solve(grad);
  // VectorXd deltap = - learn_rate_ * hessian_approx_.ldlt().solve(grad);
  // VectorXd deltap = - learn_rate_ * hessian_approx_.householderQr().solve(grad); // no requirements on matrix for this

  // we might only want to do this line search if we see an increasing value f(p+dp) > f(p)
  // , in which case a golden section search might be more efficient
  auto f_line = [&](double x){return f(par + x*deltap.array());};
  double alpha_step = line_search( f_line, 0.7071067, 1.0 );  

  deltap *= alpha_step;
  
  Mat<N, 1> deltagrad = (g(par+deltap.array()).matrix() - grad);

  // storing hessian by its square root could  possibly help numerical stability
  // might be good to check for divide-by-zero here, even if it's not likely to happen
  hessian_approx_ += (deltagrad * deltagrad.transpose())/deltagrad.dot(deltap)
    - hessian_approx_ * deltap * deltap.transpose() * hessian_approx_ / (deltap.transpose() * hessian_approx_ * deltap);
  
  return deltap.array();
}

// template <int N>
// Array<N> Bfgs<N>::getDeltaPar( const IParFunc<N>& fg, Array<N> par ) {
//   Mat<N, 1> grad = fg.getFuncGrad(par).g.matrix();
//   // the hessian approximation should remain positive definite. LLT requires positive-definiteness.
//   // LLT actually yields NaN solutions at times so perhaps our hessian is not always pos-def.
//   // using a line search seems to have alleviated this issue.
//   Mat<N, 1> deltap = - learn_rate_ * hessian_approx_.llt().solve(grad);
//   // VectorXd deltap = - learn_rate_ * hessian_approx_.ldlt().solve(grad);
//   // VectorXd deltap = - learn_rate_ * hessian_approx_.householderQr().solve(grad); // no requirements on matrix for this

//   // we might only want to do this line search if we see an increasing value f(p+dp) > f(p)
//   // , in which case a golden section search might be more efficient
//   auto f_line = [&](double x){return fg.getFuncOnly(par + x*deltap.array());};
//   double alpha_step = line_search( f_line, 0.7071067, 1.0 );  

//   deltap *= alpha_step;
  
//   // Mat<N, 1> deltagrad = (g(par+deltap.array()).matrix() - grad);
//   Mat<N, 1> deltagrad = (fg.getFuncGrad(par+deltap.array()).g.matrix() - grad);

//   // storing hessian by its square root could  possibly help numerical stability
//   // might be good to check for divide-by-zero here, even if it's not likely to happen
//   hessian_approx_ += (deltagrad * deltagrad.transpose())/deltagrad.dot(deltap)
//     - hessian_approx_ * deltap * deltap.transpose() * hessian_approx_ / (deltap.transpose() * hessian_approx_ * deltap);
  
//   return deltap.array();
// }
