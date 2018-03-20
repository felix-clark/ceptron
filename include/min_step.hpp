// header file for minimizers
#include <functional>
#include <Eigen/Dense>

using Eigen::Array;
using Eigen::Matrix;

template <size_t N>
using parvec = Array<double, N, 1>;

template <size_t N>
using func_t = std::function<double(parvec<N>)>;
template <size_t N>
using grad_t = std::function<parvec<N>(parvec<N>)>;

// we should also implement a golden section search
double line_search( std::function<double(double)>, double, double, size_t maxiter = 16, double tol=1e-6 );

// in the future some may report a Hessian approximation
// this should be a separate interface
// a full Newton method would take a Hessian as well as a gradient (probably not worth implementing)

// an interface class for each of these that take a gradient function and a vector of parameters
template <size_t N>
class IMinStep
{
public:
  // do we need to initialize with the number of parameters? some steppers
  IMinStep(){}
  virtual ~IMinStep(){}
  // step_pars returns the next value of the parameter vector
  virtual parvec<N> getDeltaPar( func_t<N>, grad_t<N>, parvec<N> ) = 0;
  // remove any cached information, which some minimizers use to optimize the step rates
  virtual void resetCache() = 0;
};

// ---- basic gradient descent ----

template <size_t N>
class GradientDescent : public IMinStep<N>
{
public:
  GradientDescent() {}
  ~GradientDescent() {};
  virtual parvec<N> getDeltaPar( func_t<N>, grad_t<N>, parvec<N> ) override;
  virtual void resetCache() override {}; // simple gradient descent uses no cache
  void setLearnRate(double lr) {learn_rate_ = lr;}
private:
  double learn_rate_ = 0.01;
};

template <size_t N>
parvec<N> GradientDescent<N>::getDeltaPar( func_t<N>, grad_t<N> g, parvec<N> pars ) {
  return - learn_rate_ * g(pars);
}


// ---- adding momentum term to naive gradient descent to help with "canyons" ----

template <size_t N>
class GradientDescentWithMomentum : public IMinStep<N>
{
public:
  GradientDescentWithMomentum() {};
  ~GradientDescentWithMomentum() {};
  virtual parvec<N> getDeltaPar( func_t<N>, grad_t<N>, parvec<N> ) override;
  virtual void resetCache() override {momentum_term_.setZero();}
  void setLearnRate(double lr) {learn_rate_ = lr;}
  void setMomentumScale(double ms) {momentum_scale_ = ms;}
private:
  parvec<N> momentum_term_ = parvec<N>::Zero();
  double learn_rate_ = 0.005;
  double momentum_scale_ = 0.875;
};

template <size_t N>
parvec<N> GradientDescentWithMomentum<N>::getDeltaPar( func_t<N>, grad_t<N> g, parvec<N> pars ) {
  momentum_term_ = momentum_scale_ * momentum_term_ + learn_rate_ * g(pars);
  return - momentum_term_;
}


// // ---- Nesterov's accelerated gradient ----
// // an improved version of the momentum addition

template <size_t N>
class AcceleratedGradient : public IMinStep<N>
{
public:
  AcceleratedGradient() {};
  // AcceleratedGradient(double learn_rate=0.01,
  // 		      double momentum_scale=0.875);
  ~AcceleratedGradient() {};
  virtual parvec<N> getDeltaPar( func_t<N>, grad_t<N>, parvec<N> ) override;
  virtual void resetCache() override {momentum_term_.setZero();}
  void setLearnRate(double lr) {learn_rate_ = lr;}
  void setMomentumScale(double ms) {momentum_scale_ = ms;}
private:
  parvec<N> momentum_term_ = parvec<N>::Zero();
  double learn_rate_ = 0.005;
  double momentum_scale_ = 0.875;
};

template <size_t N>
parvec<N> AcceleratedGradient<N>::getDeltaPar( func_t<N>, grad_t<N> g, parvec<N> pars ) {
  momentum_term_ *= momentum_scale_; // exponentially reduce momentum term
  momentum_term_ += learn_rate_ * g(pars - momentum_term_);
  // momentum_term_ += learn_rate_ * g(pars - 0.5*momentum_term_); // using an average between new and old points can help with convergence on rosenbrock function
  return - momentum_term_;
}


// ---- ADADELTA has an adaptive, unitless learning rate ----
// it is probably often a good first choice because it tends to be insensitive to the hyperparameters

template <size_t N>
class AdaDelta : public IMinStep<N>
{
public:
  AdaDelta() {};
  ~AdaDelta() {};
  virtual parvec<N> getDeltaPar( func_t<N>, grad_t<N>, parvec<N> ) override;
  virtual void resetCache() override;
  void setDecayScale(double ds) {decay_scale_ = ds;}
  void setLearnRate(double lr) {learn_rate_ = lr;}
  void setEpsilon(double ep) {ep_ = ep;}
private:
  parvec<N> accum_grad_sq_ = parvec<N>::Zero();
  parvec<N> accum_dpar_sq_ = parvec<N>::Zero();
  parvec<N> last_delta_par_ = parvec<N>::Zero();
  double decay_scale_ = 0.9375; // similar to window average of last 16 values. 0.875 for scale of 8 previous values
  double learn_rate_ = 1.0; // a default value that can be adjusted down if necessary
  double ep_ = 1e-6;
};

template <size_t N>
void AdaDelta<N>::resetCache()
{
  accum_grad_sq_.setZero();
  accum_dpar_sq_.setZero();
  last_delta_par_.setZero();
}

template <size_t N>
parvec<N> AdaDelta<N>::getDeltaPar( func_t<N> f, grad_t<N> g, parvec<N> pars )
{
  parvec<N> grad = g(pars); // an improvement might be to use an accelerated version
  // element-wise learn rate
  parvec<N> adj_learn_rate = sqrt((accum_dpar_sq_ + ep_)/(accum_grad_sq_ + ep_));

  // scaling down helps convergence but does seem to slow things down.
  // perhaps a line search (golden section) would be an improvement
  parvec<N> dp = - learn_rate_ * adj_learn_rate * grad;
  // we can do an explicit check to keep from over-stepping

  // if ( f(pars+dp) > f(pars) ) {
  //   // a rapidly-truncated golden section search would probably be better since we can restrict 0 < alpha < 1
  //   // this actually ends up slowing down easy convergences so perhaps it's not the best approach
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
  last_delta_par_ = dp;
  return dp;
  
}


// ---- BFGS ----
// a quasi-Newton method that uses an approximation for the Hessian, but uses a lot of memory to store it (N^2)

template <size_t N>
class Bfgs : public IMinStep<N>
{
public:
  Bfgs() {};
  ~Bfgs() {};
  virtual parvec<N> getDeltaPar( func_t<N>, grad_t<N>, parvec<N> ) override;
  virtual void resetCache() override;
private:
  Matrix<double, N, N> hessian_approx_ = decltype(hessian_approx_)::Identity();
  // VectorXd lastpar_ = ...Zero();
  // VectorXd lastgrad_;
  double learn_rate_ = 1.0;
};

template <size_t N>
void Bfgs<N>::resetCache()
{
  hessian_approx_.setIdentity();
  // lastpar_.setZero();
  // lastgrad_.setZero();
}

template <size_t N>
parvec<N> Bfgs<N>::getDeltaPar( func_t<N> f, grad_t<N> g, parvec<N> par )
{
  Matrix<double, N, 1> grad = g(par).matrix();
  // the hessian approximation should remain positive definite. LLT requires positive-definiteness.
  // LLT actually yields NaN solutions at times so perhaps our hessian is not always pos-def.
  Matrix<double, N, 1> deltap = - learn_rate_ * hessian_approx_.llt().solve(grad);
  // VectorXd deltap = - learn_rate_ * hessian_approx_.ldlt().solve(grad);
  // VectorXd deltap = - learn_rate_ * hessian_approx_.householderQr().solve(grad); // no requirements on matrix for this

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

  // we might only want to do this line search if we see an increasing value f(p+dp) > f(p)
  // , in which case a golden section search might be more efficient
  auto f_line = [&](double x){return f(par + x*deltap.array());};
  double alpha_step = line_search( f_line, 0.7071067, 1.0 );  
  // if (fabs(alpha_step) < 1e-2) {
  //   std::cout << "small step in line search: " << alpha_step << std::endl;
  //   for (double xl = -2.0e-2; xl < 0.1; xl += 1.0e-2) {
  //     std::cout << "f(" << xl << ") = " << f_line(xl) << std::endl;
  //   }
  // }
  deltap *= alpha_step;
  
  Matrix<double, N, 1> deltagrad = (g(par+deltap.array()).matrix() - grad);

  // storing hessian by its square root could  possibly help numerical stability
  // might be good to check for divide-by-zero here, even if it's not likely to happen
  hessian_approx_ += (deltagrad * deltagrad.transpose())/deltagrad.dot(deltap)
    - hessian_approx_ * deltap * deltap.transpose() * hessian_approx_ / (deltap.transpose() * hessian_approx_ * deltap);
  
  return deltap.array();
}
