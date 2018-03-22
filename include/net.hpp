#include "activator.hpp"
#include <Eigen/Dense>

#include <iostream> // again, temporary for debugging only

namespace { // protect these definitions locally, at least for now until we come up with a better global scheme
  template <size_t M, size_t N>
  using Mat = Eigen::Matrix<double, M, N>;
  template <size_t M>
  using Vec = Mat<M, 1>;
  
  template <size_t M> // don't think we need anything but column arrays
  using Array = Eigen::Array<double, M, 1>;

  using Eigen::Map;
}




// move this code to a class determining regression type
// exclusive softmax: it assumes that categories are exclusive and that "none" is a possible category.
// most direct extension of logit for P > 1 dimensions.
// intended for use at output layer only : derivative has convenient cancellation
// template <typename Derived>
// Eigen::ArrayBase<Derived> softmax_ex(const Derived& in) { // i think this should work fine, but does require template argument to function
//   using Eigen::exp;
//   // Derived expvals = exp(in);
//   // // Array<1> expvals = exp(in);
//   // double normfact = 1.0 + expvals.sum();
//   // return expvals / normfact; // traditionally, softmax does not have this extra 1 term
//   return exp(in)/(1.0 + exp(in).sum());
// }
// for some reason we have to specify the Array below and can't use the pattern above... too tired to get it rn
template <size_t N>
Array<N> softmax_ex(const Array<N>& in) { // i think this should work fine, but does require template argument to function
  using Eigen::exp;
  Array<N> expvals = exp(in).eval();
  return expvals / (1.0 + expvals.sum()); // traditionally, softmax does not have this extra 1 term
}


enum class Regression {Categorical, LeastSquares};
// "Categorical" is "exclusive categorical", meaning |y| <= 0 and an implied "none" category is used
// for non-exclusive classification, independent NNs can be used.

// we will really want to break this apart (into layers?) and generalize it but first let's get a simple working example.
// maybe this model should be called a "FeedForward" NN
// this is a "single hidden layer feedforward network" (SLFN) with 1 output
// specifically, right now it's an (exclusive) classifier which implies logit/softmax on the output gate.
// for a y-value of zero, it is taken to mean that the data represents none of the output classes.
// N is input size, M is hidden layer size, P is output size
template <size_t N, size_t M, size_t P=1/*,
					  Regression Reg=Regression::Categorical*/,
	    InternalActivator Act=InternalActivator::Tanh>
// should inherit from some more general neural network interface once we get that going, as well as a template interface depending only on the number of input and output layers.
class SingleHiddenLayer : protected ActivFunc<Act>
{
private:
  static constexpr size_t size_w1_ = M*(N+1); // first layer
  static constexpr size_t size_w2_ = P*(M+1); // output layer
  static constexpr size_t size_ = size_w1_ + size_w2_; // total size of data required to store net
public:
  // SingleHiddenLayer() {};
  // ~SingleHiddenLayer() {};
  constexpr static size_t size() {return size_;} // needs to be declared explicitly static to work
  void randomInit() {net_.setRandom();};
  const Vec<P>& getOutput() const {return  output_layer_cache_.eval();}; // returns cache
  // maybe we should have a function like setInput() and otherwise don't allow new input arrays, so it's clear what changes the cache
  // double doPropForward(const Vec<N>&); // should maybe take an array instead of a vector? idk
  // Array<size_> doPropBackward(double); // our output right now has only 1 value // or return gradient?
  void propagateData(const Vec<N>& x0, const Vec<P>& y);
  double getCostFuncVal() const;
  const auto& getCostFuncGrad() const {return net_grad_cache_.eval();};
  // there is a type (extra stage) of backprop that would give derivatives w.r.t. *inputs* (not net values).
  // this could be useful for indicating which data elements are most relevant.
  Array<N+1> getInputLayerActivation() const {return input_layer_cache_;};
  Array<M+1> getHiddenLayerActivation() const {return hidden_layer_cache_;};
  // using "auto" may not be best when returning a map view -- TODO: look into this more?
  auto getFirstSynapses() const {return Map< const Mat<M, N+1> >(net_.data());};
  auto getSecondSynapses() const
  {return Map< const Mat<P, M+1> >(net_.template segment<size_w2_>(size_w1_).data());};
  const Array<size_>& getNetValue() const {return net_.eval();}  // these functions are the ones that need to be worked out for non-trivial passing of matrix data as an array. TODO: try an example
  // void setNetValue(const Eigen::Ref<const Array<size_>>& in) {net_ = in;}
  auto& accessNetValue() {return net_;} // allows use of in-place operations
  void resetCache();
  void setL2RegParam(double lambda); // set a parameter to add a regularization term lambda*sum(net values squared). TODO: implement this
private:
  // we could also store these as matrices and map to an array using segment<>
  Array<size_> net_ = Array<size_>::Zero();  
  // we should also cache the activation layer values. this will be necessary for efficient backpropagation.
  // possibly will need to do this for multiple input/output sets at once, for minibatch gradient descent,
  // although this loop may be able to work externally
  //  store the bias terms in the cache for convenience ?
  Vec<N+1> input_layer_cache_ = Vec<N+1>::Identity(); // trick to set 1st element to 1 be default: bias term
  Vec<M+1> hidden_layer_cache_ = Vec<M+1>::Identity();
  Vec<P> output_layer_cache_ = Vec<P>::Zero(); // no bias layer in output. normalization will be implied w/ an extra softmax term
  Array<size_> net_grad_cache_ = Array<size_>::Zero(); // may not be necessary for this SLFN, but when we break up into layers it might be
  Vec<P> output_value_cache_ = Vec<P>::Zero();
  // possible other parameters: regularization; normalization of cost function
};

// this operation just resets all data to zero and so may not be particularly useful except perhaps for validation
template <size_t N, size_t M, size_t P/*,
					Regression Reg*/,
	    InternalActivator Act>
void SingleHiddenLayer<N,M,P/*,Reg*/,Act>::resetCache() {
  input_layer_cache_ = Vec<N+1>::Identity();
  hidden_layer_cache_ = Vec<M+1>::Identity();
  output_layer_cache_ = Vec<P>::Zero();
  output_value_cache_ = Vec<P>::Zero();
  net_grad_cache_ = Array<size_>::Zero();
}


// we should be able to build up multi-layer networks by having each layer work as its own and having their forward and backward propagations interact with each other
// this function performs both the forward then the backward propagation for a given data point
// generalization to mini-batches might be able to occur outside

// should really consider changing the interface to have the parameters be external, but this is probably only useful for validation;
//  for real training, it will help to have the gradient stored (except perhaps accelerated gradient)
template <size_t N, size_t M, size_t P,/*,
					Regression Reg,*/
	  InternalActivator Act>
void SingleHiddenLayer<N,M,P/*,Reg*/,Act>::propagateData(const Vec<N>& x0, const Vec<P>& y) {
  // resetCache(); // ? shouldn't actually be necessary; should compare in checks

  // if ExclusiveClassifier:
  //   check that sum(y) <= 1
  
  // propagate forwards
  Mat<M, N+1> w1 = getFirstSynapses(); // don't think there is copying here for a Map object... but we should verify how it actually works.
  Mat<P, M+1> w2 = getSecondSynapses(); // const references shouldn't be necessary for the expression templates to work

  // input_layer_cache_(0) = 1.;
  // hidden_layer_cache_(0) = 1.;
  input_layer_cache_.template segment<N>(1) = x0; // offset bias term
  // at some point we might wish to experiment with column-major (default) vs. row-major data storage order
  // Vec<M> a1 = w1.template leftCols<1>() // bias term
  //   + w1.template rightCols<N>() * input_layer_cache; // matrix mult term
  Vec<M> a1 = w1 * input_layer_cache_; // matrix mult term
  // apply activation function element-wise; need to specify the template type
  // hidden_layer_cache_.template segment<M>(1) = logit< M >(a1.array()).matrix().eval();  // offset with bias term
  hidden_layer_cache_.template segment<M>(1) =
    ActivFunc<Act>::template activ< Array<M> >(a1.array()).matrix().eval();
  // Vec<1> a2 = w2.template leftCols<1>()
  //   + w2.template rightCols<M>() * hidden_layer_cache_;
  assert( input_layer_cache_(0) == 1. && hidden_layer_cache_(0) == 1.);
  Vec<P> a2 = w2 * hidden_layer_cache_;

  // normalize output layer by including an additional category not in the output layer (so P=1 is identical to the basic case)
  output_layer_cache_ = softmax_ex<P>(a2.array()).matrix();
  output_value_cache_ = y;

  // now do backwards propagation
  // the dependence of the cost function on the data y depends on the type of regression, though it is quite similar for logistic and linear.
  // we will assume logistic for now, but it should require minor changes to generalize.

  // there is usually a 1/N term for the number of data points as well; perhaps should be handled externally
  // or maybe this should be 1/N(output layers), since a 2-output probability net should be the same as a 1-output one
  
  /// note that this term has exact cancellation with 1st term of backprop w/ logit activation
  // , since 1st term is sigma'(a^2) = sigma*(1-sigma)
  // it probably makes sense to use a logit activation on the final layer (or softmax for multi-dimensional) regardless of internal activation functions
  // double logistic_term = (output_layer_cache_ - y)/(output_layer_cache_*(1-output_layer_cache_)); // the denominator cancels out of logit/softmax
  // there are cross-terms for softmax if the cost function is the sum of that in all nodes. check this for the multidimensional case! -- yes, the cancellation works just the same.
  // for the multidimensional case, should there be an implied additional case for noise? (i.e. for which y=0 in all bins) it probably won't affect logistic regression, but the cost function is different.
  // this might actually have the same cancelation in least-square...?
  
  // the other term in the product is dx^out/d(p), which has a layer-dependent form
    // derivative of softmax sm(x) is dsm(a_j)/d(a_k) = sm(a_j)*[delta_jk - sm(a_k)]
  // we actually will want a softmax with a non-interacting extra input layer w/ value 0 for normalization (for non-classified, or y=0)

  Vec<P> delta_out = output_layer_cache_ - output_value_cache_; // x - y
  // the above is a vector of length (output size) that must be contracted with the LHS of these matrices
  // it turns out we can contract this vector pretty early on
  
  // with layer L being the output layer:
  // dx_j^(L)/dw_mn^(L) = sigma'[a_j^(L)] * delta_jm * x_n^(L-1)
  // dx_j^(L)/dw_mn^(L-1) = sigma'[a_j^(L)] * w_jm^(L) * sigma'[a_m^(L-1)] * x_n^(L-2)
  // dx_j^(L)/dw_mn^(L-2) = sigma'[a_j^(L)] * Sum_k{  w_jk^(L) * sigma'[a_k^(L-1)] * w_km^(L-1) } * sigma'[a_m^(L-2)] * x_n^(L-3)
  /// more generally:
  // dx_j^(L)/dw_mn^(L') = O^(L')_jm * x^(L')_n    where O^(L)_jm = delta_jm and
  //   O^(L-1)_jm = O^(L)_jk * w^(L)_km sigma'[a^(L)_m]  (matrix multiplication)
  
  Vec<P> d2 = delta_out; // this is potentially a redundant copy operation, except for expression templates

  // Vec<M> e1 = logit_to_d<M>(hidden_layer_cache_.template segment<M>(1).array()).matrix(); // ignores bias term explicitly
  // ignores bias term explicitly:
  Vec<M> e1 = ActivFunc<Act>::template activ_to_d< Array<M> >(hidden_layer_cache_.template segment<M>(1).array()).matrix();
  
  Vec<M> d1 = e1.asDiagonal() * (w2.template rightCols<M>()).transpose() * d2;

  auto grad_w2 = Map< Mat<P, M+1> >(net_grad_cache_.template segment<size_w2_>(size_w1_).data());
  auto grad_w1 = Map< Mat<M, N+1> >(net_grad_cache_.template segment<size_w1_>(0).data());

  grad_w2 = d2 * hidden_layer_cache_.transpose();
  grad_w1 = d1 * input_layer_cache_.transpose();
  
  // // equivalently:
  // Mat<P, M+1> grad_w2 = o2 * hidden_layer_cache_.transpose();
  // Mat<M, N+1> grad_w1 = o1 * input_layer_cache_.transpose();
  // net_grad_cache_.template segment<size_w1_>(0) = Map< Array<size_w1_> >(grad_w1.data());
  // net_grad_cache_.template segment<size_w2_>(size_w1_) = Map< Array<size_w2_> >(grad_w2.data());
  
}

// define these here for now; they may stay this simple but they also may not.
// should cache outputs and then template-specialize this based on the settings (i.e. classifier vs. least-squares)
template <size_t N, size_t M, size_t P, InternalActivator Act>
// we can't partially-specialize a member function... this is going to get really annoying.
// will need to find a workaround.
// see first: https://stackoverflow.com/questions/165101/invalid-use-of-incomplete-type-error-with-partial-template-specialization
// seems like we need to separate template parameters around w/ inheritance structure
double SingleHiddenLayer<N,M,P/*,Regression::Categorical*/,Act>::getCostFuncVal() const {
  return  - (output_value_cache_.array()*log(output_layer_cache_.array())).sum()
    - (1-output_value_cache_.array().sum())*log1p(-output_layer_cache_.array().sum());
    // we shouldn't need a guard for when sum of output layers = 1 w/ softmax gate
}

