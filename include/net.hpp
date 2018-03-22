#include "regression.hpp"
#include "activation.hpp"
#include <Eigen/Dense>

#include <iostream>
#include <fstream>

namespace { // protect these definitions locally, at least for now until we come up with a better global scheme
  template <size_t M, size_t N>
  using Mat = Eigen::Matrix<double, M, N>;
  template <size_t M>
  using Vec = Mat<M, 1>;
  
  template <size_t M> // don't think we need anything but column arrays
  using Array = Eigen::Array<double, M, 1>;

  using Eigen::Map;

  using std::cout;
  using std::endl;
  using std::string;
  using std::istream;
  using std::ostream;
  using std::ifstream;
  using std::ofstream;
  using std::ios;
}

// an interface class for NNs that takes only number of inputs and number of outputs.
// we could probably use a better name for  this
template <size_t Nin, size_t Nout, size_t Ntot>
class IFeedForward
{
public:
  constexpr static size_t size() {return Ntot;} // needs to be declared explicitly static to work. we may need to just make size_ public
  virtual const Vec<Nout>& getOutput() const = 0; // returns cache
  virtual void propagateData(const Vec<Nin>& x0, const Vec<Nout>& y) = 0;
  virtual double getCostFuncVal() const = 0;
  virtual const Array<Ntot>& getCostFuncGrad() const = 0;
  // Array<Nin+1> getInputLayerActivation() const = 0; // do we actually need this in the interface?
  virtual const Array<Ntot>& getNetValue() const = 0;
  virtual Array<Ntot>& accessNetValue() = 0; // allows use of in-place operations
  // do we need interfaces for read/write/serialization in here?
};

// we will really want to break this apart (into layers?) and generalize it but first let's get a simple working example.
// this is a "single hidden layer feedforward network" (SLFN) with 1 output
// N is input size, M is hidden layer size, P is output size
template <size_t N, size_t M, size_t P=1,
	  RegressionType Reg=RegressionType::Categorical,
	  InternalActivator Act=InternalActivator::Tanh>
class SingleHiddenLayer
  : public virtual IFeedForward<N, P, M*(N+1)+P*(M+1)>
    // we have a shortcut for the total size parameter M*(N+1)+P*(M+1) below;
    //  it's not ideal to manually compute the size twice.
    // could get around it w/ C macros, but ideally we could avoid that
    , protected Regressor<Reg>
    , protected ActivFunc<Act>
{
private:
  static constexpr size_t size_w1_ = M*(N+1); // first layer
  static constexpr size_t size_w2_ = P*(M+1); // output layer
  static constexpr size_t size_ = size_w1_ + size_w2_; // total size of data required to store net
public:
  // SingleHiddenLayer() {};
  // ~SingleHiddenLayer() {};
  void randomInit() {net_.setRandom();};
  const Vec<P>& getOutput() const override {return  output_layer_cache_.eval();}; // returns cache
  // maybe we should have a function like setInput() and otherwise don't allow new input arrays, so it's clear what changes the cache
  // double doPropForward(const Vec<N>&); // should maybe take an array instead of a vector? idk
  // Array<size_> doPropBackward(double); // our output right now has only 1 value // or return gradient?
  void propagateData(const Vec<N>& x0, const Vec<P>& y) override;
  double getCostFuncVal() const override;
  // const auto& getCostFuncGrad() const {return net_grad_cache_.eval();};
  const Array<size_>& getCostFuncGrad() const override {return net_grad_cache_.eval();};
  // there is a type (extra stage) of backprop that would give derivatives w.r.t. *inputs* (not net values).
  // this could be useful for indicating which data elements are most relevant.
  Array<N+1> getInputLayerActivation() const {return input_layer_cache_;};
  Array<M+1> getHiddenLayerActivation() const {return hidden_layer_cache_;};
  // using "auto" may not be best when returning a map view -- TODO: look into this more?
  auto getFirstSynapses() const {return Map< const Mat<M, N+1> >(net_.data());};
  auto getSecondSynapses() const
  {return Map< const Mat<P, M+1> >(net_.template segment<size_w2_>(size_w1_).data());};
  const Array<size_>& getNetValue() const override {return net_.eval();}  // these functions are the ones that need to be worked out for non-trivial passing of matrix data as an array. TODO: try an example
  // void setNetValue(const Eigen::Ref<const Array<size_>>& in) {net_ = in;}
  // auto& accessNetValue() {return net_;} // allows use of in-place operations
  Array<size_>& accessNetValue() override {return net_;} // allows use of in-place operations
  void resetCache();
  void setL2RegParam(double lambda); // set a parameter to add a regularization term lambda*sum(net values squared). TODO: implement this

  bool operator==(const SingleHiddenLayer<N,M,P,Reg,Act>& other) const;
  bool operator!=(const SingleHiddenLayer<N,M,P,Reg,Act>& other) const {return !(this->operator==(other));}

  // using stream operators is not a precise or efficient way to store doubles.
  // we'll save/load as binary, and there are better ways to do so directly.
  // see: http://www.cplusplus.com/doc/tutorial/files/
  // overload streaming operators for input/output
  // not re-doing this template declaration leads to warnings. we might be fine without it, however.
  template<size_t Nf, size_t Mf, size_t Pf,
  	   RegressionType Regf,
  	   InternalActivator Actf>
  friend istream& operator>>(istream& in, SingleHiddenLayer<Nf, Mf, Pf, Regf, Actf>& me);
  template<size_t Nf, size_t Mf, size_t Pf,
  	   RegressionType Regf,
  	   InternalActivator Actf>
  friend ostream& operator<<(ostream& out, const SingleHiddenLayer<Nf, Mf, Pf, Regf, Actf>& me);
  void toFile(const string& fname) const;
  void fromFile(const string& fname);
  
private:
  // we could also store these as matrices and map to an array using segment<>
  Array<size_> net_ = Array<size_>::Zero();  
  // we should also cache the activation layer values. this will be necessary for efficient backpropagation.
  // possibly will need to do this for multiple input/output sets at once, for minibatch gradient descent,
  // although this loop may be able to work externally
  //  store the bias terms in the cache for convenience
  Vec<N+1> input_layer_cache_ = Vec<N+1>::Identity(); // trick to set 1st element to 1 be default: bias term
  Vec<M+1> hidden_layer_cache_ = Vec<M+1>::Identity();
  Vec<P> output_layer_cache_ = Vec<P>::Zero(); // no bias layer in output.
  Array<size_> net_grad_cache_ = Array<size_>::Zero(); // may not be necessary for this SLFN, but when we break up into layers it might be
  Vec<P> output_value_cache_ = Vec<P>::Zero();
  // possible other parameters: regularization; normalization of cost function
};

// this operation just resets all data to zero and so may not be particularly useful except perhaps for validation and sanity checks
template <size_t N, size_t M, size_t P,
	  RegressionType Reg,
	  InternalActivator Act>
void SingleHiddenLayer<N,M,P,Reg,Act>::resetCache() {
  input_layer_cache_ = Vec<N+1>::Identity();
  hidden_layer_cache_ = Vec<M+1>::Identity();
  output_layer_cache_ = Vec<P>::Zero();
  output_value_cache_ = Vec<P>::Zero();
  net_grad_cache_ = Array<size_>::Zero();
}

template <size_t N, size_t M, size_t P,
	  RegressionType Reg,
	  InternalActivator Act>
double SingleHiddenLayer<N,M,P,Reg,Act>::getCostFuncVal() const {
  return Regressor<Reg>::template costFuncVal< Array<P> >(output_layer_cache_.array(), output_value_cache_.array());
}


// we should be able to build up multi-layer networks by having each layer work as its own and having their forward and backward propagations interact with each other
// this function performs both the forward then the backward propagation for a given data point
// generalization to mini-batches might be able to occur outside

// should really consider changing the interface to have the parameters be external, but this is probably only useful for validation;
//  for real training, it will help to have the gradient stored (except perhaps accelerated gradient)
template <size_t N, size_t M, size_t P,
	  RegressionType Reg,
	  InternalActivator Act>
void SingleHiddenLayer<N,M,P,Reg,Act>::propagateData(const Vec<N>& x0, const Vec<P>& y) {
  // resetCache(); // ? shouldn't actually be necessary; should compare in checks

  if (Reg == RegressionType::Categorical) {
    // this if statement should be optimized away at compile-time
    if (y.sum() > 1.0) {
      // multiple nets can be used for non-exclusive categories
      std::cout << "warning: classification data breaks unitarity. this net assumes mutually exclusive categories." << std::endl;
      std::cout << "debug: y values:" << y.transpose() << std::endl;
    }
  }
  
  // propagate forwards
  Mat<M, N+1> w1 = getFirstSynapses(); // don't think there is copying here for a Map object... but we should verify how it actually works.
  Mat<P, M+1> w2 = getSecondSynapses(); // const references shouldn't be necessary for the expression templates to work

  input_layer_cache_.template segment<N>(1) = x0; // offset bias term
  // at some point we might wish to experiment with column-major (default) vs. row-major data storage order
  Vec<M> a1 = w1 * input_layer_cache_; // matrix mult term
  // apply activation function element-wise; need to specify the template type
  hidden_layer_cache_.template segment<M>(1) =
    ActivFunc<Act>::template activ< Array<M> >(a1.array()).matrix().eval();
  
  assert( input_layer_cache_(0) == 1. && hidden_layer_cache_(0) == 1.);
  
  Vec<P> a2 = w2 * hidden_layer_cache_;

  output_layer_cache_ = Regressor<Reg>::template outputGate< Array<P> >(a2.array()).matrix();
  output_value_cache_ = y;

  // now do backwards propagation
  // the dependence of the cost function on the data y depends on the type of regression, though it is quite similar for logistic and linear.
  // we will assume logistic for now, but it should require minor changes to generalize.

  // there is usually a 1/N term for the number of data points as well; perhaps should be handled externally
  // or maybe this should be 1/N(output layers), since a 2-output probability net should be the same as a 1-output one
  
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
  Vec<M> e1 = ActivFunc<Act>::template activToD< Array<M> >(hidden_layer_cache_.template segment<M>(1).array()).matrix();
  
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


template <size_t N, size_t M, size_t P,
	  RegressionType Reg,
	  InternalActivator Act>	  
bool SingleHiddenLayer<N,M,P,Reg,Act>::operator==(const SingleHiddenLayer<N,M,P,Reg,Act>& other) const {
  return (this->net_ == other.net_).all();
}

template <size_t N, size_t M, size_t P,
	  RegressionType Reg,
	  InternalActivator Act>	  
istream& operator>>(istream& in, SingleHiddenLayer<N, M, P, Reg, Act>& me) {
  for (size_t i=0; i<me.size_; ++i) {
    // the fact that this one line doesn't work is actually either a bug in g++ or a flaw in the standards.
    // it may be fixed in C++ 17 but g++ isn't there yet.
    // in >> std::hexfloat >> me.net_(i);    
    // see: https://stackoverflow.com/questions/42604596/read-and-write-using-stdhexfloat for link to discussion as well as this workaround suggestion.
    
    std::string s;
    in >> std::hexfloat >> s;
    me.net_(i) = std::strtod(s.data(), nullptr);
  }
  return in;
}

template <size_t N, size_t M, size_t P,
	  RegressionType Reg,
	  InternalActivator Act>	  
ostream& operator<<(ostream& out, const SingleHiddenLayer<N, M, P, Reg, Act>& me) {
  for (size_t i=0; i<me.size_; ++i) {
    // we do want to go hexfloat, otherwise we suffer a precision loss
    out << std::hexfloat << me.net_(i) << '\n';// removing newline doesn't work even w/ binary, possibly because of the issue discussed in operation >>.
  }
  return out;
}


template <size_t N, size_t M, size_t P,
	  RegressionType Reg,
	  InternalActivator Act>	  
void SingleHiddenLayer<N,M,P,Reg,Act>::toFile(const string& fname) const {
  // ios::trunc erases any previous content in the file.
  ofstream fout(fname , ios::binary | ios::trunc );
  if (!fout.is_open()) {
    cout << "could not open file " << fname << " for writing." << endl;
  }
  fout << *this;
  fout.close();
}

template <size_t N, size_t M, size_t P,
	  RegressionType Reg,
	  InternalActivator Act>	  
void SingleHiddenLayer<N,M,P,Reg,Act>::fromFile(const string& fname) {
  ifstream fin(fname/*, ios::binary*/);
  if (!fin.is_open()) {
    cout << "could not open file " << fname << " for reading." << endl;
    return;
  }
  // the ios::ate flag puts us at the end which will let us count the total size
  // fin.seekg(0,ios::end);
  // std::streampos size = fin.tellg();
  // cout << "size = " << size << endl;
  // cout << "should be: " << this->size() << endl;
  // fin.seekg(0, ios::beg); // put us a distance 0 away from the beginning
  // fin.read( net_.data(), size ); // copy directly into data array
  fin >> *this; // streams are not efficient
  fin.close();
}
