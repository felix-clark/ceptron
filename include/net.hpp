#pragma once
#include "global.hpp"
#include "regression.hpp"
#include "activation.hpp"
#include <boost/log/trivial.hpp>
// #include <boost/log/sources/basic_logger.hpp> // could be useful in class defs
#include <Eigen/Dense>

#include <fstream>

namespace { // protect these definitions locally; don't pollute global namespace
  using namespace ceptron;
  using std::istream;
  using std::ostream;
  using std::ifstream;
  using std::ofstream;
  using std::ios;
}

// an interface class for NNs that takes only number of inputs and number of outputs.
// we could probably use a better name for  this
// we want the batch size to be dynamic so as to not incur difficult constraints on the input data. for large network sizes, compile-time constants won't actually help (and may in some cases hurt), so we'll use a different interface for large nets (dimension >~ 32)
// we can't blindly plug in -1 for dynamic, because in some cases we use N+1 to include a bias term. a dynamic network will probably use a generic dynamic interface.
// template <size_t Nin, size_t Nout, size_t Ntot>
// class ISmallFeedForward
// {
// public:
//   constexpr static size_t size() {return Ntot;} // needs to be declared explicitly static to work. we may need to just make size_ public
//   // virtual const Vec<Nout>& getOutput() const = 0; // returns cache. complicated w/ batch input; do we really need this?
//   // virtual ceptron::func_grad_res<Ntot> costFuncAndGrad(const BatchVec<Nin>& x0, const BatchVec<Nout>& y) const = 0;
//   // virtual double costFunc(const BatchVec<Nin>& x0, const BatchVec<Nout>& y) const = 0; // some minimizers will perform a quick line search and only need the function value
//   // virtual double getCostFuncVal() const = 0;
//   // virtual const Array<Ntot>& getCostFuncGrad() const = 0;
//   // Array<Nin+1> getInputLayerActivation() const = 0; // do we actually need this in the interface?
//   virtual const Array<Ntot>& getNetValue() const = 0;
//   virtual Array<Ntot>& accessNetValue() = 0; // allows use of in-place operations
//   // do we need interfaces for read/write/serialization in here?
// };

// we will really want to break this apart (into layers?) and generalize it but first let's get a simple working example.
// this is a "single hidden layer feedforward network" (SLFN) with 1 output
// N is input size, M is output layer size, P is the hidden layer size
template <size_t N, size_t M=1, size_t P=N>
class SingleHiddenLayer
{
private:
  static constexpr size_t size_w1_ = P*(N+1); // first layer
  static constexpr size_t size_w2_ = M*(P+1); // output layer
  static constexpr size_t size_ = size_w1_ + size_w2_; // total size of data required to store net
public:
  constexpr static size_t size() {return size_;}
  // SingleHiddenLayer() {};
  // ~SingleHiddenLayer() {};
  void randomInit() {net_.setRandom();};
  // this could be useful for indicating which data elements are most relevant.
  Array<N> getInputLayerActivation(const Vec<N>& in) const; // returns activation from single input. these will no longer be cached
  Array<P> getHiddenLayerActivation(const Vec<N>& in) const;
  // using "auto" may not be best when returning a map view -- TODO: look into this more?
  auto getFirstSynapses() const {return Map< const Mat<P, N+1> >(net_.data());};
  auto getSecondSynapses() const
  {return Map< const Mat<M, P+1> >(net_.template segment<size_w2_>(size_w1_).data());};
  const Array<size_>& getNetValue() const {return net_.eval();}  // these functions are the ones that need to be worked out for non-trivial passing of matrix data as an array. TODO: try an example
  Array<size_>& accessNetValue() {return net_;} // allows use of in-place operations
  // we should probably handle regularization in the minimizer
  // void setL2RegParam(double lambda); // set a parameter to add a regularization term lambda*sum(net values squared). TODO: implement this (but it should possibly happen in the minimizer/trainer)

  bool operator==(const SingleHiddenLayer<N,M,P>& other) const;
  bool operator!=(const SingleHiddenLayer<N,M,P>& other) const {return !(this->operator==(other));}

  // using stream operators is not the most precise or efficient way to store doubles.
  // we'll save/load as binary, and there are better ways to do so directly.
  // overload streaming operators for input/output
  // not re-doing this template declaration leads to warnings. we might be fine without it, however.
  template<size_t Nf, size_t Mf, size_t Pf>
  friend istream& operator>>(istream& in, SingleHiddenLayer<Nf, Mf, Pf>& me);
  template<size_t Nf, size_t Mf, size_t Pf>
  friend ostream& operator<<(ostream& out, const SingleHiddenLayer<Nf, Mf, Pf>& me);
  void toFile(const std::string& fname) const;
  void fromFile(const std::string& fname);
  
private:
  // we could also store these as matrices and map to an array using segment<>
  Array<size_> net_ = Array<size_>::Zero();  
};

// we should be able to build up multi-layer networks by having each layer work as its own and having their forward and backward propagations interact with each other
// this function performs both the forward then the backward propagation for a given data point
// generalization to mini-batches might be able to occur outside



// should really consider changing the interface to have the parameters be external, but this is probably only useful for validation;
//  for real training, it will help to have the gradient stored (except perhaps accelerated gradient)
template <size_t N, size_t M, size_t P,
	  RegressionType Reg,
	  InternalActivator Act>
ceptron::func_grad_res<SingleHiddenLayer<N,M,P>::size()>
costFuncAndGrad(const SingleHiddenLayer<N,M,P>& net, const BatchVec<N>& x0, const BatchVec<M>& y) {
  const auto batchSize = x0.cols();
  assert( batchSize == y.cols() );

  if (Reg == RegressionType::Categorical) {
    // this if statement should be optimized away at compile-time
    if ((y.colwise().sum().array() > 1.0).any()) {
      // multiple nets can be used for non-exclusive categories
      // TODO: implement logging system, and suppress this warning
      BOOST_LOG_TRIVIAL(warning) << "warning: classification data breaks unitarity. this net assumes mutually exclusive categories." << std::endl;
      BOOST_LOG_TRIVIAL(debug) << "y values:" << y.transpose();
    }
  }
  
  // propagate forwards
  Mat<P, N+1> w1 = net.getFirstSynapses(); // don't think there is copying here for a Map object... but we should verify how it actually works.
  Mat<M, P+1> w2 = net.getSecondSynapses(); // const references shouldn't be necessary for the expression templates to work

  // at some point we might wish to experiment with column-major (default) vs. row-major data storage order
  BatchVec<P> a1 = w1.template leftCols<1>() * BatchVec<1>::Ones(1,batchSize) // bias terms
    + w1.template rightCols<N>() * x0; // weights term
  // apply activation function element-wise; need to specify the template type
  BatchVec<P> x1 = ActivFunc<Act>::template activ< BatchArray<P> >(a1.array()).matrix();
  
  BatchVec<M> a2 = w2.template leftCols<1>() * BatchVec<1>::Ones(1, batchSize)
    + w2.template rightCols<P>() * x1;
  assert( a2.cols() == batchSize );
  
  BatchVec<M> x2 = Regressor<Reg>::template outputGate< BatchArray<M> >(a2.array()).matrix();
  assert( x2.cols() == batchSize );

  double costFuncVal = Regressor<Reg>::template costFuncVal< BatchArray<M> >(x2.array(), y.array());

  // now do backwards propagation

  // with layer L being the output layer:
  // dx_j^(L)/dw_mn^(L) = sigma'[a_j^(L)] * delta_jm * x_n^(L-1)
  // dx_j^(L)/dw_mn^(L-1) = sigma'[a_j^(L)] * w_jm^(L) * sigma'[a_m^(L-1)] * x_n^(L-2)
  // dx_j^(L)/dw_mn^(L-2) = sigma'[a_j^(L)] * Sum_k{  w_jk^(L) * sigma'[a_k^(L-1)] * w_km^(L-1) } * sigma'[a_m^(L-2)] * x_n^(L-3)
  /// more generally:
  // dx_j^(L)/dw_mn^(L') = O^(L')_jm * x^(L')_n    where O^(L)_jm = delta_jm and
  //   O^(L-1)_jm = O^(L)_jk * w^(L)_km sigma'[a^(L)_m]  (matrix multiplication)

  // error term for output layer
  BatchVec<M> d2 = x2 - y;
  
  BatchVec<P> e1 = ActivFunc<Act>::template activToD< BatchArray<P> >(x1.array()).matrix();
  BatchVec<P> d1 = e1.cwiseProduct( (w2.template rightCols<P>()).transpose() * d2 );

  Array<SingleHiddenLayer<N,M,P>::size()> costFuncGrad;
  // Vec<P> gb1 = d1.rowwise().sum();
  Mat<P,N> gw1 = d1 * x0.transpose(); // this operation contracts along the axis of different batch data points
  // Vec<M> gb2 = d2.rowwise().sum();
  Mat<M,P> gw2 = d2 * x1.transpose();
  // it would be nice to make this procedure more readable:
  costFuncGrad << d1.rowwise().sum() // gb1 // this comes from the bias terms
    , Map< Vec<P*N> >(gw1.data()) // is this really safe?
    , d2.rowwise().sum() //gb2
    , Map< Vec<M*P> >(gw2.data());

  // normalize by batch size
  costFuncVal /= batchSize;
  costFuncGrad /= batchSize;
  return {costFuncVal, costFuncGrad};  
}

// we need to be careful w/ this function because it's similar to the version w/ gradient, but stops sooner.
// a compositional style of this function inside the one that includes the gradient doesn't work trivially since the backprop needs some intermediate results from this calculation
template <size_t N, size_t M, size_t P,
	  RegressionType Reg,
	  InternalActivator Act>
double costFunc(const SingleHiddenLayer<N,M,P>& net, const BatchVec<N>& x0, const BatchVec<M>& y) {
  // we won't do as many checks in this version -- the whole point is to be fast.
  const auto batchSize = x0.cols();
  // propagate forwards
  Mat<P, N+1> w1 = net.getFirstSynapses(); // don't think there is copying here for a Map object... but we should verify how it actually works.
  Mat<M, P+1> w2 = net.getSecondSynapses(); // const references shouldn't be necessary for the expression templates to work

  // a_n should just be temporary expressions
  BatchVec<P> a1 = w1.template leftCols<1>() * BatchVec<1>::Ones(1,batchSize) // bias terms
    + w1.template rightCols<N>() * x0; // weights term
  BatchVec<P> x1 = ActivFunc<Act>::template activ< BatchArray<P> >(a1.array()).matrix(); // net output of layer 1
  BatchVec<M> a2 = w2.template leftCols<1>() * BatchVec<1>::Ones(1, batchSize)
    + w2.template rightCols<P>() * x1;
  BatchVec<M> x2 = Regressor<Reg>::template outputGate< BatchArray<M> >(a2.array()).matrix();

  // we might return f and gradient together... or maybe we just cache them
  double costFuncVal = Regressor<Reg>::template costFuncVal< BatchArray<M> >(x2.array(), y.array());
  costFuncVal /= batchSize;
  return costFuncVal;
}

// this returns a prediction for a single data point x0
template <size_t N, size_t M, size_t P,
	  RegressionType Reg,
	  InternalActivator Act>
Vec<M> getPrediction(const SingleHiddenLayer<N,M,P>& net, const Vec<N>& x0) {
  Mat<P, N+1> w1 = net.getFirstSynapses();
  Mat<M, P+1> w2 = net.getSecondSynapses();

  // a_n should just be temporary expressions
  Vec<P> a1 = w1.template leftCols<1>() // bias terms
    + w1.template rightCols<N>() * x0; // weights term
  Vec<P> x1 = ActivFunc<Act>::template activ< Array<P> >(a1.array()).matrix(); // net output of layer 1
  Vec<M> a2 = w2.template leftCols<1>()
    + w2.template rightCols<P>() * x1;
  Vec<M> x2 = Regressor<Reg>::template outputGate< Array<M> >(a2.array()).matrix();
  return x2;
}


template <size_t N, size_t M, size_t P>	  
bool SingleHiddenLayer<N,M,P>::operator==(const SingleHiddenLayer<N,M,P>& other) const {
  return (this->net_ == other.net_).all();
}

template <size_t N, size_t M, size_t P>
istream& operator>>(istream& in, SingleHiddenLayer<N, M, P>& me) {
  // TODO: some metadata at the top would be nice for verification
  // let's wait until we have a more general setup (e.g. multilayer) before we worry about that.
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

template <size_t N, size_t M, size_t P>
ostream& operator<<(ostream& out, const SingleHiddenLayer<N, M, P>& me) {
  for (size_t i=0; i<me.size(); ++i) {
    // we do want to go hexfloat, otherwise we suffer a precision loss
    out << std::hexfloat << me.net_(i) << '\n';// removing newline doesn't work even w/ binary, possibly because of the issue discussed in operatior>>().
  }
  return out;
}


template <size_t N, size_t M, size_t P>
void SingleHiddenLayer<N,M,P>::toFile(const std::string& fname) const {
  // ios::trunc erases any previous content in the file.
  ofstream fout(fname , ios::binary | ios::trunc );
  if (!fout.is_open()) {
    BOOST_LOG_TRIVIAL(error) << "could not open file " << fname << " for writing.";
  }
  fout << *this;
  fout.close();
}

template <size_t N, size_t M, size_t P>	  
void SingleHiddenLayer<N,M,P>::fromFile(const std::string& fname) {
  ifstream fin(fname, ios::binary);
  if (!fin.is_open()) {
    BOOST_LOG_TRIVIAL(error) << "could not open file " << fname << " for reading.";
    return;
  }
  fin >> *this; // streams are not efficient
  fin.close();
}
