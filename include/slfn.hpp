#pragma once
#include "global.hpp"
#include "regression.hpp"
#include "activation.hpp"
#include <boost/log/trivial.hpp>
// #include <boost/log/sources/basic_logger.hpp> // could be useful in class defs
#include <Eigen/Dense>

#include <fstream>

namespace { // protect these definitions locally; don't pollute global namespace
  using std::istream;
  using std::ostream;
  using std::ifstream;
  using std::ofstream;
  using std::ios;
}

namespace ceptron {

  // could a base case using the CRTP help avoid some of the awkwardness in the definitions below?
  // template <typename Derived>
  // class SlfnBase
  // {
  // public:
  //   static constexpr size_t size = Derived::size;
  //   // using Derived::randomInit; // this doesn't work
  // };
  
  // we will really want to break this apart (into layers?) and generalize it but first let's get a simple working example.
  // this is a "single hidden layer feedforward network" (SLFN) with 1 output
  // a multi-layer generalization is a FFNN
  // N is input size, M is output layer size, P is the hidden layer size
  template <size_t N, size_t M=1, size_t P=(N+M)/2,
	    RegressionType Reg=RegressionType::Categorical,
	    InternalActivator Act=InternalActivator::Tanh>
  class SlfnStatic // : public SlfnBase<SlfnStatic<N,M,P,Reg,Act>>
  {
  public:
    static constexpr size_t inputs = N;
    static constexpr size_t outputs = M;
    static constexpr size_t hiddens = P; // this member won't generalize well, so we should replace typedefs for 1st, 2nd layer matrix types, for instance
    static constexpr RegressionType RegType = Reg;
    static constexpr InternalActivator ActType = Act;
  private:
    using this_t = SlfnStatic<N,M,P,Reg,Act>; // just a shorthand
    static constexpr size_t size_w1_ = P*(N+1); // first layer
    static constexpr size_t size_w2_ = M*(P+1); // output layer
    // static constexpr size_t size_ = size_w1_ + size_w2_; // total size of data required to store net
  public:
    constexpr static size_t size = size_w1_ + size_w2_;
    // constexpr static size_t size() {return size_;}
    SlfnStatic() {randomInit();} // forgetting to randomly initialize can be bad, so we'll do it automatically right now.
    SlfnStatic(const Array</*size_*/>& ar) {net_=ar;}; // to construct/convert directly from array
    // ~SlfnStatic() {};
    void randomInit();
    // this could be useful for indicating which data elements are most relevant.
    Array<N> inputLayerActivation(const Vec<N>& in) const; // returns activation from single input. these will no longer be cached
    Array<P> hiddenLayerActivation(const Vec<N>& in) const; // they don't actually need to be member functions
    // using "auto" may not be best when returning a map view -- TODO: look into this more?
    auto getFirstSynapses() const {return Map< const Mat<P, N+1> >(net_.data());};
    auto getSecondSynapses() const
    {return Map< const Mat<M, P+1> >(net_.template segment<size_w2_>(size_w1_).data());};
    // const Array<size_>& getNetValue() const {return net_;}
    const Array<>& getNetValue() const {return net_;}
    Array<>& accessNetValue() {return net_;} // allows use of in-place operations

    bool operator==(const this_t& other) const;
    bool operator!=(const this_t& other) const {return !(this->operator==(other));}

    void toFile(const std::string& fname) const;
    void fromFile(const std::string& fname);
  
  private:
    // we will likely move to not storing the network data directly in this class.
    // it may make sense to declare a struct (union) of functions + net data
    Array<> net_ = Array<size>::Zero(size);

  public:
    // define stream operators here at the end
    // defining this in the class template is annoying but it allows us to avoid introducing
    //  another layer of template parameters. known as the "making new friends" idiom.
    friend istream& operator>> (istream& in, this_t& me) {
      // TODO: some metadata at the top would be nice for verification
      // let's wait until we have a more general setup (e.g. multilayer) before we worry about that.
      for (size_t i=0; i<me.size; ++i) {
	// the fact that this one line doesn't work is actually either a bug in g++ or a flaw in the standards.
	// it may be fixed in C++ 17 but g++ isn't there yet.
	// in >> std::hexfloat >> me.net_(i);
	// see: https://stackoverflow.com/questions/42604596/read-and-write-using-stdhexfloat
	//  for link to discussion as well as this workaround suggestion:
	std::string s;
	in >> std::hexfloat >> s;
	me.net_(i) = std::strtod(s.data(), nullptr);
      }
      return in;
    }
    friend ostream& operator<<(ostream& out, const this_t& me) {
      for (size_t i=0; i<me.size; ++i) {
	// we do want to go hexfloat, otherwise we suffer a precision loss
	// removing newline doesn't work even w/ binary, possibly because of the issue discussed in operatior>>().
	out << std::hexfloat << me.net_(i) << '\n';
      }
      return out;
    }

  };


  template <size_t N, size_t M, size_t P,
  	    RegressionType Reg,
  	    InternalActivator Act>
  void SlfnStatic<N,M,P,Reg,Act>::randomInit() {
    // it is better to scale random initialization by 1/sqrt(n) where n is the number of inputs in a synapse.
    //   this results in a variance of the neuron output that is not dependent on the number of inputs.
    //   dividing by the total size is just a rough approximation
    // We also only want to randomize the weights -- the initial biases should be kept at zero.
    // TODO
    net_ = (1./size)*Array<size>::Random();
  }


  // we should be able to build up multi-layer networks by having each layer work as its own and having their forward and backward propagations interact with each other


  template <typename Net>
  ceptron::func_grad_res<>
  costFuncAndGrad(const Net& net, const BatchVec<Net::inputs>& x0, const BatchVec<Net::outputs>& y, double l2reg = 0.0) {
    const auto batchSize = x0.cols();
    assert( batchSize == y.cols() );

    constexpr size_t N = Net::inputs;
    constexpr size_t M = Net::outputs;
    constexpr size_t P = Net::hiddens;
  
    if (Net::RegType == RegressionType::Categorical) {
      // this if statement should be optimized away at compile-time
      if ((y.colwise().sum().array() > 1.0).any()) {
	// multiple nets can be used for non-exclusive categories
	// TODO: implement logging system, and suppress this warning
	BOOST_LOG_TRIVIAL(warning) << "warning: classification data breaks unitarity. this net assumes mutually exclusive categories." << std::endl;
	BOOST_LOG_TRIVIAL(debug) << "y values:" << y.transpose();
      }
    }
  
    // propagate forwards
    // it may be simpler to split into weights and biases
    // these matrix types should be typedefs in Net
    Mat<P, N+1> w1 = net.getFirstSynapses(); // don't think there is copying here for a Map object... but we should verify how it actually works.
    Mat<M, P+1> w2 = net.getSecondSynapses(); // const references shouldn't be necessary for the expression templates to work

    // at some point we might wish to experiment with column-major (default) vs. row-major data storage order
    BatchVec<P> a1 = w1.template leftCols<1>() * BatchVec<1>::Ones(1,batchSize) // bias terms
      + w1.template rightCols<N>() * x0; // weights term
    // apply activation function element-wise; need to specify the template type
    BatchVec<P> x1 = ActivFunc<Net::ActType>::template activ< BatchArray<P> >(a1.array()).matrix();
  
    BatchVec<M> a2 = w2.template leftCols<1>() * BatchVec<1>::Ones(1, batchSize)
      + w2.template rightCols<P>() * x1;
    assert( a2.cols() == batchSize );
  
    BatchVec<M> x2 = Regressor<Net::RegType>::template outputGate< BatchArray<M> >(a2.array()).matrix();
    assert( x2.cols() == batchSize );

    double costFuncVal = Regressor<Net::RegType>::template costFuncVal< BatchArray<M> >(x2.array(), y.array());
    double costFuncReg = w1.template rightCols<N>().array().square().sum()
      + w2.template rightCols<P>().array().square().sum(); // lambda*sum (weights^2), but we don't include the bias terms
    costFuncVal += l2reg * costFuncReg;
  
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
  
    BatchVec<P> e1 = ActivFunc<Net::ActType>::template activToD< BatchArray<P> >(x1.array()).matrix();
    BatchVec<P> d1 = e1.cwiseProduct( (w2.template rightCols<P>()).transpose() * d2 );

    // Array<SlfnStatic<N,M,P>::size()> costFuncGrad;
    Array<Net::size> costFuncGrad;
    // Vec<P> gb1 = d1.rowwise().sum(); // this operation contracts along the axis of different batch data points
    Mat<P,N> gw1 = d1 * x0.transpose() + 2.0*l2reg*w1.template rightCols<N>(); // add regularization term
    // Vec<M> gb2 = d2.rowwise().sum();
    Mat<M,P> gw2 = d2 * x1.transpose() + 2.0*l2reg*w2.template rightCols<P>();
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
  template <typename Net>
  double costFunc(const Net& net, const BatchVec<Net::inputs>& x0, const BatchVec<Net::outputs>& y, double l2reg = 0.0) {

    constexpr size_t N = Net::inputs;
    constexpr size_t M = Net::outputs;
    constexpr size_t P = Net::hiddens;
  
    // we won't do as many checks in this version -- the whole point is to be fast.
    const auto batchSize = x0.cols();
    // propagate forwards
    Mat<P, N+1> w1 = net.getFirstSynapses(); // don't think there is copying here for a Map object... but we should verify how it actually works.
    Mat<M, P+1> w2 = net.getSecondSynapses(); // const references shouldn't be necessary for the expression templates to work

    // a_n should just be temporary expressions
    BatchVec<P> a1 = w1.template leftCols<1>() * BatchVec<1>::Ones(1,batchSize) // bias terms
      + w1.template rightCols<N>() * x0; // weights term
    BatchVec<P> x1 = ActivFunc<Net::ActType>::template activ< BatchArray<P> >(a1.array()).matrix(); // net output of layer 1
    BatchVec<M> a2 = w2.template leftCols<1>() * BatchVec<1>::Ones(1, batchSize)
      + w2.template rightCols<P>() * x1;
    BatchVec<M> x2 = Regressor<Net::RegType>::template outputGate< BatchArray<M> >(a2.array()).matrix();

    // we might return f and gradient together... or maybe we just cache them
    double costFuncVal = Regressor<Net::RegType>::template costFuncVal< BatchArray<M> >(x2.array(), y.array());
    double costFuncReg = w1.template rightCols<N>().array().square().sum()
      + w2.template rightCols<P>().array().square().sum(); // lambda*sum (weights^2), but we don't include the bias terms
    costFuncVal += l2reg * costFuncReg;
    // normalize after regularization term, which seems like a strange convention
    costFuncVal /= batchSize;
    return costFuncVal;
  }

  // this returns a prediction for a single data point x0
  template <typename Net>
  Vec<Net::outputs> prediction(const Net& net, const Vec<Net::inputs>& x0) {
    constexpr size_t N = Net::inputs;
    constexpr size_t M = Net::outputs;
    constexpr size_t P = Net::hiddens;

    // again, these typedefs should come out of Net, not N, M, and P
    Mat<P, N+1> w1 = net.getFirstSynapses();
    Mat<M, P+1> w2 = net.getSecondSynapses();

    // a_n should just be temporary expressions
    Vec<P> a1 = w1.template leftCols<1>() // bias terms
      + w1.template rightCols<N>() * x0; // weights term
    Vec<P> x1 = ActivFunc<Net::ActType>::template activ< Array<P> >(a1.array()).matrix(); // net output of layer 1
    Vec<M> a2 = w2.template leftCols<1>()
      + w2.template rightCols<P>() * x1;
    Vec<M> x2 = Regressor<Net::RegType>::template outputGate< Array<M> >(a2.array()).matrix();
    return x2;
  }

  // instead of all this, we could declare a base class and use the CRTP
  template <size_t N, size_t M, size_t P,
	    RegressionType Reg,
	    InternalActivator Act>
  bool SlfnStatic<N,M,P,Reg,Act>::operator==(const SlfnStatic<N,M,P,Reg,Act>& other) const {
    return (this->net_ == other.net_).all();
  }


  template <size_t N, size_t M, size_t P,
	    RegressionType Reg,
	    InternalActivator Act>
  void SlfnStatic<N,M,P,Reg,Act>::toFile(const std::string& fname) const {
    // ios::trunc erases any previous content in the file.
    ofstream fout(fname , ios::binary | ios::trunc );
    if (!fout.is_open()) {
      BOOST_LOG_TRIVIAL(error) << "could not open file " << fname << " for writing.";
      return;
    }
    fout << *this;
    fout.close();
  }

  template <size_t N, size_t M, size_t P,
	    RegressionType Reg,
	    InternalActivator Act>	  
  void SlfnStatic<N,M,P,Reg,Act>::fromFile(const std::string& fname) {
    ifstream fin(fname, ios::binary);
    if (!fin.is_open()) {
      BOOST_LOG_TRIVIAL(error) << "could not open file " << fname << " for reading.";
      return;
    }
    fin >> *this; // streams are not efficient
    fin.close();
  }

} // namespace ceptron
