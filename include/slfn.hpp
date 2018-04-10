#pragma once
#include "global.hpp"
#include "regression.hpp"
#include "activation.hpp"
#include "log.hpp"
#include <Eigen/Dense>


namespace ceptron {

  // could a base case using the CRTP help avoid some of the awkwardness in the definitions below?
  template <typename Derived>
  class SlfnBase
  {
  public:
    // static constexpr size_t size = Derived::size; // this pattern does actually work. is it useful? // it actually fails in clang 3. we aren't using it now anyway so we'll let it stay commented.
  private:
    SlfnBase() = default;  // making the constructor private and the derived class a friend ensures
    friend Derived;        // that a derived class can only give its own type as a parameter
  };
  
  // we will really want to break this apart (into layers?) and generalize it but first let's get a simple working example.
  // this is a "single hidden layer feedforward network" (SLFN) with 1 output
  // a multi-layer generalization is a FFNN
  // N is input size, M is output layer size, P is the hidden layer size
  template <size_t N, size_t M=1, size_t P=(N+M)/2,
	    RegressionType Reg=RegressionType::Categorical,
	    InternalActivator Act=InternalActivator::Tanh>
  class SlfnStatic : public SlfnBase<SlfnStatic<N,M,P,Reg,Act>>
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
  public:
    constexpr static size_t size = size_w1_ + size_w2_;
    // constexpr static size_t size() {return size_;}
    SlfnStatic() = default; // don't forget to randomly initialize
    // ~SlfnStatic() = default;
    // static Array<size> randomWeights(); // returns an appropriate random initialization // make non-member function
    // this could be useful for indicating which data elements are most relevant.
    Array<N> inputLayerActivation(const Vec<N>& in) const; // returns activation from single input. these will no longer be cached
    Array<P> hiddenLayerActivation(const Vec<N>& in) const; // they don't actually need to be member functions
    // using "auto" may not be best when returning a map view -- TODO: look into this more?
    auto getFirstSynapses(const ArrayX& net) const {return Map< const Mat<P, N+1> >(net.data());};
    auto getSecondSynapses(const ArrayX& net) const
    {return Map< const Mat<M, P+1> >(net.template segment<size_w2_>(size_w1_).data());};
    // const Array<>& getNetValue() const {return net_;}
    // Array<>& accessNetValue() {return net_;} // allows use of in-place operations

    void setL2Reg(double lambda) {l2_lambda_=lambda;}
    double getL2Reg() const {return l2_lambda_;}

  private:
    // l2 regularization parameter
    double l2_lambda_=0.;
    
    // we will likely move to not storing the network data directly in this class.
    // it may make sense to declare a struct (union) of functions + net data
    // Array<> net_ = Array<size>::Zero(size);
  }; // class SlfnStatic

  template <typename Net>
  Array<Net::size> randomWeights()
  {
    constexpr size_t N = Net::inputs;
    constexpr size_t P = Net::hiddens;
    constexpr size_t M = Net::outputs;
    // using Net = SlfnStatic<N,M,P,Reg,Act>;
    Array<Net::size> net = Array<Net::size>::Zero(); // set all to zero - biases should be initialized to zero
    // the weight variances should be scaled down by a factor of the square root of the # of their inputs,
    //  so that the variance of their output is not a function of the number of inputs.
    Map< Mat<P, N> >((net.template segment<P*N>(P)).data()) = Mat<P, N>::Random()/sqrt(N);
    Map< Mat<M, P> >((net.template segment<M*P>(P*(N+1)+M)).data()) = Mat<M, P>::Random()/sqrt(P);
    return net;
  }

  // we should be able to build up multi-layer networks by having each layer work as its own and having their forward and backward propagations interact with each other

  // possibly we can move this back to being a member variable, but it's not clear to me rn what would be gained/lost.
  // the runtime version has that organization, so perhaps we'll keep them different and see which we like best.
  template <typename Net>
  ceptron::func_grad_res
  costFuncAndGrad(const Net& net, const ArrayX& netvals, const BatchVec<Net::inputs>& x0, const BatchVec<Net::outputs>& y) {
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
    	LOG_DEBUG("classification data breaks unitarity. this net assumes mutually exclusive categories.");
    	LOG_TRACE("y values:" << y.transpose());
      }
    }
  
    // propagate forwards
    // it may be simpler to split into weights and biases
    // these matrix types should be typedefs in Net
    Mat<P, N+1> w1 = net.getFirstSynapses(netvals);
    Mat<M, P+1> w2 = net.getSecondSynapses(netvals);

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
    costFuncVal += net.getL2Reg() * costFuncReg;
  
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
    Mat<P,N> gw1 = d1 * x0.transpose() + 2.0*net.getL2Reg()*w1.template rightCols<N>(); // add regularization term
    // Vec<M> gb2 = d2.rowwise().sum();
    Mat<M,P> gw2 = d2 * x1.transpose() + 2.0*net.getL2Reg()*w2.template rightCols<P>();
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
  double costFunc(const Net& net, const ArrayX& netvals, const BatchVec<Net::inputs>& x0, const BatchVec<Net::outputs>& y) {

    constexpr size_t N = Net::inputs;
    constexpr size_t M = Net::outputs;
    constexpr size_t P = Net::hiddens;
  
    // we won't do as many checks in this version -- the whole point is to be fast.
    const auto batchSize = x0.cols();
    // propagate forwards
    Mat<P, N+1> w1 = net.getFirstSynapses(netvals);
    Mat<M, P+1> w2 = net.getSecondSynapses(netvals);

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
    costFuncVal += net.getL2Reg() * costFuncReg;
    // normalize after regularization term, which seems like a strange convention
    costFuncVal /= batchSize;
    return costFuncVal;
  }

  // this returns a prediction for a single data point x0
  template <typename Net>
  Vec<Net::outputs> prediction(const Net& net, const ArrayX& netvals, const Vec<Net::inputs>& x0) {
    constexpr size_t N = Net::inputs;
    constexpr size_t M = Net::outputs;
    constexpr size_t P = Net::hiddens;

    // again, these typedefs should come out of Net, not N, M, and P
    Mat<P, N+1> w1 = net.getFirstSynapses(netvals);
    Mat<M, P+1> w2 = net.getSecondSynapses(netvals);

    // a_n should just be temporary expressions
    Vec<P> a1 = w1.template leftCols<1>() // bias terms
      + w1.template rightCols<N>() * x0; // weights term
    Vec<P> x1 = ActivFunc<Net::ActType>::template activ< Array<P> >(a1.array()).matrix(); // net output of layer 1
    Vec<M> a2 = w2.template leftCols<1>()
      + w2.template rightCols<P>() * x1;
    Vec<M> x2 = Regressor<Net::RegType>::template outputGate< Array<M> >(a2.array()).matrix();
    return x2;
  }

} // namespace ceptron
