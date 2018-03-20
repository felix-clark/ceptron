#include <Eigen/Dense>

namespace { // protect these definitions locally, at least for now until we come up with a better global scheme
  template <size_t M, size_t N>
  using Mat = Eigen::Matrix<double, M, N>;

  template <size_t M> // don't think we need anything but column arrays
  using Array = Eigen::Array<double, M, 1>;

  using Eigen::Map;
}

template <typename Derived> // should work for scalars and eigen arrays
Eigen::ArrayBase<Derived> sigmoid(const Eigen::Ref<const Derived>& in) {
  using Eigen::exp;
  Derived result = 1.0/(1.0 + exp(-in));
  return result;
}

double sigmoid(double in) {
  return 1.0/(1.0 + exp(-in));
}

// N is input size, M is hidden layer size; single output function
template <size_t N, size_t M>
// should inherit from some neural network interface once we get that going
class SingleHiddenLayer
{
private:
  static constexpr size_t size_ = (N+1)*M + (M+1); // total size of net
public:
  SingleHiddenLayer();
  ~SingleHiddenLayer() {};
  void randomInit() {A1_.setRandom(); A2_.setRandom();};
  double getOutput(const Mat<N, 1>&) const; // should maybe take an array instead of a vector? idk
  auto getFirstSynapses() const {return A1_;};
  auto getSecondSynapses() const {return A2_;};
  // Array<size_> getNetValue(); // if we specify directly that it is of size size_, the compiler can not associate with the function definition properly. "auto" just works, apparently.
  const auto& getNetValue() /*const*/; // operator<< doesn't work with const?
  void setNetValue(const Array<size_>&);
private:
  // at some point we might wish to experiment with column-major (default) vs. row-major data storage order
  Mat<M, N+1> A1_ = decltype(A1_)::Zero(); // maps input to hidden
  Mat<1, M+1> A2_ = decltype(A2_)::Zero(); // maps hidden to output
  // maybe it'll be easier if our only storage is the actual 1D array, then we view it as matrices when needed (rather than vice-versa)
  // perhaps there should be a member Map<> to simply convert from 1D array form to matrix storage?
  // Map< Array<size_> > netmap_; // is there a way to permanently have this point to A1 and A2? maybe not, at least safely...
};

template <size_t N, size_t M>
SingleHiddenLayer<N,M>::SingleHiddenLayer() {
  // possibly we can set things up here to go back and forth between our  matrix data and a pre-configured map.
  // however if the net gets moved, we need to be careful. possibly need custom move/copy constructors.
}

// simply return the value of the network for a given input. we'll use a simple sigmoid activation function for now (make this adjustable later)
template <size_t N, size_t M>
double SingleHiddenLayer<N,M>::getOutput(const Mat<N, 1>& x0) const {
  const Mat<M, 1> y1 = A1_.template leftCols<1>() // bias term
    + A1_.template rightCols<N>() * x0; // matrix mult term
  const Mat<M, 1> x1 = sigmoid< Array<M> >(y1.array()).matrix(); // apply activation function element-wise; need to specify the template type
  Mat<1,1> y2 = A2_.template leftCols<1>()
    + A2_.template rightCols<M>() * x1;

  return sigmoid(y2(0,0)); // this version is outputting a scalar; just access it directly
}

// arranges synapse values into a 1D array for use in gradient descent.
template <size_t N, size_t M>
const auto& SingleHiddenLayer<N,M>::getNetValue() /*const*/ {
  // this seems inefficient -- isn't this memory getting copied over completely?
  Array<size_> output;
  output.setZero(); // until we can verify that all elements are getting "copied" properly
  output << Map< const Array<M*(N+1)> >(A1_.data())
    , Map< const Array<M+1> >(A2_.data());
  return output.array();
}

template <size_t N, size_t M>
void SingleHiddenLayer<N,M>::setNetValue(const Array<size_>& in) {
  
}
