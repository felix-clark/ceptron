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
  static constexpr size_t size_w1_ = M*(N+1); // first layer
  static constexpr size_t size_w2_ = M+1; // output layer
  static constexpr size_t size_ = size_w1_ + size_w2_; // total size of net
public:
  SingleHiddenLayer() {};
  ~SingleHiddenLayer() {};
  void randomInit() {net_.setRandom();};
  double getOutput(const Mat<N, 1>&) const; // should maybe take an array instead of a vector? idk
  // auto getFirstSynapses() const {return A1_;};
  // using "auto" may not be best when returning a map view -- TODO: look into this more.
  auto getFirstSynapses() const {return Map< const Mat<M, N+1> >(net_.data());};
  auto getSecondSynapses() const
  {return Map< const Mat<1, M+1> >(net_.template segment<size_w2_>(size_w1_).data());};
  // Array<size_> getNetValue(); // if we specify directly that it is of size size_, the compiler can not associate with the function definition properly. "auto" just works, apparently.
  const auto& getNetValue() const {return net_;}
  void setNetValue(const Array<size_>& in) {net_ = in;}
private:
  Array<size_> net_ = Array<size_>::Zero();  

};


// simply return the value of the network for a given input. we'll use a simple sigmoid activation function for now (make this adjustable later)
template <size_t N, size_t M>
double SingleHiddenLayer<N,M>::getOutput(const Mat<N, 1>& x0) const {
  // at some point we might wish to experiment with column-major (default) vs. row-major data storage order
  const Mat<M, N+1>& A1 = getFirstSynapses();
  const Mat<1, M+1>& A2 = getSecondSynapses();
  const Mat<M, 1> y1 = A1.template leftCols<1>() // bias term
    + A1.template rightCols<N>() * x0; // matrix mult term
  const Mat<M, 1> x1 = sigmoid< Array<M> >(y1.array()).matrix(); // apply activation function element-wise; need to specify the template type
  Mat<1,1> y2 = A2.template leftCols<1>()
    + A2.template rightCols<M>() * x1;

  return sigmoid(y2(0,0)); // this version is outputting a scalar; just access it directly
}

