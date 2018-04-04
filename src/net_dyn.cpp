#include "net_dyn.hpp"

namespace {
  using namespace ceptron;
} // namespace

// ---- FfnDyn ----

int FfnDyn::getNumOutputs() const {
  return first_layer_->getNumEndOutputs();
}

int FfnDyn::getNumInputs() const {
  return first_layer_->getNumInputs();
}

// double FfnDyn::costFunc(const Eigen::Ref<const ArrayX>& netvals, const BatchArrayX& xin, const BatchArrayX& yin) const {
// a normal reference should be enough -- we don't need to pass in blocks of matrices at this time
double FfnDyn::costFunc(const ArrayX& netvals, const BatchVecX& xin, const BatchVecX& yin) const {
  assert( netvals.size() == num_weights() );
  const int batchSize = xin.cols();
  double totalCost = first_layer_->costFunction(netvals, xin, yin);
  return totalCost / batchSize;
}

ArrayX FfnDyn::costFuncGrad(const ArrayX& netvals, const BatchVecX& xin, const BatchVecX& yin) const {
  assert( netvals.size() == num_weights() );
  const int batchSize = xin.cols();
  // set aside memory for gradient ahead of time, then recursively fill
  ArrayX grad(num_weights());
  first_layer_->getCostFunctionGrad(netvals, xin, yin, grad);
  return grad / batchSize;
}


// ---- Layer ----

FfnDyn::Layer::Layer(InternalActivator act, RegressionType reg,
	     size_t ins, size_t outs)
  : inputs_(ins)
  , outputs_(outs)
  , num_weights_(outputs_*(inputs_+1))
  , is_output_(true)
  , act_(act)
  , reg_(reg)
{
  BOOST_LOG_TRIVIAL(trace) << "in output layer constructor with " << ins << " inputs and " << outs << " outputs.";
}

size_t FfnDyn::Layer::getNumWeightsForward() const {
  const size_t thisSize = getNumWeights();
  return is_output_ ? thisSize : thisSize + next_layer_->getNumWeightsForward();
}

size_t FfnDyn::Layer::getNumEndOutputs() const {
  return is_output_ ? outputs_ : next_layer_->getNumEndOutputs();
}

double FfnDyn::Layer::costFunction(const Eigen::Ref<const ArrayX>& netvals, const BatchVecX& xin, const BatchVecX& yin) const { // , regularization etc.)
  // const auto batchsize = xin.cols();
  // could assert that xin.cols() == yin.cols(). maybe just in slower gradient version
  VecX bias = Eigen::Map<const VecX>(netvals.segment(0,outputs_).data(), outputs_, 1);
  MatX weights = Eigen::Map<const MatX>(netvals.segment(outputs_, inputs_*outputs_).data(),
					outputs_, inputs_);
  const auto insize = netvals.size();
  BatchVecX a1 = weights*xin.matrix();// + bias;
  a1.colwise() += bias;
  if (is_output_) {
    assert( insize == getNumWeights() );
    BatchVecX x1 = outputGate(reg_, a1); // would be nice to generalize outputGate and activ. member function at initialization?
    return costFuncVal(reg_, x1, yin); // we need to add regularization terms as well
  } else {
    assert( insize > getNumWeights() );
    BatchArrayX x1 = activ(act_, a1.array());
    const Eigen::Ref<const ArrayX> remNet = netvals.segment(num_weights_, insize-num_weights_); // remaining parameters to be passed on
    return next_layer_->costFunction( remNet, x1, yin ); // add l2 regularization terms for weight matrix
  }
}

void FfnDyn::Layer::getCostFunctionGrad(const Eigen::Ref<const ArrayX>& netvals, const BatchVecX& xin, const BatchVecX& yin, /*const*/ Eigen::Ref<ArrayX>/*&*/ gradnet) const { // , regularization etc.)
  // const auto batchsize = xin.cols();
  assert ( xin.cols() == yin.cols() );
  assert ( netvals.size() == gradnet.size() );
  VecX bias = Eigen::Map<const VecX>(netvals.segment(0,outputs_).data(), outputs_, 1);
  MatX weights = Eigen::Map<const MatX>(netvals.segment(outputs_, inputs_*outputs_).data(),
					outputs_, inputs_);
  const auto insize = netvals.size();
  BatchVecX a1 = weights*xin.matrix();
  a1.colwise() += bias; // a is the output node's value before the activation gate is applied
  if (is_output_) {
    assert( insize == getNumWeights() );
    BatchVecX x1 = outputGate(reg_, a1.array()).matrix(); // would be nice to generalize outputGate and activ. member function at initialization?
    // in principle not every regression type might work out to have the bias error term be so simple, but most are constructed this way.
    BatchVecX delta = x1 - yin; // a function of output x
    gradnet.segment(0,outputs_) = delta.rowwise().sum().array(); // bias gradient
    MatX gw = delta * xin.matrix().transpose(); // add l2 regularization terms
    gradnet.segment(outputs_, inputs_*outputs_) = Map<ArrayX>(gw.data(), num_weights_);
  } else {
    assert( insize > getNumWeights() );
    BatchArrayX x1 = activ(act_, a1.array());
    BatchVecX e = activToD(act_, x1).matrix(); // seems like this could be combined with the above call
    throw std::runtime_error( "unimplemented gradient");
    // next_layer_=>getCostFunction(...);
    
    // BatchVecX delta = e.cwiseProduct( weights.transpose() * nextdelta ); // need to get nextdelta from recursive call
    
    // return next_layer_->
    //   getCostFunction( netvals.segment(num_weights_, insize-num_weights_),
    // 		       activ(act_, x1.array()), yin );
  }
}


