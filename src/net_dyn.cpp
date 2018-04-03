#include "net_dyn.hpp"

namespace {
  using namespace ceptron;
} // namespace

// ---- FfnDyn ----

size_t FfnDyn::getNumOutputs() const {
  return first_layer_->getNumEndOutputs();
}

double FfnDyn::costFunc(const Eigen::Ref<const ArrayX>& netvals, const BatchArrayX& xin, const BatchArrayX& yin) const {
  assert( netvals.size() == num_weights() );
  return first_layer_->getCostFunction(netvals, xin, yin);
}

// ---- Layer ----

// template <>
Layer::Layer(InternalActivator act, RegressionType reg,
	     size_t ins, size_t outs)
  : inputs_(ins)
  , outputs_(outs)
  , is_output_(true)
  // , bias_(data.data(), outs)
  // , weights_(data.segment(outs, ins*outs).data(), outs, ins)
  , act_(act)
  , reg_(reg)
{
  // do something with reg to save it. possibly point to appropriate functions.
  BOOST_LOG_TRIVIAL(debug) << "in output layer constructor with " << ins << " inputs and " << outs << " outputs.";
}

size_t Layer::getNumWeightsForward() const {
  const size_t thisSize = getNumWeights();
  return is_output_ ? thisSize : thisSize + next_layer_->getNumWeightsForward();
}

size_t Layer::getNumEndOutputs() const {
  return is_output_ ? outputs_ : next_layer_->getNumEndOutputs();
}

double Layer::getCostFunction(const Eigen::Ref<const ArrayX>& netvals, const BatchArrayX& xin, const BatchArrayX& yin) const { // , regularization etc.) 
  ArrayX bias = netvals.segment(0,outputs_);
  MatX weights = Eigen::Map<const MatX>(netvals.segment(outputs_, inputs_*outputs_).data(), outputs_, inputs_);
  if (is_output_) {
    assert( netvals.size() == getNumWeights() );
  } else {
    assert( netvals.size() > getNumWeights() );
  }
}


