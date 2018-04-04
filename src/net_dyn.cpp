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
  const auto batchsize = xin.cols();
  // could assert that xin.cols() == yin.cols(). maybe in gradient version
  BatchVecX bias =
    Eigen::Map<const VecX>(netvals.segment(0,outputs_).data(), outputs_, 1)
    * BatchVecX::Ones(1,batchsize);
  MatX weights = Eigen::Map<const MatX>(netvals.segment(outputs_, inputs_*outputs_).data(), outputs_, inputs_);
  const int thissize = outputs_*(inputs_+1); // this could be computed/saved ahead of time, like at construction
  const auto insize = netvals.size();
  BatchVecX x1 = weights*xin.matrix() + bias;
  if (is_output_) {
    assert( insize == getNumWeights() );
    return costFuncVal(reg_, xin, yin);
  } else {
    assert( insize > getNumWeights() );
    return next_layer_->
      getCostFunction( netvals.segment(thissize, insize-thissize),
		       activ(act_, x1.array()), yin );
  }
}


