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

double FfnDyn::Layer::getCostFunction(const Eigen::Ref<const ArrayX>& netvals, const BatchArrayX& xin, const BatchArrayX& yin) const { // , regularization etc.)
  // const auto batchsize = xin.cols();
  // could assert that xin.cols() == yin.cols(). maybe just in slower gradient version
  VecX bias = Eigen::Map<const VecX>(netvals.segment(0,outputs_).data(), outputs_, 1);
  MatX weights = Eigen::Map<const MatX>(netvals.segment(outputs_, inputs_*outputs_).data(),
					outputs_, inputs_);
  const auto insize = netvals.size();
  BatchVecX x1 = weights*xin.matrix();// + bias;
  x1.colwise() += bias;
  if (is_output_) {
    assert( insize == getNumWeights() );
    return costFuncVal(reg_, x1, yin);
  } else {
    assert( insize > getNumWeights() );
    return next_layer_->
      getCostFunction( netvals.segment(num_weights_, insize-num_weights_),
		       activ(act_, x1.array()), yin );
  }
}


