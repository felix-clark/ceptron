#include "ffn_dyn.hpp"

namespace {
using namespace ceptron;
}  // namespace

// ---- FfnDyn ----

int FfnDyn::numOutputs() const { return first_layer_->numEndOutputs(); }

int FfnDyn::numInputs() const { return first_layer_->numInputs(); }

void FfnDyn::setL2Reg(scalar l) { first_layer_->setL2RegRecurse(l); }

// scalar FfnDyn::costFunc(const Eigen::Ref<const ArrayX>& netvals, const
// BatchArrayX& xin, const BatchArrayX& yin) const {
// a normal reference should be enough -- we don't need to pass in blocks of
// matrices at this time
scalar FfnDyn::costFunc(const ArrayX& netvals, const BatchVecX& xin,
                        const BatchVecX& yin) const {
  assert(netvals.size() == num_weights());
  const int batchSize = xin.cols();
  scalar totalCost = first_layer_->costFuncRecurse(netvals, xin, yin);
  return totalCost / batchSize;
}

ArrayX FfnDyn::costFuncGrad(const ArrayX& netvals, const BatchVecX& xin,
                            const BatchVecX& yin) const {
  assert(netvals.size() == num_weights());
  const int batchSize = xin.cols();
  // set aside memory for gradient ahead of time, then recursively fill
  ArrayX grad(num_weights());
  first_layer_->getCostFuncGradRecurse(netvals, xin, yin, grad);
  return grad / batchSize;
}

VecX FfnDyn::operator()(const ArrayX& netvals, const VecX& xin) const {
  return first_layer_->predictRecurse(netvals, xin);
}

ArrayX FfnDyn::randomWeights() const {
  return first_layer_->randParsRecurse(num_weights());
}

// ---- Layer ----

FfnDyn::Layer::Layer(InternalActivator, RegressionType reg, size_t ins,
                     size_t outs)
    : inputs_(ins),
      outputs_(outs),
      num_weights_(outputs_ * (inputs_ + 1)),
      is_output_(true)
      // , act_(act)
      ,
      reg_(reg) {
  activation_func_ = std::bind(outputGate, reg, std::placeholders::_1);
}

size_t FfnDyn::Layer::numWeightsRecurse() const {
  const size_t thisSize = numWeights();
  return is_output_ ? thisSize : thisSize + next_layer_->numWeightsRecurse();
}

size_t FfnDyn::Layer::numEndOutputs() const {
  return is_output_ ? outputs_ : next_layer_->numEndOutputs();
}

void FfnDyn::Layer::setL2RegRecurse(scalar l) {
  l2_lambda_ = l;
  if (!is_output_) next_layer_->setL2RegRecurse(l);
}

ArrayX FfnDyn::Layer::randParsRecurse(int pars_left) const {
  if (pars_left <= 0) {
    throw std::runtime_error(
        "logic error in recursive call to randomize weight parameters");
  }
  // set to zeros
  ArrayX pars = ArrayX::Zero(num_weights_);
  // weights need to be scaled by the inverse sqrt of the number of inputs,
  //  or else the variance of the matrix multiplication will scale with size of
  //  the previous layer
  // pars.segment(outputs_, inputs_*outputs_) =
  // ArrayX::Random(inputs_*outputs_)/sqrt(inputs_);
  // Glorot and Bengio suggest the following instead (helpful for both
  // directions of propagation):
  pars.segment(outputs_, inputs_ * outputs_) =
      ArrayX::Random(inputs_ * outputs_) * sqrt(6.0 / (inputs_ + outputs_));
  if (is_output_) {
    assert(static_cast<int>(num_weights_) == pars_left);
    return pars;
  }
  ArrayX combPars(pars_left);
  // this method of initialization is likely not the most efficient, but this
  // function
  //   should only be called ~once per initialization so it shouldn't be a
  //   bottleneck.
  combPars << pars, next_layer_->randParsRecurse(pars_left - num_weights_);
  return combPars;
}

scalar FfnDyn::Layer::costFuncRecurse(
    const Eigen::Ref<const ArrayX>& netvals, const BatchVecX& xin,
    const BatchVecX& yin) const {  // , regularization etc.)
  // const auto batchsize = xin.cols();
  // could assert that xin.cols() == yin.cols(). maybe just in slower gradient
  // version
  VecX bias =
      Eigen::Map<const VecX>(netvals.segment(0, outputs_).data(), outputs_, 1);
  MatX weights = Eigen::Map<const MatX>(
      netvals.segment(outputs_, inputs_ * outputs_).data(), outputs_, inputs_);
  const auto insize = netvals.size();
  BatchVecX a1 = weights * xin.matrix();  // + bias;
  a1.colwise() += bias;

  scalar regTerm = l2_lambda_ * weights.array().square().sum();
  BatchVecX x1 = activation_func_(a1);
  if (is_output_) {
    assert(insize == numWeights());
    // BatchVecX x1 = outputGate(reg_, a1); // would be nice to generalize
    // outputGate and activ. member function at initialization?
    return regTerm + costFuncVal(reg_, x1, yin);
  } else {
    assert(insize > numWeights());
    // BatchArrayX x1 = activ(act_, a1.array());
    const Eigen::Ref<const ArrayX>& remNet = netvals.segment(
        num_weights_,
        insize - num_weights_);  // remaining parameters to be passed on
    return regTerm + next_layer_->costFuncRecurse(remNet, x1, yin);
  }
}

MatX FfnDyn::Layer::getCostFuncGradRecurse(
    const Eigen::Ref<const ArrayX>& netvals, const BatchVecX& xin,
    const BatchVecX& yin, Eigen::Ref<ArrayX> gradnet)
    const {  // , regularization etc.)
  // const auto batchsize = xin.cols();
  assert(xin.cols() == yin.cols());
  assert(netvals.size() == gradnet.size());
  VecX bias =
      Eigen::Map<const VecX>(netvals.segment(0, outputs_).data(), outputs_, 1);
  MatX weights = Eigen::Map<const MatX>(
      netvals.segment(outputs_, inputs_ * outputs_).data(), outputs_, inputs_);
  const auto insize = netvals.size();
  BatchVecX a1 = weights * xin;
  a1.colwise() += bias;  // a is the output node's value before the activation
                         // gate is applied

  BatchVecX x1 = activation_func_(a1.array()).matrix();
  BatchVecX delta = BatchVecX::Zero(x1.cols(), x1.rows());
  if (is_output_) {
    assert(insize == numWeights());
    // in principle not every regression type might work out to have the bias
    // error term be so simple, but most are constructed this way.
    // in general this could be a non-trivial function that we will need to get
    // from the RegressionType.
    delta = x1 - yin;  // a function of output x
  } else {
    assert(insize > numWeights());
    BatchVecX e = activ_to_d_func_(x1.array()).matrix();
    const Eigen::Ref<const ArrayX>& nextNet =
        netvals.segment(num_weights_, insize - num_weights_);
    Eigen::Ref<ArrayX> nextGrad =
        gradnet.segment(num_weights_, insize - num_weights_);
    // recursive part: we need the W^T * delta of the next layer to get the
    // delta for this layer
    delta = e.cwiseProduct(
        next_layer_->getCostFuncGradRecurse(nextNet, x1, yin, nextGrad));
  }
  MatX gw = delta * xin.transpose() + 2.0 * l2_lambda_ * weights;
  gradnet.segment(0, outputs_) =
      delta.rowwise().sum().array();  // bias gradient
  gradnet.segment(outputs_, inputs_ * outputs_) =
      Map<ArrayX>(gw.data(), inputs_ * outputs_);
  return weights.transpose() * delta;
}

VecX FfnDyn::Layer::predictRecurse(const Eigen::Ref<const ArrayX>& netvals,
                                   const VecX& xin) const {
  VecX bias =
      Eigen::Map<const VecX>(netvals.segment(0, outputs_).data(), outputs_, 1);
  MatX weights = Eigen::Map<const MatX>(
      netvals.segment(outputs_, inputs_ * outputs_).data(), outputs_, inputs_);
  const auto insize = netvals.size();
  BatchVecX a1 = weights * xin;
  a1.colwise() += bias;
  BatchVecX x1 = activation_func_(a1.array()).matrix();
  if (is_output_) {
    assert(insize == numWeights());
    return x1;
  }
  // we are in an internal layer
  assert(insize > numWeights());
  const Eigen::Ref<const ArrayX> remNet = netvals.segment(
      num_weights_,
      insize - num_weights_);  // remaining parameters to be passed on
  return next_layer_->predictRecurse(
      remNet, x1);  // add l2 regularization terms for weight matrix
}
