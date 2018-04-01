#pragma once
#include "global.hpp"

#include "regression.hpp"
#include "activation.hpp"
#include <boost/log/trivial.hpp>
#include <vector>

namespace {
  using namespace ceptron;
  
} // namespace

// a runtime-sized feedforward network
class FfnDyn
{
public:
  FfnDyn(size_t ins, size_t outs); // right now just have a testing constructor w/ a single layer
  ~FfnDyn() = default;
  FfnDyn(const FfnDyn&) = delete; // we do some non-trivial memory address operations so don't allow copying
  // FfnDyn(FfnDyn&&) = delete; // or move-ops. actually this is not defined when we implement or delete the copy-constructor
  // might need to disable both assignment and move varieties of operator= too?
  // although if we define it so make the net sizes and values the same then that might be fine with this organization
private:
  size_t size_; // saving this may be redundant
  ArrayX net_;
  std::vector<Layer> layers_;
};

/// move these defs to a cpp source file once we get off the ground
FfnDyn::FfnDyn(size_t ins, size_t outs)
  : size_(outs*(ins+1))
  , net_(size_)
{
  layers_.push_back(Layer(ins, outs, InternalActivator::Tanh, net_));
}


// these layers should possibly all be non-exposed classes used only by the network
class Layer
{
public:
  Layer(size_t ins, size_t outs, InternalActivator act, Eigen::Ref<ArrayX> data);
  ~Layer() = default;
private:
  // should the data be held externally, or should this have its own
  Map<VecX> bias_;
  Map<MatX> weights_;
  InternalActivator act_;
};

Layer::Layer(size_t ins, size_t outs, InternalActivator act, Eigen::Ref<ArrayX> data)
  : act_(act)
  , bias_(data.data(), outs)
  , weights_(data.segment(outs, ins*outs).data(), outs, ins)
{
    
}

// perhaps this shouldn't be a separate class and we should combine internal activators w/ output ones.
//  output layers may need only a single additional function for summing up the cost function
class OutputLayer
{
public:
  OutputLayer(size_t ins, size_t outs, RegressionType reg);
  ~OutputLayer() = default;
private:
}
