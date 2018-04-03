#pragma once
#include "global.hpp"
#include "activation.hpp"
#include "regression.hpp"

#include <boost/log/trivial.hpp>

#include <vector>
#include <memory>

namespace {
  using namespace ceptron;
  
} // namespace

class Layer;
// class OutputLayer;

// a runtime-sized feedforward network
// not sure  yet that we actually need a separate class for this
class FfnDyn
{
public:
  template <typename...Ts> FfnDyn(RegressionType reg, InternalActivator act, Ts... layer_sizes);
  // to correctly unpack variadic parameter packs we need to use templates, which it would be nice to avoid for these static cases.
  ~FfnDyn() = default;
  FfnDyn(const FfnDyn&) = delete; // we do some non-trivial memory operations w/ shared_ptrs so don't allow copying (not sure if it works)
  // if we make this functional so that it doesn't own the data, we might not have to worry about copying
  // FfnDyn(FfnDyn&&) = delete; // move ops are not defined when we implement or delete the copy-constructor
  // might need to disable both assignment and move varieties of operator= too?
  // although if we define it so make the net sizes and values the same then that might be fine with this organization
  int num_weights() const {return size_;} // number of weights in net (size of gradient)
  size_t getNumOutputs() const;
  double costFunc(const Eigen::Ref<const ArrayX>& netvals, const BatchArrayX& xin, const BatchArrayX& yin) const;
private:
  // ArrayX net_; // we probably shouldn't save data internally.
  // std::vector<Layer> layers_; // and perhaps we need only point to the first layer
  // these pointers could probably be unique for simple feed-forward nets
  std::shared_ptr<Layer> first_layer_;
  size_t size_; // saving this may be redundant, but useful for quick assertions
};

// a "Layer" here refers to the structure that includes the weight matrix taking "inputs_" values and mapping to "outputs_" values, with an activation function on top.
// these layers should possibly all be non-exposed classes used only by the network
class Layer
{
public:
   // constructor for output layer
  Layer(InternalActivator act, RegressionType reg, size_t ins, size_t outs);
  // constructor for internal
  template <typename ...Ts> Layer(InternalActivator act, RegressionType reg, size_t ins, size_t n1, size_t n2, Ts... sizes);
  virtual ~Layer() = default;

  double getCostFunction(const Eigen::Ref<const ArrayX>& net, const BatchArrayX& xin, const BatchArrayX& yin) const;
  
  size_t getNumInputs() const {return inputs_;}
  size_t getNumOutputs() const {return outputs_;}
  size_t getNumEndOutputs() const;
  int getNumWeights() const {return outputs_*(inputs_+1);}
  size_t getNumWeightsForward() const; // recursively compute the total number of weight elements including this layer and all in front
protected:
  size_t inputs_;
  size_t outputs_;
private:
  // the output layer won't need to access these. in fact they will be somewhat wasted memory.
  // if it was significant we could split this into InputLayer, but that's probably not a big deal.
  
  bool is_output_ = false; // is this the last layer in the net?
  // Map<VecX> bias_;// map values can be passed in at call time; don't save as data
  // Map<MatX> weights_;
  InternalActivator act_;
  RegressionType reg_; // it's redundant saving this in every layer but it makes things easier for now.
  // we could do a dynamic check on sizeof...(others) to specialize Layer w/ a separate OutputLayer class
  std::shared_ptr<Layer> next_layer_;
  // is a pointer to the last layer useful for anything? seems we can get most everything recursively
};


template <typename... Ts> FfnDyn::FfnDyn(RegressionType reg, InternalActivator act, Ts... layer_sizes)
{
  static_assert( sizeof...(layer_sizes) > 1, "network needs more than a single layer to be meaningful" );
  first_layer_ = std::make_shared<Layer>(act, reg, layer_sizes...);
  size_ = first_layer_->getNumWeightsForward();
}


// could maybe switch order of parameters w/ a template, but it's probably not worth it
template <typename...Ts>
Layer::Layer(InternalActivator act, RegressionType reg,
	     size_t ins, size_t n1, size_t n2,
	     Ts... others)
  : inputs_(ins)
  , is_output_(false)
  // , bias_(data.data(), outs)
  // , weights_(data.segment(outs, ins*outs).data(), outs, ins)
  , act_(act)
  , reg_(reg)
{
  // static_assert( sizeof...(Ts) > 0, "this should only get called for intermediate layers" );
  next_layer_ = std::make_shared<Layer>(act, reg, n1, n2, others...);
  outputs_ = n1;
  // outputs_ = next_layer_->getNumInputs(); // this should give us the first element of others...
	  
  BOOST_LOG_TRIVIAL(debug) << "in intermediate layer constructor with " << inputs_ << " inputs and " << outputs_ << " outputs.";
}

