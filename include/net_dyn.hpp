#pragma once
#include "global.hpp"
#include "activation.hpp"
#include "regression.hpp"

#include <boost/log/trivial.hpp>

#include <vector>
#include <memory>

namespace ceptron {
  // a runtime-sized feedforward network
  // not sure  yet that we actually need a separate class for this as opposed to using first-layer only,
  //  but trimming down the interface is probably worth it.
  class FfnDyn
  {
  public:
    template <typename...Ts> FfnDyn(RegressionType reg, InternalActivator act, Ts... layer_sizes);
    ~FfnDyn() = default;
    FfnDyn(const FfnDyn&) = delete; // we do some non-trivial memory operations w/ shared_ptrs so don't allow copying (not sure if it works)
    // FfnDyn(FfnDyn&&) = delete; // move ops are not defined when we implement or delete the copy-constructor
    // might need to disable operator= too?
    int num_weights() const {return size_;} // number of weights in net (size of gradient)
    int getNumInputs() const;
    int getNumOutputs() const;
    ArrayX randomWeights() const; // returns a parameter array with weights (but not biases) randomized, as a suggested initialization
    double costFunc(const ArrayX& netvals, const BatchVecX& xin, const BatchVecX& yin) const;
    ArrayX costFuncGrad(const ArrayX& netvals, const BatchVecX& xin, const BatchVecX& yin) const;
    VecX prediction(const ArrayX& netvals, const VecX& xin) const;
  private:
    class Layer;// we need to forward-declare the class
    // these pointers could probably be unique for simple feed-forward nets
    std::shared_ptr<Layer> first_layer_;
    size_t size_; // saving this may be redundant, but useful for quick assertions

    // a "Layer" here refers to the structure that includes the weight matrix taking "inputs_" values and mapping to "outputs_" values, with an activation function on top.
    class Layer
    {
    public:
      // constructor for output layer
      Layer(InternalActivator act, RegressionType reg, size_t ins, size_t outs);
      // recursive constructor for internal layers
      template <typename ...Ts> Layer(InternalActivator act, RegressionType reg, size_t ins, size_t n1, size_t n2, Ts... sizes);
      virtual ~Layer() = default;

      ArrayX randParsRecurse(int) const;
      // recursive calls to retrieve cost function, gradient, prediction
      double costFuncRecurse(const Eigen::Ref<const ArrayX>& net, const BatchVecX& xin, const BatchVecX& yin) const;
      // returns the matrix needed for backprop
      // takes a non-const reference to fill the gradient with (hence "get" in the function name)
      MatX getCostFuncGradRecurse(const Eigen::Ref<const ArrayX>& net, const BatchVecX& xin, const BatchVecX& yin, /*const*/ Eigen::Ref<ArrayX>/*&*/ gradnet) const;
      VecX predictRecurse(const Eigen::Ref<const ArrayX>& net, const VecX& xin) const;
      
      size_t getNumInputs() const {return inputs_;}
      size_t getNumOutputs() const {return outputs_;}
      size_t getNumEndOutputs() const;
      int getNumWeights() const {return outputs_*(inputs_+1);}
      size_t getNumWeightsRecurse() const; // recursively compute the total number of weight elements including this layer and all in front
    private:
      size_t inputs_;
      size_t outputs_;
      size_t num_weights_;
      // the output layer won't need to access these. in fact they will be somewhat wasted memory.
      // if it was significant we could split this into InputLayer, but that's probably not a big deal.
    
      bool is_output_ = false; // is this the last layer in the net?
      InternalActivator act_;
      RegressionType reg_; // it's redundant saving this in every layer but it makes things easier for now.
      // we could do a dynamic check on sizeof...(others) to specialize Layer w/ a separate OutputLayer class
      std::shared_ptr<Layer> next_layer_;
      // is a pointer to the last layer useful for anything? seems we can get most everything recursively
    }; // class Layer

  }; // class FfnDyn


  template <typename... Ts> FfnDyn::FfnDyn(RegressionType reg, InternalActivator act, Ts... layer_sizes)
  {
    static_assert( sizeof...(layer_sizes) > 1, "network needs more than a single layer to be meaningful" );
    first_layer_ = std::make_shared<Layer>(act, reg, layer_sizes...);
    size_ = first_layer_->getNumWeightsRecurse();
  }


  // could maybe switch order of parameters w/ a template, but it's probably not worth it
  template <typename...Ts>
  FfnDyn::Layer::Layer(InternalActivator act, RegressionType reg,
		       size_t ins, size_t n1, size_t n2,
		       Ts... others)
    : inputs_(ins)
    , is_output_(false)
    , act_(act)
    , reg_(reg)
  {
    // static_assert( sizeof...(Ts) > 0, "this should only get called for intermediate layers" ); // this assert is no longer necessary
    next_layer_ = std::make_shared<Layer>(act, reg, n1, n2, others...);
    outputs_ = n1;
    // outputs_ = next_layer_->getNumInputs(); // this would also give us the first element of others...
    num_weights_ = outputs_*(inputs_+1);
    BOOST_LOG_TRIVIAL(trace) << "in intermediate layer constructor with " << inputs_ << " inputs and " << outputs_ << " outputs.";
  }

} // namespace ceptron
