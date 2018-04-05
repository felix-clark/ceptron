#pragma once
#include "global.hpp"
#include "activation.hpp"
#include "regression.hpp"

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
    int numInputs() const;
    int numOutputs() const;

    // set an l2 regularization parameter, which adds a sum-of-squares of weight terms (but not biases) to the cost function.
    void setL2Reg(double lambda); // technically could be declared const, since only the layers will care
    
    // convenience function to return a parameter array with weights (but not biases) randomized,
    //  as a suggested initialization. numbers are scaled to have a variance that does not depend on number of inputs.
    //  use std::srand before calling to set the RNG seed.
    ArrayX randomWeights() const;

    // these methods are the workhorses of the NN, and are implemented recursively on the layers
    double costFunc(const ArrayX& netvals, const BatchVecX& xin, const BatchVecX& yin) const;
    ArrayX costFuncGrad(const ArrayX& netvals, const BatchVecX& xin, const BatchVecX& yin) const;
    VecX prediction(const ArrayX& netvals, const VecX& xin) const;
  private:
    class Layer;// we need to forward-declare the class to declare the smart ptr
    // this is only a simple feed-forward net, so unique pointers are sufficient.
    std::unique_ptr<Layer> first_layer_;
    size_t size_=0; // saving this may be redundant, but useful for quick assertions
    
    // ---- recursive definition of layer
    
    // a "Layer" here refers to the structure that includes the weight matrix taking "inputs_" values and mapping to "outputs_" values, with an activation function on top.
    class Layer
    {
    public:
      // constructor for output layer
      Layer(InternalActivator act, RegressionType reg, size_t ins, size_t outs);
      // recursive constructor for internal layers
      template <typename ...Ts> Layer(InternalActivator act, RegressionType reg, size_t ins, size_t n1, size_t n2, Ts... sizes);
      virtual ~Layer() = default;

      void setL2RegRecurse(double l);
      
      ArrayX randParsRecurse(int) const;
      // recursive calls to retrieve cost function, gradient, prediction
      double costFuncRecurse(const Eigen::Ref<const ArrayX>& net, const BatchVecX& xin, const BatchVecX& yin) const;
      // returns the matrix needed for backprop
      // takes a non-const reference to fill the gradient with (hence "get" in the function name)
      MatX getCostFuncGradRecurse(const Eigen::Ref<const ArrayX>& net, const BatchVecX& xin, const BatchVecX& yin, Eigen::Ref<ArrayX> gradnet) const;
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
      RegressionType reg_; // it's redundant saving this in every layer but it makes things easier for now.
      // perhaps we don't have to store both the above if we have this activation function stored:
      std::function<BatchArrayX(BatchArrayX)> activation_func_; // we could similarly store the cost function calculation for the output layer only, so we don't have to save reg_.
      std::function<BatchArrayX(BatchArrayX)> activ_to_d_func_;
      
      std::unique_ptr<Layer> next_layer_;
      // is a pointer to the last layer useful for anything? seems we can get most everything recursively

      // l2 regularization parameter
      double l2_lambda_=0.;
      
      // TODO: dropout probability: probability that a node is zeroed out for a calculation.
      // when getting cost function and gradient at the same time we need to make the masks identical
      // this might play poorly w/ calculations that require multiple calculations of the objective function (contour tracing)
      // we need to scale the values of neurons by (1-p) when extracting predictions
      // double dropout_p_=0.;
    
    }; // class Layer

  }; // class FfnDyn


  template <typename... Ts> FfnDyn::FfnDyn(RegressionType reg, InternalActivator act, Ts... layer_sizes)
  {
    static_assert( sizeof...(layer_sizes) > 1, "network needs more than a single layer to be meaningful" );
    first_layer_ = std::make_unique<Layer>(act, reg, layer_sizes...);
    size_ = first_layer_->getNumWeightsRecurse();
  }


  // could maybe switch order of parameters w/ a template, but it's probably not worth it
  template <typename...Ts>
  FfnDyn::Layer::Layer(InternalActivator act, RegressionType reg,
		       size_t ins, size_t n1, size_t n2,
		       Ts... others)
    : inputs_(ins)
    , is_output_(false)
    , reg_(reg)
  {
    next_layer_ = std::make_unique<Layer>(act, reg, n1, n2, others...);
    outputs_ = n1;
    num_weights_ = outputs_*(inputs_+1);

    activation_func_ = std::bind(activ, act, std::placeholders::_1);
    activ_to_d_func_ = std::bind(activToD, act, std::placeholders::_1);
  }

} // namespace ceptron
