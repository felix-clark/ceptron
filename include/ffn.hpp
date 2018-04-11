#pragma once
// the static (compile-time) version of an arbitrary-size FFNN
#include "global.hpp"
#include "activation.hpp"
#include "regression.hpp"
#include "log.hpp"
#include <Eigen/Dense>

namespace ceptron {

  // a layer should be able to be defined solely by its size and its activation function.
  //  internally it may need to know how many elements are in the layer in front or back of it,
  //   but its definition should require only these two.
  template <size_t n, InternalActivator act>
  struct FfnLayerDef : public ActivFunc<act> {
    FfnLayerDef() = default;
    ~FfnLayerDef() = default;
    static constexpr size_t size = n;
    // import activation functions and derivatives
    // std::function<Array<n>(const Array<n>&)> activ = ActivFunc<act>::activ;
    using ActivFunc<act>::activ;
    using ActivFunc<act>::activToD;
 };

  // create a different type which specializes output layers
  template <size_t n, RegressionType reg>
  struct FfnOutputLayerDef : public Regressor<reg> {
    FfnOutputLayerDef() = default;
    ~FfnOutputLayerDef() = default;
    static constexpr size_t size = n;
    // import activation functions and derivatives
    // pull regression functions into local class namespace
    using Regressor<reg>::outputGate;
    using Regressor<reg>::costFuncVal; // so far we don't require a special derivative function
 };

  // template <typename Derived>
  // class FfnStaticBase {

  // }; // class FfnStaticBase

  template <size_t NetIn, typename... Layers>
  class FfnStatic /*: privatepublic FfnStaticBase<FfnStatic<Nin, Layers...>>*/ {
    static_assert( sizeof...(Layers) > 0,
    		   "a FFNN needs more than an input layer" );
  private:
    // forward-declare recursive template
    template <size_t Nl, typename L1, typename... Rest> class LayerRec;
    LayerRec<NetIn, Layers...> first_layer_;
  

    // perhaps should generalize InternalActivator to Activator which includes internal and output activation functions
    // this would necessitate migrating some functionality currently in the regression file into this new class.
    // cost function computation would be external (property of net only?) and we'd need more work to specialize in order to cancel common factors in e.g. logistic regression backprop
    // recursive definition
    template <size_t N0,
	      typename L0, typename L1, typename... Rest>
    class LayerRec<N0, L0, L1, Rest...> : public L0 {
    public:
      static constexpr size_t Nin = N0;
      static constexpr size_t Nout = L0::size;
      // LayerRec() = default;
    private:
      // a private constructor may be appropriate if we will only instantiate this layer class from within the public FfnStatic class
      
      using L0::activ;
      using L0::activToD;

      LayerRec<Nout, L1, Rest...> next_layer_;
    
    }; // class LayerRec

    // end case for recursive layer: output layer will have RegressionType instead of InternalActivator
    template <size_t N, typename L0>
    class LayerRec<N, L0> : public L0 {
    public:
      static constexpr size_t Nin = N;
      static constexpr size_t Nout = L0::size;
      // L0 should have type FfnOutputLayerDef
    private:
    
      // there is no next layer. we'll define functions based on the regression type.
      using L0::outputGate; // we'd need to inherit from L0 (which is an FfnOutputLayerDef)
      using L0::costFuncVal;
    }; // class LayerRec
    
  };
  
  
} // namespace ceptron

