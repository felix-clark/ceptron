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
  struct FfnLayerDef {
    FfnLayerDef() = default;
    ~FfnLayerDef() = default;
    static constexpr size_t size = n;
    // std::function<Array<n>(const Array<n>&)> activ = ActivFunc<act>::activ;
    using ActivFunc<act>::activ;
    using ActivFunc<act>::activToD;
 };

  class FfnStaticBase {

  }; // class FfnStaticBase
  
  // perhaps should generalize InternalActivator to Activator which includes internal and output activation functions
  // this would necessitate migrating some functionality currently in the regression file into this new class.
  // cost function computation would be external (property of net only?) and we'd need more work to specialize in order to cancel common factors in e.g. logistic regression backprop
  // recursive definition
  template <size_t N /*number of inputs; need this be stored here?*/, typename layer, typename... Layers>
  class FfnLayerRec {
  public:
    static constexpr size_t Nin = N;
    static constexpr size_t Nout = layer::size;
  private:
    // a private constructor may be appropriate if we will only instantiate this layer class from within the public FfnStatic class
    FfnLayerRec() = default;
    friend FfnStaticBase;

    FfnLayerRec<Nout, Layers...> next_layer_;
    
  }; // class FfnLayer

  template <size_t Nin, size_t Nout>
  class FfnStatic : private/*public*/ FfnStaticBase {

  };
  
  
} // namespace ceptron

