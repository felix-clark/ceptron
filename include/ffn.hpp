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
    // we don't actually need to do this w/ inheritance, and in fact it probably 
    //  requires less header code to remove them.
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

  // ---- FfnStatic ----
  // a template class defining a static-size feedforward net of arbitrarily many layers.
  template <size_t N, typename... Layers>
  class FfnStatic {
    static_assert( sizeof...(Layers) > 0,
    		   "a FFNN needs more than an input layer" );
  private:
    // forward-declare recursive template so that we can specialize later
    template <size_t N_, typename L1_, typename... Rest_> class LayerRec;

    // we need to define a traits class to make a static interface for the layer helper classes,
    //  since the base case needs access to these traits but they aren't yet defined when the derived classes are defined
    template <typename Derived> struct LayerTraits;
    template <size_t N_, typename L1_> struct LayerTraits < LayerRec<N_,L1_> > {
      static constexpr size_t Nin = N_;
      static constexpr size_t Nout = L1_::size;
      static constexpr size_t Nfinal = Nout; // this specialization is for the output layer
    };
    template <size_t N_, typename L1_, typename L2_, typename... Rest_> struct LayerTraits < LayerRec<N_,L1_,L2_,Rest_...> > {
      static constexpr size_t Nin = N_;
      static constexpr size_t Nout = L1_::size;
      using next_layer_t = LayerRec<Nout, L2_, Rest_...>;
      // static constexpr size_t Nfinal = LayerTraits< typename LayerRec<N_,L1_,L2_,Rest_...>::next_layer_t >::Nfinal;
      static constexpr size_t Nfinal = LayerTraits< next_layer_t >::Nfinal;
    };

    // static interface class for Layer classes
    template <typename Derived> struct LayerBase {
    private:
      // we only have one base class so we can just define this helper function in-class instead of making an additional helper class.
      // will we need a const version?
      // Derived& derived() const {return *static_cast<Derived*>(this);}
      Derived& derived() const {return static_cast<Derived>(*this);}
    protected:
      // it will be convenient to export these into derived classes,
      //  but they will need to be "used" explicitly
      static constexpr size_t Nin = LayerTraits<Derived>::Nin;
      static constexpr size_t Nout = LayerTraits<Derived>::Nout;
      static constexpr size_t Nfinal = LayerTraits<Derived>::Nfinal;
    public:
      scalar costFuncRecurse(const Eigen::Ref<const ArrayX>& net, const BatchVec<Nin>& xin, const BatchVec<Nfinal>& yin) const {
	// return static_cast< Derived* >(this)->costFuncRecurse(net, xin, yin);
	return derived().costFuncRecurse(net, xin, yin);
      }
      Vec<Nfinal> predictRecurse(const Eigen::Ref<const ArrayX>& net, const Vec<Nin>& xin) const {
	return derived()->predictRecurse(net, xin);
      }
    }; // class LayerBase
  
    // perhaps should generalize InternalActivator to Activator which includes internal and output activation functions
    // this would necessitate migrating some functionality currently in the regression file into this new class.
    // cost function computation would be external (property of net only?) and we'd need more work to specialize in order to cancel common factors in e.g. logistic regression backprop
    // recursive definition
    template <size_t N_, typename L0_, typename L1_, typename... Rest_>
    class LayerRec<N_, L0_, L1_, Rest_...>
      : public LayerBase< LayerRec<N_, L0_, L1_, Rest_...> >
      , public L0_ // this direct inheritance is unnecessary, since we can access the static functions of ActivFunc<> regardless
    {
      using this_t = LayerRec<N_, L0_, L1_, Rest_...>;
      using traits_t = LayerTraits<this_t>;
      using base_t = LayerBase<this_t>;
    public:
      using base_t::Nin;
      using base_t::Nout;
      using base_t::Nfinal;
      // LayerRec() = default;
      scalar costFuncRecurse(const Eigen::Ref<const ArrayX>& net, const BatchVec<Nin>& xin, const BatchVec<Nfinal>& yin) const;
    private:
      // a private constructor may be appropriate if we will only instantiate this layer class from within the public FfnStatic class
      
      using L0_::activ;
      using L0_::activToD;

      using next_layer_t = LayerRec<Nout, L1_, Rest_...>;
      next_layer_t next_layer_;
    
    }; // class LayerRec (internal case)

    // end case for recursive layer: output layer will have RegressionType instead of InternalActivator
    template <size_t N_, typename L0_>
    class LayerRec<N_, L0_>
      : public LayerBase< LayerRec<N_, L0_> >
      , public L0_
    {
      using this_t = LayerRec<N_, L0_>;
      using traits_t = LayerTraits<this_t>;
      using base_t = LayerBase<this_t>;
      // these traits may or may not need to be exposed for member function definitions. probably yes if we want to define member functions below (i.e. outside of the FfnStatic class definition).
      using base_t::Nin; // could also get from traits_t::Nin, though not with "using" since this doesn't inherit
      using base_t::Nout;
      using base_t::Nfinal;
    public:
      scalar costFuncRecurse(const Eigen::Ref<const ArrayX>& net, const BatchVec<Nin>& xin, const BatchVec<Nfinal>& yin) const;
    private:
    
      // there is no next layer. we'll define functions based on the regression type.
      using L0_::outputGate; // we'd need to inherit from L0 (which is an FfnOutputLayerDef)
      using L0_::costFuncVal;
    }; // class LayerRec (output case)

    
    LayerRec<N, Layers...> first_layer_;// clang requires this member to be declared after the definitions. g++ just segfaults w/ the CRTP.
    
  };
  
  
} // namespace ceptron

