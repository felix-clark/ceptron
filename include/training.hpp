#pragma once
#include "global.hpp"
#include "net.hpp"
#include "min_step.hpp"

// serves as a wrapper to get a function of the parameters from a batch and a small net
// honestly this is somewhat due to a bit of an awkward interface with the NNs holding their own parameters.
// it makes it convenient to save/load but this might be worth a refactor
// we could certainly think of NNs as pure functions of par, x_batch, y_batch ...
// indeed, the more i think about it the more i think it'd be better to separate the functions
//  that compute f and g and the structure that stores the net value.
// then various gradient descent algorithms can be utilized with simple binds
// that will require some more planning about how to deal with the activation function. maybe they can just become template functions.
template <size_t Npar>
class IParFunc {
public:
  virtual ceptron::func_grad_res<Npar> getFuncGrad( const Array<Npar>& ) const = 0;
  virtual double getFuncOnly( const Array<Npar>& ) const = 0;
};

template <size_t Npar, size_t Nin, size_t Nout>
class ParFunc : public IParFunc<Npar>
{
public:
  ceptron::func_grad_res<Npar> getFuncGrad( const Array<Npar>& );
  double getFuncOnly( const Array<Npar>& );
};
