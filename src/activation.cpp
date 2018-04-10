#include "activation.hpp"

namespace {
  using namespace ceptron;
} // namespace


ceptron::BatchArrayX ceptron::activ(InternalActivator act, const Eigen::Ref<const BatchArrayX>& in) {
  switch (act) {
  case InternalActivator::Logit:
    return ActivFunc<InternalActivator::Logit>::activ(in);
  case InternalActivator::Tanh:
    return ActivFunc<InternalActivator::Tanh>::activ(in);
  case InternalActivator::ReLU:
    return ActivFunc<InternalActivator::ReLU>::activ(in);
  case InternalActivator::Softplus:
    return ActivFunc<InternalActivator::Softplus>::activ(in);
  case InternalActivator::LReLU:
    return ActivFunc<InternalActivator::LReLU>::activ(in);
  case InternalActivator::Softsign:
    return ActivFunc<InternalActivator::Softsign>::activ(in);
  }
  throw std::runtime_error("unimplemented runtime activation function");
}


ceptron::BatchArrayX ceptron::activToD(InternalActivator act, const Eigen::Ref<const BatchArrayX>& x) {
  switch (act) {
  case InternalActivator::Logit:
    return ActivFunc<InternalActivator::Logit>::activToD(x);
  case InternalActivator::Tanh:
    return ActivFunc<InternalActivator::Tanh>::activToD(x);
  case InternalActivator::ReLU:
    return ActivFunc<InternalActivator::ReLU>::activToD(x);
  case InternalActivator::Softplus:
    return ActivFunc<InternalActivator::Softplus>::activToD(x);
  case InternalActivator::LReLU:
    return ActivFunc<InternalActivator::LReLU>::activToD(x);    
  case InternalActivator::Softsign:
    return ActivFunc<InternalActivator::Softsign>::activToD(x);
  }
  throw std::runtime_error("unimplemented runtime activation derivative function");  
}
