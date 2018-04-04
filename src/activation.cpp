#include "activation.hpp"

namespace {
  using namespace ceptron;
} // namespace


ceptron::BatchArrayX activ(InternalActivator act, const Eigen::Ref<const BatchArrayX>& in) {
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
  }
  throw std::runtime_error("unimplemented runtime activation funtion");
}


ceptron::BatchArrayX activToD(InternalActivator act, const Eigen::Ref<const BatchArrayX>& g) {
  throw std::runtime_error("unimplemented runtime activation derivative function");  
}
