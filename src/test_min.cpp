#include "min_step.hpp"
#include "test_functions.hpp"

#include <iostream>

using std::cout;
using std::endl;

parvec test_min_step( IMinStep* minstep, func_t f, grad_t g, parvec initpars, double abstol=1e-6 ) {
  assert( minstep != nullptr );
  parvec pars = initpars;
  bool finished = false;
  while( !finished ) {
    parvec dpars = minstep->getDeltaPar( g, pars );

    pars += dpars;
    
    // just check the gradient magnitude for now
    // ArrayXd does not have a squaredNorm() function like VectorXd
    finished = g(pars).square().sum() < abstol*abstol;
  }
  return pars;
}

int main( int argc, char** argv ) {

  size_t ndim = 4;
  parvec initpars(ndim);
  cout << "testing basic sphere function" << endl;
  
  
  return 0;
}
