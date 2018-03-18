// test functions to compare minimizers with
#include <Eigen/Dense>

using parvec = Eigen::ArrayXd;

// an easy case that should converge to the origin
double sphere( const parvec& pars );
parvec grad_sphere( const parvec& pars );

// should converge to all parameters having value 1.0
double rosenbrock( const parvec& pars, double scale=100.0 );
parvec grad_rosenbrock( const parvec& pars, double scale=100.0 );

// a highly oscillating function.
// global minimum at origin but has many local minima.
// probably too pathological for ML minimizers
double rastrigin( const parvec& pars, double scale=10.0 );
parvec grad_rastrigin( const parvec& pars, double scale=10.0 );
