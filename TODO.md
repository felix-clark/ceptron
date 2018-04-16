# TODO

## make scalar type customizable (and default single-precision instead of double)

Most ML does just fine with only single-precision, and doubles are not supported on GPUs.

Half-precision floats could even be considered, but this might take some extra work and compromise portability.

## work on error messages for initializing arbitrary-size compile-time NNs

## implement dropout regularization

the tricky part here is figuring out when the masks can be re-rolled

## implement resursive patterns (RNNs)

this can have non-trivial effects on backprop and dropout (mask needs to persist throughout additional unrolled calls).
