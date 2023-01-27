# CUDA Vector

### Intro

Lightweight CUDA/C++ header-only library for vector computations on the GPU. 

### How to use it

Clone the repository and include `CudaVector.h` wherever you need.

### How it works

There are 2 types of container available:

1. `CudaVectorXD`;
2. `CudaAlgVectorXD` for aligned memory accesses.

Where `X` (Dimension) can either be 1, 2, or 3.
Every memory operation is carried out on the `host` code, and each of those classes has a `device` counterpart that can access the memory on the GPU.

Check out `Test.cu` to see an example usage.

### Next features

Integration with `std::vector`, iterators, capacity are not yet implemented.
