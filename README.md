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
Check out `Test.cu` to see example usages.

Nsight Compute metrics for memory ops on `CudaVector2D` and `CudaAlgVector2D` (taken from the example available in `Test.cu`):

![UntitledMis](https://user-images.githubusercontent.com/77488235/216616265-d92439af-a405-40fc-91ed-f263fbf841d9.png)
CudaVector2D

![Untitled](https://user-images.githubusercontent.com/77488235/216616232-6f7711b7-32ee-4e25-8f6f-a643c3bf3a06.png)
CudaAlgVector2D

### Next features

Integration with `std::vector`, iterators, capacity are not yet implemented.
