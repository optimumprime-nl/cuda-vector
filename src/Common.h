#pragma once

#pragma warning( disable : 4002 )

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <assert.h>


const int CACHE_LINE_GLOBAL_BYTES = 128;


#if defined(__CUDACC__)
#define KERNEL          __global__
#define HOST_DEVICE     __host__ __device__     
#define DEVICE_ONLY     __device__              
#define HOST_ONLY       __host__                
#define INLINE          __inline__
#else
#define KERNEL          
#define HOST_DEVICE         
#define DEVICE_ONLY                   
#define HOST_ONLY                     
#define INLINE          inline 
#endif


// This macro is a printf() wrapper
// - Prints the threadIdx and the blockIdx (till the 2nd dimension)
// - No need to comment it out for the release build
// - Use dprintf(...); as printf(...);


#ifdef _DEBUG

#ifdef __CUDA_ARCH__

#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define __BLOCK__  blockIdx.x + gridDim.x * blockIdx.y
#define __THREAD__ threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z
#define dprintf(format, ...) printf(ANSI_COLOR_MAGENTA "[%i, %i] " ANSI_COLOR_RESET format, __BLOCK__, __THREAD__, __VA_ARGS__);

#else

#define dprintf(args) printf(args)

#endif

#else

#define dprintf(...)

#endif


// Generic CUDA APIs error catcher
#define CC(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA API ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

namespace cuda {

    HOST_ONLY INLINE unsigned int ComputeStride(unsigned int rowLength, int sizeOfType)
    {
        if (rowLength == 0) { return 0; }

        int stride = 0;
        int minimumBytesToAllocate = rowLength * sizeOfType;
        if (minimumBytesToAllocate % CACHE_LINE_GLOBAL_BYTES != 0)
        {
            // Get the closest (and bigger) cache line starting address (address must be multiple of CACHE_LINE_GLOBAL_BYTES)
            int NearestBiggerCacheline = minimumBytesToAllocate + (CACHE_LINE_GLOBAL_BYTES - minimumBytesToAllocate % CACHE_LINE_GLOBAL_BYTES);
            stride = (NearestBiggerCacheline - minimumBytesToAllocate) / sizeOfType;

            assert((NearestBiggerCacheline - minimumBytesToAllocate) % sizeOfType == 0);
        }

        return stride;
    }

}



