#pragma once

#include "Common.h"




// Cuda vector base class 

namespace cuda
{
    template<typename Type>
    class CudaVectorBase
    {
        static_assert(std::is_pod<Type>::value, "'Type' must be POD");

    protected:

        HOST_ONLY CudaVectorBase();
        HOST_ONLY virtual ~CudaVectorBase();

    public:
        // Getters

        HOST_DEVICE INLINE Type* Data() const { return data_; }
        HOST_DEVICE INLINE unsigned int Size() const { return size_; }; // For aligned vectors it does not correspond to the actual number of elements

    protected:

        HOST_ONLY void Clear();

    protected:

        Type* data_;
        unsigned int size_;
    };

} // cuda


// template definitions

template<typename Type>
HOST_ONLY cuda::CudaVectorBase<Type>::CudaVectorBase()
    : data_(nullptr), size_(0)
{
}

template<typename Type>
HOST_ONLY cuda::CudaVectorBase<Type>::~CudaVectorBase()
{
    if (data_ != nullptr)
    {
        CC(cudaFree(data_));
    }
}

template<typename Type>
HOST_ONLY void cuda::CudaVectorBase<Type>::Clear()
{
    CC(cudaFree(data_));
    data_ = nullptr;
    size_ = 0;
}
