#pragma once

#include "CudaVectorBase.h"
#include "Dimensions.h"




// Memory manager class

namespace cuda
{
    template<typename Type, typename DimType>
    class MemoryManager : public CudaVectorBase<Type>
    {
    public:

        HOST_ONLY void Resize(DimType dimension);
        HOST_ONLY void Clear();
        HOST_DEVICE bool Empty() const;

    protected:

        // Allocates memory (if not allocated already) on the device
        // Frees existing memory first
        HOST_ONLY void AllocateMemory(DimType dimension);

        HOST_ONLY MemoryManager();
        HOST_ONLY MemoryManager(DimType dimension);
        HOST_ONLY virtual ~MemoryManager();

    protected:

        DimType dimension_;
    };

} // cuda


// template definitions

template<typename Type, typename DimType>
HOST_ONLY void cuda::MemoryManager<Type, DimType>::Resize(DimType dimension)
{
    if (dimension_ == dimension) { return; }
    
    AllocateMemory(dimension);    
}

template<typename Type, typename DimType>
HOST_ONLY void cuda::MemoryManager<Type, DimType>::Clear()
{
    cuda::CudaVectorBase<Type>::Clear();
    dimension_.Clear();
}

template<typename Type, typename DimType>
HOST_DEVICE bool cuda::MemoryManager<Type, DimType>::Empty() const
{ 
    return CudaVectorBase<Type>::size_ == 0; 
}

template<typename Type, typename DimType>
HOST_ONLY void cuda::MemoryManager<Type, DimType>::AllocateMemory(DimType dimension)
{
    unsigned int size = dimension.Unroll();

    if (size <= 0) { throw std::runtime_error("Trying to allocate non-positive bytes of memory"); }

    if (CudaVectorBase<Type>::size_ == size && dimension_ == dimension)
    {
        return;
    }

    dimension_ = dimension;

    Type*& devicePtr = CudaVectorBase<Type>::data_;

    CudaVectorBase<Type>::size_ = size;
    CC(cudaFree(devicePtr));
    CC(cudaMalloc(&devicePtr, CudaVectorBase<Type>::size_ * sizeof(Type)));
}

template<typename Type, typename DimType>
HOST_ONLY cuda::MemoryManager<Type, DimType>::MemoryManager<Type, DimType>() 
{
}

template<typename Type, typename DimType>
HOST_ONLY cuda::MemoryManager<Type, DimType>::MemoryManager<Type, DimType>(DimType dimension)
    : dimension_(dimension)
{
}

template<typename Type, typename DimType>
HOST_ONLY cuda::MemoryManager<Type, DimType>::~MemoryManager()
{
}