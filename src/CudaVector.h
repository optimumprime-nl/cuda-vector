#pragma once

#include "MemoryManager.h"

#include "assert.h"




namespace cuda {

    // Forward declarations

    template<typename Type> class CudaVector1DKernel;
    template<typename Type> class CudaVector2DKernel;
    template<typename Type> class CudaVector3DKernel;
    template<typename Type> class CudaAlgVector2DKernel;
    template<typename Type> class CudaAlgVector3DKernel;


    // Host classes that manage device memory

    // 1D vector
    template<typename Type>
    class CudaVector1D : public MemoryManager<Type, Dim1>
    {
    public:

        HOST_ONLY CudaVector1D();
        HOST_ONLY CudaVector1D(Dim1 dimension);

        HOST_ONLY operator CudaVector1DKernel<Type>() const;
        HOST_ONLY operator CudaVector1DKernel<Type const>() const;

        // Getters

        HOST_ONLY INLINE unsigned int X() const { return dimension_.x_; };
    };

    // 2D vector
    template<typename Type>
    class CudaVector2D : public MemoryManager<Type, Dim2>
    {
    public:

        HOST_ONLY CudaVector2D();
        HOST_ONLY CudaVector2D(Dim2 dimension);

        HOST_ONLY operator CudaVector2DKernel<Type>() const;
        HOST_ONLY operator CudaVector2DKernel<Type const>() const;

        // Getters

        HOST_ONLY INLINE unsigned int X() const { return dimension_.x_; }
        HOST_ONLY INLINE unsigned int Y() const { return dimension_.y_; }
    };

    // 3D vector
    template<typename Type>
    class CudaVector3D : public MemoryManager<Type, Dim3>
    {
    public:

        HOST_ONLY CudaVector3D();
        HOST_ONLY CudaVector3D(Dim3 dimension);

        HOST_ONLY operator CudaVector3DKernel<Type>() const;
        HOST_ONLY operator CudaVector3DKernel<Type const>() const;

        // Getters

        HOST_ONLY INLINE unsigned int X() const { return dimension_.x_; }
        HOST_ONLY INLINE unsigned int Y() const { return dimension_.y_; }
        HOST_ONLY INLINE unsigned int Z() const { return dimension_.z_; }
    };

    // 2D Vector Aligned
    template<typename Type>
    class CudaAlgVector2D : public MemoryManager<Type, Dim2_Aligned>
    {
    public:

        HOST_ONLY CudaAlgVector2D();
        HOST_ONLY CudaAlgVector2D(Dim2_Aligned dimension);

        HOST_ONLY operator CudaAlgVector2DKernel<Type>() const;
        HOST_ONLY operator CudaAlgVector2DKernel<Type const>() const;

        // Getters

        HOST_ONLY INLINE unsigned int X() const { return dimension_.x_; };
        HOST_ONLY INLINE unsigned int Y() const { return dimension_.y_; };
        HOST_ONLY INLINE unsigned int Stride() const { return dimension_.stride_; };
        HOST_ONLY INLINE unsigned int Length() const { return dimension_.length_; };
        HOST_ONLY INLINE unsigned int WastedBytes() const { return dimension_.wastedBytes_; };
    };

    // 3D Vector Aligned
    template<typename Type>
    class CudaAlgVector3D : public MemoryManager<Type, Dim3_Aligned>
    {
    public:

        HOST_ONLY CudaAlgVector3D();
        HOST_ONLY CudaAlgVector3D(Dim3_Aligned dimension);

        HOST_ONLY operator CudaAlgVector3DKernel<Type>() const;
        HOST_ONLY operator CudaAlgVector3DKernel<Type const>() const;

        // Getters

        HOST_ONLY INLINE unsigned int X() const { return dimension_.x_; };
        HOST_ONLY INLINE unsigned int Y() const { return dimension_.y_; };
        HOST_ONLY INLINE unsigned int Z() const { return dimension_.z_; };
        HOST_ONLY INLINE unsigned int Stride() const { return dimension_.stride_; };
        HOST_ONLY INLINE unsigned int Length() const { return dimension_.length_; };
        HOST_ONLY INLINE unsigned int WastedBytes() const { return dimension_.wastedBytes_; };
    };


    // Lightweight classes that allow device memory access
    // Constructed by copy from a CudaVector with same dimension_ and Type

    // 1D vector kernel
    template<typename Type>
    class CudaVector1DKernel
    {
    public:

        DEVICE_ONLY Type& At(unsigned int x);

        HOST_ONLY CudaVector1DKernel<Type>(Type* data, Dim1 dimension);

    private:

        CudaVector1DKernel() = default;

    public:
        // Getters

        DEVICE_ONLY INLINE unsigned int X() const { return dimension_.x_; };

    private:

        Type* data_;		    
        Dim1 dimension_;
    };

    // 2D vector kernel 
    template<typename Type>
    class CudaVector2DKernel
    {
    public:

        // x = row index
        // y = column index

        DEVICE_ONLY Type& At(unsigned int x, unsigned int y);

        HOST_ONLY CudaVector2DKernel<Type>(Type* data, Dim2 dimension);

    private:

        CudaVector2DKernel() = default;

    public:
        // Getters

        DEVICE_ONLY INLINE unsigned int X() const { return dimension_.x_; };
        DEVICE_ONLY INLINE unsigned int Y() const { return dimension_.y_; };

    private:

        Type* data_;
        Dim2 dimension_;
    };

    // 3D vector kernel
    template<typename Type>
    class CudaVector3DKernel
    {
    public:

        // x = row index
        // y = column index
        // z = layer index

        DEVICE_ONLY Type& At(unsigned int x, unsigned int y, unsigned int z);

        HOST_ONLY CudaVector3DKernel<Type>(Type* data, Dim3 dimension);

    private:

        CudaVector3DKernel() = default;

    public:
        // Getters

        DEVICE_ONLY INLINE unsigned int X() const { return dimension_.x_; };
        DEVICE_ONLY INLINE unsigned int Y() const { return dimension_.y_; };
        DEVICE_ONLY INLINE unsigned int Z() const { return dimension_.z_; };

    private:

        Type* data_;
        Dim3 dimension_;
    };

    // 2D aligned vector kernel
    template<typename Type>
    class CudaAlgVector2DKernel
    {
    public:

        // x = row index
        // y = column index

        DEVICE_ONLY Type& At(unsigned int x, unsigned int y);

        HOST_ONLY CudaAlgVector2DKernel<Type>(Type* data, Dim2_Aligned dimension);

    private:

        CudaAlgVector2DKernel() = default;

    public:
        // Getters

        DEVICE_ONLY INLINE unsigned int X() const { return dimension_.x_; };
        DEVICE_ONLY INLINE unsigned int Y() const { return dimension_.y_; };

    private:

        Type* data_;
        Dim2_Aligned dimension_;
    };

    // 3D aligned vector kernel
    template<typename Type>
    class CudaAlgVector3DKernel
    {
    public:

        // x = row index
        // y = column index
        // z = layer index

        DEVICE_ONLY Type& At(unsigned int x, unsigned int y, unsigned int z);

        HOST_ONLY CudaAlgVector3DKernel<Type>(Type* data, Dim3_Aligned dimension);

    private:

        CudaAlgVector3DKernel() = default;

    public:
        // Getters

        DEVICE_ONLY INLINE unsigned int X() const { return dimension_.x_; };
        DEVICE_ONLY INLINE unsigned int Y() const { return dimension_.y_; };
        DEVICE_ONLY INLINE unsigned int Z() const { return dimension_.z_; };

    private:

        Type* data_;
        Dim3_Aligned dimension_;
    };

} // cuda


// Template definitions

// 1D vector

template<typename Type>
HOST_ONLY cuda::CudaVector1D<Type>::CudaVector1D()
{
}

template<typename Type>
HOST_ONLY cuda::CudaVector1D<Type>::CudaVector1D(Dim1 dimension)
    : MemoryManager<Type, Dim1>(dimension)
{
    MemoryManager<Type, Dim1>::AllocateMemory(dimension);
}

template<typename Type>
HOST_ONLY cuda::CudaVector1D<Type>::operator CudaVector1DKernel<Type>() const
{
    return CudaVector1DKernel<Type>{ CudaVectorBase<Type>::Data(), MemoryManager<Type, Dim1>::dimension_ };
}

template<typename Type>
HOST_ONLY cuda::CudaVector1D<Type>::operator CudaVector1DKernel<Type const>() const
{
    return CudaVector1DKernel<Type const>{ CudaVectorBase<Type>::Data(), MemoryManager<Type, Dim1>::dimension_ };
}


// 2D vector

template<typename Type>
HOST_ONLY cuda::CudaVector2D<Type>::CudaVector2D()
{
}

template<typename Type>
HOST_ONLY cuda::CudaVector2D<Type>::CudaVector2D(Dim2 dimension)
    : MemoryManager<Type, Dim2>(dimension)
{
    MemoryManager<Type, Dim2>::AllocateMemory(dimension);
}

template<typename Type>
HOST_ONLY cuda::CudaVector2D<Type>::operator CudaVector2DKernel<Type>() const
{
    return CudaVector2DKernel<Type>{ CudaVectorBase<Type>::Data(), MemoryManager<Type, Dim2>::dimension_ };
}

template<typename Type>
HOST_ONLY cuda::CudaVector2D<Type>::operator CudaVector2DKernel<Type const>() const
{
    return CudaVector2DKernel<Type const>{ CudaVectorBase<Type>::Data(), MemoryManager<Type, Dim2>::dimension_ };
}

// 3D vector

template<typename Type>
HOST_ONLY cuda::CudaVector3D<Type>::CudaVector3D()
{
}

template<typename Type>
HOST_ONLY cuda::CudaVector3D<Type>::CudaVector3D(Dim3 dimension)
    : MemoryManager<Type, Dim3>(dimension)
{
    MemoryManager<Type, Dim3>::AllocateMemory(dimension);
}

template<typename Type>
HOST_ONLY cuda::CudaVector3D<Type>::operator CudaVector3DKernel<Type>() const
{
    return CudaVector3DKernel<Type>{ CudaVectorBase<Type>::Data(), MemoryManager<Type, Dim3>::dimension_ };
}

template<typename Type>
HOST_ONLY cuda::CudaVector3D<Type>::operator CudaVector3DKernel<Type const>() const
{
    return CudaVector3DKernel<Type const>{ CudaVectorBase<Type>::Data(), MemoryManager<Type, Dim3>::dimension_ };
}

// 2D aligned vector

template<typename Type>
HOST_ONLY cuda::CudaAlgVector2D<Type>::CudaAlgVector2D()
{
}

template<typename Type>
HOST_ONLY cuda::CudaAlgVector2D<Type>::CudaAlgVector2D(Dim2_Aligned dimension)
    : MemoryManager<Type, Dim2_Aligned>(dimension)
{
    MemoryManager<Type, Dim2_Aligned>::AllocateMemory(dimension);
}

template<typename Type>
HOST_ONLY cuda::CudaAlgVector2D<Type>::operator CudaAlgVector2DKernel<Type>() const
{
    return CudaAlgVector2DKernel<Type>{ CudaVectorBase<Type>::Data(), MemoryManager<Type, Dim2_Aligned>::dimension_ };
}

template<typename Type>
HOST_ONLY cuda::CudaAlgVector2D<Type>::operator CudaAlgVector2DKernel<Type const>() const
{
    return CudaAlgVector2DKernel<Type const>{ CudaVectorBase<Type>::Data(), MemoryManager<Type, Dim2_Aligned>::dimension_ };
}

// 3D aligned vector

template<typename Type>
HOST_ONLY cuda::CudaAlgVector3D<Type>::CudaAlgVector3D()
{
}

template<typename Type>
HOST_ONLY cuda::CudaAlgVector3D<Type>::CudaAlgVector3D(Dim3_Aligned dimension)
    : MemoryManager<Type, Dim3_Aligned>(dimension)
{
    MemoryManager<Type, Dim3_Aligned>::AllocateMemory(dimension);
}

template<typename Type>
HOST_ONLY cuda::CudaAlgVector3D<Type>::operator CudaAlgVector3DKernel<Type>() const
{
    return CudaAlgVector3DKernel<Type>{ CudaVectorBase<Type>::Data(), MemoryManager<Type, Dim3_Aligned>::dimension_ };
}

template<typename Type>
HOST_ONLY cuda::CudaAlgVector3D<Type>::operator CudaAlgVector3DKernel<Type const>() const
{
    return CudaAlgVector3DKernel<Type const>{ CudaVectorBase<Type>::Data(), MemoryManager<Type, Dim3_Aligned>::dimension_ };
}

// 1D vector kernel

template<typename Type>
DEVICE_ONLY INLINE Type& cuda::CudaVector1DKernel<Type>::At(unsigned int x)
{
    assert(x < dimension_.x_);

    return data_[x];
}

template<typename Type>
HOST_ONLY cuda::CudaVector1DKernel<Type>::CudaVector1DKernel<Type>(Type* data, Dim1 dimension)
    : data_(data), dimension_(dimension)
{
}

// 2D vector kernel

template<typename Type>
DEVICE_ONLY INLINE Type& cuda::CudaVector2DKernel<Type>::At(unsigned int x, unsigned int y)
{
    assert(x < dimension_.x_);
    assert(y < dimension_.y_);

    return data_[dimension_.y_ * x + y];
}

template<typename Type>
HOST_ONLY cuda::CudaVector2DKernel<Type>::CudaVector2DKernel(Type* data, Dim2 dimension)
    : data_(data), dimension_(dimension)
{
}

// 3D vector kernel

template<typename Type>
DEVICE_ONLY INLINE Type& cuda::CudaVector3DKernel<Type>::At(unsigned int x, unsigned int y, unsigned int z)
{
    assert(x < dimension_.x_);
    assert(y < dimension_.y_);
    assert(z < dimension_.z_);

    return data_[(dimension_.x_ * dimension_.y_ * z) + (dimension_.y_ * x + y)];
}

template<typename Type>
HOST_ONLY cuda::CudaVector3DKernel<Type>::CudaVector3DKernel(Type* data, Dim3 dimension)
    : data_(data), dimension_(dimension)
{
}

// 2D aligned vector kernel

template<typename Type>
DEVICE_ONLY INLINE Type& cuda::CudaAlgVector2DKernel<Type>::At(unsigned int x, unsigned int y)
{
    assert(x < dimension_.x_);
    assert(y < dimension_.y_);

    return data_[(dimension_.y_ + dimension_.stride_) * x + y];
}

template<typename Type>
HOST_ONLY cuda::CudaAlgVector2DKernel<Type>::CudaAlgVector2DKernel(Type* data, Dim2_Aligned dimension)
    : data_(data), dimension_(dimension)
{
}

// 3D aligned vector kernel

template<typename Type>
DEVICE_ONLY INLINE Type& cuda::CudaAlgVector3DKernel<Type>::At(unsigned int x, unsigned int y, unsigned int z)
{
    assert(x < dimension_.x_);
    assert(y < dimension_.y_);
    assert(z < dimension_.z_);

    return data_[((dimension_.y_ + dimension_.stride_) * dimension_.x_ * z) + ((dimension_.y_ + dimension_.stride_) * x + y)];
}

template<typename Type>
HOST_ONLY cuda::CudaAlgVector3DKernel<Type>::CudaAlgVector3DKernel(Type* data, Dim3_Aligned dimension)
    : data_(data), dimension_(dimension)
{
}