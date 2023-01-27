#pragma once

#include "Common.h"



namespace cuda
{
    struct Dim1
    {
        HOST_ONLY Dim1(unsigned int x = 0);

        HOST_ONLY unsigned int Unroll() const;
        HOST_ONLY void Clear();

        HOST_DEVICE bool operator==(const Dim1& other) const;

        unsigned int x_;
    };

    struct Dim2
    {
        HOST_ONLY Dim2(unsigned int x = 0, unsigned int y = 0);

        HOST_ONLY unsigned int Unroll() const;
        HOST_ONLY void Clear();

        HOST_DEVICE bool operator==(const Dim2& other) const;

        unsigned int x_;
        unsigned int y_;
    };

    struct Dim3
    {
        HOST_ONLY Dim3(unsigned int x = 0, unsigned int y = 0, unsigned int z = 0);

        HOST_ONLY INLINE unsigned int Unroll() const;
        HOST_ONLY void Clear();

        HOST_DEVICE bool operator==(const Dim3& other) const;

        unsigned int x_;
        unsigned int y_;
        unsigned int z_;
    };

    // Aligned versions for multi-dimension vectors

    struct Dim2_Aligned
    {
        // x_ = number of rows
        // y_ = number of columns

        HOST_ONLY Dim2_Aligned(unsigned int x = 0, unsigned int y = 0, int SizeOfType = 0);

        HOST_ONLY unsigned int Unroll() const;
        HOST_ONLY void Clear();

        HOST_DEVICE bool operator==(const Dim2_Aligned& other) const;

        unsigned int x_;
        unsigned int y_;
        unsigned int stride_;   // Row stride to garantee memory alignment on the device (measured in number of elements of type Type)
        unsigned int length_;   // Number of actual elements in the vector (differs from size_)
#ifdef _DEBUG
        unsigned int wastedBytes_;
#endif
    };

    struct Dim3_Aligned
    {
        // x_ = number of rows
        // y_ = number of columns
        // z_ = number of layers

        HOST_ONLY Dim3_Aligned(unsigned int x = 0, unsigned int y = 0, unsigned int z = 0, int sizeOfType = 0);

        HOST_ONLY unsigned int Unroll() const;
        HOST_ONLY void Clear();

        HOST_DEVICE bool operator==(const Dim3_Aligned& other) const;


        unsigned int x_;
        unsigned int y_;
        unsigned int z_;
        unsigned int stride_;   // Row stride to garantee memory alignment on the device (measured in number of elements of type Type)
        unsigned int length_;   // Number of actual elements in the vector (differs from size_)
#ifdef _DEBUG
        unsigned int wastedBytes_;
#endif
    };

} // cuda



// template definitions

// Dim1

HOST_ONLY cuda::Dim1::Dim1(unsigned int x)
    : x_(x)
{
}

HOST_ONLY unsigned int cuda::Dim1::Unroll() const
{
    return x_;
}

HOST_ONLY void cuda::Dim1::Clear()
{
    x_ = 0;
}

HOST_DEVICE bool cuda::Dim1::operator==(const Dim1& other) const
{
    return this->x_ == other.x_;
}

// Dim2

HOST_ONLY cuda::Dim2::Dim2(unsigned int x, unsigned int y)
    : x_(x), y_(y)
{
}

HOST_ONLY unsigned int cuda::Dim2::Unroll() const
{
    return x_ * y_;
}

HOST_ONLY void cuda::Dim2::Clear()
{
    x_ = y_ = 0;
}

HOST_DEVICE bool cuda::Dim2::operator==(const Dim2& other) const
{
    return
        this->x_ == other.x_ &&
        this->y_ == other.y_;
}

// Dim3

HOST_ONLY cuda::Dim3::Dim3(unsigned int x, unsigned int y, unsigned int z)
    : x_(x), y_(y), z_(z)
{
}

HOST_ONLY unsigned int cuda::Dim3::Unroll() const
{
    return x_ * y_ * z_;
}

HOST_ONLY void cuda::Dim3::Clear()
{
    x_ = y_ = z_ = 0;
}

HOST_DEVICE bool cuda::Dim3::operator==(const Dim3& other) const
{
    return
        this->x_ == other.x_ &&
        this->y_ == other.y_ &&
        this->z_ == other.z_;
}

// Dim2_Aligned

HOST_ONLY cuda::Dim2_Aligned::Dim2_Aligned(unsigned int x, unsigned int y, int sizeOfType)
    : x_(x)
    , y_(y)
    , stride_(cuda::ComputeStride(y, sizeOfType))
    , length_(x * y)
#ifdef _DEBUG
    , wastedBytes_(x * stride_)
#endif
{
}

HOST_ONLY unsigned int cuda::Dim2_Aligned::Unroll() const
{
    return x_ * (y_ + stride_);
}

HOST_ONLY void cuda::Dim2_Aligned::Clear()
{
    x_ = y_ = stride_ = length_ = 0;
#ifdef _DEBUG
    wastedBytes_ = 0;
#endif
}

HOST_DEVICE bool cuda::Dim2_Aligned::operator==(const Dim2_Aligned& other) const
{
    return
        this->x_ == other.x_ &&
        this->y_ == other.y_ &&
        this->stride_ == other.stride_ &&
        this->length_ == other.length_;
}

// Dim3_Aligned

HOST_ONLY cuda::Dim3_Aligned::Dim3_Aligned(unsigned int x, unsigned int y, unsigned int z, int sizeOfType)
    : x_(x)
    , y_(y)
    , z_(z)
    , stride_(cuda::ComputeStride(y, sizeOfType))
    , length_(x * y * z)
#ifdef _DEBUG
    , wastedBytes_(x * z * stride_)
#endif
{
}

HOST_ONLY unsigned int cuda::Dim3_Aligned::Unroll() const
{
    return x_ * z_ * (y_ + stride_);
}

HOST_ONLY void cuda::Dim3_Aligned::Clear()
{
    x_ = y_ = z_ = stride_ = length_ = 0;
#ifdef _DEBUG
    wastedBytes_ = 0;
#endif
}

HOST_DEVICE bool cuda::Dim3_Aligned::operator==(const Dim3_Aligned& other) const
{
    return
        this->x_ == other.x_ &&
        this->y_ == other.y_ &&
        this->z_ == other.z_ &&
        this->stride_ == other.stride_ &&
        this->length_ == other.length_;
}
