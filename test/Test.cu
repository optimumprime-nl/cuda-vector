
// Example usage 

#include "cudaVector.h"


__global__ void Kernel(
	cuda::CudaVector1DKernel<double> v1,
	cuda::CudaVector2DKernel<double> v2,
	cuda::CudaVector3DKernel<double> v3,
	cuda::CudaAlgVector2DKernel<int> v2A,
	cuda::CudaAlgVector3DKernel<float> v3A
	)
{
	v1.At(2) = 832;
	v2.At(2, 4) = 8.213421;
	v3.At(2, 10, 0) = 0.00012213;
	v2A.At(0, 0) = v3A.Y();
	v3A.At(1, 12, 1) = 0.122f;

	dprintf("%f\n", v1.At(2));
	dprintf("%f\n", v2.At(2, 4));
	dprintf("%f\n", v3.At(2, 10, 0));
	dprintf("%d\n", v2A.At(0, 0));
	dprintf("%f\n", v3A.At(1, 12, 1));
}



int main()
{
	try
	{
		cuda::CudaVector1D<double> vec1(3);
		cuda::CudaVector2D<double> vec2;
		cuda::CudaVector3D<double> vec3({ 3, 12 , 1 });
		cuda::CudaAlgVector2D<int> vec2Aligned({ 2, 1, sizeof(int) });
		cuda::CudaAlgVector3D<float> vec3Aligned;
		vec2.Resize({ 3, 10 });
		vec3.Clear();
		vec3.Resize({ 4, 13 , 2 });
		vec3Aligned.Resize({ 10, 13, 10, sizeof(float) });

		Kernel << <1, 1 >> > (vec1, vec2, vec3, vec2Aligned, vec3Aligned);
		cudaDeviceSynchronize();

#ifdef _DEBUG
		std::cout << "2D wasted bytes: " << vec2Aligned.WastedBytes() << std::endl;
		std::cout << "3D wasted bytes: " << vec3Aligned.WastedBytes() << std::endl;
#endif
	}
	catch (std::exception& exception)
	{
		std::cout << exception.what() << std::endl;
	}	
}