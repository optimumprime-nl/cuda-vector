
// Example usage 

#include "cudaVector.h"

#include <random>
#include <algorithm>
#include <chrono>

using namespace cuda;


// ============================== EXAMPLE 1: How to use it ================================

KERNEL void SmallExampleKernel(
    CudaVector1DKernel<unsigned int> vec1,
    CudaVector2DKernel<int> vec2, 
    CudaVector3DKernel<double> vec3
)
{
    if (threadIdx.x + blockDim.x * blockIdx.x >= 1) return;

    for (int x = 0; x < vec3.X(); x++)
    {
        vec1.At(x) = x;

        for (int y = 0; y < vec3.Y(); y++)
        {
            vec2.At(x, y) = y * x;

            for (int z = 0; z < vec3.Z(); z++)
            {
                vec3.At(x, y, z) = static_cast<double>(x * vec2.At(0, y) * z);
            }
        }
    }
}

void SmallExample()
{
    unsigned int x = 100, y = 100, z = 100;

    CudaVector1D<unsigned int> vec1(x);
    CudaVector2D<int> vec2;
    vec2.Resize({ x, y });
    CudaVector3D<double> vec3({ x, y, z });

    SmallExampleKernel << <1, 1 >> > (vec1, vec2, vec3);
    CC(cudaDeviceSynchronize());
}



// ================= EXAMPLE 2: Unoptimized vs Optimized (aligned) =========================

typedef float Real;

struct Params
{
    Real T, r, dt, sigma, sqrtdt;
};

// Fills stockPrice vector with simulated values
template<typename Vector2DType>
__global__ void SimulationKernel(
    Params pms,
    Vector2DType stockPrice,
    Vector2DType randoms)
{
    // Grid-stride loop over the paths
    for (int path = blockIdx.x * blockDim.x + threadIdx.x;
        path < stockPrice.Y();
        path += blockDim.x * gridDim.x)
    {
        // Init stock (log-)price at time 0
        stockPrice.At(0, path) = static_cast<Real>(100);

        // Simulate step by step, 1 GPU thread evaluates multiple paths
        for (int step = 0; step < stockPrice.X() - 1; step++)
        {
            const Real& stock = stockPrice.At(step, path);

            Real drift = (pms.r - static_cast<Real>(0.5) * pms.sigma * pms.sigma) * pms.dt;
            Real diffusion = pms.sigma * pms.sqrtdt * randoms.At(step, path);

            stockPrice.At(step + 1, path) = stock * exp(drift + diffusion);
        }
    }
}

void Simulation()
{
    // Init parameters
    const int simulationDates = 5000;
    const int numberOfPaths = 100001;
    Params params;
    params.T = static_cast<Real>(1);
    params.dt = static_cast<Real>(params.T / simulationDates);
    params.sqrtdt = sqrt(params.dt);
    params.sigma = static_cast<Real>(0.25);
    params.r = static_cast<Real>(0.05);

    // Misaligned version
    CudaVector2D<Real> stockPrice({ simulationDates , numberOfPaths });
    CudaVector2D<Real> randoms({ simulationDates - 1, numberOfPaths });

    // Aligned version
    CudaAlgVector2D<Real> stockPriceOpt({ simulationDates , numberOfPaths, sizeof(Real) });
    CudaAlgVector2D<Real> randomsOpt({ simulationDates - 1, numberOfPaths, sizeof(Real) });

    // Init randoms
    std::default_random_engine defaultEngine{ static_cast<long unsigned int>(123) };
    std::normal_distribution<Real> dist(0.0, 1.0);
    auto gen = [&dist, &defaultEngine]() { return dist(defaultEngine); };
    std::vector<Real> normals(randomsOpt.Size());
    std::generate(std::begin(normals), std::end(normals), gen);
    CC(cudaMemcpy(randoms.Data(), normals.data(), randoms.Size() * sizeof(Real), cudaMemcpyHostToDevice));
    CC(cudaMemcpy(randomsOpt.Data(), normals.data(), randomsOpt.Size() * sizeof(Real), cudaMemcpyHostToDevice));

    // Get times for both versions
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::chrono::steady_clock::time_point tic, toc;
    std::chrono::duration<double> time;
    {
        CC(cudaDeviceSynchronize());
        tic = std::chrono::steady_clock::now();
        SimulationKernel<CudaVector2DKernel<Real>> << < props.multiProcessorCount * 12, 128 >> > (
            params,
            static_cast<CudaVector2DKernel<Real>>(stockPrice),
            static_cast<CudaVector2DKernel<Real>>(randoms)
            );
        CC(cudaDeviceSynchronize());
        toc = std::chrono::steady_clock::now();
        time = toc - tic;
        std::cout << "Misaligned version time: " << time.count() << "s\n";
    }
    {
        CC(cudaDeviceSynchronize());
        tic = std::chrono::steady_clock::now();
        SimulationKernel<CudaAlgVector2DKernel<Real>> << < props.multiProcessorCount * 12, 128 >> > (params,
            static_cast<CudaAlgVector2DKernel<Real>>(stockPriceOpt),
            static_cast<CudaAlgVector2DKernel<Real>>(randomsOpt)
            );
        CC(cudaDeviceSynchronize());
        toc = std::chrono::steady_clock::now();
        time = toc - tic;
        std::cout << "Aligned version time: " << time.count() << "s\n";
        std::cout << "Stride: " << stockPriceOpt.Stride() << std::endl;
#ifdef _DEBUG
        std::cout << "Wasted KB: " << stockPriceOpt.WastedBytes() / 1e3 << std::endl;
#endif
    }
}



int main()
{
    try 
    {
        { 
            SmallExample(); 
        }
        {
            Simulation(); 
        }
    }
    catch (std::exception& exception)
    {
        std::cout << exception.what() << std::endl;
    }
}
