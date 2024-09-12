#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cmath>

// CUDA Kernels
__global__ void initializeAssetPrices(double* prices, double* values, double S, double K, double u, double d, int steps, int optionType);

__global__ void backwardInduction(double* values, double* prices, int step, double S, double K, double p, double r, double dt, double u, double d, int optionType);

#endif // KERNELS_CUH