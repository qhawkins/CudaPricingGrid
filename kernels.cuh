#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cmath>

// CUDA Kernels
__global__ void calculatePrice(int steps, int batchSize, double* price, double* batchedS, double* batchedK, double* batchedR, double* batchedQ, double* batchedT, int* batchedType, double* batchedSigma);

__global__ void calculateSinglePrice(int steps, double* price, double* S, double* K, double* r, double* q, double* T, int* optionType, double sigma, int index);

#endif // KERNELS_CUH