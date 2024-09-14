#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// Initialize asset prices and option values

__global__ void computeImpliedVolatilityKernel(int steps, int batchSize, double* marketPrices, 
                                              double* S, double* K, double* r, double* q, 
                                              double* T, int* optionType, double* ivResults, 
                                              double tol, int max_iter);

__global__ void priceOptionsKernel(int steps, int batchSize, 
                                   double* S, double* K, double* r, 
                                   double* q, double* T, double* sigma, 
                                   int* optionType, double* prices);                                     
#endif // KERNELS_CUH

