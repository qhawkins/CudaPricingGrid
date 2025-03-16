#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include "structs.h"

// Device function to compute option price using a binomial tree
__device__ double deviceOptionPrice(int steps, double S, double K, double r, double q, double T, double sigma, int optionType);

// Kernel to compute implied volatility using bisection
__global__ void computeImpliedVolatilityKernel(int steps, int batchSize, double* marketPrices, 
                                              double* S, double* K, double* r, double* q, 
                                              double* T, int* optionType, double* ivResults, 
                                              double tol, int max_iter);

// Kernel to price options
__global__ void priceOptionsKernel(int steps, int batchSize, 
                                  double* S, double* K, double* r, 
                                  double* q, double* T, double* sigma, 
                                  int* optionType, double* prices);

// Optimized fused kernel that computes IV and Greeks in one pass
__global__ void fusedComputationKernel(
    int steps, int batchSize, 
    const double* marketPrices, 
    const double* S, const double* K, 
    const double* r, const double* q, 
    const double* T, const int* optionType,
    double* ivResults, double* deltaResults, 
    double* gammaResults, double* thetaResults, 
    double* vegaResults,
    double dSpot, double dVol,
    double tol, int max_iter);

// Shared memory version of option pricing for better performance
__global__ void sharedMemoryOptionPricingKernel(
    int steps, int batchSize,
    const double* S, const double* K,
    const double* r, const double* q,
    const double* T, const double* sigma,
    const int* optionType, double* prices);

#endif // KERNELS_CUH

