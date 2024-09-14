#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cmath>

// Initialize asset prices and option values
__global__ void initializeAssetPrices(double* prices, double* values, 
                                      double S, double K, double u, double d, 
                                      int steps, int optionType, int batchIndex, 
                                      int steps_plus_one);

// Backward induction to compute option values
__global__ void backwardInduction(double* values, double* prices, 
                                  int step, double p, double r, double dt, 
                                  int steps_plus_one, int batchIndex);

// Kernel to calculate option prices for a batch
__global__ void calculatePrice(int steps, int batchSize, double* price, 
                               double* batchedS, double* batchedK, double* batchedR, 
                               double* batchedQ, double* batchedT, int* batchedType, 
                               double* batchedSigma, double* prices, double* values);

#endif // KERNELS_CUH
