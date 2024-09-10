#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cmath>

__global__ void initializeAssetPrices(double* prices, double* values, double S, double K, double u, double d, int steps, int optionType) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= steps) {
        prices[idx] = S * pow(u, steps - idx) * pow(d, idx);
        if (optionType == 0) { // Call
            values[idx] = fmax(0.0, prices[idx] - K);
        } else { // Put
            values[idx] = fmax(0.0, K - prices[idx]);
        }
    }
}

__global__ void backwardInduction(double* values, double* prices, int step, double S, double K, double p, double r, double dt, double u, double d, int optionType) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= step) {
        double spotPrice = S * std::pow(u, step - idx) * std::pow(d, idx);
        double expectedValue = p * values[idx] + (1 - p) * values[idx + 1];
        expectedValue *= exp(-r * dt);
        double intrinsicValue;
        if (optionType == 0) { // Call
            intrinsicValue = fmax(0.0, spotPrice - K);
        } else { // Put
            intrinsicValue = fmax(0.0, K - spotPrice);
        }
        values[idx] = fmax(expectedValue, intrinsicValue);
    }
}

#endif // KERNELS_CUH