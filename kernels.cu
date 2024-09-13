#include "kernels.cuh"
#include <iostream>
// CUDA Kernels
__device__ void initializeAssetPrices(double* prices, double* values, double S, double K, double u, double d, int steps, int optionType) {
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

__device__ void initializeAssetPricesSingle(double* prices, double* values, double S, double K, double u, double d, int steps, int optionType) {
    for (int idx = 0; idx <= steps; idx++) {
        prices[idx] = S * pow(u, steps - idx) * pow(d, idx);
        if (optionType == 0) { // Call
            values[idx] = fmax(0.0, prices[idx] - K);
        } else { // Put
            values[idx] = fmax(0.0, K - prices[idx]);
        }
    }
}


__device__ void backwardInduction(double* values, double* prices, int step, double S, double K, double p, double r, double dt, double u, double d, int optionType) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= step) {
        double spotPrice = S * pow(u, step - idx) * pow(d, idx);
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

__global__ void calculatePrice(int steps, int batchSize, double* price, double* batchedS, double* batchedK, double* batchedR, double* batchedQ, double* batchedT, int* batchedType, double* batchedSigma) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < batchSize) {
        double S = batchedS[idx];
        double K = batchedK[idx];
        double r = batchedR[idx];
        double q = batchedQ[idx];
        double T = batchedT[idx];
        double sigma = batchedSigma[idx];
        int optionType = batchedType[idx];
        double dt = T / steps;
        double u = exp(sigma * sqrt(dt));
        double d = 1/u;
        double p = (exp((r - q) * dt) - d) / (u - d);
        double* prices = new double[steps+1];
        double* values = new double[steps+1];

        initializeAssetPrices(prices, values, S, K, u, d, steps, optionType);
        
        for (int i = steps - 1; i >= 0; i--) {
            backwardInduction(values, prices, i, S, K, p, r, dt, u, d, optionType);
        }
        price[idx] = values[0];
    }
}

__global__ void calculateSinglePrice(int steps, double* price, double* S, double *K, double *r, double *q, double *T, int *optionType, double sigma, int index) {
    //if (index == 0) {
    //    printf("Index is zero\n");
    //}
    double dt = T[index] / steps;
    double u = exp(sigma * sqrt(dt));
    double d = 1/u;
    double p = (exp((r - q) * dt) - d) / (u - d);
    extern __shared__ double prices[];
    extern __shared__ double values[];
    //printf("Index is %d\n", index);
    initializeAssetPricesSingle(prices, values, S[index], K[index], u, d, steps, optionType[index]);
    for (int i = steps - 1; i >= 0; i--) {
        backwardInduction(values, prices, i, S[index], K[index], p, r[index], dt, u, d, optionType[index]);
    }
    //printf("Price for index %d is %f\n", index, values[0]);
    price = &values[0];
}