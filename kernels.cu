#include "kernels.cuh"

// Initialize asset prices and option values
__global__ void initializeAssetPrices(double* prices, double* values, 
                                      double S, double K, double u, double d, 
                                      int steps, int optionType, int batchIndex, 
                                      int steps_plus_one) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < steps_plus_one) {
        prices[batchIndex * steps_plus_one + idx] = S * pow(u, steps - idx) * pow(d, idx);
        if (optionType == 0) { // Call
            values[batchIndex * steps_plus_one + idx] = fmax(0.0, 
                prices[batchIndex * steps_plus_one + idx] - K);
        } else { // Put
            values[batchIndex * steps_plus_one + idx] = fmax(0.0, 
                K - prices[batchIndex * steps_plus_one + idx]);
        }
    }
}

// Backward induction to compute option values
__global__ void backwardInduction(double* values, double* prices, 
                                  int step, double p, double r, double dt, 
                                  int steps_plus_one, int batchIndex) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < step + 1) {
        double expectedValue = p * values[batchIndex * steps_plus_one + idx] + 
                               (1.0 - p) * values[batchIndex * steps_plus_one + idx + 1];
        expectedValue *= exp(-r * dt);
        // Update the option value at this node
        values[batchIndex * steps_plus_one + idx] = expectedValue;
    }
}

// Kernel to calculate option prices for a batch
__global__ void calculatePrice(int steps, int batchSize, double* price, 
                               double* batchedS, double* batchedK, double* batchedR, 
                               double* batchedQ, double* batchedT, int* batchedType, 
                               double* batchedSigma, double* prices, double* values) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int steps_plus_one = steps + 1;
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
        double d = 1.0 / u;
        double p = (exp((r - q) * dt) - d) / (u - d);
        
        // Initialize asset prices and option values
        int threadsPerBlock = 256;
        int blocksPerGrid = (steps_plus_one + threadsPerBlock - 1) / threadsPerBlock;
        initializeAssetPrices<<<blocksPerGrid, threadsPerBlock>>>(
            prices, values, S, K, u, d, steps, optionType, idx, steps_plus_one
        );
        __syncthreads();
        //cudaDeviceSynchronize(); // Ensure initialization is complete

        // Perform backward induction
        for (int step = steps - 1; step >= 0; step--) {
            int current_step_plus_one = step + 1;
            blocksPerGrid = (current_step_plus_one + threadsPerBlock - 1) / threadsPerBlock;
            backwardInduction<<<blocksPerGrid, threadsPerBlock>>>(
                values, prices, step, p, r, dt, steps_plus_one, idx
            );
            __syncthreads();
        }

        // Assign the computed option price
        price[idx] = values[idx * steps_plus_one];
    }
}
