#include "kernels.cuh"

// Device function to compute option price using a binomial tree
__device__ double deviceOptionPrice(int steps, double S, double K, double r, double q, double T, double sigma, int optionType) {
    double dt = T / steps;
    double u = exp(sigma * sqrt(dt));
    double d = 1.0 / u;
    double p = (exp((r - q) * dt) - d) / (u - d);

    // Validate p
    if (p < 0.0 || p > 1.0) {
        return NAN;
    }

    // Initialize option values at maturity
    double option_values[1024]; // Assuming steps <= 1023
    if (steps >= 1024) {
        // Exceeded maximum steps
        return NAN;
    }

    for (int i = 0; i <= steps; i++) {
        double price = S * pow(u, steps - i) * pow(d, i);
        if (optionType == 0) { // Call
            option_values[i] = fmax(0.0, price - K);
        } else { // Put
            option_values[i] = fmax(0.0, K - price);
        }
    }

    // Perform backward induction
    for (int step = steps - 1; step >= 0; step--) {
        for (int i = 0; i <= step; i++) {
            option_values[i] = (p * option_values[i] + (1.0 - p) * option_values[i + 1]) * exp(-r * dt);
        }
    }

    return option_values[0];
}

// Kernel to compute implied volatility using bisection
__global__ void computeImpliedVolatilityKernel(int steps, int batchSize, double* marketPrices, 
                                              double* S, double* K, double* r, double* q, 
                                              double* T, int* optionType, double* ivResults, 
                                              double tol, int max_iter) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize) return;

    double target = marketPrices[idx];
    double a = 0.01;
    double b = 3.0;
    double fa = deviceOptionPrice(steps, S[idx], K[idx], r[idx], q[idx], T[idx], a, optionType[idx]) - target;
    double fb = deviceOptionPrice(steps, S[idx], K[idx], r[idx], q[idx], T[idx], b, optionType[idx]) - target;

    if (isnan(fa) || isnan(fb)) {
        // Root not bracketed or invalid price
        ivResults[idx] = -1.0;
        return;
    }

    double c, fc;
    for (int iter = 0; iter < max_iter; iter++) {
        c = 0.5 * (a + b);
        fc = deviceOptionPrice(steps, S[idx], K[idx], r[idx], q[idx], T[idx], c, optionType[idx]) - target;

        if (fabs(fc) < tol) {
            ivResults[idx] = c;
            return;
        }

        if (fa * fc < 0.0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    ivResults[idx] = c; // Best estimate after max iterations
}

__global__ void priceOptionsKernel(int steps, int batchSize, 
                                   double* S, double* K, double* r, 
                                   double* q, double* T, double* sigma, 
                                   int* optionType, double* prices) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize) return;

    prices[idx] = deviceOptionPrice(steps, S[idx], K[idx], r[idx], q[idx], T[idx], sigma[idx], optionType[idx]);
}

