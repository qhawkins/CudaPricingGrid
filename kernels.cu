#include "kernels.cuh"
#include <math.h>

// Optimized device function to compute option price using a binomial tree
__device__ double deviceOptionPrice(int steps, double S, double K, double r, double q, double T, double sigma, int optionType) {
    // Calculate time step
    double dt = T / steps;
    
    // Precompute constants for efficiency
    double u = exp(sigma * sqrt(dt));
    double d = 1.0 / u;
    double pu = (exp((r - q) * dt) - d) / (u - d);
    double pd = 1.0 - pu;
    double discount = exp(-r * dt);
    
    // Validate probability
    if (pu < 0.0 || pu > 1.0) {
        return NAN;
    }

    // Check if steps exceeds our array size limit
    if (steps >= 1024) {
        return NAN;
    }
    
    // Thread-local array for option values
    double option_values[1024];  // Fixed size array
    
    // Initialize option values at maturity (time step N)
    for (int i = 0; i <= steps; i++) {
        // Compute final asset price at this node
        double asset_price = S * pow(u, steps - i) * pow(d, i);
        
        // Determine option payoff based on type
        if (optionType == 0) {  // Call option
            option_values[i] = fmax(0.0, asset_price - K);
        } else {  // Put option
            option_values[i] = fmax(0.0, K - asset_price);
        }
    }
    
    // Backward induction to price the option
    for (int step = steps - 1; step >= 0; step--) {
        for (int i = 0; i <= step; i++) {
            // Calculate option value at this node using risk-neutral valuation
            option_values[i] = discount * (pu * option_values[i] + pd * option_values[i + 1]);
        }
    }
    
    // Return the option price (value at root node)
    return option_values[0];
}

// Kernel to compute implied volatility using bisection - optimized
__global__ void computeImpliedVolatilityKernel(int steps, int batchSize, double* marketPrices, 
                                              double* S, double* K, double* r, double* q, 
                                              double* T, int* optionType, double* ivResults, 
                                              double tol, int max_iter) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize) return;
    
    // Load option parameters (coalesced memory access)
    double s = S[idx];
    double k = K[idx];
    double rate = r[idx];
    double yield = q[idx];
    double time = T[idx];
    double target = marketPrices[idx];
    int type = optionType[idx];
    
    // Use Newton-Raphson method first for faster convergence, then fallback to bisection
    double sigma = 0.3;  // Initial guess
    double price, vega;
    
    // Newton-Raphson iterations (faster convergence)
    for (int iter = 0; iter < 10; iter++) {
        price = deviceOptionPrice(steps, s, k, rate, yield, time, sigma, type);
        
        // Calculate vega numerically
        double sigma_up = sigma * 1.001;
        double price_up = deviceOptionPrice(steps, s, k, rate, yield, time, sigma_up, type);
        vega = (price_up - price) / (sigma_up - sigma);
        
        // Check for convergence
        double diff = price - target;
        if (fabs(diff) < tol) {
            ivResults[idx] = sigma;
            return;
        }
        
        // Avoid division by very small vega
        if (fabs(vega) < 1e-8) break;
        
        // Newton step
        double new_sigma = sigma - diff / vega;
        
        // Ensure sigma stays in reasonable bounds
        if (new_sigma < 0.001) new_sigma = 0.001;
        if (new_sigma > 5.0) new_sigma = 5.0;
        
        // Check for convergence of sigma
        if (fabs(new_sigma - sigma) < 1e-6) {
            ivResults[idx] = new_sigma;
            return;
        }
        
        sigma = new_sigma;
    }
    
    // If Newton-Raphson didn't converge, fallback to bisection method
    double a = 0.01;  // Lower bound for IV
    double b = 3.0;   // Upper bound
    
    double fa = deviceOptionPrice(steps, s, k, rate, yield, time, a, type) - target;
    double fb = deviceOptionPrice(steps, s, k, rate, yield, time, b, type) - target;
    
    // Check if solution is bracketed
    if (fa * fb > 0.0 || isnan(fa) || isnan(fb)) {
        ivResults[idx] = -1.0;  // No solution or invalid parameters
        return;
    }
    
    // Bisection method
    double c, fc;
    for (int iter = 0; iter < max_iter; iter++) {
        c = 0.5 * (a + b);
        fc = deviceOptionPrice(steps, s, k, rate, yield, time, c, type) - target;
        
        // Check for convergence
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
    
    // Return best estimate after max iterations
    ivResults[idx] = c;
}

// Optimized kernel to price options
__global__ void priceOptionsKernel(int steps, int batchSize, 
                                  double* S, double* K, double* r, 
                                  double* q, double* T, double* sigma, 
                                  int* optionType, double* prices) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize) return;
    
    // Compute option price
    prices[idx] = deviceOptionPrice(steps, S[idx], K[idx], r[idx], q[idx], T[idx], sigma[idx], optionType[idx]);
}

// Fused kernel implementation that calculates IV and Greeks in one pass
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
    double tol, int max_iter) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize) return;
    
    // Load option parameters
    double s = S[idx];
    double k = K[idx];
    double rate = r[idx];
    double yield = q[idx];
    double time = T[idx];
    double target = marketPrices[idx];
    int type = optionType[idx];
    
    // 1. Compute implied volatility
    double a = 0.01;
    double b = 3.0;
    double fa = deviceOptionPrice(steps, s, k, rate, yield, time, a, type) - target;
    double fb = deviceOptionPrice(steps, s, k, rate, yield, time, b, type) - target;
    
    double sigma = -1.0; // Default invalid value
    
    if (!isnan(fa) && !isnan(fb)) {
        double c, fc;
        for (int iter = 0; iter < max_iter; iter++) {
            c = 0.5 * (a + b);
            fc = deviceOptionPrice(steps, s, k, rate, yield, time, c, type) - target;
            
            if (fabs(fc) < tol) {
                sigma = c;
                break;
            }
            
            if (fa * fc < 0.0) {
                b = c;
                fb = fc;
            } else {
                a = c;
                fa = fc;
            }
        }
        
        sigma = 0.5 * (a + b); // Best estimate after iterations
    }
    
    ivResults[idx] = sigma;
    
    // Only compute Greeks if we found a valid IV
    if (sigma > 0) {
        // 2. Compute original price with the computed IV
        double price = deviceOptionPrice(steps, s, k, rate, yield, time, sigma, type);
        
        // 3. Compute shifted prices for Greeks
        double S_up = s * (1.0 + dSpot);
        double S_down = s * (1.0 - dSpot);
        double sigma_up = sigma * (1.0 + dVol);
        double sigma_down = sigma * (1.0 - dVol);
        double reducedT = fmax(time - 1.0/252.0, 0.0);
        
        // Calculate shifted prices
        double price_S_up = deviceOptionPrice(steps, S_up, k, rate, yield, time, sigma, type);
        double price_S_down = deviceOptionPrice(steps, S_down, k, rate, yield, time, sigma, type);
        double price_sigma_up = deviceOptionPrice(steps, s, k, rate, yield, time, sigma_up, type);
        double price_sigma_down = deviceOptionPrice(steps, s, k, rate, yield, time, sigma_down, type);
        double price_T_reduced = deviceOptionPrice(steps, s, k, rate, yield, reducedT, sigma, type);
        
        // 4. Compute Greeks
        deltaResults[idx] = (price_S_up - price_S_down) / (2.0 * dSpot * s);
        gammaResults[idx] = (price_S_up - 2.0 * price + price_S_down) / (dSpot * dSpot * s * s);
        thetaResults[idx] = (price_T_reduced - price);
        vegaResults[idx] = (price_sigma_up - price_sigma_down) / (2.0 * dVol * sigma);
    } else {
        // Set invalid values for Greeks if IV computation failed
        deltaResults[idx] = NAN;
        gammaResults[idx] = NAN;
        thetaResults[idx] = NAN;
        vegaResults[idx] = NAN;
    }
}

// Shared memory version of option pricing for better performance
__global__ void sharedMemoryOptionPricingKernel(
    int steps, int batchSize,
    const double* S, const double* K,
    const double* r, const double* q,
    const double* T, const double* sigma,
    const int* optionType, double* prices) {
    
    extern __shared__ double shared_mem[];
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize) return;
    
    // Load option parameters to shared memory for faster access
    int tid = threadIdx.x;
    shared_mem[tid] = S[idx];               // S
    shared_mem[tid + blockDim.x] = K[idx];  // K
    shared_mem[tid + 2 * blockDim.x] = r[idx];  // r
    shared_mem[tid + 3 * blockDim.x] = q[idx];  // q
    shared_mem[tid + 4 * blockDim.x] = T[idx];  // T
    shared_mem[tid + 5 * blockDim.x] = sigma[idx];  // sigma
    
    // Use local variable for option type
    int type = optionType[idx];
    
    __syncthreads();  // Ensure all data is loaded before computation
    
    // Retrieve parameters from shared memory
    double s = shared_mem[tid];
    double k = shared_mem[tid + blockDim.x];
    double rate = shared_mem[tid + 2 * blockDim.x];
    double yield = shared_mem[tid + 3 * blockDim.x];
    double time = shared_mem[tid + 4 * blockDim.x];
    double vol = shared_mem[tid + 5 * blockDim.x];
    
    // Compute option price
    prices[idx] = deviceOptionPrice(steps, s, k, rate, yield, time, vol, type);
}