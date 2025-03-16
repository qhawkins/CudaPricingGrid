#ifndef OPTION_PRICER_CUH
#define OPTION_PRICER_CUH

#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include "structs.h"

// Macro for checking CUDA errors
#define CHECK_CUDA_ERROR(val) { checkCudaError((val), __FILE__, __LINE__); }
inline void checkCudaError(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << " at: " << file << ":" << line << std::endl;
        exit(result);
    }
}

// Enhanced CRROptionPricer class with optimized memory management
class OptimizedCRROptionPricer {
private:
    int batch_size;
    bool use_pinned_memory;
    
    // Host pinned memory for all inputs
    double *h_S, *h_K, *h_r, *h_q, *h_T, *h_marketPrices;
    int *h_optionType;
    
    // Device memory pointers
    double *d_S, *d_K, *d_r, *d_q, *d_T;
    double *d_price, *d_prices, *d_values;
    double *d_sigma;
    double *d_price_low, *d_price_high, *d_price_mid;
    double *d_ivResults, *d_marketPrices;
    int *d_optionType;
    
    // Stream for asynchronous operations
    cudaStream_t stream;
    
    // Parameters
    int steps;
    double tol;
    int max_iter;
    
    // Greeks parameter buffers
    double *d_greeks_delta, *d_greeks_gamma, *d_greeks_theta, *d_greeks_vega;
    
    // Private methods
    void allocateMemory();
    void freeMemory();

public:
    OptimizedCRROptionPricer(int batch_size_, int steps_, 
                            double tol_, int max_iter_, 
                            cudaStream_t stream_, bool use_pinned_memory_ = true);
    ~OptimizedCRROptionPricer();
    
    // Set data for computation
    void setData(const std::vector<double>& S, const std::vector<double>& K,
                 const std::vector<double>& r, const std::vector<double>& q,
                 const std::vector<double>& T, const std::vector<int>& optionType,
                 const std::vector<double>& marketPrices);
    
    // Compute implied volatility and greeks in a single batch
    void computeAllInOne(std::vector<double>& impliedVols, 
                         std::vector<Greeks>& greeks,
                         double dSpot = 0.01, double dVol = 0.01);
                         
    // Fused kernel method to compute both IV and Greeks
    void launchFusedComputation();
};

// Launch a fused kernel to compute IV and Greeks directly
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

#endif // OPTION_PRICER_CUH