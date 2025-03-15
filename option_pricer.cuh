#ifndef OPTION_PRICER_CUH
#define OPTION_PRICER_CUH

#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "structs.h"

// Define a macro for CUDA error checking
#define CHECK_CUDA_ERROR(val) { checkCudaError((val), __FILE__, __LINE__); }

// Helper function for CUDA error checking
inline void checkCudaError(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

// Forward declarations of CUDA kernels
__global__ void priceOptionsKernel(
    int steps, int num_options, 
    double* S, double* K, double* r, double* q, 
    double* T, double* sigma, int* optionType, 
    double* prices
);

__global__ void computeImpliedVolatilityKernel(
    int steps, int num_options, 
    double* marketPrices, double* S, double* K, 
    double* r, double* q, double* T, 
    int* optionType, double* ivResults, 
    double tol, int max_iter
);

// CRROptionPricer class declaration
class CRROptionPricer {
private:
    // Host data
    double *S, *K, *r, *q, *T, *marketPrices;
    int *optionType;
    int steps, max_iter;
    double tol;
    int batch_size;

    // Device data
    double *d_S, *d_K, *d_r, *d_q, *d_T;
    int *d_optionType;
    double *d_price, *d_prices, *d_values;
    double *d_price_low, *d_price_high, *d_price_mid;
    double *d_sigma;
    double *d_ivResults;
    double *d_marketPrices;
    double *d_shifted_S, *d_shifted_K, *d_shifted_r, *d_shifted_q, *d_shifted_T, *d_shifted_sigma;
    int *d_shifted_optionType;
    double *d_shifted_prices;

    cudaStream_t stream;

public:
    // Constructor
    CRROptionPricer(int batch_size_, double* marketPrices_, double* S_, double* K_, 
                   double* r_, double* q_, double* T_, int steps_, 
                   int* type_, double tol_, int max_iter_, cudaStream_t stream_);

    // Destructor
    ~CRROptionPricer();

    // Function to retrieve option prices asynchronously
    void retrieveOptionPrices(std::vector<double>& host_price);

    // Function to calculate all Greeks
    std::vector<Greeks> calculateAllGreeks(const GreekParams& params, const std::vector<double>& sigma);

    // Function to print Greeks for an option
    void printGreeks(const Greeks& greeks);

    // Function to compute implied volatility
    std::vector<double> computeImpliedVolatilityDevice();
};

#endif // OPTION_PRICER_CUH