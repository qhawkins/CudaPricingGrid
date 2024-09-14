#ifndef CRROPTIONPRICER_CUH
#define CRROPTIONPRICER_CUH

#include "kernels.cuh"
#include "structs.h"
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <functional>
#include <cmath>

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Helper function for option pricing (forward declaration)
double optionPriceFunction(int steps, double S, double K, double r, double q, double T, double sigma, int optionType, cudaStream_t stream);

// CRROptionPricer class definition
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



    // Stream
    cudaStream_t stream;

public:
    CRROptionPricer(int batch_size_, double* marketPrices_, double* S_, double* K_, 
                   double* r_, double* q_, double* T_, int steps_, 
                   int* type_, double tol_, int max_iter_, cudaStream_t stream_)
        : batch_size(batch_size_), marketPrices(marketPrices_), S(S_), K(K_), 
          r(r_), q(q_), T(T_), steps(steps_), optionType(type_), 
          tol(tol_), max_iter(max_iter_), stream(stream_) 
    {
        // Allocate device memory asynchronously
        size_t size = batch_size * sizeof(double);
        size_t steps_plus_one = steps + 1;
        size_t prices_size = batch_size * steps_plus_one * sizeof(double);
        size_t int_size = batch_size * sizeof(int);

        CHECK_CUDA_ERROR(cudaMallocAsync(&d_S, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_K, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_r, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_q, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_T, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_prices, prices_size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_values, prices_size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_low, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_high, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_mid, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_optionType, int_size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_ivResults, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_marketPrices, size, stream));

        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_S, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_K, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_r, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_q, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_T, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_sigma, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_optionType, int_size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_prices, size, stream));

        // Copy data to device asynchronously
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_S, S, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_K, K, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_r, r, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_q, q, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_T, T, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_optionType, optionType, batch_size * sizeof(int), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_marketPrices, marketPrices, size, 
                                        cudaMemcpyHostToDevice, stream));

    }

    ~CRROptionPricer() {
        // Free device memory asynchronously
        cudaFreeAsync(d_S, stream);
        cudaFreeAsync(d_K, stream);
        cudaFreeAsync(d_r, stream);
        cudaFreeAsync(d_q, stream);
        cudaFreeAsync(d_T, stream);
        cudaFreeAsync(d_optionType, stream);
        cudaFreeAsync(d_price, stream);
        cudaFreeAsync(d_prices, stream);
        cudaFreeAsync(d_values, stream);
        cudaFreeAsync(d_sigma, stream);
        cudaFreeAsync(d_price_low, stream);
        cudaFreeAsync(d_price_high, stream);
        cudaFreeAsync(d_price_mid, stream);
        cudaFreeAsync(d_ivResults, stream);
        cudaFreeAsync(d_marketPrices, stream);

        cudaFreeAsync(d_shifted_S, stream);
        cudaFreeAsync(d_shifted_K, stream);
        cudaFreeAsync(d_shifted_r, stream);
        cudaFreeAsync(d_shifted_q, stream);
        cudaFreeAsync(d_shifted_T, stream);
        cudaFreeAsync(d_shifted_sigma, stream);
        cudaFreeAsync(d_shifted_optionType, stream);
        cudaFreeAsync(d_shifted_prices, stream);
    }

    // Function to retrieve option prices asynchronously
    void retrieveOptionPrices(std::vector<double>& host_price) {
        host_price.resize(batch_size);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(host_price.data(), d_price, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
    }

    std::vector<Greeks> calculateAllGreeks(const GreekParams& params, const std::vector<double>& sigma) {
        std::vector<Greeks> greeks(batch_size, Greeks());

        double *d_greek_prices;
        double *d_sigma;
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_greek_prices, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma, batch_size * sizeof(double), stream));

        // Copy sigma to device
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma, sigma.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));

        // Step 1: Price Original Options
        // Launch pricing kernel for original options
        priceOptionsKernel<<<(batch_size + 255) / 256, 256, 0, stream>>>(
            steps, batch_size, d_S, d_K, d_r, d_q, d_T, 
            d_sigma, d_optionType, d_greek_prices
        );
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Copy original prices back to host
        std::vector<double> price_original(batch_size);
        //std::cout << "batch size: " << batch_size << std::endl;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_original.data(), d_greek_prices, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        //std::cout << "Original Prices: " << price_original[0] << std::endl;

        // Step 2: Prepare Shifted Parameters
        std::vector<double> shifted_S_up(batch_size);
        std::vector<double> shifted_S_down(batch_size);
        std::vector<double> shifted_sigma_up(batch_size);
        std::vector<double> shifted_sigma_down(batch_size);

        for(int i =0; i < batch_size; ++i){
            shifted_S_up[i] = S[i] + params.dSpot[i];
            shifted_S_down[i] = S[i] - params.dSpot[i];
            shifted_sigma_up[i] = sigma[i] + params.dVol[i];
            shifted_sigma_down[i] = sigma[i] - params.dVol[i];
        }

        // Step 3: Allocate Host Arrays for All Shifted Parameters
        // Total shifted parameter sets: 4 * batch_size (S_up, S_down, sigma_up, sigma_down)
        int total_shifted = 4 * batch_size;
        std::vector<double> shifted_S_total(total_shifted);
        std::vector<double> shifted_sigma_total(total_shifted);
        std::vector<int> shifted_optionType_total(total_shifted);
        std::vector<double> shifted_K_total(total_shifted);
        std::vector<double> shifted_r_total(total_shifted);
        std::vector<double> shifted_q_total(total_shifted);
        std::vector<double> shifted_T_total(total_shifted);

        for(int i =0; i < batch_size; ++i){
            // Shift S up
            shifted_S_total[i] = shifted_S_up[i];
            shifted_sigma_total[i] = sigma[i];
            shifted_optionType_total[i] = optionType[i];
            shifted_K_total[i] = K[i];
            shifted_r_total[i] = r[i];
            shifted_q_total[i] = q[i];
            shifted_T_total[i] = T[i];
            
            // Shift S down
            shifted_S_total[batch_size + i] = shifted_S_down[i];
            shifted_sigma_total[batch_size + i] = sigma[i];
            shifted_optionType_total[batch_size + i] = optionType[i];
            shifted_K_total[batch_size + i] = K[i];
            shifted_r_total[batch_size + i] = r[i];
            shifted_q_total[batch_size + i] = q[i];
            shifted_T_total[batch_size + i] = T[i];
            
            // Shift sigma up
            shifted_S_total[2 * batch_size + i] = S[i];
            shifted_sigma_total[2 * batch_size + i] = shifted_sigma_up[i];
            shifted_optionType_total[2 * batch_size + i] = optionType[i];
            shifted_K_total[2 * batch_size + i] = K[i];
            shifted_r_total[2 * batch_size + i] = r[i];
            shifted_q_total[2 * batch_size + i] = q[i];
            shifted_T_total[2 * batch_size + i] = T[i];
            
            // Shift sigma down
            shifted_S_total[3 * batch_size + i] = S[i];
            shifted_sigma_total[3 * batch_size + i] = shifted_sigma_down[i];
            shifted_optionType_total[3 * batch_size + i] = optionType[i];
            shifted_K_total[3 * batch_size + i] = K[i];
            shifted_r_total[3 * batch_size + i] = r[i];
            shifted_q_total[3 * batch_size + i] = q[i];
            shifted_T_total[3 * batch_size + i] = T[i];
        }

        // Step 4: Allocate Device Memory for Shifted Parameters and Shifted Prices
        double *d_shifted_S_total, *d_shifted_sigma_total;
        int *d_shifted_optionType_total;
        double *d_shifted_K_total, *d_shifted_r_total, *d_shifted_q_total, *d_shifted_T_total;
        double *d_shifted_prices_total;
        
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_S_total, total_shifted * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_sigma_total, total_shifted * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_optionType_total, total_shifted * sizeof(int), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_K_total, total_shifted * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_r_total, total_shifted * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_q_total, total_shifted * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_T_total, total_shifted * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_shifted_prices_total, total_shifted * sizeof(double), stream));
        
        // Step 5: Copy All Shifted Parameters to Device
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_shifted_S_total, shifted_S_total.data(), 
                                        total_shifted * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_shifted_sigma_total, shifted_sigma_total.data(), 
                                        total_shifted * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_shifted_optionType_total, shifted_optionType_total.data(), 
                                        total_shifted * sizeof(int), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_shifted_K_total, shifted_K_total.data(), 
                                        total_shifted * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_shifted_r_total, shifted_r_total.data(), 
                                        total_shifted * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_shifted_q_total, shifted_q_total.data(), 
                                        total_shifted * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_shifted_T_total, shifted_T_total.data(), 
                                        total_shifted * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        
        // Step 6: Launch Pricing Kernel for All Shifted Options
        priceOptionsKernel<<<(total_shifted + 255) / 256, 256, 0, stream>>>(
            steps, total_shifted, d_shifted_S_total, d_shifted_K_total, 
            d_shifted_r_total, d_shifted_q_total, d_shifted_T_total, 
            d_shifted_sigma_total, d_shifted_optionType_total, d_shifted_prices_total
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Step 7: Copy All Shifted Prices Back to Host
        std::vector<double> shifted_prices_total(total_shifted);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(shifted_prices_total.data(), d_shifted_prices_total, 
                                        total_shifted * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        
        // Step 8: Compute Greeks
        for(int i = 0; i < batch_size; ++i){
            // Indices for shifted prices
            int S_up_idx = i;
            int S_down_idx = batch_size + i;
            int sigma_up_idx = 2 * batch_size + i;
            int sigma_down_idx = 3 * batch_size + i;
            
            double price_S_up = shifted_prices_total[S_up_idx];
            double price_S_down = shifted_prices_total[S_down_idx];
            double price_sigma_up = shifted_prices_total[sigma_up_idx];
            double price_sigma_down = shifted_prices_total[sigma_down_idx];
            
            double original_price = price_original[i];
            
            // Compute Delta
            greeks[i].delta = (price_S_up - price_S_down) / (2.0 * params.dSpot[i]);
            
            // Compute Vega
            greeks[i].vega = (price_sigma_up - price_sigma_down) / (2.0 * params.dVol[i]);
            
            // Compute Gamma
            greeks[i].gamma = (price_S_up - 2.0 * original_price + price_S_down) / (params.dSpot[i] * params.dSpot[i]);
        }
        
        // Step 9: Free Temporary Device Memory
        cudaFree(d_shifted_S_total);
        cudaFree(d_shifted_sigma_total);
        cudaFree(d_shifted_optionType_total);
        cudaFree(d_shifted_K_total);
        cudaFree(d_shifted_r_total);
        cudaFree(d_shifted_q_total);
        cudaFree(d_shifted_T_total);
        cudaFree(d_shifted_prices_total);
        cudaFree(d_greek_prices);
        cudaFree(d_sigma);
        
        return greeks;
    }

    // Function to print Greeks for an option
    void printGreeks(const Greeks& greeks) {
        std::cout << "First-order Greeks:" << std::endl;
        std::cout << "Delta: " << greeks.delta << std::endl;
        std::cout << "Theta: " << greeks.theta << std::endl;
        std::cout << "Vega: " << greeks.vega << std::endl;
        std::cout << "Rho: " << greeks.rho << std::endl;

        std::cout << "\nSecond-order Greeks:" << std::endl;
        std::cout << "Gamma: " << greeks.gamma << std::endl;
        std::cout << "Vanna: " << greeks.vanna << std::endl;
        std::cout << "Charm: " << greeks.charm << std::endl;
        std::cout << "Vomma: " << greeks.vomma << std::endl;
        std::cout << "Veta: " << greeks.veta << std::endl;
        std::cout << "Vera: " << greeks.vera << std::endl;

        std::cout << "\nThird-order Greeks:" << std::endl;
        std::cout << "Speed: " << greeks.speed << std::endl;
        std::cout << "Zomma: " << greeks.zomma << std::endl;
        std::cout << "Color: " << greeks.color << std::endl;
        std::cout << "Ultima: " << greeks.ultima << std::endl;
    }

    // Function to compute implied volatility for each option using bisection method
    std::vector<double> computeImpliedVolatilityDevice() {
        std::vector<double> impliedVols(batch_size, -1.0); // Initialize with -1 indicating failure

        // Launch the IV computation kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
        computeImpliedVolatilityKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            steps, batch_size, d_marketPrices, d_S, d_K, d_r, d_q, d_T, 
            d_optionType, d_ivResults, tol, max_iter
        );
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Asynchronously copy IV results back to host
        CHECK_CUDA_ERROR(cudaMemcpyAsync(impliedVols.data(), d_ivResults, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));

        // Synchronize to ensure IV computation is complete
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        //std::cout << "Implied Volatilities: " << impliedVols[0] << std::endl;

        return impliedVols;
    }
};

#endif // CRROPTIONPRICER_H
