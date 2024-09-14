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

        CHECK_CUDA_ERROR(cudaMallocAsync(&d_S, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_K, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_r, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_q, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_T, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_optionType, batch_size * sizeof(int), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_prices, prices_size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_values, prices_size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_low, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_high, size, stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_mid, size, stream));
        // Copy data to device asynchronously
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_S, S, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_K, K, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_r, r, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_q, q, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_T, T, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_optionType, optionType, batch_size * sizeof(int), 
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
    }

    // Add an output buffer parameter
    void computeOptionPrices(double* d_output_price) {
        int threadsPerBlock = 1024; // Max threads per block for many GPUs
        int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;

        calculatePrice<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            steps, batch_size, d_output_price, d_S, d_K, d_r, d_q, d_T, 
            d_optionType, d_sigma, d_prices, d_values
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    // Function to retrieve option prices asynchronously
    void retrieveOptionPrices(std::vector<double>& host_price) {
        host_price.resize(batch_size);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(host_price.data(), d_price, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
    }

    // Function to calculate all Greeks
    std::vector<Greeks> calculateAllGreeks(const GreekParams& params) {
        std::vector<Greeks> greeks(batch_size, Greeks());

        // Define shifts for finite differences
        std::vector<double> shiftS = params.dSpot;    // Delta
        std::vector<double> shiftVol = params.dVol;   // Vega
        std::vector<double> shiftT = params.dTime;    // Theta
        std::vector<double> shiftR = params.dRate;    // Rho
        std::vector<double> shiftQ = params.dYield;   // Veta

        // Allocate device memory for shifted sigmas
        double *d_sigma_upS, *d_sigma_downS;
        double *d_sigma_upVol, *d_sigma_downVol;
        double *d_sigma_upT, *d_sigma_downT;
        double *d_sigma_upR, *d_sigma_downR;
        double *d_sigma_upQ, *d_sigma_downQ;

        // Allocate device memory for shifted prices
        double *d_price_upS, *d_price_downS;
        double *d_price_upVol, *d_price_downVol;
        double *d_price_upT, *d_price_downT;
        double *d_price_upR, *d_price_downR;
        double *d_price_upQ, *d_price_downQ;

        // Allocate all shifted sigmas
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma_upS, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma_downS, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma_upVol, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma_downVol, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma_upT, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma_downT, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma_upR, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma_downR, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma_upQ, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma_downQ, batch_size * sizeof(double), stream));

        // Allocate shifted price storage
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_upS, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_downS, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_upVol, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_downVol, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_upT, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_downT, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_upR, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_downR, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_upQ, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_downQ, batch_size * sizeof(double), stream));

        // Prepare shifted sigmas
        std::vector<double> sigma_upS(batch_size);
        std::vector<double> sigma_downS(batch_size);
        std::vector<double> sigma_upVol(batch_size);
        std::vector<double> sigma_downVol(batch_size);
        std::vector<double> sigma_upT(batch_size);
        std::vector<double> sigma_downT(batch_size);
        std::vector<double> sigma_upR(batch_size);
        std::vector<double> sigma_downR(batch_size);
        std::vector<double> sigma_upQ(batch_size);
        std::vector<double> sigma_downQ(batch_size);

        for (int i = 0; i < batch_size; ++i) {
            sigma_upS[i] = S[i] + shiftS[i];
            sigma_downS[i] = S[i] - shiftS[i];
            sigma_upVol[i] = params.dVol[i] + shiftVol[i];
            sigma_downVol[i] = params.dVol[i] - shiftVol[i];
            sigma_upT[i] = params.dTime[i] + shiftT[i];
            sigma_downT[i] = params.dTime[i] - shiftT[i];
            sigma_upR[i] = params.dRate[i] + shiftR[i];
            sigma_downR[i] = params.dRate[i] - shiftR[i];
            sigma_upQ[i] = params.dYield[i] + shiftQ[i];
            sigma_downQ[i] = params.dYield[i] - shiftQ[i];
        }

        // Copy shifted sigmas to device
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma_upS, sigma_upS.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma_downS, sigma_downS.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma_upVol, sigma_upVol.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma_downVol, sigma_downVol.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma_upT, sigma_upT.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma_downT, sigma_downT.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma_upR, sigma_upR.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma_downR, sigma_downR.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma_upQ, sigma_upQ.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma_downQ, sigma_downQ.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));

        // Launch kernels to compute shifted prices
        // Define a lambda to launch compute for each shifted parameter
        auto compute_shifted_prices = [&](double* d_sigma_shift, double* d_price_shift) {
            calculatePrice<<<(batch_size + 255) / 256, 256, 0, stream>>>(
                steps, batch_size, d_price_shift, d_S, d_K, d_r, d_q, d_T, 
                d_optionType, d_sigma_shift, d_prices, d_values
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
        };

        // Compute shifted prices
        compute_shifted_prices(d_sigma_upS, d_price_upS);
        compute_shifted_prices(d_sigma_downS, d_price_downS);
        compute_shifted_prices(d_sigma_upVol, d_price_upVol);
        compute_shifted_prices(d_sigma_downVol, d_price_downVol);
        compute_shifted_prices(d_sigma_upT, d_price_upT);
        compute_shifted_prices(d_sigma_downT, d_price_downT);
        compute_shifted_prices(d_sigma_upR, d_price_upR);
        compute_shifted_prices(d_sigma_downR, d_price_downR);
        compute_shifted_prices(d_sigma_upQ, d_price_upQ);
        compute_shifted_prices(d_sigma_downQ, d_price_downQ);

        // Synchronize to ensure all shifted price computations are complete
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        // Retrieve all shifted prices
        std::vector<double> price_upS(batch_size), price_downS(batch_size);
        std::vector<double> price_upVol(batch_size), price_downVol(batch_size);
        std::vector<double> price_upT(batch_size), price_downT(batch_size);
        std::vector<double> price_upR(batch_size), price_downR(batch_size);
        std::vector<double> price_upQ(batch_size), price_downQ(batch_size);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_upS.data(), d_price_upS, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_downS.data(), d_price_downS, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_upVol.data(), d_price_upVol, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_downVol.data(), d_price_downVol, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_upT.data(), d_price_upT, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_downT.data(), d_price_downT, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_upR.data(), d_price_upR, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_downR.data(), d_price_downR, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_upQ.data(), d_price_upQ, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_downQ.data(), d_price_downQ, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));

        // Synchronize to ensure all data is copied
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        // Calculate Greeks on host
        for (int i = 0; i < batch_size; ++i) {
            Greeks g;
            // First-order Greeks
            g.delta = (price_upS[i] - price_downS[i]) / (2.0 * params.dSpot[i]);
            g.vega = (price_upVol[i] - price_downVol[i]) / (2.0 * params.dVol[i]);
            g.theta = -(price_upT[i] - price_downT[i]) / (2.0 * params.dTime[i]);
            g.rho = (price_upR[i] - price_downR[i]) / (2.0 * params.dRate[i]);

            // Second-order Greeks
            g.gamma = (price_upS[i] - 2.0 * marketPrices[i] + price_downS[i]) 
                      / (params.dSpot[i] * params.dSpot[i]);
            g.vanna = (price_upVol[i] - price_downVol[i] - price_upS[i] + price_downS[i]) 
                      / (4.0 * params.dSpot[i] * params.dVol[i]);
            g.charm = (price_upT[i] - price_downT[i] - price_upS[i] + price_downS[i]) 
                      / (4.0 * params.dSpot[i] * params.dTime[i]);
            g.vomma = (price_upVol[i] - 2.0 * marketPrices[i] + price_downVol[i]) 
                      / (params.dVol[i] * params.dVol[i]);
            g.veta = (price_upQ[i] - price_downQ[i]) / (2.0 * params.dYield[i]);
            g.vera = (price_upR[i] - price_downR[i]) / (2.0 * params.dRate[i]);

            // Third-order Greeks
            g.speed = (price_upS[i] - 3.0 * price_upS[i] + 3.0 * price_downS[i] - price_downS[i]) 
                     / (2.0 * pow(params.dSpot[i], 3));
            g.zomma = (price_upVol[i] - price_downVol[i] - price_upS[i] + price_downS[i]) 
                      / (4.0 * params.dSpot[i] * params.dVol[i]);
            g.color = (price_upT[i] - 3.0 * price_upT[i] + 3.0 * price_downT[i] - price_downT[i]) 
                     / (2.0 * params.dSpot[i] * pow(params.dTime[i], 2));
            g.ultima = (price_upVol[i] - 3.0 * price_upVol[i] + 3.0 * price_downVol[i] - price_downVol[i]) 
                      / (2.0 * pow(params.dVol[i], 3));

            // Assign Greeks to vector
            greeks[i] = g;
        }

        // Free temporary device memory
        cudaFreeAsync(d_sigma_upS, stream);
        cudaFreeAsync(d_sigma_downS, stream);
        cudaFreeAsync(d_sigma_upVol, stream);
        cudaFreeAsync(d_sigma_downVol, stream);
        cudaFreeAsync(d_sigma_upT, stream);
        cudaFreeAsync(d_sigma_downT, stream);
        cudaFreeAsync(d_sigma_upR, stream);
        cudaFreeAsync(d_sigma_downR, stream);
        cudaFreeAsync(d_sigma_upQ, stream);
        cudaFreeAsync(d_sigma_downQ, stream);

        cudaFreeAsync(d_price_upS, stream);
        cudaFreeAsync(d_price_downS, stream);
        cudaFreeAsync(d_price_upVol, stream);
        cudaFreeAsync(d_price_downVol, stream);
        cudaFreeAsync(d_price_upT, stream);
        cudaFreeAsync(d_price_downT, stream);
        cudaFreeAsync(d_price_upR, stream);
        cudaFreeAsync(d_price_downR, stream);
        cudaFreeAsync(d_price_upQ, stream);
        cudaFreeAsync(d_price_downQ, stream);
        
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
    void computeImpliedVolatility(std::vector<double>& impliedVols) {
        impliedVols.resize(batch_size, -1.0); // Initialize with -1 indicating failure

        // Define root-finding parameters
        double sigma_low = 0.01;
        double sigma_high = 3.0;

        // Initialize sigma_low and sigma_high on host
        std::vector<double> sigma_low_vec(batch_size, sigma_low);
        std::vector<double> sigma_high_vec(batch_size, sigma_high);
    
        // Copy sigma_low and sigma_high to device
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma, sigma_low_vec.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        // Launch kernel to compute price_low
        computeOptionPrices(d_price_low); 

        // Wait for price_low to be computed
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        // Copy sigma_high to device
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma, sigma_high_vec.data(), 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyHostToDevice, stream));
        // Launch kernel to compute price_high
        computeOptionPrices(d_price_high); // This should write to d_price_high

        // Wait for price_high to be computed
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        // Retrieve prices from device
        std::vector<double> price_low(batch_size);
        std::vector<double> price_high(batch_size);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_low.data(), d_price_low, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(price_high.data(), d_price_high, 
                                        batch_size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, stream));

        // Synchronize to ensure data is copied
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        std::cout << "Price Low: " << price_low[0] << ", Price High: " << price_high[0] << std::endl;
        // Perform bisection on host
        for (int i = 0; i < batch_size; ++i) {
            double target = marketPrices[i];
            double f_low = price_low[i] - target;
            double f_high = price_high[i] - target;

            if (f_low * f_high > 0) {
                // Root not bracketed
                impliedVols[i] = -1.0;
                continue;
            }

            double a = sigma_low;
            double b = sigma_high;
            double fa_val = f_low;
            double fb_val = f_high;
            double c, fc_val;

            for (int iter = 0; iter < max_iter; ++iter) {
                c = 0.5 * (a + b);
                double sigma_mid = c;

                // Allocate device memory for sigma_mid
                double *d_sigma_mid;
                CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma_mid, batch_size * sizeof(double), stream));

                // Initialize sigma_mid on host
                std::vector<double> sigma_mid_vec(batch_size, sigma_mid);
                CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma_mid, sigma_mid_vec.data(), 
                                                batch_size * sizeof(double), 
                                                cudaMemcpyHostToDevice, stream));

                // Copy sigma_mid to d_sigma
                CHECK_CUDA_ERROR(cudaMemcpyAsync(d_sigma, d_sigma_mid, 
                                                batch_size * sizeof(double), 
                                                cudaMemcpyDeviceToDevice, stream));

                // Launch kernel to compute price_mid
                computeOptionPrices(d_price_mid); // This should write to d_price_mid

                // Wait for price_mid to be computed
                CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

                // Retrieve price_mid from device
                std::vector<double> price_mid(batch_size);
                CHECK_CUDA_ERROR(cudaMemcpyAsync(price_mid.data(), d_price_mid, 
                                                batch_size * sizeof(double), 
                                                cudaMemcpyDeviceToHost, stream));
                CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

                fc_val = price_mid[i] - target;

                if (fabs(fc_val) < tol) {
                    impliedVols[i] = c;
                    CHECK_CUDA_ERROR(cudaFreeAsync(d_sigma_mid, stream));
                    break;
                }

                if (fa_val * fc_val < 0) {
                    b = c;
                    fb_val = fc_val;
                } else {
                    a = c;
                    fa_val = fc_val;
                }

                CHECK_CUDA_ERROR(cudaFreeAsync(d_sigma_mid, stream));
            }
        }

        // Free temporary device memory
    }

};

#endif // CRROPTIONPRICER_H
