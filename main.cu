#include <cmath>
#include <stdexcept>
#include "structs.h"
#include "grid.cuh"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

__global__ void updateBoundaryConditions(float* V, float K, int N, int t, int M, float r, float dt, int contractType) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // S = 0 boundary
        if (contractType == 0) { // Call option
            V[0 * (M + 1) + t] = 0.0f;
        } else { // Put option
            V[0 * (M + 1) + t] = K * exp(-r * (M - t) * dt);
        }
    } else if (idx == 1) {
        // S = Smax boundary
        if (contractType == 0) { // Call option
            V[N * (M + 1) + t] = V[N * (M + 1) + M] - K * exp(-r * (M - t) * dt);
        } else { // Put option
            V[N * (M + 1) + t] = 0.0f;
        }
    }
}

GridParams initializeGridParameters(float S0, float K, float T, float r, float sigma, int M, int N) {
    GridParams params;

    // Assign input parameters
    params.S0 = S0;
    params.K = K;
    params.T = T;
    params.r = r;
    params.sigma = sigma;
    params.M = M;
    params.N = N;

    // Calculate derived parameters
    params.dt = T / M;
    params.Smax = S0 * exp(2 * sigma * sqrt(T));  // Rule of thumb for max stock price
    params.dS = params.Smax / N;

    // Validate parameters
    if (S0 <= 0 || K <= 0 || T <= 0 || r < 0 || sigma <= 0 || M <= 0 || N <= 0) {
        throw std::invalid_argument("Invalid input parameters");
    }

    return params;
}

float priceAmericanOption(const GridParams& params) {
    std::cout << "S0: " << params.S0 << std::endl;

    // Allocate host memory
    float *h_V, *h_S;
    allocateHostMemory(params, h_V, h_S);

    // Initialize option values
    initializeOptionValues(h_V, h_S, params);

    std::cout << "Initial V values:" << std::endl;
    for (int i = 0; i <= params.N; i += params.N / 10) {
        std::cout << "V[" << i << "] = " << h_V[i] << ", S[" << i << "] = " << h_S[i] << std::endl;
    }

    // Allocate device memory
    float *d_V, *d_S, *d_a, *d_b, *d_c, *d_y;
    allocateDeviceMemory(params, d_V, d_S);
    cudaMalloc((void**)&d_a, params.N * sizeof(float));
    cudaMalloc((void**)&d_b, params.N * sizeof(float));
    cudaMalloc((void**)&d_c, params.N * sizeof(float));
    cudaMalloc((void**)&d_y, (params.N + 1) * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_S, h_S, (params.N + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, (params.M + 1) * (params.N + 1) * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up tridiagonal matrix coefficients
    //launchSetupTridiagonalMatrix(d_a, d_b, d_c, d_S, params);
    //std::cout << "params sigma: " <<   params.sigma << std::endl;
    // Main pricing loop
    for (int t = params.M - 1; t >= 0; --t) {
        // Update boundary conditions
        updateBoundaryConditions<<<1, 2>>>(d_V, params.K, params.N, t, params.M, params.r, params.dt, params.contractType);

        // Solve tridiagonal system
        launchThomasSolver(d_a, d_b, d_c, &d_V[0 * (params.M + 1) + (t+1)], &d_V[0 * (params.M + 1) + t], params);

        // Apply early exercise condition
        launchApplyEarlyExercise(d_V, d_S, params, t);

        // Debug: Check values every 100 time steps
        if (t % 100 == 0 || t == 0) {
            cudaMemcpy(h_V, d_V, (params.N + 1) * (params.M + 1) * sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "V values at t = " << t << ":" << std::endl;
            for (int j = 0; j <= params.N; j += params.N / 10) {
                std::cout << "V[" << j << "] = " << h_V[j * (params.M + 1) + t] << ", S[" << j << "] = " << h_S[j] << std::endl;
            }
        }
    }

    // Copy final results back to host
    cudaMemcpy(h_V, d_V, (params.M + 1) * (params.N + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "V array at t=0 (first few values):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "V[" << i << "] = " << h_V[i] << std::endl;
    }

    // Interpolate to get the option price at S0
    float option_price = interpolatePrice(params.S0, h_S, h_V, params.N, params.M);


    // Free device memory
    cudaFree(d_V);
    cudaFree(d_S);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_y);

    // Free host memory
    free(h_V);
    free(h_S);

    return option_price;
}

std::atomic<int> completed_calculations(0);
std::chrono::steady_clock::time_point start_time;

std::vector<OptionData> read_csv(const std::string& filename) {
    std::vector<OptionData> options;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return options;
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        OptionData option;
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(iss, token, ',')) {
            //skip the first line
            if (token == "Contract") {
                break;
            }
            tokens.push_back(token);
        }

        if (tokens.size() == 8) {
            try {
                option.market_price = std::stod(tokens[0]);
                option.strike_price = std::stod(tokens[1]);
                option.underlying_price = std::stod(tokens[2]);
                option.years_to_expiration = std::stod(tokens[3]);
                if (option.years_to_expiration <= .01) {
                    continue;
                }
                option.rfr = std::stod(tokens[7])/100;
                char contract_type = tokens[4][0];
                option.contract_id = tokens[5];
                option.timestamp = tokens[6];
                option.option_type = (contract_type == 'C' || contract_type == 'c') ? "call" : "put";
                double itm_perc = (option.option_type == "call") ? (option.underlying_price - option.strike_price) / option.underlying_price : (option.strike_price - option.underlying_price) / option.underlying_price;
                //std::cout << "ITM Percentage: " << itm_perc << std::endl;
                if (itm_perc < -.05 || itm_perc > 0.05) {
                    continue;
                }
                options.push_back(option);
                if (options.size() == 256) {
                    break;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing line: " << line << " - " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Error parsing line: " << line << " - Incorrect number of fields" << std::endl;
            std::cout << tokens.size();
            exit(1938);
        }
    }

    file.close();
    return options;
}

float calculate_IV(GridParams params)
{
    float lower_bound = 0.0001;
    float upper_bound = 2.0;  // Reduced from 5.0
    float tolerance = 1e-5;
    int max_iterations = 512;
    float sigma;
    
    for (int i = 0; i < max_iterations; ++i)
    {
        if (i == 0) {
            sigma = 0.125;
        }
        else {
            sigma = (lower_bound + upper_bound) / 2;
        }
        params.sigma = sigma;
        params.dt = params.T / params.M;
        params.Smax = params.S0 * exp(2 * sigma * sqrt(params.T));
        params.dS = log(params.Smax / params.S0) / (params.N / 2.0f);

        std::cout << "Iteration " << i << ": Sigma = " << params.sigma 
                  << ", Smax = " << params.Smax
                  << ", dS = " << params.dS << std::endl;

        float simulated_price = priceAmericanOption(params);

        std::cout << "Simulated Price: " << simulated_price 
                  << ", Target Price: " << params.optionPrice 
                  << ", Difference: " << fabs(simulated_price - params.optionPrice) << std::endl;
        std::cout << "Simulated Price: " << simulated_price << " Sigma: " << params.sigma << std::endl;
        if (fabs(simulated_price - params.optionPrice) < tolerance) {
            //std::cout << "Converged after " << i << " iterations" << std::endl;
            return sigma;
        }
        
        if (simulated_price > params.optionPrice) {
            upper_bound = sigma;
        }
        else {
            lower_bound = sigma;                
        }
        
        if (upper_bound - lower_bound < tolerance) {
            //std::cout << "Converged after " << i << " iterations" << std::endl;
            return sigma;
        }
        if (sigma > 4) {
            exit(100);
        }
    }
    return -1; // Indicate failure to converge
}

int main() {
        
    std::string input_filename = "/home/qhawkins/Desktop/GMEStudy/timed_opra_clean_mc.csv";
    std::string output_filename = "/home/qhawkins/Desktop/GMEStudy/implied_volatilities_mc.csv";
    std::vector<OptionData> options = read_csv(input_filename);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < options.size(); i++) {
        OptionData& option = options[i];
        try {
            GridParams params;
            params.S0 = option.underlying_price;
            params.K = option.strike_price;
            params.T = option.years_to_expiration;
            params.r = option.rfr;
            params.sigma = option.impliedVolatility;
            params.M = 1000;
            params.N = 1000;
            params.contractType = (option.option_type == "call") ? 0 : 1;
            params.optionPrice = option.market_price;

            float iv = calculate_IV(params);

            option.impliedVolatility = iv;
            break;
        } catch (const std::exception& e) {
            std::cerr << "Error calculating option price: " << e.what() << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Elapsed time (s): " << duration_ns.count()/1e9 << " s\n";
    std::cout << "Time per calculation (ns): " << duration_ns.count() / options.size() << std::endl;
    for (int i = 128; i < 148; i++) {
        std::cout << "Option " << i << " Contract Type: " << options[i].option_type << " IV: " << options[i].impliedVolatility << " Market Price: " << options[i].market_price << " Strike Price: " << options[i].strike_price << " Underlying Price: " << options[i].underlying_price << " Years to Expiration: " << options[i].years_to_expiration << " RFR: " << options[i].rfr << " Delta: " << options[i].delta << " Gamma: " << options[i].gamma << " Theta: " << options[i].theta << " Vega: " << options[i].vega << " Rho: " << options[i].rho << " Vanna: " << options[i].vanna << " Charm: " << options[i].charm << " Vomma: " << options[i].vomma << " Veta: " << options[i].veta << " Speed: " << options[i].speed << " Zomma: " << options[i].zomma << " Color: " << options[i].color << " Ultima: " << options[i].ultima << std::endl;
        std::cout << "\n\n\n" << std::endl;
    }
}