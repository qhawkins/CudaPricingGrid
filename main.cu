#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include "structs.h"
#include "kernels.cuh"
#include "CRROptionPricer.cuh"
#include "concurrent_queue.h"
#include "threadpool.h"

struct BatchResult {
    std::vector<OptionData> data;
    bool is_sentinel;
};

// Function to process a single batch
void processBatch(std::vector<OptionData> batch) 
{
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int batch_size = batch.size();
    std::vector<double> S(batch_size);
    std::vector<double> K(batch_size);
    std::vector<double> r(batch_size);
    std::vector<double> q(batch_size, 0.0); // Assuming no dividends
    std::vector<double> T(batch_size);
    std::vector<int> type(batch_size);
    std::vector<double> marketPrices(batch_size);

    std::vector<double> dSpot(batch_size);
    std::vector<double> dStrike(batch_size);
    std::vector<double> dRate(batch_size, 0.0001);
    std::vector<double> dYield(batch_size, 0.0001);
    std::vector<double> dTime(batch_size, 1.0 / 365.0);
    std::vector<double> dVol(batch_size, 0.01);

    for (int i = 0; i < batch_size; i++) {
        S[i] = batch[i].underlying_price;
        K[i] = batch[i].strike_price;
        r[i] = batch[i].rfr;
        T[i] = batch[i].years_to_expiration;
        type[i] = batch[i].option_type;
        marketPrices[i] = batch[i].market_price;
        dSpot[i] = 0.01 * S[i];
        dStrike[i] = 0.01 * K[i];
    }

    // Initialize pricer
    CRROptionPricer* pricer = new CRROptionPricer(batch_size, marketPrices.data(), S.data(), K.data(), 
                          r.data(), q.data(), T.data(), 1000, type.data(), 
                          1e-5, 1000, stream);
    // Compute implied volatilities
    std::vector<double> impliedVols;
    impliedVols = pricer->computeImpliedVolatilityDevice();
    // Define GreekParams
    GreekParams params;
    params.dSpot = dSpot;
    params.dStrike = dStrike;
    params.dRate = dRate;
    params.dYield = dYield;
    params.dTime = dTime;
    params.dVol = dVol;
    //std::cout << "Implied Vols: " << impliedVols.size() << std::endl;
    // Calculate Greeks
    std::vector<Greeks> greeks = pricer->calculateAllGreeks(params, impliedVols);

    for (int i = 0; i < batch_size; i++) {
        OptionData &option = batch[i];
        //option.modelPrice = host_price[i];
        option.impliedVolatility = impliedVols[i];
        if (option.impliedVolatility < 0) {
            // Handle failure to compute IV
            continue;
        }
        // Assign Greeks
        option.delta = greeks[i].delta;
        option.theta = greeks[i].theta;
        option.vega = greeks[i].vega;
        option.rho = greeks[i].rho;

        // Second-order Greeks
        option.gamma = greeks[i].gamma;
        option.vanna = greeks[i].vanna;
        option.charm = greeks[i].charm;
        option.vomma = greeks[i].vomma;
        option.veta = greeks[i].veta;
        option.vera = greeks[i].vera;

        // Third-order Greeks
        option.speed = greeks[i].speed;
        option.zomma = greeks[i].zomma;
        option.color = greeks[i].color;
        option.ultima = greeks[i].ultima;

    }
    // Clean up
    delete pricer;
    cudaStreamDestroy(stream);

}

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
                //if (option.years_to_expiration <= .01) {
                //    continue;
                //}
                option.rfr = std::stod(tokens[7])/100;
                char contract_type = tokens[4][0];
                option.contract_id = tokens[5];
                option.timestamp = tokens[6];
                option.option_type = (contract_type == 'C' || contract_type == 'c') ? 0 : 1;
                //double itm_perc = (option.option_type == "call") ? (option.underlying_price - option.strike_price) / option.underlying_price : (option.strike_price - option.underlying_price) / option.underlying_price;
                //std::cout << "ITM Percentage: " << itm_perc << std::endl;
                //if (itm_perc < -.05 || itm_perc > 0.05) {
                //    continue;
               // }
                options.push_back(option);
                if (options.size() == 16384) {
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

int main() {
    std::string input_filename = "/home/qhawkins/Desktop/GMEStudy/timed_opra_clean_mc.csv";
    std::string output_filename = "/home/qhawkins/Desktop/GMEStudy/implied_volatilities_mc.csv";

    // Step 1: Read options from CSV
    std::vector<OptionData> options = read_csv(input_filename);
    if (options.empty()) {
        std::cerr << "No options data found. Exiting." << std::endl;
        return 1;
    }

    // Step 2: Define batch size and calculate number of batches
    const int BATCH_SIZE = 128;
    int num_batches = (options.size() + BATCH_SIZE - 1) / BATCH_SIZE;

    // Step 3: Prepare futures for asynchronous tasks
    std::vector<std::future<std::vector<OptionData>>> futures;
    futures.reserve(num_batches); // Reserve space to avoid reallocations

    auto start_time = std::chrono::high_resolution_clock::now();

    // Step 4: Launch asynchronous tasks for each batch
    for(int i = 0; i < num_batches; ++i){
        size_t start_idx = i * BATCH_SIZE;
        size_t end_idx = std::min(start_idx + BATCH_SIZE, (size_t)options.size());
        std::vector<OptionData> batch(options.begin() + start_idx, options.begin() + end_idx);

        // Launch async task with launch::async to ensure it runs asynchronously
        futures.emplace_back(
            std::async(std::launch::async, processBatch, batch)
        );
    }

    // Step 5: Collect results from all futures
    std::vector<OptionData> final_results;
    final_results.reserve(options.size()); // Reserve space for efficiency
    std::mutex results_mutex; // Mutex to protect final_results in case of concurrent access

    for(auto &fut : futures){
        try {
            // Retrieve the processed batch
            std::vector<OptionData> processed_batch = fut.get();

            // Append the processed batch to final_results
            // Using a mutex in case multiple threads try to append simultaneously
            // However, since this loop runs in a single thread, mutex might not be necessary
            // It's included here for demonstration and future-proofing
            std::lock_guard<std::mutex> lock(results_mutex);
            final_results.insert(final_results.end(),
                                 std::make_move_iterator(processed_batch.begin()),
                                 std::make_move_iterator(processed_batch.end()));
        } catch(const std::exception& e) {
            std::cerr << "Error processing batch: " << e.what() << std::endl;
            // Handle exceptions as needed (e.g., skip this batch, retry, etc.)
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Step 6: Output or save results as needed
    // Example: Print the first few results
    size_t print_limit = std::min<size_t>(5, final_results.size());
    for (size_t i = 0; i < print_limit; ++i) {
        const auto& result = final_results[i];
        std::cout << "Contract ID: " << result.contract_id << std::endl;
        std::cout << "Timestamp: " << result.timestamp << std::endl;
        std::cout << "Market Price: " << result.market_price << std::endl;
        std::cout << "Implied Volatility: " << result.impliedVolatility << std::endl;
        std::cout << "Delta: " << result.delta << std::endl;
        std::cout << "Gamma: " << result.gamma << std::endl;
        std::cout << "Theta: " << result.theta << std::endl;
        std::cout << "Vega: " << result.vega << std::endl;
        std::cout << "Rho: " << result.rho << std::endl;
        std::cout << "Vanna: " << result.vanna << std::endl;
        std::cout << "Charm: " << result.charm << std::endl;
        std::cout << "Vomma: " << result.vomma << std::endl;
        std::cout << "Veta: " << result.veta << std::endl;
        std::cout << "Vera: " << result.vera << std::endl;
        std::cout << "Speed: " << result.speed << std::endl;
        std::cout << "Zomma: " << result.zomma << std::endl;
        std::cout << "Color: " << result.color << std::endl;
        std::cout << "Ultima: " << result.ultima << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;

    return 0;
}