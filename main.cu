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
#include "option_pricer.cuh"
#include "gpu_optimizer.cuh"

// Thread-safe queue for parallel processing
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cond;
    bool done = false;

public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::move(item));
        cond.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this] { return !queue.empty() || done; });
        
        if (queue.empty() && done) {
            return false;
        }
        
        item = std::move(queue.front());
        queue.pop();
        return true;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
        cond.notify_all();
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }
};

// Function to read options from CSV file
std::vector<OptionData> read_csv(const std::string &filename) {
    std::vector<OptionData> options;
    std::ifstream file(filename);
    std::string line;
    bool skip_flag = true;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return options;
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        OptionData option;
        std::string token;
        std::vector<std::string> tokens;

        if (skip_flag) {
            skip_flag = false;
            continue;
        }

        while (std::getline(iss, token, ',')) {
            // skip the first line
            tokens.push_back(token);
        }

        if (tokens.size() == 8) {
            try {
                option.market_price = std::stod(tokens[0]);
                option.strike_price = std::stod(tokens[1]);
                option.underlying_price = std::stod(tokens[2]);
                option.years_to_expiration = std::stod(tokens[3]);
                option.rfr = std::stod(tokens[7]) / 100;
                char contract_type = tokens[4][0];
                option.contract_id = tokens[5];
                option.timestamp = tokens[6];
                option.option_type = (contract_type == 'C' || contract_type == 'c') ? 0 : 1;
                options.push_back(option);
            }
            catch (const std::exception &e) {
                std::cerr << "Error parsing line: " << line << " - " << e.what() << std::endl;
            }
        }
    }

    file.close();
    return options;
}

// Worker function to process batches using multiple streams
void worker_function(int gpu_id, int batch_size, int num_streams, 
                     ThreadSafeQueue<std::vector<OptionData>>& input_queue,
                     ThreadSafeQueue<std::vector<OptionData>>& output_queue) {
    
    cudaSetDevice(gpu_id);
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }
    
    // Create option pricers for each stream
    std::vector<OptimizedCRROptionPricer*> pricers(num_streams);
    for (int i = 0; i < num_streams; i++) {
        pricers[i] = new OptimizedCRROptionPricer(batch_size, 1000, 1e-5, 1000, streams[i], true);
    }
    
    int stream_idx = 0;
    std::vector<OptionData> batch;
    
    while (input_queue.pop(batch)) {
        // Process the batch using the current stream
        OptimizedCRROptionPricer* pricer = pricers[stream_idx];
        
        std::vector<double> S(batch_size);
        std::vector<double> K(batch_size);
        std::vector<double> r(batch_size);
        std::vector<double> q(batch_size, 0.0); // Assuming no dividends
        std::vector<double> T(batch_size);
        std::vector<int> type(batch_size);
        std::vector<double> marketPrices(batch_size);
        
        // Prepare data for calculation
        for (int i = 0; i < batch.size(); i++) {
            S[i] = batch[i].underlying_price;
            K[i] = batch[i].strike_price;
            r[i] = batch[i].rfr;
            T[i] = batch[i].years_to_expiration;
            type[i] = batch[i].option_type;
            marketPrices[i] = batch[i].market_price;
        }
        
        // Set parameters for spot and vol bumps based on asset price
        std::vector<double> dSpot(batch_size);
        for (int i = 0; i < batch_size; i++) {
            dSpot[i] = 0.01 * S[i]; // 1% bump
        }
        
        // Set data in the pricer
        pricer->setData(S, K, r, q, T, type, marketPrices);
        
        // Compute implied volatilities and Greeks in one go
        std::vector<double> impliedVols;
        std::vector<Greeks> greeks;
        pricer->computeAllInOne(impliedVols, greeks);
        
        // Copy results back to the batch
        for (int i = 0; i < batch.size(); i++) {
            OptionData &option = batch[i];
            option.impliedVolatility = impliedVols[i];
            
            // Only assign Greeks if IV calculation succeeded
            if (impliedVols[i] > 0) {
                option.delta = greeks[i].delta;
                option.gamma = greeks[i].gamma;
                option.theta = greeks[i].theta;
                option.vega = greeks[i].vega;
            }
        }
        
        // Push results to output queue
        output_queue.push(batch);
        
        // Move to the next stream
        stream_idx = (stream_idx + 1) % num_streams;
    }
    
    // Clean up
    for (int i = 0; i < num_streams; i++) {
        delete pricers[i];
        cudaStreamDestroy(streams[i]);
    }
}

int main(int argc, char* argv[]) {
    std::string input_filename = (argc > 1) ? argv[1] : "/home/qhawkins/Desktop/MonteCarloCuda/timed_opra_clean_mc_small.csv";
    std::string output_filename = (argc > 2) ? argv[2] : "/home/qhawkins/Desktop/GMEStudy/implied_volatilities_mc.csv";

    // Step 1: Get GPU information
    std::vector<GPUInfo> gpuInfos = getGPUInfo();
    if (gpuInfos.empty()) {
        std::cerr << "No CUDA-capable GPU found. Exiting." << std::endl;
        return 1;
    }
    
    // Print GPU information
    for (const auto& info : gpuInfos) {
        printGPUInfo(info);
    }
    
    // Step 2: Read options from CSV
    std::vector<OptionData> options = read_csv(input_filename);
    if (options.empty()) {
        std::cerr << "No options data found. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Read " << options.size() << " options from file." << std::endl;
    
    // Step 3: Determine optimal batch size and number of streams
    size_t memoryEstimate = getOptionMemoryEstimate(1000); // 1000 steps per option
    auto [batch_size, num_streams] = determineOptimalBatchSize(0, memoryEstimate);
    
    std::cout << "Optimal batch size: " << batch_size << std::endl;
    std::cout << "Number of streams: " << num_streams << std::endl;
    
    // Step 4: Create input and output queues
    ThreadSafeQueue<std::vector<OptionData>> input_queue;
    ThreadSafeQueue<std::vector<OptionData>> output_queue;
    
    // Step 5: Prepare batches and push to input queue
    for (size_t i = 0; i < options.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, options.size());
        std::vector<OptionData> batch(options.begin() + i, options.begin() + end);
        
        // If the last batch is smaller than batch_size, pad it
        if (batch.size() < batch_size) {
            // Create a copy of the last option for padding
            OptionData padding = batch.back();
            while (batch.size() < batch_size) {
                batch.push_back(padding);
            }
        }
        
        input_queue.push(batch);
    }
    
    // Signal that all batches have been queued
    input_queue.finish();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Step 6: Launch worker thread
    std::thread worker(worker_function, 0, batch_size, num_streams, 
                      std::ref(input_queue), std::ref(output_queue));
    
    // Step 7: Collect results
    std::vector<OptionData> results;
    std::vector<OptionData> batch_result;
    size_t num_batches = (options.size() + batch_size - 1) / batch_size;
    size_t batches_processed = 0;
    
    while (batches_processed < num_batches) {
        if (output_queue.pop(batch_result)) {
            // Only add the original number of options (to handle the padding in the last batch)
            size_t start_idx = batches_processed * batch_size;
            size_t valid_options = std::min(batch_size, int(options.size() - start_idx));
            
            results.insert(results.end(), batch_result.begin(), batch_result.begin() + valid_options);
            batches_processed++;
            
            // Print progress
            std::cout << "\rProcessed " << batches_processed << "/" << num_batches 
                      << " batches (" << (100.0 * batches_processed / num_batches) << "%)";
            std::cout.flush();
        }
    }
    
    std::cout << std::endl;
    
    // Step 8: Wait for worker to finish
    worker.join();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Step 9: Output results
    std::cout << "Processed " << results.size() << " options in " << elapsed.count() << " seconds." << std::endl;
    std::cout << "Average time per option: " << (elapsed.count() * 1000 / results.size()) << " ms" << std::endl;
    
    // Print the first few results
    size_t print_limit = std::min<size_t>(5, results.size());
    for (size_t i = 0; i < print_limit; ++i) {
        const auto &result = results[i];
        std::cout << "Contract ID: " << result.contract_id << std::endl;
        std::cout << "Market Price: " << result.market_price << std::endl;
        std::cout << "Implied Volatility: " << result.impliedVolatility << std::endl;
        std::cout << "Delta: " << result.delta << std::endl;
        std::cout << "Gamma: " << result.gamma << std::endl;
        std::cout << "Theta: " << result.theta << std::endl;
        std::cout << "Vega: " << result.vega << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }
    
    return 0;
}