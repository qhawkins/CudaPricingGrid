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


class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i)
            workers.emplace_back(
                [this]
                {
                    for(;;)
                    {
                        std::packaged_task<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock,
                                [this]{ return this->stop || !this->tasks.empty(); });
                            if(this->stop && this->tasks.empty())
                                return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }
                        task();
                    }
                }
            );
    };

    std::future<std::vector<OptionData>> enqueue(std::function<std::vector<OptionData>()> task) {
        std::packaged_task<std::vector<OptionData>()> packaged_task(std::move(task));
        std::future<std::vector<OptionData>> res = packaged_task.get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace(std::move(packaged_task));
        }
        condition.notify_one();
        return res;
    };

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    };

private:
    std::vector<std::thread> workers;
    std::queue<std::packaged_task<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// Function to process a single batch
std::vector<OptionData> processBatch(const std::vector<OptionData>& batch, 
                                     cudaStream_t& stream) 
{
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
    CRROptionPricer pricer(batch_size, marketPrices.data(), S.data(), K.data(), 
                          r.data(), q.data(), T.data(), 1000, type.data(), 
                          1e-5, 1000, stream);
    // Compute implied volatilities
    std::vector<double> impliedVols;
    pricer.computeImpliedVolatility(impliedVols);

    // Define GreekParams
    GreekParams params;
    params.dSpot = dSpot;
    params.dStrike = dStrike;
    params.dRate = dRate;
    params.dYield = dYield;
    params.dTime = dTime;
    params.dVol = dVol;

    // Calculate Greeks
    std::vector<Greeks> greeks = pricer.calculateAllGreeks(params);

    // Assign results to options
    std::vector<OptionData> results;
    results.reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
        OptionData option = batch[i];
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

        results.push_back(option);
    }

    return results;
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
    std::vector<OptionData> options = read_csv(input_filename);

    const int NUM_STREAMS = 32; // Adjust based on your GPU capabilities
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    const int BATCH_SIZE = 512;

    ThreadPool* pool = new ThreadPool(16);

    std::vector<std::future<std::vector<OptionData>>> futures;
    for (size_t i = 0; i < options.size(); i += BATCH_SIZE) {
        size_t end = std::min(i + BATCH_SIZE, options.size());
        //i = i >= end ? end-BATCH_SIZE : i; 
        std::vector<OptionData> batch(options.begin() + i, options.begin() + end);
        //std::cout << "Batch S: " << batch[0].underlying_price << " K: " << batch[0].strike_price << " r: " << batch[0].rfr << " T: " << batch[0].years_to_expiration << std::endl;
        futures.push_back(pool->enqueue([batch, &streams, i, NUM_STREAMS]()->std::vector<OptionData> {
            return processBatch(batch, streams[i / BATCH_SIZE % NUM_STREAMS]);
        }));
    }
    std::cout << "Queue created" << std::endl;
    // Collect results in order
    std::vector<std::vector<OptionData>> results;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < futures.size(); ++i) {
        futures[i].wait();
    }
    for (size_t i = 0; i < futures.size(); ++i) {
        std::cout << "Waiting for future " << i << std::endl;
        results.push_back(futures[i].get());
    }
    std::vector<OptionData> final_results;
    for (const auto& result : results) {
        for (const auto& option : result) {
            final_results.push_back(option);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Clean up CUDA streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
    // Print or save results
    for (const auto& result : final_results) {
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
        std::cout << std::endl;
    }
    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;


    return 0;
}