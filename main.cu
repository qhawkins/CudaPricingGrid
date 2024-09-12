#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include "structs.h"
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

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

    std::future<void> enqueue(std::function<void()> task) {
        std::packaged_task<void()> packaged_task(std::move(task));
        std::future<void> res = packaged_task.get_future();
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

// CUDA Kernels
__global__ void initializeAssetPrices(double* prices, double* values, double S, double K, double u, double d, int steps, int optionType) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= steps) {
        prices[idx] = S * pow(u, steps - idx) * pow(d, idx);
        if (optionType == 0) { // Call
            values[idx] = fmax(0.0, prices[idx] - K);
        } else { // Put
            values[idx] = fmax(0.0, K - prices[idx]);
        }
    }
}

__global__ void backwardInduction(double* values, double* prices, int step, double S, double K, double p, double r, double dt, double u, double d, int optionType) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= step) {
        double spotPrice = S * pow(u, step - idx) * pow(d, idx);
        double expectedValue = p * values[idx] + (1 - p) * values[idx + 1];
        expectedValue *= exp(-r * dt);
        double intrinsicValue;
        if (optionType == 0) { // Call
            intrinsicValue = fmax(0.0, spotPrice - K);
        } else { // Put
            intrinsicValue = fmax(0.0, K - spotPrice);
        }
        values[idx] = fmax(expectedValue, intrinsicValue);
    }
}

struct GreekParams {
    double dSpot;       // Step size for spot price
    double dStrike;     // Step size for strike price
    double dRate;       // Step size for interest rate
    double dYield;      // Step size for dividend yield
    double dTime;       // Step size for time to maturity
    double dVol;        // Step size for volatility
};

class CRROptionPricer {
private:
    double S, K, r, q, T, tol;
    int steps, max_iter, optionType;
    cudaStream_t stream;

    struct PriceCache {
        double price;
        double S, K, r, q, T, sigma;
        PriceCache(double p, double s, double k, double r, double q, double t, double sig)
            : price(p), S(s), K(k), r(r), q(q), T(t), sigma(sig) {}
    };

    std::vector<PriceCache> priceCache;

public:
    double getCachedPrice(double shiftS, double shiftK, double shiftR, double shiftQ, double shiftT, double shiftSigma) {
        double currentS = S + shiftS;
        double currentK = K + shiftK;
        double currentR = r + shiftR;
        double currentQ = q + shiftQ;
        double currentT = T + shiftT;
        double currentSigma = shiftSigma;

        for (const auto& cache : priceCache) {
            if (std::abs(cache.S - currentS) < 1e-10 &&
                std::abs(cache.K - currentK) < 1e-10 &&
                std::abs(cache.r - currentR) < 1e-10 &&
                std::abs(cache.q - currentQ) < 1e-10 &&
                std::abs(cache.T - currentT) < 1e-10 &&
                std::abs(cache.sigma - currentSigma) < 1e-10) {
                return cache.price;
            }
        }

        double price = calculatePrice(currentS, currentK, currentR, currentQ, currentT, currentSigma);
        priceCache.emplace_back(price, currentS, currentK, currentR, currentQ, currentT, currentSigma);
        return price;
    }
    double calculatePrice(double S, double K, double r, double q, double T, double sigma) {
        double dt = T / steps;
        double u = std::exp(sigma * std::sqrt(dt));
        double d = 1.0 / u;
        double p = (std::exp((r - q) * dt) - d) / (u - d);

        double* d_prices;
        double* d_values;

        CHECK_CUDA_ERROR(cudaMallocAsync(&d_prices, (steps + 1) * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_values, (steps + 1) * sizeof(double), stream));

        int block_size = 256;
        int grid_size = (steps + block_size - 1) / block_size;

        initializeAssetPrices<<<grid_size, block_size, 0, stream>>>(d_prices, d_values, S, K, u, d, steps, optionType);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        for (int i = steps - 1; i >= 0; i--) {
            backwardInduction<<<grid_size, block_size, 0, stream>>>(d_values, d_prices, i, S, K, p, r, dt, u, d, optionType);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));    
        }   

        double result;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(&result, d_values, sizeof(double), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        try {
            CHECK_CUDA_ERROR(cudaFreeAsync(d_prices, stream));
            CHECK_CUDA_ERROR(cudaFreeAsync(d_values, stream));
        }
        catch (const std::exception& e) {
            std::cerr << "Error freeing memory: " << e.what() << std::endl;
        }
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        return result;
    }

    CRROptionPricer(double S, double K, double r, double q, double T, int steps, int type, double tol, int max_iter, cudaStream_t stream)
        : S(S), K(K), r(r), q(q), T(T), steps(steps), optionType(type), tol(tol), max_iter(max_iter), stream(stream) {}


    std::unordered_map<std::string, double> calculateAllGreeks(double sigma, const GreekParams& params) {
        std::unordered_map<std::string, double> greeks;
        
        // Base price
        double basePrice = getCachedPrice(0, 0, 0, 0, 0, sigma);
        
        // 1st-order Greeks
        double priceUpS = getCachedPrice(params.dSpot, 0, 0, 0, 0, sigma);
        double priceDownS = getCachedPrice(-params.dSpot, 0, 0, 0, 0, sigma);
        greeks["delta"] = (priceUpS - priceDownS) / (2 * params.dSpot);
        
        double priceUpSigma = getCachedPrice(0, 0, 0, 0, 0, sigma + params.dVol);
        double priceDownSigma = getCachedPrice(0, 0, 0, 0, 0, sigma - params.dVol);
        greeks["vega"] = ((priceUpSigma - priceDownSigma) / (2 * params.dVol))/100;
        
        double priceUpT = getCachedPrice(0, 0, 0, 0, params.dTime, sigma);
        greeks["theta"] = -(priceUpT - basePrice) / params.dTime;
        
        double priceUpR = getCachedPrice(0, 0, params.dRate, 0, 0, sigma);
        greeks["rho"] = ((priceUpR - basePrice) / params.dRate)/100;
        
        double priceUpQ = getCachedPrice(0, 0, 0, params.dYield, 0, sigma);
        greeks["epsilon"] = (priceUpQ - basePrice) / params.dYield;
        
        greeks["lambda"] = greeks["delta"] * S / basePrice;

        // 2nd-order Greeks
        greeks["gamma"] = (priceUpS - 2 * basePrice + priceDownS) / (params.dSpot * params.dSpot);
        
        greeks["vanna"] = (getCachedPrice(params.dSpot, 0, 0, 0, 0, sigma + params.dVol) - getCachedPrice(params.dSpot, 0, 0, 0, 0, sigma - params.dVol)
                        - getCachedPrice(-params.dSpot, 0, 0, 0, 0, sigma + params.dVol) + getCachedPrice(-params.dSpot, 0, 0, 0, 0, sigma - params.dVol)) 
                        / (4 * params.dSpot * params.dVol);
        
        greeks["charm"] = (getCachedPrice(params.dSpot, 0, 0, 0, params.dTime, sigma) - getCachedPrice(params.dSpot, 0, 0, 0, 0, sigma)
                        - getCachedPrice(-params.dSpot, 0, 0, 0, params.dTime, sigma) + getCachedPrice(-params.dSpot, 0, 0, 0, 0, sigma)) 
                        / (2 * params.dSpot * params.dTime);
        
        greeks["vomma"] = (priceUpSigma - 2 * basePrice + priceDownSigma) / (params.dVol * params.dVol);
        
        greeks["veta"] = (getCachedPrice(0, 0, 0, 0, params.dTime, sigma + params.dVol) - getCachedPrice(0, 0, 0, 0, params.dTime, sigma - params.dVol)
                        - getCachedPrice(0, 0, 0, 0, 0, sigma + params.dVol) + getCachedPrice(0, 0, 0, 0, 0, sigma - params.dVol)) 
                        / (2 * params.dVol * params.dTime);

        // 3rd-order Greeks
        double priceUp2S = getCachedPrice(2*params.dSpot, 0, 0, 0, 0, sigma);
        double priceDown2S = getCachedPrice(-2*params.dSpot, 0, 0, 0, 0, sigma);
        greeks["speed"] = (priceUp2S - 3*priceUpS + 3*priceDownS - priceDown2S) / (2 * params.dSpot * params.dSpot * params.dSpot);
        
        greeks["zomma"] = (getCachedPrice(params.dSpot, 0, 0, 0, 0, sigma + params.dVol) - 2*getCachedPrice(0, 0, 0, 0, 0, sigma + params.dVol) + getCachedPrice(-params.dSpot, 0, 0, 0, 0, sigma + params.dVol)
                        - getCachedPrice(params.dSpot, 0, 0, 0, 0, sigma - params.dVol) + 2*getCachedPrice(0, 0, 0, 0, 0, sigma - params.dVol) - getCachedPrice(-params.dSpot, 0, 0, 0, 0, sigma - params.dVol)) 
                        / (2 * params.dSpot * params.dSpot * params.dVol);
        
        greeks["color"] = (getCachedPrice(params.dSpot, 0, 0, 0, params.dTime, sigma) - 2*getCachedPrice(0, 0, 0, 0, params.dTime, sigma) + getCachedPrice(-params.dSpot, 0, 0, 0, params.dTime, sigma)
                        - getCachedPrice(params.dSpot, 0, 0, 0, 0, sigma) + 2*basePrice - getCachedPrice(-params.dSpot, 0, 0, 0, 0, sigma)) 
                        / (params.dSpot * params.dSpot * params.dTime);
        
        greeks["ultima"] = ((getCachedPrice(0, 0, 0, 0, 0, sigma + 2*params.dVol) - 3*getCachedPrice(0, 0, 0, 0, 0, sigma + params.dVol) 
                        + 3*getCachedPrice(0, 0, 0, 0, 0, sigma - params.dVol) - getCachedPrice(0, 0, 0, 0, 0, sigma - 2*params.dVol)) 
                        / (params.dVol * params.dVol * params.dVol))/(100*100);

        // Vera (as requested)
        double priceUpRSigmaUp = getCachedPrice(0, 0, params.dRate, 0, 0, sigma + params.dVol);
        double priceUpRSigmaDown = getCachedPrice(0, 0, params.dRate, 0, 0, sigma - params.dVol);
        double priceDownRSigmaUp = getCachedPrice(0, 0, -params.dRate, 0, 0, sigma + params.dVol);
        double priceDownRSigmaDown = getCachedPrice(0, 0, -params.dRate, 0, 0, sigma - params.dVol);
        greeks["vera"] = ((priceUpRSigmaUp - priceUpRSigmaDown) - (priceDownRSigmaUp - priceDownRSigmaDown)) 
                        / (4 * params.dRate * params.dVol);

        return greeks;
    }

    std::unordered_map<std::string, double> calculateAllGreeksOld(double sigma, double h = 1.0) {
        std::unordered_map<std::string, double> greeks;
        
        // Base price
        double basePrice = getCachedPrice(0, 0, 0, 0, 0, sigma);
        
        // Delta (for $1 move in underlying)
        double priceUpS = getCachedPrice(h, 0, 0, 0, 0, sigma);
        double priceDownS = getCachedPrice(-h, 0, 0, 0, 0, sigma);
        greeks["delta"] = (priceUpS - priceDownS) / (2 * h);
        
        // Gamma (for $1 move in underlying)
        greeks["gamma"] = (priceUpS - 2 * basePrice + priceDownS) / (h * h);
        
        // Theta (for 1 day)
        double priceUpT = getCachedPrice(0, 0, 0, 0, 1.0/365.0, sigma);
        greeks["theta"] = -(priceUpT - basePrice) / (1.0/365.0);
        
        // Vega (for 1% change in volatility)
        double priceUpSigma = getCachedPrice(0, 0, 0, 0, 0, sigma + 0.01);
        greeks["vega"] = (priceUpSigma - basePrice);
        
        // Rho (for 1% change in interest rate)
        double priceUpR = getCachedPrice(0, 0, 0.01, 0, 0, sigma);
        greeks["rho"] = (priceUpR - basePrice);
        
        // Higher order Greeks
        greeks["vanna"] = (getCachedPrice(h, 0, 0, 0, 0, sigma + 0.01) - getCachedPrice(h, 0, 0, 0, 0, sigma - 0.01)
                        - getCachedPrice(-h, 0, 0, 0, 0, sigma + 0.01) + getCachedPrice(-h, 0, 0, 0, 0, sigma - 0.01)) / (4 * h * 0.02);
        
        greeks["charm"] = (getCachedPrice(h, 0, 0, 0, 1.0/365.0, sigma) - getCachedPrice(h, 0, 0, 0, 0, sigma)
                        - getCachedPrice(-h, 0, 0, 0, 1.0/365.0, sigma) + getCachedPrice(-h, 0, 0, 0, 0, sigma)) / (2 * h * 1.0/365.0);
        
        greeks["vomma"] = (priceUpSigma - 2 * basePrice + getCachedPrice(0, 0, 0, 0, 0, sigma - 0.01));

        greeks["veta"] = (getCachedPrice(0, 0, 0, 0, 1.0/365.0, sigma + 0.01) - 2*basePrice + getCachedPrice(0, 0, 0, 0, 1.0/365.0, sigma - 0.01));


        
        // Speed
        double priceUp2S = getCachedPrice(2*h, 0, 0, 0, 0, sigma);
        double priceDown2S = getCachedPrice(-2*h, 0, 0, 0, 0, sigma);
        greeks["speed"] = (priceUp2S - 3*priceUpS + 3*priceDownS - priceDown2S) / (2 * h * h * h);
        
        // Zomma
        greeks["zomma"] = (getCachedPrice(h, 0, 0, 0, 0, sigma + 0.01) - 2*getCachedPrice(0, 0, 0, 0, 0, sigma + 0.01) + getCachedPrice(-h, 0, 0, 0, 0, sigma + 0.01)
                        - getCachedPrice(h, 0, 0, 0, 0, sigma - 0.01) + 2*getCachedPrice(0, 0, 0, 0, 0, sigma - 0.01) - getCachedPrice(-h, 0, 0, 0, 0, sigma - 0.01)) / (2 * h * h * 0.02);
        
        // Color
        greeks["color"] = (getCachedPrice(h, 0, 0, 0, 1.0/365.0, sigma) - 2*getCachedPrice(0, 0, 0, 0, 1.0/365.0, sigma) + getCachedPrice(-h, 0, 0, 0, 1.0/365.0, sigma)
                        - getCachedPrice(h, 0, 0, 0, 0, sigma) + 2*basePrice - getCachedPrice(-h, 0, 0, 0, 0, sigma)) / (h * h * 1.0/365.0);
        
        // Ultima
        greeks["ultima"] = (getCachedPrice(0, 0, 0, 0, 0, sigma + 0.02) - 3*getCachedPrice(0, 0, 0, 0, 0, sigma + 0.01) 
                        + 3*getCachedPrice(0, 0, 0, 0, 0, sigma - 0.01) - getCachedPrice(0, 0, 0, 0, 0, sigma - 0.02));

        return greeks;
    }

    void printGreeks(const std::unordered_map<std::string, double>& greeks) {
        std::cout << "First-order Greeks:" << std::endl;
        std::cout << "Delta: " << greeks.at("delta") << std::endl;
        std::cout << "Theta: " << greeks.at("theta") << std::endl;
        std::cout << "Vega: " << greeks.at("vega") << std::endl;
        std::cout << "Rho: " << greeks.at("rho") << std::endl;

        std::cout << "\nSecond-order Greeks:" << std::endl;
        std::cout << "Gamma: " << greeks.at("gamma") << std::endl;
        std::cout << "Vanna: " << greeks.at("vanna") << std::endl;
        std::cout << "Charm: " << greeks.at("charm") << std::endl;
        std::cout << "Vomma: " << greeks.at("vomma") << std::endl;
        std::cout << "Veta: " << greeks.at("veta") << std::endl;
        std::cout << "Vera: " << greeks.at("vera") << std::endl;

        std::cout << "\nThird-order Greeks:" << std::endl;
        std::cout << "Speed: " << greeks.at("speed") << std::endl;
        std::cout << "Color: " << greeks.at("color") << std::endl;
        std::cout << "Zomma: " << greeks.at("zomma") << std::endl;
        std::cout << "Ultima: " << greeks.at("ultima") << std::endl;
    }

    double computeIV(double optionPrice) {
        auto f = [this, optionPrice](double sigma) {
            return getCachedPrice(0, 0, 0, 0, 0, sigma) - optionPrice;
        };
        // Initial guess
        double a = 0.1;
        double b = 10.0;
        double fa = f(a);
        double fb = f(b);
        //std::cout << "Initial fa: " << fa << std::endl;
        //std::cout << "Initial fb: " << fb << std::endl;
        // If not bracketed, expand the interval
        int bracket_attempts = 0;
        while (fa * fb > 0 && bracket_attempts < 50) {
            if (std::abs(fa) < std::abs(fb)) {
                a -= (b - a);
                fa = f(a);
            } else {
                b += (b - a);
                fb = f(b);
            }
            bracket_attempts++;
        }

        if (fa * fb > 0) {
            return -1;  // Root not bracketed after attempts
        }

        double c = b,
        fc = fb;
        double d, e;

        for (int iter = 0; iter < max_iter; iter++) {
            if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
                c = a; fc = fa;
                d = b - a; e = d;
            }
            if (std::abs(fc) < std::abs(fb)) {
                a = b; b = c; c = a;
                fa = fb; fb = fc; fc = fa;
            }

            double tol1 = 2 * std::numeric_limits<double>::epsilon() * std::abs(b) + 0.5 * tol;
            double xm = 0.5 * (c - b);
            
            if (std::abs(xm) <= tol1 || fb == 0) {
                return b;  // Found a solution
            }
            
            if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
                double s = fb / fa;
                double p, q;
                if (a == c) {
                    p = 2 * xm * s;
                    q = 1 - s;
                } else {
                    q = fa / fc;
                    double r = fb / fc;
                    p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1));
                    q = (q - 1) * (r - 1) * (s - 1);
                }
                if (p > 0) q = -q;
                p = std::abs(p);
                
                if (2 * p < std::min(3 * xm * q - std::abs(tol1 * q), std::abs(e * q))) {
                    e = d;
                    d = p / q;
                } else {
                    d = xm;
                    e = d;
                }
            } else {
                d = xm;
                e = d;
            }

            a = b;
            fa = fb;
            b += (std::abs(d) > tol1) ? d : (xm > 0 ? tol1 : -tol1);
            fb = f(b);
        }

        return -2;  // Max iterations reached
    }
};

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
                if (options.size() == 16385) {
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

void processBatch(std::vector<OptionData>& batch, cudaStream_t& stream) {
    for (auto& option : batch) {
        double S = option.underlying_price;
        double K = option.strike_price;
        double r = option.rfr;
        double q = 0.0;
        double T = option.years_to_expiration;
        int steps = 1000;
        //std::cout << "S: " << S << " K: " << K << " r: " << r << " T: " << T << std::endl;
        int type = option.option_type;
        //int type = (option.option_type == "call") ? 0 : 1;
        GreekParams params;
        params.dSpot = 0.01 * S;
        params.dStrike = 0.01 * K;
        params.dRate = 0.0001;
        params.dYield = 0.0001;
        params.dTime = 1.0 / 365.0;
        params.dVol = 0.01;
        
        CRROptionPricer pricer(S, K, r, q, T, steps, type, .00001, 1000, stream);
        
        double iv = pricer.computeIV(option.market_price);
        
        std::unordered_map<std::string, double> greeks = pricer.calculateAllGreeks(iv, params);
        
        option.delta = greeks["delta"];
        option.gamma = greeks["gamma"];
        option.theta = greeks["theta"];
        option.vega = greeks["vega"];
        option.rho = greeks["rho"];

        //second order greeks
        option.vanna = greeks["vanna"];
        option.charm = greeks["charm"];
        option.vomma = greeks["vomma"];
        option.veta = greeks["veta"];
        option.vera = greeks["vera"];

        //Third order greeks
        option.speed = greeks["speed"];
        option.zomma = greeks["zomma"];
        option.color = greeks["color"];
        option.ultima = greeks["ultima"];

    }

}

int main() {
    std::string input_filename = "/home/qhawkins/Desktop/GMEStudy/timed_opra_clean_mc.csv";
    std::string output_filename = "/home/qhawkins/Desktop/GMEStudy/implied_volatilities_mc.csv";
    std::vector<OptionData> options = read_csv(input_filename);

    const int NUM_STREAMS = 16; // Adjust based on your GPU capabilities
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    const int BATCH_SIZE = 8;

    ThreadPool* pool = new ThreadPool(16);

    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < options.size(); i += BATCH_SIZE) {
        size_t end = std::min(i + BATCH_SIZE, options.size());
        i = i >= end ? end-BATCH_SIZE : i; 
        std::cout << "I " << i << " End" << end << std::endl;
        std::vector<OptionData> batch(options.begin() + i, options.begin() + end);
        //std::cout << "Batch S: " << batch[0].underlying_price << " K: " << batch[0].strike_price << " r: " << batch[0].rfr << " T: " << batch[0].years_to_expiration << std::endl;
        futures.push_back(pool->enqueue([&batch, &streams, i, NUM_STREAMS]() {
            processBatch(batch, streams[i / BATCH_SIZE % NUM_STREAMS]);
        }));
    }
    std::cout << "Queue created" << std::endl;
    // Collect results in order
    for (size_t i = 0; i < futures.size(); ++i) {
        futures[i].wait();
    }

    // Clean up CUDA streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }

    // Print or save results
    for (const auto& result : options) {
        std::cout << "Contract ID: " << result.contract_id << std::endl;
        std::cout << "Timestamp: " << result.timestamp << std::endl;
        std::cout << "Market Price: " << result.market_price << std::endl;
        std::cout << "Implied Volatility: " << result.impliedVolatility << std::endl;
        std::cout << "Delta: " << result.delta << std::endl;
        std::cout << "Gamma: " << result.gamma << std::endl;
        std::cout << "Theta: " << result.theta << std::endl;
        std::cout << "Vega: " << result.vega << std::endl;
        std::cout << "Rho: " << result.rho << std::endl;
        std::cout << "Epsilon: " << result.epsilon << std::endl;
        std::cout << "Lambda: " << result.lambda << std::endl;
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

    return 0;
}