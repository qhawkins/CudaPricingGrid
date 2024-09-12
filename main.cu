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

struct GreekParams {
    double* dSpot;       // Step size for spot price
    double* dStrike;     // Step size for strike price
    double* dRate;       // Step size for interest rate
    double* dYield;      // Step size for dividend yield
    double* dTime;       // Step size for time to maturity
    double* dVol;        // Step size for volatility
};

class CRROptionPricer {
private:
    double *S, *K, *r, *q, *T, tol, *marketPrices;
    int steps, max_iter, *optionType;
    cudaStream_t stream;
    double *d_prices, *d_values;
    int batch_size;
    double *sigma;


    struct PriceCache {
        double* price;
        double *S, *K, *r, *q, *T, *sigma;
        PriceCache(double* p, double* s, double* k, double* r, double* q, double* t, double* sig)
            : price(p), S(s), K(k), r(r), q(q), T(t), sigma(sig) {}
    };

    std::vector<PriceCache> priceCache;

public:
    double* getCachedPrice(double* shiftS, double* shiftK, double* shiftR, double* shiftQ, double* shiftT, double* shiftSigma, int* optionType) {
        double* currentS;
        double* currentK;
        double* currentR;
        double* currentQ;
        double* currentT;
        double* currentSigma;
        double* price;
        int* type;

        for (int i = 0; i < batch_size; i++) {
            currentS[i] = S[i] + shiftS[i];
            currentK[i] = K[i] + shiftK[i];
            currentR[i] = r[i] + shiftR[i];
            currentQ[i] = q[i] + shiftQ[i];
            currentT[i] = T[i] + shiftT[i];

            // Update sigma, currently array, may need to be changed
            currentSigma[i] = shiftSigma[i];


            price[i] = 0;
            type[i] = optionType[i];
        }

        double* to_calc_S;
        double* to_calc_K;
        double* to_calc_R;
        double* to_calc_Q;
        double* to_calc_T;
        double* to_calc_Sigma;
        double** to_calc_Price;
        int* to_calc_Type;
    
        int index = 0;

        // Check if price is already cached
        for (auto& cache : priceCache) {
            for (int i = 0; i < batch_size; i++) {
                if (std::abs(cache.S[i] - currentS[i]) > 1e-10 &&
                    std::abs(cache.K[i] - currentK[i]) > 1e-10 &&
                    std::abs(cache.r[i] - currentR[i]) > 1e-10 &&
                    std::abs(cache.q[i] - currentQ[i]) > 1e-10 &&
                    std::abs(cache.T[i] - currentT[i]) > 1e-10 &&
                    std::abs(cache.sigma[i] - currentSigma[i]) > 1e-10) {
                    
                        to_calc_S[index] = currentS[i];
                        to_calc_K[index] = currentK[i];
                        to_calc_R[index] = currentR[i];
                        to_calc_Q[index] = currentQ[i];
                        to_calc_T[index] = currentT[i];
                        to_calc_Sigma[index] = currentSigma[i];
                        to_calc_Price[index] = &price[i];
                        to_calc_Type[index] = type[i];
                        index++;
                }
            }
        }
        int blockSize = 256;
        int numBlocks = (index + blockSize - 1) / blockSize;

        double* d_to_calc_S;
        double* d_to_calc_K;
        double* d_to_calc_R;
        double* d_to_calc_Q;
        double* d_to_calc_T;
        double* d_to_calc_Sigma;
        double** d_to_calc_Price;
        int* d_to_calc_Type;

        CHECK_CUDA_ERROR(cudaMallocAsync(&d_to_calc_S, index * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_to_calc_K, index * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_to_calc_R, index * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_to_calc_Q, index * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_to_calc_T, index * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_to_calc_Sigma, index * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_to_calc_Price, index * sizeof(double*), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_to_calc_Type, index * sizeof(int), stream));
        
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_to_calc_S, to_calc_S, index * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_to_calc_K, to_calc_K, index * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_to_calc_R, to_calc_R, index * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_to_calc_Q, to_calc_Q, index * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_to_calc_T, to_calc_T, index * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_to_calc_Sigma, to_calc_Sigma, index * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_to_calc_Price, to_calc_Price, index * sizeof(double*), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_to_calc_Type, to_calc_Type, index * sizeof(int), cudaMemcpyHostToDevice, stream));

        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        calculatePrice<<<numBlocks, blockSize, 0, stream>>>(steps, batch_size, price, to_calc_S, to_calc_K, to_calc_R, to_calc_Q, to_calc_T, to_calc_Type, to_calc_Sigma);
        
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        CHECK_CUDA_ERROR(cudaFree(d_to_calc_S));
        CHECK_CUDA_ERROR(cudaFree(d_to_calc_K));
        CHECK_CUDA_ERROR(cudaFree(d_to_calc_R));
        CHECK_CUDA_ERROR(cudaFree(d_to_calc_Q));
        CHECK_CUDA_ERROR(cudaFree(d_to_calc_T));
        CHECK_CUDA_ERROR(cudaFree(d_to_calc_Sigma));
        CHECK_CUDA_ERROR(cudaFree(d_to_calc_Price));
        CHECK_CUDA_ERROR(cudaFree(d_to_calc_Type));

        priceCache.emplace_back(price, currentS, currentK, currentR, currentQ, currentT, currentSigma);
        return price;
    }

    CRROptionPricer(int batch_size, double* marketPrices, double* S, double* K, double* r, double* q, double* T, int steps, int* type, double tol, int max_iter, cudaStream_t stream)
        : S(S), K(K), r(r), q(q), T(T), steps(steps), optionType(type), tol(tol), max_iter(max_iter), stream(stream), batch_size(batch_size), marketPrices(marketPrices) {
        // Allocate device memory
        double *d_S, *d_K, *d_r, *d_q, *d_T;
        int *d_optionType;
    
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_prices, (steps + 1) * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_values, (steps + 1) * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_S, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_K, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_r, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_q, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_T, batch_size * sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaMallocAsync(&d_optionType, batch_size * sizeof(int), stream));

        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_S, S, batch_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_K, K, batch_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_r, r, batch_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_q, q, batch_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_T, T, batch_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_optionType, type, batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
        
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }
    
    ~CRROptionPricer() {
        // Free device memory
        CHECK_CUDA_ERROR(cudaFreeAsync(d_prices, stream));
        CHECK_CUDA_ERROR(cudaFreeAsync(d_values, stream));
    }

    std::unordered_map<std::string, double>* calculateAllGreeks(const GreekParams& params) {
        std::unordered_map<std::string, double>* greeks;
        
        double* neg_dSpot;
        double* neg_dVol;
        double* neg_dTime;
        double* neg_dRate;
        double* neg_dYield;
        double* plus_sigma;
        double* minus_sigma;
        double* plusd2Spot;
        double* plusd2Vol;
        double* minusd2Spot;
        double* minusd2Vol;


        for (int i = 0; i < batch_size; i++) {
            neg_dSpot[i] = -params.dSpot[i];
            neg_dVol[i] = -params.dVol[i];
            neg_dTime[i] = -params.dTime[i];
            neg_dRate[i] = -params.dRate[i];
            neg_dYield[i] = -params.dYield[i];
            plus_sigma[i] = sigma[i] + params.dVol[i];
            minus_sigma[i] = sigma[i] - params.dVol[i];
            plusd2Spot[i] = 2 * params.dSpot[i];
            plusd2Vol[i] = sigma[i] + 2 * params.dVol[i];
            minusd2Spot[i] = -2 * params.dSpot[i];
            minusd2Vol[i] = sigma[i] - 2 * params.dVol[i];
        }

        // Base price
        double* basePrice = getCachedPrice(0, 0, 0, 0, 0, sigma, optionType);
        
        // 1st-order Greeks
        double* priceUpS = getCachedPrice(params.dSpot, 0, 0, 0, 0, sigma, optionType);
        double* priceDownS = getCachedPrice(neg_dSpot, 0, 0, 0, 0, sigma, optionType);

        double* priceUpSigma = getCachedPrice(0, 0, 0, 0, 0, plus_sigma, optionType);
        double* priceDownSigma = getCachedPrice(0, 0, 0, 0, 0, minus_sigma, optionType);
        double* priceUpT = getCachedPrice(0, 0, 0, 0, params.dTime, sigma, optionType);
        double* priceUpR = getCachedPrice(0, 0, params.dRate, 0, 0, sigma, optionType);
        double* priceUpQ = getCachedPrice(0, 0, 0, params.dYield, 0, sigma, optionType);

        double* upSpotUpVol = getCachedPrice(params.dSpot, 0, 0, 0, 0, plus_sigma, optionType);
        double* upSpotDownVol = getCachedPrice(params.dSpot, 0, 0, 0, 0, minus_sigma, optionType);
        double* downSpotUpVol = getCachedPrice(neg_dSpot, 0, 0, 0, 0, plus_sigma, optionType);
        double* downSpotDownVol = getCachedPrice(neg_dSpot, 0, 0, 0, 0, minus_sigma, optionType);
        double* upSpotUpTime = getCachedPrice(params.dSpot, 0, 0, 0, params.dTime, sigma, optionType);
        double* upSpot = getCachedPrice(params.dSpot, 0, 0, 0, 0, sigma, optionType);
        double* downSpot = getCachedPrice(neg_dSpot, 0, 0, 0, 0, sigma, optionType);
        double* downSpotUpTime = getCachedPrice(neg_dSpot, 0, 0, 0, params.dTime, sigma, optionType);
        double* upTimeUpVol = getCachedPrice(0, 0, 0, 0, params.dTime, plus_sigma, optionType);
        double* upTimeDownVol = getCachedPrice(0, 0, 0, 0, params.dTime, minus_sigma, optionType);
        double* upVol = getCachedPrice(0, 0, 0, 0, 0, plus_sigma, optionType);
        double* downVol = getCachedPrice(0, 0, 0, 0, 0, minus_sigma, optionType);
        double* up2Spot = getCachedPrice(plusd2Spot, 0, 0, 0, 0, sigma, optionType);
        double* down2Spot = getCachedPrice(minusd2Spot, 0, 0, 0, 0, sigma, optionType);
        double* up2Vol = getCachedPrice(0, 0, 0, 0, 0, plusd2Vol, optionType);
        double* down2Vol = getCachedPrice(0, 0, 0, 0, 0, minusd2Vol, optionType);
        double* upRateUpVol = getCachedPrice(0, 0, params.dRate, 0, 0, plus_sigma, optionType);
        double* upRateDownVol = getCachedPrice(0, 0, params.dRate, 0, 0, minus_sigma, optionType);
        double* downRateUpVol = getCachedPrice(0, 0, neg_dRate, 0, 0, plus_sigma, optionType);
        double* downRateDownVol = getCachedPrice(0, 0, neg_dRate, 0, 0, minus_sigma, optionType);
            
        for (int i = 0; i < batch_size; i++) {
            greeks[i]["iv"] = sigma[i];
            //First order greek computations
            greeks[i]["delta"] = (priceUpS[i] - priceDownS[i]) / (2 * params.dSpot[i]);
            greeks[i]["vega"] = ((priceUpSigma[i] - priceDownSigma[i]) / (2 * params.dVol[i]))/100;
            greeks[i]["theta"] = -(priceUpT[i] - basePrice[i]) / params.dTime[i];
            greeks[i]["rho"] = ((priceUpR[i] - basePrice[i]) / params.dRate[i])/100;
            greeks[i]["veta"] = ((priceUpQ[i] - basePrice[i]) / params.dYield[i])/100;
        
            // Second-order Greeks
            greeks[i]["gamma"] = (priceUpS[i] - 2 * basePrice[i] + priceDownS[i]) 
                                / (params.dSpot[i] * params.dSpot[i]);

            greeks[i]["vanna"] = (upSpotUpVol[i] - upSpotDownVol[i] - downSpotUpVol[i] + downSpotDownVol[i]) 
                                / (4 * params.dSpot[i] * params.dVol[i]);
            
            greeks[i]["charm"] = (upSpotUpTime[i] - upSpot[i] - downSpotUpTime[i] + downSpot[i]) 
                                / (2 * params.dSpot[i] * params.dTime[i]);
            
            greeks[i]["vomma"] = (upVol[i] - 2 * basePrice[i] + downVol[i]) 
                                / (params.dVol[i] * params.dVol[i]);
            
            greeks[i]["veta"] = (upTimeUpVol[i] - upTimeDownVol[i] - upVol[i] + downVol[i]) 
                                / (2 * params.dVol[i] * params.dTime[i]);
            
            greeks[i]["vera"] = ((upRateUpVol[i] - upRateDownVol[i]) - (downRateUpVol[i] - downRateDownVol[i])) 
                    / (4 * params.dRate[i] * params.dVol[i]);


            // Third-order Greeks
            greeks[i]["speed"] = (up2Spot[i] - 3*upSpot[i] + 3*downSpot[i] - down2Spot[i]) 
                                / (2 * params.dSpot[i] * params.dSpot[i] * params.dSpot[i]);
            
            greeks[i]["zomma"] = (upSpotUpVol[i] - 2*upVol[i] + downSpotUpVol[i]
                                - upSpotDownVol[i] + 2*downVol[i] - downSpotDownVol[i]) 
                                / (2 * params.dSpot[i] * params.dSpot[i] * params.dVol[i]);
            
            greeks[i]["color"] = (upSpotUpTime[i] - 2*upTimeUpVol[i] + downSpotUpTime[i]
                                - upSpot[i] + 2*basePrice[i] - downSpot[i]) 
                                / (params.dSpot[i] * params.dSpot[i] * params.dTime[i]);
            
            greeks[i]["ultima"] = ((up2Vol[i] - 3*upVol[i] + 3*downVol[i] - down2Vol[i]) 
                                / (params.dVol[i] * params.dVol[i] * params.dVol[i])) / (100*100);
        }
        
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

    void batchedComputeIV() {
        for (int i = 0; i < batch_size; i++) {
            sigma[i] = computeIV(i, marketPrices, S, K, r, q, T, optionType);
        }
    }

    double computeIV(int index, double* optionPrice, double* d_S, double* d_K, double* d_r, double* d_q, double* d_T, int* d_optionType) {
        double* price = 0;
        double* d_price;

        CHECK_CUDA_ERROR(cudaMallocAsync(&d_price, sizeof(double), stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_price, price, sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));



        auto f = [this, optionPrice, price, d_price, d_S, d_K, d_r, d_q, d_T, d_optionType, index](double sigma) {            
            calculateSinglePrice<<<1, 1, 0, stream>>>(steps, price, d_S, d_K, d_r, d_q, d_T, d_optionType, sigma, index);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(price, d_price, sizeof(double), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
            return *price - optionPrice[index];
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
                if (options.size() == 1024) {
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

std::vector<OptionData> processBatch(std::vector<OptionData> batch, cudaStream_t& stream) {
    std::vector<OptionData> results;
    double* S = new double[batch.size()];
    double* K = new double[batch.size()];
    double* r = new double[batch.size()];
    double* q = new double[batch.size()];
    double* T = new double[batch.size()];
    int* type = new int[batch.size()];
    double* marketPrices = new double[batch.size()];
    
    double* dSpot = new double[batch.size()];
    double* dStrike = new double[batch.size()];
    double* dRate = new double[batch.size()];
    double* dYield = new double[batch.size()];
    double* dTime = new double[batch.size()];
    double* dVol = new double[batch.size()];

    for (int i = 0; i < batch.size(); i++) {
        S[i] = batch[i].underlying_price;
        K[i] = batch[i].strike_price;
        r[i] = batch[i].rfr;
        q[i] = 0.0;
        T[i] = batch[i].years_to_expiration;
        type[i] = batch[i].option_type;
        marketPrices[i] = batch[i].market_price;
        dSpot[i] = 0.01 * batch[i].underlying_price;
        dStrike[i] = 0.01 * batch[i].strike_price;
        dRate[i] = 0.0001;
        dYield[i] = 0.0001;
        dTime[i] = 1.0 / 365.0;
        dVol[i] = 0.01;
    }
        

    GreekParams params;
    params.dSpot = dSpot;
    params.dStrike = dStrike;
    params.dRate = dRate;
    params.dYield = dYield;
    params.dTime = dTime;
    params.dVol = dVol;

    std::unordered_map<std::string, double>* greeks;

    CRROptionPricer pricer(batch.size(), marketPrices, S, K, r, q, T, 1000, type, .00001, 1000, stream);
    pricer.batchedComputeIV();
    greeks = pricer.calculateAllGreeks(params);

    for (int i = 0; i < batch.size(); i++) {
        OptionData option = batch[i];
        option.impliedVolatility = greeks[i]["iv"];
        if (option.impliedVolatility < 0) {
            continue;
        }
        option.delta = greeks[i]["delta"];
        option.theta = greeks[i]["theta"];
        option.vega = greeks[i]["vega"];
        option.rho = greeks[i]["rho"];

        //second order greeks
        option.gamma = greeks[i]["gamma"];
        option.vanna = greeks[i]["vanna"];
        option.charm = greeks[i]["charm"];
        option.vomma = greeks[i]["vomma"];
        option.veta = greeks[i]["veta"];
        option.vera = greeks[i]["vera"];

        //Third order greeks
        option.speed = greeks[i]["speed"];
        option.zomma = greeks[i]["zomma"];
        option.color = greeks[i]["color"];
        option.ultima = greeks[i]["ultima"];

        results.push_back(option);
    }
    return results;
}

int main() {
    std::string input_filename = "/home/qhawkins/Desktop/GMEStudy/timed_opra_clean_mc.csv";
    std::string output_filename = "/home/qhawkins/Desktop/GMEStudy/implied_volatilities_mc.csv";
    std::vector<OptionData> options = read_csv(input_filename);

    const int NUM_STREAMS = 16; // Adjust based on your GPU capabilities
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    const int BATCH_SIZE = 64;

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