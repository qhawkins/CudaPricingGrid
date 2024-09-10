#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

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

        CHECK_CUDA_ERROR(cudaFreeAsync(d_prices, stream));
        CHECK_CUDA_ERROR(cudaFreeAsync(d_values, stream));

        return result;
    }

    CRROptionPricer(double S, double K, double r, double q, double T, int steps, int type, double tol, int max_iter, cudaStream_t stream)
        : S(S), K(K), r(r), q(q), T(T), steps(steps), optionType(type), tol(tol), max_iter(max_iter), stream(stream) {}

    std::unordered_map<std::string, double> calculateAllGreeks(double sigma, double h = 1.0) {
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

        greeks["vera"] = (getCachedPrice(0, 0, 0, 0, 0, sigma + 0.01) - 2*basePrice + getCachedPrice(0, 0, 0, 0, 0, sigma - 0.01));

        
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
        std::cout << "Initial fa: " << fa << std::endl;
        std::cout << "Initial fb: " << fb << std::endl;
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

int main() {
    double S = 100;  // Current stock price
    double K = 100;  // Strike price
    double r = 0.05;  // Risk-free rate
    double q = 0;     // Dividend yield
    double T = 1.0;   // Time to maturity in years
    int steps = 1000;  // Number of steps in the binomial tree

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    std::vector<double> test_volatilities = {0.1, 0.2, 0.3, 0.4, 0.5};

    CRROptionPricer callPricer(S, K, r, q, T, steps, 0, .00001, 1000, stream);
    CRROptionPricer putPricer(S, K, r, q, T, steps, 1, .00001, 1000, stream);
    std::cout << "Call Greeks:" << std::endl;
    auto greeks = callPricer.calculateAllGreeks(0.2);
    callPricer.printGreeks(greeks);

    std::cout << "\nPut Greeks:" << std::endl;
    greeks = putPricer.calculateAllGreeks(0.2);
    putPricer.printGreeks(greeks);
    
    // Test implied volatility calculation
    double marketPrice = 10.0;  // Example market price
    std::cout << "\nImplied Volatility Calculation:" << std::endl;
    std::cout << "Call IV: " << callPricer.computeIV(marketPrice) << std::endl;
    std::cout << "Put IV: " << putPricer.computeIV(marketPrice) << std::endl;

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return 0;
}