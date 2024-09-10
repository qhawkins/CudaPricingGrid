#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

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

class CRROptionPricer {
private:
    double S, K, r, q, T, tol;
    int steps, max_iter, optionType;
    cudaStream_t stream;

public:
    CRROptionPricer(double S, double K, double r, double q, double T, int steps, int type, double tol, int max_iter, cudaStream_t stream)
        : S(S), K(K), r(r), q(q), T(T), steps(steps), optionType(type), tol(tol), max_iter(max_iter), stream(stream) {}

    double price(double sigma) {
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

        std::vector<double> h_prices(steps + 1);
        std::vector<double> h_values(steps + 1);

        for (int i = steps - 1; i >= 0; i--) {
            backwardInduction<<<grid_size, block_size, 0, stream>>>(d_values, d_prices, i, S, K, p, r, dt, u, d, optionType);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));    
        }   

        double result[steps+1];
        CHECK_CUDA_ERROR(cudaMemcpyAsync(&result, d_values, (steps+1) * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        CHECK_CUDA_ERROR(cudaFreeAsync(d_prices, stream));
        CHECK_CUDA_ERROR(cudaFreeAsync(d_values, stream));

        return result[0];
    }

    double computeIV(double optionPrice) {
        auto f = [this, optionPrice](double sigma) {
            return price(sigma) - optionPrice;
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

        double c = b, fc = fb;
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

    std::cout << "Call Option Prices:" << std::endl;
    CRROptionPricer callPricer(S, K, r, q, T, steps, 0, .00001, 1000, stream);
    for (double vol : test_volatilities) {
        double price = callPricer.price(vol);
        std::cout << "Volatility: " << vol << ", Price: " << price << std::endl;
    }

    //std::cout << "\nPut Option Prices:" << std::endl;
    CRROptionPricer putPricer(S, K, r, q, T, steps, 1, .00001, 1000, stream);
    for (double vol : test_volatilities) {
        double price = putPricer.price(vol);
        std::cout << "Volatility: " << vol << ", Price: " << price << std::endl;
    }

    // Test implied volatility calculation
    double marketPrice = 10.0;  // Example market price
    std::cout << "\nImplied Volatility Calculation:" << std::endl;
    std::cout << "Call IV: " << callPricer.computeIV(marketPrice) << std::endl;
    std::cout << "Put IV: " << putPricer.computeIV(marketPrice) << std::endl;

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return 0;
}