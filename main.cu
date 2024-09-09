#include <cmath>
#include <stdexcept>
#include "structs.h"
#include "grid.cuh"


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