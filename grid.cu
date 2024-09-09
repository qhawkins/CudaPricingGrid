#include "grid.cuh"


void allocateHostMemory(const GridParams& params, float*& V, float*& S) {
    // Allocate memory for option values
    V = (float*)malloc((params.M + 1) * (params.N + 1) * sizeof(float));
    if (V == nullptr) {
        throw std::runtime_error("Failed to allocate memory for option values");
    }

    // Allocate memory for stock prices
    S = (float*)malloc((params.N + 1) * sizeof(float));
    if (S == nullptr) {
        free(V);  // Free previously allocated memory
        throw std::runtime_error("Failed to allocate memory for stock prices");
    }

    // Initialize stock prices
    for (int i = 0; i <= params.N; ++i) {
        S[i] = i * params.dS;
    }
}

void initializeOptionValues(float* V, const float* S, const GridParams& params) {
    // Set terminal condition (at expiry)
    for (int j = 0; j <= params.N; ++j) {
        V[params.M * (params.N + 1) + j] = std::max(params.K - S[j], 0.0f);
    }

    // Set boundary conditions
    for (int i = 0; i <= params.M; ++i) {
        // At S = 0
        V[i * (params.N + 1)] = params.K;

        // At S = Smax
        V[i * (params.N + 1) + params.N] = 0.0f;
    }

    // Initialize interior points with payoff (for American option)
    for (int i = 0; i < params.M; ++i) {
        for (int j = 1; j < params.N; ++j) {
            V[i * (params.N + 1) + j] = std::max(params.K - S[j], 0.0f);
        }
    }
}