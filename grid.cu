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

    float center_index = params.N / 2.0f;
    for (int i = 0; i <= params.N; ++i) {
        S[i] = params.S0 * exp((i - center_index) * params.dS);
    }

}

void initializeOptionValues(float* V, const float* S, const GridParams& params) {
    for (int j = 0; j <= params.N; ++j) {
        float payoff = (params.contractType == 0) ? 
            std::max(S[j] - params.K, 0.0f) : 
            std::max(params.K - S[j], 0.0f);
        for (int t = 0; t <= params.M; ++t) {
            V[j * (params.M + 1) + t] = payoff;
        }
        if (j % 100 == 0 || j == params.N) {
            std::cout << "Initial V[" << j << "] = " << payoff
                      << ", S[" << j << "] = " << S[j] << std::endl;
        }
    }
}

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


void allocateDeviceMemory(const GridParams& params, float*& d_V, float*& d_S) {
    size_t size_V = (params.M + 1) * (params.N + 1) * sizeof(float);
    size_t size_S = (params.N + 1) * sizeof(float);

    // Allocate device memory for option values
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_V, size_V));

    // Allocate device memory for stock prices
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_S, size_S));

    // Allocate additional arrays for tridiagonal solver if needed
    // For example:
    // float *d_a, *d_b, *d_c;
    // CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, params.N * sizeof(float)));
    // CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, params.N * sizeof(float)));
    // CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, params.N * sizeof(float)));
}

void copyHostToDevice(float* V, float* S, float* d_V, float* d_S, const GridParams& params) {
    size_t size_V = (params.M + 1) * (params.N + 1) * sizeof(float);
    size_t size_S = (params.N + 1) * sizeof(float);

    // Copy option values from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, V, size_V, cudaMemcpyHostToDevice));

    // Copy stock prices from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_S, S, size_S, cudaMemcpyHostToDevice));

    // Synchronize to ensure the copy is complete
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

__global__ void setupTridiagonalMatrix(float* a, float* b, float* c, const float* S, const GridParams params) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j > 0 && j < params.N) {
        float sigma_sqr = params.sigma * params.sigma;
        float dt = params.T / params.M;
        float dS = (S[j+1] - S[j-1]) / 2.0f;  // Central difference
        
        float alpha = 0.5f * sigma_sqr * S[j] * S[j] * dt / (dS * dS);
        float beta = params.r * S[j] * dt / (2.0f * dS);
        
        a[j] = -alpha + beta;
        b[j] = 1.0f + 2.0f * alpha + params.r * dt;
        c[j] = -alpha - beta;
    }
}

void launchSetupTridiagonalMatrix(float* d_a, float* d_b, float* d_c, float* d_S, const GridParams& params) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (params.N + threadsPerBlock - 1) / threadsPerBlock;
    //std::cout << "params sigma: " << params.sigma << std::endl;
    setupTridiagonalMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, d_S, params);
    
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "setupTridiagonalMatrix launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setupTridiagonalMatrix!\n", cudaStatus);
    }
}

__global__ void thomasSolverForward(float* a, float* b, float* c, float* d, float* c_prime, float* d_prime, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // Start from j=1

    if (j < N) {
        float w;
        if (j == 1) {
            c_prime[j] = c[j] / b[j];
            d_prime[j] = d[j] / b[j];
        } else {
            w = 1.0f / (b[j] - a[j] * c_prime[j-1]);
            c_prime[j] = c[j] * w;
            d_prime[j] = (d[j] - a[j] * d_prime[j-1]) * w;
        }
    }
}

__global__ void thomasSolverBackward(float* x, float* c_prime, float* d_prime, int N) {
    int j = N - 2 - (blockIdx.x * blockDim.x + threadIdx.x); // Start from N-2, going backward

    if (j >= 0) {
        x[j] = d_prime[j] - c_prime[j] * x[j+1];
    }
}

void launchThomasSolver(float* d_a, float* d_b, float* d_c, float* d_d, float* d_x, const GridParams& params) {
    int N = params.N + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate auxiliary arrays
    float *d_c_prime, *d_d_prime;
    cudaMalloc((void**)&d_c_prime, N * sizeof(float));
    cudaMalloc((void**)&d_d_prime, N * sizeof(float));

    // Forward sweep
    thomasSolverForward<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, d_d, d_c_prime, d_d_prime, N);

    // Synchronize
    cudaDeviceSynchronize();

    // Set the last element of x
    cudaMemcpy(&d_x[N-1], &d_d_prime[N-1], sizeof(float), cudaMemcpyDeviceToDevice);

    // Backward sweep
    thomasSolverBackward<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_c_prime, d_d_prime, N);

    // Synchronize
    cudaDeviceSynchronize();

    // Free auxiliary arrays
    cudaFree(d_c_prime);
    cudaFree(d_d_prime);
}

__device__ float Max(float a, float b) {
    return (a > b) ? a : b;
}

__global__ void applyEarlyExercise(float* V, const float* S, const GridParams params, int t) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j <= params.N) {
        int index = j * (params.M + 1) + t;
        float payoff;
        if (params.contractType == 0) { // Call option
            payoff = max(S[j] - params.K, 0.0f);
        } else { // Put option
            payoff = max(params.K - S[j], 0.0f);
        }
        V[index] = max(V[index], payoff);
    }
}

void launchApplyEarlyExercise(float* d_V, float* d_S, const GridParams& params, int timeStep) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (params.N + 1 + threadsPerBlock - 1) / threadsPerBlock;
    
    applyEarlyExercise<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_S, params, timeStep);
    
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "applyEarlyExercise launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching applyEarlyExercise!\n", cudaStatus);
    }
}

float interpolatePrice(float S0, float* S, float* V, int N, int M) {
    // Find the two nearest grid points
    int j = 0;
    while (j < N && S[j] < S0) ++j;
    
    if (j == 0 || j == N) {
        std::cout << "Warning: S0 is outside the grid. S[0] = " << S[0] << ", S[N] = " << S[N] << std::endl;
        return (j == 0) ? V[0 * (M + 1) + 0] : V[N * (M + 1) + 0];
    }

    // Linear interpolation at t=0
    float alpha = (S0 - S[j-1]) / (S[j] - S[j-1]);
    float v1 = V[(j-1) * (M + 1) + 0];
    float v2 = V[j * (M + 1) + 0];
    float interpolated_price = v1 * (1 - alpha) + v2 * alpha;

    std::cout << "Interpolation: S[" << j-1 << "] = " << S[j-1] << ", S[" << j << "] = " << S[j] << std::endl;
    std::cout << "V[" << j-1 << "] = " << v1 << ", V[" << j << "] = " << v2 << std::endl;
    std::cout << "Alpha: " << alpha << ", Interpolated price: " << interpolated_price << std::endl;

    return interpolated_price;
}


