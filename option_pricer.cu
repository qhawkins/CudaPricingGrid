#include "option_pricer.cuh"
#include "kernels.cuh"

OptimizedCRROptionPricer::OptimizedCRROptionPricer(int batch_size_, int steps_, 
                                                 double tol_, int max_iter_, 
                                                 cudaStream_t stream_, bool use_pinned_memory_)
    : batch_size(batch_size_), steps(steps_), tol(tol_), max_iter(max_iter_), 
      stream(stream_), use_pinned_memory(use_pinned_memory_) {
    allocateMemory();
}

OptimizedCRROptionPricer::~OptimizedCRROptionPricer() {
    freeMemory();
}

void OptimizedCRROptionPricer::allocateMemory() {
    size_t size = batch_size * sizeof(double);
    size_t int_size = batch_size * sizeof(int);
    
    // Allocate host memory (pinned if requested)
    if (use_pinned_memory) {
        CHECK_CUDA_ERROR(cudaMallocHost(&h_S, size));
        CHECK_CUDA_ERROR(cudaMallocHost(&h_K, size));
        CHECK_CUDA_ERROR(cudaMallocHost(&h_r, size));
        CHECK_CUDA_ERROR(cudaMallocHost(&h_q, size));
        CHECK_CUDA_ERROR(cudaMallocHost(&h_T, size));
        CHECK_CUDA_ERROR(cudaMallocHost(&h_marketPrices, size));
        CHECK_CUDA_ERROR(cudaMallocHost(&h_optionType, int_size));
    }
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_S, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_K, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_r, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_q, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_T, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_optionType, int_size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_marketPrices, size, stream));
    
    // Pricing buffers
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_price, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_sigma, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_low, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_high, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_price_mid, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_ivResults, size, stream));
    
    // Binomial tree workspace
    size_t prices_size = batch_size * (steps + 1) * sizeof(double);
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_prices, prices_size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_values, prices_size, stream));
    
    // Greeks buffers
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_greeks_delta, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_greeks_gamma, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_greeks_theta, size, stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_greeks_vega, size, stream));
}

void OptimizedCRROptionPricer::freeMemory() {
    // Free host memory
    if (use_pinned_memory) {
        cudaFreeHost(h_S);
        cudaFreeHost(h_K);
        cudaFreeHost(h_r);
        cudaFreeHost(h_q);
        cudaFreeHost(h_T);
        cudaFreeHost(h_marketPrices);
        cudaFreeHost(h_optionType);
    }
    
    // Free device memory asynchronously
    cudaFreeAsync(d_S, stream);
    cudaFreeAsync(d_K, stream);
    cudaFreeAsync(d_r, stream);
    cudaFreeAsync(d_q, stream);
    cudaFreeAsync(d_T, stream);
    cudaFreeAsync(d_optionType, stream);
    cudaFreeAsync(d_marketPrices, stream);
    
    cudaFreeAsync(d_price, stream);
    cudaFreeAsync(d_prices, stream);
    cudaFreeAsync(d_values, stream);
    cudaFreeAsync(d_sigma, stream);
    cudaFreeAsync(d_price_low, stream);
    cudaFreeAsync(d_price_high, stream);
    cudaFreeAsync(d_price_mid, stream);
    cudaFreeAsync(d_ivResults, stream);
    
    cudaFreeAsync(d_greeks_delta, stream);
    cudaFreeAsync(d_greeks_gamma, stream);
    cudaFreeAsync(d_greeks_theta, stream);
    cudaFreeAsync(d_greeks_vega, stream);
}

void OptimizedCRROptionPricer::setData(const std::vector<double>& S, const std::vector<double>& K,
                                      const std::vector<double>& r, const std::vector<double>& q,
                                      const std::vector<double>& T, const std::vector<int>& optionType,
                                      const std::vector<double>& marketPrices) {
    
    size_t size = batch_size * sizeof(double);
    size_t int_size = batch_size * sizeof(int);
    
    // Copy to host buffers if using pinned memory
    if (use_pinned_memory) {
        std::copy(S.begin(), S.end(), h_S);
        std::copy(K.begin(), K.end(), h_K);
        std::copy(r.begin(), r.end(), h_r);
        std::copy(q.begin(), q.end(), h_q);
        std::copy(T.begin(), T.end(), h_T);
        std::copy(optionType.begin(), optionType.end(), h_optionType);
        std::copy(marketPrices.begin(), marketPrices.end(), h_marketPrices);
        
        // Copy from pinned host memory to device (faster)
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_S, h_S, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_K, h_K, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_r, h_r, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_q, h_q, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_T, h_T, size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_optionType, h_optionType, int_size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_marketPrices, h_marketPrices, size, cudaMemcpyHostToDevice, stream));
    } else {
        // Copy directly from host vectors to device
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_S, S.data(), size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_K, K.data(), size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_r, r.data(), size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_q, q.data(), size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_T, T.data(), size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_optionType, optionType.data(), int_size, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_marketPrices, marketPrices.data(), size, cudaMemcpyHostToDevice, stream));
    }
}

void OptimizedCRROptionPricer::computeAllInOne(std::vector<double>& impliedVols, 
                                             std::vector<Greeks>& greeks,
                                             double dSpot, double dVol) {
    // Launch the fused kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    
    fusedComputationKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        steps, batch_size, d_marketPrices, d_S, d_K, d_r, d_q, d_T, d_optionType,
        d_ivResults, d_greeks_delta, d_greeks_gamma, d_greeks_theta, d_greeks_vega,
        dSpot, dVol, tol, max_iter);
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Retrieve results
    impliedVols.resize(batch_size);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(impliedVols.data(), d_ivResults, 
                                    batch_size * sizeof(double), 
                                    cudaMemcpyDeviceToHost, stream));
    
    // Prepare Greeks
    greeks.resize(batch_size);
    
    // Temporary vectors for Greeks results
    std::vector<double> delta(batch_size);
    std::vector<double> gamma(batch_size);
    std::vector<double> theta(batch_size);
    std::vector<double> vega(batch_size);
    
    // Copy Greeks from device to host
    CHECK_CUDA_ERROR(cudaMemcpyAsync(delta.data(), d_greeks_delta, 
                                    batch_size * sizeof(double), 
                                    cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(gamma.data(), d_greeks_gamma, 
                                    batch_size * sizeof(double), 
                                    cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(theta.data(), d_greeks_theta, 
                                    batch_size * sizeof(double), 
                                    cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(vega.data(), d_greeks_vega, 
                                    batch_size * sizeof(double), 
                                    cudaMemcpyDeviceToHost, stream));
    
    // Wait for all copies to complete
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    // Fill Greeks structure
    for (int i = 0; i < batch_size; i++) {
        greeks[i].delta = delta[i];
        greeks[i].gamma = gamma[i];
        greeks[i].theta = theta[i];
        greeks[i].vega = vega[i];
    }
}