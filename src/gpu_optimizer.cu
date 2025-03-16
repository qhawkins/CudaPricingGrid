#include "gpu_optimizer.cuh"
#include <iostream>

std::vector<GPUInfo> getGPUInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    std::vector<GPUInfo> gpuInfos;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        GPUInfo info;
        info.deviceId = i;
        info.name = deviceProp.name;
        info.totalMemory = deviceProp.totalGlobalMem;
        
        // Get free memory
        size_t free, total;
        cudaSetDevice(i);
        cudaMemGetInfo(&free, &total);
        
        info.freeMemory = free;
        info.multiProcessorCount = deviceProp.multiProcessorCount;
        info.maxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;
        info.maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
        info.sharedMemPerBlock = deviceProp.sharedMemPerBlock;
        info.warpSize = deviceProp.warpSize;
        
        gpuInfos.push_back(info);
    }
    
    return gpuInfos;
}

void printGPUInfo(const GPUInfo& info) {
    std::cout << "====== GPU " << info.deviceId << " ======" << std::endl;
    std::cout << "Name: " << info.name << std::endl;
    std::cout << "Total Memory: " << info.totalMemory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Free Memory: " << info.freeMemory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << info.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Multiprocessor: " << info.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Threads per Block: " << info.maxThreadsPerBlock << std::endl;
    std::cout << "Shared Memory per Block: " << info.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Warp Size: " << info.warpSize << std::endl;
}

size_t getOptionMemoryEstimate(int steps) {
    // Base memory per option:
    // CRROptionPricer allocates:
    // - d_S, d_K, d_r, d_q, d_T: 5 * sizeof(double)
    // - d_price: sizeof(double)
    // - d_prices, d_values: 2 * (steps + 1) * sizeof(double)
    // - d_sigma, d_price_low, d_price_high, d_price_mid, d_ivResults: 5 * sizeof(double)
    // - d_optionType: sizeof(int)
    // - d_marketPrices: sizeof(double)
    // - Shifted parameters: 8 * sizeof(double) + sizeof(int)
    
    // Total estimate per option
    return (20 * sizeof(double) + 2 * sizeof(int) + 2 * (steps + 1) * sizeof(double));
}

std::pair<int, int> determineOptimalBatchSize(int deviceId, size_t elementMemoryEstimate) {
    cudaSetDevice(deviceId);
    
    // Get device properties
    GPUInfo info;
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);
        
        info.multiProcessorCount = deviceProp.multiProcessorCount;
        info.maxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;
        info.maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
        info.warpSize = deviceProp.warpSize;
        
        // Get free memory
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        info.freeMemory = free;
    }
    
    // Reserve some memory for the system and other processes (20%)
    size_t memoryToUse = info.freeMemory * 0.8;
    
    // Calculate max batch size based on memory
    int maxBatchSize = memoryToUse / elementMemoryEstimate;
    
    // Calculate max batch size based on compute capabilities
    int computeCapabilityBatchSize = info.multiProcessorCount * info.maxThreadsPerMultiProcessor / 2;
    
    // Choose the smaller of the two
    int batchSize = std::min(maxBatchSize, computeCapabilityBatchSize);
    
    // Ensure batch size is a multiple of the warp size
    batchSize = (batchSize / info.warpSize) * info.warpSize;
    
    // Ensure batch size is at least one warp
    batchSize = std::max(batchSize, info.warpSize);
    
    // Determine number of streams based on GPU utilization
    // Use more streams if we have fewer options per stream
    int numStreams = 1;
    if (batchSize < computeCapabilityBatchSize) {
        // Try to keep GPU busy with multiple streams
        numStreams = std::min(16, computeCapabilityBatchSize / batchSize);
    }
    
    return std::make_pair(batchSize, numStreams);
}