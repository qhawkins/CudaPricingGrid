#ifndef GPU_OPTIMIZER_CUH
#define GPU_OPTIMIZER_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>

// Structure to hold GPU device information
struct GPUInfo {
    int deviceId;
    std::string name;
    size_t totalMemory;
    size_t freeMemory;
    int multiProcessorCount;
    int maxThreadsPerMultiProcessor;
    int maxThreadsPerBlock;
    size_t sharedMemPerBlock;
    int warpSize;
};

// Function to get information about all available GPUs
std::vector<GPUInfo> getGPUInfo();

// Function to determine optimal batch size for a specific GPU
// Returns a pair of (optimal batch size, number of streams)
std::pair<int, int> determineOptimalBatchSize(int deviceId, size_t elementMemoryEstimate);

// Print GPU info
void printGPUInfo(const GPUInfo& info);

// Get memory estimate for one option calculation
size_t getOptionMemoryEstimate(int steps);

#endif // GPU_OPTIMIZER_CUH