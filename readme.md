# CUDA-Accelerated Option Pricing Engine

A high-performance, GPU-accelerated financial derivatives pricing library that implements the Cox-Ross-Rubinstein (CRR) binomial tree model for American and European options. The engine calculates option prices, implied volatilities, and Greeks (delta, gamma, theta, vega) with exceptional speed and accuracy.

## Features

- **High-Performance Pricing**: Utilizes CUDA for massively parallel option pricing on NVIDIA GPUs
- **Batch Processing**: Optimized processing of thousands of options simultaneously 
- **Implied Volatility Solver**: Fast convergence using hybrid Newton-Raphson and bisection methods
- **Greeks Calculation**: Accurate calculation of delta, gamma, theta, and vega sensitivities
- **Memory Optimization**: Smart memory management with pinned memory and asynchronous transfers
- **Multi-Stream Processing**: Overlapping computation and memory transfers for improved throughput
- **Automatic Resource Optimization**: Dynamically determines optimal batch sizes and stream counts based on available GPU resources

## Architecture

The system consists of several components working together:

### 1. Optimized CRR Option Pricer (`option_pricer.cu`)

Core pricing engine that implements the Cox-Ross-Rubinstein binomial tree model on the GPU. Key features:

- Efficient host-device memory management
- Batched processing of option contracts
- Asynchronous memory operations
- Support for pinned host memory for faster transfers

### 2. CUDA Kernels (`kernels.cu`)

Specialized GPU kernels for financial calculations:

- `deviceOptionPrice`: Optimized binomial tree implementation
- `computeImpliedVolatilityKernel`: Parallel IV solver with hybrid method
- `fusedComputationKernel`: Single-pass calculation of both IV and Greeks
- `sharedMemoryOptionPricingKernel`: Performance-optimized pricing with shared memory

### 3. GPU Optimizer (`gpu_optimizer.cu`)

Resource management utilities that ensure optimal GPU utilization:

- GPU capability detection and information reporting
- Memory requirement estimation for option pricing
- Automatic determination of optimal batch sizes
- Stream configuration for maximum throughput

### 4. Main Application (`main.cu`)

Orchestration layer that ties everything together:

- CSV data loading for option contract details
- Thread-safe queue implementation for producer-consumer pattern
- Multi-threaded GPU task distribution
- Batched result collection and reporting

## Performance

The engine is designed for high-throughput option pricing and can process thousands of options per second. Performance scales with GPU capability, with particular optimization for NVIDIA's latest architectures.

Key performance features:

- Lock-free thread coordination
- Minimized host-device synchronization
- Coalesced memory access patterns
- Optimized CUDA kernel configurations
- Multi-stream overlapping execution

## Implementation Details

### Binomial Tree Model

The Cox-Ross-Rubinstein binomial model discretizes time into multiple steps and models the underlying asset price as moving up or down at each step with specified probabilities. The option price is computed by backward induction from the final nodes.

### Implied Volatility Calculation

The system implements a hybrid approach:
1. First attempts fast convergence with Newton-Raphson method
2. Falls back to robust bisection method if needed
3. Highly parallelized to process thousands of contracts simultaneously

### Greeks Calculation

Greeks are calculated using finite difference approximations:
- **Delta**: First derivative with respect to underlying price
- **Gamma**: Second derivative with respect to underlying price
- **Theta**: Rate of option price change with respect to time
- **Vega**: Sensitivity to volatility changes

### Memory Management

The engine implements sophisticated memory handling:
- Option data is processed in optimal batch sizes
- Pinned memory for faster host-device transfers
- Asynchronous operations via CUDA streams
- Minimized memory footprint through shared memory usage

## Optimizations

- **Thread Coalescing**: Access patterns designed for minimal memory divergence
- **Kernel Fusion**: Combined operations to reduce kernel launch overhead
- **Shared Memory**: Utilization of fast on-chip memory for frequently accessed data
- **Stream Parallelism**: Overlapping computation and memory transfers
- **Dynamic Parameters**: Automatic tuning based on GPU capabilities
- **Memory Transfer Minimization**: Batch processing to amortize transfer costs

## Example Output

Below is sample output from the engine showing calculated implied volatilities and Greeks for GME options:

| Contract ID | Market Price | Implied Volatility | Delta | Gamma | Theta | Vega |
|-------------|--------------|-------------------|-------|-------|-------|------|
| GME230331P00020000 | 0.08 | 1.21894 | -0.075475 | 0.0658214 | -0.0729677 | 0.268407 |
| GME230331P00022500 | 0.57 | 1.05485 | -0.389156 | 0.192138 | -0.270041 | 0.721651 |
| GME230421C00025000 | 1.01 | 0.7577 | 0.369665 | 0.102424 | -0.0527892 | 2.18999 |
| GME230331C00023000 | 0.76 | 1.03724 | 0.510249 | 0.203841 | -0.283831 | 0.748003 |

The output demonstrates the engine's ability to accurately compute financial metrics across different option types (puts and calls) with varying strike prices and expiration dates. Note the high implied volatility values for GME options, reflecting the market's anticipation of future price movements.
