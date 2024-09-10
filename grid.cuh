#ifndef GRID_CUH
#define GRID_CUH

#include "structs.h"
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

void allocateHostMemory(const GridParams& params, float*& V, float*& S);
void initializeOptionValues(float* V, const float* S, const GridParams& params);
void allocateDeviceMemory(const GridParams& params, float*& d_V, float*& d_S);
void copyHostToDevice(const float* h_V, const float* h_S, float* d_V, float* d_S, const GridParams& params);
void launchSetupTridiagonalMatrix(float* d_a, float* d_b, float* d_c, const float* d_S, const GridParams& params);
void launchThomasSolver(float* d_a, float* d_b, float* d_c, float* d_y, float* d_x, const GridParams& params);
void launchApplyEarlyExercise(float* d_V, float* d_S, const GridParams& params, int t);
void launchSetupTridiagonalMatrix(float* d_a, float* d_b, float* d_c, float* d_S, const GridParams& params);
float interpolatePrice(float S0, float* S, float* V, int N, int M);
__global__ void updateBoundaryConditions(float* V, float K, int N, int t);


#endif // GRID_CUH