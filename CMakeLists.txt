cmake_minimum_required(VERSION 3.18)

project(CudaPricingGrid LANGUAGES CXX CUDA) # Name of your project
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_89,code=sm_89 -rdc=true")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)

link_libraries(-lcublas -lcusolver -lcudadevrt)

add_executable(CudaPricingGrid 
               main.cu
               kernels.cu
               option_pricer.cu
               gpu_optimizer.cu)

set_source_files_properties(main.cu kernels.cu option_pricer.cu gpu_optimizer.cu PROPERTIES FORCE_LANGUAGE CUDA LINKER_LANGUAGE CUDA)

include_directories(/usr/local/cuda-12.5/include)
include_directories(${PROJECT_SOURCE_DIR}/include)
