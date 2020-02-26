cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(dcn_v2_cuda_forward_v2)

#add_compile_options(-std=c++11)

#add_definitions(-D WITH_CUDA)

set(Torch_DIR /usr/local/libtorch/share/cmake/Torch) 
find_package(Torch REQUIRED)

include_directories(/usr/include/python3.5m)
include_directories(/usr/include/python2.7/)


#include_directories(/usr/local/cuda/include)
#link_directories(/usr/local/cuda/lib64)

set(CUDA_HOST_COMPILATION_CPP ON)
#set(TORCH_NVCC_FLAGS "-D__CUDA_NO_HALF_OPERATORS__")

#set(CUDA_NVCC_FLAGS -std=c++11 
#-DCUDA_HAS_FP16=1 
#-D__CUDA_NO_HALF_OPERATORS__ 
#-D__CUDA_NO_HALF_CONVERSIONS__
#-D__CUDA_NO_HALF2_OPERATORS__)

set(CUDA_NVCC_FLAGS -std=c++11
-D__CUDA_NO_HALF_OPERATORS__ )

#set(CUDA_NVCC_FLAGS -std=c++11 -Xcompiler -fexceptions -Xcompiler -fPIC 
#-gencode arch=compute_30,code=sm_30
#-gencode arch=compute_35,code=sm_35
#-gencode arch=compute_50,code=sm_50
#-gencode arch=compute_60,code=sm_60
#-gencode arch=compute_60,code=compute_60)


cuda_add_library(${PROJECT_NAME} SHARED 
vision.cpp
dcn_v2_cuda.cu
dcn_v2_im2col_cuda.cu
)

# Enable C++11
target_compile_features(${PROJECT_NAME} PRIVATE cxx_range_for)
# Link against LibTorch
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
