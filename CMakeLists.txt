cmake_minimum_required(VERSION 3.10)

# please configure the following paths
set(CUDAPATH /usr/local/cuda-10.2)
set(CUDNNPATH /usr/local/cuda-10.2)
set(TRTPATH /home/ubuntu/libs/TensorRT-7.0.0.11)

project(ios LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_COMPILER ${CUDAPATH}/bin/nvcc)
set(CUPTIPATH ${CUDAPATH}/extras/CUPTI)

find_library(TRT_LIBRARY NAMES nvinfer PATHS ${TRTPATH}/lib NO_DEFAULT_PATH)
find_library(CUDNN_LIBRARY NAMES cudnn PATHS ${CUDNNPATH}/lib64 NO_DEFAULT_PATH)
find_library(CUDART_LIBRARY NAMES cudart PATHS ${CUDAPATH}/lib64 NO_DEFAULT_PATH)
find_library(CUPTI_LIBRARY NAMES cupti PATHS ${CUPTIPATH}/lib64 NO_DEFAULT_PATH)


include_directories(${CUDAPATH}/include ${CUPTIPATH}/include ${MKLPATH}/include ${TRTPATH}/include)
link_directories(${CUDAPATH}/lib64 ${MKLPATH}/lib ${CUPTIPATH}/lib64)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" SUPPORT_CXX11)

set(CMAKE_CUDA_FLAGS "-O2 -arch compute_70 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")
set(COMMON_SRCS src/dist/jsoncpp.cpp src/utils.cc )
add_definitions(-DBUILD_SHARED_LIBS=ON)

add_library(trt_backend SHARED src/trt_backend.cc ${COMMON_SRCS})
target_link_libraries(trt_backend ${TRT_LIBRARY} ${CUDNN_LIBRARY} ${CUDART_LIBRARY})

add_library(ios_backend SHARED src/ios_backend.cu src/ops.cu ${COMMON_SRCS} src/profile.cu src/profile.h)
set_target_properties(ios_backend PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ios_backend cuda pthread ${CUDNN_LIBRARY} ${CUDART_LIBRARY} ${CUPTI_LIBRARY})

