cmake_minimum_required(VERSION 3.10)
project(ios LANGUAGES C CXX CUDA)

include(config.cmake)
file(GLOB_RECURSE IOS_SRCS src/ios/* src/utils/*)
file(GLOB_RECURSE TRT_SRCS src/trt/* src/utils/*)

if (NOT "${USE_CUDA}" STREQUAL "OFF")
    include(cmake/FindCUDA.cmake)
    find_cuda(${USE_CUDA})
    if (CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
        message(STATUS "CUDA_INCLUDE_DIR=" ${CUDA_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
    endif(CUDA_FOUND)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" SUPPORT_CXX11)
set(CMAKE_CUDA_FLAGS "-O2 -arch compute_70 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")
add_definitions(-DBUILD_SHARED_LIBS=ON)

include_directories(include)

add_library(ios_runtime SHARED ${IOS_SRCS})
target_link_libraries(ios_runtime ${CUDA_CUDA_LIBRARY} ${CUDA_CUDNN_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUPTI_LIBRARY}
#        ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64/libcupti.so
        ) #pthread
set_target_properties(ios_runtime PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(NOT "${USE_TRT}" STREQUAL "OFF")
    find_library(TRT_LIBRARY NAMES nvinfer PATHS ${USE_TRT}/lib NO_DEFAULT_PATH)
    add_library(trt_runtime SHARED ${TRT_SRCS})
    target_include_directories(trt_runtime PUBLIC ${USE_TRT}/include)
    target_link_libraries(trt_runtime ${TRT_LIBRARY} ${CUDA_CUDNN_LIBRARY} ${CUDA_CUDART_LIBRARY})
endif()


