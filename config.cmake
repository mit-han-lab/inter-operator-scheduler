# - When USE_CUDA=ON, use auto search
# - When USE_CUDA=/path/to/cuda-path, use the cuda path
set(USE_CUDA ON)

# - When USE_TRT=OFF, do not build TensorRT runtime
# - When USE_TRT=/path/to/tensorrt-path, use the tensorrt path to find tensorrt library
#   e.g., set(USE_TRT /home/ubuntu/libs/TensorRT-7.0.0.11)
set(USE_TRT OFF)

# - When DATA_TYPE=FLOAT32, the single precision float point data type is used
# - When DATA_TYPE=FLOAT16, the half precision float point data type is used
# - When DATA_TYPE=INT8, the 8-bit integer data type is used
set(USE_DATA_TYPE FLOAT32)

# - When USE_TENSOR_CORE=OFF, IOS runtime will not utilize Tensor Core in the computation
# - When USE_TENSOR_CORE=ON, IOS runtime will utilize Tensor Core in the computation.
set(USE_TENSOR_CORE OFF)
