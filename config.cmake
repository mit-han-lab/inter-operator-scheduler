# - When USE_CUDA=ON, use auto search
# - When USE_CUDA=/path/to/cuda-path, use the cuda path
set(USE_CUDA ON)

# - When USE_TRT=OFF, do not build TensorRT runtime
# - When USE_TRT=/path/to/tensorrt-path, use the tensorrt path to find tensorrt library
#   e.g., set(USE_TRT /home/ubuntu/libs/TensorRT-7.0.0.11)
set(USE_TRT OFF)
