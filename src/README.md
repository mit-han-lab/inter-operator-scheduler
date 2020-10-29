### Code Base Structure of IOS runtime (execution engine)

- src
    - ios
        - ios_runtime.cu (IOS runtime implementation)
        - ops.cu (implementation of kernels that CUDNN does not support, such as concating/add of multiple tensors)
        - profile.cu (support of CUPTI profiling to profile device utilization)
    - trt
        - trt_runtime.cc (convert IOS computation graph to TensorRT network, measure latency & inference using TensorRT)
    - utils
        - jsoncpp.cpp (jsoncpp to parse json string to C++ objects)
        - utils.cc (helper functions)
   
