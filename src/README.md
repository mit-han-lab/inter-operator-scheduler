### Code Base Structure of IOS runtime (execution engine)
- dist (the code from third-party)
    - json / json.cpp (jsoncpp to parse json string to C++ objects)
- ios_backend.cu (IOS runtime implementation)
- trt_backend.cc (convert IOS ir graph to TensorRT module, measure latency & inference using TensorRT)
- profile.h (support of CUPTI profiling to profile device utilization)
- ops.ch / ops.h (implementation of kernels that CUDNN does not support, such as concating/add of multiple tensors)
- uils.cc / utils.h (helper functions)

