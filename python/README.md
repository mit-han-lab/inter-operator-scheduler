### IOS Python Modules

- ios
    - models (define the networks in IOS computation graph format)
        - common (helper functions used to define networks)
        - inception_v3
        - nasnet
        - randwire
        - squeezenet
        - ...
    - contrib (glue to connect python and IOS runtime implemented by C++)
        - ctypes_utils (helper functions to convert python objects to ctypes objects)
        - runtime (measure latency and inference using IOS runtime)
        - trt_runtime (measure latency and inference using TensorRT)
    - cost_model (define cost models used in IOS optimizer)
    - ir (the intermediate representation of a network (IOS computation graph), the IR can be converted to torch, tvm, tensorflow, TASO, and tensorrt modules easily)
    - optimizer (optimize and schedule a computation graph)
    - visualizer (visualize an IOS computation graph with some schedule)
    - tf_utils (convert an IOS computation graph to tensorflow module, support latency measurement and inference)
    - torch_utils (convert an IOS computation graph to torch module, support latency measurement and inference)
    - tvm_utils (convert an IOS computation graph to tvm module, support tuning, latency measurement and inference)
    - utils (some utility functions)

