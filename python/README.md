### Code Base Structure of IOS

- models (define the networks in IOS computation graph format)
    - common.py (helper functions used to define networks)
    - inception_v3.py
    - nasnet.py
    - randwire.py
    - squeezenet.py
    - ...
- contrib (glue to connect python and IOS runtime implemented by C++)
    - ctypes_utils.py (helper functions to convert python objects to ctypes objects)
    - ios_backend.py (measure latency and inference using IOS runtime)
    - trt_backend.py (measure latency and inference using TensorRT)
- cost_model (define cost models)
    - cost_model.py (abstract class of cost model)
    - ios_cost_model.py (the cost model corresponding to IOS runtime)
    - random_cost_model.py (random cost model)
- ir.py (the intermediate representation of a network (IOS computation graph), the IR can be converted to torch, tvm, tensorflow, TASO, and tensorrt modules easily)
- optimizer.py (optimize and schedule a computation graph)
- visualizer.py (visualize an IOS computation graph with some schedule)
- tf_utils.py (convert an IOS computation graph to tensorflow module, support latency measurement and inference)
- torch_utils.py (convert an IOS computation graph to torch module, support latency measurement and inference)
- tvm_utils.py (convert an IOS computation graph to tvm module, support tuning, latency measurement and inference)
- utils.py (some utility functions)

