from ir import Conv, Graph
import json
import os
from .ctypes_utils import *

cur_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

if 'NO_GPU' in os.environ:
    libname_list = []
else:
    libname_list = [
        'libtrt_backend.so',
        '../build/libtrt_backend.so',
        '../../build/libtrt_backend.so',
    ]

lib = None
for libname in libname_list:
    try:
        lib = ctypes.cdll.LoadLibrary(os.path.join(cur_dir, libname))
    except OSError as e:
        pass
    else:
        # print(f"loaded {libname}")
        break
if lib is None:
    for libname in libname_list:
        try:
            lib = ctypes.cdll.LoadLibrary(os.path.join(cur_dir, libname))
        except OSError as e:
            print(e)
    raise FileNotFoundError('TensorRT Backend library not found')


def graph_latency(graph, batch_size, warmup, number, repeat):
    results_t = ctypes.c_float * int(repeat)
    results = results_t()

    lib.graph_latency(c_string(json.dumps(graph.export_config())),
                      ctypes.c_int(batch_size),
                      ctypes.c_int(warmup),
                      ctypes.c_int(number),
                      ctypes.c_int(repeat),
                      ctypes.cast(results, ctypes.POINTER(ctypes.c_float))
                      )
    return [float(v) for v in results]


def graph_inference(graph: Graph, batch_size, input: np.ndarray):
    output = np.empty(shape=(batch_size, *graph.blocks[-1].exit_node.output_shape), dtype=np.float32)

    conv_nodes: List[Conv] = list(get_nodes_by_type(graph.nodes(), Conv))

    output_data = output.ctypes.data_as(ctypes.c_void_p)

    lib.graph_inference(c_string(json.dumps(graph.export_config())),
                        ctypes.c_int(batch_size),
                        input.ctypes.data_as(ctypes.c_void_p),

                        ctypes.c_int(len(conv_nodes)),
                        c_string_list([node.name for node in conv_nodes]),
                        c_ndarray_list([node.weight for node in conv_nodes]),
                        c_ndarray_list([node.bias for node in conv_nodes]),

                        output_data
                        )
    return output
