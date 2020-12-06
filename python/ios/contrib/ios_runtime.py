from ios.ir import Conv, Graph, Block
import json
import os
from .ctypes_utils import *

cur_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

libname_list = [
    'libios_runtime.so',
    '../libios_runtime.so',
    '../../libios_runtime.so',
    '../build/libios_runtime.so',
    '../../build/libios_runtime.so',
    '../../../build/libios_runtime.so',
]
lib = None

def load_ios_library():
    for libname in libname_list:
        try:
            return ctypes.cdll.LoadLibrary(os.path.join(cur_dir, libname))
        except OSError as e:
            pass
    raise FileNotFoundError('IOS Backend library not found')

def graph_latency(graph: Graph, batch_size=1, warmup=2, number=6, repeat=6, profile_stage=False):
    global lib
    if lib is None:
        lib = load_ios_library()
    num_stages = sum(len(b.stages) for b in graph.blocks)
    results_t = ctypes.c_float * int(repeat)
    results = results_t()
    stage_results_t = ctypes.c_float * int(repeat * num_stages)
    stage_results = stage_results_t()
    for i in range(repeat * num_stages):
        stage_results[i] = 0.0

    lib.graph_latency(c_string(json.dumps(graph.export_config())),
                      ctypes.c_int(batch_size),
                      ctypes.c_int(warmup),
                      ctypes.c_int(number),
                      ctypes.c_int(repeat),
                      ctypes.c_int(1 if profile_stage else 0),
                      ctypes.cast(results, ctypes.POINTER(ctypes.c_float)),
                      ctypes.cast(stage_results, ctypes.POINTER(ctypes.c_float))
                      )
    if profile_stage:
        return [float(v) for v in results], [float(v) for v in stage_results]
    else:
        return [float(v) for v in results]


def block_latency(block: Block, batch_size, warmup, number, repeat, profile_stage=False):
    global lib
    if lib is None:
        lib = load_ios_library()
    num_stages = len(block.stages)
    results_t = ctypes.c_float * int(repeat)
    results = results_t()
    stage_results_t = ctypes.c_float * int(repeat * num_stages)
    stage_results = stage_results_t()
    for i in range(repeat * num_stages):
        stage_results[i] = 0.0

    lib.block_latency(c_string(json.dumps(block.export_config())),
                      ctypes.c_int(batch_size),
                      ctypes.c_int(warmup),
                      ctypes.c_int(number),
                      ctypes.c_int(repeat),
                      ctypes.c_int(1 if profile_stage else 0),
                      ctypes.cast(results, ctypes.POINTER(ctypes.c_float)),
                      ctypes.cast(stage_results, ctypes.POINTER(ctypes.c_float))
                      )
    if profile_stage:
        return [float(v) for v in results], [float(v) for v in stage_results]
    else:
        return [float(v) for v in results]


def stage_latency(stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False):
    global lib
    if lib is None:
        lib = load_ios_library()
    stage_seqs = []
    input_nodes = []
    for seq in stage:
        seq_nodes = []
        for node in seq:
            seq_nodes.append(node.export_config())
        stage_seqs.append(seq_nodes)
        input_nodes.extend(value.node for nd in seq for term in nd.inputs for value in term)
    stage_node_names = [nd.name for seq in stage for nd in seq]
    input_nodes: List[Node] = list(ind for ind in dict.fromkeys(input_nodes) if ind.name not in stage_node_names)
    results_t = ctypes.c_float * int(repeat)
    results = results_t()
    stage_results_t = ctypes.c_float * int(repeat)
    stage_results = stage_results_t()
    lib.stage_latency(c_string(json.dumps(stage_seqs)),
                      c_string(json.dumps({nd.name: nd.output_shape for nd in input_nodes})),
                      ctypes.c_int(batch_size),
                      ctypes.c_int(warmup),
                      ctypes.c_int(number),
                      ctypes.c_int(repeat),
                      ctypes.c_int(1 if profile_stage else 0),
                      ctypes.cast(results, ctypes.POINTER(ctypes.c_float)),
                      ctypes.cast(stage_results, ctypes.POINTER(ctypes.c_float))
                      )
    if profile_stage:
        return [float(v) for v in results], [float(v) for v in stage_results]
    else:
        return [float(v) for v in results]


def graph_inference(graph: Graph, batch_size, input: np.ndarray):
    global lib
    if lib is None:
        lib = load_ios_library()
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
