"""
Convert python objects to c objects
"""
from typing import List, Iterable
from ios.ir import Sequential, Node
import numpy as np
import ctypes


def c_int(i: int):  # to int
    return ctypes.c_int(i)


def c_string(s: str):  # to char *
    return ctypes.cast(ctypes.create_string_buffer(s.encode()), ctypes.c_char_p)


def c_string_list(sl: List[str]):  # to char **
    conv_names_t = ctypes.c_char_p * len(sl)
    name_list = conv_names_t(*[ctypes.cast(ctypes.create_string_buffer(s.encode()), ctypes.c_char_p) for s in sl])
    return ctypes.cast(name_list, ctypes.POINTER(ctypes.c_char_p))


def c_ndarray_list(al: List[np.ndarray]):  # to float **
    data_t = ctypes.c_void_p * len(al)
    return ctypes.cast(data_t(*[a.ctypes.data_as(ctypes.c_void_p) if isinstance(a, np.ndarray)
                                else ctypes.c_void_p(None) for a in al]), ctypes.POINTER(ctypes.c_void_p))


def c_ndarray(a: np.ndarray):  # to float *
    return a.ctypes.data_as(ctypes.c_void_p)


def get_nodes_by_type(nodes: Iterable[Node], node_type):
    for node in nodes:
        if isinstance(node, node_type):
            yield node
        elif isinstance(node, Sequential):
            for v in get_nodes_by_type(node.nodes, node_type):
                yield v
