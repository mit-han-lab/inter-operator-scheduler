import os
from .common import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
graph_dir = os.path.join(cur_dir, "randwire_graphs/generated")


def read_graph(fname):
    with open(fname, 'r') as f:
        n = int(f.readline())
        m = int(f.readline())
        edges = []
        for i in range(m):
            u, v = f.readline().split()
            u, v = int(u), int(v)
            edges.append((u, v))
        return n, edges


def randwire_small_front(v: Value, ch: int) -> Tuple[Value, Block]:
    block = Block(v.node, None, [], None)
    v = conv2d(block, [[v]], out_channels=ch // 2, kernel=(3, 3), stride=(2, 2), padding=(1, 1), act="relu")
    v = conv2d(block, [[v]], out_channels=ch, kernel=(3, 3), stride=(2, 2), padding=(1, 1), act="identity", is_exit=True)
    return v, block


def randwire_regular_front(v: Value, ch: int) -> Tuple[Value, Block]:
    block = Block(v.node, None, [], None)
    v = conv2d(block, [[v]], out_channels=ch // 2, kernel=(3, 3), stride=(2, 2), padding=(1, 1), act="identity", is_exit=True)
    return v, block


def randwire_tail(v: Value) -> Tuple[Value, Block]:
    block = Block(v.node, None, [], None)
    v = relu(block, [[v]])
    v = conv2d(block, [[v]], out_channels=1280, kernel=(1, 1), stride=(1, 1), padding=(0, 0), act="identity")
    v = pool2d(block, [[v]], pool_type='global_avg', is_exit=True)
    return v, block


def randwire_stage(r: Value, ch: int, graph_fname) -> Tuple[Value, Block]:
    block = Block(r.node, None, [], None)
    n, edges = read_graph(graph_fname)
    in_degree = [0 for _ in range(n)]
    out_degree = [0 for _ in range(n)]
    outs = [[] for _ in range(n)]
    ins = [[] for _ in range(n)]
    for u, v in edges:
        in_degree[v] += 1
        out_degree[u] += 1
        outs[u].append(v)
        ins[v].append(u)
    outputs = [None for _ in range(n)]
    qu = [u for u in range(n) if in_degree[u] == 0]
    while len(qu) > 0:
        u = qu.pop()
        if len(ins[u]) == 0:
            outputs[u] = rwconv2d(block, [[r]], out_channels=ch, stride=(2, 2))
        else:
            inputs = [[]]
            for v in ins[u]:
                inputs[0].append(outputs[v])
            outputs[u] = rwconv2d(block, inputs, out_channels=ch, stride=(1, 1))
        for v in outs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                qu.append(v)
    inputs = [[outputs[u] for u in range(n) if out_degree[u] == 0]]
    return identity(block, inputs, is_exit=True), block


def randwire_large():
    reset_name()
    ch = 154
    graph_fnames = [os.path.join(graph_dir, fname) for fname in ['ws-16-4-0.75-2.txt', 'ws-32-4-0.75-3.txt', 'ws-32-4-0.75-4.txt', 'ws-32-4-0.75-5.txt']]

    pv = placeholder(output_shape=(3, 224, 224))
    v, block1 = randwire_regular_front(pv, ch)
    v, block2 = randwire_stage(v, ch, graph_fnames[0])
    v, block3 = randwire_stage(v, 2 * ch, graph_fnames[1])
    v, block4 = randwire_stage(v, 4 * ch, graph_fnames[2])
    v, block5 = randwire_stage(v, 8 * ch, graph_fnames[3])
    v, block6 = randwire_tail(v)

    graph = Graph("randwire_large", pv.node, [block1, block2, block3, block4, block5, block6])
    graph.init_weights()
    return graph

def randwire_xlarge():
    reset_name()
    ch = 154
    graph_fnames = [os.path.join(graph_dir, fname) for fname in ['ws-320-20-0.75-x.txt']]

    pv = placeholder(output_shape=(3, 224, 224))
    v, block1 = randwire_regular_front(pv, ch)
    v, block2 = randwire_stage(v, ch, graph_fnames[0])
    v, block3 = randwire_tail(v)

    graph = Graph("randwire_xlarge", pv.node, [block1, block2, block3])
    graph.init_weights()
    return graph

