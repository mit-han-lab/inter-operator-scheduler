from .common import *


def alexnet() -> Graph:
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224))
    block = Block(pv.node, None, [], None)

    v = conv2d(block, [[pv]], 64, (11, 11), stride=(4, 4), padding=(2, 2))
    v = pool2d(block, [[v]], 'max_pool', (3, 3), (2, 2), (1, 1))
    v = conv2d(block, [[v]], 192, (5, 5), stride=(1, 1), padding=(2, 2))
    v = pool2d(block, [[v]], 'max_pool', (3, 3), (2, 2), (1, 1))
    v = conv2d(block, [[v]], 384, (3, 3), stride=(1, 1), padding=(0, 0))
    v = conv2d(block, [[v]], 256, (3, 3), stride=(1, 1), padding=(0, 0))
    v = conv2d(block, [[v]], 256, (3, 3), stride=(1, 1), padding=(0, 0), is_exit=True)

    graph = Graph("alexnet", pv.node, [block])
    graph.init_weights()
    return graph

