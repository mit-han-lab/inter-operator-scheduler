from .common import *


def fire_complex(v: Value, squeeze: int, expand: int) -> Tuple[Value, Block]:
    block = Block(v.node, None, [], None)
    v1 = conv2d(block, [[v]], out_channels=squeeze, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
    v1a = conv2d(block, [[v1]], out_channels=expand, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    v1b = conv2d(block, [[v1]], out_channels=expand, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
    v1 = identity(block, [[v1a], [v1b]])
    if v1.node.output_shape[0] == v.node.output_shape[0]:
        v2 = v
    else:
        v2 = conv2d(block, [[v]], out_channels=2 * expand, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
    return identity(block, [[v1, v2]], is_exit=True), block


def block_front(v):
    block = Block(v.node, None, [], None)
    v = conv2d(block, [[v]], out_channels=96, kernel=(7, 7), stride=(2, 2), padding=(3, 3))
    v = pool2d(block, [[v]], pool_type='max', kernel=(3, 3), stride=(2, 2), padding=(0, 0), is_exit=True)
    return v, block


def block_middle(v):
    block = Block(v.node, None, [], None)
    v = pool2d(block, [[v]], pool_type='max', kernel=(3, 3), stride=(2, 2), padding=(0, 0), is_exit=True)
    return v, block


def block_tail(v):
    block = Block(v.node, None, [], None)
    v = conv2d(block, [[v]], out_channels=1000, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
    v = pool2d(block, [[v]], pool_type='global_avg', is_exit=True)
    return v, block


def squeezenet():
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224))
    v, block1 = block_front(pv)
    v, block2 = fire_complex(v, 16, 64)
    v, block3 = fire_complex(v, 16, 64)
    v, block4 = fire_complex(v, 32, 128)
    v, block5 = block_middle(v)
    v, block6 = fire_complex(v, 32, 128)
    v, block7 = fire_complex(v, 48, 192)
    v, block8 = fire_complex(v, 48, 192)
    v, block9 = fire_complex(v, 64, 256)
    v, block10 = block_middle(v)
    v, block11 = fire_complex(v, 64, 256)
    v, block12 = block_tail(v)
    graph = Graph("squeezenet", pv.node, [block1, block2, block3, block4, block5, block6, block7, block8, block9, block10, block11, block12])
    graph.init_weights()
    return graph

