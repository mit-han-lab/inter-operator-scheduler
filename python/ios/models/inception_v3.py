from .common import *


def inception_front(v: Value) -> Tuple[Value, Block]:  # 3 x 299 x 299
    block = Block(v.node, None, [], None)
    v = conv2d(block, [[v]], out_channels=32, kernel=(3, 3), stride=(2, 2))  # 32 x 149 x 149
    v = conv2d(block, [[v]], out_channels=32, kernel=(3, 3))  # 32 x 147 x 147
    v = conv2d(block, [[v]], out_channels=64, kernel=(3, 3), padding=(1, 1))  # 64 x 147 x 147
    v = pool2d(block, [[v]], pool_type='max', kernel=(3, 3), stride=(2, 2))  # 64 x 73 x 73
    v = conv2d(block, [[v]], 80, kernel=(1, 1))  # 80 x 73 x 73
    v = conv2d(block, [[v]], out_channels=192, kernel=(3, 3))  # 192 x 71 x 71
    v = pool2d(block, [[v]], pool_type='max', kernel=(3, 3), stride=(2, 2), is_exit=True)  # 192 x 35 x 35
    return v, block


def inception_a(v: Value, pool_features) -> Tuple[Value, Block]:
    block = Block(v.node, None, [], None)

    v1x1 = conv2d(block, [[v]], out_channels=64, kernel=(1, 1))

    v5x5 = conv2d(block, [[v]], out_channels=48, kernel=(1, 1))
    v5x5 = conv2d(block, [[v5x5]], out_channels=64, kernel=(5, 5), padding=(2, 2))

    v3x3dbl = conv2d(block, [[v]], out_channels=64, kernel=(1, 1))
    v3x3dbl = conv2d(block, [[v3x3dbl]], out_channels=96, kernel=(3, 3), padding=(1, 1))
    v3x3dbl = conv2d(block, [[v3x3dbl]], out_channels=96, kernel=(3, 3), padding=(1, 1))

    v_pool = pool2d(block, [[v]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    v_pool = conv2d(block, [[v_pool]], out_channels=pool_features, kernel=(1, 1))
    return identity(block, [[v1x1], [v5x5], [v3x3dbl], [v_pool]], is_exit=True), block


def inception_b(v: Value) -> Tuple[Value, Block]:
    block = Block(v.node, None, [], None)

    v3x3 = conv2d(block, [[v]], out_channels=384, kernel=(3, 3), stride=(2, 2))

    v3x3dbl = conv2d(block, [[v]], out_channels=64, kernel=(1, 1))
    v3x3dbl = conv2d(block, [[v3x3dbl]], out_channels=96, kernel=(3, 3), padding=(1, 1))
    v3x3dbl = conv2d(block, [[v3x3dbl]], out_channels=96, kernel=(3, 3), stride=(2, 2))

    v_pool = pool2d(block, [[v]], pool_type='max', kernel=(3, 3), stride=(2, 2))
    return identity(block, [[v3x3], [v3x3dbl], [v_pool]], is_exit=True), block


def inception_c(v, channels_7x7):
    block = Block(v.node, None, [], None)

    v1x1 = conv2d(block, [[v]], out_channels=192, kernel=(1, 1))

    c7 = channels_7x7
    v7x7 = conv2d(block, [[v]], out_channels=c7, kernel=(1, 1))
    v7x7 = conv2d(block, [[v7x7]], out_channels=c7, kernel=(1, 7), padding=(0, 3))
    v7x7 = conv2d(block, [[v7x7]], out_channels=192, kernel=(7, 1), padding=(3, 0))

    v7x7dbl = conv2d(block, [[v]], out_channels=c7, kernel=(1, 1))
    v7x7dbl = conv2d(block, [[v7x7dbl]], out_channels=c7, kernel=(7, 1), padding=(3, 0))
    v7x7dbl = conv2d(block, [[v7x7dbl]], out_channels=c7, kernel=(1, 7), padding=(0, 3))
    v7x7dbl = conv2d(block, [[v7x7dbl]], out_channels=c7, kernel=(7, 1), padding=(3, 0))
    v7x7dbl = conv2d(block, [[v7x7dbl]], out_channels=192, kernel=(1, 7), padding=(0, 3))

    v_pool = pool2d(block, [[v]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    v_pool = conv2d(block, [[v_pool]], out_channels=192, kernel=(1, 1))
    return identity(block, [[v1x1], [v7x7], [v7x7dbl], [v_pool]], is_exit=True), block


def inception_d(v):
    block = Block(v.node, None, [], None)

    v3x3 = conv2d(block, [[v]], out_channels=192, kernel=(1, 1))
    v3x3 = conv2d(block, [[v3x3]], out_channels=320, kernel=(3, 3), stride=(2, 2))

    v7x7x3 = conv2d(block, [[v]], out_channels=192, kernel=(1, 1))
    v7x7x3 = conv2d(block, [[v7x7x3]], out_channels=192, kernel=(1, 7), padding=(0, 3))
    v7x7x3 = conv2d(block, [[v7x7x3]], out_channels=192, kernel=(7, 1), padding=(3, 0))
    v7x7x3 = conv2d(block, [[v7x7x3]], out_channels=192, kernel=(3, 3), stride=(2, 2))

    v_pool = pool2d(block, [[v]], pool_type='max', kernel=(3, 3), stride=(2, 2))
    return identity(block, [[v3x3], [v7x7x3], [v_pool]], is_exit=True), block


def inception_e(v):
    block = Block(v.node, None, [], None)

    v1x1 = conv2d(block, [[v]], out_channels=320, kernel=(1, 1))

    v3x3 = conv2d(block, [[v]], out_channels=384, kernel=(1, 1))
    v3x3a = conv2d(block, [[v3x3]], out_channels=384, kernel=(1, 3), padding=(0, 1))
    v3x3b = conv2d(block, [[v3x3]], out_channels=384, kernel=(3, 1), padding=(1, 0))

    v3x3dbl = conv2d(block, [[v]], out_channels=448, kernel=(1, 1))
    v3x3dbl = conv2d(block, [[v3x3dbl]], out_channels=384, kernel=(3, 3), padding=(1, 1))
    v3x3dbla = conv2d(block, [[v3x3dbl]], out_channels=384, kernel=(1, 3), padding=(0, 1))
    v3x3dblb = conv2d(block, [[v3x3dbl]], out_channels=384, kernel=(3, 1), padding=(1, 0))

    v_pool = pool2d(block, [[v]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    v_pool = conv2d(block, [[v_pool]], out_channels=192, kernel=(1, 1))
    return identity(block, [[v1x1], [v3x3a], [v3x3b], [v3x3dbla], [v3x3dblb], [v_pool]], is_exit=True), block


def inception_logits(v):
    block = Block(v.node, None, [], None)

    v = pool2d(block, [[v]], pool_type='global_avg', is_exit=True)
    return v, block


def inception_v3() -> Graph:
    reset_name()

    pv = placeholder(output_shape=(3, 299, 299))
    v, block1 = inception_front(pv)
    v, block2 = inception_a(v, 32)
    v, block3 = inception_a(v, 64)
    v, block4 = inception_a(v, 64)
    v, block5 = inception_b(v)
    v, block6 = inception_c(v, 128)
    v, block7 = inception_c(v, 160)
    v, block8 = inception_c(v, 160)
    v, block9 = inception_c(v, 192)
    v, block10 = inception_d(v)
    v, block11 = inception_e(v)
    v, block12 = inception_e(v)
    v, block13 = inception_logits(v)

    graph = Graph("inception_v3", pv.node, [block1, block2, block3, block4, block5, block6, block7, block8, block9, block10, block11, block12, block13])
    graph.init_weights()
    return graph

