from .common import *


def avg_conv(block: Block, inputs, out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, is_exit=False):
    names = [new_name(), new_name()]
    nodes = [
        Relu(names[0], names[0], inputs, None),
        Conv(names[1], names[1], None, out_channels=out_channels, kernel=kernel, stride=stride, padding=padding, groups=groups, act="identity", output_shape=None),
    ]
    return sequential(block, hint_name='ReluConv', nodes=nodes, is_exit=is_exit)


def branch_separable(block, x: Value, out_channels, kernel, stride, padding) -> Value:
    in_channels = x.node.output_shape[0]
    x = relu_sep_conv(block, [[x]], in_channels, kernel, stride, padding)
    x = relu_sep_conv(block, [[x]], out_channels, kernel, (1, 1), padding)
    return x


def branch_separable_stem(block, x: Value, out_channels, kernel, stride, padding) -> Value:
    x = relu_sep_conv(block, [[x]], out_channels, kernel, stride, padding)
    x = relu_sep_conv(block, [[x]], out_channels, kernel, (1, 1), padding)
    return x


def stem0(block, x: Value, num_filters) -> Value:
    x1 = relu_conv(block, [[x]], out_channels=num_filters, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

    c0l = branch_separable(block, x1, num_filters, kernel=(5, 5), stride=(2, 2), padding=(2, 2))

    c0r = branch_separable_stem(block, x, num_filters, kernel=(7, 7), stride=(2, 2), padding=(3, 3))
    c0 = identity(block, [[c0l, c0r]])

    c1l = pool2d(block, [[x1]], pool_type='max', kernel=(3, 3), stride=(2, 2), padding=(1, 1))
    c1r = branch_separable_stem(block, x, num_filters, kernel=(7, 7), stride=(2, 2), padding=(3, 3))
    c1 = identity(block, [[c1l, c1r]])

    c2l = pool2d(block, [[x1]], pool_type='avg', kernel=(3, 3), stride=(2, 2), padding=(1, 1))
    c2r = branch_separable_stem(block, x, num_filters, kernel=(5, 5), stride=(2, 2), padding=(2, 2))

    c3r = pool2d(block, [[c0]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    c4l = branch_separable(block, c0, num_filters, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    c4r = pool2d(block, [[x1]], pool_type='max', kernel=(3, 3), stride=(2, 2), padding=(1, 1))

    return identity(block, [[c1], [c2l, c2r], [c3r, c0], [c4l, c4r]])


def stem1(block, hp: Value, h: Value, num_filters) -> Value:
    x_left = relu_conv(block, [[hp]], num_filters, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
    x_right_1 = relu_conv(block, [[h]], num_filters // 2, kernel=(1, 1), stride=(2, 2), padding=(0, 0))
    x_right_2 = relu_conv(block, [[h]], num_filters // 2, kernel=(1, 1), stride=(2, 2), padding=(0, 0))
    x_right = identity(block, [[x_right_1, x_right_2]])

    c0l = branch_separable(block, x_left, num_filters, kernel=(5, 5), stride=(2, 2), padding=(2, 2))
    c0r = branch_separable(block, x_right, num_filters, kernel=(7, 7), stride=(2, 2), padding=(3, 3))
    c0 = identity(block, [[c0l, c0r]])

    c1l = pool2d(block, [[x_left]], pool_type='max', kernel=(3, 3), stride=(2, 2), padding=(1, 1))
    c1r = branch_separable(block, x_right, num_filters, kernel=(7, 7), stride=(2, 2), padding=(3, 3))
    c1 = identity(block, [[c1l, c1r]])

    c2l = pool2d(block, [[x_left]], pool_type='avg', kernel=(3, 3), stride=(2, 2), padding=(1, 1))
    c2r = branch_separable(block, x_right, num_filters, kernel=(5, 5), stride=(2, 2), padding=(2, 2))

    c3r = pool2d(block, [[c0]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    c4l = branch_separable(block, c0, num_filters, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    c4r = pool2d(block, [[x_left]], pool_type='max', kernel=(3, 3), stride=(2, 2), padding=(1, 1))

    return identity(block, [[c1], [c2l, c2r], [c3r, c1], [c4l, c4r]])


def first_cell(block, x: Value, x_prev: Value, out_channels_left, out_channels_right) -> Value:
    x_left_1 = relu_conv(block, [[x_prev]], out_channels_left, kernel=(1, 1), stride=(2, 2), padding=(0, 0))
    x_left_2 = relu_conv(block, [[x_prev]], out_channels_left, kernel=(1, 1), stride=(2, 2), padding=(0, 0))
    x_left = identity(block, [[x_left_1], [x_left_2]])
    x_right = relu_conv(block, [[x]], out_channels_right, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

    c0l = branch_separable(block, x_right, out_channels_right, kernel=(5, 5), stride=(1, 1), padding=(2, 2))
    c0r = branch_separable(block, x_right, out_channels_right, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    c1l = branch_separable(block, x_left, out_channels_right, kernel=(5, 5), stride=(1, 1), padding=(2, 2))
    c1r = branch_separable(block, x_left, out_channels_right, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    c2l = pool2d(block, [[x_right]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    c3l = pool2d(block, [[x_left]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    c3r = pool2d(block, [[x_left]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    c4l = branch_separable(block, x_right, out_channels_right, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    return identity(block, [[x_left], [c0l, c0r], [c1l, c1r], [c2l, x_left], [c3l, c3r], [c4l, x_right]])


def normal_cell(block, x: Value, x_prev: Value, out_channels_left, out_channels_right) -> Value:
    x_left = relu_conv(block, [[x_prev]], out_channels_left, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
    x_right = relu_conv(block, [[x]], out_channels_right, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

    c0l = branch_separable(block, x_right, out_channels_right, kernel=(5, 5), stride=(1, 1), padding=(2, 2))
    c0r = branch_separable(block, x_left, out_channels_left, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    c1l = branch_separable(block, x_left, out_channels_left, kernel=(5, 5), stride=(1, 1), padding=(2, 2))
    c1r = branch_separable(block, x_left, out_channels_left, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    c2l = pool2d(block, [[x_right]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    c3l = pool2d(block, [[x_left]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    c3r = pool2d(block, [[x_left]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    c4l = branch_separable(block, x_right, out_channels_right, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    return identity(block, [[c0l, c0r], [c1l, c1r], [c2l, x_left], [c3l, c3r], [c4l, x_right]])


def reduction_cell(block, x: Value, x_prev: Value, out_channels_left, out_channels_right) -> Value:
    x_left = relu_conv(block, [[x_prev]], out_channels_left, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
    x_right = relu_conv(block, [[x]], out_channels_right, kernel=(1, 1), stride=(1, 1), padding=(0, 0))

    c0l = branch_separable(block, x_right, out_channels_right, kernel=(5, 5), stride=(2, 2), padding=(2, 2))
    c0r = branch_separable(block, x_left, out_channels_right, kernel=(7, 7), stride=(2, 2), padding=(3, 3))
    c0 = identity(block, [[c0l, c0r]])

    c1l = pool2d(block, [[x_right]], pool_type='max', kernel=(3, 3), stride=(2, 2), padding=(1, 1))
    c1r = branch_separable(block, x_left, out_channels_right, kernel=(7, 7), stride=(2, 2), padding=(3, 3))
    c1 = identity(block, [[c1l, c1r]])

    c2l = pool2d(block, [[x_right]], pool_type='avg', kernel=(3, 3), stride=(2, 2), padding=(1, 1))
    c2r = branch_separable(block, x_left, out_channels_right, kernel=(5, 5), stride=(2, 2), padding=(2, 2))

    c3r = pool2d(block, [[c0]], pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))

    c4l = branch_separable(block, c0, out_channels_right, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    c4r = pool2d(block, [[x_right]], pool_type='max', kernel=(3, 3), stride=(2, 2), padding=(1, 1))

    return identity(block, [[c1l, c1r], [c2l, c2r], [c1, c3r], [c4l, c4r]])


def nasnet_large(stem_filters=96, filters=168) -> Graph:  # NasNet 6 @ 4032
    reset_name()

    v0 = placeholder(output_shape=(3, 331, 331))
    block = Block(v0.node, None, [], None)

    v1 = conv2d(block, [[v0]], out_channels=stem_filters, kernel=(3, 3), stride=(2, 2), padding=(0, 0), act="identity")
    v2 = stem0(block, v1, filters // 4)
    v3 = stem1(block, v2, v1, filters // 2)
    v4 = first_cell(block, v3, v2, filters // 2, filters)
    v5 = normal_cell(block, v4, v3, filters, filters)
    v6 = normal_cell(block, v5, v4, filters, filters)
    v7 = normal_cell(block, v6, v5, filters, filters)
    v8 = normal_cell(block, v7, v6, filters, filters)
    v9 = normal_cell(block, v8, v7, filters, filters)
    v10 = reduction_cell(block, v9, v8, 2 * filters, 2 * filters)
    v11 = first_cell(block, v10, v8, filters, 2 * filters)
    v12 = normal_cell(block, v11, v10, 2 * filters, 2 * filters)
    v13 = normal_cell(block, v12, v11, 2 * filters, 2 * filters)
    v14 = normal_cell(block, v13, v12, 2 * filters, 2 * filters)
    v15 = normal_cell(block, v14, v13, 2 * filters, 2 * filters)
    v16 = normal_cell(block, v15, v14, 2 * filters, 2 * filters)
    v17 = reduction_cell(block, v16, v15, 4 * filters, 4 * filters)
    v18 = first_cell(block, v17, v15, 2 * filters, 4 * filters)
    v19 = normal_cell(block, v18, v17, 4 * filters, 4 * filters)
    v20 = normal_cell(block, v19, v18, 4 * filters, 4 * filters)
    v21 = normal_cell(block, v20, v19, 4 * filters, 4 * filters)
    v22 = normal_cell(block, v21, v20, 4 * filters, 4 * filters)
    v23 = normal_cell(block, v22, v21, 4 * filters, 4 * filters)

    v24 = relu(block, [[v23]])
    v25 = pool2d(block, [[v24]], pool_type='global_avg', is_exit=True)

    block.parts = get_parts(block, split_vars=[v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25])
    # print("nasnet_large parts:")
    # print(" ".join(str(len(p)) for p in block.parts))
    graph = Graph("nasnet_large", v0.node, [block])

    graph.init_weights()
    return graph

