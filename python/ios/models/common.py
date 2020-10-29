from typing import Tuple, List
from ios.ir import Placeholder, Conv, Pool, Relu, Identity, Value, Node, Block, Sequential, Activation, Element, Graph

name_index = 0


def new_name():
    global name_index
    name = f'{name_index}'
    name_index += 1
    return name


def reset_name():
    global name_index
    name_index = 0


def placeholder(output_shape):
    name = new_name()
    holder = Placeholder(name, name, output_shape)
    return Value(holder, 0, holder.output_shape[0])


def conv2d(block: Block, inputs, out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, act="relu", is_exit=False):
    name = new_name()
    conv = Conv(name, name, inputs, out_channels, kernel, stride, padding, groups, act, None)
    conv.infer_shape()
    for ti, term in enumerate(inputs):
        for vi, value in enumerate(term):
            value.node.uses.append((conv, ti, vi))
    if is_exit:
        block.exit_node = conv
    else:
        block.inner_nodes.append(conv)
    return Value(conv, 0, out_channels)


def rwconv2d(block: Block, inputs, out_channels, stride=(1, 1), is_exit=False):
    return relu_sep_conv(block, inputs, out_channels, (3, 3), stride, (1, 1), is_exit)


def identity(block: Block, inputs, is_exit=False):
    name = new_name()
    ident = Identity(name, name, inputs, None)
    ident.infer_shape()
    for ti, term in enumerate(inputs):
        for vi, value in enumerate(term):
            value.node.uses.append((ident, ti, vi))
    if is_exit:
        block.exit_node = ident
    else:
        block.inner_nodes.append(ident)
    return Value(ident, 0, ident.output_shape[0])


def relu(block: Block, inputs, is_exit=False):
    name = new_name()
    rel = Relu(name, name, inputs, None)
    rel.infer_shape()
    for ti, term in enumerate(inputs):
        for vi, value in enumerate(term):
            value.node.uses.append((rel, ti, vi))
    if is_exit:
        block.exit_node = rel
    else:
        block.inner_nodes.append(rel)
    return Value(rel, 0, rel.output_shape[0])


def activation(block: Block, inputs, act_type, inplace, is_exit=False):
    name = new_name()
    rel = Activation(name, name, inputs, act_type=act_type, inplace=inplace, output_shape=None)
    rel.infer_shape()
    for ti, term in enumerate(inputs):
        for vi, value in enumerate(term):
            value.node.uses.append((rel, ti, vi))
    if is_exit:
        block.exit_node = rel
    else:
        block.inner_nodes.append(rel)
    return Value(rel, 0, rel.output_shape[0])


def multiply(block: Block, inputs, is_exit=False):
    return element(block, inputs, op_type='mul', is_exit=is_exit)


def addition(block: Block, inputs, is_exit=False):
    return element(block, inputs, op_type='add', is_exit=is_exit)


def element(block: Block, inputs, op_type, is_exit=False):
    name = new_name()
    rel = Element(name, name, inputs, op_type=op_type, output_shape=None)
    rel.infer_shape()
    for ti, term in enumerate(inputs):
        for vi, value in enumerate(term):
            value.node.uses.append((rel, ti, vi))
    if is_exit:
        block.exit_node = rel
    else:
        block.inner_nodes.append(rel)
    return Value(rel, 0, rel.output_shape[0])


def sigmoid(block: Block, inputs, inplace=False, is_exit=False):
    return activation(block, inputs, 'sigmoid', inplace, is_exit)


def tanh(block: Block, inputs, inplace=False, is_exit=False):
    return activation(block, inputs, 'tanh', inplace, is_exit)


def pool2d(block: Block, inputs, pool_type, kernel=(1, 1), stride=(1, 1), padding=(0, 0), is_exit=False):
    name = new_name()
    pool = Pool(name, name, inputs, pool_type, kernel, stride, padding, None)
    pool.infer_shape()
    for ti, term in enumerate(inputs):
        for vi, value in enumerate(term):
            value.node.uses.append((pool, ti, vi))
    if is_exit:
        block.exit_node = pool
    else:
        block.inner_nodes.append(pool)
    return Value(pool, 0, pool.output_shape[0])


def sequential(block: Block, hint_name, nodes: List[Node], is_exit=False):
    name = new_name()
    if hint_name is None:
        hint_name = name
    for i in range(len(nodes) - 1):
        nodes[i].infer_shape()
        nodes[i + 1].inputs = [[Value(nodes[i], 0, nodes[i].output_shape[0])]]
    seq = Sequential(name, hint_name, nodes, None)
    seq.infer_shape()
    for ti, term in enumerate(nodes[0].inputs):
        for vi, value in enumerate(term):
            value.node.uses.append((seq, ti, vi))
    if is_exit:
        block.exit_node = seq
    else:
        block.inner_nodes.append(seq)
    return Value(seq, 0, seq.output_shape[0])


def sep_conv(block: Block, inputs: List[List[Value]], out_channels, kernel, stride, padding, is_exit=False):
    names = [new_name(), new_name()]
    in_channels = sum(term[0].node.output_shape[0] for term in inputs)
    nodes = [
        Conv(names[0], names[0], inputs, out_channels=in_channels, kernel=kernel, stride=stride, padding=padding, groups=in_channels, act="identity", output_shape=None),
        Conv(names[1], names[1], None, out_channels=out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, act="identity", output_shape=None)
    ]
    return sequential(block, hint_name='SepConv', nodes=nodes, is_exit=is_exit)


def relu_sep_conv(block: Block, inputs: List[List[Value]], out_channels, kernel, stride, padding, is_exit=False):
    names = [new_name(), new_name(), new_name()]
    in_channels = sum(term[0].node.output_shape[0] for term in inputs)
    nodes = [
        Relu(names[0], names[0], inputs, None),
        Conv(names[1], names[1], None, out_channels=in_channels, kernel=kernel, stride=stride, padding=padding, groups=in_channels, act="identity", output_shape=None),
        Conv(names[2], names[2], None, out_channels=out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, act="identity", output_shape=None)
    ]
    return sequential(block, hint_name='ReluSepConv', nodes=nodes, is_exit=is_exit)


def relu_conv(block: Block, inputs, out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, is_exit=False):
    names = [new_name(), new_name()]
    nodes = [
        Relu(names[0], names[0], inputs, None),
        Conv(names[1], names[1], None, out_channels=out_channels, kernel=kernel, stride=stride, padding=padding, groups=groups, act="identity", output_shape=None),
    ]
    return sequential(block, hint_name='ReluConv', nodes=nodes, is_exit=is_exit)


def get_parts(block, split_vars: List[Value]):
    nodes = block.inner_nodes + [block.exit_node]
    indices = [int(v.node.name) for v in split_vars]
    parts = [[] for _ in range(len(indices) - 1)]
    for node in nodes:
        idx = int(node.name)
        pi = None
        for i in range(1, len(indices)):
            if indices[i - 1] < idx and idx <= indices[i]:
                pi = i - 1
        if pi is None:
            print(indices)
            print(node.name)
            print(node)
        assert pi is not None
        parts[pi].append(node)
    return parts
