"""
Utilities used to construct computation graph.

Please refer the definition of inception v3, nasnet, randwire, and squeezenet for the definition of network.
"""
from typing import Tuple, List
from ios.ir import Placeholder, Conv, Pool, Relu, Identity, Value, Node, Block, Sequential, Activation, Element, Graph

name_index = 0


def new_name():
    """
    Get a name for new operator.
    """
    global name_index
    name = f'{name_index}'
    name_index += 1
    return name


def reset_name():
    """
    Reset the name counter. It is recommended to call this function before construct any new network.
    """
    global name_index
    name_index = 0


def placeholder(output_shape):
    """
    Placeholder.

    :param output_shape: Tuple[int, int, int]
        The outpu shape of the placeholder, which should be consistent with the input of the network.
    :return: Value
        The value represents the placeholder output.
    """
    name = new_name()
    holder = Placeholder(name, name, output_shape)
    return Value(holder, 0, holder.output_shape[0])


def conv2d(block: Block, inputs, out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, act="relu",
           is_exit=False):
    """
    Add a convolution operator to the end of given block.

    :param block: ios.Block
        The block to add the operator

    :param inputs: Sequence[Sequence[Value]]
        The inputs of the convolution. 'inputs' contains a list of terms. A term contains a list of values. The values
        in a term are added up. The terms are concatenated along with the channel dimension.

    :param out_channels:
        The number of output channels.

    :param kernel: Tuple[int, int], default (1, 1)
        The kernel size.

    :param stride: Tuple[int, int], default (1, 1)
        The stride size.

    :param padding: Tuple[int, int], default (0, 0)

    :param groups: int, default 1
        The number of groups. It must be a common factor of the input channels and output channels.

    :param act: str, default 'relu'
        The activation applied to the output of convolution.

    :param is_exit: boolean, default False
        Whether this operator is the exit operator of the block.

    :return: Value
        A value represents the output of the operator.
    """
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
    """
    Add a Randwire operator at the end of given block. It consists of a Relu activation and a Separate convolution. The
    kernel size of the separate convolution is 3x3 and the padding is 1x1.

    :param block: ios.Block
        The block to add the operator

    :param inputs: Sequence[Sequence[Value]]
        The inputs of the convolution. 'inputs' contains a list of terms. A term contains a list of values. The values
        in a term are added up. The terms are concatenated along with the channel dimension.

    :param out_channels: int
        The number of output channels of the separate convolution.

    :param stride: Tuple[int, int], default (1, 1)
        The stride of the separate convolution.

    :param is_exit: boolean, default False
        Whether this operator is the exit operator of the block.

    :return: Value
        A value represents the output of the operator.
    """
    return relu_sep_conv(block, inputs, out_channels, (3, 3), stride, (1, 1), is_exit)


def identity(block: Block, inputs, is_exit=False):
    """
    Add Identity operator at the end of given block. Because the inputs can can do the addition and concatenation
    operation, this operator works as addition and concatenation.

    :param block: ios.Block
        The block to add the operator

    :param inputs: Sequence[Sequence[Value]]
        The inputs of the convolution. 'inputs' contains a list of terms. A term contains a list of values. The values
        in a term are added up. The terms are concatenated along with the channel dimension.

    :param is_exit: boolean, default False
        Whether this operator is the exit operator of the block.

    :return: Value
        A value represents the output of the operator.
    """
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
    """
    Add a Relu activation at the end of given block. This function is equivalent to activation(act_type='relu').

    :param block: ios.Block
        The block to add the operator

    :param inputs: Sequence[Sequence[Value]]
        The inputs of the convolution. 'inputs' contains a list of terms. A term contains a list of values. The values
        in a term are added up. The terms are concatenated along with the channel dimension.

    :param is_exit: boolean, default False
        Whether this operator is the exit operator of the block.

    :return: Value
        A value represents the output of the operator.
    """
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
    """
    Add an activation at the end of given block. This function is equivalent to activation(act_type='relu').

    :param block: ios.Block
        The block to add the operator

    :param inputs: Sequence[Sequence[Value]]
        The inputs of the convolution. 'inputs' contains a list of terms. A term contains a list of values. The values
        in a term are added up. The terms are concatenated along with the channel dimension.

    :param act_type: str, must be one of 'relu', 'tanh', and 'sigmoid'.
        The activation type string.

    :param inplace: boolean
        Whether to do the activation inplace.

    :param is_exit: boolean, default False
        Whether this operator is the exit operator of the block.

    :return: Value
        A value represents the output of the operator.
    """
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
    """
    Add an element-wise multiplication operator at the end of block.

    More information please refer to ios.element.
    """
    return element(block, inputs, op_type='mul', is_exit=is_exit)


def addition(block: Block, inputs, is_exit=False):
    """
    Add an element-wise addition operator at the end of block.

    More information please refer to ios.element.
    """
    return element(block, inputs, op_type='add', is_exit=is_exit)


def element(block: Block, inputs, op_type, is_exit=False):
    """
    Add an element-wise operator at the end of given block.

    :param block: ios.Block
        The block to add the operator

    :param inputs: Sequence[Sequence[Value]]
        The inputs of the convolution. 'inputs' contains a list of terms. A term contains a list of values. The values
        in a term are added up. The terms are concatenated along with the channel dimension.

    :param op_type: str, must be one of 'mul' and 'add'
        The element-wise operation type string.

    :param is_exit: boolean, default False
        Whether this operator is the exit operator of the block.

    :return: Value
        A value represents the output of the operator.
    """
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
    """
    Add a sigmoid activation operator at the end of given block.

    More information please refer to ios.activation.
    """
    return activation(block, inputs, 'sigmoid', inplace, is_exit)


def tanh(block: Block, inputs, inplace=False, is_exit=False):
    """
    Add a tanh activation operator at the end of given block.

    More information please refer to ios.activation.
    """
    return activation(block, inputs, 'tanh', inplace, is_exit)


def pool2d(block: Block, inputs, pool_type, kernel=(1, 1), stride=(1, 1), padding=(0, 0), is_exit=False):
    """
    Add a pooling operator at the end of given block.

    :param block: ios.Block
        The block to add the operator

    :param inputs: Sequence[Sequence[Value]]
        The inputs of the convolution. 'inputs' contains a list of terms. A term contains a list of values. The values
        in a term are added up. The terms are concatenated along with the channel dimension.

    :param pool_type: str, must be one of 'max', 'avg', 'global_max' and 'global_avg'
        The pooling type string.

    :param kernel: Tuple[int, int], default (1, 1)
        The kernel size of the pooling operator.

    :param stride: Tuple[int, int], default (1, 1)
        The stride of the pooling operator.

    :param padding: Tuple[int, int], default (1, 1)
        THe padding size of the pooling operator.

    :param is_exit: boolean, default False
        Whether this operator is the exit operator of the block.

    :return: Value
        A value represents the output of the operator.
    """
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
    """
    Add a sequential compound operator at the end of given block. The compound operator contains a sequence of operators
    that would be executed sequentially.

    :param block: ios.Block
        The block to add the operator

    :param hint_name: str
        Any string used to tell what the operator represents, which can be the same with other hint name.

    :param nodes: List[ios.Node]
        The sequence of the operators in the compound operator.

    :param is_exit: boolean, default False
        Whether this operator is the exit operator of the block.

    :return: Value
        A value represents the output of the last operator in the operator sequence.
    """
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
    """
    Add a separate convolution at the end of given block. This operator is a compound operator consists of a depth-wise
    convolution and a point-wise convolution.

    :param block: ios.Block
        The block to add the operator

    :param inputs: Sequence[Sequence[Value]]
        The inputs of the convolution. 'inputs' contains a list of terms. A term contains a list of values. The values
        in a term are added up. The terms are concatenated along with the channel dimension.

    :param out_channels:
        The number of output channels of the point-wise convolution.

    :param kernel:
        The kernel size of the depth-wise convolution.

    :param stride:
        The stride size of the depth-wise convolution.

    :param padding:
        The padding size of the depth-wise convolution.

    :param is_exit: boolean, default False
        Whether this operator is the exit operator of the block.

    :return: Value
        A value represents the output of the operator.
    """
    names = [new_name(), new_name()]
    in_channels = sum(term[0].node.output_shape[0] for term in inputs)
    nodes = [
        Conv(names[0], names[0], inputs, out_channels=in_channels, kernel=kernel, stride=stride, padding=padding,
             groups=in_channels, act="identity", output_shape=None),
        Conv(names[1], names[1], None, out_channels=out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0),
             groups=1, act="identity", output_shape=None)
    ]
    return sequential(block, hint_name='SepConv', nodes=nodes, is_exit=is_exit)


def relu_sep_conv(block: Block, inputs: List[List[Value]], out_channels, kernel, stride, padding, is_exit=False):
    """
    Add a compound operator that contains Relu operator and Separate operator at the end of given block.

    :param block: ios.Block
        The block to add the operator

    :param inputs: Sequence[Sequence[Value]]
        The inputs of the convolution. 'inputs' contains a list of terms. A term contains a list of values. The values
        in a term are added up. The terms are concatenated along with the channel dimension.

    :param out_channels: int
        The number of output channels of the separate convolution.

    :param kernel: Tuple[int, int]
        The kernel size of the separate convolution.

    :param stride: Tuple[int, int]
        The stride of the separate convolution.

    :param padding: Tuple[int, int]
        The padding size of the separate convolution.

    :param is_exit: boolean, default False
        Whether this operator is the exit operator of the block.

    :return: Value
        A value represents the output of the operator.
    """
    names = [new_name(), new_name(), new_name()]
    in_channels = sum(term[0].node.output_shape[0] for term in inputs)
    nodes = [
        Relu(names[0], names[0], inputs, None),
        Conv(names[1], names[1], None, out_channels=in_channels, kernel=kernel, stride=stride, padding=padding,
             groups=in_channels, act="identity", output_shape=None),
        Conv(names[2], names[2], None, out_channels=out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0),
             groups=1, act="identity", output_shape=None)
    ]
    return sequential(block, hint_name='ReluSepConv', nodes=nodes, is_exit=is_exit)


def relu_conv(block: Block, inputs, out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1,
              is_exit=False):
    """
    Add a compound operator that contains a Relu operator and a convolution operator at the end of given block.

    :param block: ios.Block
        The block to add the operator

    :param inputs: Sequence[Sequence[Value]]
        The inputs of the convolution. 'inputs' contains a list of terms. A term contains a list of values. The values
        in a term are added up. The terms are concatenated along with the channel dimension.

    :param out_channels: int
        The number of output channels of the separate convolution.

    :param kernel: Tuple[int, int], default (1, 1)
        The kernel size of the convolution.

    :param stride: Tuple[int, int], default (1, 1)
        The stride of the convolution.

    :param padding: Tuple[int, int], default (0, 0)
        The padding size of the convolution.

    :param is_exit: boolean, default False
        Whether this operator is the exit operator of the block.

    :return: Value
        A value represents the output of the operator.
    """
    names = [new_name(), new_name()]
    nodes = [
        Relu(names[0], names[0], inputs, None),
        Conv(names[1], names[1], None, out_channels=out_channels, kernel=kernel, stride=stride, padding=padding,
             groups=groups, act="identity", output_shape=None),
    ]
    return sequential(block, hint_name='ReluConv', nodes=nodes, is_exit=is_exit)


def get_parts(block, split_vars: List[Value]):
    """
    This function is used to deal with the computation graph that has a large number of operators in a single block. It
    split the operators in the block by according to the 'split_vars'. The parts can be used to guide the optimization.
    More information please see the construction of NasNet in ios.models.nasnet.

    :param block: ios.Block
        The block whose operators are splitted.

    :param split_vars: List[Value]
        The operators of given values are used to indicate the positions of splitting.

    :return: List[List[ios.Node]]
        Return a list of operator parts. A operator part contains a list of operators.
    """
    nodes = block.inner_nodes + [block.exit_node]
    indices = [int(v.node.name) for v in split_vars]
    parts = [[] for _ in range(len(indices) - 1)]
    for node in nodes:
        idx = int(node.name)
        pi = None
        for i in range(1, len(indices)):
            if indices[i - 1] < idx <= indices[i]:
                pi = i - 1
        if pi is None:
            print(indices)
            print(node.name)
            print(node)
        assert pi is not None
        parts[pi].append(node)
    return parts
