from ios.ir import *
import taso

verbose = False


def get_conv_pool_pads(ishape, oshape, kernel, padding, stride):
    """
    Get the convolution and pooling pads mode based on specific padding and kernel. We need this function because IOS
    convolution parameters design follows PyTorch while TASO's padding parameter design follows Tensorflow.
    """
    if verbose:
        print(f"ishape: {ishape}")
        print(f"oshape: {oshape}")
        print(f"kernel: {kernel}")
        print(f"padding: {padding}")
        print(f"stride: {stride}")

    if sum(padding) == 0 and sum(kernel) > 2:
        pads = "VALID"
    else:
        pads = "SAME"

    if verbose:
        print(f"got {pads}")
    return pads


def get_conv_pads(conv: Conv):
    """
    Get the padding mode of a convolution.
    """
    return get_conv_pool_pads(conv.input_shape, conv.output_shape, conv.kernel, conv.padding, conv.stride)


def get_pool_pads(pool: Pool):
    """
    Get the padding mode of a pooling.
    """
    return get_conv_pool_pads(pool.input_shape, pool.output_shape, pool.kernel, pool.padding, pool.stride)


def check_shape(node: Node, tensor, pads):
    """
    Check the output shape of ios.ir.Node and taso.Tensor. Used to make sure the network is converted correctly.
    """
    ashape = node.output_shape
    bshape = [tensor.dim(i) for i in range(tensor.nDim)]
    for a, b in zip(ashape, bshape[1:]):
        if a != b:
            print("Error")
            print(f"shape check failed for op {node.name}, {ashape} vs {bshape}")
            info = node.export_config()
            if isinstance(node, Conv):
                info['input_shape'] = node.input_shape
            print("\n".join(str(v) for v in info.items()))
            print(f"used pads: {pads}")
            exit(0)


def conv2d(x, node: Conv, tg):
    """
    Add convolution to TASO computation graph.
    """
    weight = tg.new_weight(dims=node.weight_shape)
    if node.act == "relu":
        acts = "RELU"
    elif node.act == "identity":
        acts = "NONE"
    else:
        raise ValueError
    pads = get_conv_pads(node)
    tensor = tg.conv2d(input=x, weight=weight, strides=node.stride, padding=pads, activation=acts)
    check_shape(node, tensor, pads)
    return tensor


def pool2d(x, pool: Pool, tg):
    """
    Add pooling to TASO computation graph.
    """
    if pool.pool_type == 'global_avg':
        pads = "VALID"
        x = tg.avgpool2d(input=x, kernels=pool.kernel, strides=[1, 1], padding=pads)
    elif pool.pool_type == 'avg':
        pads = get_pool_pads(pool)
        x = tg.avgpool2d(input=x, kernels=pool.kernel, strides=pool.stride, padding=pads)
    elif pool.pool_type == 'max':
        pads = get_pool_pads(pool)
        x = tg.maxpool2d(input=x, kernels=pool.kernel, strides=pool.stride, padding=pads)
    else:
        raise NotImplemented
    check_shape(pool, x, pads)
    return x


def sequential(x, seq: Sequential, tg):
    """
    Add Sequential operator to TASO computation graph.
    """
    for node in seq.nodes:
        x = do_layer(x, node, tg)
    return x


def do_layer(x, node: Node, tg):
    """
    Add operator to TASO computation graph.
    """
    if isinstance(node, Conv):
        x = conv2d(x, node, tg)
    elif isinstance(node, Pool):
        x = pool2d(x, node, tg)
    elif isinstance(node, Identity):
        x = x
    elif isinstance(node, Activation):
        if node.act_type == 'relu':
            x = tg.relu(x)
        elif node.act_type == 'sigmoid':
            x = tg.sigmoid(x)
        elif node.act_type == 'tanh':
            x = tg.tanh(x)
        else:
            raise ValueError
    elif isinstance(node, Element):
        x = x  # has been done in terms
    elif isinstance(node, Relu):
        x = tg.relu(x)
    elif isinstance(node, Sequential):
        x = sequential(x, node, tg)
    else:
        raise NotImplementedError
    return x


def tensor2shape(tensor):
    """
    Return the shape string of tensor.
    """
    return "(" + ",".join([str(tensor.dim(i)) for i in range(tensor.nDim)]) + ")"


def graph_ios2taso(graph: Graph, batch_size):
    """
    Convert an IOS computation graph to TASO computation graph with given batch size.

    :param graph: ios.ir.Graph
        IOS computation graph.
    :param batch_size:
        Batch size that is used in the TASO computation graph.
    :return: taso.Graph
        The equivalent TASO computation graph to IOS's.
    """
    node2var = {}
    tg = taso.new_graph()

    for node in graph.nodes():
        if node is graph.enter_node:
            node2var[node] = tg.new_input(dims=(batch_size, *node.output_shape))
        else:
            term_vars = []
            for term in node.inputs:
                value_vars = []
                for value in term:
                    if value.begin == 0 and value.end == value.node.output_shape[0]:
                        var = node2var[value.node]
                    else:
                        raise NotImplementedError
                    value_vars.append(var)
                term_var = value_vars[0]
                for value_var in value_vars[1:]:
                    if isinstance(node, Element):
                        if node.op_type == 'mul':
                            term_var = tg.mul(term_var, value_var)
                        elif node.op_type == 'add':
                            term_var = tg.add(term_var, value_var)
                        else:
                            raise ValueError
                    else:
                        term_var = tg.add(term_var, value_var)
                term_vars.append(term_var)
            if len(term_vars) > 1:
                x = tg.concat(1, term_vars)
            else:
                x = term_vars[0]
            node2var[node] = do_layer(x, node, tg)

    x = node2var[graph.exit_node]
    return tg


def graph_latency(graph: Graph, batchsize, warmup, number, repeat, optimize, alpha=1.0, budget=1000):
    """
    Measure the latency of TASO optimized computation graph in TASO framework.

    :param graph: ios.ir.Graph
        The computation graph that is going to measure the latency.

    :param batchsize: int
        The execution batch size.

    :param warmup: int
        Not used.

    :param number: int
        Not used.

    :param repeat: int
        The number of latency measurement.

    :param optimize: boolean
        When optimize=True, optimize the computation graph in TASO and measure the latency.
        When optimize=False, directly measure the latency in TASO.

    :param alpha:
        The relaxation coefficient.

    :param budget:
        The iteration budget.

    :return: List[float]
        The latency measurement results.
    """
    tg = graph_ios2taso(graph, batchsize)
    if optimize:
        tg = taso.optimize(tg, alpha=alpha, budget=budget)
    # warmup and number has already implemented in taso itself
    return [tg.run_time() for _ in range(repeat)]

