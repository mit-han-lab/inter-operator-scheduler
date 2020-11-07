from ios.ir import *
import time
import tensorflow as tf
from tensorflow.python.profiler import option_builder

print(tf.version.VERSION)

# tf.disable_eager_execution()
tf.compat.v1.disable_eager_execution()

CONST_VALUE = .00000182
GLOBAL_DATA_FORMAT = "NCHW"


def get_inputs(node, name2tensor, batch_size):
    inputs = node.inputs
    term_tensors = []
    for term in inputs:
        value_tensors = []
        for value in term:
            if value.begin == 0 and value.end == value.node.output_shape[0]:
                value_tensors.append(name2tensor[value.node.name])
            else:
                value_tensors.append(tf.slice(name2tensor[value.node.name], (0, value.begin, 0, 0), (batch_size, value.end - value.begin, value.node.output_shape[1], value.node.output_shape[2])))
        if isinstance(node, Element):
            if node.op_type == 'add':
                term_tensors.append(tf.add_n(value_tensors))
            elif node.op_type == 'mul':
                assert len(value_tensors) == 2
                term_tensors.append(tf.multiply(value_tensors[0], value_tensors[1]))
            else:
                raise ValueError
        else:
            if len(value_tensors) == 1:
                term_tensors.append(value_tensors[0])
            else:
                term_tensors.append(tf.add_n(value_tensors))
    if len(term_tensors) == 1:
        return term_tensors[0]
    else:
        return tf.concat(term_tensors, axis=1)


def get_stride(stride):
    return (1, 1) + tuple(stride)


use_constant = True


def do_conv2d(name, x, input_shape, out_channels, groups, kernel, stride, padding, act, weights_np: np.ndarray, bias_np, use_correct=False):
    weights_name = name + "_weights"
    bias_name = name + "_bias"
    conv_name = name + "_conv2d"
    bias_add_name = name + "_bias_add"
    if groups == 1:
        if use_constant:
            weights = tf.constant(value=weights_np.transpose([2, 3, 1, 0]), dtype=tf.float32)
            bias = tf.constant(value=bias_np, dtype=tf.float32)
        else:
            weights = tf.get_variable(name=weights_name, dtype=tf.float32, initializer=weights_np.transpose([2, 3, 1, 0]))
            bias = tf.get_variable(name=bias_name, dtype=tf.float32, initializer=bias_np)
        if use_correct:
            x = tf.pad(x, [[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]])
            x = tf.nn.conv2d(x, weights, get_stride(stride), "VALID", data_format=GLOBAL_DATA_FORMAT, name=conv_name)
        else:
            padding_string = "SAME" if padding[0] > 0 or padding[1] > 0 else "VALID"
            x = tf.nn.conv2d(x, weights, get_stride(stride), padding_string, data_format=GLOBAL_DATA_FORMAT, name=conv_name)
        x = tf.nn.bias_add(x, bias, data_format=GLOBAL_DATA_FORMAT, name=bias_add_name)
        if act == "identity":
            pass
        elif act == "relu":
            x = tf.nn.relu(x)
        elif act == "tanh":
            x = tf.nn.tanh(x)
        else:
            raise ValueError()
        return x
    elif groups == input_shape[0] and groups == out_channels:
        if use_constant:
            weights = tf.constant(value=weights_np.transpose([2, 3, 0, 1]), dtype=tf.float32)
            bias = tf.constant(value=bias_np, dtype=tf.float32)
        else:
            weights = tf.get_variable(name=weights_name, dtype=tf.float32, initializer=weights_np.transpose([2, 3, 0, 1]))
            bias = tf.get_variable(name=bias_name, dtype=tf.float32, initializer=bias_np)
        if use_correct:
            x = tf.pad(x, [[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]])
            x = tf.nn.depthwise_conv2d(x, weights, get_stride(stride), "VALID", data_format=GLOBAL_DATA_FORMAT, name=conv_name)
        else:
            padding_string = "SAME" if padding[0] > 0 or padding[1] > 0 else "VALID"
            x = tf.nn.depthwise_conv2d(x, weights, get_stride(stride), padding_string, data_format=GLOBAL_DATA_FORMAT, name=conv_name)
        x = tf.nn.bias_add(x, bias, data_format=GLOBAL_DATA_FORMAT, name=bias_add_name)
        if act == "identity":
            pass
        elif act == "relu":
            x = tf.nn.relu(x)
        elif act == "tanh":
            x = tf.nn.tanh(x)
        else:
            raise ValueError()
        return x
    else:
        raise NotImplementedError


def do_layer(node: Node, name2tensor, batch_size, use_correct=False):
    name = node.name
    if isinstance(node, Placeholder):
        name2tensor[name] = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size,) + tuple(node.output_shape))
    elif isinstance(node, Sequential):
        for nd in node.nodes:
            name2tensor[name] = do_layer(nd, name2tensor, batch_size, use_correct)
    else:
        x = get_inputs(node, name2tensor, batch_size)
        if isinstance(node, Conv):
            name2tensor[name] = do_conv2d(node.name, x, node.input_shape, node.out_channels, node.groups, node.kernel,
                                          node.stride, node.padding, node.act, node.weight, node.bias, use_correct=use_correct)
        elif isinstance(node, Pool):
            if "max" in node.pool_type:
                if use_correct:
                    x = tf.pad(x, [[0, 0], [0, 0], [node.padding[0], node.padding[0]], [node.padding[1], node.padding[1]]], constant_values=-1e9)
                    x = tf.nn.max_pool2d(x, node.kernel, node.stride, "VALID", data_format=GLOBAL_DATA_FORMAT)
                else:
                    padding_string = "SAME" if node.padding[0] > 0 or node.padding[1] > 0 else "VALID"
                    x = tf.nn.max_pool2d(x, node.kernel, node.stride, padding_string, data_format=GLOBAL_DATA_FORMAT)
            elif "avg" in node.pool_type:
                if use_correct:
                    x = tf.pad(x, [[0, 0], [0, 0], [node.padding[0], node.padding[0]], [node.padding[1], node.padding[1]]])
                    x = tf.nn.avg_pool2d(x, node.kernel, node.stride, "VALID", data_format=GLOBAL_DATA_FORMAT)
                else:
                    padding_string = "SAME" if node.padding[0] > 0 or node.padding[1] > 0 else "VALID"
                    x = tf.nn.avg_pool2d(x, node.kernel, node.stride, padding_string, data_format=GLOBAL_DATA_FORMAT)
            else:
                raise ValueError
            name2tensor[name] = x
        elif isinstance(node, Relu):
            x = tf.nn.relu(x)
            name2tensor[name] = x
        elif isinstance(node, Identity):
            name2tensor[name] = x
        elif isinstance(node, Element):
            name2tensor[name] = x
        elif isinstance(node, Activation):
            if node.act_type == 'relu':
                x = tf.nn.relu(x)
            elif node.act_type == 'sigmoid':
                x = tf.nn.sigmoid(x)
            elif node.act_type == 'tanh':
                x = tf.nn.tanh(x)
            else:
                raise ValueError
            name2tensor[name] = x
        else:
            raise NotImplementedError
    return name2tensor[name]


def graph_latency(graph: Graph, batch_size, warmup, number, repeat, xla=False, use_correct=False, use_profiler=False):
    """
    Measure the latency of computation graph in Tensorflow.

    :param graph:
        The computation graph to measure the latency.

    :param batch_size:
        Measure the latency under given batch size.

    :param xla: boolean, default False
        Whether turn on xla optimization.

    :param use_correct: boolean, default False
        Whether use the correct transformation.

    :param use_profiler: boolean, default False
        Whether use profiler to measure the latency.

    :return: List[float]
        The latency measurement results.
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf.compat.v1.reset_default_graph()
    if xla:
        print("open xla")
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

    results = []

#    repeat = 10
    with tf.compat.v1.Session(config=config) as sess:
        options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        builder = option_builder.ProfileOptionBuilder
        profiler_opts = builder(builder.time_and_memory()).order_by('micros').build()

        name2tensor = {}
        for node in graph.nodes():
            do_layer(node, name2tensor, batch_size, use_correct)
        input_var = name2tensor[graph.enter_node.name]
        output_var = name2tensor[graph.blocks[-1].exit_node.name]

        input_shape = (batch_size,) + tuple(graph.enter_node.output_shape)

        sess.run(tf.compat.v1.global_variables_initializer())

        for i in range(repeat + warmup):
            input_data = np.random.random_sample(input_shape).astype(np.float32)
            if use_profiler:
                run_metadata = tf.RunMetadata()
                sess.run(output_var, feed_dict={input_var: input_data}, options=options, run_metadata=run_metadata)
                rt = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='scope', options=profiler_opts)
                results.append(rt.total_exec_micros / 1000.0)
            else:
                t1 = time.time()
                for _ in range(number):
                    sess.run(output_var, feed_dict={input_var: input_data})
                t2 = time.time()
                ta = t2 - t1

                t1 = time.time()
                for _ in range(number):
                    sess.run(input_var, feed_dict={input_var: input_data})
                t2 = time.time()
                tb = t2 - t1

                results.append((ta - tb) * 1000.0 / number)

    return results[-repeat:]


def graph_inference(graph: Graph, batch_size, input_data: np.ndarray, use_correct=False):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session(config=config) as sess:
        name2tensor = {}
        for node in graph.nodes():
            do_layer(node, name2tensor, batch_size, use_correct=use_correct)
        input_var = name2tensor[graph.enter_node.name]
        output_var = name2tensor[graph.blocks[-1].exit_node.name]

        sess.run(tf.compat.v1.global_variables_initializer())
        return sess.run(output_var, {input_var: input_data})


