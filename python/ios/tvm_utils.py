import os
import numpy as np
import tvm
import tvm.rpc as rpc
import tvm.contrib.ndk
import tvm.contrib.graph_runtime as runtime
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm import autotvm
from tvm import relay

from ios.ir import *


def tvm_minor_version():
    return int(tvm.__version__.split('.')[1])


def get_tracker(host=None, port=None):
    if host and port:
        return rpc.connect_tracker(host, port)
    else:
        host = os.environ.get('RPC_HOST')
        port = os.environ.get('RPC_PORT')
        return rpc.connect_tracker(host, int(port))


def conv2d(x, node: Conv, params):
    wname = f'{node.name}.weight'
    bname = f'{node.name}.bias'
    weight = tvm.nd.array(node.weight)
    bias = tvm.nd.array(node.bias)
    params[wname] = weight
    params[bname] = bias

    x = relay.nn.conv2d(x, weight=relay.var(wname, shape=weight.shape), strides=node.stride, padding=node.padding, channels=node.out_channels, kernel_size=node.kernel, groups=node.groups)
    x = relay.nn.bias_add(x, relay.var(bname, shape=bias.shape), axis=1)
    if node.act == "relu":
        x = relay.nn.relu(x)
    elif node.act == "tanh":
        x = relay.tanh(x)
    elif node.act == "identity":
        x = x
    return x


def pool2d(x, pool: Pool):
    if pool.pool_type == 'global_avg':
        x = relay.nn.global_avg_pool2d(x)
    elif pool.pool_type == 'avg':
        x = relay.nn.avg_pool2d(x, pool_size=pool.kernel, strides=pool.stride, padding=pool.padding,
                                count_include_pad=True)
    elif pool.pool_type == 'max':
        x = relay.nn.max_pool2d(x, pool_size=pool.kernel, strides=pool.stride, padding=pool.padding)
    else:
        raise NotImplemented
    return x


def sequential(x, seq: Sequential, params):
    for node in seq.nodes:
        x = do_layer(x, node, params)
    return x


def do_layer(x, node: Node, params):
    if isinstance(node, Conv):
        x = conv2d(x, node, params)
    elif isinstance(node, Pool):
        x = pool2d(x, node)
    elif isinstance(node, Identity):
        x = x
    elif isinstance(node, Activation):
        if node.act_type == 'relu':
            x = relay.nn.relu(x)
        elif node.act_type == 'sigmoid':
            x = relay.sigmoid(x)
        elif node.act_type == 'tanh':
            x = relay.tanh(x)
        else:
            raise ValueError
    elif isinstance(node, Element):
        x = x  # has been done in terms
    elif isinstance(node, Relu):
        x = relay.nn.relu(x)
    elif isinstance(node, Sequential):
        x = sequential(x, node, params)
    else:
        raise NotImplementedError
    return x


def graph2relay(graph: Graph, batch_size):
    node2var = {}
    params = {}

    for node in graph.nodes():
        if node is graph.enter_node:
            node2var[node] = relay.var(name_hint=node.name, shape=(batch_size, *node.output_shape))
        else:
            term_vars = []
            for term in node.inputs:
                value_vars = []
                for value in term:
                    if value.begin == 0 and value.end == value.node.output_shape[0]:
                        var = node2var[value.node]
                    else:
                        var = relay.strided_slice(node2var[value.node], begin=[0, value.begin, 0, 0], end=[batch_size, value.end, *value.node.output_shape[1:]])
                    value_vars.append(var)
                term_var = value_vars[0]
                for value_var in value_vars[1:]:
                    if isinstance(node, Element):
                        if node.op_type == 'mul':
                            term_var = relay.multiply(term_var, value_var)
                        elif node.op_type == 'add':
                            term_var = term_var + value_var
                        else:
                            raise ValueError
                    else:
                        term_var = term_var + value_var
                term_vars.append(term_var)
            if len(term_vars) > 1:
                x = relay.concatenate(term_vars, axis=1)
            else:
                x = term_vars[0]
            node2var[node] = do_layer(x, node, params)

    x = node2var[graph.exit_node]
    fn = relay.Function(relay.analysis.free_vars(x), x)

    if tvm_minor_version() <= 6:
        return relay.Module.from_expr(fn), params
    else:
        return tvm.ir.IRModule.from_expr(fn), params



def block2relay(block: Block, batch_size):
    node2var = {}
    params = {}

    x = relay.var(name_hint=block.enter_node.name, shape=(batch_size, *block.enter_node.output_shape))
    node2var[block.enter_node] = x
    for node in block.inner_nodes + [block.exit_node]:
        term_vars = []
        assert isinstance(node, (Pool, Conv, Identity))
        for term in node.inputs:
            value_vars = []
            for value in term:
                if value.begin == 0 and value.end == value.node.output_shape[0]:
                    var = node2var[value.node]
                else:
                    var = relay.strided_slice(node2var[value.node], begin=[0, value.begin, 0, 0],
                                              end=[batch_size, value.end, *value.node.output_shape[1:]])
                value_vars.append(var)
            term_var = value_vars[0]
            for value_var in value_vars[1:]:
                term_var = term_var + value_var
            term_vars.append(term_var)
        if len(term_vars) > 1:
            x = relay.concatenate(term_vars, axis=1)
        else:
            x = term_vars[0]
        if isinstance(node, Conv):
            node2var[node] = conv2d(x, node, params)
        elif isinstance(node, Pool):
            node2var[node] = pool2d(x, node)
        elif isinstance(node, Identity):
            node2var[node] = x
        else:
            raise NotImplemented
    fn = relay.Function(relay.analysis.free_vars(x), x)
    if tvm_minor_version() <= 6:
        return relay.Module.from_expr(fn), params
    else:
        return tvm.ir.IRModule.from_expr(fn), params


def conv2relay(node: Conv, batch_size):
    node2var = {}
    params = {}

    for term in node.inputs:
        for value in term:
            x = relay.var(name_hint=value.node.name, shape=(batch_size, *value.node.output_shape))
            node2var[value.node] = x
    term_vars = []
    for term in node.inputs:
        value_vars = []
        for value in term:
            if value.begin == 0 and value.end == value.node.output_shape[0]:
                var = node2var[value.node]
            else:
                var = relay.strided_slice(node2var[value.node], begin=[0, value.begin, 0, 0],
                                          end=[batch_size, value.end, *value.node.output_shape[1:]])
            value_vars.append(var)
        term_var = value_vars[0]
        for value_var in value_vars[1:]:
            term_var = term_var + value_var
        term_vars.append(term_var)
    if len(term_vars) > 1:
        x = relay.concatenate(term_vars, axis=1)
    else:
        x = term_vars[0]
    x = conv2d(x, node, params)
    node2var[node] = x
    fn = relay.Function(relay.analysis.free_vars(x), x)
    if tvm_minor_version() <= 6:
        return relay.Module.from_expr(fn), params
    else:
        return tvm.ir.IRModule.from_expr(fn), params


def relay_latency(relay_module, params, target, number, repeat, target_host=None):
    with relay.build_config(opt_level=3):
        graph_json, tvm_module, params = relay.build(relay_module, target=target, target_host=target_host,
                                                     params=params)
    if target.startswith('cuda'):
        ctx = tvm.gpu()
    else:
        ctx = tvm.cpu()
    assert isinstance(tvm_module, tvm.module.Module)
    graph_module = runtime.create(graph_json_str=graph_json, libmod=tvm_module, ctx=ctx)
    graph_module.set_input(**params)
    ftimer = graph_module.module.time_evaluator('run', ctx, number, repeat)
    return [float(v) * 1000.0 for v in ftimer().results]


def tune_and_compile(graph: Graph, batch_size, target, target_host, device=None):
    #
    # this function is adopted and modified from tvm tutorial
    #
    log_dir = "./tvm_schedule_configs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{graph.name}_{device}_{batch_size}.log")
    tuning_option = {
        'log_filename': log_file,
        'tuner': 'ga',
        'n_trial': 2000,
        'early_stopping': 600,
        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
#            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
            runner=autotvm.RPCRunner(
                'v100',  # change the device key to your key
                '0.0.0.0', 9190,
                number=20, repeat=3, timeout=4),
        )
    }

    # You can skip the implementation of this function for this tutorial.
    def tune_tasks(tasks,
                   measure_option,
                   tuner,
                   n_trial,
                   early_stopping,
                   log_filename,
                   use_transfer_learning=True):
        # create tmp log file
        tmp_log_file = log_filename + ".tmp"
        for i, tsk in enumerate(reversed(tasks)):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

            # create tuner
            if tuner == 'xgb' or tuner == 'xgb-rank':
                tuner_obj = XGBTuner(tsk, loss_type='rank')
            elif tuner == 'ga':
                tuner_obj = GATuner(tsk, pop_size=100)
            elif tuner == 'random':
                tuner_obj = RandomTuner(tsk)
            elif tuner == 'gridsearch':
                tuner_obj = GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            if use_transfer_learning:
                if os.path.isfile(tmp_log_file):
                    tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

            # do tuning
            # print(f"tsk.config_space {tsk.config_space}")
            tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                           early_stopping=early_stopping,
                           measure_option=measure_option,
                           callbacks=[
                               autotvm.callback.progress_bar(n_trial, prefix=prefix),
                               autotvm.callback.log_to_file(tmp_log_file)])

        # pick best records to a cache file
        autotvm.record.pick_best(tmp_log_file, log_filename)
        os.remove(tmp_log_file)
    mod, params = graph2relay(graph, batch_size)
    input_shape = (batch_size,) + tuple(graph.enter_node.output_shape)
    out_shape = (batch_size,) + tuple(graph.blocks[-1].exit_node.output_shape)
    # print(input_shape, out_shape)

    tasks = autotvm.task.extract_from_program(mod["main"], target=target, target_host=target_host,
                                              params=params, ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    if os.path.exists(log_file):
        print(f"Tuned config found, use {log_file} as config")
    else:
        print("Tuning...")
        tune_tasks(tasks, **tuning_option)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        # print("Compile...")
        with relay.build_config(opt_level=3):  # opt_level = 3 has problem
            graph, lib, params = relay.build_module.build(mod, target=target, target_host=target_host, params=params)

        return graph, lib, params


def compile(graph: Graph, batch_size, target, target_host):
    relay_module, params = graph2relay(graph, batch_size)
    with relay.build_config(opt_level=3):
        graph_json, tvm_module, params = relay.build(relay_module, target=target, target_host=target_host, params=params)
    return graph_json, tvm_module, params


def print_source_files(module, dirname: str, name):
    with open(f'{dirname}/{name}.txt', 'w') as f:
        f.write(module.get_source())
    for index, sub_module in enumerate(module.imported_modules):
        print_source_files(sub_module, dirname, f"{name}_{index}")


def graph_inference(graph: Graph, batch_size, input: np.ndarray):
    graph_json, tvm_module, params = compile(graph, batch_size, 'llvm', target_host=None)
    graph_module = runtime.create(graph_json_str=graph_json, libmod=tvm_module, ctx=tvm.cpu())
    graph_module.set_input(**params)
    graph_module.set_input(graph.input.name, tvm.nd.array(input))
    graph_module.run()
    out: tvm.nd.NDArray = graph_module.get_output(0)
    return out.asnumpy()


def graph_latency_local(graph: Graph, batch_size, number, repeat, target, target_host=None, tune=False, device=None):
    """
    Measure the latency of graph in TVM.

    :param graph: ios.Graph
        The computation graph to measure the latency.

    :param batch_size: int
        The batch size used in the measurement.

    :param number: int
        The number of execution in one 'repeat'.

    :param repeat: int
        The number of 'repeat' in the measurement.

    :param target: str
        The compile target.
        When target='cuda', TVM compile the computation graph to NVIDIA CUDA platform.
        When target='cuda -libs=cudnn', TVM will use the cuDNN library to execute convolutions.

    :param target_host: str or None, default None
        The target host.

    :param tune: boolean, default False
        Whether to tune the network before measure the latency.

    :param device: str or None, default None
        The string used to represent execution device. Used in the schedule config name.

    :return: List[float]
        The measurement results.
    """
    if tune:
        graph_json, tvm_module, params = tune_and_compile(graph, batch_size, target, target_host, device=device)
    else:
        graph_json, tvm_module, params = compile(graph, batch_size, target, target_host)
    if target.startswith('cuda'):
        ctx = tvm.gpu()
    else:
        ctx = tvm.cpu()
    graph_module = runtime.create(graph_json_str=graph_json, libmod=tvm_module, ctx=ctx)
    graph_module.set_input(**params)
    ftimer = graph_module.module.time_evaluator('run', ctx, number, repeat)
    return [float(v) * 1000.0 for v in ftimer().results]


def graph_latency_rpc(graph: Graph, batch_size, number, repeat, target, tracker: tvm.rpc.TrackerSession, device_key,
                      target_host=None):
    graph_json, tvm_module, params = compile(graph, batch_size, target, target_host=target_host)
    session = tracker.request(device_key)
    if target.startswith('cuda'):
        ctx = session.gpu()
    else:
        ctx = session.cpu()
    lib_fname = f"/tmp/{graph.name}.so"
    tvm_module.export_library(lib_fname, tvm.contrib.ndk.create_shared)
    session.upload(lib_fname)
    session_lib = session.load_module(os.path.basename(lib_fname))
    graph_module = runtime.create(graph_json, session_lib, ctx)
    graph_module.set_input(**params)
    graph_module.run()
    ftimer = graph_module.module.time_evaluator('run', ctx, number=number, repeat=repeat)
    results = list(float(v) * 1000.0 for v in ftimer().results)
    print(results)
    return results
