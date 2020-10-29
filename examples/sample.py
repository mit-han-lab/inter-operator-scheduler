import numpy as np
import ios


def sample_network():
    v = ios.placeholder(output_shape=(375, 15, 15))
    block = ios.Block(enter_node=v.node)
    v1 = ios.conv2d(block, inputs=[[v]], out_channels=375, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v2 = ios.conv2d(block, inputs=[[v]], out_channels=750, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v3 = ios.conv2d(block, inputs=[[v]], out_channels=375, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v1 = ios.conv2d(block, inputs=[[v1]], out_channels=750, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    out = ios.identity(block, inputs=[[v1], [v2], [v3]], is_exit=True)  # concat v1, v2, and v3
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


# define computation graph
graph = sample_network()

# optimize execution schedule
optimized_graph = ios.optimize(graph, batch_size=1, opt_type='dp_parallel', compute_weight=True)

# measure latency
graph.sequential_schedule()
seq_latency, stage_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=6, profile_stage=True)
print(graph)
print(f'Sequential schedule: {np.mean(seq_latency):.3f} ms')
print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(6, -1), axis=0)}\n')

opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph, batch_size=1, repeat=6, profile_stage=True)
print(optimized_graph)
print(f'Optimized schedule: {np.mean(opt_latency):.3f} ms')
print(f'     Stage latency: {np.mean(np.array(stage_latency).reshape(6, -1), axis=0)}')

# inference on ios runtime
dummy_inputs = np.random.randn(1, 375, 15, 15)
output = ios.ios_runtime.graph_inference(optimized_graph, batch_size=1, input=dummy_inputs)

