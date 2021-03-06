import ios
import ios.tvm_utils
import ios.models
# import ios.contrib.ios_runtime
import ios.taso_utils
from pprint import pprint

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


def inception_v3() -> ios.Graph:
    v = ios.placeholder(output_shape=(768, 17, 17))
    block = ios.Block(enter_node=v.node)
    v = ios.conv2d(block, [[v]], out_channels=128, kernel=(1, 1), stride=(1, 1), padding=(0, 0), act='relu')
    v = ios.conv2d(block, [[v]], out_channels=128, kernel=(1, 7), stride=(1, 1), padding=(0, 3), act='relu', is_exit=True)
    graph = ios.Graph(name="inception_v3", input=block.enter_node, blocks=[block])
    graph.init_weights()
    return graph


def main():
    # graph = sample_network()
    graph = ios.models.inception_v3()
    # graph = inception_v3()
    graph.sequential_schedule()
    # print(ios.contrib.ios_runtime.graph_latency(graph, batch_size=16, number=1, repeat=15))
    print(ios.tvm_utils.graph_latency_local(graph, batch_size=16, number=1, repeat=15, target='cuda --libs=cudnn'))
    # print(ios.taso_utils.graph_latency(graph, batchsize=16, warmup=1, number=1, repeat=15, optimize=False, budget=-1))

if __name__ == '__main__':
    main()
