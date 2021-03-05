import argparse
import os
from ios.cost_model import IOSCostModel
from ios.models.common import *


def model_a(ch, r) -> Graph:
    reset_name()
    v = placeholder(output_shape=(ch, r, r))
    block = Block(v.node, None, [], None)
    v1 = conv2d(block, [[v]], out_channels=ch, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', is_exit=False)         # 1
    v2 = conv2d(block, [[v]], out_channels=ch * 2, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', is_exit=False)     # 2
    v3 = conv2d(block, [[v]], out_channels=ch, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', is_exit=False)         # 3
    v1 = conv2d(block, [[v1]], out_channels=ch * 2, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', is_exit=False)    # 4
    vf = identity(block, [[v1], [v2], [v3]], is_exit=True)
    graph = Graph("demo", v.node, [block])
    graph.init_weights()
    return graph


expr_dir = f'./outputs'
os.makedirs(expr_dir, exist_ok=True)

event2index = {
    "active_warps_pm": 0,
    "l2_subp0_read_sector_misses": 1,
    "l2_subp0_total_read_sector_queries": 2,
    "l2_subp1_read_sector_misses": 3,
    "l2_subp1_total_read_sector_queries": 4
}


argparser = argparse.ArgumentParser()
argparser.add_argument('--ename', type=str, required=True)
argparser.add_argument('--event', type=int, choices=list(event2index.values()), default=event2index['active_warps_pm'])
args = argparser.parse_args()

batch_size = 1
warmup = 10000 + args.event  # magic number to enable profiling. warmup = 10000 + event_index
number = 1500
repeat = 1

def main(channels, resolutions):
    cost_model = IOSCostModel()
    # print(f"channels: {channels}   resolutions: {resolutions}")
    for ch in channels:
        for r in resolutions:
            graph = model_a(ch, r)

            graph_seq = model_a(ch, r)
            graph_seq.sequential_schedule()
            graph_seq.blocks[0].stages = [[["1"]], [["4"]], [["3"]], [["2"]], [["5"]]]
            graph_seq.name = graph.name + "_sequential"

            graph_ios = model_a(ch, r)
            graph_ios.sequential_schedule()
            graph_ios.blocks[0].stages = [[["1"], ["2"]], [["3"], ["4"]], [["5"]]]
            graph_ios.name = graph.name + "_ios"

            if args.ename == 'seq':
                latency = cost_model.get_graph_latency(graph_seq, batch_size, warmup, number, repeat, profile_stage=False)
            elif args.ename == 'ios':
                latency = cost_model.get_graph_latency(graph_ios, batch_size, warmup, number, repeat, profile_stage=False)
            else:
                raise ValueError
            # print(latency)


if __name__ == "__main__":
    main(channels=[384], resolutions=[15])

