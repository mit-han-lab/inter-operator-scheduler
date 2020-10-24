import json
import time
import argparse
import numpy as np
import os
from contrib import trt_backend
from optimizer import optimize
from models import inception_v3, randwire_large, nasnet_large, squeezenet
from visualizer import draw, draw_block

argparser = argparse.ArgumentParser()

argparser.add_argument('--eid', type=str, default="eid")
argparser.add_argument('--device', type=str, required=True,
                       choices=['1080ti', '1080', 'v100', 'titanxp', '2080ti', 'cpu'])
argparser.add_argument('--model', type=str, required=True,
                       choices=['inception_v3', 'randwire', 'nasnet', 'squeezenet'])
argparser.add_argument('--bs', type=int, required=False, default=1)
argparser.add_argument('--opt_type', type=str, required=True,
                       choices=['trt', 'sequential', 'greedy', 'dp_parallel_merge', 'dp_parallel', 'dp_merge'])
argparser.add_argument('--r', type=int, required=True)
argparser.add_argument('--s', type=int, required=True)
argparser.add_argument('--warmup', type=int, required=False, default=2)
argparser.add_argument('--number', type=int, required=False, default=4)
argparser.add_argument('--repeat', type=int, required=False, default=4)

args = argparser.parse_args()
expr_dir = f'./outputs/{args.eid}-{args.device}-{args.model}-{args.bs}-{args.opt_type}-s-{args.s}r-{args.r}-{args.warmup}-{args.number}-{args.repeat}'
os.makedirs("./outputs", exist_ok=True)
os.makedirs(expr_dir, exist_ok=True)

name2model = {
        'inception_v3': inception_v3,
        'randwire': randwire_large,
        'nasnet': nasnet_large,
        'squeezenet': squeezenet
}


def main():

    summary = []
    line = f'Argument: {args}'
    print(line)
    summary.append(line)

    graph = name2model[args.model]()

    if args.opt_type == 'trt':
        t1 = time.time()
        graph_latency = trt_backend.graph_latency(graph, args.bs, args.warmup, args.number, args.repeat)
        t2 = time.time()
        line = f" {args.model} optimized with {args.opt_type} in {t2 - t1:.0f} secs: {np.mean(graph_latency):.4f}\n"
        print(line)
        summary.append(line)
    else:
        from cost_model.ios_cost_model import IOSCostModel
        cost_model = IOSCostModel()
        t1 = time.time()
        if args.opt_type == 'sequential':
            graph.sequential_schedule()
        elif args.opt_type == 'greedy':
            graph.greedy_schedule()
        else:
            graph = optimize(graph, args.bs, cost_model, args.opt_type, args.warmup, args.number, args.repeat,
                             max_num_streams=args.s, max_part_size=50, compute_weight=False, max_seq_length=args.r,
                             debug_dp_info=None, verbose=False)
        t2 = time.time()
        name = graph.name

        # draw graph
        profile_stage = False

        if profile_stage:
            graph_latency_list, stages_latency_ = cost_model.get_graph_latency(graph, args.bs, warmup=args.warmup, number=args.number, repeat=args.repeat, profile_stage=True)
            graph_latency = np.mean(graph_latency_list)
            num_stages = len(stages_latency_) // args.repeat
            stages_latency = [float(np.mean([stages_latency_[j * num_stages + i] for j in range(args.repeat)])) for i in range(num_stages)]
            label = "\n".join([
                f'{name}',
                f'latency {graph_latency:.3f}',
                f'stage latency',
                *[f"{latency:.3f}" for latency in stages_latency]
            ])
            draw(graph, f"{expr_dir}/{name}.png", label)
        else:
            graph_latency_list = cost_model.get_graph_latency(graph, args.bs, warmup=args.warmup, number=args.number, repeat=args.repeat, profile_stage=False)
            graph_latency = np.mean(graph_latency_list)
            label = "\n".join([f'{name}', f'latency {graph_latency:.3f}'])
            draw(graph, f"{expr_dir}/{name}.png", label)

        block_latency_list = []
        for bindex, block in enumerate(graph.blocks):
            if profile_stage:
                block_latency, stages_latency_ = cost_model.get_block_latency(block, args.bs, args.warmup, args.number, args.repeat, profile_stage=True)
                block_latency = np.mean(block_latency)
                block_latency_list.append(block_latency)
                num_stages = len(stages_latency_) // args.repeat
                stages_latency = [float(np.mean([stages_latency_[j * num_stages + i] for j in range(args.repeat)])) for i in range(num_stages)]
                block_dir = f'{expr_dir}/{name}_blocks'
                os.makedirs(block_dir, exist_ok=True)
                label = "\n".join([
                    f'{name}',
                    f'block {bindex}',
                    f'latency {block_latency:.3f}',
                    f'stages latency {" ".join([f"{latency:.3f}" for latency in stages_latency])}'
                ])
                draw_block(block, f'{block_dir}/{bindex}.png', label=label)
            else:
                block_latency = cost_model.get_block_latency(block, args.bs, args.warmup, args.number, args.repeat, profile_stage=False)
                block_latency = np.mean(block_latency)
                block_latency_list.append(block_latency)
                block_dir = f'{expr_dir}/{name}_blocks'
                os.makedirs(block_dir, exist_ok=True)
                label = "\n".join([
                    f'{name}',
                    f'block {bindex}',
                    f'latency {block_latency:.3f}'
                ])
                draw_block(block, f'{block_dir}/{bindex}.png', label=label)

        latency = {
            'graph latency': graph_latency_list,
            'block latency': block_latency_list
        }

        line = f" {args.model} IOS optimized with {args.opt_type} in {t2 - t1:.0f} secs: {graph_latency:.4f}\n"
        print(line)
        summary.append(line)

        with open(f"{expr_dir}/{name}.json", "w") as f:
            json.dump(graph.export_config(), f, indent=2)

    with open(f'{expr_dir}/latency.json', 'w') as f:
        json.dump(latency, f, indent=2)
    with open(f'{expr_dir}/summary.txt', 'w') as f:
        f.write('\n'.join(summary))
    with open(f'{expr_dir}/arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)



main()

