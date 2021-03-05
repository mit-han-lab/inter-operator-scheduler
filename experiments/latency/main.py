import json
import time
import argparse
import numpy as np
import os
import logging
from ios.models import inception_v3, randwire_large, nasnet_large, squeezenet
from ios.visualizer import draw, draw_block

logging.disable(logging.WARNING)

argparser = argparse.ArgumentParser()
argparser.add_argument('--ename', type=str, required=True)
argparser.add_argument('--index', type=int, required=True)
argparser.add_argument('--device', type=str, required=True,
                       choices=['1080ti', '1080', 'v100', 'titanxp', '2080ti', 'cpu', 'k80'])
argparser.add_argument('--model', type=str, required=True,
                       choices=['inception_v3', 'randwire', 'nasnet', 'squeezenet'])
argparser.add_argument('--bs', type=int, required=False, default=1)
argparser.add_argument('--opt_type', type=str, required=True,
                       choices=['trt', 'tf', 'tf_xla', 'tvm_cudnn', 'taso', 'taso_seq', 'tvm_tune', 'sequential', 'greedy', 'dp_parallel_merge', 'dp_parallel', 'dp_merge'])
argparser.add_argument('--r', type=int, default=3)
argparser.add_argument('--s', type=int, default=8)
argparser.add_argument('--warmup', type=int, required=False, default=2)
argparser.add_argument('--number', type=int, required=False, default=6)
argparser.add_argument('--repeat', type=int, required=False, default=6)

args = argparser.parse_args()
expr_dir = f'./outputs/{args.ename}/{args.device}-{args.model}-bs{args.bs}-{args.opt_type}-idx{args.index}-r{args.r}-s{args.s}-{args.warmup}-{args.number}-{args.repeat}'
os.makedirs("./outputs", exist_ok=True)
os.makedirs(f"./outputs/{args.ename}", exist_ok=True)
os.makedirs(expr_dir, exist_ok=True)

name2model = {
    'inception_v3': inception_v3,
    'randwire': randwire_large,
    'nasnet': nasnet_large,
    'squeezenet': squeezenet
}


def summary_str(opt_cost, mean_latency):
    def min_width(s, w):
        return s + ' ' * max((w - len(s)), 0)

    opt_dict = {
        'trt': 'TensorRT',
        'tf': 'Tensorflow',
        'tf_xla': 'Tensorflow-XLA',
        'tvm_cudnn': 'TVM-cuDNN',
        'taso': 'TASO',
        'taso_seq': 'TASO-SEQ',
        'tvm_tune': 'TVM-AutoTune',
        'sequential': 'Sequential',
        'greedy': 'Greedy',
        'dp_parallel_merge': 'IOS-Both',
        'dp_parallel': 'IOS-Parallel',
        'dp_merge': 'IOS-Merge'
    }

    opt_name = opt_dict[args.opt_type]
    if opt_name.startswith('IOS') and args.ename=='prune':
        opt_name = min_width(f'{opt_name}(r={args.r}, s={args.s})', 20)
    else:
        opt_name = min_width(f'{opt_name}', 15)

    items_dict = {
        'Model': min_width(args.model, 12),
        'Optimization': opt_name,
        'Batchsize': f'{args.bs:<3}',
        'Optimization cost': min_width(f'{opt_cost:.0f} sec', 8),
        'Latency': min_width(f'{mean_latency:.2f} ms', 10)
    }
    return " | ".join([f"{a}: {b}" for a, b in items_dict.items()])


def main():
    logs = {}
    graph = name2model[args.model]()

    if args.opt_type == 'trt':
        from ios.contrib import trt_runtime
        t1 = time.time()
        graph_latency = trt_runtime.graph_latency(graph, args.bs, args.warmup, args.number, args.repeat)
        t2 = time.time()
        name = f'{graph.name}_{args.opt_type}'
        logs[name] = {}
        logs[name]['latency'] = graph_latency
        logs[name]['opt_time'] = t2 - t1
        summary = summary_str(opt_cost=t2 - t1, mean_latency=np.mean(graph_latency))
    elif args.opt_type in ['tf', 'tf_xla']:
        import ios.tf_utils
        t1 = time.time()
        graph_latency = ios.tf_utils.graph_latency(graph, args.bs, warmup=args.warmup, number=args.number, repeat=args.repeat, xla=('xla' in args.opt_type))
        t2 = time.time()
        name = f'{graph.name}_{args.opt_type}'
        logs[name] = {}
        logs[name]['latency'] = graph_latency
        logs[name]['opt_time'] = t2 - t1
        summary = summary_str(opt_cost=t2 - t1, mean_latency=np.mean(graph_latency))
    elif args.opt_type in ['taso_seq', 'taso']:
        import ios.taso_utils
        opt_type = args.opt_type
        t1 = time.time()
        budget = 1000
        if args.model == 'nasnet':
            budget = 8  # the budget of nasnet is set to 8 to avoid out of memory.
        graph_latency = ios.taso_utils.graph_latency(graph, batchsize=args.bs, warmup=args.warmup, number=args.number, repeat=args.repeat, optimize=(opt_type == 'taso'), budget=budget)
        t2 = time.time()
        name = f'{graph.name}_{args.opt_type}'
        logs[name] = {}
        logs[name]['latency'] = graph_latency
        logs[name]['opt_time'] = t2 - t1
        summary = summary_str(opt_cost=t2 - t1, mean_latency=np.mean(graph_latency))
    elif args.opt_type in ['tvm_cudnn', 'tvm_tune']:
        import ios.tvm_utils
        if args.opt_type == 'tvm_tune':
            target = 'cuda'
            tune = True
        else:
            target = 'cuda -libs=cudnn'
            tune = False
        name = f'{graph.name}_{args.opt_type}'
        t1 = time.time()
        graph_latency = ios.tvm_utils.graph_latency_local(graph, args.bs, number=args.number, repeat=args.repeat, target=target, tune=tune)
        t2 = time.time()
        logs[name] = {}
        logs[name]['latency'] = graph_latency
        logs[name]['opt_time'] = t2 - t1
        summary = summary_str(opt_cost=t2 - t1, mean_latency=np.mean(graph_latency))
    else:
        from ios.cost_model import IOSCostModel
        from ios.optimizer import optimize
        cost_model = IOSCostModel()
        t1 = time.time()
        if args.opt_type == 'sequential':
            graph.sequential_schedule()
        elif args.opt_type == 'greedy':
            graph.greedy_schedule()
        else:
            graph = optimize(graph, args.bs, cost_model, args.opt_type, args.warmup, args.number, args.repeat,
                             max_num_groups=args.s,
                             max_part_size=50,
                             max_group_size=args.r,
                             compute_weight=False,
                             verbose=False)
        t2 = time.time()
        name = graph.name
        graph_latency = cost_model.get_graph_latency(graph, args.bs, warmup=args.warmup, number=args.number, repeat=args.repeat)
        block_latency = [np.mean(cost_model.get_block_latency(block, args.bs, args.warmup, args.number, args.repeat)) for block in graph.blocks]
        logs[name] = {}
        logs[name]['latency'] = graph_latency
        logs[name]['opt_time'] = t2 - t1
        logs[name]['mean'] = float(np.mean(graph_latency))
        logs[name]['std'] = float(np.std(graph_latency))
        logs[name]['block_latency'] = block_latency
        summary = summary_str(opt_cost=t2 - t1, mean_latency=np.mean(graph_latency))
        for bindex, block in enumerate(graph.blocks):
            block_dir = f'{expr_dir}/{name}_blocks'
            os.makedirs(block_dir, exist_ok=True)
            draw_block(block, f'{block_dir}/{bindex}.png', f'{name} block {bindex}, latency {block_latency[bindex]:.3f}')
        draw(graph, f"{expr_dir}/{name}.png", label=f'{name}, latency {float(np.mean(graph_latency)):.3f}')
        with open(f"{expr_dir}/{name}.json", "w") as f:
            json.dump(graph.export_config(), f, indent=2)

    print(summary)
    with open(f'{expr_dir}/summary.txt', 'w') as f:
        f.write(summary + "\n")
    with open(f'{expr_dir}/latency.json', 'w') as f:
        json.dump(logs, f, indent=2)
    with open(f'{expr_dir}/arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


main()
