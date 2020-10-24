from typing import List, Optional, Dict, Tuple, Set, FrozenSet, List
import functools
import operator
import numpy as np
from ir import Graph, Block, Conv, Value, Pool, Placeholder, Node, Identity, Sequential
from cost_model import CostModel
import logging
import itertools
from utils import iter_subset

logging.disable(logging.WARNING)

IGraph = Dict[int, List[int]]


def count_bits(s):
    cnt = 0
    while s > 0:
        s -= s & (-s)
        cnt += 1
    return cnt


def state2iset(s):
    iset = []
    i = 0
    while s > 0:
        if (s & 1) != 0:
            iset.append(i)
        s = s >> 1
        i += 1
    return iset


def get_input(ssnodes, block, nid, idn):
    # merge the inputs of ssnodes and drop duplicate inputs
    terms = list(itertools.chain(*[nd.inputs for nd in ssnodes]))
    iterms = [tuple((-1 if value.node is block.enter_node else nid[value.node], value.begin, value.end) for value in term) for term in terms]
    iterms = dict.fromkeys(sorted(iterms, key=lambda iterm: (len(iterm), iterm)))  # git rid of duplicates terms
    terms = [[Value(block.enter_node if ivalue[0] == -1 else idn[ivalue[0]], ivalue[1], ivalue[2]) for ivalue in iterm] for iterm in iterms]
    return terms


def log(info, verbose):
    if verbose:
        print(info)


def state2nset(s, idn):
    return [idn[i] for i in state2iset(s)]


def check_merge(s, idn) -> bool:  # check whether a set of nodes is valid to merge
    found = False
    in_size = None
    out_size = None
    stride = None
    act = None
    nds = [idn[i] for i in state2iset(s)]
    if len(nds) == 1:
        return True
    for nd in nds:
        if not isinstance(nd, Conv):  # current only merge conv
            return False
        if not found:
            in_size = nd.input_shape[1], nd.input_shape[2]
            out_size = nd.output_shape[1], nd.output_shape[2]
            stride = nd.stride[0], nd.stride[1]
            act = nd.act
            found = True
        else:
            # all input resolution, output resolution and stride must be the same
            if in_size[0] != nd.input_shape[1] or in_size[1] != nd.input_shape[2]:
                return False
            if out_size[0] != nd.output_shape[1] or out_size[1] != nd.output_shape[2]:
                return False
            if stride[0] != nd.stride[0] or stride[1] != nd.stride[1]:
                return False
            if nd.groups != 1 or act != nd.act:
                return False
            if len(nd.inputs) > 1 or len(nd.inputs[0]) > 1 or not (nd.inputs[0][0] == nds[0].inputs[0][0]):
                return False
    return True


def check_parallel(ss, successor_dict, max_num_streams):
    iset = state2iset(ss)
    if len(iset) > max_num_streams:  # stream number requirement
        return False
    suc_list = [successor_dict[u] for u in iset]
    num = len(iset)
    for i in range(num):
        for j in range(i + 1, num):
            if not suc_list[i].isdisjoint(suc_list[j]):  # successors keep disjoint
                return False
    return True


def latency(stage: Tuple[List[List[int]], str], block, merge_latency, parallel_latency, cost_model, idn, nid, batch_size, warmup, number, repeat) -> float:
    stage_seqs, qtype = stage
    ss = sum(1 << u for u in itertools.chain(*stage_seqs))
    if qtype == 'merge':
        if ss in merge_latency:
            return merge_latency[ss]
        snodes = state2nset(ss, idn)
        if len(stage_seqs) == 1:
            assert len(snodes) == 1
            merge_latency[ss] = float(np.mean(cost_model.get_stage_latency([[snodes[0]]], batch_size, warmup, number, repeat)))
        else:
            convs = [nd for nd in snodes if isinstance(nd, Conv)]
            assert len(convs) == len(snodes)
            terms = get_input(convs, block, nid, idn)
            out_channels = sum(nd.out_channels for nd in convs)
            kernel = (max(nd.kernel[0] for nd in convs), max(nd.kernel[1] for nd in convs))
            stride = (convs[0].stride[0], convs[0].stride[1])
            padding = (max(nd.padding[0] for nd in convs), max(nd.padding[1] for nd in convs))
            groups = convs[0].groups
            act = convs[0].act
            conv = Conv('c', '', inputs=terms, out_channels=out_channels, kernel=kernel, stride=stride, padding=padding, groups=groups, act=act, output_shape=None)
            merge_latency[ss] = float(np.mean(cost_model.get_stage_latency([[conv]], batch_size, warmup, number, repeat)))
        return merge_latency[ss]
    elif qtype == 'parallel':
        if ss in parallel_latency:
            return parallel_latency[ss]
        stage_seqs_nodes = []
        for seq in stage_seqs:
            seq_nodes = []
            for uid in seq:
                seq_nodes.append(idn[uid])
            stage_seqs_nodes.append(seq_nodes)
        parallel_latency[ss] = float(np.mean(cost_model.get_stage_latency(stage_seqs_nodes, batch_size, warmup, number, repeat)))
        return parallel_latency[ss]
    else:
        raise ValueError


def build_graph(all_nodes: List[Node], nid):
    g: Dict[int, List[int]] = {}
    for nu in all_nodes:
        iu = nid[nu]
        g[iu] = []
        for use in nu.uses:
            nv = use[0]
            if nv in all_nodes:
                iv = nid[nv]
                g[iu].append(iv)
        g[iu] = list(set(g[iu]))  # dump duplicate targets
    return g


def topological_order(graph: IGraph) -> List[int]:
    in_degree = {u: 0 for u in graph.keys()}
    for u in graph.keys():
        for v in graph[u]:
            in_degree[v] += 1
    qu = [u for u in graph.keys() if in_degree[u] == 0]
    order = []
    while len(qu) > 0:
        u = qu.pop()
        order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                qu.append(v)
    assert len(order) == len(graph)  # no loop
    return order


def graph_transitive_closure(graph: IGraph, include_self=False) -> Dict[int, Set[int]]:
    tc: Dict[int, Set[int]] = {u: set() for u in graph.keys()}
    for u in reversed(topological_order(graph)):
        if include_self:
            tc[u].update([u])
        for v in graph[u]:
            tc[u].update(tc[v])
            tc[u].update([v])
    return tc


def transitive_closure_to_graph(tc):
    graph: IGraph = {}
    for u in tc:
        graph[u] = list(tc[u])
    return graph


def longest_chain(graph: IGraph) -> List[int]:
    depth: Dict[int, int] = {u: 1 for u in graph}
    comes: Dict[int, int] = {u: None for u in graph}
    for u in reversed(topological_order(graph)):
        for v in graph[u]:
            if depth[u] < depth[v] + 1:
                depth[u] = depth[v] + 1
                comes[u] = v
    u = max(depth.keys(), key=lambda u: depth[u])
    chain = []
    while u is not None:
        chain.append(u)
        u = comes[u]
    return chain


def sub_graph(graph: IGraph, uset) -> IGraph:
    new_graph = {}
    for u in graph:
        if u in uset:
            new_graph[u] = [v for v in graph[u] if v in uset]
    return new_graph


def graph_chain_decomposition(graph: IGraph) -> List[List[int]]:
    chains = []
    graph = transitive_closure_to_graph(graph_transitive_closure(graph))

    while len(graph) > 0:
        chain = longest_chain(graph)
        chains.append(chain)
        graph = sub_graph(graph, set(graph.keys()) - set(chain))
    return chains


def ending_iterator(
        s: int,
        chains: List[List[int]],
        nid: Dict[Node, int],
        idn: Dict[int, Node],
        max_seq_length: int,
        max_num_streams: int):
    iset = state2iset(s)
    begins = []
    ends = []
    lengths = []

    # get the range for each chain
    for ichain, chain in enumerate(chains):
        end = 0
        for iu, u in enumerate(chain):
            if u in iset:
                end = iu + 1
            else:
                break
        begin = max(0, end - max_seq_length)
        begins.append(begin)
        ends.append(end)
        lengths.append(end - begin)
    bases = [length + 1 for length in lengths]
    strides = list(itertools.accumulate(bases, operator.mul))
    total = strides[-1]

    # build sub graph and transitive clousure
    tc = graph_transitive_closure(build_graph(state2nset(s, idn), nid), include_self=True)

    # enuermate ending
    for w in range(total):
        end_list = []
        for i, chain in enumerate(chains):
            div = strides[i-1] if i >= 1 else 1
            idx = (w % strides[i]) // div
            if idx == lengths[i]:  # empty
                continue
            end_list.append(chain[begins[i] + idx])
        if len(end_list) == 0:
            continue
        if len(end_list) > max_num_streams:
            continue
        isdisjoint = True
        for i in range(len(end_list)):
            for j in range(i+1, len(end_list)):
                if not tc[end_list[i]].isdisjoint(tc[end_list[j]]):
                    isdisjoint = False
                    break
            if not isdisjoint:
                break
        if isdisjoint:
            groups = [sorted(tc[u]) for u in end_list]
            if any(len(group) > max_seq_length for group in groups):
                continue
            yield groups


def dop(s: int,
        block, chains, on_debug, debug_dp_info, idn, nid, dp, ep, opt_type, max_seq_length, max_num_streams, merge_latency, parallel_latency, cost_model, batch_size, warmup, number, repeat) -> float:
    if s == 0:
        return 0.0
    if s in dp:
        return dp[s]
    if on_debug:
        debug_dp_info['#states'][-1] += 1
        debug_dp_info['meta'][-1][s] = 0

    iset = state2iset(s)
    # print(iset)
    successor_dict: Dict[int, Set] = {u: set() for u in iset}
    for u in reversed(iset):
        successors = successor_dict[u]
        successors.add(u)
        for use in idn[u].uses:
            if use[0] in nid and nid[use[0]] in iset:
                successors.update(successor_dict[nid[use[0]]])

    dpv = 1e19
    s1 = sum(1 << u for u in iset if len(successor_dict[u]) == 1)
    s2 = sum(1 << u for u in iset if len(successor_dict[u]) <= max_seq_length)
    if "merge" in opt_type:
        for ss in iter_subset(s1):
            if check_merge(ss, idn):
                stage = [[u] for u in state2iset(ss)], 'merge'
                val1 = dop(s - ss, block, chains, on_debug, debug_dp_info, idn, nid, dp, ep, opt_type, max_seq_length, max_num_streams, merge_latency, parallel_latency, cost_model, batch_size, warmup, number, repeat)
                val2 = latency(stage, block, merge_latency, parallel_latency, cost_model, idn, nid, batch_size, warmup, number, repeat)
                val = val1 + val2
                if on_debug:
                    debug_dp_info['#transitions'][-1] += 1
                    debug_dp_info['meta'][-1][s] += debug_dp_info['meta'][-1][s - ss]
                if val < dpv:
                    dpv = val
                    ep[s] = stage
    # if "parallel" in opt_type:
    #     for ss in iter_subset(s2):
    #         if check_parallel(ss, successor_dict, max_num_streams):
    #             stage = [list(sorted(list(successor_dict[u]))) for u in state2iset(ss)], 'parallel'
    #             consumed = sum(1 << u for u in itertools.chain(*stage[0]))
    #             val1 = dop(s - consumed, block, chains, on_debug, debug_dp_info, idn, nid, dp, ep, opt_type, max_seq_length, max_num_streams, merge_latency, parallel_latency, cost_model, batch_size, warmup, number, repeat)
    #             val2 = latency(stage, block, merge_latency, parallel_latency, cost_model, idn, nid, batch_size, warmup, number, repeat)
    #             val = val1 + val2
    #             if on_debug:
    #                 debug_dp_info['#transitions'][-1] += 1
    #                 debug_dp_info['meta'][-1][s] += debug_dp_info['meta'][-1][s - consumed]
    #                 debug_dp_info['width'][-1] = max(debug_dp_info['width'][-1], len(stage[0]))
    #             if val < dpv:
    #                 dpv = val
    #                 ep[s] = stage
    if "parallel" in opt_type:
        for groups in ending_iterator(s, chains, nid, idn, max_seq_length, max_num_streams):
            stage = groups, 'parallel'
            consumed = sum(1 << u for u in itertools.chain(*stage[0]))
            val1 = dop(s - consumed, block, chains, on_debug, debug_dp_info, idn, nid, dp, ep, opt_type, max_seq_length, max_num_streams, merge_latency, parallel_latency, cost_model, batch_size, warmup, number, repeat)
            val2 = latency(stage, block, merge_latency, parallel_latency, cost_model, idn, nid, batch_size, warmup, number, repeat)
            val = val1 + val2
            if on_debug:
                debug_dp_info['#transitions'][-1] += 1
                debug_dp_info['meta'][-1][s] += debug_dp_info['meta'][-1][s - consumed]
                debug_dp_info['width'][-1] = max(debug_dp_info['width'][-1], len(stage[0]))
            if val < dpv:
                dpv = val
                ep[s] = stage
    dp[s] = dpv
    return dpv


def get_stage_list(ep, s):
    stage_list = []
    while s != 0:
        stage = ep[s]
        stage_list.append(stage)
        s = s - sum(1 << u for seq in ep[s][0] for u in seq)
    stage_list = list(reversed(stage_list))
    return stage_list


def optimize(graph: Graph,
             batch_size,
             cost_model: Optional[CostModel],
             opt_type: str,
             warmup, number, repeat,
             max_num_streams, max_part_size, compute_weight,  max_seq_length=3,
             debug_dp_info=None, verbose=True) -> Graph:
    graph_enter = Placeholder(graph.input.name, graph.input.hint_name, graph.input.output_shape)
    graph_enter.output_shape = graph.enter_node.output_shape
    blocks = []
    on_debug = debug_dp_info is not None
    if on_debug:
        assert isinstance(debug_dp_info, dict)
        debug_dp_info['dp'] = {}
        debug_dp_info['stage_latency'] = {}
        debug_dp_info['#states'] = []
        debug_dp_info['#transitions'] = []
        debug_dp_info['#schedules'] = []
        debug_dp_info['width'] = []
        debug_dp_info['#operators'] = []
        debug_dp_info['meta'] = []
    log(f"optimize {opt_type} on {graph.name}", verbose)
    assert 'parallel' in opt_type or 'merge' in opt_type

    for bindex, block in enumerate(graph.blocks):
        all_nodes = block.inner_nodes + [block.exit_node]
        node_parts = block.parts

        nid: Dict[Node, int] = {node: i for i, node in enumerate(all_nodes)}
        idn: Dict[int, Node] = {i: node for i, node in enumerate(all_nodes)}

        if node_parts is None:
            node_parts = []
            for idx_part in range((len(all_nodes) + max_part_size - 1) // max_part_size):
                begin = idx_part * max_part_size
                end = min((idx_part + 1) * max_part_size, len(all_nodes))
                node_parts.append([all_nodes[i] for i in range(begin, end)])
        log(f"block {bindex} with {len(all_nodes)} nodes {len(node_parts)} parts", verbose)

        stage_list = []

        for part_index, npart in enumerate(node_parts):
            log(f"part {part_index} with {len(npart)} nodes", verbose)
            ipart = [nid[nd] for nd in npart]

            dp: Dict[int, float] = {}
            ep: Dict[int, Tuple[List[int], str]] = {}
            merge_latency: Dict[int, float] = {}
            parallel_latency: Dict[int, float] = {}
            part_graph = build_graph(npart, nid)
            chains = graph_chain_decomposition(part_graph)

            max_num_endings = functools.reduce(operator.mul, [len(chain)+1 for chain in chains])
            if verbose:
                print(f"#Chains: {len(chains)}")
                print(f"Max number of endings: {max_num_endings}")

            if on_debug:
                debug_dp_info['#states'].append(0)
                debug_dp_info['#transitions'].append(0)
                debug_dp_info['#schedules'].append(0)
                debug_dp_info['width'].append(0)
                debug_dp_info['#operators'].append(len(npart))
                debug_dp_info['meta'].append({0: 1})

            ustate = sum(1 << i for i in ipart)
            dop(ustate, block, chains, on_debug, debug_dp_info, idn, nid, dp, ep, opt_type, max_seq_length, max_num_streams, merge_latency, parallel_latency, cost_model, batch_size, warmup, number, repeat)
            stage_list.extend(get_stage_list(ep, ustate))

            if on_debug:
                for ss in parallel_latency:
                    stage_name = ' '.join(['parallel'] + [nd.name for nd in state2nset(ss, idn)])
                    debug_dp_info['stage_latency'][stage_name] = parallel_latency[ss]
                for ss in merge_latency:
                    stage_name = ' '.join(['merge'] + [nd.name for nd in state2nset(ss, idn)])
                    debug_dp_info['stage_latency'][stage_name] = merge_latency[ss]
                for s in dp:
                    state_name = ' '.join(nd.name for nd in state2nset(s, idn))
                    stage_name = ' '.join(nd.name for nd in state2nset(sum(1 << u for u in itertools.chain(*ep[s][0])), idn))
                    debug_dp_info['dp'][state_name] = f'{dp[s]:.3f} ({ep[s][1]} {stage_name})'
                debug_dp_info['#schedules'][-1] = debug_dp_info['meta'][-1][ustate]

        new_block = construct(stage_list, block, blocks, graph_enter, idn, nid, compute_weight)
        blocks.append(new_block)

    new_graph = Graph(graph.name + "_" + opt_type, graph_enter, blocks)
    new_graph.infer_shape()
    return new_graph


def construct(stage_list: List[Tuple[List[List[int]], str]], block, constructed_blocks, graph_enter, idn, nid, compute_weight) -> Block:
    inner_nodes = []
    stages = []
    if len(constructed_blocks) == 0:
        new_enter_node = graph_enter
    else:
        new_enter_node = constructed_blocks[-1].exit_node
    out_dict = {block.enter_node: (new_enter_node, 0, new_enter_node.output_shape[0])}

    def merge_inputs(inputs: List[List[Value]]):
        merge_inputs_flag = True
        if merge_inputs_flag:
            while True:  # merge input
                merged = False
                for i in range(1, len(inputs)):
                    if len(inputs[i - 1]) > 1 or len(inputs[i]) > 1:
                        continue
                    va, vb = inputs[i - 1][0], inputs[i][0]
                    if va.node == vb.node and va.end == vb.begin:
                        vc = Value(va.node, va.begin, vb.end)
                        inputs = inputs[:i - 1] + [[vc]] + inputs[i + 1:]
                        merged = True
                        break
                if not merged:
                    break
        return inputs

    def get_new_terms(terms, new_node, do_sort=True):
        nterms = []
        for ti, term in enumerate(terms):
            nterm = []
            for vi, value in enumerate(term):
                nv = out_dict[value.node]
                nterm.append(Value(nv[0], nv[1] + value.begin, nv[1] + value.end))
            nterms.append(nterm)
        if do_sort:
            nterms = sorted(nterms, key=lambda nterm: (len(nterm), nterm[0].node.name))  # git rid of duplicates terms
        for ti, term in enumerate(nterms):
            for vi, value in enumerate(term):
                value.node.uses.append((new_node, ti, vi))
        return nterms

    def copy_weights(dst_node, src_node):
        if isinstance(dst_node, Conv):
            assert isinstance(src_node, Conv)
            dst_node.weight = src_node.weight.copy()
            dst_node.bias = src_node.bias.copy()

    for stage_seqs, qtype in stage_list:
        if qtype == 'merge' and len(stage_seqs) > 1:  # only merge convolutions
            inodes = list(itertools.chain(*stage_seqs))
            snodes = [nd for nd in [idn[i] for i in inodes] if isinstance(nd, Conv)]
            assert len(snodes) == len(inodes)
            #  get the parameters of merged conv
            out_channels = sum(nd.out_channels for nd in snodes)
            kernel = (max(nd.kernel[0] for nd in snodes), max(nd.kernel[1] for nd in snodes))
            stride = snodes[0].stride[0], snodes[0].stride[1]
            padding = (max(nd.padding[0] for nd in snodes), max(nd.padding[1] for nd in snodes))
            groups = 1
            #  construct merged conv
            terms = get_input(snodes, block, nid, idn)
            new_node = Conv(snodes[0].name, " ".join(nd.hint_name for nd in snodes), None, out_channels, kernel, stride, padding, groups, snodes[0].act, None)
            new_node.inputs = get_new_terms(terms, new_node)
            new_node.infer_shape()
            if compute_weight:
                new_node.weight = np.zeros(shape=new_node.weight_shape, dtype=np.float32)
                new_node.bias = np.zeros(shape=new_node.bias_shape, dtype=np.float32)
            #  set weights and out_dict
            out_begin = 0
            for node in snodes:
                for term in node.inputs:
                    in_begin = 0
                    for ti, t in enumerate(terms):
                        found = True
                        if len(term) != len(t):
                            found = False
                        else:
                            for va, vb in zip(term, t):
                                if not (va.node == vb.node and va.begin == vb.begin and va.end == vb.end):
                                    found = False
                                    break
                        if found:
                            break
                        in_begin += t[0].length
                    in_end = in_begin + term[0].length
                    out_end = out_begin + node.out_channels
                    kernel_begin = (kernel[0] - node.kernel[0]) // 2, (kernel[1] - node.kernel[1]) // 2
                    kernel_end = kernel_begin[0] + node.kernel[0], kernel_begin[1] + node.kernel[1]
                    if compute_weight:
                        new_node.weight[out_begin:out_end, in_begin:in_end, kernel_begin[0]:kernel_end[0],
                        kernel_begin[1]:kernel_end[1]] = node.weight.copy()
                        new_node.bias[out_begin:out_end] = node.bias.copy()
                out_dict[node] = (new_node, out_begin, out_begin + node.out_channels)
                out_begin += node.out_channels
            new_node.inputs = merge_inputs(new_node.inputs)
            new_node.infer_shape()
            inner_nodes.append(new_node)
            stages.append([[new_node.name]])
        else:
            seq_in_stage = []
            for seq in stage_seqs:
                inodes = seq
                snodes = [idn[i] for i in inodes]
                new_nodes = []
                for snode in snodes:
                    snode_config = snode.export_config()
                    if isinstance(snode, Sequential):
                        snode_config["nodes"][0]["inputs"] = []
                        new_node = Node.from_config(snode_config, {})
                        new_node.nodes[0].inputs = merge_inputs(get_new_terms(snode.nodes[0].inputs, new_node, do_sort=False))
                        new_node.inputs = new_node.nodes[0].inputs
                        if compute_weight:
                            for dst_nd, src_nd in zip(new_node.nodes, snode.nodes):
                                copy_weights(dst_nd, src_nd)
                        new_node.infer_shape()
                        new_nodes.append(new_node)
                        out_dict[snode] = (new_node, 0, new_node.output_shape[0])
                    else:
                        snode_config["inputs"] = []
                        new_node = Node.from_config(snode_config, {})
                        new_node.inputs = merge_inputs(get_new_terms(snode.inputs, new_node, do_sort=False))
                        if compute_weight:
                            copy_weights(new_node, snode)
                        new_node.infer_shape()
                        new_nodes.append(new_node)
                        out_dict[snode] = (new_node, 0, new_node.output_shape[0])
                inner_nodes.extend(new_nodes)
                seq_in_stage.append([new_node.name for new_node in new_nodes])
            stages.append(seq_in_stage)
    new_exit_node = inner_nodes.pop()
    return Block(new_enter_node, new_exit_node, inner_nodes, stages)
