from typing import List
from .cost_model import CostModel
from ir import Graph, Block, Node
from contrib.ios_backend import graph_latency, stage_latency, block_latency


class IOSCostModel(CostModel):
    def get_graph_latency(self, graph: Graph, batch_size, warmup, number, repeat, profile_stage=False):
        return graph_latency(graph, batch_size, warmup, number, repeat, profile_stage)

    def get_block_latency(self, block: Block, batch_size, warmup, number, repeat, profile_stage=False):
        return block_latency(block, batch_size, warmup, number, repeat, profile_stage)

    def get_stage_latency(self, stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False):
        return stage_latency(stage, batch_size, warmup, number, repeat, profile_stage)


