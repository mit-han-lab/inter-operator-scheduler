from typing import List
from ir import Block, Graph, Node
import numpy as np
from cost_model import CostModel


class RandomCostModel(CostModel):
    def get_graph_latency(self, graph: Graph, batch_size, warmup, number, repeat, profile_stage=False):
        return [float(np.random.rand()) for i in range(repeat)]

    def get_block_latency(self, block: Block, batch_size, warmup, number, repeat, profile_stage=False):
        return [float(np.random.rand()) for i in range(repeat)]

    def get_stage_latency(self, stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False):
        return [float(np.random.rand()) for i in range(repeat)]

