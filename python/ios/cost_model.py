from typing import List
import numpy as np
from ios.ir import Graph, Block, Node
from ios.contrib.ios_runtime import graph_latency, stage_latency, block_latency


class CostModel:
    def get_stage_latency(self, stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False) -> List[float]:
        raise NotImplementedError

    def get_block_latency(self, block: Block, batch_size, warmup, number, repeat, profile_stage=False) -> List[float]:
        raise NotImplementedError

    def get_graph_latency(self, graph: Graph, batch_size, warmup, number, repeat, profile_stage=False) -> List[float]:
        raise NotImplementedError


class RandomCostModel(CostModel):
    def get_graph_latency(self, graph: Graph, batch_size, warmup, number, repeat, profile_stage=False):
        return [float(np.random.rand()) for i in range(repeat)]

    def get_block_latency(self, block: Block, batch_size, warmup, number, repeat, profile_stage=False):
        return [float(np.random.rand()) for i in range(repeat)]

    def get_stage_latency(self, stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False):
        return [float(np.random.rand()) for i in range(repeat)]


class IOSCostModel(CostModel):
    def get_graph_latency(self, graph: Graph, batch_size, warmup, number, repeat, profile_stage=False):
        return graph_latency(graph, batch_size, warmup, number, repeat, profile_stage)

    def get_block_latency(self, block: Block, batch_size, warmup, number, repeat, profile_stage=False):
        return block_latency(block, batch_size, warmup, number, repeat, profile_stage)

    def get_stage_latency(self, stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False):
        return stage_latency(stage, batch_size, warmup, number, repeat, profile_stage)


