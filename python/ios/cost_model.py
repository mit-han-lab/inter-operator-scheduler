from typing import List
import numpy as np
from ios.ir import Graph, Block, Node
from ios.contrib.ios_runtime import graph_latency, stage_latency, block_latency


class CostModel:
    """
    Cost model is used to measure the latency of a stage, block, and computation graph. Cost model is used to guide the
    optimization of IOS.
    """
    def get_stage_latency(self, stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False) -> List[float]:
        raise NotImplementedError

    def get_block_latency(self, block: Block, batch_size, warmup, number, repeat, profile_stage=False) -> List[float]:
        raise NotImplementedError

    def get_graph_latency(self, graph: Graph, batch_size, warmup, number, repeat, profile_stage=False) -> List[float]:
        raise NotImplementedError


class RandomCostModel(CostModel):
    """
    Random cost model return random results, which can be used to debug and generate random schedules.
    """
    def get_graph_latency(self, graph: Graph, batch_size, warmup, number, repeat, profile_stage=False):
        return [float(np.random.rand()) for i in range(repeat)]

    def get_block_latency(self, block: Block, batch_size, warmup, number, repeat, profile_stage=False):
        return [float(np.random.rand()) for i in range(repeat)]

    def get_stage_latency(self, stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False):
        return [float(np.random.rand()) for i in range(repeat)]


class IOSCostModel(CostModel):
    """
    IOS runtime cost model, which measure the latency by directly executing the stage on the hardware.
    """
    def get_graph_latency(self, graph: Graph, batch_size, warmup, number, repeat, profile_stage=False):
        return graph_latency(graph, batch_size, warmup, number, repeat, profile_stage)

    def get_block_latency(self, block: Block, batch_size, warmup, number, repeat, profile_stage=False):
        return block_latency(block, batch_size, warmup, number, repeat, profile_stage)

    def get_stage_latency(self, stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False):
        return stage_latency(stage, batch_size, warmup, number, repeat, profile_stage)


