from ir import Conv, Block, Graph, Node
from typing import List


class CostModel:
    def get_stage_latency(self, stage: List[List[Node]], batch_size, warmup, number, repeat, profile_stage=False) -> List[float]:
        raise NotImplementedError

    def get_block_latency(self, block: Block, batch_size, warmup, number, repeat, profile_stage=False) -> List[float]:
        raise NotImplementedError

    def get_graph_latency(self, graph: Graph, batch_size, warmup, number, repeat, profile_stage=False) -> List[float]:
        raise NotImplementedError


