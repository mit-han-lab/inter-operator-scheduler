import torch
import torch.nn as nn
import torch.utils.data
from ios.ir import *


def get_layer(node: Node, name2layer) -> nn.Module:
    if isinstance(node, Conv):
        conv = nn.Conv2d(in_channels=node.input_shape[0],
                         out_channels=node.out_channels,
                         kernel_size=node.kernel,
                         groups=node.groups,
                         stride=node.stride,
                         padding=node.padding,
                         bias=True)
        if node.act == 'identity':
            act = nn.Sequential()
        elif node.act == 'relu':
            act = torch.nn.ReLU()
        elif node.act == 'tanh':
            act = torch.nn.Tanh()
        else:
            raise ValueError()
        layer = nn.Sequential(conv, act)
    elif isinstance(node, Pool):
        if node.pool_type == 'global_avg':
            layer = nn.AdaptiveAvgPool2d((1, 1))
        elif node.pool_type == 'avg':
            layer = nn.AvgPool2d(node.kernel, node.stride, node.padding)
        elif node.pool_type == 'max':
            layer = nn.MaxPool2d(node.kernel, node.stride, node.padding)
        else:
            raise NotImplemented
    elif isinstance(node, Identity):
        layer = nn.Sequential()
    elif isinstance(node, Relu):
        layer = nn.ReLU()
    elif isinstance(node, Activation):
        if node.act_type == 'relu':
            layer = nn.ReLU()
        elif node.act_type == 'sigmoid':
            layer = nn.Sigmoid()
        elif node.act_type == 'tanh':
            layer = nn.Tanh()
        else:
            raise ValueError
    elif isinstance(node, Element):
        layer = nn.Sequential()  # the op has been done in terms
    elif isinstance(node, Sequential):
        layers = []
        for snode in node.nodes:
            slayer = get_layer(snode, name2layer)
            name2layer[snode.name] = slayer
            layers.append(slayer)
        layer = nn.Sequential(*layers)
    else:
        raise NotImplementedError
    return layer


class Net(nn.Module):
    """
    Construct equivalent torch Module.
    """
    def __init__(self, graph: Graph):
        super(Net, self).__init__()
        name2layer = {}
        for node in graph.nodes():
            if node == graph.enter_node:
                continue
            name2layer[node.name] = get_layer(node, name2layer)
        self.graph = graph
        self.node2layer = nn.ModuleDict(name2layer)

    def forward(self, x):
        node2output = {}

        def sum_up(lst):
            if len(lst) == 1:
                return lst[0]
            else:
                return torch.sum(torch.stack(lst, dim=0), dim=0, keepdim=False)

        def prod_up(lst):
            if len(lst) == 1:
                return lst[0]
            else:
                return torch.prod(torch.stack(lst, dim=0), dim=0, keepdim=False)

        def narrow(v: torch.Tensor, axis, begin, length):
            if v.size(axis) == length and begin == 0:
                return v
            else:
                return torch.narrow(v, axis, begin, length)

        for node in self.graph.nodes():
            if node is self.graph.enter_node:
                node2output[node.name] = x
            else:
                term_vars = []
                for term in node.inputs:
                    value_vars = []
                    for value in term:
                        value_vars.append(narrow(node2output[value.node.name], 1, value.begin, value.length))
                    if isinstance(node, Element):
                        if node.op_type == 'add':
                            term_vars.append(sum_up(value_vars))
                        elif node.op_type == 'mul':
                            term_vars.append(prod_up(value_vars))
                        else:
                            raise ValueError
                    else:
                        term_vars.append(sum_up(value_vars))

                if len(term_vars) == 1:
                    input = term_vars[0]
                else:
                    input = torch.cat(term_vars, dim=1)
                layer = self.node2layer[node.name]
                output = layer(input)
                assert isinstance(output, torch.Tensor)
                node2output[node.name] = output

        x = node2output[self.graph.exit_node.name]
        return x

    def write_weights_to_torch(self):
        def ir2torch(node):
            if isinstance(node, Conv):
                conv = self.node2layer[node.name][0]
                conv.weight = nn.Parameter(torch.from_numpy(node.weight))
                conv.bias = nn.Parameter(torch.from_numpy(node.bias))
            elif isinstance(node, Sequential):
                for nd in node.nodes:
                    ir2torch(nd)

        for node in self.graph.nodes():
            ir2torch(node)

    def read_weights_from_torch(self):
        def torch2ir(node):
            if isinstance(node, Conv):
                conv = self.node2layer[node.name][0]
                node.weight = conv.weight.detach().cpu().numpy()
                node.bias = conv.bias.detach().cpu().numpy()
            elif isinstance(node, Sequential):
                for nd in node.nodes:
                    torch2ir(nd)

        for node in self.graph.nodes():
            torch2ir(node)

