"""
This module defines the computation graph used in IOS. Any neural network is represented by a computation graph in IOS.
A computation graph is represented in a instance of Graph class. There are two ways to construct a computation graph:
    1. Use the interfaces provided in ios.ir.common, which can also be used in ios.ir namespace.
    2. Construct from a config file via ios.ir.Graph.from_config(config), where config is exported previously by
       ios.ir.Graph.export_config().

Note:
A Graph contains a list of Block. A Block contains a set of operators. The dependency relation of the operators in a
Block is a directed acyclic graph (DAG). All operator classes share the same base class Node. Each operator is
defined as a separate class, such as Conv, Pool, and Sequential. Sequential is a compound operator, which contains a
sequence of operators. Other operators are normal operators.
"""

from typing import List, Iterable, Sequence, Tuple, Dict, Optional
import numpy as np


class Value:
    """
    Represent a tensor value by a triple (node, begin, end), where node is an operator, begin and end specify the
    channels. In other words, (node, begin, end) represents a slide [begin, end) of the node's output on channel
    dimension (dimension with index 0).

    Parameters
    ----------
    node: ios.ir.Node
        The operator whose output tensor value is referred by Value.

    begin: int
        The slice begin in the channel dimension.

    end: int
        The slice end in the channel dimension.
    """
    __slots__ = ['node', 'begin', 'end']

    def __init__(self, node, begin, end):
        self.node: Node = node
        self.begin = begin
        self.end = end

    @property
    def length(self):
        """
        Return the number of channels in the tensor value

        :return: the number of channels.
        """
        return self.end - self.begin

    @staticmethod
    def from_config(config, name2nodes):
        """
        Construct the value from a value config.

        :param config: the config exported by Value.export_config(...).
        :param name2nodes: a dict that maps the name to constructed operators, which may be referred in the Value by its
            name.
        :return: the new constructed Value.
        """
        return Value(name2nodes[config[0]], config[1], config[2])

    def export_config(self):
        """
        Export the config of value.

        :return: the config of value, which contains the triple (node name, begin, end).
        """
        return [self.node.name, self.begin, self.end]

    def __eq__(self, other):
        if not isinstance(other, Value):
            return False
        return self.node == other.node and self.begin == other.begin and self.end == other.end

    def __repr__(self):
        return f'({self.node.hint_name}:{self.begin}:{self.end})'

    def __str__(self):
        return self.__repr__()


class Node:
    """
    Node is the base class of all operators.

    Parameters
    ----------
    type: str
        The type string of the operator.

    name: str
        The name of the operator, which is a identifier of the operator in the computation graph.

    hint_name: str
        The hint name of the operator.

    inputs: Optional[Sequence[Sequence[Value]]]
        Represents the inputs of the operator. The inputs contains a list of terms. A term contains a list of Values.
        Each term sums up the values first, and then the inputs concat the terms along with the channel dimension. The
        result of concatenation is used as the input of the operator.

    output_shape: None or Tuple[int, int, int]
        The output shape of the operator. It can be None because this can be inferred by the input shapes.
    """
    __slots__ = ['type', 'name', 'hint_name', 'inputs', 'output_shape', 'uses']

    def __init__(self, type, name, hint_name, inputs, output_shape):
        self.type = type
        self.name = name
        self.hint_name = hint_name
        self.output_shape: Tuple[int, int, int] = output_shape
        self.inputs: Sequence[Sequence[Value]] = inputs
        self.uses: List[Tuple[Node, int, int]] = []

    @staticmethod
    def from_config(config, name2node):
        """
        Construct an operator based on config.

        :param config: the config used to construct the new operator
        :param name2node: a dict that maps the name to constructed operator, which may be referred in the new
            constructed operator
        :return: the new constructed operator
        """
        if config['type'] == 'placeholder':
            return Placeholder.from_config(config, name2node)
        elif config['type'] == 'conv':
            return Conv.from_config(config, name2node)
        elif config['type'] == 'pool':
            return Pool.from_config(config, name2node)
        elif config['type'] == 'identity':
            return Identity.from_config(config, name2node)
        elif config['type'] == 'element':
            return Element.from_config(config, name2node)
        elif config['type'] == 'relu':
            return Relu.from_config(config, name2node)
        elif config['type'] == 'activation':
            return Activation.from_config(config, name2node)
        elif config['type'] == 'sequential':
            return Sequential.from_config(config, name2node)
        else:
            raise ValueError(f"unrecognized node type {config['type']}")

    def export_config(self):
        """
        Export the config of the operator.
        """
        raise NotImplementedError

    def infer_shape(self):
        """
        Infer the output shape of the operator based on the shape of inputs
        """
        raise NotImplementedError

    def flops(self):
        """
        Return the number of floating point operations in the operator.
        """
        raise NotImplementedError

    def memory_access(self):
        """
        Return the number of memory access operations.
        """
        raise NotImplementedError

    def kernels(self):
        """
        Return the number of kernel calls in the operator. It may be greater than 1, because there are compound
        operator such as Sequential.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.name

    def input_flops(self):
        """
        Calculate the flops in the input part of the operator.
        """
        ret = 0
        for term in self.inputs:
            term_size = term[0].node.output_shape[1] * term[0].node.output_shape[2] * term[0].length
            ret += len(term) * term_size
        return ret

    def input_memory_access(self):
        """
        Calculate the number of memory access operations in the input part of the operator
        """
        ret = 0
        term_sizes = []
        for term in self.inputs:
            term_size = term[0].node.output_shape[1] * term[0].node.output_shape[2] * term[0].length
            if len(term) > 1:
                ret += term_size * len(term)
            term_sizes.append(term_size)
        if len(self.inputs) > 1:
            ret += sum(term_sizes)
        return ret

    def input_kernels(self):
        """
        Calculate the number of kernel calls in the input part
        """
        ret = 0
        for term in self.inputs:
            if len(term) > 1:
                ret += 1
        if len(self.inputs) > 1:
            ret += 1
        return ret

    def input_readable_str(self, merge_op='+'):
        """
        Return a readable string represents the inputs :param merge_op: the merge operation of terms, can be '+' (for
        all operators except Element with 'mul' op_type) or '*' for Element operator with 'mul' op_type. :return:
        """
        terms_str = []
        for term in self.inputs:
            values_str = []
            for value in term:
                values_str.append(value.node.hint_name)
            terms_str.append(merge_op.join(values_str))
        if len(terms_str) == 1:
            if len(self.inputs[0]) > 1:
                return '(' + terms_str[0] + ')'
            else:
                return terms_str[0]
        else:
            return "Concat(" + ",".join(terms_str) + ")"

    def readable_lines(self, indent) -> List[str]:
        """
        Return a list of lines to represent the operator.

        :param indent: the indent for inner operators for compound operator.
        """
        raise NotImplementedError


class Placeholder(Node):
    """
    The placeholder that represents the input of the computation graph.
    """
    def __init__(self, name, hint_name, output_shape):
        super().__init__("placeholder", name, hint_name, [[]], output_shape)

    @staticmethod
    def from_config(config, name2node):
        return Placeholder(config['name'], config['hint_name'], output_shape=config['output_shape'])

    def export_config(self):
        return {
            'type': 'placeholder',
            'name': self.name,
            'hint_name': self.hint_name,
            'output_shape': self.output_shape
        }

    def flops(self):
        return 0.0

    def memory_access(self):
        return 0.0

    def kernels(self):
        return 0

    def infer_shape(self):
        pass

    def readable_lines(self, indent) -> List[str]:
        return [f'[{self.hint_name}]Placeholder']


class Conv(Node):
    """
    Convolution operator.

    Parameters
    ----------
    out_channels: int
        The number of output channels

    kernel: Tuple[int, int]
        The kernel size

    stride: Tuple[int, int]
        The strides for each dimension

    padding: Tuple[int, int]
        The padding for each dimension

    groups: int
        The number of groups in the convolution, which must be a common factor of input channels and output channels

    act: str, must be one of 'relu', 'sigmoid', 'tanh', 'identity'
        The activation function applied to the output of convolution
    """
    __slots__ = ['out_channels', 'kernel', 'stride', 'padding', 'groups', 'act', 'weight', 'bias']

    def __init__(self, name, hint_name, inputs, out_channels, kernel, stride, padding, groups, act, output_shape):
        super().__init__("conv", name, hint_name, inputs, output_shape)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.out_channels = out_channels
        self.act = act
        assert act in ['relu', 'sigmoid', 'tanh', 'identity']

        #
        # Both weight and bias are instances of numpy.ndarray. It can be None when the computation graph is used to
        # measure the latency. However, it must be set properly before inference.
        #
        self.weight = None
        self.bias = None

    @staticmethod
    def from_config(config, name2node):
        node = Conv(
            name=config['name'],
            hint_name=config['hint_name'],
            inputs=[[Value.from_config(value_config, name2node) for value_config in term_config] for term_config in
                    config['inputs']],
            out_channels=config['out_channels'],
            kernel=config['kernel'],
            stride=config['stride'],
            padding=config['padding'],
            groups=config['groups'],
            act=config['act'],
            output_shape=config['output_shape']
        )
        for ti, term in enumerate(node.inputs):
            for vi, value in enumerate(term):
                name2node[value.node.name].uses.append((node, ti, vi))
        return node

    def export_config(self):
        config = {
            'type': 'conv',
            'name': self.name,
            'hint_name': self.hint_name,
            'inputs': None if self.inputs is None else [[value.export_config() for value in term] for term in
                                                        self.inputs],
            'out_channels': self.out_channels,
            'kernel': self.kernel,
            'stride': self.stride,
            'padding': self.padding,
            'groups': self.groups,
            'act': self.act,
            'output_shape': self.output_shape
        }
        return config

    def init_weights(self):
        bound = np.sqrt(1) / np.sqrt(self.kernel[0] * self.kernel[1] * self.input_shape[0] // self.groups)
        self.weight = np.random.uniform(low=-bound, high=bound, size=self.weight_shape).astype(np.float32)
        self.bias = np.random.uniform(low=-bound, high=bound, size=self.bias_shape).astype(np.float32)

    @property
    def input_shape(self):
        return (sum(term[0].length for term in self.inputs), *self.inputs[0][0].node.output_shape[1:])

    @property
    def weight_shape(self):
        return (self.out_channels, self.input_shape[0] // self.groups, *self.kernel)

    @property
    def bias_shape(self):
        return self.out_channels,

    def flops(self):
        flops_per_output = (1 + self.kernel[0] * self.kernel[1] * self.input_shape[0] // self.groups)
        flops_of_conv = self.output_shape[0] * self.output_shape[1] * self.output_shape[2] * flops_per_output
        return flops_of_conv + self.input_flops()

    def memory_access(self):
        return self.input_shape[0] * self.input_shape[1] * self.input_shape[2] + self.input_memory_access()

    def kernels(self):
        return 1 + self.input_kernels()

    def infer_shape(self):
        self.output_shape = (
            self.out_channels,
            1 + (self.input_shape[1] - self.kernel[0] + self.padding[0] * 2) // self.stride[0],
            1 + (self.input_shape[2] - self.kernel[1] + self.padding[1] * 2) // self.stride[1]
        )

    def readable_lines(self, indent) -> List[str]:
        return [f'[{self.hint_name}]Conv2d({self.input_readable_str()})']


class Sequential(Node):
    """
    A compound operator that contains multiple operators.

    Parameters
    ----------
    nodes: List[Node]
        A list of operators in the Sequential operator. The operators are executed sequentially in the occurring order.
    """
    __slots__ = ['nodes']

    def __init__(self, name, hint_name, nodes: List[Node], output_shape):
        super().__init__('sequential', name, hint_name, nodes[0].inputs, output_shape)
        self.nodes = nodes

    @staticmethod
    def from_config(config, name2node):
        nodes = []
        for snode_config in config['nodes']:
            snode = Node.from_config(snode_config, name2node)
            name2node[snode.name] = snode
            nodes.append(snode)
        node = Sequential(
            name=config['name'],
            hint_name=config['hint_name'],
            nodes=nodes,
            output_shape=config['output_shape']
        )
        for ti, term in enumerate(node.inputs):
            for vi, value in enumerate(term):
                name2node[value.node.name].uses.append((node, ti, vi))
        return node

    def export_config(self):
        return {
            'type': self.type,
            'name': self.name,
            'hint_name': self.hint_name,
            'nodes': [nd.export_config() for nd in self.nodes],
            'output_shape': self.output_shape
        }

    def infer_shape(self):
        for node in self.nodes:
            node.infer_shape()
        self.output_shape = self.nodes[-1].output_shape

    def flops(self):
        return sum(node.flops() for node in self.nodes)

    def memory_access(self):
        return sum(node.memory_access() for node in self.nodes)

    def kernels(self):
        return sum(node.kernels() for node in self.nodes)

    def readable_lines(self, indent) -> List[str]:
        lines = [f'[{self.hint_name}]Sequential(']
        for node in self.nodes:
            node_lines = node.readable_lines(indent)
            lines.extend([' ' * indent + line for line in node_lines])
        lines.append(')')
        return lines


class Pool(Node):
    """
    Pooling operator.

    Parameters
    ----------
    pool_type: str
        The pooling type string. It can be one of 'max', 'avg', 'global_max', 'global_avg'.

    kernel: Tuple[int, int]
        Kernel size.

    stride: Tuple[int, int]
        Stride size.

    padding: Tuple[int, int]
        Padding size.
    """
    __slots__ = ['name', 'inputs', 'pool_type', 'kernel', 'stride', 'padding', 'uses']
    #
    #   pool_type:
    #       max, avg, global_max, global_avg
    #

    def __init__(self, name, hint_name, inputs, pool_type, kernel, stride, padding, output_shape):
        super().__init__('pool', name, hint_name, inputs, output_shape)
        self.name = name
        self.pool_type = pool_type
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    @staticmethod
    def from_config(config, name2node):
        node = Pool(
            name=config['name'],
            hint_name=config['hint_name'],
            inputs=[[Value.from_config(value_config, name2node) for value_config in term_config]
                    for term_config in config['inputs']],
            pool_type=config['pool_type'],
            kernel=config['kernel'],
            stride=config['stride'],
            padding=config['padding'],
            output_shape=config['output_shape']
        )
        for ti, term in enumerate(node.inputs):
            for vi, value in enumerate(term):
                name2node[value.node.name].uses.append((node, ti, vi))
        return node

    def export_config(self):
        return {
            'type': 'pool',
            'name': self.name,
            'hint_name': self.hint_name,
            'inputs': [[value.export_config() for value in term] for term in self.inputs],
            'pool_type': self.pool_type,
            'kernel': self.kernel,
            'stride': self.stride,
            'padding': self.padding,
            'output_shape': self.output_shape
        }

    @property
    def input_shape(self):
        return (sum(term[0].length for term in self.inputs), *self.inputs[0][0].node.output_shape[1:])

    def infer_shape(self):
        if self.pool_type == 'global_avg':
            self.stride = (1, 1)
            self.padding = (0, 0)
            self.kernel = self.input_shape[1:]

        # infer output shape
        self.output_shape = [
            self.input_shape[0],
            1 + (self.input_shape[1] - self.kernel[0] + self.padding[0] * 2) // self.stride[0],
            1 + (self.input_shape[2] - self.kernel[1] + self.padding[1] * 2) // self.stride[1]
        ]

    def flops(self):
        output_size = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        if 'global' in self.pool_type:
            return output_size + self.input_flops()
        else:
            return output_size * self.kernel[0] * self.kernel[1] + self.input_flops()

    def memory_access(self):
        return self.input_shape[0] * self.input_shape[1] * self.input_shape[2] + self.input_memory_access()

    def kernels(self):
        return 1 + self.input_kernels()

    def readable_lines(self, indent) -> List[str]:
        namedict = {
            'max': 'MaxPool2d',
            'avg': 'AvgPool2d',
            'global_max': 'GlobalMaxPool2d',
            'global_avg': 'GlobalAvgPool2d'
        }
        return [f'[{self.hint_name}]{namedict[self.pool_type]}({self.input_readable_str()})']


class Element(Node):
    """
    Element-wise operator.

    Parameters
    ----------
    op_type: str
        The element-wise operator type. It can be one of 'add' and 'mul'. Element operator will calculate the
        sum (op_type='add') or product (op_type='mul') for each terms in the inputs.
    """
    __slots__ = ['op_type']

    def __init__(self, name, hint_name, inputs, op_type, output_shape):
        super().__init__("element", name, hint_name, inputs, output_shape)
        self.op_type: str = op_type

    @staticmethod
    def from_config(config, name2node):
        node = Element(
            name=config['name'],
            hint_name=config['hint_name'],
            inputs=[[Value.from_config(value_config, name2node) for value_config in term_config]
                    for term_config in config['inputs']],
            op_type=config['op_type'],
            output_shape=config['output_shape']
        )
        for ti, term in enumerate(node.inputs):
            for vi, value in enumerate(term):
                name2node[value.node.name].uses.append((node, ti, vi))
        return node

    def export_config(self):
        return {
            'type': 'element',
            'name': self.name,
            'hint_name': self.hint_name,
            'inputs': [[value.export_config() for value in term] for term in self.inputs],
            'op_type': self.op_type,
            'output_shape': self.output_shape
        }

    @property
    def input_shape(self):
        return (sum(term[0].length for term in self.inputs), *self.inputs[0][0].node.output_shape[1:])

    def infer_shape(self):
        self.output_shape = self.input_shape

    def flops(self):
        return 0.0 + self.input_flops()

    def memory_access(self):
        return 0.0 + self.input_memory_access()

    def kernels(self):
        return self.input_kernels()

    def readable_lines(self, indent) -> List[str]:
        if self.op_type == 'mul':
            merge_op = '*'
        else:
            merge_op = '+'
        return [f'[{self.hint_name}]{self.input_readable_str(merge_op)}']


class Identity(Node):
    """
    Identity operator. It can also works as the Concat operator when there are multiple terms in the inputs.
    """
    def __init__(self, name, hint_name, inputs, output_shape):
        super().__init__("identity", name, hint_name, inputs, output_shape)

    @staticmethod
    def from_config(config, name2node):
        node = Identity(
            name=config['name'],
            hint_name=config['hint_name'],
            inputs=[[Value.from_config(value_config, name2node) for value_config in term_config] for term_config in
                    config['inputs']],
            output_shape=config['output_shape']
        )
        for ti, term in enumerate(node.inputs):
            for vi, value in enumerate(term):
                name2node[value.node.name].uses.append((node, ti, vi))
        return node

    def export_config(self):
        return {
            'type': 'identity',
            'name': self.name,
            'hint_name': self.hint_name,
            'inputs': [[value.export_config() for value in term] for term in self.inputs],
            'output_shape': self.output_shape
        }

    @property
    def input_shape(self):
        return (sum(term[0].length for term in self.inputs), *self.inputs[0][0].node.output_shape[1:])

    def infer_shape(self):
        self.output_shape = self.input_shape

    def flops(self):
        return 0.0 + self.input_flops()

    def memory_access(self):
        return 0.0 + self.input_memory_access()

    def kernels(self):
        return self.input_kernels()

    def readable_lines(self, indent) -> List[str]:
        input_str = self.input_readable_str()
        if input_str[0] != 'C':
            input_str = '(' + input_str + ')'
        return [f'[{self.hint_name}]{input_str}']


class Activation(Node):
    """
    Activation operator.

    Parameters
    ----------
    act_type: str
        The activation type string. It can be one of 'relu', 'tanh', and 'sigmoid'.
    inplace: boolean
        Whether to do the activation inplace.
    """
    __slots__ = ['act_type', 'inplace']

    def __init__(self, name, hint_name, inputs, act_type, inplace, output_shape):
        super().__init__('activation', name, hint_name, inputs, output_shape)
        self.act_type = act_type
        self.inplace = inplace

    @staticmethod
    def from_config(config, name2node):
        node = Activation(
            name=config['name'],
            hint_name=config['hint_name'],
            inputs=[[Value.from_config(value_config, name2node) for value_config in term_config]
                    for term_config in config['inputs']],
            act_type=config['act_type'],
            inplace=config['inplace'],
            output_shape=config['output_shape']
        )
        for ti, term in enumerate(node.inputs):
            for vi, value in enumerate(term):
                name2node[value.node.name].uses.append((node, ti, vi))
        return node

    def export_config(self):
        return {
            'type': self.type,
            'name': self.name,
            'hint_name': self.hint_name,
            'inputs': [[value.export_config() for value in term] for term in self.inputs],
            'act_type': self.act_type,
            'inplace': self.inplace,
            'output_shape': self.output_shape
        }

    @property
    def input_shape(self):
        return (sum(term[0].length for term in self.inputs), *self.inputs[0][0].node.output_shape[1:])

    def infer_shape(self):
        self.output_shape = self.input_shape

    def flops(self):
        return 4 * self.input_shape[0] * self.input_shape[1] * self.input_shape[2] + self.input_flops()

    def memory_access(self):
        return self.input_shape[0] * self.input_shape[1] * self.input_shape[2] + self.input_memory_access()

    def kernels(self):
        return 1 + self.input_kernels()

    def readable_lines(self, indent) -> List[str]:
        return [f'{self.act_type.capitalize()}@{self.hint_name}({self.input_readable_str()})']


class Relu(Node):
    """
    Relu activation operator. It is equivalent to Activation(act_type='relu').
    """
    def __init__(self, name, hint_name, inputs, output_shape):
        super().__init__('relu', name, hint_name, inputs, output_shape)

    @staticmethod
    def from_config(config, name2node):
        node = Relu(
            name=config['name'],
            hint_name=config['hint_name'],
            inputs=[[Value.from_config(value_config, name2node) for value_config in term_config] for term_config in
                    config['inputs']],
            output_shape=config['output_shape']
        )
        for ti, term in enumerate(node.inputs):
            for vi, value in enumerate(term):
                name2node[value.node.name].uses.append((node, ti, vi))
        return node

    def export_config(self):
        return {
            'type': self.type,
            'name': self.name,
            'hint_name': self.hint_name,
            'inputs': [[value.export_config() for value in term] for term in self.inputs],
            'output_shape': self.output_shape
        }

    @property
    def input_shape(self):
        return (sum(term[0].length for term in self.inputs), *self.inputs[0][0].node.output_shape[1:])

    def infer_shape(self):
        self.output_shape = self.input_shape

    def flops(self):
        return 4 * self.input_shape[0] * self.input_shape[1] * self.input_shape[2] + self.input_flops()

    def memory_access(self):
        return self.input_shape[0] * self.input_shape[1] * self.input_shape[2] + self.input_memory_access()

    def kernels(self):
        return 1 + self.input_kernels()

    def readable_lines(self, indent) -> List[str]:
        return [f'[{self.hint_name}]Relu({self.input_readable_str()})']


class Block:
    """
    A block in the computation graph. A block is a set of operators. The dependency relation among the operators is a
    directed acyclic graph (DAG). There is only one enter operator and one exit operator for a block. We consider the
    exit operator belongs to the block while the enter operator does not belong to the block.

    Parameters
    ----------
    enter_node: ios.ir.Node
        The enter operator of the block.

    exit_node: Optional[ios.ir.Node]
        The exit operator of the block.

    inner_nodes: List[ios.ir.Node]
        The inner operators of the block.

    stages: Optional[List[List[List[str]]]]
        The execution schedule for the block. It contains a list of stages. Each stage contains of list of groups. Each
        group contains a list of operator names. Stages are executed sequentially. Groups in a stage are executed
        concurrently. Operators in the same group are executed sequentially.
    """
    __slots__ = ['enter_node', 'exit_node', 'inner_nodes', 'stages', 'parts']

    def __init__(self, enter_node, exit_node=None, inner_nodes=None, stages=None):
        if inner_nodes is None:
            inner_nodes = []
        self.enter_node: Optional[Node] = enter_node
        self.exit_node: Optional[Node] = exit_node
        self.inner_nodes: List[Node] = inner_nodes
        self.stages = stages
        self.parts = None

    @staticmethod
    def from_config(config, name2node):
        """
        Construct a block from a previously exported config.

        :param config: dict
            the config exported previously.
        :param name2node: Dict[str, ios.ir.Node]
            a dict that maps the operator name to operator (i.e. ios.ir.Node).
        :return: the constructed block.
        """
        enter_node = name2node[config['enter_name']]
        inner_nodes = []
        for inner_config in config['inner_nodes']:
            node = Node.from_config(inner_config, name2node)
            inner_nodes.append(node)
            name2node[node.name] = node
        exit_node = Node.from_config(config['exit_node'], name2node)
        stages = config['stages']
        name2node[exit_node.name] = exit_node
        return Block(enter_node, exit_node, inner_nodes, stages)

    def export_config(self):
        """
        Export the config of the block.
        """
        return {
            'enter_name': self.enter_node.name,
            'enter_node': self.enter_node.export_config(),
            'inner_nodes': [node.export_config() for node in self.inner_nodes],
            'exit_node': self.exit_node.export_config(),
            'stages': self.stages
        }

    def sequential_schedule(self):
        """
        Generate the sequential schedule for the block. The schedule is stored in self.stages.
        """
        self.stages = []
        for node in self.inner_nodes:
            self.stages.append([[node.name]])
        self.stages.append([[self.exit_node.name]])

    def greedy_schedule(self, max_num_stream):
        """
        Generate the greedy schedule for the block. The schedule is stored in self.stages.

        :param max_num_stream: int
            The maximum number of groups in a stage. If there are more potential groups in a stage, only put
            max_num_stream number of groups and leave remaining groups in the subsequent stages.
        """
        self.stages = []
        indeg = {node: sum(len(term) for term in node.inputs) for node in self.inner_nodes + [self.exit_node]}
        qu = [self.enter_node] + [node for node in self.inner_nodes + [self.exit_node] if len(node.inputs) == 0]
        while len(qu) > 0:
            stage = [node for node in qu if node is not self.enter_node]
            if len(stage) > 0:
                stage, remain = stage[:max_num_stream], stage[max_num_stream:]
                self.stages.append([[node.name] for node in stage])
                for u in remain:
                    qu.remove(u)
            else:
                remain = []
            nqu = []
            for u in qu:
                if u is self.exit_node:
                    continue
                for v in u.uses:
                    if v[0] not in indeg:   # v[0] is the node in Sequential
                        continue
                    indeg[v[0]] -= 1
                    if indeg[v[0]] == 0:
                        nqu.append(v[0])
            qu = remain + nqu
        assert sum(len(s) for s in self.stages) == len(self.inner_nodes) + 1

    def flops(self):
        """
        Return the number of flops of the block.
        """
        return sum(node.flops() for node in self.inner_nodes + [self.exit_node])

    def memory_access(self):
        """
        Return the number of memory access of the block.
        """
        return sum(node.memory_access() for node in self.inner_nodes + [self.exit_node])

    def kernels(self):
        """
        Return the number of kernel calls of the block.
        """
        return sum(node.kernels() for node in self.inner_nodes + [self.exit_node])

    def schedule_operators(self):
        """
        Return the number of schedule units in the block. This is different with the number of operators because
        the inner operators in the compound operator are not counted.
        """
        return sum(1 for _ in self.inner_nodes + [self.exit_node])

    def readable_lines(self, indent=0):
        """
        Return a list of lines representing the block in a human friendly manner.
        """
        name2node: Dict[str, Node] = {node.name: node for node in self.inner_nodes + [self.exit_node]}

        def group_readable_lines(group):
            if len(group) > 1:
                group_lines = [f'Sequential (']
                for name in group:
                    node = name2node[name]
                    node_lines = node.readable_lines(indent)
                    group_lines.extend([' ' * indent + line for line in node_lines])
                group_lines.append(')')
                return group_lines
            else:
                return name2node[group[0]].readable_lines(indent)

        def stage_readable_lines(stage):
            if len(stage) > 1:
                stage_lines = ['Parallel(']
                for group in stage:
                    group_lines = group_readable_lines(group)
                    stage_lines.extend([' ' * indent + line for line in group_lines])
                stage_lines.append(')')
                return stage_lines
            else:
                return group_readable_lines(stage[0])

        def block_readable_lines():
            use_block_level = False
            if use_block_level:
                block_lines = ['Sequential(']
                for stage in self.stages:
                    stage_lines = stage_readable_lines(stage)
                    block_lines.extend([' ' * indent + line for line in stage_lines])
                block_lines.append(')')
            else:
                block_lines = []
                for stage in self.stages:
                    block_lines.extend(stage_readable_lines(stage))
            return block_lines

        return block_readable_lines()


class Graph:
    """
    Graph represents the computation graph used in IOS. Any neural network is represented by computation graph.
    Please use the functions in ios.ir.common to construct the computation graph.

    Parameters
    ----------
    name: str
        The name of the computation graph, which is used as a hint.

    input: ios.ir.Placeholder
        The input of the computation graph. Currently, the computation graph only support a single input.

    blocks: List[ios.ir.Block]
        The list of blocks in the computation graph. For two continuous blocks b1 and b2. The exit operator of b1 must
        be the enter operator of b2.
    """
    __slots__ = ['name', 'input', 'input_shape', 'blocks']

    def __init__(self, name, input, blocks):
        self.name = name
        self.input: Placeholder = input
        self.blocks: List[Block] = blocks

    def nodes(self) -> Iterable[Node]:
        """
        An iterator of the operators in computation graph. The iteration order follows the topological order of the
        computation graph.
        """
        if len(self.blocks) == 0:
            return
        yield self.blocks[0].enter_node
        for block in self.blocks:
            for node in block.inner_nodes:
                yield node
            yield block.exit_node

    @property
    def enter_node(self) -> Node:
        return self.blocks[0].enter_node

    @property
    def exit_node(self) -> Node:
        return self.blocks[-1].exit_node

    @staticmethod
    def from_config(config):
        """
        Construct the computation graph by the graph config exported previously.
        """
        name2node = {}
        input = Placeholder.from_config(config['input'], name2node)
        name2node[input.name] = input
        blocks = [Block.from_config(block_config, name2node) for block_config in config['blocks']]
        graph = Graph(config['name'], input, blocks)
        graph.infer_shape()
        return graph

    def export_config(self):
        """
        Export the computation graph config.
        """
        return {
            'name': self.name,
            'input': self.input.export_config(),
            'blocks': [block.export_config() for block in self.blocks],
        }

    def infer_shape(self):
        """
        Infer the output shape of each operator.
        """
        for node in self.nodes():
            if node == self.blocks[0].enter_node:
                continue
            node.infer_shape()

    def init_weights(self):
        """
        Initialize the weights for each operator in the computation graph randomly.
        """
        need_init = (Conv,)

        def init_all(nodes: Iterable[Node]):
            for node in nodes:
                if isinstance(node, need_init):
                    node.init_weights()
                if isinstance(node, Sequential):
                    init_all(node.nodes)

        init_all(self.nodes())

    def sequential_schedule(self):
        """
        Set the execution schedule to be Sequential schedule, which executes the operator sequentially according to some
        topological order.
        """
        for block in self.blocks:
            block.sequential_schedule()

    def greedy_schedule(self, max_num_stream=10):
        """
        Set the execution schedule to be Greedy schedule, which executes the operators in a greedy manner. More
        specifically, Greedy schedule would execute at most max_num_stream number of operators at a time whenever
        possible.
        :param max_num_stream: int
            The maximum number of groups for each stage.
        """
        for block in self.blocks:
            block.greedy_schedule(max_num_stream)

    def flops(self):
        """
        Return the float point operators of the computation graph.
        """
        return sum(block.flops() for block in self.blocks)

    def memory_access(self):
        """
        Return the number of memory access operations of the computation graph.
        """
        return sum(block.memory_access() for block in self.blocks)

    def kernels(self):
        """
        Return the number of kernel calls of the computation graph.
        """
        return sum(block.kernels() for block in self.blocks)

    def schedule_operators(self):
        """
        The number of schedule operators (units) of the computation graphs. The inner operators in a compound operator
        such as Sequential are not counted in it.
        :return:
        """
        return sum(block.schedule_operators() for block in self.blocks)

    def readable_lines(self, indent=0):
        """
        Return a list of string lines in a human friendly manner.
        """
        lines = ['Sequential(']
        for block in self.blocks:
            block_lines = block.readable_lines(indent)
            lines.extend([' ' * indent + line for line in block_lines])
        lines.append(')')
        return lines

    def __str__(self):
        """
        Return a string represents the computation graph and execution schedule.
        """
        return "\n".join(self.readable_lines(2))
