from .common import *

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg_net(cfg, name):
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224))
    block = Block(pv.node, None, [], None)

    v = pv
    for c in cfg:
        if c == 'M':
            v = pool2d(block, [[v]], pool_type='max', kernel=(2, 2), stride=(2, 2))
        else:
            v = conv2d(block, [[v]], c, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act="relu")
    v = pool2d(block, [[v]], pool_type='global_avg', is_exit=True)
    graph = Graph(name, pv.node, [block])
    graph.init_weights()
    return graph


def vgg_11():
    return vgg_net(cfgs['A'], 'vgg_11')


def vgg_13():
    return vgg_net(cfgs['B'], 'vgg_13')


def vgg_16():
    return vgg_net(cfgs['D'], 'vgg_16')


def vgg_19():
    return vgg_net(cfgs['E'], 'vgg_19')
