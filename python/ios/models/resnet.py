from .common import *

def resnet_front(v):
    block = Block(v.node, None, [], None)
    v = conv2d(block, [[v]], out_channels=64, kernel=(7, 7), stride=(2, 2), padding=(3, 3), act='relu')
    v = pool2d(block, [[v]], pool_type='max', kernel=(3, 3), stride=(2, 2), padding=(1, 1), is_exit=True)
    return v, block, 64

def basic_block(block, v, channels, stride, downsample, is_exit):
    skip = v
    if downsample is not None:
        skip = downsample
    v = conv2d(block, [[v]], out_channels=channels, kernel=(3, 3), stride=stride, padding=(1, 1), act='relu')
    v = conv2d(block, [[v]], out_channels=channels, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='identity')
    v = addition(block, [[v, skip]], is_exit=is_exit)
    return v

def bottleneck(block, v, channels, stride, downsample, is_exit):
    skip = v
    if downsample is not None:
        skip = downsample
    v = conv2d(block, [[v]], out_channels=channels, kernel=(1, 1), stride=stride, padding=(0, 0), act='relu')
    v = conv2d(block, [[v]], out_channels=channels, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v = conv2d(block, [[v]], out_channels=channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), act='identity')
    v = addition(block, [[v, skip]])
    v = activation(block, [[v]], act_type='relu', inplace=True, is_exit=is_exit)
    return v

def resnet_block(v, block_func, expansion, channels, layers, in_channels, stride):
    block = Block(v.node, None, [], None)
    if max(stride) != 1 or in_channels != channels * expansion:
        downsample = conv2d(block, [[v]], out_channels=channels * expansion, kernel=(1, 1), stride=stride, padding=(0, 0), act='relu')
    else:
        downsample = None
    v = block_func(block, v, channels, stride, downsample, is_exit=False)
    for t in range(1, layers):
        v = block_func(block, v, channels, stride=(1, 1), downsample=None, is_exit=(t == layers-1))
    return v, block, channels

def resnet18():
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224))
    v, block1, out_channels = resnet_front(pv)
    v, block2, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=64,  layers=2, in_channels=out_channels, stride=(1, 1))
    v, block3, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=128, layers=2, in_channels=out_channels, stride=(2, 2))
    v, block4, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=256, layers=2, in_channels=out_channels, stride=(2, 2))
    v, block5, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=512, layers=2, in_channels=out_channels, stride=(2, 2))

    graph = Graph("resnet18", pv.node, [block1, block2, block3, block4, block5])
    graph.init_weights()
    graph.sequential_schedule()
    return graph

def resnet34():
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224))
    v, block1, out_channels = resnet_front(pv)
    v, block2, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=64,  layers=3, in_channels=out_channels, stride=(1, 1))
    v, block3, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=128, layers=4, in_channels=out_channels, stride=(2, 2))
    v, block4, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=256, layers=6, in_channels=out_channels, stride=(2, 2))
    v, block5, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=512, layers=3, in_channels=out_channels, stride=(2, 2))

    graph = Graph("resnet34", pv.node, [block1, block2, block3, block4, block5])
    graph.init_weights()
    graph.sequential_schedule()
    return graph

def resnet50():
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224))
    v, block1, out_channels = resnet_front(pv)
    v, block2, out_channels = resnet_block(v, block_func=bottleneck, expansion=4, channels=64,  layers=3, in_channels=out_channels, stride=(1, 1))
    v, block3, out_channels = resnet_block(v, block_func=bottleneck, expansion=4, channels=128, layers=4, in_channels=out_channels, stride=(2, 2))
    v, block4, out_channels = resnet_block(v, block_func=bottleneck, expansion=4, channels=256, layers=6, in_channels=out_channels, stride=(2, 2))
    v, block5, out_channels = resnet_block(v, block_func=bottleneck, expansion=4, channels=512, layers=3, in_channels=out_channels, stride=(2, 2))

    graph = Graph("resnet50", pv.node, [block1, block2, block3, block4, block5])
    graph.init_weights()
    graph.sequential_schedule()
    return graph
