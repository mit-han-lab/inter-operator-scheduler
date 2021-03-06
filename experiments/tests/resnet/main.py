import numpy as np
import ios
from ios.models.resnet import resnet34, resnet18, resnet50


for net in [resnet18, resnet34, resnet50]:
    graph = net()
    ios.draw(graph, fname=f'{graph.name}.png', label=graph.name)

    latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=6)

    optimized_graph = ios.optimize(graph, batch_size=1, verbose=True)
    optimized_latency = ios.ios_runtime.graph_latency(optimized_graph, batch_size=1, repeat=6)
    ios.draw(optimized_graph, fname=f'{optimized_graph.name}.png', label=graph.name)

    print(graph.name)
    print(f' Sequential schedule: {np.mean(latency):.3f} ms')
    print(f'  Optimized schedule: {np.mean(optimized_latency):.3f} ms')



