from numpy.lib.function_base import vectorize
import ios
import numpy as np
import ios.models.randwire

graph = ios.models.randwire.randwire_xlarge()
graph.sequential_schedule()
latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=6)

optimized_graph = ios.optimize(graph, batch_size=1, verbose=True)
optimized_latency = ios.ios_runtime.graph_latency(optimized_graph, batch_size=1, repeat=6)


print(f' Sequential schedule: {np.mean(latency):.3f} ms')
print(f'  Optimized schedule: {np.mean(optimized_latency):.3f} ms')

