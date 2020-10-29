# IOS: An Inter-Operator Scheduler for CNN Acceleration

With the increasing computational capacity, the sequential execution of CNNs no longer 
provides sufficient parallelization opportunities to fully utilize all the computation resources. 
We propose IOS that combines intra- and inter-operator parallelism and adapt dynamic programming to 
ﬁnd an efficient schedule that better utilizes the hardware. Experiments show that IOS can improve the 
GPU utilization and speedup modern CNN inference from 1.1 to 1.5x compared to the state-of-the-art 
libraries (e.g., TensorRT).

<div align="center">
  <img src="https://github.com/idy002/inter-operator-scheduler/blob/main/figures/frameworks_comparison.png" width=600>
  
  End-to-end inference performance comparison on a NVIDIA V100 GPU.
</div>

## 1 Installation 

Please follow this section to build IOS from source code.

### Prerequisites

- Cmake 3.10 or higher 
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0 or higher
- [cuDNN](https://developer.nvidia.com/cudnn) 7.6.5 or higher

### Build IOS runtime
To get started, clone the IOS source code from Github.
```shell script
git clone https://github.com/mit-han-lab/inter-operator-scheduler.git ios
cd ios
```
Then build the IOS runtime:
```shell script
mkdir build
cd build; 
cmake ..; make -j4
cd ..
```

### Install IOS python package
Once the IOS runtime has been built, run following commands to install the IOS python package.
```shell script
cd python; 
python setup.py install --user
```


## 3 Usage 
IOS optimizes user-defined computation graph and does inference on IOS backend. The following code snip is a sample, in which user 
1. defines the computation graph first,
2. then optimizes the execution schedule,
3. and does inference on IOS runtime at last.

<div align="center">
  <img src="https://github.com/idy002/inter-operator-scheduler/blob/main/figures/demo.png">
  Timeline of the sample network under Sequential, Greedy, and IOS optimized schedule.
</div>

The following code snip demonstrates the usage of IOS. It first define the computation graph of the network, then optimizes it with IOS, and finally does the inference. The network defined in the code is the sample network shown above.

```python
import numpy as np
import ios

def sample_network():
    v = ios.placeholder(output_shape=(375, 15, 15))
    block = ios.Block(enter_node=v.node)
    v1 = ios.conv2d(block, inputs=[[v]], out_channels=375, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v2 = ios.conv2d(block, inputs=[[v]], out_channels=750, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v3 = ios.conv2d(block, inputs=[[v]], out_channels=375, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v1 = ios.conv2d(block, inputs=[[v1]], out_channels=750, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    out = ios.identity(block, inputs=[[v1], [v2], [v3]], is_exit=True)  # concat v1, v2, and v3
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

# define computation graph
graph = sample_network()

# optimize execution schedule
optimized_graph = ios.optimize(graph, batch_size=1, opt_type='dp_parallel', compute_weight=True)

# measure latency
graph.sequential_schedule()
seq_latency, stage_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=6, profile_stage=True)
print(graph)
print(f'Sequential schedule: {np.mean(seq_latency):.3f} ms')
print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(6, -1), axis=0)}\n')

opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph, batch_size=1, repeat=6, profile_stage=True)
print(optimized_graph)
print(f'Optimized schedule: {np.mean(opt_latency):.3f} ms')
print(f'     Stage latency: {np.mean(np.array(stage_latency).reshape(6, -1), axis=0)}')

# inference on ios runtime
dummy_inputs = np.random.randn(1, 375, 15, 15)
output = ios.ios_runtime.graph_inference(optimized_graph, batch_size=1, input=dummy_inputs)
```
An output of this program:
```text
Sequential(
  [1]Conv2d(0)
  [2]Conv2d(0)
  [3]Conv2d(0)
  [4]Conv2d(1)
  [5]Concat(4,2,3)
)
Sequential schedule: 0.486 ms
      Stage latency: [0.11070578 0.12603733 0.10604089 0.12549689 0.01794844]

Sequential(
  Parallel(
    [1]Conv2d(0)
    [2]Conv2d(0)
  )
  Parallel(
    [4]Conv2d(1)
    [3]Conv2d(0)
  )
  [5]Concat(4,2,3)
)
Optimized schedule: 0.333 ms
     Stage latency: [0.16145067 0.15448178 0.01732267]
```

## 4 Experiments
See [instructions](experiments/README.md) to reproduce all the experiments.
