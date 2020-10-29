


## Experiments
The following parts shows the commands to reproduce all experiments and ablation study. 

### Experiment Environment Setup

In this experiment, we compared IOS with different frameworks as following

- [TensorRT](https://developer.nvidia.com/tensorrt)
- [TVM](https://docs.tvm.ai/install/from_source.html)
- [TASO](https://github.com/jiazhihao/TASO)
- [Tensorflow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)

All experiments all conducted under following environment.
- Python 3.7
- NVIDIA Driver 450.51.05
- CUDA Toolkit 10.2
- CUDNN 7.6.5
- TensorRT 7.0.0.11
- TVM v0.6
- TASO v1.0
- Tensorflow 2.3
- PyTorch 1.6.0

The perquisites for each experiment(from 1 to 7) are
- Experiment 1, 3, 4, 5 do not require any other frameworks/libraries
- Experiment 2 requires TensorRT, TVM, TASO, Tensorflow, and PyTorch (you can ignore any of them if you do not want to compare IOS with it)
- Experiment 6 requires TensorRT (you can ignore it if you only compare Sequential schedule and IOS optimized schedule)
- Experiment 7 reqruies TVM

We recommend you reproduce the experiments in a conda environment:
```shell script
conda create -n ios python=3.7
conda activate ios
```

#### Install TensorRT runtime in IOS
1. Download the [TensorRT](https://developer.nvidia.com/tensorrt) from NVIDIA website. We recommend to download the tar archive.
2. Extract the TensorRT archive to somewhere ().
   ```shell script
   tar xvzf ~/Downloads/TensorRT-7.0.0.11.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz /path/to/ios
   ```
3. Configure the `config.cmake` file in ios root directory. Change `set(USE_TRT OFF)` to `set(USE_TRT /path/to/ios/TensorRT-TensorRT-7.0.0.11)`.
4. Rebuild IOS runtime and TRT runtime and reinstall IOS python package:
   ```shell script
   cd /path/to/ios; 
   mkdir -p build; cd build; cmake ..; make -j4; cd ..
   cd python; python setup.py install; cd ..
   ```

Now we finish the installation of TensorRT runtime in IOS. We can infer the IOS computation graph and measure its latency using `ios.trt_runtime` module as follows
```python
import numpy as np
import ios
graph = ios.models.inception_v3()
# measure latency
latency = ios.trt_runtime.graph_latency(graph, batch_size=1, repeat=5)
# inference
outputs = ios.trt_runtime.graph_inference(graph, batch_size=1, input=np.random.randn(1, 3, 299, 299))
```
Module `ios.trt_runtime` converts the IOS computation graph `ios.Graph` to the corresponding TensorRT network, measures the latency and infers the network using TensorRT library.

#### Install TVM
Please refer the [TVM installation guide](https://tvm.apache.org/docs/install/from_source.html) for the instructions to install TVM. 
Because we need to customize the installation configuration (step 3 bellow), we put the installation commands here for simplicity. 
1. Clone the TVM source code from Github.
   ```shell script
   git clone https://github.com/apache/incubator-tvm.git tvm
   cd tvm; 
   git checkout v0.6  # you can change v0.6 to v0.7 or v0.8 to use higher version of TVM
   git submodule update --resursive --init
   mkdir build; cp cmake/config.cmake build; 
   ```
2. Install `llvm` by `sudo apt install llvm`.
3. Configure `build/config.cmake`.
   1. Replace `set(USE_CUDA OFF)` by `set(USE_CUDA ON)` or `set(USE_CUDA /path/to/a/specific/cuda_toolkit)`.
   2. Replace `set(USE_CUDNN OFF)` to `set(USE_CUDNN ON)`.
   3. Replace `set(USE_LLVM OFF)` to `set(USE_LLVM ON)`. 
4. Build and Install TVM.
   ```shell script
   cd build; cmake ..; make -j; cd ..;
   cd python; python setup.py install --user; cd ..;
   cd topi/python; python setup.py install --user; cd ../.. # for tvm v0.6, ignore for tvm v0.7 or higher
   ```
5. Validate that you have successfully installed TVM by
   ```python
   import tvm
   print(tvm.__version__)
   ```

#### Install TASO
Please refer the [TASO installation guide](https://github.com/jiazhihao/TASO/blob/master/INSTALL.md) for the instructions to install TASO. 

#### Install Tensorflow
```shell script
pip install tensorflow
```

#### Install PyTorch
```shell script
conda install pytorch torchvision -c pytorch
```


### Lock GPU Clock Rate
Because modern GPU can adjust the execution clock rate dynamically to reduce energy consumption when the device is not busy. 
We can lock the clock rate to make the experiment results more accurate and consistent.
Before conducting the experiments, running the following command (need sudo-privilege).
```shell script
sudo nvidia-smi --lock-gpu-clocks MIN_CLOCK,MAX_CLOCK
```
This command lock the gpu clocks in the specified range `[MIN_CLOCK, MAX_CLOCK]`. 
In our experiments, we set both `MIN_CLOCK` and `MAX_CLOCK` to 1530, 
which is the maximum clock rate NVIDIA Tesla V100 SXM2 supports. 
You can use the following command to query the clock rates supported by your NVIDIA GPU,
```shell script
nvidia-smi --query --display SUPPORTED_CLOCKS
```
and use this command to watch the current GPU clock rate:
```shell script
watch nvidia-smi -q -d CLOCK
```
After the experiments, you can run the following command to reset your GPU clock
```shell script
sudo nvidia-smi --reset-gpu-clocks
```
Refer [here](https://www.microway.com/hpc-tech-tips/nvidia-smi_control-your-gpus/) for more information.

### 1 Comparison of Different Schedules
<div align="center">
  <img src="https://github.com/idy002/inter-operator-scheduler/blob/main/experiments/figures/schedules.png" width=600>
  
  Comparison of different schedules
</div>

Command:
```shell script
cd experiments/latency; sh run_expr_schedules.sh; cd ../..
```
This experiment compare the following schedules: Sequential, Greedy, IOS-Merge, IOS-Parallel, and IOS-Both. 
For fair comparison, all schedules are executed in the same execution engine (IOS runtime).

### 2 Comparison of cuDNN-based Frameworks
<div align="center">
  <img src="https://github.com/idy002/inter-operator-scheduler/blob/main/figures/frameworks_comparison.png" width=600>
  
  Comparison of different frameworks 
</div>

Command:
```shell script
cd experiments/latency; sh run_expr_frameworks.sh; cd ../..
```
This experiment compare IOS with other cuDNN-based frameworks/libraries: Tensorflow, TVM-cuDNN, TASO, and TensorRT. 
TVM-cuDNN is the TVM framework, but convolution uses the cuDNN kernel (`target = 'cuda -libs=cudnn'`). 

### 3 Utilization Profiling
<div align="center">
  <img src="https://github.com/idy002/inter-operator-scheduler/blob/main/experiments/figures/utilization.png" width=600>
  
  The profiling of 
  <a href="https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/issueefficiency.htm">active warps</a> 
  for demo network. 
</div>

Command:
```shell script
cd experiments/utilization; sh run_expr_utilization.sh; cd ../..
```
Above command would generate a plot image named `active_warps.png`, which can reflect the real device utilization.

### 4 Specialized Scheduling is Beneficial
IOS support specialized scheduling for different devices and different batch sizes. 
<div align="center">
  <img src="https://github.com/idy002/inter-operator-scheduler/blob/main/experiments/figures/specialization.png" width=600>
  
  Latency (ms) of specialized schedules for batch size 1, 32 and 128, and specialized schedule for NVIDIA Tesla K80 and V100.
</div>
<div align="center">
  <img src="https://github.com/idy002/inter-operator-scheduler/blob/main/experiments/figures/specialization_example.png" width=600>
  
  The schedules found by IOS for the last block of Inception V3.
</div>

To explore the specialization for different batch sizes, run the following command:
```shell script
cd experiments/specialization; sh run_expr_spec_batchsize.sh; cd ../..
```
We first optimize for different batch sizes (1, 16, 32, 64, and 128) to get the schedule specialized for different batch sizes (for your simplicity, we have put the schedules we got in the `schedules` directory). 
Then we execute the network Inception V3 with different batch sizes and specialized schedules (there are 25 combinations, 5 by 5). 

To explore the specialization for different devices, we need a different GPU device. In our experiment, we take NVIDIA Tesla K80 as the second device.
We first optimize the network on different devices to get the specialized schedules (we also put them in `schedules` directory). 
Then we execute the network with different specialized schedules on the two devices (there are 4 combinations, 2 by 2).
```shell script
cd experiments/specialization; sh run_expr_spec_device.sh; cd ../..
```
Experiments show that specialized scheduling is beneficial.

### 5 Schedule Pruning Reduce Search Time
<div align="center">
  <img src="https://github.com/idy002/inter-operator-scheduler/blob/main/experiments/figures/reduce_optimization_cost.png" width=600>
  
  Trade-off between the optimized latency and the optimization cost for Inception V3 and NasNet.
</div>

Command:
```shell script
cd experiments/prune; sh run_expr_prune.sh; cd ../..
```
To allow users to trade off the search time and optimized schedule latency, we introduce the schedule pruning strategy to reduce the search time. 
This experiment shows the trade-off between the search time and schedule latency.

### 6 Consistent Improvement for Different Batch Sizes
<div align="center">
  <img src="https://github.com/idy002/inter-operator-scheduler/blob/main/experiments/figures/large_batchsize.png" width=600>
  
  The throughput comparison of Sequential schedule, TensorRT and IOS on batch size 1, 16, 32, 64, 128 for Inception V3.
</div>

Command:
```shell script
cd experiments/latency; sh run_expr_batchsize.sh; cd ../..
```
IOS can achieve consistent improvement for different batch sizes. In this experiment, we measure the latency of Inception V3 on batch size 1, 16, 32, 64, 128. 
Experiment result show that IOS consistently outperforms TensorRT in terms of throughput.

### 7 Intra- and Inter-Operator Parallelism
<div align="center">
  <img src="https://github.com/idy002/inter-operator-scheduler/blob/main/experiments/figures/autotvm.png" width=600>
  
  End-to-end performance comparison between TVM-AutoTune and IOS. The optimizations of TVM-AutoTune and IOS are orthogonal.
</div>

Command:
```shell script
cd experiments/latency; sh run_expr_autotvm.sh; cd ../..
```
AutoTVM is specialized for improvement the efficiency of the kernel by searching a highly optimized schedule for the kernel itself. 
Current IOS is implemented based on vendor-provided library cuDNN. 
We compare both of them to give us more insight about the intra- and inter-operator parallelism.
Because AutoTVM is time consuming (it takes us 26 hours on a 8-V100 server to optimize the four benchmark networks), we provide the schedule configs in `tvm_schedule_configs` directory. 
Please note that these schedule configs are optimized for NVIDIA Tesla V100 SXM2 with driver 450.51.05 and cuda toolkit 10.2. 


