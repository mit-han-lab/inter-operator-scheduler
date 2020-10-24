# IOS: An Inter-Operator Scheduler for CNN Acceleration

With the increasing computational capacity, the sequential execution of CNNs no longer 
provides sufﬁcient parallelization opportunities to fully utilize all the computation resources. 
We propose IOS that combines intra- and interoperator parallelism and adapt dynamic programming to 
ﬁnd an efﬁcient schedule that better utilizes the hardware. Experiments show that IOS can improve the 
GPU utilization and speedup modern CNN inference from 1.1 to 1.5x compared to the state-of-the-art 
libraries (e.g., TensorRT).

## 1 Prerequisites

Current implementation of IOS is based on cuda platform and cuDNN.

- [CUDA Toolkit]()
- [cuDNN](https://developer.nvidia.com/cudnn)

The following frameworks/libraries are used as baseline:

- [TensorRT](https://developer.nvidia.com/tensorrt)
- [TVM](https://docs.tvm.ai/install/from_source.html)
- [TASO](https://github.com/jiazhihao/TASO)
- [Tensorflow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)

When installing TVM, please turn on the support for cuDNN and LLVM.

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

Please use the following command to install dependent python packages:

``
conda install pydot tqdm matplotlib
``

## 2 Build IOS

Please configure `CMakeLists.txt` by setting the following parameters
- Set the `CUDAPATH` to the CUDA toolkit root directory
- Set the `TRTPATH` to the TensorRT root directory (download and extract archive to somewhere and point TRTPATH to it)

Then execute the following commands to build IOS 
```bash
mkdir build
cd build
cmake ..
make -j8
```

Finally, add the path of `inter-operator-scheduler/python` to environment variable `PYTHONPATH` (You may want to config it in .bashrc file)

## 3 Experiments
The following parts shows the commands to reproduce all experiments and ablation study. Please run the shell commands in the root directory of inter-operator-scheduler.

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
You can use the following command to query the clock rates supported by your NVIDIA GPU.
```shell script
nvidia-smi --query --display SUPPORTED_CLOCKS
```
After the experiments, you can run the following command to reset your GPU clock
```shell script
sudo nvidia-smi --reset-gpu-clocks
```

### Comparison of Different Schedules
Command:
```shell script
cd experiments/latency; sh run_expr_schedules.sh; cd ../..
```
This experiment compare the following schedules: Sequential, Greedy, IOS-Merge, IOS-Parallel, and IOS-Both. 
For fair comparison, all schedules are executed in the same execution engine (IOS runtime).

### Comparison of cuDNN-based Frameworks
Command:
```shell script
cd experiments/latency; sh run_expr_frameworks.sh; cd ../..
```
This experiment compare IOS with other cuDNN-based frameworks/libraries: Tensorflow, TVM-cuDNN, TASO, and TensorRT. 
TVM-cuDNN is the TVM framework, but convolution uses the cuDNN kernel (`target = 'cuda -libs=cudnn'`). 

### Utilization Profiling
Command:
```shell script
cd experiments/utilization; sh run_expr_utilization.sh; cd ../..
```
Above command would generate a plot image named `active_warps.png`, which can reflect the real device utilization.

### Specialized Scheduling is Beneficial
IOS support specialized scheduling for different devices and different batch sizes. 

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

### Schedule Pruning Reduce Search Time
Command:
```shell script
cd experiments/prune; sh run_expr_prune.sh; cd ../..
```
To allow users to trade off the search time and optimized schedule latency, we introduce the schedule pruning strategy to reduce the search time. 
This experiment shows the trade-off between the search time and schedule latency.

### Consistent Improvement for Different Batch Sizes
Command:
```shell script
cd experiments/latency; sh run_expr_batchsize.sh; cd ../..
```
IOS can achieve consistent improvement for different batch sizes. In this experiment, we measure the latency of Inception V3 on batch size 1, 16, 32, 64, 128. 
Experiment result show that IOS consistently outperforms TensorRT in terms of throughput.

### Intra- and Inter-Operator Parallelism
Command:
```shell script
cd experiments/latency; sh run_expr_autotvm.sh; cd ../..
```
AutoTVM is specialized for improvement the efficiency of the kernel by searching a highly optimized schedule for the kernel itself. 
Current IOS is implemented based on vendor-provided library cuDNN. 
We compare both of them to give us more insight about the intra- and inter-operator parallelism.
Because AutoTVM is time consuming (it takes us 26 hours on a 8-V100 server to optimize the four benchmark networks), we provide the schedule configs in `tvm_schedule_configs` directory. 
Please note that these schedule configs are optimized for NVIDIA Tesla V100 SXM2 with driver 450.51.05 and cuda toolkit 10.2. 


