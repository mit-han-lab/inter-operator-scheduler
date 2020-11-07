#ifndef MUSIC_OPS_H
#define MUSIC_OPS_H

#include <driver_types.h>
#include <cuda.h>
#include <cuda_fp16.h>

#if defined(USE_FLOAT16)
typedef __half data_type;
#elif defined(USE_INT8)
typedef int8_t data_type;
#else
typedef float data_type;
#endif
extern cudnnDataType_t cudnn_data_type;


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int BLOCK_SIZE_LIMIT = 32768;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
    int ret = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    return (ret > BLOCK_SIZE_LIMIT) ? BLOCK_SIZE_LIMIT : ret;
}

__global__ void assign_with_stride_src(data_type *dst, const data_type *src, int n, int dst_blk_size, int src_blk_size);
__global__ void assign_with_stride_dst(data_type *dst, const data_type *src, int n, int dst_blk_size, int src_blk_size);
__global__ void accumulate_sum_2(data_type *dst, const data_type *src1, const data_type *src2, int n);
__global__ void accumulate_sum_3(data_type *dst, const data_type *src1, const data_type *src2, const data_type *src3, int n);
__global__ void accumulate_sum_4(data_type *dst, const data_type *src1, const data_type *src2, const data_type *src3, const data_type *src4, int n);
__global__ void accumulate_sum_5(data_type *dst, const data_type *src1, const data_type *src2, const data_type *src3, const data_type *src4, const data_type *src5, int n);


#endif //MUSIC_OPS_H
