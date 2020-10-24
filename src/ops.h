//
// Created by yyding on 8/24/19.
//

#ifndef MUSIC_OPS_H
#define MUSIC_OPS_H

#include <driver_types.h>
#include <cuda.h>

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

__global__ void assign_with_stride_src(float *dst, const float *src, int n, int dst_blk_size, int src_blk_size);
__global__ void assign_with_stride_dst(float *dst, const float *src, int n, int dst_blk_size, int src_blk_size);
__global__ void accumulate_sum_2(float *dst, const float *src1, const float *src2, int n);
__global__ void accumulate_sum_3(float *dst, const float *src1, const float *src2, const float *src3, int n);
__global__ void accumulate_sum_4(float *dst, const float *src1, const float *src2, const float *src3, const float *src4, int n);
__global__ void accumulate_sum_5(float *dst, const float *src1, const float *src2, const float *src3, const float *src4, const float *src5, int n);


#endif //MUSIC_OPS_H
