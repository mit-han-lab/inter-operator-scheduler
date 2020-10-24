#include <cudnn.h>
#include "ops.h"
#include "utils.h"
#include <assert.h>

__global__ void assign_with_stride_dst(float *dst, const float *src, int n, int dst_blk_size, int src_blk_size) {
    CUDA_KERNEL_LOOP(i, n) {
        int blk_idx = i / dst_blk_size;
        int blk_offset = i % dst_blk_size;
        int src_offset = blk_idx * src_blk_size + blk_offset;
        int dst_offset = blk_idx * dst_blk_size + blk_offset;
        dst[dst_offset] = src[src_offset];
    }
}

__global__ void assign_with_stride_src(float *dst, const float *src, int n, int dst_blk_size, int src_blk_size) {
    CUDA_KERNEL_LOOP(i, n) {
        int blk_idx = i / src_blk_size;
        int blk_offset = i % src_blk_size;
        int src_offset = blk_idx * src_blk_size + blk_offset;
        int dst_offset = blk_idx * dst_blk_size + blk_offset;
        dst[dst_offset] = src[src_offset];
    }
}

__global__ void accumulate_sum_2(float *dst, const float *src1, const float *src2, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        dst[i] = src1[i] + src2[i];
    }
}

__global__ void accumulate_sum_3(float *dst, const float *src1, const float *src2, const float *src3, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        dst[i] = src1[i] + src2[i] + src3[i];
    }
}

__global__ void accumulate_sum_4(float *dst, const float *src1, const float *src2, const float *src3, const float *src4, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        dst[i] = src1[i] + src2[i] + src3[i] + src4[i];
    }
}

__global__ void accumulate_sum_5(float *dst, const float *src1, const float *src2, const float *src3, const float *src4, const float *src5, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        dst[i] = src1[i] + src2[i] + src3[i] + src4[i] + src5[i];
    }
}

