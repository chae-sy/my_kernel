#pragma once
#include <cuda_fp16.h>

__global__ void matmul_0_1(const half *A, const half *B, half *C, int M, int N, int K);
__global__ void mma_matmul_3_0(const half *A, const half *B, float *C,
                               int M, int N, int K);

// 3.1:  N-stage pipeline + 4× tiling (128×128)
__global__ void mma_matmul_3_1(const half *A, const half *B, float *C,
                               int M, int N, int K);

// 3.2:  Two-stage FP16×FP16→FP32 accumulate every iteration
__global__ void mma_matmul_3_2(const half *A, const half *B, float *C,
                               int M, int N, int K);

// 3.3:  FP16×FP16 with FP16 accumulation, convert to FP32 output
__global__ void mma_matmul_3_3(const half *A, const half *B, float *C,
                               int M, int N, int K);

// 3.4:  Two-stage FP16×FP16→FP32 accumulate every accStep iterations
__global__ void mma_matmul_3_4(const half *A, const half *B, float *C,
                               int M, int N, int K);
