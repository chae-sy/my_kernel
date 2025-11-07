#pragma once
#include <cuda_fp16.h>

__global__ void matmul_0_1(const half *A, const half *B, half *C, int M, int N, int K);
