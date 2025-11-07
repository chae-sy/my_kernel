#pragma once
#include <cuda_fp16.h>

__global__ void reference_kernel(const half *A, const half *B, half *C, int M, int N, int K);
