#include <cuda_fp16.h>     // half 자료형, __half2float(), __float2half() 함수
#include <cuda_runtime.h>  // CUDA 런타임 API (cudaMalloc, cudaMemcpy 등)
#include <stdio.h>         // printf(), fprintf() 등 기본 I/O

__global__ void reference_kernel(const half *A, const half *B, half *C, int M, int N, int K) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if ( row < M && col < N ) {
        float acc = 0.f;
        for (int i = 0; i< K; i++) {
        acc += __half2float(A[row*K + i]) * __half2float(B[col + i*N]); }
    C [row * N + col] = __float2half(acc);

    }
    
}