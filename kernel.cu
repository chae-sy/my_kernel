#include <cuda_fp16.h>     
#include <cuda_runtime.h>  
#include <stdio.h>         

__global__ void matmul_0_1(const half *A, const half *B, half *C, int M, int N, int K) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if ( row < M && col < N ) {
        float acc = 0.f;
        for (int i = 0; i< K; i++) {
        __half a = A[row*K + i];
        __half b = B[col + i*N];
        float psum = __half2float(__hmul(a, b)); 
        acc += psum; } // row major indexing, FP16 × FP16 → FP32
    C [row * N + col] = __float2half(acc); // FP32 → FP16

    }
    
}