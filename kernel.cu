#include <cuda_fp16.h>     
#include <cuda_runtime.h>  
#include <stdio.h>         

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