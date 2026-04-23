#include <cuda_fp16.h>     
#include <cuda_runtime.h>  
#include <stdio.h>  
#include "kernel.h"

int main(){
    int m = 16;
    int n = 8;
    int k = 16;
    
    half *hA, *hB, *hC;
    hA = (half*)malloc(m * k * sizeof(half));
    hB = (half*)malloc(k * n * sizeof(half));
    hC = (half*)malloc(m * n * sizeof(half));

    half *dA, *dB, *dC;
    cudaMalloc(&dA, m * k * sizeof(half));
    cudaMalloc(&dB, k * n * sizeof(half));
    cudaMalloc(&dC, m * n * sizeof(half));
    
   
    
    // 작은 값으로 초기화(포화 방지)
    for (int r=0; r<m; ++r)
      for (int c=0; c<k; ++c)
        hA[r*k + c] = __float2half( ( (r+c) % 7 ) * 0.1f );

    // B를 column-major로 채움: index = col*K + row
    for (int c=0; c<n; ++c)
      for (int r=0; r<k; ++r)
        hB[c*k + r] = __float2half( ( (r+c) % 5 ) * 0.1f );

    cudaMalloc(&dA, m * k * sizeof(half));
    cudaMalloc(&dB, k * n * sizeof(half));
    cudaMalloc(&dC, m * n * sizeof(half));

    cudaMemcpy(dA, hA, m * k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, k * n * sizeof(half), cudaMemcpyHostToDevice);

    dim3 block(32); // 하나의 block이 16개의 열(x), 16개의 행(y)을 처리
    dim3 grid(n/8, m/16); 
    /* grid(1,1), block(16,16) → 총 256 threads가 있지만,
    if(row<m && col<n) 조건문이 있으니까
    필요한 6개 thread만 실제 연산에 참여합니다 */
    mma_matmul_1_0<<<grid, block>>>(dA, dB, dC, m, n, k);

    cudaDeviceSynchronize(); // GPU 연산이 모두 끝날 때까지 기다리기

    cudaMemcpy(hC, dC, m*n*sizeof(half), cudaMemcpyDeviceToHost);

    printf("Result (C = A × B):\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.2f ", __half2float(hC[i * n + j]));
        }
        printf("\n");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}