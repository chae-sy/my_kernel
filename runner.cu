#include <cuda_fp16.h>     
#include <cuda_runtime.h>  
#include <stdio.h>  
#include "kernel.cu"

int main(){
    int m = 3;
    int n = 2;
    int k = 2;
    
    half *hA, *hB, *hC;
    hA = (half*)malloc(m * k * sizeof(half));
    hB = (half*)malloc(k * n * sizeof(half));
    hC = (half*)malloc(m * n * sizeof(half));

    half *dA, *dB, *dC;
    cudaMalloc(&dA, m * k * sizeof(half));
    cudaMalloc(&dB, k * n * sizeof(half));
    cudaMalloc(&dC, m * n * sizeof(half));
    
    float A_vals[6] = {1, 2, 3, 4, 5, 6}; // half로 선언 불가, printf 불가 
    float B_vals[4] = {7, 8, 9, 10};
    for (int i = 0; i < 6; i++) hA[i] = __float2half(A_vals[i]);
    for (int i = 0; i < 4; i++) hB[i] = __float2half(B_vals[i]);

    cudaMalloc(&dA, m * k * sizeof(half));
    cudaMalloc(&dB, k * n * sizeof(half));
    cudaMalloc(&dC, m * n * sizeof(half));

    cudaMemcpy(dA, hA, m * k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, k * n * sizeof(half), cudaMemcpyHostToDevice);

    dim3 block(16, 16); // 하나의 block이 16개의 열(x), 16개의 행(y)을 처리
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y); 
    /* grid(1,1), block(16,16) → 총 256 threads가 있지만,
    if(row<m && col<n) 조건문이 있으니까
    필요한 6개 thread만 실제 연산에 참여합니다 */
    reference_kernel<<<grid, block>>>(dA, dB, dC, m, n, k);

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